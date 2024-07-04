//===- OpenEdgeScheduler.cpp - Declare the class for ops schedule *- C++
//-*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements class for schedule functions.
//
//===----------------------------------------------------------------------===//

#include "compigra/CgraDialect.h"
#include "compigra/CgraOps.h"
#include "compigra/Scheduler/KernelSchedule.h"
#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

using namespace mlir;
using namespace compigra;

namespace compigra {
Operation *getCntOpIndirectly(Operation *userOp, Operation *op) {
  Operation *cntOp = userOp;
  // If the userOp is branchOp or conditionalOp, analyze which operation uses
  // the block argument
  if (isa<LLVM::BrOp>(userOp)) {
    // get argument index
    Block *currBlock = userOp->getBlock();
    Block *userBlock = userOp->getBlock()->getSuccessor(0);
    unsigned argIndex =
        std::distance(userOp->getOperands().begin(),
                      std::find(userOp->getOperands().begin(),
                                userOp->getOperands().end(), op->getResult(0)));
    Operation *useOp = nullptr;
    for (auto &op : userBlock->getOperations()) {
      if (op.getOperand(0) == userBlock->getArgument(argIndex)) {
        useOp = &op;
        break;
      }
    }
    cntOp = useOp;
  }

  return cntOp;
}

SmallVector<Operation *, 4> getCntOpIndirectly(Value val) {
  SmallVector<Operation *, 4> cntOps;
  if (!val.isa<BlockArgument>()) {
    cntOps.push_back(val.getDefiningOp());
    return cntOps;
  }

  // if the value is not block argument, return empty vector

  Block *block = val.getParentBlock();
  Value propVal;

  int argInd = val.cast<BlockArgument>().getArgNumber();
  for (auto pred : block->getPredecessors()) {
    Operation *termOp = pred->getTerminator();
    // Return operation does not propagate block argument
    if (isa<LLVM::ReturnOp>(termOp))
      continue;

    if (isa<LLVM::BrOp>(termOp)) {
      // the corresponding block argument is the argInd'th operator
      propVal = termOp->getOperand(argInd);
      auto defOps = getCntOpIndirectly(propVal);
      cntOps.append(defOps.begin(), defOps.end());
    } else if (isa<cgra::BneOp, cgra::BeqOp, cgra::BltOp, cgra::BgeOp>(
                   termOp)) {
      // The terminator would be beq, bne, blt, bge, etc, the propagated value
      // is counted from 2nd operand.
      propVal = termOp->getOperand(argInd + 2);
      auto defOps = getCntOpIndirectly(propVal);
      cntOps.append(defOps.begin(), defOps.end());
    }
  }

  return cntOps;
}
} // namespace compigra

static Instruction initVoidInstruction(std::string name) {
  Instruction inst;
  inst.name = name;
  inst.time = INT_MAX;
  inst.pe = -1;
  inst.Rout = -1;
  inst.opA = "Unknown";
  inst.opB = "Unknown";
  return inst;
}

static Value getCorrelatedVal(Value val) {
  //  Return the correlated value when the val is propagated through branch
  //  operations as block arguments
  for (auto user : val.getUsers()) {
    if (isa<LLVM::BrOp>(user)) {
      // get argument index
      Block *currBlock = user->getBlock();
      Block *userBlock = user->getBlock()->getSuccessor(0);
      unsigned argIndex =
          std::distance(user->getOperands().begin(),
                        std::find(user->getOperands().begin(),
                                  user->getOperands().end(), val));
      auto blockArg = userBlock->getArgument(argIndex);
      return blockArg;
    }
  }
  return val;
}

int OpenEdgeKernelScheduler::getConnectedBlock(int pe, std::string direction) {
  int row = pe / nCol;
  int col = pe % nCol;
  if (direction == "ROUT") {
    return pe;
  }

  // If direction == R0, R1, ...return pe
  if (direction.find("R") != std::string::npos)
    if (direction.size() > 1 && std::isdigit(direction[1])) {
      int reg = std::stoi(direction.substr(1));
      if (reg < maxReg)
        return pe;
    }

  if (direction == "RCT") {
    row = (row - 1 + nRow) % nRow; // Move up, wrap around if needed
  } else if (direction == "RCB") {
    row = (row + 1) % nRow; // Move down, wrap around if needed
  } else if (direction == "RCL") {
    col = (col - 1 + nCol) % nCol; // Move left, wrap around if needed
  } else if (direction == "RCR") {
    col = (col + 1) % nCol; // Move right, wrap around if needed
  } else {
    return -1;
  }

  return row * nCol + col;
}

void OpenEdgeKernelScheduler::assignSchedule(
    mlir::Block::OpListType &ops, std::map<int, Instruction> instructions) {
  for (auto [ind, op] : llvm::enumerate(ops)) {
    knownRes[&op] = instructions[ind];
  }
}

#ifdef HAVE_GUROBI
void OpenEdgeKernelScheduler::initVariables(
    GRBModel &model, std::map<Block *, GRBVar> &timeBlkEntry,
    std::map<Block *, GRBVar> &timeBlkExit,
    std::map<Operation *, GRBVar> &timeOpVar,
    std::map<Operation *, GRBVar> &spaceOpVar) {
  for (auto [bbId, block] : llvm::enumerate(region.getBlocks())) {

    // init block entry and exit time
    timeBlkEntry[&block] =
        model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER,
                     "b" + std::to_string(bbId) + "_entry");
    timeBlkExit[&block] =
        model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER,
                     "b" + std::to_string(bbId) + "_exit");
    for (auto [opId, op] : llvm::enumerate(block.getOperations())) {
      // Skip the constant operations which is mapped to Imm field
      if (isa<LLVM::ConstantOp>(op))
        continue;
      // Create the variable for the operation
      timeOpVar[&op] = model.addVar(
          -GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER,
          "time_" + std::to_string(bbId) + "_" + std::to_string(opId));
      //  Add constraints with entry and exit time
      model.addConstr(timeBlkEntry[&block] <= timeOpVar[&op]);
      model.addConstr(timeOpVar[&op] <= timeBlkExit[&block]);
      spaceOpVar[&op] = model.addVar(0.0, nCol * nRow - 1, 0.0, GRB_INTEGER,
                                     "pe_" + std::to_string(bbId) + "_" +
                                         std::to_string(opId));
    }
  }
}

void OpenEdgeKernelScheduler::initKnownSchedule(
    GRBModel &model, std::map<Operation *, GRBVar> &timeOpVar,
    std::map<Operation *, GRBVar> &spaceOpVar) {
  for (auto [op, inst] : knownRes) {
    model.addConstr(timeOpVar[op] == inst.time);
    model.addConstr(spaceOpVar[op] == inst.pe);
  }
}

void OpenEdgeKernelScheduler::initOpTimeConstraints(
    GRBModel &model, std::map<Operation *, GRBVar> &timeOpVar,
    std::map<Block *, GRBVar> &timeBlkEntry,
    std::map<Block *, GRBVar> &timeBlkExit) {
  Operation *returnOp = nullptr;
  for (auto [op, var] : timeOpVar) {
    if (isa<LLVM::ReturnOp>(op)) {
      returnOp = op;
    }
    // Add the constraint based on the successor
    // constraints  for the successor
    if (op->getBlock()->getTerminator() == op) {
      // If the operation is the terminator, the operation execution is the same
      // as the block exit time
      model.addConstr(var == timeBlkExit[op->getBlock()]);
    }
    // if the result is known, skip
    if (knownRes.find(op) != knownRes.end())
      continue;
    for (Operation *userOp : op->getUsers())
      model.addConstr(var + 1 <= timeOpVar[userOp]);
  }

  // Add the constraint based on the block
  for (auto &blk : region.getBlocks()) {
    for (auto sucBlk : blk.getSuccessors()) {
      // Skip the self-loop
      if (&blk == sucBlk)
        continue;
      model.addConstr(timeBlkExit[&blk] + 1 == timeBlkEntry[sucBlk]);
    }
  }

  // The returnOp is mapped to be EXIT, and must be execute alone
  for (auto [op, var] : timeOpVar) {
    if (op == returnOp)
      continue;
    model.addConstr(var <= timeOpVar[returnOp] - 1);
  }
}

static void addNeighborConstraints(GRBModel &model, Operation *consumer,
                                   std::vector<Operation *> &producers,
                                   int nRow, int nCol,
                                   std::map<Operation *, GRBVar> &timeOpVar,
                                   std::map<Operation *, GRBVar> &spaceOpVar) {
  // Create helper variables for the possible neighbors
  GRBVar left = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
  GRBVar right = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
  GRBVar top = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
  GRBVar bottom = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);

  // Auxiliary variables for calculations
  GRBVar xRow = model.addVar(0, nRow - 1, 0, GRB_INTEGER);
  GRBVar xCol = model.addVar(0, nCol - 1, 0, GRB_INTEGER);

  auto x = spaceOpVar[consumer];
  // Constraints to calculate row and column indices of x
  // xRow == x / nCol
  model.addConstr(xRow == (x - xCol) / nCol);
  // xCol == x % nCol
  GRBVar u = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
  model.addConstr(x == u * nCol + xCol);

  std::vector<GRBVar> leftVars;
  std::vector<GRBVar> rightVars;
  std::vector<GRBVar> topVars;
  std::vector<GRBVar> bottomVars;
  for (auto prodOp : producers) {
    auto y = spaceOpVar[prodOp];
    // Calculate left neighbor (wrap around if needed)
    GRBVar leftCol = model.addVar(0, nCol - 1, 0, GRB_INTEGER);
    GRBVar uLeft = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
    model.addConstr(xCol - 1 == nCol * uLeft + leftCol);
    model.addConstr(left == xRow * nCol + leftCol);

    // Calculate right neighbor (wrap around if needed)
    GRBVar rightCol = model.addVar(0, nCol - 1, 0, GRB_INTEGER);
    GRBVar uRight = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
    model.addConstr(xCol + 1 == nCol * uRight + rightCol);
    model.addConstr(right == xRow * nCol + rightCol);

    // Calculate top neighbor (wrap around if needed)
    GRBVar topRow = model.addVar(0, nRow - 1, 0, GRB_INTEGER);
    GRBVar uTop = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
    model.addConstr(xRow - 1 == nRow * uTop + topRow);
    model.addConstr(top == topRow * nCol + xCol);

    // Calculate bottom neighbor (wrap around if needed)
    GRBVar bottomRow = model.addVar(0, nRow - 1, 0, GRB_INTEGER);
    GRBVar uBottom = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
    model.addConstr(xRow + 1 == nRow * uBottom + bottomRow);
    model.addConstr(bottom == bottomRow * nCol + xCol);

    // Add constraints that y must be one of the neighbors
    GRBVar chooseLeft = model.addVar(0, 1, 0, GRB_BINARY);
    GRBVar chooseRight = model.addVar(0, 1, 0, GRB_BINARY);
    GRBVar chooseTop = model.addVar(0, 1, 0, GRB_BINARY);
    GRBVar chooseBottom = model.addVar(0, 1, 0, GRB_BINARY);
    leftVars.push_back(chooseLeft);
    rightVars.push_back(chooseRight);
    topVars.push_back(chooseTop);
    bottomVars.push_back(chooseBottom);

    GRBVar chooseSelf = model.addVar(0, 1, 0, GRB_BINARY);

    model.addQConstr(y == left * chooseLeft + right * chooseRight +
                              top * chooseTop + bottom * chooseBottom +
                              x * chooseSelf);
    model.addConstr(
        chooseLeft + chooseRight + chooseTop + chooseBottom + chooseSelf == 1);

    // Between the time gap of the producer and consumer, the producer PE cannot
    // execute any other operations
    auto startT = timeOpVar[prodOp];
    auto endT = timeOpVar[consumer];
    for (auto [op, tVar] : timeOpVar) {

      if (op == consumer || op == prodOp)
        continue;
      GRBVar pe = spaceOpVar[op];
      // A helper variable to indicate the time gap between the producer and the
      // consumer, where helper = 1 means startT <= t <= endT.
      GRBVar helper = model.addVar(0, 1, 0, GRB_BINARY);
      model.addConstr(tVar >= startT - 1e9 * (1 - helper));
      model.addConstr(tVar <= endT + 1e9 * (1 - helper));

      // Set constraints for pe != y if helper == 1
      GRBVar helper2 = model.addVar(0, 1, 0, GRB_BINARY);
      model.addQConstr(helper * pe <= helper * (y - 1e-4 + 1e9 * helper2));
      model.addQConstr(helper * pe >=
                       helper * (y + 1e-4 - 1e9 * (1 - helper2)));
    }
  }
}

void OpenEdgeKernelScheduler::initOpSpaceConstraints(
    GRBModel &model, std::map<Operation *, GRBVar> &spaceOpVar,
    std::map<Operation *, GRBVar> &timeOpVar) {

  for (auto [op, var] : spaceOpVar) {
    // BrOp does not require to consider data dependency
    if (isa<LLVM::BrOp>(op))
      continue;

    // The operation should be assigned to the PE that is connected to.(PE of
    // predecessors' neighbor or itself)
    std::vector<Operation *> producers;
    for (auto [ind, opVal] : llvm::enumerate(op->getOperands())) {
      if (opVal.getDefiningOp() && isa<LLVM::ConstantOp>(opVal.getDefiningOp()))
        continue;

      // Ignore the propagation of the block argument
      if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(op) &&
          ind >= 2)
        break;

      // The getCntOpIndirectly gets definition operations of the operand which
      // should be unique unless the operand is block argument.
      auto defOps = getCntOpIndirectly(opVal);
      auto cntOp = getCntOpIndirectly(opVal)[0];
      // If the operand is block argument, the defOps must be produced in the
      // same PE.
      if (defOps.size() > 1)
        for (auto defOp : defOps)
          if (spaceOpVar.find(defOp) != spaceOpVar.end())
            model.addConstr(spaceOpVar[defOp] == spaceOpVar[cntOp]);

      if (spaceOpVar.find(cntOp) == spaceOpVar.end())
        continue;
      producers.push_back(cntOp);
    }
    addNeighborConstraints(model, op, producers, nRow, nCol, timeOpVar,
                           spaceOpVar);

    // The result is known, skip
    if (knownRes.find(op) != knownRes.end() && knownRes[op].Rout >= 0)
      continue;

    // assign the space w.r.t to its user's PE if it can be inferred.
    for (auto userOp : op->getUsers()) {
      // Get the real userOp if the userOp is branchOp or conditionalOp
      auto cntOp = getCntOpIndirectly(userOp, op);
      if (knownRes.find(cntOp) != knownRes.end() && knownRes[cntOp].Rout >= 0) {
        knownRes[op] = initVoidInstruction(op->getName().getStringRef().str());

        // If the result is stored in the register, assign the same PE
        bool leftOp =
            cntOp->getOperand(0) == getCorrelatedVal(op->getResult(0));
        std::string direct = leftOp ? knownRes[cntOp].opA : knownRes[cntOp].opB;
        // Get the last char of direct
        int dstPE = getConnectedBlock(knownRes[cntOp].pe, direct);
        model.addConstr(var == dstPE);
        knownRes[op].pe = dstPE;
        break;
      }
    }
  }
}

/// The combination of time and space variables should be unique
void OpenEdgeKernelScheduler::initOpTimeSpaceConstraints(
    GRBModel &model, std::map<Operation *, GRBVar> &timeOpVar,
    std::map<Operation *, GRBVar> &spaceOpVar) {
  // Only consider the operations in the same basic block, as the operations in
  // different basic block have been considered in the time constraints.
  for (size_t i = 0; i < timeOpVar.size(); i++) {
    auto timePair = *std::next(timeOpVar.begin(), i);
    auto spacePair = *std::next(spaceOpVar.begin(), i);

    GRBVar t1 = timePair.second;
    GRBVar s1 = spacePair.second;

    Block *block = timePair.first->getBlock();
    for (size_t j = i + 1; j < timeOpVar.size(); j++) {
      auto timePair2 = *std::next(timeOpVar.begin(), j);
      auto spacePair2 = *std::next(spaceOpVar.begin(), j);

      // If the operations are in different blocks, skip
      if (block != timePair2.first->getBlock())
        continue;

      GRBVar t2 = timePair2.second;
      GRBVar s2 = spacePair2.second;

      GRBVar t_eq = model.addVar(0, 1, 0, GRB_BINARY);
      GRBVar s_eq = model.addVar(0, 1, 0, GRB_BINARY);

      GRBVar diffTAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(diffTAbs >= t1 - t2);
      model.addConstr(diffTAbs >= t2 - t1);

      GRBVar diffSAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(diffSAbs >= s1 - s2);
      model.addConstr(diffSAbs >= s2 - s1);

      // If diffAbs is zero, t_eq and s_eq should be one, 1e9 is a selected
      // large number to force the binary variable to be one
      model.addConstr(t_eq >= 1 - diffTAbs / 1e9);
      model.addConstr(s_eq >= 1 - diffSAbs / 1e9);
      model.addConstr(t_eq + s_eq <= 1);
    }
  }
}

void OpenEdgeKernelScheduler::initObjectiveFunction(
    GRBModel &model, GRBVar &funcStartT, GRBVar &funcEndT,
    std::map<Operation *, GRBVar> &timeOpVar,
    std::map<Block *, GRBVar> &timeBlkEntry,
    std::map<Block *, GRBVar> &timeBlkExit, GRBLinExpr &objExpr) {
  GRBLinExpr obj = objExpr;
  for (auto [blk, entry] : timeBlkEntry) {
    model.addConstr(funcStartT <= entry);
  }
  for (auto [blk, exit] : timeBlkExit) {
    model.addConstr(exit <= funcEndT);
  }
  obj = funcEndT - funcStartT;
  // Add the objective function to minimize the total execution time
  double coef = 1e-3;
  for (auto [op, var] : timeOpVar) {
    obj += coef * var;
  }
  model.setObjective(obj, GRB_MINIMIZE);
}

LogicalResult OpenEdgeKernelScheduler::createSchedulerAndSolve() {
  GRBEnv env = GRBEnv("./gurobi.log");
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel model = GRBModel(env);

  // Objective function
  GRBLinExpr obj = 0;

  std::map<Operation *, GRBVar> timeVarMap;
  std::map<Operation *, GRBVar> peVarMap;

  std::map<Block *, GRBVar> timeBlkEntry;
  std::map<Block *, GRBVar> timeBlkExit;
  initVariables(model, timeBlkEntry, timeBlkExit, timeVarMap, peVarMap);
  // assign the known schedule
  initKnownSchedule(model, timeVarMap, peVarMap);
  // create time constraints
  initOpTimeConstraints(model, timeVarMap, timeBlkEntry, timeBlkExit);
  // create space constraints
  initOpSpaceConstraints(model, peVarMap, peVarMap);
  // create time and space constraints
  initOpTimeSpaceConstraints(model, timeVarMap, peVarMap);
  // create the objective function
  GRBVar funcStartT =
      model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER, "t0");
  GRBVar funcEndT =
      model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER, "t1");
  initObjectiveFunction(model, funcStartT, funcEndT, timeVarMap, timeBlkEntry,
                        timeBlkExit, obj);

  // Optimize the model
  model.optimize();
  model.write("model.lp");

  // Check if the optimization status indicates infeasibility
  if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE ||
      model.get(GRB_IntAttr_Status) == GRB_INF_OR_UNBD)
    return failure();

  // If the model is infeasible, write the model to solution
  for (auto [op, var] : timeVarMap) {
    writeOpResult(op, var.get(GRB_DoubleAttr_X),
                  peVarMap[op].get(GRB_DoubleAttr_X), -1);
  }
  return success();
}
#endif