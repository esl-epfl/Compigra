//===- OpenEdgeScheduler.cpp - Declare the class for ops schedule *- C++-*-===//
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
#include "compigra/Scheduler/ModuloScheduleAdapter.h"
#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif
// for schedule result output
#include "fstream"
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace compigra;

namespace compigra {
Operation *getCntUseOpIndirectly(OpOperand &useOpr) {
  Operation *cntOp = useOpr.getOwner();
  unsigned argIndex = useOpr.getOperandNumber();
  // If the userOp is branchOp or conditionalOp, analyze which operation uses
  // the block argument
  if (isa<LLVM::BrOp>(cntOp)) {
    Block *userBlock = cntOp->getBlock()->getSuccessor(0);
    auto users = userBlock->getArgument(argIndex).getUsers();
    cntOp = *users.begin();
  }

  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(cntOp)) {
    if (argIndex >= 2 && argIndex < 2 + condBr.getNumTrueDestOperands()) {
      // if the argument if propagated to true branch
      Block *userBlock = condBr.getTrueDest();
      auto users = userBlock->getArgument(argIndex - 2).getUsers();
      cntOp = *users.begin();

    } else if (argIndex >= 2 + condBr.getNumTrueDestOperands()) {
      // if the argument if propagated to false branch
      Block *userBlock = condBr.getFalseDest();
      unsigned prefix = 2 + condBr.getNumTrueDestOperands();
      auto users = userBlock->getArgument(argIndex - prefix).getUsers();
      cntOp = *users.begin();
    }
  }
  return cntOp;
}

SmallPtrSet<Operation *, 4> getCntUserIndirectly(Value val) {
  SmallPtrSet<Operation *, 4> cntOps;
  // SmallPtrSet<Operation *, 4> visited;
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    auto argIndex = use.getOperandNumber();
    if (isa<LLVM::BrOp>(user)) {
      Block *currBlock = user->getBlock();
      Block *userBlock = user->getBlock()->getSuccessor(0);
      // find the corresponding users that use the block argument
      auto users = getCntUserIndirectly(userBlock->getArgument(argIndex));
      cntOps.insert(users.begin(), users.end());
      continue;
    }
    if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(user)) {
      if (argIndex >= 2 && argIndex < 2 + condBr.getNumTrueDestOperands()) {
        // if the argument if propagated to true branch
        Block *currBlock = user->getBlock();
        Block *userBlock = condBr.getTrueDest();
        auto users = getCntUserIndirectly(userBlock->getArgument(argIndex - 2));
        cntOps.insert(users.begin(), users.end());
      } else if (argIndex >= 2 + condBr.getNumTrueDestOperands()) {
        // if the argument if propagated to false branch
        Block *currBlock = user->getBlock();
        Block *userBlock = condBr.getFalseDest();
        unsigned prefix = 2 + condBr.getNumTrueDestOperands();
        auto users =
            getCntUserIndirectly(userBlock->getArgument(argIndex - prefix));
        cntOps.insert(users.begin(), users.end());
      } else
        cntOps.insert(user);
      continue;
    }

    cntOps.insert(user);
  }
  return cntOps;
}

SmallVector<Operation *, 4> getCntDefOpIndirectly(Value val,
                                                  Block *targetBlock) {
  SmallVector<Operation *, 4> cntOps;
  if (!val.isa<BlockArgument>()) {
    cntOps.push_back(val.getDefiningOp());
    return cntOps;
  }

  // if the value is not block argument, return empty vector
  // TODO[@YYY]: remove targetBlock from parameter list
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
      auto defOps = getCntDefOpIndirectly(propVal, targetBlock);
      cntOps.append(defOps.begin(), defOps.end());
    } else if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(termOp)) {
      // The terminator would be beq, bne, blt, bge, etc, the propagated value
      // is counted from 2nd operand.
      if (targetBlock == condBr.getTrueDest())
        propVal = condBr.getTrueOperand(argInd);
      else if (targetBlock == condBr.getFalseDest()) {
        propVal = condBr.getFalseOperand(argInd);
      } else
        continue;

      auto defOps = getCntDefOpIndirectly(propVal, targetBlock);
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
    return row * nCol + col;
  }

  if (direction == "RCB") {
    row = (row + 1) % nRow; // Move down, wrap around if needed
    return row * nCol + col;
  }

  if (direction == "RCL") {
    col = (col - 1 + nCol) % nCol; // Move left, wrap around if needed
    return row * nCol + col;
  }

  if (direction == "RCR") {
    col = (col + 1) % nCol; // Move right, wrap around if needed
    return row * nCol + col;
  }

  return -1;
}

static bool useValueForCmp(Operation *userOp, Value val) {
  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(userOp))
    for (auto [ind, op] : llvm::enumerate(condBr.getOperands())) {
      if (ind >= 2)
        break;
      if (op == val)
        return true;
    }
  return false;
}

void OpenEdgeKernelScheduler::assignSchedule(
    std::vector<opWithId> ops, const int II, int &curPC,
    std::map<int, int> opExec, const std::map<int, Instruction> instructions,
    std::vector<int> &totalExec, int gap) {
  for (auto [op, opId] : ops) {
    knownRes[op] = instructions.at(opId);
    knownRes[op].time = opExec.at(opId) + II * totalExec[opId] + gap;
    if (isa<cgra::ConditionalBranchOp>(op) && knownRes[op].opB == "ROUT")
      knownRes[op].opB = "Unknown";
    curPC = std::max(curPC, knownRes[op].time);
    totalExec[opId]++;
    // llvm::errs() << *op << " (" << opId << ")"
    //              << " : [" << knownRes[op].time << ", " << knownRes[op].pe
    //              << "]\n";
  }
}

void OpenEdgeKernelScheduler::assignSchedule(
    mlir::Block::OpListType &ops,
    const std::map<int, Instruction> instructions) {
  for (auto [ind, op] : llvm::enumerate(ops)) {
    knownRes[&op] = instructions.at(ind);
    if (isa<cgra::ConditionalBranchOp>(op) && knownRes[&op].opB == "ROUT")
      knownRes[&op].opB = "Unknown";
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
      varNamePost[&op] = std::to_string(bbId) + "_" + std::to_string(opId);
    }
  }
}

void OpenEdgeKernelScheduler::initKnownSchedule(
    GRBModel &model, const std::map<Operation *, GRBVar> timeOpVar,
    const std::map<Operation *, GRBVar> spaceOpVar) {
  for (auto [op, inst] : knownRes) {
    model.addConstr(timeOpVar.at(op) == inst.time);
    model.addConstr(spaceOpVar.at(op) == inst.pe);
  }
}

void OpenEdgeKernelScheduler::initOpTimeConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> timeOpVar,
    const std::map<Block *, GRBVar> timeBlkEntry,
    const std::map<Block *, GRBVar> timeBlkExit) {
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
      model.addConstr(var == timeBlkExit.at(op->getBlock()));
    }

    // if (knownRes.find(op) != knownRes.end())
    //   continue;
    for (auto &use : op->getUses()) {
      auto userOp = use.getOwner();
      // Skip the parameter propagation use
      if (isa<LLVM::BrOp>(userOp))
        continue;
      if (isa<cgra::ConditionalBranchOp>(userOp) && use.getOperandNumber() >= 2)
        continue;
      if (userOp->getBlock() == op->getBlock())
        model.addConstr(var + 1 <= timeOpVar.at(userOp));
    }
  }

  // Add the constraint based on the block
  for (auto [id1, blk1] : llvm::enumerate(region.getBlocks())) {
    if (id1 == 0)
      continue;
    auto prevNode = blk1.getPrevNode();
    model.addConstr(timeBlkEntry.at(&blk1) >= timeBlkExit.at(prevNode) + 1);
  }

  // The returnOp is mapped to be EXIT, and must be execute alone
  for (auto [op, var] : timeOpVar) {
    if (op == returnOp)
      continue;
    model.addConstr(var <= timeOpVar.at(returnOp) - 1);
  }
}

/// Determine whether the srcBlk is the predecessor of dstBlk
static bool isBackEdge(Block *srcBlk, Block *dstBlk) {
  auto &entryBlock = srcBlk->getParent()->front();
  // start DFS from entry block, if dstBlk is visited before srcBlk, it is a
  // back edge
  std::unordered_set<Block *> visited;
  std::stack<Block *> stack;
  stack.push(&entryBlock);
  while (!stack.empty()) {
    auto currBlk = stack.top();
    stack.pop();
    if (visited.find(currBlk) != visited.end())
      continue;
    visited.insert(currBlk);
    for (auto succBlk : currBlk->getSuccessors()) {
      if (succBlk == dstBlk)
        return visited.find(srcBlk) == visited.end();
      stack.push(succBlk);
    }
  }
  return false;
}

static bool isBackEdge(Operation *srcOp, Operation *dstOp) {
  /// if the dstOp directly consumes the result of srcOp, and they are in the
  /// same block, it is not a back edge
  if (srcOp->getBlock() == dstOp->getBlock()) {
    for (auto opr : dstOp->getOperands()) {
      if (opr == srcOp->getResult(0))
        return false;
    }
    return true;
  }

  return isBackEdge(srcOp->getBlock(), dstOp->getBlock());
}

static std::stack<Block *> getBlockPath(Block *srcBlk, Block *dstBlk) {
  std::stack<Block *> path;
  std::unordered_set<Block *> visited;
  std::unordered_map<Block *, Block *>
      parent; // To store the parent of each block

  std::stack<Block *> dfsStack;
  dfsStack.push(srcBlk);
  visited.insert(srcBlk);

  while (!dfsStack.empty()) {
    Block *current = dfsStack.top();
    dfsStack.pop();

    // If we reached the destination block, build the path
    if (current == dstBlk) {
      while (current != nullptr) {
        path.push(current);
        current = parent[current];
      }
      return path;
    }

    for (Block *successor : current->getSuccessors()) {
      if (visited.find(successor) == visited.end()) {
        visited.insert(successor);
        parent[successor] = current;
        dfsStack.push(successor);
      }
    }
  }

  // If no path found, return an empty stack
  return std::stack<Block *>();
}

static bool isLoopBlock(Block *blk) {
  for (auto sucBlk : blk->getSuccessors())
    if (sucBlk == blk)
      return true;
  return false;
}

static bool viaArgPropagate(Operation *srcOp, Operation *dstOp) {
  for (auto user : srcOp->getUsers()) {
    if (user == dstOp)
      return false;
  }
  return true;
}

static std::vector<std::pair<GRBVar, GRBVar>>
getTimeGapBetween(Operation *srcOp, Operation *dstOp,
                  std::map<Operation *, GRBVar> &timeOpVar,
                  std::map<Operation *, GRBVar> &spaceOpVar,
                  std::map<Block *, GRBVar> timeBlkEntry,
                  std::map<Block *, GRBVar> timeBlkExit) {
  if (srcOp->getBlock() == dstOp->getBlock()) {
    // determine whether operands of dstOp directly produced by srcOp, if yes,
    // time gap is <TsrcOp, TdstOp>
    if (!isBackEdge(srcOp, dstOp)) {
      return {{timeOpVar[srcOp], timeOpVar[dstOp]}};
    }
    // if it is a back edge, the time gap is <TsrcOp, TBlockExit>, <TBlockEntry,
    // TdstOp>
    return {{timeOpVar[srcOp], timeBlkExit[srcOp->getBlock()]},
            {timeBlkEntry[dstOp->getBlock()], timeOpVar[dstOp]}};
  }

  // first seek the block path from srcOp to dstOp
  auto path = getBlockPath(srcOp->getBlock(), dstOp->getBlock());

  // if srcOp and dstOp are in different blocks, and is not a back edge,
  // the block is connected by path bb0 -> bb1 -> ... -> bbn
  // time gap is <TsrcOp, Tb0Exit>, <Tb1Entry, Tb1Exit>, ... <TbnEntry, TdstOp>
  if (!isBackEdge(srcOp->getBlock(), dstOp->getBlock())) {
    std::vector<std::pair<GRBVar, GRBVar>> timeGaps;
    timeGaps.push_back({timeOpVar[srcOp], timeBlkExit[srcOp->getBlock()]});

    path.pop(); // pop the srcOp block
    for (int i = 0; i < path.size() - 1; i++) {
      timeGaps.push_back({timeBlkEntry[path.top()], timeBlkExit[path.top()]});
      path.pop();
    }
    timeGaps.push_back({timeBlkEntry[dstOp->getBlock()], timeOpVar[dstOp]});
    // if dstOp block is self loop, add the time gap <TdstOp, TbExit>
    if (isLoopBlock(dstOp->getBlock()) && !viaArgPropagate(srcOp, dstOp)) {
      timeGaps.push_back({timeOpVar[dstOp], timeBlkExit[dstOp->getBlock()]});
      timeGaps.push_back(
          {timeBlkExit[dstOp->getBlock()], timeBlkExit[dstOp->getBlock()]});
    }
    return timeGaps;
  }

  // if it is a back edge, the time gap is <TsrcOp, TBlockExit>,<Tb1Entry,
  // Tb1Exit>,...<TbnEntry, TdstOp>
  std::vector<std::pair<GRBVar, GRBVar>> timeGaps;
  timeGaps.push_back({timeOpVar[srcOp], timeBlkExit[srcOp->getBlock()]});
  path.pop(); // pop the srcOp block
  for (int i = 0; i < path.size() - 1; i++) {
    timeGaps.push_back({timeBlkEntry[path.top()], timeBlkExit[path.top()]});
    path.pop();
  }
  timeGaps.push_back({timeBlkEntry[dstOp->getBlock()], timeOpVar[dstOp]});
  return timeGaps;
}

static LogicalResult
addNeighborConstraints(GRBModel &model, Operation *consumer,
                       std::vector<Operation *> &producers, int nRow, int nCol,
                       std::map<Operation *, GRBVar> timeOpVar,
                       std::map<Operation *, GRBVar> spaceOpVar,
                       std::map<Block *, GRBVar> timeBlkEntry,
                       std::map<Block *, GRBVar> timeBlkExit,
                       std::map<Operation *, std::string> varName) {
  // Create helper variables for the possible neighbors
  GRBVar left = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
  GRBVar right = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
  GRBVar top = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
  GRBVar bottom = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);

  // Auxiliary variables for calculations
  GRBVar xRow = model.addVar(0, nRow - 1, 0, GRB_INTEGER);
  GRBVar xCol = model.addVar(0, nCol - 1, 0, GRB_INTEGER);

  auto &x = spaceOpVar[consumer];
  // Constraints to calculate row and column indices of x
  // xRow == x / nCol
  model.addConstr(xRow == (x - xCol) / nCol);
  // xCol == x % nCol
  GRBVar u = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
  model.addConstr(x == u * nCol + xCol);

  for (auto prodOp : producers) {
    auto &y = spaceOpVar[prodOp];
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
    GRBVar chooseLeft =
        model.addVar(0, 1, 0, GRB_BINARY,
                     "left" + varName[consumer] + "for" + varName[prodOp]);
    GRBVar chooseRight =
        model.addVar(0, 1, 0, GRB_BINARY,
                     "right" + varName[consumer] + "for" + varName[prodOp]);
    GRBVar chooseTop =
        model.addVar(0, 1, 0, GRB_BINARY,
                     "top" + varName[consumer] + "for" + varName[prodOp]);
    GRBVar chooseBottom =
        model.addVar(0, 1, 0, GRB_BINARY,
                     "bottom" + varName[consumer] + "for" + varName[prodOp]);
    GRBVar chooseSelf =
        model.addVar(0, 1, 0, GRB_BINARY,
                     "self" + varName[consumer] + "for" + varName[prodOp]);

    model.addQConstr(y == left * chooseLeft + right * chooseRight +
                              top * chooseTop + bottom * chooseBottom +
                              x * chooseSelf);

    model.addConstr(
        chooseLeft + chooseRight + chooseTop + chooseBottom + chooseSelf == 1);

    // Between the time gap of the producer and consumer, the producer PE cannot
    // execute any other operations
    auto timeGaps = getTimeGapBetween(prodOp, consumer, timeOpVar, spaceOpVar,
                                      timeBlkEntry, timeBlkExit);
    // if x ! = y, consumer and producer are not in the same PE
    GRBVar diffXY = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
    model.addConstr(diffXY == x - y);
    GRBVar diffXYAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
    model.addGenConstrAbs(diffXYAbs, diffXY);
    GRBVar ifPEEq = model.addVar(0, 1, 0, GRB_BINARY);
    model.addConstr(ifPEEq >= diffXYAbs / 1e6);

    int constrInd = 0;
    for (auto [op, tVar] : timeOpVar) {
      if (op == consumer || op == prodOp)
        continue;
      for (auto it = timeGaps.begin(); it != timeGaps.end(); ++it) {
        auto [startT, endT] = *it;
        GRBVar &pe = spaceOpVar[op];
        // A helper variable to indicate the time gap between the producer and
        // the consumer, where helper = 1 means startT <= t <= endT.
        GRBVar h1 = model.addVar(0, 1, 0, GRB_BINARY);
        model.addConstr(tVar >= startT - 1e3 * (1 - h1));
        model.addConstr(tVar <= startT + 1e3 * h1 - 1e-2);

        GRBVar h2 = model.addVar(0, 1, 0, GRB_BINARY);
        model.addConstr(endT >= tVar - 1e3 * (1 - h2));
        if (std::next(it) == timeGaps.end()) {
          model.addConstr(endT <= tVar + 1e3 * h2);
        } else {
          model.addConstr(endT <= tVar + 1e3 * h2 - 1e-2);
        }

        GRBVar h = model.addVar(0, 1, 0, GRB_BINARY);

        // if t_start<=t<=t_end && x!=y

        GRBVar xvars[3] = {h1, h2, ifPEEq};
        model.addGenConstrAnd(h, xvars, 3);

        // Set constraints for pe != y if helper == 1
        GRBVar diff = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
        GRBVar diffAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
        model.addConstr(diff == pe - y,
                        "diff" + varName[op] + "_" + varName[consumer] + "_" +
                            varName[prodOp] + "_" + std::to_string(constrInd));
        model.addGenConstrAbs(diffAbs, diff);

        model.addGenConstrIndicator(h, 1, diffAbs, GRB_GREATER_EQUAL, 1e-4,
                                    "indicator_" + varName[consumer] + "with" +
                                        varName[prodOp] + "id" +
                                        std::to_string(constrInd));

        constrInd++;
      }
    }
  }
  return success();
}

void OpenEdgeKernelScheduler::initOpSpaceConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> spaceOpVar,
    const std::map<Operation *, GRBVar> timeOpVar,
    const std::map<Block *, GRBVar> timeBlkEntry,
    const std::map<Block *, GRBVar> timeBlkExit) {

  for (auto [op, var] : spaceOpVar) {
    // BrOp does not require to consider data dependency
    if (isa<LLVM::BrOp>(op))
      continue;

    // The operation should be assigned to the PE that is connected to.(PE
    // of predecessors' neighbor or itself)
    std::vector<Operation *> producers;
    for (auto [ind, opVal] : llvm::enumerate(op->getOperands())) {
      if (opVal.getDefiningOp() && isa<LLVM::ConstantOp>(opVal.getDefiningOp()))
        continue;

      // Ignore the propagation of the block argument
      if (isa<cgra::ConditionalBranchOp>(op) && ind >= 2)
        break;

      // The getCntDefOpIndirectly gets definition operations of the operand
      // which should be unique unless the operand is block argument.
      auto defOps = getCntDefOpIndirectly(opVal, op->getBlock());
      for (auto defOp : defOps) {
        producers.push_back(defOp);
      }

      auto cntOp = defOps[0];
      if (spaceOpVar.find(cntOp) == spaceOpVar.end())
        continue;
      // If the operand is block argument, the defOps must be produced in the
      // same PE.
      if (defOps.size() > 1)
        for (auto defOp : defOps)
          if (spaceOpVar.find(defOp) != spaceOpVar.end())
            model.addConstr(spaceOpVar.at(defOp) == spaceOpVar.at(cntOp));
    }

    if (failed(addNeighborConstraints(model, op, producers, nRow, nCol,
                                      timeOpVar, spaceOpVar, timeBlkEntry,
                                      timeBlkExit, varNamePost))) {
      llvm::errs() << "Failed to add neighbor constraints\n";
      return;
    }
    model.optimize();
    if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
      llvm::errs() << "can not fit" << *op << "\n";
      return;
    }

    // The result is known, skip
    if (knownRes.find(op) != knownRes.end() && knownRes[op].Rout >= 0)
      continue;

    // assign the space w.r.t to its user's PE if it can be inferred.
    for (auto &use : op->getUses()) {
      auto userOp = use.getOwner();
      // ignore if it is for the parameter propagation
      if (isa<LLVM::BrOp>(userOp) ||
          (isa<cgra::ConditionalBranchOp>(userOp) &&
           !useValueForCmp(userOp, op->getResult(0))))
        continue;
      // Get the real userOp if the userOp is branchOp or conditionalOp
      // TODO[@YYY]: check the cntOp's validity and determine whether delete
      // the previous continue
      // auto cntOp = getCntUseOpIndirectly(use);
      auto cntOp = userOp;
      if (knownRes.find(cntOp) != knownRes.end() && knownRes[cntOp].Rout >= 0) {
        knownRes[op] = initVoidInstruction(op->getName().getStringRef().str());

        // If the result is stored in the register, assign the same PE
        int leftOpInd = 0;
        if (isa<cgra::BzfaOp, cgra::BsfaOp>(cntOp))
          leftOpInd = 1;
        bool leftOp =
            cntOp->getOperand(leftOpInd) == getCorrelatedVal(op->getResult(0));
        std::string direct = leftOp ? knownRes[cntOp].opA : knownRes[cntOp].opB;
        // if cannot refer from its user's operands, skip it
        if (direct == "Unknown")
          continue;
        // Get the last char of direct
        int dstPE = getConnectedBlock(knownRes[cntOp].pe, direct);
        llvm::errs() << *cntOp << " in " << knownRes[cntOp].pe << " >> "
                     << direct << " ";
        llvm::errs() << "Assign " << *op << " to " << dstPE << "\n";
        model.addConstr(var == dstPE);
        knownRes[op].pe = dstPE;
        model.optimize();
        if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
          llvm::errs() << "ASSIGN OP" << *op << "FAILED\n";
          return;
        }
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

      GRBVar diffT = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
      GRBVar diffTAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(diffT == t1 - t2);
      model.addGenConstrAbs(diffTAbs, diffT);

      GRBVar diffS = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
      GRBVar diffSAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(diffS == s1 - s2);
      model.addGenConstrAbs(diffSAbs, diffS);

      // If diffAbs is zero, t_eq and s_eq should be one,
      model.addConstr(t_eq >= 1 - diffTAbs);
      model.addConstr(s_eq >= 1 - diffSAbs);
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

  model.setObjective(obj, GRB_MINIMIZE);
}

LogicalResult OpenEdgeKernelScheduler::createSchedulerAndSolve() {
  GRBEnv env = GRBEnv("./gurobi.log");
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel model = GRBModel(env);
  int time_limit = 1500;
  model.set(GRB_DoubleParam_TimeLimit, time_limit);

  // Objective function
  GRBLinExpr obj = 0;

  std::map<Operation *, GRBVar> timeVarMap;
  std::map<Operation *, GRBVar> peVarMap;

  std::map<Block *, GRBVar> timeBlkEntry;
  std::map<Block *, GRBVar> timeBlkExit;
  initVariables(model, timeBlkEntry, timeBlkExit, timeVarMap, peVarMap);
  // assign the known schedule
  initKnownSchedule(model, timeVarMap, peVarMap);
  // llvm::errs() << "init known schedule\n";

  // create time constraints
  initOpTimeConstraints(model, timeVarMap, timeBlkEntry, timeBlkExit);
  // llvm::errs() << "Time constraints are initialized\n";

  // create space constraints
  initOpSpaceConstraints(model, peVarMap, timeVarMap, timeBlkEntry,
                         timeBlkExit);
  // llvm::errs() << "Space constraints are initialized\n";

  // create time and space constraints
  initOpTimeSpaceConstraints(model, timeVarMap, peVarMap);
  // llvm::errs() << "Time and Space constraints are initialized\n";

  // create the objective function
  GRBVar funcStartT =
      model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER, "t0");
  GRBVar funcEndT =
      model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER, "t1");
  initObjectiveFunction(model, funcStartT, funcEndT, timeVarMap, timeBlkEntry,
                        timeBlkExit, obj);

  // Optimize the model
  model.write("model.lp");
  model.optimize();

  // Check if the optimization status indicates infeasibility
  if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL ||
      model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL ||
      model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT) {
    if (model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT) {
      int solCount = model.get(GRB_IntAttr_SolCount);
      if (solCount == 0) {
        return failure();
      }
    }
  } else {
    return failure();
  }

  llvm::errs() << model.get(GRB_DoubleAttr_Runtime) << "s\n";

  std::ofstream csvFile("output.csv");
  // If the model is infeasible, write the model to solution
  for (auto [op, var] : timeVarMap) {
    writeOpResult(op, var.get(GRB_DoubleAttr_X),
                  peVarMap[op].get(GRB_DoubleAttr_X), -1);
    std::string str;
    llvm::raw_string_ostream rso(str);
    rso << *op;
    csvFile << rso.str() << "&" << var.get(GRB_StringAttr_VarName) << "&"
            << solution[op].time << "&"
            << peVarMap[op].get(GRB_StringAttr_VarName) << "&"
            << solution[op].pe << "\n";
  }
  csvFile.close();
  return success();
}
#endif