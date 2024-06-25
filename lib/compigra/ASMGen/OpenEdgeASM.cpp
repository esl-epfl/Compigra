//===- OpenEdge.cpp - Declare the functions for gen OpenEdge ASM*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements assembly generation functions for OpenEdge.
//
//===----------------------------------------------------------------------===//

#include "compigra/ASMGen/OpenEdgeASM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

using namespace mlir;
using namespace compigra;

/// Function to print the map of instructions
static void
printInstructions(const std::map<Operation *, Instruction> &instructions) {
  for (const auto &pair : instructions) {
    const Instruction &inst = pair.second;
    llvm::errs() << "Op: " << *pair.first << "\n"
                 << "Name: " << inst.name << " "
                 << "Time: " << inst.time << " "
                 << "PE: " << inst.pe << " "
                 << "Rout: " << inst.Rout << " "
                 << "OpA: " << inst.opA << " "
                 << "OpB: " << inst.opB << " "
                 << "Immediate: " << inst.immediate << "\n\n";
  }
}

/// Function to parse the scheduled results produced by SAT-MapIt line by line
/// and store the instruction in the map.
static LogicalResult readMapFile(std::string mapResult, unsigned maxReg,
                                 std::map<int, Instruction> &instructions) {
  std::ifstream file(mapResult);
  llvm::errs() << mapResult << "\n";
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file\n";
    return failure();
  }

  std::string line;

  bool parsing = false;

  // Read each line and parse it into the map
  while (std::getline(file, line)) {
    parsing = false;
    if (line.find("Id:") != std::string::npos) {
      parsing = true;
    }

    if (parsing)
      satmapit::parseLine(line, instructions, maxReg);
  }

  return success();
}

static Instruction initVoidInstruction(std::string name) {
  Instruction inst;
  inst.name = name;
  inst.time = INT_MAX;
  inst.pe = -1;
  inst.Rout = -1;
  inst.opA = "Unkown";
  inst.opB = "Unkown";
  inst.immediate = 0;
  return inst;
}

/// Initialize the block arguments to be SADD
static LogicalResult initBlockArgs(Block *block,
                                   std::map<int, Instruction> &instructions,
                                   OpBuilder &builder) {
  // Get the phi nodes in the block
  int nPhi = 0;
  for (auto [index, inst] : instructions) {
    if (inst.name != "phi")
      break;
    nPhi++;
  }

  if (nPhi != block->getNumArguments())
    return failure();

  Block *prevNode = block->getPrevNode();
  // Initialize the block arguments to be SADD
  for (size_t i = 0; i < nPhi; i++) {
    // insert constant Zero and SADD
    builder.setInsertionPoint(prevNode->getTerminator());
    auto zeroOp = builder.create<LLVM::ConstantOp>(
        prevNode->getTerminator()->getLoc(), builder.getIntegerType(32),
        builder.getI32IntegerAttr(0));

    builder.setInsertionPointToStart(block);
    auto addOp = builder.create<LLVM::AddOp>(
        block->getArgument(i).getLoc(), builder.getIntegerType(32),
        block->getArgument(i), zeroOp.getResult());
    // Replace the block argument with the add operation
    block->getArgument(i).replaceUsesWithIf(
        addOp.getResult(),
        [&](OpOperand &operand) { return operand.getOwner() != addOp; });
    // Revise the instruction map
    instructions[i].name = "add";
  }
  return success();
}

int OpenEdgeASMGen::getEarliestExecutionTime(Operation *op) {
  if (solution.find(op) != solution.end())
    return solution[op].time;
  return INT_MAX;
};

int OpenEdgeASMGen::getEarliestExecutionTime(Value val) {
  if (val.getDefiningOp())
    return getEarliestExecutionTime(val.getDefiningOp());
  return INT_MAX;
};

int OpenEdgeASMGen::getEarliestExecutionTime(Block *block) {
  int time = INT_MAX;
  // Get the earliest execution of operations in the block
  for (Operation &op : block->getOperations()) {
    time = std::min(time, getEarliestExecutionTime(&op));
  }
  return time;
}

int OpenEdgeASMGen::getConnectedBlock(int pe, std::string direction) {
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
  } else if (direction == "RCT") {
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

static Operation *getCntOpIndirectly(Operation *userOp, Operation *op) {
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
    llvm::errs() << "correspond argIndex: " << argIndex << "\n";
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

#ifdef HAVE_GUROBI
void OpenEdgeASMGen::initVariables(GRBModel &model,
                                   std::map<Block *, GRBVar> &timeBlkEntry,
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
      llvm::errs() << "init: " << bbId << "_" << opId << " : " << op << "\n";
      spaceOpVar[&op] = model.addVar(0.0, nCol * nRow - 1, 0.0, GRB_INTEGER,
                                     "pe_" + std::to_string(bbId) + "_" +
                                         std::to_string(opId));
    }
  }
}

void OpenEdgeASMGen::initKnownSchedule(
    GRBModel &model, std::map<Operation *, GRBVar> &timeOpVar,
    std::map<Operation *, GRBVar> &spaceOpVar) {
  for (auto [op, inst] : knownRes) {
    model.addConstr(timeOpVar[op] == inst.time);
    model.addConstr(spaceOpVar[op] == inst.pe);
  }
}

void OpenEdgeASMGen::initOpTimeConstraints(
    GRBModel &model, std::map<Operation *, GRBVar> &timeOpVar,
    std::map<Block *, GRBVar> &timeBlkEntry,
    std::map<Block *, GRBVar> &timeBlkExit) {
  for (auto [op, var] : timeOpVar) {
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
}

/// Add constraints to the model that y must be one of the neighbors of x
static void addNeighborConstraints(GRBModel &model, GRBVar &x, GRBVar &y,
                                   int nRow, int nCol) {
  // Create helper variables for the possible neighbors
  GRBVar left = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
  GRBVar right = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
  GRBVar top = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
  GRBVar bottom = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);

  // Auxiliary variables for calculations
  GRBVar xRow = model.addVar(0, nRow - 1, 0, GRB_INTEGER);
  GRBVar xCol = model.addVar(0, nCol - 1, 0, GRB_INTEGER);

  // Constraints to calculate row and column indices of x
  // xRow == x / nCol
  model.addConstr(xRow == (x - xCol) / nCol);
  // xCol == x % nCol
  GRBVar u = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
  model.addConstr(x == u * nCol + xCol);

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

  // Add constraints that y must be one of the neighbors
  GRBVar chooseLeft = model.addVar(0, 1, 0, GRB_BINARY);
  GRBVar chooseRight = model.addVar(0, 1, 0, GRB_BINARY);
  GRBVar chooseTop = model.addVar(0, 1, 0, GRB_BINARY);
  GRBVar chooseBottom = model.addVar(0, 1, 0, GRB_BINARY);
  GRBVar chooseSelf = model.addVar(0, 1, 0, GRB_BINARY);

  model.addQConstr(y == left * chooseLeft + right * chooseRight +
                            top * chooseTop + bottom * chooseBottom +
                            x * chooseSelf);
  model.addConstr(
      chooseLeft + chooseRight + chooseTop + chooseBottom + chooseSelf == 1);
}

void OpenEdgeASMGen::initOpSpaceConstraints(
    GRBModel &model, std::map<Operation *, GRBVar> &spaceOpVar) {

  for (auto [op, var] : spaceOpVar) {
    // BrOp does not require to consider data dependency; or if found op in
    // result, the space hase been assigned, continue
    if (isa<LLVM::BrOp>(op) || knownRes.find(op) != knownRes.end())
      continue;

    knownRes[op] = initVoidInstruction(op->getName().getStringRef().str());
    // assign the space w.r.t to its successor's PE
    bool findPE = false;
    for (auto userOp : op->getUsers()) {
      // Get the real userOp if the userOp is branchOp or conditionalOp
      auto cntOp = getCntOpIndirectly(userOp, op);
      if (knownRes.find(cntOp) != knownRes.end() && knownRes[cntOp].Rout >= 0) {
        // If the result is stored in the register, assign the same PE
        bool leftOp =
            cntOp->getOperand(0) == getCorrelatedVal(op->getResult(0));
        std::string direct = leftOp ? knownRes[cntOp].opA : knownRes[cntOp].opB;
        // Get the last char of direct
        int dstPE = getConnectedBlock(knownRes[cntOp].pe, direct);
        model.addConstr(var == dstPE);
        knownRes[op].pe = dstPE;

        findPE = true;
        break;
      }
    }

    // If the PE has been assigned, continue
    if (findPE)
      continue;

    // DEBUG: assign around its predecessor
    for (auto [ind, opVal] : llvm::enumerate(op->getOperands())) {
      if (isa<LLVM::ConstantOp>(opVal.getDefiningOp()))
        continue;
      // TODO: consider the effect of branchOp and conditionalOp
      auto cntOp = opVal.getDefiningOp();
      auto cntPE = spaceOpVar[cntOp];
      // Get the left, right, top, bottom PE

      addNeighborConstraints(model, cntPE, var, nRow, nCol);
    }
  }
}

/// The combination of time and space variables should be unique
void OpenEdgeASMGen::initOpTimeSpaceConstraints(
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

void OpenEdgeASMGen::initObjectiveFunction(
    GRBModel &model, GRBVar &funcStartT, GRBVar &funcEndT,
    std::map<Operation *, GRBVar> &timeOpVar,
    std::map<Block *, GRBVar> &timeBlkEntry,
    std::map<Block *, GRBVar> &timeBlkExit) {
  GRBLinExpr obj = 0;
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

LogicalResult OpenEdgeASMGen::createSchedulerAndSolve() {
  GRBEnv env = GRBEnv("./gurobi.log");
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel model = GRBModel(env);
  std::map<Operation *, GRBVar> timeVarMap;
  std::map<Operation *, GRBVar> peVarMap;

  std::map<Block *, GRBVar> timeBlkEntry;
  std::map<Block *, GRBVar> timeBlkExit;
  initVariables(model, timeBlkEntry, timeBlkExit, timeVarMap, peVarMap);
  llvm::errs() << "initVariables\n";
  // assign the known schedule
  initKnownSchedule(model, timeVarMap, peVarMap);
  llvm::errs() << "initKnownSchedule\n";
  // create time constraints
  initOpTimeConstraints(model, timeVarMap, timeBlkEntry, timeBlkExit);
  llvm::errs() << "initOpTimeConstraints\n";
  // create space constraints
  initOpSpaceConstraints(model, peVarMap);
  llvm::errs() << "initOpSpaceConstraints\n";
  // create time and space constraints
  initOpTimeSpaceConstraints(model, timeVarMap, peVarMap);

  // create the objective function
  GRBVar funcStartT =
      model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER, "t0");
  GRBVar funcEndT =
      model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_INTEGER, "t1");
  initObjectiveFunction(model, funcStartT, funcEndT, timeVarMap, timeBlkEntry,
                        timeBlkExit);
  // Optimize the model
  model.optimize();
  model.write("model.lp");

  // Check if the optimization status indicates infeasibility
  if (model.get(GRB_IntAttr_Status) == GRB_INFEASIBLE) {
    // Model is infeasible
    return failure();
  }

  // If the model is infeasible, write the model to solution
  for (auto [op, var] : timeVarMap) {
    writeOpResult(op, var.get(GRB_DoubleAttr_X),
                  peVarMap[op].get(GRB_DoubleAttr_X), -1);
  }
}
#endif

LogicalResult
OpenEdgeASMGen::assignSchedule(mlir::Block::OpListType &ops,
                               std::map<int, Instruction> instructions) {
  for (auto [ind, op] : llvm::enumerate(ops)) {
    knownRes[&op] = instructions[ind];
  }
  return success();
}

/// Get the loop block of the region
static Block *getLoopBlock(Region &region) {
  for (auto &block : region)
    for (auto suc : block.getSuccessors())
      if (suc == &block)
        return &block;
  return nullptr;
}

namespace {
struct OpenEdgeASMGenPass
    : public compigra::impl::OpenEdgeASMGenBase<OpenEdgeASMGenPass> {

  explicit OpenEdgeASMGenPass(StringRef funcName, StringRef mapResult) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    unsigned maxReg = 3;

    llvm::errs() << mapResult << "\n";
    std::map<int, Instruction> instructions;
    if (failed(readMapFile(mapResult, maxReg, instructions)))
      return signalPassFailure();

    for (auto funcOp :
         llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>())) {
      if (funcOp.getName() != funcName)
        continue;
      auto &r = funcOp.getBody();
      // init block arguments to be SADD
      if (failed(initBlockArgs(getLoopBlock(r), instructions, builder)))
        return signalPassFailure();
      // init OpenEdgeASMGen
      OpenEdgeASMGen asmGen(r, maxReg, 4);
      asmGen.assignSchedule(getLoopBlock(r)->getOperations(), instructions);
      printInstructions(asmGen.knownRes);
      asmGen.createSchedulerAndSolve();

      // llvm::errs() << funcOp << "\n";
    }
    // Assign operations in the init phase
  }
};
} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createOpenEdgeASMGen(StringRef funcName,
                                                 StringRef mapResult) {
  return std::make_unique<OpenEdgeASMGenPass>(funcName, mapResult);
}
} // namespace compigra