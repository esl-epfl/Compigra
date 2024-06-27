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
#include "compigra/CgraDialect.h"
#include "compigra/CgraOps.h"
#include "compigra/Scheduler/KernelSchedule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

#include <iomanip>

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
                 << "OpB: " << inst.opB << "\n\n";
  }
}

static bool isRegisterDigit(const std::string &reg, unsigned maxReg) {
  if (reg.empty() || reg[0] != 'R')
    return false;
  for (size_t i = 1; i < reg.size(); i++) {
    if (!isdigit(reg[i]))
      return false;
  }
  return std::stoi(reg.substr(1)) <= maxReg;
}

LogicalResult OpenEdgeASMGen::allocateRegisters(
    std::map<Operation *, Instruction> restriction) {

  // First write restriction to the solution
  for (auto [op, inst] : restriction) {
    instSolution[op] = inst;
    solution[op].reg = inst.Rout;
  }

  // Allocate registers based on each PE
  for (size_t pe = 0; pe < nRow * nCol; pe++) {
    auto ops = getOperationsAtPE(pe);
    // If the operation is used by user in other PE, it must be stored in the
    // Rout
    for (auto [time, op] : ops) {
      // Check whether the PE of the operation is known, if yes, skil allocation
      if (solution[op].reg != -1)
        continue;

      for (auto &use : llvm::make_early_inc_range(op->getUses())) {
        Operation *user = getCntOpIndirectly(use.getOwner(), op);
        // if the user PE is not restricted, don't allocate register for now
        int userPE = solution[user].pe;
        if (solution[user].reg == -1)
          continue;

        // Check the PE of the user
        if (userPE != pe) {
          solution[op].reg = maxReg;
        } else {
          // The result operand should match with use operand
          unsigned operandIndex = use.getOperandNumber();
          auto regStr = operandIndex == 0 ? instSolution[user].opA
                                          : instSolution[user].opB;
          if (regStr == "ROUT")
            solution[op].reg = maxReg;
          else {
            if (isRegisterDigit(regStr, maxReg))
              solution[op].reg = std::stoi(regStr.substr(1));
            else
              return failure();
          }
        }
        // If the correspond PE is set, stop seeking from its other users

        break;
      }
    }
  }

  // TODO[@Yuxuan]: Register allocation algorithm required for operations cannot
  // derived from restrictions.
  for (auto [op, sol] : solution) {
    if (op->getNumResults() > 0 && sol.reg == -1)
      return failure();
    instSolution[op].Rout = sol.reg;
  }
  llvm::errs() << "Register allocation done\n";
  // write register allocation results to instructions
  convertToInstructionMap();
  return success();
}

int OpenEdgeASMGen::getEarliestExecutionTime(Operation *op) {
  if (solution.find(op) != solution.end())
    return solution[op].time;
  return INT_MAX;
};

int OpenEdgeASMGen::getKernelStart() {
  int time = INT_MAX;
  for (auto [op, inst] : solution)
    if (inst.time < time)
      time = inst.time;
  return time;
};

int OpenEdgeASMGen::getKernelEnd() {
  int time = getKernelStart();
  for (auto [op, inst] : solution)
    if (inst.time > time)
      time = inst.time;
  return time;
};

int OpenEdgeASMGen::getEarliestExecutionTime(Block *block) {
  int time = INT_MAX;
  // Get the earliest execution of operations in the block
  for (Operation &op : block->getOperations()) {
    time = std::min(time, getEarliestExecutionTime(&op));
  }
  return time;
}

/// Get the loop block of the region
static Block *getLoopBlock(Region &region) {
  for (auto &block : region)
    for (auto suc : block.getSuccessors())
      if (suc == &block)
        return &block;
  return nullptr;
}

std::map<int, Operation *> OpenEdgeASMGen::getOperationsAtTime(int time) {
  std::map<int, Operation *> ops;
  for (auto [op, inst] : solution)
    if (inst.time == time)
      ops[inst.pe] = op;

  return ops;
}

std::map<int, Operation *> OpenEdgeASMGen::getOperationsAtPE(int pe) {
  std::map<int, Operation *> ops;
  for (auto [op, inst] : solution)
    if (inst.pe == pe)
      ops[inst.time] = op;

  return ops;
}

// Helper function to pad strings to a fixed width
static std::string padString(const std::string &str, size_t width) {
  if (str.length() >= width) {
    return str;
  }
  return str +
         std::string(width - str.length(), ' '); // Pad with spaces on the right
}

static int getOpIndex(Operation *op) {
  auto block = op->getBlock();
  int index = 0;
  for (auto &iterOp : block->getOperations()) {
    if (op == &iterOp)
      return index;
    index++;
  }
  return -1;
}

/// Function for retrieve the operand knowning the src operation and user
/// operand's definition operand's PES.
/// e.g. Suppose %a = op %b, ..., return the string of the %b  for %a, such as
/// R0, R1,... or RCT, RCB, RCR, RCL.
static std::string getOperandSrcReg(int peA, int peB, int srcReg, int nRow,
                                    int nCol, unsigned maxReg) {
  if (peA == peB)
    if (srcReg == maxReg)
      return "Rout";
    else
      return "R" + std::to_string(srcReg);

  // check userPE in the neighbour of srcPE
  // check whether peB on peA right
  if ((peA + 1) % nCol == peB % nCol && (peA / nCol == peB / nCol))
    return "RCR";

  // check whether peB on peA left
  if ((peB + 1) % nCol == peA % nCol && (peA / nCol == peB / nCol))
    return "RCL";

  // check whether peB on peA top
  if (((peA - nCol + (nRow * nCol)) % (nRow * nCol)) == peB)
    return "RCT";

  // check whether peB on peA bottom
  if (((peA + nCol) % (nRow * nCol)) == peB)
    return "RCB";

  return "ERROR";
}

LogicalResult OpenEdgeASMGen::convertToInstructionMap() {
  for (auto [op, unit] : solution) {
    Instruction inst;
    if (instSolution.find(op) != instSolution.end())
      inst = instSolution[op];
    else
      inst = Instruction{op->getName().getStringRef().str(),
                         unit.time,
                         unit.pe,
                         unit.reg,
                         "Unknown",
                         "Unknown"};

    if (isa<LLVM::BrOp, LLVM::ConstantOp>(op)) {
      continue;
    }

    // Get defition operation
    if (op->getNumOperands() > 0 && inst.opA == "Unknown") {
      auto producerA = op->getOperand(0).getDefiningOp();
      if (isa<LLVM::ConstantOp>(producerA))
        // assign opA to be Imm
        inst.opA = std::to_string(
            producerA->getAttrOfType<IntegerAttr>("value").getInt());
      else if (solution.find(producerA) != solution.end()) {
        inst.opA =
            getOperandSrcReg(unit.pe, solution[producerA].pe,
                             solution[producerA].reg, nRow, nCol, maxReg);
        llvm::errs() << unit.pe << " " << solution[producerA].pe << " "
                     << solution[producerA].reg << "\n";
      } else
        return failure();
    }

    if (op->getNumOperands() > 1 && inst.opB == "Unknown") {
      auto producerB = op->getOperand(1).getDefiningOp();
      if (isa<LLVM::ConstantOp>(producerB))
        // assign opA to be Imm
        inst.opB = std::to_string(
            producerB->getAttrOfType<IntegerAttr>("value").getInt());
      else if (solution.find(producerB) != solution.end()) {
        inst.opB =
            getOperandSrcReg(unit.pe, solution[producerB].pe,
                             solution[producerB].reg, nRow, nCol, maxReg);

        llvm::errs() << unit.pe << " " << solution[producerB].pe << " "
                     << solution[producerB].reg << "\n";
      } else
        return failure();
    }

    instSolution[op] = inst;
  }
  return success();
}

std::string OpenEdgeASMGen::printInstructionToISA(Operation *op,
                                                  bool dropNeighbourBr) {
  // If it is return, return EXIT
  if (isa<LLVM::ReturnOp>(op))
    return "EXIT";
  // Drop the dialect prefix
  size_t pos = op->getName().getStringRef().find(".");
  auto opName = op->getName().getStringRef().substr(pos + 1).str();
  if (isa<LLVM::AddOp, LLVM::MulOp>(op))
    opName = "s" + opName;
  // make opName capital
  for (auto &c : opName) {
    c = std::toupper(c);
  }

  // If the operation is branchOp, find the branch target
  if (auto brOp = dyn_cast<LLVM::BrOp>(op)) {
    auto block = brOp->getSuccessor(0);
    int sucTime = getEarliestExecutionTime(block);
    int curTime = getEarliestExecutionTime(op);
    if (sucTime == curTime + 1 && dropNeighbourBr)
      return "NOP";
    return "BR " + std::to_string(sucTime + baseTime);
  }

  std::string ROUT = "";
  if (op->getNumResults() > 0)
    ROUT = instSolution[op].Rout == maxReg
               ? " ROUT,"
               : " R" + std::to_string(instSolution[op].Rout) + ",";

  std::string opA = "";
  if (op->getNumOperands() > 0) {
    // check whether the operand is immediate
    auto cntOp = getCntOpIndirectly(op->getOperand(0))[0];
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(cntOp)) {
      int imm = constOp.getValueAttr().dyn_cast<IntegerAttr>().getInt();
      opA = imm ? " " + std::to_string(imm) : " ZERO";
    } else
      opA = " " + instSolution[op].opA;
  }

  std::string opB = "";
  if (op->getNumOperands() > 1) {
    auto cntOp = getCntOpIndirectly(op->getOperand(1))[0];
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(cntOp)) {
      int imm = constOp.getValueAttr().dyn_cast<IntegerAttr>().getInt();
      opB = imm ? " " + std::to_string(imm) : " ZERO";
    } else
      opB = "," + instSolution[op].opB;
  }

  std::string addition = "";
  if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(op)) {
    auto block = op->getSuccessor(0);
    int sucTime = getEarliestExecutionTime(block);
    addition = " " + std::to_string(sucTime + baseTime);
  }

  return opName + ROUT + opA + opB + addition;
}

/// Print the known schedule
void OpenEdgeASMGen::printKnownSchedule(bool GridLIke, int startPC) {
  // For each time step
  initBaseTime(startPC);
  std::vector<std::vector<std::string>> asmCode;
  for (int t = getKernelStart(); t <= getKernelEnd(); t++) {
    // Get the operations scheduled at the time step
    auto ops = getOperationsAtTime(t);
    std::vector<std::string> asmCodeLine;
    bool isNOP = true;
    for (int i = 0; i < nRow; i++) {
      for (int j = 0; j < nCol; j++) {
        if (ops.find(i * nCol + j) != ops.end()) {
          std::string isa = printInstructionToISA(ops[i * nCol + j]);
          isNOP = (isa != "NOP") ? false : isNOP;
          asmCodeLine.push_back(isa);
        } else {
          asmCodeLine.push_back("NOP");
        }
      }
    }
    if (!isNOP)
      asmCode.push_back(asmCodeLine);
    // If the time step is empty, skip it and revise the base time
    else
      baseTime--;
  }

  for (int t = 0; t < asmCode.size(); t++) {
    // Print the instruction for each time step
    llvm::errs() << "Time = " << startPC + t << "\n";
    for (int i = 0; i < asmCode[t].size(); i++) {
      llvm::errs() << padString(asmCode[t][i], 20);
      if (!GridLIke)
        llvm::errs() << "\n";
      else if (i % nCol == nCol - 1)
        llvm::errs() << "\n";
    }
  }
}

/// Function to parse the scheduled results produced by SAT-MapIt line by line
/// and store the instruction in the map.
static LogicalResult readMapFile(std::string mapResult, unsigned maxReg,
                                 std::map<int, Instruction> &instructions) {
  std::ifstream file(mapResult);
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
      // init scheduler
      OpenEdgeKernelScheduler scheduler(r, maxReg, 4);
      scheduler.assignSchedule(getLoopBlock(r)->getOperations(), instructions);
      scheduler.createSchedulerAndSolve();
      // assign schedule
      asmGen.setSolution(scheduler.getSolution());
      asmGen.allocateRegisters(scheduler.knownRes);
      asmGen.printKnownSchedule();
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