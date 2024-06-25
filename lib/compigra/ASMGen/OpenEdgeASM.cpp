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
#include "compigra/Scheduler/KernelSchedule.h"
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
      // init scheduler
      OpenEdgeKernelScheduler scheduler(r, maxReg, 4);
      scheduler.assignSchedule(getLoopBlock(r)->getOperations(), instructions);
      scheduler.createSchedulerAndSolve();
      printInstructions(scheduler.knownRes);

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