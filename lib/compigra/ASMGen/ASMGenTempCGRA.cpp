//===- ASMGenOpenEdge.cpp - Implements the functions for temporal CGRA ASM
// generation *- C++ -*----------------------------------------------------===//
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

#include "compigra/ASMGen/ASMGenTempCGRA.h"
#include "compigra/ASMGen/OpenEdgeASM.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraOps.h"
#include "compigra/Scheduler/KernelSchedule.h"
#include "compigra/Scheduler/ModuloScheduleAdapter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <set>

using namespace mlir;
using namespace compigra;

static LogicalResult
outputDATE2023DAG(func::FuncOp funcOp, std::string outputDAG,
                  std::string pythonExectuable, Region &r, OpBuilder &builder,
                  unsigned peGridSize = 4, unsigned maxReg = 3) {

  // Find the loop block
  int bbInd = -1;
  for (auto &blk : llvm::make_early_inc_range(funcOp.getBlocks())) {
    bbInd++;
    bool isLoop =
        std::find(blk.getSuccessors().begin(), blk.getSuccessors().end(),
                  &blk) != blk.getSuccessors().end();
    if (!isLoop)
      continue;
    // Get the oeprations in the loop block
    SmallVector<Operation *> nodes;
    for (Operation &op : blk.getOperations()) {
      nodes.push_back(&op);
    }

    // initialize print function
    satmapit::PrintSatMapItDAG printer(blk.getTerminator(), nodes);
    printer.init();
    if (failed(printer.printDAG(outputDAG + "/bb" + std::to_string(bbInd))))
      continue;

    // detect whether the python executable exist
    std::string command = pythonExectuable + " --path " + outputDAG +
                          "/ --bench bb" + std::to_string(bbInd) + " --unit " +
                          std::to_string(peGridSize) + " > " + outputDAG +
                          "/out_raw_bb" + std::to_string(bbInd) + ".sat\n";

    // call the python code script to solve the MS
    llvm::errs() << "Running the SAT-Solver\n";
    int result = system(command.c_str());
    if (result != 0)
      continue;
    llvm::errs() << "SAT-solver done\n";
    // read the result and update the schedule
    std::string mapResult =
        outputDAG + "/out_raw_bb" + std::to_string(bbInd) + ".sat";
    int opSize = blk.getOperations().size();

    int II;
    std::map<int, Instruction> instructions;
    std::map<int, std::unordered_set<int>> opTimeMap;
    std::vector<std::unordered_set<int>> bbTimeMap = {{}};
    if (failed(readMapFile(mapResult, maxReg,
                           opSize + blk.getNumArguments() - 1, II, opTimeMap,
                           bbTimeMap, instructions)))
      continue;
    // does not overlap the loop execution, not necessary to update the schedule
    if (!kernelOverlap(bbTimeMap))
      continue;

    if (failed(initBlockArgs(&blk, instructions, builder)))
      return failure();

    // Adapt the CFG with the loop modulo schedule
    std::map<int, int> execTime = getLoopOpUnfoldExeTime(opTimeMap);
    ModuloScheduleAdapter adapter(r, builder, blk.getOperations(), II, execTime,
                                  opTimeMap, bbTimeMap);
    llvm::errs() << "Adapt the loop block with the schedule result\n";

    if (failed(adapter.adaptCFGWithLoopMS()))
      return failure();
    llvm::errs() << "Adapt one block\n";
  }
  return success();
}

namespace {
struct ASMGenTemporalCGRAPass
    : public compigra::impl::ASMGenTemporalCGRABase<ASMGenTemporalCGRAPass> {

  explicit ASMGenTemporalCGRAPass(int nRow, int nCol, int mem, StringRef msOpt,
                                  StringRef asmOutDir) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    auto funcOp = *modOp.getOps<func::FuncOp>().begin();
    if (asmOutDir.empty())
      asmOutDir = "out";
    std::string outDir = asmOutDir;

    Region &region = funcOp.getBody();
    OpBuilder builder(funcOp);

    size_t lastSlashPos = outDir.find_last_of("/");
    if (failed(outputDATE2023DAG(
            funcOp, outDir.substr(0, lastSlashPos) + "/IR_opt/satmapit",
            msOpt.substr(1, msOpt.size() - 2), region, builder, nRow, 3)))
      return signalPassFailure();
    return;

    TemporalCGRAScheduler scheduler(region, 3, nRow, nCol, builder);
    scheduler.setReserveMem(mem);
    if (failed(scheduler.createSchedulerAndSolve())) {
      llvm::errs() << "Failed to create scheduler and solve\n";
      return signalPassFailure();
    }

    // assign schedule results and produce assembly
    // scheduler.readScheduleResult("temporalSpatialSchedule.csv");
    OpenEdgeASMGen asmGen(region, 3, nRow);
    asmGen.setSolution(scheduler.getSolution());
    if (failed(asmGen.allocateRegisters(scheduler.knownRes))) {
      llvm::errs() << "Failed to allocate registers\n";
      return signalPassFailure();
    }
    asmGen.printKnownSchedule(true, 0, outDir);
  }
};

} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createASMGenTemporalCGRA(int nRow, int nCol,
                                                     int mem, StringRef msOpt,
                                                     StringRef asmOutDir) {
  return std::make_unique<ASMGenTemporalCGRAPass>(nRow, nCol, mem, msOpt,
                                                  asmOutDir);
}
} // namespace compigra
