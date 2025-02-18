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

static LogicalResult outputDATE2023DAG(func::FuncOp funcOp,
                                       std::string outputDAG,
                                       std::string pythonExectuable,
                                       unsigned peGridSize = 4) {

  // Find the loop block
  for (auto [ind, blk] : llvm::enumerate(funcOp.getBlocks())) {

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
    if (failed(printer.printDAG(outputDAG + "/bb" + std::to_string(ind))))
      continue;

    // detect whether the python executable exist
    std::string command = pythonExectuable + " --path " + outputDAG +
                          "/ --bench bb" + std::to_string(ind) + " --unit " +
                          std::to_string(peGridSize) + " > " + outputDAG +
                          "/out_raw_bb" + std::to_string(ind) + ".sat\n";
    llvm::errs() << "Running the SAT-Solver\n";
    int result = system(command.c_str());
    if (result != 0)
      continue;

    // call the python code script to solve the MS
  }
  return success();
}
namespace {
struct ASMGenTemporalCGRAPass
    : public compigra::impl::ASMGenTemporalCGRABase<ASMGenTemporalCGRAPass> {

  explicit ASMGenTemporalCGRAPass(int nRow, int nCol, int mem,
                                  StringRef asmOutDir) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    auto funcOp = *modOp.getOps<func::FuncOp>().begin();
    if (asmOutDir.empty())
      asmOutDir = "out";
    std::string outDir = asmOutDir;
    std::string pythonExectuable =
        "python3  /home/yuxuan/Projects/24S/SAT-MapIt/Mapper/main.py";

    Region &region = funcOp.getBody();
    OpBuilder builder(funcOp);
    TemporalCGRAScheduler scheduler(region, 3, nRow, nCol, builder);
    scheduler.setReserveMem(mem);

    size_t lastSlashPos = outDir.find_last_of("/");
    if (failed(outputDATE2023DAG(
            funcOp, outDir.substr(0, lastSlashPos) + "/IR_opt/satmapit",
            pythonExectuable)))
      return signalPassFailure();
    return;

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
std::unique_ptr<mlir::Pass>
createASMGenTemporalCGRA(int nRow, int nCol, int mem, StringRef asmOutDir) {
  return std::make_unique<ASMGenTemporalCGRAPass>(nRow, nCol, mem, asmOutDir);
}
} // namespace compigra
