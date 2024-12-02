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

namespace {
struct ASMGenTemporalCGRAPass
    : public compigra::impl::ASMGenTemporalCGRABase<ASMGenTemporalCGRAPass> {

  explicit ASMGenTemporalCGRAPass(int nRow, int nCol) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    auto funcOp = *modOp.getOps<func::FuncOp>().begin();
    std::string outDir = "out";

    Region &region = funcOp.getBody();
    OpBuilder builder(funcOp);
    TemporalCGRAScheduler scheduler(region, 3, nRow, nCol, builder);
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
std::unique_ptr<mlir::Pass> createASMGenTemporalCGRA(int nRow, int nCol) {
  return std::make_unique<ASMGenTemporalCGRAPass>(nRow, nCol);
}
} // namespace compigra
