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

    Region &region = funcOp.getBody();
    OpBuilder builder(funcOp);
    TemporalCGRAScheduler scheduler(region, 3, 3, 3, builder);
    scheduler.createSchedulerAndSolve();
    llvm::errs() << modOp << "\n";
    // if (failed(scheduler.createSchedulerAndSolve()))
    //   return signalPassFailure();
  }
};

} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createASMGenTemporalCGRA(int nRow, int nCol) {
  return std::make_unique<ASMGenTemporalCGRAPass>(nRow, nCol);
}
} // namespace compigra
