//===- compigra-opt.cpp - The polygeist-opt driver ------------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'compigra-opt' tool, which is the polygeist analog
// of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/Dialect.h"
// #include "polygeist/PolygeistOpsDialect.h.inc"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "compigra/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::LLVM::LLVMDialect, mlir::affine::AffineDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect>();

  // mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  return failed(
      mlir::MlirOptMain(argc, argv, "Compigra optimizer driver\n", registry));
}
