//===- compigra-opt.cpp - The compigra-opt driver -------------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'compigra-opt' tool, which is the compigra analog
// of mlir-opt, used to drive compiler passes.
//
//===----------------------------------------------------------------------===//

#include "InitAllDialects.h"
#include "InitAllPasses.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  mlir::registerCSEPass();
  // Register affine passes correctly
  // mlir::affine::registerAffinePasses();
  mlir::registerConvertAffineToStandardPass();

  compigra::registerAllPasses();
  compigra::registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "Compigra optimizer driver\n", registry));
}
