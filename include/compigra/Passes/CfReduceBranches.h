//===- CfReduceBranches.h - Reduce branches for bb merge   ------*- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --reduce-branches pass.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_CFREDUCEBRANCHES_H
#define COMPIGRA_CFREDUCEBRANCHES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

namespace compigra {

#define GEN_PASS_DEF_CFREDUCEBRANCHES
#define GEN_PASS_DECL_CFREDUCEBRANCHES
#include "compigra/Passes/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCfReduceBranches();

} // namespace compigra

#endif // COMPIGRA_CFREDUCEBRANCHES_H