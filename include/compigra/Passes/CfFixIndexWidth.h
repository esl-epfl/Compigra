//===- CfFixIndexWidth.h - Fix index to CGRA PE bitwidth --------*- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --fix-index-width pass.
//
//===----------------------------------------------------------------------===//
#ifndef COMPIGRA_CFFIXINDEXWIDTH_H
#define COMPIGRA_CFFIXINDEXWIDTH_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace compigra {

#define GEN_PASS_DEF_CFFIXINDEXWIDTH
#define GEN_PASS_DECL_CFFIXINDEXWIDTH
#include "compigra/Passes/Passes.h.inc"
// #define GEN_PASS_DEF_CFMAPTOFULLPREDICT

std::unique_ptr<mlir::Pass> createCfFixIndexWidth();

} // namespace compigra

#endif // COMPIGRA_CFFIXINDEXWIDTH_H