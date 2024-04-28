//===- CfMapToFullPredict.h - Fix index to CGRA PE bitwidth -----*- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --map-to-full-predict pass.
//
//===----------------------------------------------------------------------===//
#ifndef COMPIGRA_CFMAPTOFULLPREDICT_H
#define COMPIGRA_CFMAPTOFULLPREDICT_H

#include "compigra/Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace compigra {

std::unique_ptr<mlir::Pass> createCfMapToFullPredict();

} // namespace compigra
} // namespace mlir

#endif // COMPIGRA_CFMAPTOFULLPREDICT_H