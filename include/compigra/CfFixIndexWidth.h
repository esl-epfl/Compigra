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
#ifndef COMPIGRA_SCFFIXINDEXWIDTH_H
#define COMPIGRA_SCFFIXINDEXWIDTH_H

#include "compigra/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace compigra {



std::unique_ptr<mlir::Pass> createCfFixIndexWidth();

} // namespace compigra
} // namespace mlir

#endif // COMPIGRA_SCFFIXINDEXWIDTH_H