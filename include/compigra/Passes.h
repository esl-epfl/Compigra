//===-  Passes.h - Passes registration --------------------------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_PASSES_H
#define COMPIGRA_PASSES_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace compigra {

#define GEN_PASS_DEF_CFFIXINDEXWIDTH
#include "compigra/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCfFixIndexWidth();
} // end namespace compigra
} // end namespace mlir

#endif // COMPIGRA_PASSES_H
