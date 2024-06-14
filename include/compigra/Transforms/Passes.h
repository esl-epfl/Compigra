//===-  Passes.h - Tranformation Passes registration ------------*- C++ -*-===//
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

#ifndef COMPIGRA_TRANSFORMS_PASSES_H
#define COMPIGRA_TRANSFORMS_PASSES_H

#include "compigra/Transforms/CfFixIndexWidth.h"
#include "compigra/Transforms/CfMapToFullPredict.h"
#include "compigra/Transforms/CfReduceBranches.h"
#include "compigra/Transforms/CgraFitToOpenEdge.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace compigra {

#define GEN_PASS_REGISTRATION
#include "compigra/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCfFixIndexWidth();
std::unique_ptr<mlir::Pass> createCfMapToFullPredict();
std::unique_ptr<mlir::Pass> createCgraFitToOpenEdge(StringRef outputDAG);

} // end namespace compigra
#endif // COMPIGRA_TRANSFORMS_PASSES_H
