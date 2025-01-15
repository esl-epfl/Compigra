//===- Utils.h - Declares helper functions *- C++-* -----------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions for temporal CGRA schedule.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_H
#define UTILS_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;

namespace compigra {
/// Determine whether the value is block argument, or the source operand of the
/// block argument
bool isPhiRelatedValue(Value val);

/// Determine whether the srcBlk is the predecessor of dstBlk
bool isBackEdge(Block *srcBlk, Block *dstBlk);

bool isBackEdge(Operation *srcOp, Operation *dstOp);
} // namespace compigra

#endif // UTILS_H
