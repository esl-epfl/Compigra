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
#include <stack>

using namespace mlir;

namespace compigra {
/// Determine whether the value is block argument, or the source operand of the
/// block argument
bool isPhiRelatedValue(Value val);

/// Given a value(block argument or the source operand of the block argument),
/// get all the related values (all source operands and the block arguments)
void getAllPhiRelatedValues(Value val, SetVector<Value> &relatedVals);

/// Get the execution path from srcBlk to dstBlk
std::stack<Block *> getBlockPath(Block *srcBlk, Block *dstBlk);

/// Get all the paths from the operation to the end of the block
void getAllPathsToBlockEnd(Operation *op,
                           std::vector<SmallVector<Operation *, 8>> &allPaths);

/// From the start of op's block to the earliest start execution time of op
unsigned getEarliestStartTime(Operation *op);

/// The latest end time of op to the end of the block
unsigned getLatestEndTime(Operation *op);

/// Get the critical path of the block execution
SmallVector<Operation *, 8> getCriticalPath(Block *blk);

/// Calculate the shortest execution path from srcOp to dstOp. dstOp must
/// consume srcOp's result. Return UINT64_MAX if there is no path.
unsigned getShortestLiveHops(Operation *srcOp, Operation *dstOp);

/// Determine whether the srcBlk is the predecessor of dstBlk
bool isBackEdge(Block *srcBlk, Block *dstBlk);

bool isBackEdge(Operation *srcOp, Operation *dstOp);
} // namespace compigra

#endif // UTILS_H
