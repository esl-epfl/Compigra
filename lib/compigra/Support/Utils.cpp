//===- Utils.cpp - Implement helper functions *- C++-* --------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helper functions for temporal CGRA schedule.
//
//===----------------------------------------------------------------------===//

#include "compigra/Support/Utils.h"
#include <stack>
#include <unordered_set>

namespace compigra {
bool isPhiRelatedValue(Value val) {
  if (val.isa<BlockArgument>())
    return true;

  for (auto &use : val.getUses()) {
    // cf.br carries the block argument
    if (isa<cf::BranchOp>(use.getOwner()))
      return true;

    // cgra.cond_br carries the block argument after the condition
    if (isa<cgra::ConditionalBranchOp>(use.getOwner()) &&
        use.getOperandNumber() > 1)
      return true;
  }
  return false;
}

bool isBackEdge(Block *srcBlk, Block *dstBlk) {
  auto &entryBlock = srcBlk->getParent()->front();
  // start DFS from entry block, if dstBlk is visited before srcBlk, it is a
  // back edge
  std::unordered_set<Block *> visited;
  std::stack<Block *> stack;
  stack.push(&entryBlock);
  while (!stack.empty()) {
    auto currBlk = stack.top();
    stack.pop();
    if (visited.find(currBlk) != visited.end())
      continue;
    visited.insert(currBlk);
    for (auto succBlk : currBlk->getSuccessors()) {
      if (succBlk == dstBlk)
        return visited.find(srcBlk) == visited.end();
      stack.push(succBlk);
    }
  }
  return false;
}

bool isBackEdge(Operation *srcOp, Operation *dstOp) {
  /// if the dstOp directly consumes the result of srcOp, and they are in the
  /// same block, it is not a back edge
  if (srcOp->getBlock() == dstOp->getBlock()) {
    for (auto opr : dstOp->getOperands()) {
      if (opr == srcOp->getResult(0))
        return false;
    }
    return true;
  }

  return isBackEdge(srcOp->getBlock(), dstOp->getBlock());
}
} // namespace compigra
