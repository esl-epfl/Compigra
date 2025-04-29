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

void getAllPhiRelatedValues(Value val, SetVector<Value> &relatedVals) {
  if (relatedVals.count(val))
    return;

  if (val.isa<BlockArgument>()) {
    relatedVals.insert(val);

    // Get the related values from the predecessor blocks
    unsigned ind = val.cast<BlockArgument>().getArgNumber();
    Block *block = val.getParentBlock();
    for (Block *pred : block->getPredecessors()) {
      Operation *branchOp = pred->getTerminator();
      Value operand = nullptr;
      if (auto br = dyn_cast<cf::BranchOp>(branchOp)) {
        operand = br.getOperand(ind);
      } else if (auto cbr = dyn_cast<cgra::ConditionalBranchOp>(branchOp)) {
        if (block == cbr.getTrueDest())
          operand = cbr.getTrueOperand(ind);
        else
          operand = cbr.getFalseOperand(ind);
      }
      relatedVals.insert(operand);
      // recursively get the related values
      getAllPhiRelatedValues(operand, relatedVals);
    }

    // Get the related values from the successor blocks
    // if val is used as branch argument
    for (auto &use : val.getUses()) {
      if (auto br = dyn_cast<cf::BranchOp>(use.getOwner())) {
        getAllPhiRelatedValues(
            br.getSuccessor()->getArgument(use.getOperandNumber()),
            relatedVals);
      }
      if (auto cbr = dyn_cast<cgra::ConditionalBranchOp>(use.getOwner())) {
        if (use.getOperandNumber() < 2)
          continue;
        if (use.getOperandNumber() < 2 + cbr.getNumTrueDestOperands()) {
          getAllPhiRelatedValues(
              cbr.getTrueDest()->getArgument(use.getOperandNumber() - 2),
              relatedVals);
        } else {
          getAllPhiRelatedValues(
              cbr.getFalseDest()->getArgument(use.getOperandNumber() -
                                              cbr.getNumTrueDestOperands() - 2),
              relatedVals);
        }
      }
    }
    return;
  }

  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    llvm::errs() << "Can not track " << val << " for phi related chain\n";
    return;
  }
  relatedVals.insert(val);
  for (auto &use : val.getUses()) {
    unsigned index = use.getOperandNumber();
    if (auto br = dyn_cast_or_null<cf::BranchOp>(use.getOwner())) {
      getAllPhiRelatedValues(br->getSuccessor(0)->getArgument(index),
                             relatedVals);
    }
    if (auto cbr =
            dyn_cast_or_null<cgra::ConditionalBranchOp>(use.getOwner())) {
      // Only for flag comparison not for phi value propagation
      if (index < 2)
        continue;
      bool isTrueOpr = index >= 2 && index < 2 + cbr.getNumTrueDestOperands();
      if (isTrueOpr) {
        getAllPhiRelatedValues(cbr.getTrueDest()->getArgument(index - 2),
                               relatedVals);
      } else {
        getAllPhiRelatedValues(cbr.getFalseDest()->getArgument(
                                   index - cbr.getNumTrueDestOperands() - 2),
                               relatedVals);
      }
    }
  }
}

std::stack<Block *> getBlockPath(Block *srcBlk, Block *dstBlk) {
  std::stack<Block *> path;
  std::unordered_set<Block *> visited;
  std::unordered_map<Block *, Block *>
      parent; // To store the parent of each block

  std::stack<Block *> dfsStack;
  dfsStack.push(srcBlk);
  visited.insert(srcBlk);

  while (!dfsStack.empty()) {
    Block *current = dfsStack.top();
    dfsStack.pop();

    // If we reached the destination block, build the path
    if (current == dstBlk) {
      while (current != nullptr) {
        path.push(current);
        current = parent[current];
      }
      return path;
    }

    for (Block *successor : current->getSuccessors()) {
      if (visited.find(successor) == visited.end()) {
        visited.insert(successor);
        parent[successor] = current;
        dfsStack.push(successor);
      }
    }
  }

  // If no path found, return an empty stack
  return std::stack<Block *>();
}

unsigned getEarliestStartTime(Operation *op) {
  SmallVector<Value, 4> useOprs(op->operand_begin(), op->operand_end());
  unsigned hop = 0;
  // back track the use operands to get the longest path
  for (auto &use : useOprs) {
    unsigned curPath = 0;
    if (auto defOp = use.getDefiningOp()) {
      // if the defOp has been produced in other block, the hop increase 0.
      if (defOp->getBlock() != op->getBlock())
        continue;

      unsigned prodTime = getEarliestStartTime(defOp);
      curPath = prodTime == UINT64_MAX ? UINT64_MAX : prodTime + 1;
    } else if (isa<BlockArgument>(use)) {
      // if the operand is block argument, the distance is 0
      curPath = 0;
    } else {
      // invalid situation
      return UINT64_MAX;
    }
    // if the defOp is in the different block, get the longest path
    hop = std::max(hop, curPath);
  }
  return hop;
}

unsigned getLatestEndTime(Operation *op) {
  SmallVector<Operation *, 4> userOps(op->user_begin(), op->user_end());
  unsigned hop = 0;
  for (auto &user : userOps) {
    unsigned curPath = 0;
    if (user->getBlock() != op->getBlock() ||
        user == op->getBlock()->getTerminator())
      continue;

    unsigned userTime = getLatestEndTime(user);
    curPath = userTime == UINT64_MAX ? UINT64_MAX : userTime + 1;

    hop = std::max(hop, curPath);
  }
  return hop;
}

void getAllPathsToBlockEnd(Operation *op,
                           std::vector<SmallVector<Operation *, 8>> &allPaths) {
  if (!op) {
    return; // Handle null root operation
  }

  SmallVector<Operation *, 8> tempPath;

  // Helper function to perform DFS recursively
  std::function<void(Operation *, SmallVector<Operation *, 8> &)> dfs =
      [&](Operation *current, SmallVector<Operation *, 8> &path) {
        if (!current) {
          return;
        }

        // Add the current node to the path
        path.push_back(current);

        // If the current node is a leaf (no users), store the path
        if (current->getUsers().empty()) {
          allPaths.push_back(path);
        } else {
          // Otherwise, continue DFS on each child node
          for (Operation *user : current->getUsers()) {
            dfs(user, path);
          }
        }

        // Backtrack to explore other branches
        path.pop_back();
      };

  // Start DFS from the root operation
  dfs(op, tempPath);

  // Sort all paths by their length in descending order
  std::sort(
      allPaths.begin(), allPaths.end(),
      [](const SmallVector<Operation *, 8> &a,
         const SmallVector<Operation *, 8> &b) { return a.size() > b.size(); });
}

SmallVector<Operation *, 8> getCriticalPath(Block *blk) {
  SmallVector<Operation *, 8> criticalPath;

  // the startNodes should not use any value produced in current block
  SmallVector<Operation *> startNodes;
  for (auto &op : blk->getOperations()) {
    bool isStartNode = true;
    for (auto opr : op.getOperands()) {
      if (opr.getDefiningOp() && opr.getDefiningOp()->getBlock() == blk) {
        isStartNode = false;
      }
    }

    if (isStartNode) {
      std::vector<SmallVector<Operation *, 8>> allPaths;
      // start from the start node DFS to find the critical path
      getAllPathsToBlockEnd(&op, allPaths);

      if (allPaths[0].size() > criticalPath.size())
        criticalPath = allPaths[0];
    }
  }

  return criticalPath;
}

unsigned getValueLiveLength(Value val,
                            std::map<Block *, SetVector<Value>> &liveIns,
                            std::map<Block *, SetVector<Value>> &liveOuts) {
  SmallVector<Block *, 8> blocks;
  for (auto [block, _] : liveIns)
    blocks.push_back(block);
  // if val is phi related, ensure it produced the same attributes for all
  // related values.
  if (isa<BlockArgument>(val)) {
    unsigned totalLength = 0;
    for (auto block : blocks) {
      if (liveIns[block].count(val) && liveOuts[block].count(val))
        totalLength += getCriticalPath(block).size();
      if (liveIns[block].count(val) && !liveOuts[block].count(val)) {
        unsigned latestUse = 0;
        for (auto user : val.getUsers())
          if (user->getBlock() == block)
            latestUse = std::max(latestUse, getEarliestStartTime(user));
        totalLength += latestUse;
      }
    }

    return totalLength;
  }

  unsigned maxHop = 0;
  Operation *defOp = val.getDefiningOp();

  for (auto user : defOp->getUsers()) {
    // calculate the theoretical live path length
    maxHop = std::max(maxHop, getShortestLiveHops(defOp, user));
  }

  // For all the values in relatedVals, their theoretical live path cannot
  // exceed certain hops.
  return maxHop;
}

unsigned getShortestLiveHops(Operation *srcOp, Operation *dstOp) {
  unsigned hops = UINT64_MAX;
  if (srcOp->getBlock() == dstOp->getBlock()) {
    if (isBackEdge(srcOp, dstOp))
      // the distance is srcOp -> its exit ->
      return getLatestEndTime(dstOp) + getEarliestStartTime(srcOp);
    return 1;
  }

  auto route = getBlockPath(srcOp->getBlock(), dstOp->getBlock());
  hops = getEarliestStartTime(dstOp);
  route.pop(); // pop the dstOp block
  for (size_t i = 1; i < route.size() - 1; i++) {
    hops += getCriticalPath(route.top()).size();
    route.pop();
  }
  hops += getCriticalPath(srcOp->getBlock()).size();
  return hops;
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

void removeBlockArgs(Operation *term, std::vector<unsigned> argId, Block *dest,
                     OpBuilder &builder) {
  builder.setInsertionPoint(term);
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    SmallVector<Value> operands;
    for (auto [ind, opr] : llvm::enumerate(br->getOperands())) {
      if (std::find(argId.begin(), argId.end(), (unsigned)ind) == argId.end()) {
        operands.push_back(opr);
        llvm::errs() << "NOT FOUND " << ind;
      }
    }

    builder.create<cf::BranchOp>(term->getLoc(), operands, br.getSuccessor());
    term->erase();
    return;
  }

  if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
    SmallVector<Value> trueOperands;
    for (auto [ind, opr] : llvm::enumerate(condBr.getTrueDestOperands()))
      if (condBr.getTrueDest() != dest ||
          std::find(argId.begin(), argId.end(), ind) == argId.end())
        trueOperands.push_back(opr);
    SmallVector<Value> falseOperands;
    for (auto [ind, opr] : llvm::enumerate(condBr.getFalseDestOperands()))
      if (condBr.getFalseDest() != dest ||
          std::find(argId.begin(), argId.end(), ind) == argId.end())
        falseOperands.push_back(opr);
    builder.create<cf::CondBranchOp>(term->getLoc(), condBr.getOperand(0),
                                     condBr.getTrueDest(), trueOperands,
                                     condBr.getFalseDest(), falseOperands);
    term->erase();
    return;
  }
}

void removeUselessBlockArg(Region &region, OpBuilder &builder) {
  for (auto &block : region) {
    if (block.isEntryBlock())
      continue;
    if (block.getArguments().size() == 0 ||
        std::distance(block.getPredecessors().begin(),
                      block.getPredecessors().end()) > 1)
      continue;

    // the block has only one predecessor and have arguments
    // get the corresponding value in the predecessor
    auto prevTerm = (*block.getPredecessors().begin())->getTerminator();
    auto oprIndBase = 0;
    if (auto condBr = dyn_cast<cf::CondBranchOp>(prevTerm)) {
      // the block is the false dest
      oprIndBase = 1;
      if (condBr.getFalseDest() == &block)
        oprIndBase += condBr.getTrueDestOperands().size();
    }

    // find the corresponding value in the predecessor
    std::vector<unsigned> argId;
    unsigned oprInd = 0;
    for (auto arg : llvm::make_early_inc_range(block.getArguments())) {
      arg.replaceAllUsesWith(prevTerm->getOperand(oprIndBase + oprInd));
      argId.push_back(oprInd);
      oprInd++;
    }
    // remove all block arguments
    llvm::BitVector bitVec(argId.size(), true);
    block.eraseArguments(bitVec);
    removeBlockArgs(prevTerm, argId, &block, builder);
  }
}
} // namespace compigra
