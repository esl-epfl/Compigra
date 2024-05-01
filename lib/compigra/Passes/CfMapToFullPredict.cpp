//===-  CfMapToFullPredict.cpp - Fix index to CGRA PE bitwidth --*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --map-to-full-predict pass, which reduce the number of bbs
// by applying full predict on the DAG.
//
//===----------------------------------------------------------------------===//

#include "compigra/Passes/CfMapToFullPredict.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ilist.h"
#include <stack>

// for printing debug informations
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace compigra;

int getBlockIndex(std::vector<bbInfo> &bbs, Block *block) {
  for (auto bb : bbs) {
    if (bb.block == block)
      return bb.index;
  }
  return -1;
}

// Low-efficiency implementation of detecting loop in the control flow graph
BlockStage getBlockStage(Block *block, std::vector<bbInfo> &bbs) {
  if (block->isEntryBlock())
    return BlockStage::Init;

  std::vector<int> visited(bbs.size(), 0);

  // dfs to track the stage of the block
  std::stack<Block *> stack;
  stack.push(block);

  while (!stack.empty()) {
    Block *curBlock = stack.top();
    stack.pop();

    if (visited[getBlockIndex(bbs, curBlock)] == 1)
      return BlockStage::Loop;

    visited[getBlockIndex(bbs, curBlock)] = 1;

    for (auto successor : curBlock->getSuccessors()) {
      stack.push(successor);
    }
  }

  return BlockStage::Fini;
}

// Need to be optimized
bool isBackEdge(Block *block, std::vector<bbInfo> &bbs) {
  for (auto succ : block->getSuccessors()) {
    if (getBlockIndex(bbs, succ) < getBlockIndex(bbs, block))
      return true;
  }
  return false;
}

Operation *getDefOperation(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (defOp == nullptr)
    return nullptr;
  if (isa<cf::BranchOp, cf::CondBranchOp>(defOp))
    return nullptr;
  return defOp;
}

// validate whether the control flow graph can be scheduled by SAT-MapIt
LogicalResult validateSatMapCtrlGraph(std::vector<bbInfo> &bbs,
                                      int &outLoopBB) {
  int initBB = -1;
  int finiBB = -1;

  // check whether there is only one init block and one fini block
  for (auto &bb : bbs) {
    if (bb.stage == BlockStage::Init) {
      if (initBB != -1)
        return failure();
      initBB = bb.index;
    }
    if (bb.stage == BlockStage::Fini) {
      if (finiBB != -1)
        return failure();
      finiBB = bb.index;
    }
  }
  // verify init=0 and fini=bbs.size() block
  if (initBB != 0 || finiBB != static_cast<int>(bbs.size() - 1))
    return failure();

  // check whether init block has only one successor
  if (bbs[initBB].block->getSuccessors().size() != 1)
    return failure();

  // check whether fini block has only one predecessor
  outLoopBB = -1;
  for (int i = 1; i < static_cast<int>(bbs.size()) - 1; i++) {
    auto &bb = bbs[i];
    if (bb.stage != BlockStage::Loop)
      return failure();
    for (auto suc : bb.block->getSuccessors())
      if (bbs[getBlockIndex(bbs, suc)].stage == BlockStage::Fini) {
        if (outLoopBB != -1)
          return failure();
        outLoopBB = getBlockIndex(bbs, bb.block);
      }
  }

  return success();
}

LogicalResult mapToFullPredict(std::vector<bbInfo> &bbs, OpBuilder &builder,
                               int outLoopBB) {
  // determine whether the block
  Location loc = builder.getUnknownLoc();
  SmallVector<Operation *, 8> replaceOps;
  SmallVector<Operation *, 8> removeBranchOps;

  for (int i = 1; i < static_cast<int>(bbs.size()) - 1; i++) {
    auto &bb = bbs[i];

    if (getBlockIndex(bbs, bb.block) <= outLoopBB)
      loc = bbs[1].block->getTerminator()->getLoc();

    // move the operations to the end of the block
    Block *cntBlock;
    if (i <= outLoopBB)
      cntBlock = bbs[1].block;
    else
      cntBlock = bbs[outLoopBB + 1].block;
    // auto cntBlock = bbs[outLoopBB + 1].block;

    loc = cntBlock->getTerminator()->getLoc();

    // allow the loop block to have arguments
    if (getBlockIndex(bbs, bb.block) != 1 &&
        getBlockIndex(bbs, bb.block) != outLoopBB + 1)
      for (size_t i = 0; i < bb.block->getNumArguments(); i++) {
        std::vector<Value> operands;
        // get the operands from its successors
        for (auto pred : bb.block->getPredecessors()) {
          auto defOp = pred->getTerminator();
          operands.push_back(defOp->getOperand(i));
        }
        if (operands.size() == 1)
          continue;
        builder.setInsertionPoint(cntBlock->getTerminator());
        auto mergeOp = builder.create<cgra::MergeOp>(loc, operands);
        loc = mergeOp.getLoc();
        // replace the argument with the result of mergeOp
        bb.block->getArgument(i).replaceAllUsesWith(mergeOp.getResult());
      }

    for (auto &op : bb.block->getOperations()) {
      // remove unnecessary branchOp and condBranchOp
      if (isa<cf::BranchOp, cf::CondBranchOp>(op) &&
          !isBackEdge(bb.block, bbs) &&
          getBlockIndex(bbs, bb.block) != outLoopBB)
        removeBranchOps.push_back(&op);
      else {
        if (getBlockIndex(bbs, bb.block) == 1 ||
            getBlockIndex(bbs, bb.block) == outLoopBB + 1)
          continue;
        // move the operations to first successor of the outLoopBB
        builder.setInsertionPoint(cntBlock->getTerminator());
        auto newOp = builder.clone(op);
        newOp->setLoc(loc);
        op.replaceAllUsesWith(newOp);
        loc = newOp->getLoc();
      }
    }
  }

  // delte unnecessary branchOp and condBranchOp
  for (auto op : removeBranchOps)
    op->erase();

  // remove the unnecessary blocks
  for (int i = 1; i < static_cast<int>(bbs.size()) - 1; i++) {
    auto &bb = bbs[i];
    if (getBlockIndex(bbs, bb.block) != 1 &&
        getBlockIndex(bbs, bb.block) != outLoopBB + 1)
      bb.block->erase();
  }

  return success();
}

namespace {
/// Driver for the index bitwidth fix pass.
struct CfMapToFullPredictPass
    : public compigra::impl::CfMapToFullPredictBase<CfMapToFullPredictPass> {
  void runOnOperation() override;
};

void CfMapToFullPredictPass::runOnOperation() {
  auto *ctx = &getContext();
  OpBuilder builder(ctx);
  // PatternRewriter rewriter(ctx);

  std::vector<Operation *> ops;
  std::vector<bbInfo> bbs;

  func::FuncOp funcOp;
  getOperation()->walk([&](Operation *op) {
    if (isa<func::FuncOp>(op))
      funcOp = cast<func::FuncOp>(op);
  });

  funcOp->walk([&](Operation *op) {
    if (!isa<func::FuncOp>(op)) {
      auto curBlock = op->getBlock();
      if (getBlockIndex(bbs, curBlock) == -1) {
        bbs.push_back(
            bbInfo{static_cast<int>(bbs.size()), BlockStage::Init, curBlock});
      }
    }
  });

  // track the stage of the block
  for (auto &bb : bbs)
    bb.stage = getBlockStage(bb.block, bbs);

  int outLoopBB = -1;
  if (failed(validateSatMapCtrlGraph(bbs, outLoopBB)))
    return signalPassFailure();

  if (failed(mapToFullPredict(bbs, builder, outLoopBB)))
    return signalPassFailure();
}

} // namespace

namespace mlir {
namespace compigra {
std::unique_ptr<mlir::Pass> createCfMapToFullPredict() {
  return std::make_unique<CfMapToFullPredictPass>();
}
} // namespace compigra
} // namespace mlir