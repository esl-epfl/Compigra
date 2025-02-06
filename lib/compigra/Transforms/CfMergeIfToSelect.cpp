//===-  CfMergeIfToSelect.cpp - Fix index to CGRA PE bitwidth --*- C++ --*-===//
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

#include "compigra/Transforms/CfMergeIfToSelect.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <stack>
#include <unordered_set>

// for printing debug informations
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace compigra;

static unsigned getPredBlkSize(Block *blk) {
  return std::distance(blk->pred_begin(), blk->pred_end());
}

static unsigned getSuccBlkSize(Block *blk) {
  return std::distance(blk->succ_begin(), blk->succ_end());
}

static bool isWriteFreeBlk(Block *blk) {
  for (auto &op : blk->without_terminator()) {
    if (isa<memref::StoreOp>(op))
      return false;
  }
  return true;
}

bool isSingleConnected(Block *srcBlk, Block *dstBlk, OpBuilder &builder) {
  // srcBlk only have one successor dstBlk;
  // dstBlk only have one predecessor srcBlk
  return getSuccBlkSize(srcBlk) == 1 && getPredBlkSize(dstBlk) == 1 &&
         srcBlk->getSuccessors().front() == dstBlk;
}

LogicalResult compigra::mergeSinglePathBBs(Block *srcBB, Block *dstBB,
                                           OpBuilder &builder) {
  // move all operations in dstBB to srcBB, erase the terminator of srcBB
  auto termOp = srcBB->getTerminator();
  for (auto opr : llvm::make_early_inc_range(dstBB->getArguments()))
    opr.replaceAllUsesWith(termOp->getOperand(opr.getArgNumber()));

  for (auto &op : llvm::make_early_inc_range(dstBB->getOperations()))
    op.moveBefore(srcBB->getTerminator());
  termOp->erase();
  dstBB->erase();

  return success();
}

LogicalResult compigra::mergeTwoExecutePath(Block *mergeBB, Block *src1,
                                            Block *src2, OpBuilder &builder) {
  // first determine src1 and src2 are fusible
  // path 1: src1 -> dstBB and path 2: src2 -> dstBB should be write-free to
  // fuse them.
  if (!isWriteFreeBlk(src1) || !isWriteFreeBlk(src2))
    return failure();

  Block *dstBlk;
  // check they share the same destination block
  if (mergeBB == src2)
    std::swap(src1, src2);

  if (getSuccBlkSize(src2) != 1)
    return failure();
  dstBlk = src2->getSuccessors().front();

  if (mergeBB == src1) {
    // mergeBB must also point the dstBlk
    bool findDstBlk = std::find(mergeBB->succ_begin(), mergeBB->succ_end(),
                                dstBlk) != mergeBB->succ_end();
    bool findSrc2 = std::find(mergeBB->succ_begin(), mergeBB->succ_end(),
                              src2) != mergeBB->succ_end();
    if (!findDstBlk || !findSrc2)
      return failure();

  } else {
    bool findSrc1 = std::find(mergeBB->succ_begin(), mergeBB->succ_end(),
                              src1) != mergeBB->succ_end();
    bool findSrc2 = std::find(mergeBB->succ_begin(), mergeBB->succ_end(),
                              src2) != mergeBB->succ_end();
    bool src1IsDstBlk = (getSuccBlkSize(src2) == 1) &&
                        (src2->getSuccessors().front() == dstBlk);
    if (!findSrc1 || !findSrc2 || !src1IsDstBlk)
      return failure();

    // move operations in src1 to mergeBB, change the topology from (B) to (A).
    for (auto &op : llvm::make_early_inc_range(src1->without_terminator()))
      op.moveBefore(mergeBB->getTerminator());
    // change the mergeBB->getTerminator();
    cf::CondBranchOp origOp =
        dyn_cast<cf::CondBranchOp>(mergeBB->getTerminator());
    cf::BranchOp src1TermOp = dyn_cast<cf::BranchOp>(src1->getTerminator());
    auto jumpOprs = src1TermOp.getOperands();

    bool mergeTrueBranch = origOp.getTrueDest() == src1;
    auto trueDest = mergeTrueBranch ? dstBlk : origOp.getTrueDest();
    auto trueOprs = mergeTrueBranch ? jumpOprs : origOp.getTrueOperands();

    auto falseDest = mergeTrueBranch ? origOp.getFalseDest() : dstBlk;
    auto falseOprs = mergeTrueBranch ? origOp.getFalseOperands() : jumpOprs;

    builder.setInsertionPointToEnd(mergeBB);
    builder.create<cf::CondBranchOp>(origOp.getLoc(), origOp.getCondition(),
                                     trueDest, trueOprs, falseDest, falseOprs);
    origOp.erase();
    src1TermOp.erase();
    src1->erase();
  }

  // merge src2 to mergeBB, then the mergebb can directly jump to succBlk
  cf::CondBranchOp mergeTermOp =
      dyn_cast<cf::CondBranchOp>(mergeBB->getTerminator());
  cf::BranchOp src2TermOp = dyn_cast<cf::BranchOp>(src2->getTerminator());
  auto jumpOprs = src2TermOp.getOperands();
  builder.setInsertionPoint(mergeBB->getTerminator());
  for (auto &op : llvm::make_early_inc_range(src2->without_terminator())) {
    op.moveBefore(mergeTermOp);
  }
  bool src2IsFalseBranch = mergeTermOp.getFalseDest() == src2;
  // insert selectOps to select what to propagate to the dstBlk
  SmallVector<Value, 4> selectOprs;
  for (auto [ind, opr] : llvm::enumerate(jumpOprs)) {
    auto selection1 = src2IsFalseBranch ? mergeTermOp.getTrueOperand(ind) : opr;
    auto selection2 =
        src2IsFalseBranch ? opr : mergeTermOp.getFalseOperand(ind);
    auto selOpr = builder.create<arith::SelectOp>(mergeTermOp->getLoc(),
                                                  mergeTermOp.getCondition(),
                                                  selection1, selection2);
    selectOprs.push_back(selOpr.getResult());
  }
  // create a new branch to the dstBlk
  builder.create<cf::BranchOp>(mergeTermOp->getLoc(), dstBlk, selectOprs);
  mergeTermOp.erase();
  src2TermOp.erase();
  src2->erase();
  return success();
}

std::optional<Block *> compigra::getCommonAncestor(Block *src1, Block *src2) {
  if (!src1 || !src2)
    return std::nullopt;

  std::unordered_set<Block *> visited;

  // first check whether src1 is the src2's ancestor
  Block *ancestor = src1;
  if (std::find(src2->pred_begin(), src2->pred_end(), src1) !=
      src2->pred_end()) {
    // src1 should be the only predecessor of src2
    if (getPredBlkSize(src2) == 1 && getSuccBlkSize(src2) == 1)
      return src1;
    return std::nullopt;
  }

  // then check whether src2 is the src1's ancestor
  ancestor = src2;
  if (std::find(src1->pred_begin(), src1->pred_end(), src2) !=
      src1->pred_end()) {
    // src2 should be the only predecessor of src1
    if (getPredBlkSize(src1) == 1 && getSuccBlkSize(src1) == 1)
      return src2;
    return std::nullopt;
  }

  // then check whether src1 and src2 have a common ancestor
  if (getPredBlkSize(src1) == 1 && getSuccBlkSize(src1) == 1 &&
      getPredBlkSize(src2) == 1 && getSuccBlkSize(src2) == 1)
    // src1 and src2 should have the same predecessor
    if (*src1->pred_begin() == *src2->pred_begin())
      return *src1->pred_begin();

  return std::nullopt;
}

bool existMergeHorizontal(func::FuncOp funcOp, OpBuilder &builder) {
  // iterate basic blocks in the function
  for (auto &curBlk : funcOp.getBlocks()) {

    // first get successor blocks
    for (auto sucBlk : curBlk.getSuccessors()) {
      for (auto upperBlk : sucBlk->getPredecessors()) {
        if (upperBlk == &curBlk)
          continue;

        if (auto ancestor = getCommonAncestor(&curBlk, upperBlk);
            ancestor.has_value()) {
          if (succeeded(mergeTwoExecutePath(ancestor.value(), &curBlk, upperBlk,
                                            builder)))
            return true;
        }
      }
    }
  }
  return false;
}

bool existMergeVertical(func::FuncOp funcOp, OpBuilder &builder) {
  // iterate basic blocks in the function
  for (auto &curBlk : funcOp.getBlocks()) {

    // first get successor blocks
    for (auto sucBlk : curBlk.getSuccessors()) {
      if (isSingleConnected(&curBlk, sucBlk, builder)) {
        llvm::errs() << "Vertical merge " << curBlk.getOperations().front()
                     << " and " << sucBlk->getOperations().front() << "\n";
        if (succeeded(mergeSinglePathBBs(&curBlk, sucBlk, builder)))
          return true;
      }
    }
  }
  return false;
}

namespace {
/// Driver for the cf DAG rewrite pass.
struct CfMergeIfToSelectPass
    : public compigra::impl::CfMergeIfToSelectBase<CfMergeIfToSelectPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // std::vector<bbInfo> bbs;

    func::FuncOp funcOp;
    getOperation()->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        funcOp = cast<func::FuncOp>(op);
    });

    unsigned maxTry = 10;
    while (existMergeHorizontal(funcOp, builder) ||
           existMergeVertical(funcOp, builder)) {
      if (maxTry-- == 0)
        break;
    }
  }
};

} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createCfMergeIfToSelect() {
  return std::make_unique<CfMergeIfToSelectPass>();
}
} // namespace compigra
