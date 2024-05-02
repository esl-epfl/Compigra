//===- CfReduceBranches.cpp - Reduce branches for bb merge   ----*- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --reduce-branches pass, which merge basic blocks with for
// control flow simplification through branch reduction.
//
//===----------------------------------------------------------------------===//

#include "compigra/Passes/CfReduceBranches.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ilist.h"

// Debugging support
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "BRANCH_REDUCTION"

using namespace mlir;
using namespace compigra;

LogicalResult irrelavantBlock(Block *block1, Block *block2) {
  return success();
}

namespace {
// Remove the conditional branch operation if it branches to the same block with
// different arguments, merge the successor block with the current block and use
// select operation to merge the two branches
struct CondBranchOpRewrite : public RewritePattern {
  CondBranchOpRewrite(MLIRContext *context)
      : RewritePattern(cf::CondBranchOp::getOperationName(), 1, context){};

  LogicalResult match(Operation *op) const override {
    if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op))
      if (condBranchOp.getTrueDest() == condBranchOp.getFalseDest())
        return success();
    return failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    // use select operation to merge the two branches
    auto condBranchOp = cast<cf::CondBranchOp>(op);
    Value flag = op->getOperand(0);

    Block *sucBlock = op->getBlock()->getSuccessors()[0];
    for (size_t i = 0; i < condBranchOp.getTrueDestOperands().size(); i++) {
      rewriter.setInsertionPoint(op);
      auto selecOp = rewriter.create<arith::SelectOp>(
          op->getLoc(), flag, condBranchOp.getTrueOperand(i),
          condBranchOp.getFalseOperand(i));
      rewriter.replaceAllUsesWith(sucBlock->getArgument(i), selecOp);
    }

    sucBlock->walk([&](Operation *sucOp) {
      sucOp->moveBefore(op->getBlock()->getTerminator());
    });
    rewriter.eraseOp(op);
    rewriter.eraseBlock(sucBlock);
  }
};

// Merge a basic block with its predecessor if the predecessor unconditionally
// branches to it, or the merge not affect the other basic block the predecessor
// block branches to.
struct BranchOpRewrite : public RewritePattern {
  BranchOpRewrite(MLIRContext *context)
      : RewritePattern(cf::BranchOp::getOperationName(), 1, context) {}

  LogicalResult match(Operation *op) const override {
    if (!isa<cf::BranchOp>(op))
      return failure();
    else {
      // branchOp should have only one successor
      if (op->getBlock()->getSuccessors().size() != 1) {
        return failure();
      }

      Block *curBlock = op->getBlock();
      if (curBlock->isEntryBlock() || curBlock->getArguments().size() > 0) {
        return failure();
      }

      Block *prevBlock = *curBlock->getPredecessors().begin();
      if (isa<cf::CondBranchOp>(prevBlock->getTerminator()) &&
          failed(irrelavantBlock(prevBlock, curBlock)))
        return failure();
    }
    return success();
  }

  // merge current block with the successor block
  void rewrite(Operation *branchOp, PatternRewriter &rewriter) const override {

    // Get the predecessor block of the branch operation
    Block *curBlock = branchOp->getBlock();
    Block *prevBlock = *curBlock->getPredecessors().begin();
    Operation *prevTerminator = prevBlock->getTerminator();
    std::vector<Operation *> opsToMove;

    for (auto &op : curBlock->getOperations())
      opsToMove.push_back(&op);

    auto loc = prevTerminator->getLoc();

    if (isa<cf::BranchOp>(prevTerminator)) {
      for (auto op : opsToMove) {
        if (op == branchOp) {
          // rewrite its successor's branch and cond_branch
          if (isa<cf::BranchOp>(prevTerminator)) {
            SmallVector<Value, 4> operands;
            // add the arguments of current block to its successor's branch and
            // cond_branch
            for (auto arg : op->getOperands())
              operands.push_back(arg);
            // update the succcessor of block
            auto sucBranchOp = cast<cf::BranchOp>(prevTerminator);
            sucBranchOp.setDest(curBlock->getSuccessors()[0]);
            //   update the branch arguments
            sucBranchOp->insertOperands(sucBranchOp->getNumOperands(),
                                        operands);
            prevTerminator->setLoc(loc);
            continue;
          }
        }
        op->moveBefore(prevTerminator);
        loc = op->getLoc();
      }
    } else if (isa<cf::CondBranchOp>(prevTerminator)) {
      for (auto op : opsToMove) {
        if (op == branchOp) {
          SmallVector<Value, 4> operands;
          for (auto arg : op->getOperands())
            operands.push_back(arg);
          // the other branch should not be affected
          rewriter.setInsertionPoint(prevTerminator);

          cf::CondBranchOp newOp;
          auto prevCondBranchOp = cast<cf::CondBranchOp>(prevTerminator);
          //   update the partial conditional branch operation in the
          //   successor
          if (curBlock == prevCondBranchOp.getTrueDest()) {
            newOp = rewriter.create<cf::CondBranchOp>(
                loc, prevTerminator->getOperand(0),
                curBlock->getSuccessors()[0], operands,
                prevBlock->getSuccessors()[1],
                prevCondBranchOp.getFalseOperands());
          } else {
            newOp = rewriter.create<cf::CondBranchOp>(
                loc, prevTerminator->getOperand(0),
                prevBlock->getSuccessors()[0],
                prevCondBranchOp.getTrueDestOperands(),
                curBlock->getSuccessors()[0], operands);
          }
          rewriter.replaceOp(prevTerminator, newOp);
          continue;
        }
        op->moveBefore(prevTerminator);
        loc = op->getLoc();
      }
    }

    rewriter.eraseOp(branchOp);
    rewriter.eraseBlock(curBlock);
  }
};

struct CfReduceBranchesPass
    : public compigra::impl::CfReduceBranchesBase<CfReduceBranchesPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns{ctx};
    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;

    patterns.add<BranchOpRewrite>(ctx);
    patterns.add<CondBranchOpRewrite>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      LLVM_DEBUG(llvm::dbgs() << "BranchOp or CondBranchOp failed to meet "
                                 "transformation requirement\n");
      return signalPassFailure();
    }
  }
};

} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createCfReduceBranches() {
  return std::make_unique<CfReduceBranchesPass>();
}
} // namespace compigra
