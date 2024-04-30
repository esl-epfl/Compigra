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

// validate whether the control flow graph can be scheduled by SAT-MapIt
LogicalResult validateSatMapCtrlGraph(Block *block) { return failure(); }

LogicalResult mapToFullPredict(std::vector<bbInfo> &bbs, OpBuilder &builder) {
  // determine whether the block
  int outLoopBB = -1;
  Location loc = builder.getUnknownLoc();
  SmallVector<Operation *, 8> replaceOps;

  for (auto &bb : bbs) {
    if (bb.stage == BlockStage::Init || bb.stage == BlockStage::Fini)
      continue;

    auto beginOp = bb.block->getOperations().begin();
    auto endOp = bb.block->getOperations().rbegin();

    llvm::errs() << "\nbeginOp: " << *beginOp << "\n";

    if (isa<cf::CondBranchOp>(*endOp))
      for (auto suc : bb.block->getSuccessors()) {
        if (bbs[getBlockIndex(bbs, suc)].stage == BlockStage::Fini) {
          // CHECK
          outLoopBB = getBlockIndex(bbs, bb.block);
        }
      }

    if (outLoopBB == -1)
      continue;

    // move the operations to the end of the block
    auto cntBlock = bbs[outLoopBB + 1].block;

    if (bb.index == getBlockIndex(bbs, cntBlock) || bb.index == outLoopBB)
      continue;

    loc = cntBlock->getTerminator()->getLoc();

    // if block has arguments, insert mergeOp to merge the arguments
    for (size_t i = 0; i < bb.block->getNumArguments(); i++) {
      std::vector<Value> operands;
      // get the operands from its successors
      for (auto pred : bb.block->getPredecessors()) {
        auto defOp = pred->getTerminator();
        operands.push_back(defOp->getOperand(i));
      }
      builder.setInsertionPoint(cntBlock->getTerminator());
      // auto mergeOp = builder.create<cgra::MergeOp>(loc, operands);
      // llvm::errs() << "new created MergeOp: " << mergeOp << "\n";
      // loc = mergeOp.getLoc(); // update insert point
    }

    // move all the operations before the loc
    for (auto &op : bb.block->getOperations()) {
      if (isa<cf::BranchOp, cf::CondBranchOp>(op))
        continue;
      replaceOps.push_back(&op);
    }
    // if (getBlockIndex(bbs, bb.block) == 5)
    //   break;
  }

  // delte unnecessary branchOp and condBranchOp
  for (auto op : replaceOps) {
    auto newOp = builder.clone(*op);
    newOp->setLoc(loc);
    op->replaceAllUsesWith(newOp);
    loc = newOp->getLoc();
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
    // auto funcOp
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

  if (failed(mapToFullPredict(bbs, builder)))
    return signalPassFailure();

  llvm::errs() << "After mapToFullPredict\n";

  getOperation()->walk([&](Operation *op) { llvm::errs() << *op << "\n"; });
}

} // namespace

namespace mlir {
namespace compigra {
std::unique_ptr<mlir::Pass> createCfMapToFullPredict() {
  return std::make_unique<CfMapToFullPredictPass>();
}
} // namespace compigra
} // namespace mlir