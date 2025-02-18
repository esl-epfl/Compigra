//===-  CfFuseLoopHeadBody.cpp - Fuse the loop head and body --*- C++ --*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --fuse-loop pass, which fuses the head and body of a loop into
// one basic block.
//
//===----------------------------------------------------------------------===//

#include "compigra/Transforms/CfFuseLoopHeadBody.h"
#include "compigra/Support/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace compigra;

static bool hasNoSideEffect(Block *loopHead) {
  // check whether operations in the head block are only for control purposes
  auto termOp = loopHead->getTerminator();
  // check whether other operations are only used for control purposes
  for (auto &op : loopHead->without_terminator()) {
    if (isa<memref::StoreOp>(op))
      return false;
    for (auto user : op.getUsers()) {
      if (user != termOp)
        return false;
    }
  }
  return true;
}

std::optional<Block *> getFusibleBodyBlock(func::FuncOp funcOp) {
  for (auto &blk : funcOp.getBlocks()) {
    // Ensure the block has exactly one predecessor and one successor
    if (!llvm::hasSingleElement(blk.getPredecessors()) ||
        !llvm::hasSingleElement(blk.getSuccessors())) {
      continue;
    }

    // Verify predecessor and successor are the same block (head)
    Block *headBlk = *blk.pred_begin();
    if (headBlk != *blk.succ_begin())
      continue;

    // Check for control-only operations
    if (hasNoSideEffect(headBlk))
      return &blk;
  }
  return std::nullopt;
}

// if the result of the condition is determined, revise the terminator to
// jump to the corresponding block
bool removeCertainCondBr(func::FuncOp funcOp, OpBuilder &builder) {
  for (auto &blk : funcOp.getBlocks()) {
    auto termOp = blk.getTerminator();
    auto condBrOp = dyn_cast_or_null<cf::CondBranchOp>(termOp);
    if (!condBrOp)
      continue;

    auto cmpOp = dyn_cast_or_null<arith::CmpIOp>(
        condBrOp.getCondition().getDefiningOp());
    if (!cmpOp)
      continue;

    auto predicate = cmpOp.getPredicate();
    auto opr1 = cmpOp.getOperand(0);
    auto opr2 = cmpOp.getOperand(1);
    // Check if both operands are constant
    auto constOpr1 = dyn_cast_or_null<arith::ConstantOp>(opr1.getDefiningOp());
    auto constOpr2 = dyn_cast_or_null<arith::ConstantOp>(opr2.getDefiningOp());
    if (!constOpr1 || !constOpr2)
      continue;

    // Get the constant values
    auto val1 = constOpr1.getValue().cast<IntegerAttr>().getInt();
    auto val2 = constOpr2.getValue().cast<IntegerAttr>().getInt();

    // Determine the result of the comparison
    bool result;
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      result = (val1 == val2);
      break;
    case arith::CmpIPredicate::ne:
      result = (val1 != val2);
      break;
    case arith::CmpIPredicate::slt:
      result = (val1 < val2);
      break;
    case arith::CmpIPredicate::sle:
      result = (val1 <= val2);
      break;
    case arith::CmpIPredicate::sgt:
      result = (val1 > val2);
      break;
    case arith::CmpIPredicate::sge:
      result = (val1 >= val2);
      break;
    default:
      continue;
    }

    // Replace the conditional branch with an unconditional branch
    auto Operands =
        result ? condBrOp.getTrueOperands() : condBrOp.getFalseOperands();
    builder.setInsertionPoint(termOp);
    builder.create<cf::BranchOp>(
        termOp->getLoc(),
        result ? condBrOp.getTrueDest() : condBrOp.getFalseDest(), Operands);
    termOp->erase();
    return true;
  }
  // does not exist certain conditional branch
  return false;
}

Operation *mergeHeadAndBody(Block *headBlk, Block *bodyBlk,
                            SmallVector<Operation *, 4> &copyOps,
                            OpBuilder &builder) {
  builder.setInsertionPointToStart(bodyBlk);
  auto loc = bodyBlk->getOperations().front().getLoc();

  SmallVector<Operation *, 4> opsToMove;
  for (auto &op : headBlk->without_terminator())
    opsToMove.push_back(&op);

  // copy the block arguments from the head block to the body block
  SmallVector<Value, 4> bodyArgs;
  for (auto arg : headBlk->getArguments()) {
    auto newArg = bodyBlk->addArgument(arg.getType(), loc);
    bodyArgs.push_back(newArg);
  }

  // copy the operations from the head block to the body block
  auto origHeadTerm = dyn_cast<cf::CondBranchOp>(headBlk->getTerminator());
  builder.setInsertionPoint(bodyBlk->getTerminator());
  auto phiSource = bodyBlk->getTerminator()->getOperands();
  Operation *bodyCond;
  for (auto op : opsToMove) {
    // create a operation in the body block with the corresponding operands
    auto newOp = builder.clone(*op);
    if (op == origHeadTerm.getCondition().getDefiningOp())
      bodyCond = newOp;

    for (auto &opr : op->getOpOperands()) {
      auto val = opr.get();
      auto defOp = val.getDefiningOp();
      // replace the block argument with the corresponding value
      if (auto blockArg = dyn_cast_or_null<BlockArgument>(val);
          blockArg && blockArg.getOwner() == headBlk) {
        newOp->setOperand(opr.getOperandNumber(),
                          phiSource[blockArg.getArgNumber()]);
      }
    }
    copyOps.push_back(newOp);
  }

  // adapted the original operations in the body block
  for (auto &op : bodyBlk->getOperations()) {
    for (auto &opr : op.getOpOperands()) {
      auto val = opr.get();
      auto defOp = val.getDefiningOp();
      // replace the block argument with the corresponding value
      if (auto blockArg = dyn_cast_or_null<BlockArgument>(val);
          blockArg && blockArg.getOwner() == headBlk)
        op.setOperand(opr.getOperandNumber(),
                      bodyArgs[blockArg.getArgNumber()]);
    }
  }
  return bodyCond;
}

cf::CondBranchOp setUpNewHeadTerminator(Block *headBlk, Block *bodyBlk,
                                        OpBuilder &builder, bool ifTrueToBody) {
  // change the propagated operands in head block to the new block arguments
  Block *initBlk;
  Block *tailBlk;
  for (auto pred : headBlk->getPredecessors())
    if (pred != bodyBlk) {
      initBlk = pred;
      break;
    }
  for (auto succ : headBlk->getSuccessors())
    if (succ != bodyBlk) {
      tailBlk = succ;
      break;
    }

  auto origHeadTerm = dyn_cast<cf::CondBranchOp>(headBlk->getTerminator());
  SmallVector<Value, 4> initOprs;
  if (auto branchOp = dyn_cast<cf::BranchOp>(initBlk->getTerminator())) {
    initOprs.append(branchOp.getOperands().begin(),
                    branchOp.getOperands().end());
  } else if (auto condBranchOp =
                 dyn_cast<cf::CondBranchOp>(initBlk->getTerminator())) {
    if (condBranchOp.getTrueDest() == headBlk) {
      initOprs.append(condBranchOp.getTrueOperands().begin(),
                      condBranchOp.getTrueOperands().end());
    } else {
      initOprs.append(condBranchOp.getFalseOperands().begin(),
                      condBranchOp.getFalseOperands().end());
    }
  }
  builder.setInsertionPoint(headBlk->getTerminator());
  SmallVector<Value, 4> trueOprs(initOprs.begin(), initOprs.end());
  SmallVector<Value, 4> falseOprs(origHeadTerm.getFalseOperands().begin(),
                                  origHeadTerm.getFalseOperands().end());

  // operands in the tail block should be replaced with the new block arguments
  for (auto &op : tailBlk->getOperations())
    for (auto &opr : op.getOpOperands()) {
      auto val = opr.get();
      auto defOp = val.getDefiningOp();
      auto blockArg = dyn_cast_or_null<BlockArgument>(val);
      // replace the block argument with the corresponding value
      if ((blockArg && blockArg.getOwner() == headBlk) ||
          (defOp && defOp->getBlock() == headBlk)) {
        // add block argument to the tail block
        auto loc = tailBlk->getOperations().front().getLoc();
        tailBlk->addArgument(blockArg.getType(), loc);
        // concatenate val to the falseOprs
        falseOprs.push_back(val);
        // set the operand in tailBlk to the new block argument
        op.setOperand(opr.getOperandNumber(), tailBlk->getArguments().back());
      }
    }

  if (!ifTrueToBody)
    std::swap(trueOprs, falseOprs);

  auto newHeadTerm = builder.create<cf::CondBranchOp>(
      origHeadTerm->getLoc(), origHeadTerm.getCondition(),
      origHeadTerm.getTrueDest(), trueOprs, origHeadTerm.getFalseDest(),
      falseOprs);

  return newHeadTerm;
}

LogicalResult fuseHeadToBody(Block *headBlk, Block *bodyBlk,
                             OpBuilder &builder) {
  auto origHeadTerm = dyn_cast<cf::CondBranchOp>(headBlk->getTerminator());
  auto phiSource = bodyBlk->getTerminator()->getOperands();
  bool ifTrueToBody = origHeadTerm.getTrueDest() == bodyBlk;

  SmallVector<Operation *, 4> copyOps;
  auto bodyCond = mergeHeadAndBody(headBlk, bodyBlk, copyOps, builder);

  auto newHeadTerm =
      setUpNewHeadTerminator(headBlk, bodyBlk, builder, ifTrueToBody);

  // change the terminator of the body block to be conditional branch
  builder.setInsertionPoint(bodyBlk->getTerminator());
  auto origBodyTerm = dyn_cast<cf::BranchOp>(bodyBlk->getTerminator());

  SmallVector<Value, 4> trueBodyOprs(origBodyTerm.getOperands().begin(),
                                     origBodyTerm.getOperands().end());
  SmallVector<Value, 4> falseBodyOprs(newHeadTerm.getFalseOperands().begin(),
                                      newHeadTerm.getFalseOperands().end());
  // if true to continue the loop(body)
  for (auto &opr : falseBodyOprs) {
    auto blockArg = dyn_cast_or_null<BlockArgument>(opr);
    auto defOp = opr.getDefiningOp();
    if ((blockArg && blockArg.getOwner() == headBlk)) {
      opr = phiSource[blockArg.getArgNumber()];
    } else if (defOp && defOp->getBlock() == headBlk) {
      // replace the use of defOp with the corresponding value in copyOps
      auto &ops = headBlk->getOperations();
      auto opIt =
          llvm::find_if(ops, [&](mlir::Operation &op) { return &op == defOp; });

      if (opIt == ops.end())
        return failure();

      int defOpInd = static_cast<int>(std::distance(ops.begin(), opIt));

      opr = copyOps[defOpInd]->getResult(0);
    }
  }
  if (!ifTrueToBody)
    std::swap(trueBodyOprs, falseBodyOprs);

  auto newBodyTerm = builder.create<cf::CondBranchOp>(
      origHeadTerm->getLoc(), bodyCond->getResult(0),
      origHeadTerm.getTrueDest(), trueBodyOprs, origHeadTerm.getFalseDest(),
      falseBodyOprs);

  origHeadTerm->erase();
  origBodyTerm->erase();
  return success();
}

namespace {
/// Driver for the cf DAG rewrite pass.
struct CfFuseLoopHeadBodyPass
    : public compigra::impl::CfFuseLoopHeadBodyBase<CfFuseLoopHeadBodyPass> {

  void runOnOperation() override {
    func::FuncOp funcOp;
    getOperation()->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op))
        funcOp = cast<func::FuncOp>(op);
    });
    OpBuilder builder(&getContext());

    unsigned maxTry = 10;
    // if get the fusible block, fuse it
    while (maxTry--) {
      auto fusibleBody = getFusibleBodyBlock(funcOp);
      if (!fusibleBody.has_value() && !removeCertainCondBr(funcOp, builder))
        break;
      if (fusibleBody.has_value()) {
        auto bodyBlk = fusibleBody.value();
        if (failed(fuseHeadToBody(*bodyBlk->pred_begin(), bodyBlk, builder)))
          return signalPassFailure();
      }
    }
    removeUselessBlockArg(funcOp.getBody(), builder);
  };
};

} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createCfFuseLoopHeadBody() {
  return std::make_unique<CfFuseLoopHeadBodyPass>();
}
} // namespace compigra