//===- CfToCgraConversion.cpp - Convert Cf to Cgra ops   --------*- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --convert-llvm-to-cgra pass, which converts the operations not
// supported in CGRA in llvm dialects to customized cgra dialect.
//
//===----------------------------------------------------------------------===//

#include "compigra/Conversion/CfToCgraConversion.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"

// memory interface support
#include "nlohmann/json.hpp"
#include <fstream>

// Debugging support
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace compigra;
using json = nlohmann::json;

namespace {
/// Get the (true) destination block of the cgra branch operation, if block is
/// a innermost block, return the last block of the successors.
static Block *getCgraBranchDstBlock(Block *block) {
  auto *nextNode = block->getNextNode();

  auto *sucNode = block->getSuccessors().front();
  return nextNode == sucNode ? block->getSuccessors().back() : sucNode;
}

/// Check whether the block should be forced to jump to the next block.
/// cf.cond_br to cgra.cond_br adaptation is composed of single
/// conditional + branch. If the negative successor is the next node, the
/// scheduling guarantee it by PC+1. Otherwise, a new basic block is inserted
/// after with an unconditional branch to the negative successor.
static bool forceJumpToNextBlock(Block *block) {
  auto *nextNode = block->getNextNode();
  for (auto *succ : block->getSuccessors())
    if (succ != nextNode)
      return false;
  return true;
}

static arith::CmpIPredicate reverseCmpFlag(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    predicate = arith::CmpIPredicate::ne;
    break;
  case arith::CmpIPredicate::ne:
    predicate = arith::CmpIPredicate::eq;
    break;
  case arith::CmpIPredicate::slt:
    predicate = arith::CmpIPredicate::sge;
    break;
  case arith::CmpIPredicate::sgt:
    predicate = arith::CmpIPredicate::sle;
    break;
  case arith::CmpIPredicate::sge:
    predicate = arith::CmpIPredicate::slt;
    break;
  case arith::CmpIPredicate::sle:
    predicate = arith::CmpIPredicate::sgt;
    break;
  case arith::CmpIPredicate::ult:
    predicate = arith::CmpIPredicate::uge;
    break;
  case arith::CmpIPredicate::ugt:
    predicate = arith::CmpIPredicate::ule;
    break;
  case arith::CmpIPredicate::uge:
    predicate = arith::CmpIPredicate::ult;
    break;
  case arith::CmpIPredicate::ule:
    predicate = arith::CmpIPredicate::ugt;
    break;
  }
  return predicate;
}

/// CGRA branch only support bne, beq, blt, bge. All predicates must be
/// converted to eq, ne, lt, ge, where in some cases the comparison operands are
/// swapped.
static cgra::CondBrPredicate getCgraBrPredicate(arith::CmpIPredicate pred,
                                                Value &val1, Value &val2) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return cgra::CondBrPredicate::eq;
  case arith::CmpIPredicate::ne:
    return cgra::CondBrPredicate::ne;
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return cgra::CondBrPredicate::lt;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    std::swap(val1, val2);
    return cgra::CondBrPredicate::lt;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return cgra::CondBrPredicate::ge;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    std::swap(val1, val2);
    return cgra::CondBrPredicate::ge;
  }
}

/// Lower cf::CondBranchOp to cgra::CondBranchOp
struct CfCondBrOpConversion : OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern<cf::CondBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp condBrOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // create a new block for the default branch
    Block *condBrBlock = getCgraBranchDstBlock(condBrOp->getBlock());
    bool forceJump = forceJumpToNextBlock(condBrOp->getBlock());
    Block *falseBlk = condBrBlock == condBrOp.getTrueDest()
                          ? condBrOp.getFalseDest()
                          : condBrOp.getTrueDest();
    if (forceJump) {
      rewriter.setInsertionPoint(condBrOp);
      falseBlk = rewriter.createBlock(condBrOp->getBlock()->getNextNode());
    }

    // get predicate from the condition
    arith::CmpIOp cmpOp =
        condBrOp.getCondition().getDefiningOp<arith::CmpIOp>();
    arith::CmpIPredicate predicate = cmpOp.getPredicate();

    Value cmpOpr0 = cmpOp.getOperand(0);
    Value cmpOpr1 = cmpOp.getOperand(1);
    if (cmpOp.getOperand(0).getType().isa<IndexType>()) {
      rewriter.setInsertionPoint(condBrOp);
      arith::IndexCastOp castOp0 = rewriter.create<arith::IndexCastOp>(
          cmpOp->getLoc(), rewriter.getIntegerType(32), cmpOp.getOperand(0));
      arith::IndexCastOp castOp1 = rewriter.create<arith::IndexCastOp>(
          cmpOp->getLoc(), rewriter.getIntegerType(32), cmpOp.getOperand(1));
      cmpOpr0 = castOp0.getResult();
      cmpOpr1 = castOp1.getResult();
    }

    bool isTrueDest = condBrBlock == condBrOp.getTrueDest();
    if (!isTrueDest)
      predicate = reverseCmpFlag(predicate);

    auto condBrArgs = isTrueDest ? condBrOp.getTrueDestOperands()
                                 : condBrOp.getFalseDestOperands();
    auto jumpArgs = isTrueDest ? condBrOp.getFalseDestOperands()
                               : condBrOp.getTrueDestOperands();
    Block *jumpBlock =
        isTrueDest ? condBrOp.getFalseDest() : condBrOp.getTrueDest();

    // replace cf.condbr with cgra.condbr
    rewriter.setInsertionPoint(condBrOp);
    cgra::CondBrPredicate cgraPred =
        getCgraBrPredicate(predicate, cmpOpr0, cmpOpr1);
    auto newCondBr = rewriter.replaceOpWithNewOp<cgra::ConditionalBranchOp>(
        condBrOp, cgraPred, cmpOpr0, cmpOpr1, condBrBlock, condBrArgs, falseBlk,
        SmallVector<Value>());

    if (cmpOp->hasOneUse()) {
      rewriter.eraseOp(cmpOp);
    }

    // llvm::errs() << "CmpOp" << cmpOp->hasOneUse() << " " << cmpOp.use_empty()
    //              << "\n";
    if (forceJump) {
      rewriter.setInsertionPointToStart(falseBlk);
      auto defaultBr = rewriter.create<cf::BranchOp>(newCondBr->getLoc(),
                                                     jumpArgs, jumpBlock);
      defaultBr->moveAfter(&falseBlk->getOperations().front());
    }

    return success();
  }
};

} // namespace

void compigra::populateCfToCgraConversionPatterns(RewritePatternSet &patterns) {
  //   patterns.insert<cf::CondBranchOp>(context);
  llvm::errs() << "populateCfToCgraConversionPatterns\n";
  patterns.add<CfCondBrOpConversion>(patterns.getContext());
}

void CfToCgraConversionPass::runOnOperation() {
  llvm::errs() << "CfToCgraConversionPass\n";
  ConversionTarget target(getContext());
  target.addIllegalOp<cf::CondBranchOp>();
  target.addLegalOp<cgra::ConditionalBranchOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
//   target.addIllegalOp<arith::CmpIOp>();

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  populateCfToCgraConversionPatterns(patterns);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

  ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
  llvm::errs() << modOp << "\n";
}

namespace compigra {
std::unique_ptr<mlir::Pass> createCfToCgraConversion(StringRef memAlloc) {
  return std::make_unique<CfToCgraConversionPass>(memAlloc);
}
} // namespace compigra