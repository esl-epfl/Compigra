//===- LLVMToCgraConversion.cpp - Convert LLVM to Cgra ops   ----*- C++ -*-===//
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

#include "compigra/Conversion/LLVMToCgraConversion.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"

// memory interface support
#include "nlohmann/json.hpp"
#include <fstream>

// Debugging support
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "LLVM_TO_CGRA_CONVERSION"

using namespace mlir;
using namespace compigra;
using json = nlohmann::json;

namespace {
/// Conversion target for lowering a func::FuncOp to a handshake::FuncOp.
struct LowerFuncOpTarget : public ConversionTarget {
  explicit LowerFuncOpTarget(MLIRContext &context) : ConversionTarget(context) {
    loweredFuncs.clear();
    addLegalDialect<cgra::CgraDialect, LLVM::LLVMDialect>();
    addDynamicallyLegalOp<LLVM::LLVMFuncOp>(
        [&](const auto &op) { return loweredFuncs.contains(op); });
  }

  SmallPtrSet<Operation *, 4> loweredFuncs;
};

/// Conversion pattern for partially lowering a func::FuncOp to a
/// cgra::FuncOp. Lowering is achieved by a provided partial lowering
/// function.
struct PartialLowerFuncOp : public OpConversionPattern<LLVM::LLVMFuncOp> {
  using PartialLoweringFunc = std::function<LogicalResult(
      LLVM::LLVMFuncOp, ConversionPatternRewriter &)>;

  PartialLowerFuncOp(LowerFuncOpTarget &target, MLIRContext *context,
                     const PartialLoweringFunc &fun)
      : OpConversionPattern<LLVM::LLVMFuncOp>(context), target(target),
        loweringFunc(fun) {}
  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    // Dialect conversion scheme requires the matched root operation to be
    // replaced or updated if the match was successful; this ensures that
    // happens even if the lowering function does not modify the root operation
    LogicalResult res = failure();
    rewriter.modifyOpInPlace(op, [&] { res = loweringFunc(op, rewriter); });

    // Signal to the conversion target that the conversion pattern
    target.loweredFuncs.insert(op);

    // Success status of conversion pattern determined by success of partial
    // lowering function
    return res;
  };

private:
  /// The conversion target for this pattern.
  LowerFuncOpTarget &target;
  /// The rewrite function.
  PartialLoweringFunc loweringFunc;
};

/// Conversion target for lowering a region.
struct LowerRegionTarget : public ConversionTarget {
  explicit LowerRegionTarget(MLIRContext &context, Region &region)
      : ConversionTarget(context), region(region) {
    // The root operation is marked dynamically legal to ensure
    // the pattern on its region is only applied once.
    markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (op != region.getParentOp())
        return true;
      return regionLowered;
    });
  }

  /// Whether the region's parent operation was lowered.
  bool regionLowered = false;
  /// The region being lowered.
  Region &region;
};

/// Allows to partially lower a region by matching on the parent operation to
/// then call the provided partial lowering function with the region and the
/// rewriter.
///
/// The interplay with the target is similar to `PartialLowerFuncOp`.
struct PartialLowerRegion : public ConversionPattern {
  using PartialLoweringFunc =
      std::function<LogicalResult(Region &, ConversionPatternRewriter &)>;

  PartialLowerRegion(LowerRegionTarget &target, MLIRContext *context,
                     LogicalResult &loweringResRef,
                     const PartialLoweringFunc &fun)
      : ConversionPattern(target.region.getParentOp()->getName().getStringRef(),
                          1, context),
        target(target), loweringRes(loweringResRef), fun(fun) {}
  using ConversionPattern::ConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    // Dialect conversion scheme requires the matched root operation to be
    // replaced or updated if the match was successful; this ensures that
    // happens even if the lowering function does not modify the root operation
    rewriter.modifyOpInPlace(
        op, [&] { loweringRes = fun(target.region, rewriter); });

    // Signal to the conversion target that the conversion pattern ran
    target.regionLowered = true;

    // Success status of conversion pattern determined by success of partial
    // lowering function
    return loweringRes;
  };

private:
  LowerRegionTarget &target;
  LogicalResult &loweringRes;
  PartialLoweringFunc fun;
};
} // namespace

static LogicalResult
partiallyLowerOp(const PartialLowerFuncOp::PartialLoweringFunc &loweringFunc,
                 LLVM::LLVMFuncOp funcOp) {
  MLIRContext *ctx = funcOp->getContext();
  RewritePatternSet patterns(ctx);
  LowerFuncOpTarget target(*ctx);
  patterns.add<PartialLowerFuncOp>(target, ctx, loweringFunc);
  return applyPartialConversion(funcOp, target, std::move(patterns));
}

// Get the stage of the block in SATMapIt DAG
enum class SATLoopBlock { Unkown, Init, Loop, Fini };
static SATLoopBlock getSATMapItBlockType(Block *block) {
  if (block->isEntryBlock()) {
    return SATLoopBlock::Init;
  }

  if (isa<LLVM::ReturnOp>(block->getTerminator())) {
    return SATLoopBlock::Fini;
  }

  if (auto condBranchOp = dyn_cast<LLVM::CondBrOp>(block->getTerminator()))
    for (auto attr : condBranchOp.getAttributeNames())
      if (attr.contains(StringRef("loop_annotation")))
        return SATLoopBlock::Loop;

  return SATLoopBlock::Unkown;
}

LogicalResult CgraLowering::reorderBBs(ConversionPatternRewriter &rewriter) {
  // can only reorder the basic blocks if the region satisfies the SAT DAG
  Block *loopBlock = nullptr;
  for (auto &block : region)
    if (getSATMapItBlockType(&block) == SATLoopBlock::Loop) {
      loopBlock = &block;
      break;
    }

  // No need to reorder BBs if there is no loop block
  if (loopBlock == nullptr)
    return success();

  // ensure the successor of the loop block is its nextNode
  Block *nextNode = nullptr;
  for (auto suc : loopBlock->getSuccessors()) {
    if (suc != loopBlock) {
      nextNode = suc;
      break;
    }
  }

  // next Node of BB shouldn't be nullptr
  if (nextNode == nullptr)
    return failure();

  if (loopBlock->getNextNode() != nextNode)
    rewriter.moveBlockBefore(loopBlock, nextNode);

  return success();
}

/// Get the users of the ICmp operation with the given type.
template <typename T>
static SmallVector<Operation *> getICmpOpUsers(Value val) {
  SmallVector<Operation *> cmpOps;
  for (auto &u : val.getUses()) {
    if (auto cmpOp = dyn_cast<T>(u.getOwner()))
      cmpOps.push_back(cmpOp);
  }
  return cmpOps;
}

Operation *CgraLowering::getConstantOp() {
  for (Operation &op : region.getOps())
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(&op))
      return constOp;
  return nullptr;
}

LogicalResult
CgraLowering::raiseConstOnlyUse(ConversionPatternRewriter &rewriter) {
  for (auto constOp : region.getOps<LLVM::ConstantOp>()) {
    // if the constant operation has only one user, skip it
    if (constOp->hasOneUse())
      continue;

    // iterate through the uses of the constant operation, and replace the uses
    // from the second one
    for (auto &use : llvm::make_early_inc_range(constOp->getUses())) {
      // skip the first use to keep the constant operation
      // if (ind == 0)
      //   continue;
      auto user = use.getOwner();
      rewriter.setInsertionPoint(constOp);
      auto insertCstOp = rewriter.create<LLVM::ConstantOp>(
          constOp.getLoc(), constOp.getType(), constOp.getValue());
      //  replace the user's use of constOp result with insertCstOp result
      user->setOperand(use.getOperandNumber(), insertCstOp.getResult());
    }
  }
  return success();
}

/// Get the destination block of the cgra branch operation, which shouldn't be
/// the block below it
static Block *getCgraBranchDstBlock(Block *block) {
  auto *nextNode = block->getNextNode();
  auto *sucNode = block->getSuccessors().front();
  return nextNode == sucNode ? block->getSuccessors().back() : sucNode;
}

LogicalResult CgraLowering::replaceCmpOps(ConversionPatternRewriter &rewriter) {
  // Store all the cmp operations to be replaced
  SmallVector<Operation *> cmpOps;
  SmallVector<Operation *> selectOps;
  SmallVector<Operation *> eraseBrOps;

  for (Operation &op : llvm::make_early_inc_range(region.getOps())) {
    if (!isa<LLVM::ICmpOp>(op))
      continue;

    auto cmpOp = dyn_cast<LLVM::ICmpOp>(op);
    cmpOps.push_back(&op);

    // remove the cmp operation if it has no users
    if (op.getUsers().empty())
      continue;

    auto predicate = cmpOp.getPredicate();

    // substitute the cmp operation with sub operation and using its zero/sign
    // flag to signal the result of comparison.
    auto selOps = getICmpOpUsers<LLVM::SelectOp>(cmpOp.getResult());
    auto condBrOps = getICmpOpUsers<LLVM::CondBrOp>(cmpOp.getResult());

    bool existNonSelAndBrUser =
        static_cast<size_t>(
            std::distance(op.getUsers().begin(), op.getUsers().end())) >
        selOps.size() + condBrOps.size();

    rewriter.setInsertionPoint(cmpOp);
    LLVM::SubOp subOp = nullptr;
    // Reverse the operands order for greater than or equal to and greater
    if (predicate == LLVM::ICmpPredicate::uge ||
        predicate == LLVM::ICmpPredicate::sge ||
        predicate == LLVM::ICmpPredicate::ugt ||
        predicate == LLVM::ICmpPredicate::sgt) {
      subOp = rewriter.create<LLVM::SubOp>(cmpOp.getLoc(), cmpOp.getOperand(1),
                                           cmpOp.getOperand(0));
    } else
      subOp = rewriter.create<LLVM::SubOp>(cmpOp.getLoc(), cmpOp.getOperand(0),
                                           cmpOp.getOperand(1));

    // insert additional bzfa operation to conclude equal case of the
    // comparison.
    auto selectFlag = subOp.getResult();
    if (predicate == LLVM::ICmpPredicate::uge ||
        predicate == LLVM::ICmpPredicate::uge ||
        predicate == LLVM::ICmpPredicate::ule ||
        predicate == LLVM::ICmpPredicate::ule) {
      // create constant -1 to indicate the equal case
      auto resType = subOp.getResult().getType();
      LLVM::ConstantOp constOp = rewriter.create<LLVM::ConstantOp>(
          cmpOp.getLoc(), resType, APInt(resType.getIntOrFloatBitWidth(), -1));

      // create bzfa operation to include the equal case
      cgra::BzfaOp bzfaOp = rewriter.create<cgra::BzfaOp>(
          cmpOp.getLoc(), subOp.getResult().getType(), subOp.getResult(),
          SmallVector<Value>({constOp.getResult(), subOp.getResult()}));
      selectFlag = bzfaOp.getResult();
    }

    // Replace the select operation with bsfa/bzfa operation.
    if (!selOps.empty()) {
      // replace the bsfa with select operation
      for (auto selOp : selOps) {
        rewriter.setInsertionPoint(selOp);
        if (predicate == LLVM::ICmpPredicate::eq) {
          rewriter.replaceOpWithNewOp<cgra::BzfaOp>(
              selOp, selOp->getResult(0).getType(), selectFlag,
              SmallVector<Value>({selOp->getOperand(1), selOp->getOperand(2)}));
        } else if (predicate == LLVM::ICmpPredicate::ne) {
          rewriter.replaceOpWithNewOp<cgra::BzfaOp>(
              selOp, selOp->getResult(0).getType(), selectFlag,
              SmallVector<Value>({selOp->getOperand(2), selOp->getOperand(1)}));
        } else {
          rewriter.replaceOpWithNewOp<cgra::BsfaOp>(
              selOp, selOp->getResult(0).getType(), selectFlag,
              SmallVector<Value>({selOp->getOperand(1), selOp->getOperand(2)}));
        }
      }
    }

    // Replace the cond_br operation with corresponding branch operation in cgra
    if (replaceBranch && !condBrOps.empty()) {
      for (auto brOp : condBrOps) {
        eraseBrOps.push_back(brOp);
        LLVM::CondBrOp condBrOp = dyn_cast<LLVM::CondBrOp>(brOp);

        auto predicate = cmpOp.getPredicate();

        // Get the conditional branch block for cgra bge, blt, etc, which should
        // not be the next node of current block.
        Block *condBrBlock = getCgraBranchDstBlock(condBrOp->getBlock());

        // If the next node is not the trueDest, revise the conditional flag to
        // be the opposite.
        // e.g if a==b branch to ^2 else ^1 => if a!=b branch to ^1 else ^2
        // e.g if a>b branch to ^2 else ^1 => if a<=b branch to ^1 else ^2
        // The rewrite does not change the functionality.
        bool isTrueDest = condBrBlock == condBrOp.getTrueDest();
        if (!isTrueDest) {
          switch (predicate) {
          case LLVM::ICmpPredicate::eq:
            predicate = LLVM::ICmpPredicate::ne;
            break;
          case LLVM::ICmpPredicate::ne:
            predicate = LLVM::ICmpPredicate::eq;
            break;
          case LLVM::ICmpPredicate::slt:
            predicate = LLVM::ICmpPredicate::sge;
            break;
          case LLVM::ICmpPredicate::sgt:
            predicate = LLVM::ICmpPredicate::sle;
            break;
          case LLVM::ICmpPredicate::sge:
            predicate = LLVM::ICmpPredicate::slt;
            break;
          case LLVM::ICmpPredicate::sle:
            predicate = LLVM::ICmpPredicate::sgt;
            break;
          case LLVM::ICmpPredicate::ult:
            predicate = LLVM::ICmpPredicate::uge;
            break;
          case LLVM::ICmpPredicate::ugt:
            predicate = LLVM::ICmpPredicate::ule;
            break;
          case LLVM::ICmpPredicate::uge:
            predicate = LLVM::ICmpPredicate::ult;
            break;
          case LLVM::ICmpPredicate::ule:
            predicate = LLVM::ICmpPredicate::ugt;
            break;
          }
        }

        auto condBrArgs = isTrueDest ? condBrOp.getTrueDestOperands()
                                     : condBrOp.getFalseDestOperands();

        // Add a new basic block after the beq, bne, blt, bge, block as the
        // false branch rewriter.
        Block *jumpBlock =
            isTrueDest ? condBrOp.getFalseDest() : condBrOp.getTrueDest();
        auto jumpArgs = isTrueDest ? condBrOp.getFalseDestOperands()
                                   : condBrOp.getTrueDestOperands();

        //  get type range of jumpArgs
        auto jumpArgsType = jumpArgs.getTypes();
        // Create a vector of locations and fill it with the location of the
        // terminator

        auto newDefaltBlk =
            rewriter.createBlock(brOp->getBlock()->getNextNode());

        switch (predicate) {
        case LLVM::ICmpPredicate::eq: {
          rewriter.setInsertionPoint(brOp);
          auto op = rewriter.create<cgra::ConditionalBranchOp>(
              brOp->getLoc(), cgra::CondBrPredicate::eq, cmpOp.getOperand(0),
              cmpOp.getOperand(1), condBrBlock, condBrArgs, newDefaltBlk,
              SmallVector<Value>());
          break;
        }
        case LLVM::ICmpPredicate::ne: {
          rewriter.setInsertionPoint(brOp);
          auto op = rewriter.create<cgra::ConditionalBranchOp>(
              brOp->getLoc(), cgra::CondBrPredicate::ne, cmpOp.getOperand(0),
              cmpOp.getOperand(1), condBrBlock, condBrArgs, newDefaltBlk,
              SmallVector<Value>());

          break;
        }
        case LLVM::ICmpPredicate::slt:
        case LLVM::ICmpPredicate::ult: {
          rewriter.setInsertionPoint(brOp);
          auto op = rewriter.create<cgra::ConditionalBranchOp>(
              brOp->getLoc(), cgra::CondBrPredicate::lt, cmpOp.getOperand(0),
              cmpOp.getOperand(1), condBrBlock, condBrArgs, newDefaltBlk,
              SmallVector<Value>());
          break;
        }
        case LLVM::ICmpPredicate::sgt:
        case LLVM::ICmpPredicate::ugt: {
          rewriter.setInsertionPoint(brOp);
          auto op = rewriter.create<cgra::ConditionalBranchOp>(
              brOp->getLoc(), cgra::CondBrPredicate::lt, cmpOp.getOperand(1),
              cmpOp.getOperand(0), condBrBlock, condBrArgs, newDefaltBlk,
              SmallVector<Value>());
          break;
        }
        case LLVM::ICmpPredicate::sge:
        case LLVM::ICmpPredicate::uge: {
          rewriter.setInsertionPoint(brOp);
          auto op = rewriter.create<cgra::ConditionalBranchOp>(
              brOp->getLoc(), cgra::CondBrPredicate::ge, cmpOp.getOperand(0),
              cmpOp.getOperand(1), condBrBlock, condBrArgs, newDefaltBlk,
              SmallVector<Value>());
          break;
        }
        case LLVM::ICmpPredicate::sle:
        case LLVM::ICmpPredicate::ule: {
          rewriter.setInsertionPoint(brOp);
          auto op = rewriter.create<cgra::ConditionalBranchOp>(
              brOp->getLoc(), cgra::CondBrPredicate::ge, cmpOp.getOperand(1),
              cmpOp.getOperand(0), condBrBlock, condBrArgs, newDefaltBlk,
              SmallVector<Value>());
          break;
        }
        default:
          return failure();
        }

        // insert branch operation to the new block
        rewriter.setInsertionPointToStart(newDefaltBlk);
        auto defaultBr =
            rewriter.create<LLVM::BrOp>(brOp->getLoc(), jumpArgs, jumpBlock);
        defaultBr->moveAfter(&newDefaltBlk->getOperations().front());
      }
    }

    // if exists user Op except select and branch, create a bsfa operation to
    // select either 0 or 1
    if (existNonSelAndBrUser) {
      rewriter.setInsertionPoint(getConstantOp());
      LLVM::ConstantOp constOp0 = rewriter.create<LLVM::ConstantOp>(
          cmpOp.getLoc(), rewriter.getI1Type(), APInt(1, 0));
      LLVM::ConstantOp constOp1 = rewriter.create<LLVM::ConstantOp>(
          cmpOp.getLoc(), rewriter.getI1Type(), APInt(1, 1));
      rewriter.setInsertionPoint(cmpOp);
      Operation *binSelOp = nullptr;
      if (predicate == LLVM::ICmpPredicate::eq) {
        // insert bzfa %selFlag, 1, 0;
        binSelOp = rewriter.create<cgra::BzfaOp>(
            cmpOp.getLoc(), rewriter.getI1Type(), selectFlag,
            SmallVector<Value>({constOp1.getResult(), constOp0.getResult()}));
      } else if (predicate == LLVM::ICmpPredicate::ne) {
        // insert bzfa %selFlag, 0, 1;
        binSelOp = rewriter.create<cgra::BzfaOp>(
            cmpOp.getLoc(), rewriter.getI1Type(), selectFlag,
            SmallVector<Value>({constOp0.getResult(), constOp1.getResult()}));
      } else {
        // insert bsfa %selFlag, 1, 0;
        binSelOp = rewriter.create<cgra::BsfaOp>(
            cmpOp.getLoc(), rewriter.getI1Type(), selectFlag,
            SmallVector<Value>({constOp1.getResult(), constOp0.getResult()}));
      }

      rewriter.replaceAllUsesWith(cmpOp.getResult(), binSelOp->getResult(0));
    }
  }

  for (Operation *op : cmpOps)
    rewriter.eraseOp(op);
  for (Operation *op : selectOps)
    rewriter.eraseOp(op);
  for (Operation *op : eraseBrOps)
    rewriter.eraseOp(op);

  return success();
}

LogicalResult
compigra::partiallyLowerRegion(const RegionLoweringFunc &loweringFunc,
                               Region &region) {
  Operation *op = region.getParentOp();
  MLIRContext *ctx = region.getContext();
  RewritePatternSet patterns(ctx);
  LowerRegionTarget target(*ctx, region);
  LogicalResult partialLoweringSuccessfull = success();
  patterns.add<PartialLowerRegion>(target, ctx, partialLoweringSuccessfull,
                                   loweringFunc);
  return success(
      applyPartialConversion(op, target, std::move(patterns)).succeeded() &&
      partialLoweringSuccessfull.succeeded());
}

template <typename T> static size_t getArgsIndex(T val, SmallVector<T> set) {
  for (auto [ind, arg] : llvm::enumerate(set)) {
    if (arg == val)
      return ind;
  }
}

LogicalResult
CgraLowering::addMemoryInterface(ConversionPatternRewriter &rewriter) {
  auto funcOp = region.getParentOfType<cgra::FuncOp>();
  DenseMap<Value, Operation *> arrayBaseAddrs;

  SmallVector<Operation *> gepOps;
  SmallVector<Operation *> loadOps;
  Operation *entryOp = getConstantOp();

  for (auto loadOp :
       llvm::make_early_inc_range(region.getOps<LLVM::LoadOp>())) {
    auto arg = loadOp.getOperand();

    // if it is a function argument, replace lwi directly
    if (std::find(funcOp.getArguments().begin(), funcOp.getArguments().end(),
                  arg) != funcOp.getArguments().end()) {
      auto argIndex = getArgsIndex<Value>(
          arg, SmallVector<Value>{funcOp.getArguments().begin(),
                                  funcOp.getArguments().end()});

      // insert lwi operation to load the integer variable
      rewriter.setInsertionPoint(entryOp);
      LLVM::ConstantOp constOp = rewriter.create<LLVM::ConstantOp>(
          entryOp->getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(memInterface.startAddr[argIndex]));
      constOp->setAttr("hostValue", rewriter.getStringAttr(
                                        "arg" + std::to_string(argIndex)));

      rewriter.setInsertionPoint(loadOp);

      cgra::LwiOp lwiOp = rewriter.create<cgra::LwiOp>(
          loadOp.getLoc(), loadOp.getResult().getType(), constOp.getResult());

      rewriter.replaceOp(loadOp, lwiOp.getResult());
      continue;
    }

    // calculate the address if it is a GEP operation
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(arg.getDefiningOp())) {
      gepOps.push_back(gepOp);
      Value baseAddr = gepOp.getOperand(0);
      auto argIndex = getArgsIndex<Value>(
          baseAddr, SmallVector<Value>{funcOp.getArguments().begin(),
                                       funcOp.getArguments().end()});
      rewriter.setInsertionPoint(entryOp);
      // Add offset constant
      LLVM::ConstantOp addrEle = rewriter.create<LLVM::ConstantOp>(
          entryOp->getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(4));
      // Add base address constant
      LLVM::ConstantOp baseOp = rewriter.create<LLVM::ConstantOp>(
          entryOp->getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(memInterface.startAddr[argIndex]));
      baseOp->setAttr("hostValue",
                      rewriter.getStringAttr("arg" + std::to_string(argIndex)));
      rewriter.setInsertionPoint(loadOp);
      LLVM::MulOp offsetOp = rewriter.create<LLVM::MulOp>(
          loadOp.getLoc(), rewriter.getI32Type(), gepOp.getOperand(1),
          addrEle.getResult());
      LLVM::AddOp addrOp = rewriter.create<LLVM::AddOp>(
          loadOp.getLoc(), rewriter.getI32Type(), baseOp.getResult(),
          offsetOp.getResult());

      cgra::LwiOp lwiOp = rewriter.create<cgra::LwiOp>(
          loadOp.getLoc(), loadOp.getResult().getType(), addrOp.getResult());

      rewriter.replaceOp(loadOp, lwiOp.getResult());
    }
  }

  // calculate the address if it is a GEP operation
  for (auto storeOp :
       llvm::make_early_inc_range(region.getOps<LLVM::StoreOp>())) {
    auto arg = storeOp.getOperand(1);
    // if it is a function argument, replace swi directly
    if (std::find(funcOp.getArguments().begin(), funcOp.getArguments().end(),
                  arg) != funcOp.getArguments().end()) {

      auto argIndex = getArgsIndex<Value>(
          arg, SmallVector<Value>{funcOp.getArguments().begin(),
                                  funcOp.getArguments().end()});

      // insert lwi operation to load the integer variable
      rewriter.setInsertionPoint(entryOp);
      LLVM::ConstantOp constOp = rewriter.create<LLVM::ConstantOp>(
          storeOp.getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(memInterface.startAddr[argIndex]));
      constOp->setAttr("returnValue", rewriter.getStringAttr(
                                          "arg" + std::to_string(argIndex)));
      rewriter.setInsertionPoint(storeOp);
      rewriter.create<cgra::SwiOp>(storeOp.getLoc(), storeOp.getOperand(0),
                                   constOp.getResult());

      rewriter.eraseOp(storeOp);
      continue;
    }

    // calculate the address if it is a GEP operation
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(arg.getDefiningOp())) {
      gepOps.push_back(gepOp);
      Value baseAddr = gepOp.getOperand(0);
      auto argIndex = getArgsIndex<Value>(
          baseAddr, SmallVector<Value>{funcOp.getArguments().begin(),
                                       funcOp.getArguments().end()});
      rewriter.setInsertionPoint(entryOp);
      // Add offset constant TODO: CHECK THIS
      LLVM::ConstantOp addrEle = rewriter.create<LLVM::ConstantOp>(
          entryOp->getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(4));
      // Add base address constant
      LLVM::ConstantOp baseOp = rewriter.create<LLVM::ConstantOp>(
          entryOp->getLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(memInterface.startAddr[argIndex]));
      baseOp->setAttr("returnValue",
                      rewriter.getStringAttr("arg" + std::to_string(argIndex)));
      rewriter.setInsertionPoint(storeOp);
      LLVM::MulOp offsetOp = rewriter.create<LLVM::MulOp>(
          storeOp.getLoc(), rewriter.getI32Type(), gepOp.getOperand(1),
          addrEle.getResult());
      LLVM::AddOp addrOp = rewriter.create<LLVM::AddOp>(
          storeOp.getLoc(), rewriter.getI32Type(), baseOp.getResult(),
          offsetOp.getResult());
      cgra::SwiOp swiOp = rewriter.create<cgra::SwiOp>(
          storeOp.getLoc(), storeOp.getOperand(0), addrOp.getResult());
      rewriter.eraseOp(storeOp);
    }
  }
  for (Operation *op : gepOps)
    rewriter.eraseOp(op);

  return success();
}

LogicalResult
CgraLowering::removeUnusedOps(ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> eraseOps;
  for (auto &op : region.getOps()) {
    if (op.getBlock()->getTerminator() == &op || isa<cgra::SwiOp>(op))
      continue;

    if (op.use_empty()) {
      eraseOps.push_back(&op);

      // Backtrack the definition Op if its only user has been erased
      SmallVector<Operation *> toProcess = eraseOps;
      while (!toProcess.empty()) {
        Operation *op = toProcess.back();
        toProcess.pop_back();

        for (Value operand : op->getOperands()) {
          Operation *predOp = operand.getDefiningOp();
          if (predOp && predOp->hasOneUse()) {
            // Erase an operation once
            if (std::find(eraseOps.begin(), eraseOps.end(), predOp) ==
                eraseOps.end())
              eraseOps.push_back(predOp);
            toProcess.push_back(predOp);
          }
        }
      }
    }
  }

  for (Operation *op : eraseOps)
    rewriter.eraseOp(op);

  return success();
}

/// Run partial lowering functions on the region.
static LogicalResult lowerRegion(CgraLowering &cl) {
  if (failed(runPartialLowering(cl, &CgraLowering::reorderBBs)))
    return failure();

  if (failed(runPartialLowering(cl, &CgraLowering::addMemoryInterface)))
    return failure();

  llvm::errs() << "add memory interface success\n";
  if (failed(runPartialLowering(cl, &CgraLowering::replaceCmpOps)))
    return failure();

  if (failed(runPartialLowering(cl, &CgraLowering::raiseConstOnlyUse)))
    return failure();

  if (failed(runPartialLowering(cl, &CgraLowering::removeUnusedOps)))
    return failure();

  return success();
}

/// Lower a func::FuncOp to a cgra::FuncOp.
static LogicalResult lowerFuncOp(LLVM::LLVMFuncOp funcOp, StringRef outputDAG,
                                 MemoryInterface &memInterface,
                                 MLIRContext *ctx) {

  // The cgra function only retains the original function's symbol and
  // function type
  SmallVector<NamedAttribute, 4> attributes;
  for (const NamedAttribute &attr : funcOp->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == funcOp.getFunctionTypeAttrName())
      continue;
    attributes.push_back(attr);
  }

  // Get function arguments
  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (auto &argType : funcOp.getArgumentTypes())
    argTypes.push_back(argType);

  // Get function results
  llvm::SmallVector<mlir::Type, 8> resTypes;
  for (auto resType : funcOp.getResultTypes())
    resTypes.push_back(resType);

  // Get existing attributes into a new vector and add the new attribute.
  llvm::SmallVector<mlir::Attribute, 4> argAttrs;

  // Replaces the func-level function with a corresponding Handshake-level
  // function.
  cgra::FuncOp newFuncOp = nullptr;

  auto funcLowering = [&](LLVM::LLVMFuncOp funcOp, PatternRewriter &rewriter) {
    if (!argTypes.empty())
      for (auto argAttr : funcOp.getAllArgAttrs())
        argAttrs.push_back(argAttr);

    auto funcType = rewriter.getFunctionType(argTypes, resTypes);
    newFuncOp = rewriter.create<cgra::FuncOp>(funcOp.getLoc(), funcOp.getName(),
                                              funcType, attributes);

    if (!argTypes.empty())
      newFuncOp.setArgAttrsAttr(ArrayAttr::get(ctx, argAttrs));

    if (funcOp.isExternal())
      return success();

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (!newFuncOp.isExternal()) {
      newFuncOp.resolveArgAndResNames();
    }

    return success();
  };

  if (failed(partiallyLowerOp(funcLowering, funcOp)))
    return failure();

  funcOp.erase();
  if (!newFuncOp.isExternal()) {
    CgraLowering cl(newFuncOp.getBody());
    cl.initMemoryInterface(memInterface);
    return lowerRegion(cl);
  }
  return success();
};

static LogicalResult parseMemoryInterface(MemoryInterface &memInterface,
                                          json &memAttr,
                                          std::string parseFunc) {
  // print the json file
  for (auto &element : memAttr.items()) {
    auto funcName = element.key();
    auto value = element.value();

    bool isParseFunc = funcName == parseFunc;
    if (isParseFunc) {

      // parse the activeArrTail, which is an array
      if (value.contains("startAddr") && value["startAddr"].is_array()) {
        std::vector<int> startAddr;
        for (auto &arrTail : value["startAddr"]) {
          std::string arrTailVal = arrTail;
          startAddr.push_back(std::stoi(arrTailVal, nullptr, 16));
        }
        memInterface.startAddr = startAddr;
      } else {
        return failure();
      }

      // parse the endAddr, which is for exceed boundary check (optional)
      if (value.contains("endAddr") && value["endAddr"].is_array()) {
        std::vector<int> endAddr;
        for (auto &arrTail : value["endAddr"]) {
          std::string arrTailVal = arrTail;
          endAddr.push_back(std::stoi(arrTailVal, nullptr, 16));
        }
        memInterface.endAddr = endAddr;
      }
      return success();
    }
  }
  return failure();
}

void LLVMToCgraConversionPass::runOnOperation() {
  ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());

  // Parse Memory Interface from input json file
  std::ifstream infile(memAlloc.getValue());
  if (!infile.is_open()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to open the memory description file: "
                            << memAlloc << "\n");
    return signalPassFailure();
  }

  json memJson = json::parse(infile);
  MemoryInterface memInterface;
  if (failed(parseMemoryInterface(memInterface, memJson, funcName.getValue())))
    return signalPassFailure();

  for (auto global : llvm::make_early_inc_range(modOp.getOps<LLVM::GlobalOp>()))
    // remove global definition
    global.erase();
  // rewrite funcOp to cgra::FuncOp
  SmallVector<Operation *> eraseFuncOps;
  for (auto funcOp :
       llvm::make_early_inc_range(modOp.getOps<LLVM::LLVMFuncOp>())) {
    // Not lower the function if it is not required
    if (funcName == funcOp.getName()) {
      if (failed(lowerFuncOp(funcOp, funcName, memInterface, &getContext())))
        return signalPassFailure();
    } else
      // erase unused function
      eraseFuncOps.push_back(funcOp);
  }

  // erase the function
  for (auto funcOp : eraseFuncOps) {
    funcOp->dropAllUses();
    funcOp->erase();
  }
};

namespace compigra {
std::unique_ptr<mlir::Pass> createLLVMToCgraConversion(StringRef funcName,
                                                       StringRef memAlloc) {
  return std::make_unique<LLVMToCgraConversionPass>(funcName, memAlloc);
}
} // namespace compigra