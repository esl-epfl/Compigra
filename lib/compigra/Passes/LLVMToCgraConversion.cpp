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

#include "compigra/Passes/LLVMToCgraConversion.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/ilist.h"
#include "llvm/Support/Format.h"

// memory interface support
#include "nlohmann/json.hpp"

// Debugging support
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace compigra;

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

/// Get the number of predecessor blocks of a block.
static unsigned getBlockPredecessorCount(Block *block) {
  auto predecessors = block->getPredecessors();
  return std::distance(predecessors.begin(), predecessors.end());
}

LogicalResult CgraLowering::addMergeOps(ConversionPatternRewriter &rewriter) {
  ValueMap mergePairs;

  for (Block &block : region) {
    rewriter.setInsertionPointToStart(&block);
    auto insertLoc = block.front().getLoc();

    for (size_t i = 0; i < block.getNumArguments(); i++) {
      std::vector<Value> operands;
      // get the operands from its successors
      for (auto pred : block.getPredecessors()) {
        if (auto branchOp = dyn_cast<LLVM::BrOp>(pred->getTerminator())) {
          operands.push_back(branchOp.getOperand(i));
          continue;
        } else if (auto condBranchOp =
                       dyn_cast<LLVM::CondBrOp>(pred->getTerminator())) {
          if (&block == condBranchOp.getTrueDest()) {
            operands.push_back(condBranchOp.getTrueDestOperands()[i]);
          } else if (&block == condBranchOp.getFalseDest()) {
            operands.push_back(condBranchOp.getFalseDestOperands()[i]);
          } else {
            return failure();
          }
        }
      }
      if (operands.size() <= 1)
        continue;
      auto mergeOp = rewriter.create<cgra::MergeOp>(insertLoc, operands);
      insertLoc = mergeOp.getLoc();
      // replace the argument with the result of mergeOp
      block.getArgument(i).replaceAllUsesWith(mergeOp.getResult());
    }
  }
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

LogicalResult CgraLowering::replaceCmpOps(ConversionPatternRewriter &rewriter) {
  // Store all the cmp operations to be replaced
  SmallVector<Operation *> cmpOps;
  SmallVector<Operation *> selectOps;

  for (Operation &op : region.getOps()) {
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
    auto branchOps = getICmpOpUsers<LLVM::CondBrOp>(cmpOp.getResult());

    bool existNonSelectUser =
        static_cast<size_t>(std::distance(op.getUsers().begin(),
                                          op.getUsers().end())) > selOps.size();

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

    auto selectFlag = subOp.getResult();
    if (predicate == LLVM::ICmpPredicate::uge ||
        predicate == LLVM::ICmpPredicate::uge ||
        predicate == LLVM::ICmpPredicate::ule ||
        predicate == LLVM::ICmpPredicate::ule) {
      // create constant -1
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
        rewriter.replaceOpWithNewOp<cgra::BsfaOp>(
            selOp, selOp->getResult(0).getType(), selectFlag,
            SmallVector<Value>({selOp->getOperand(1), selOp->getOperand(2)}));
      }
    }

    // Replace the cond_br operation with cgra branch operation.
    if (!branchOps.empty()) {
      auto jumpOprand = op.getBlock()->getOperations().front().getResult(0);
      switch (predicate) {
      case LLVM::ICmpPredicate::eq:
        for (auto branchOp : branchOps) {
          rewriter.setInsertionPoint(branchOp);
          rewriter.create<cgra::BeqOp>(
              branchOp->getLoc(), jumpOprand.getType(),
              SmallVector<Value>(
                  {cmpOp.getOperand(0), cmpOp.getOperand(1), jumpOprand}));
        }
        break;
      case LLVM::ICmpPredicate::ne:
        for (auto branchOp : branchOps) {
          rewriter.setInsertionPoint(branchOp);
          rewriter.create<cgra::BneOp>(
              branchOp->getLoc(), jumpOprand.getType(),
              SmallVector<Value>(
                  {cmpOp.getOperand(0), cmpOp.getOperand(1), jumpOprand}));
        }
        break;
      default:
        return failure();
      }
    }

    // if exists user Op except select, create a bsfa operation to select either
    // 0 or 1
    if (existNonSelectUser) {
      rewriter.setInsertionPoint(getConstantOp());
      LLVM::ConstantOp constOp0 = rewriter.create<LLVM::ConstantOp>(
          cmpOp.getLoc(), rewriter.getI1Type(), APInt(1, 0));
      LLVM::ConstantOp constOp1 = rewriter.create<LLVM::ConstantOp>(
          cmpOp.getLoc(), rewriter.getI1Type(), APInt(1, 1));
      rewriter.setInsertionPoint(cmpOp);
      cgra::BsfaOp bsfaOp = rewriter.create<cgra::BsfaOp>(
          cmpOp.getLoc(), rewriter.getI1Type(), selectFlag,
          SmallVector<Value>({constOp0.getResult(), constOp1.getResult()}));
      rewriter.replaceAllUsesWith(cmpOp.getResult(), bsfaOp.getResult());
    }
  }

  for (Operation *op : cmpOps)
    rewriter.eraseOp(op);
  for (Operation *op : selectOps)
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

// Get the stage of the block in SATMapIt DAG
enum class SATLoopBlock { Unkown, Init, Loop, Fini };
static SATLoopBlock SATMapItBlock(Block *block) {
  if (block->isEntryBlock()) {
    if (block->getSuccessors().size() == 1)
      return SATLoopBlock::Init;
    else
      return SATLoopBlock::Unkown;
  }

  if (isa<LLVM::ReturnOp>(block->getTerminator())) {
    if (getBlockPredecessorCount(block) == 1)
      return SATLoopBlock::Fini;
    else
      return SATLoopBlock::Unkown;
  }

  if (auto condBranchOp = dyn_cast<LLVM::CondBrOp>(block->getTerminator()))
    for (auto attr : condBranchOp.getAttributeNames())
      if (attr.contains(StringRef("loop_annotation")))
        return SATLoopBlock::Loop;

  return SATLoopBlock::Unkown;
}

LogicalResult
CgraLowering::createSATMapItDAG(ConversionPatternRewriter &rewriter) {

  // rewrite branchOp
  SmallVector<Block *> eraseBlocks;
  SmallVector<Operation *> eraseOps;

  Block &entryBlock = region.getBlocks().front();
  Operation *entryTerminator = entryBlock.getTerminator();
  if (isa<LLVM::ReturnOp>(entryTerminator))
    return success();

  for (auto &block : region) {
    Operation *termOp = block.getTerminator();

    // check whether one of the predecessors is unconditional branch
    if (SATMapItBlock(termOp->getBlock()) == SATLoopBlock::Unkown)
      return failure();

    if (SATMapItBlock(termOp->getBlock()) == SATLoopBlock::Init) {
      for (Operation &op : block) {
        op.setAttr("stage", rewriter.getStringAttr("init"));
      }
      eraseOps.push_back(termOp);
    }

    if (SATMapItBlock(termOp->getBlock()) == SATLoopBlock::Fini) {
      for (Operation &op : block) {
        op.setAttr("stage", rewriter.getStringAttr("fini"));
      }
      rewriter.moveOpAfter(termOp, entryBlock.getTerminator());
      eraseBlocks.push_back(&block);
    }
    if (SATMapItBlock(termOp->getBlock()) == SATLoopBlock::Loop) {
      std::vector<Operation *> opsToMove;
      for (auto &op : block.getOperations())
        if (&op != termOp)
          opsToMove.push_back(&op);
      for (auto op : opsToMove) {
        op->setAttr("stage", rewriter.getStringAttr("loop"));
        op->moveBefore(entryTerminator);
      }
      eraseBlocks.push_back(&block);
    }
  }

  for (Block *block : eraseBlocks)
    rewriter.eraseBlock(block);
  for (Operation *op : eraseOps)
    rewriter.eraseOp(op);
  return success();
}

/// Get the address of the integer variable with the given index.
static int getCgraIntVarAddress(MemoryInterface &memInterface, int idx) {
  int addr = memInterface.varHeadAddr + idx * 4;
  if (addr >= memInterface.varTailAddr)
    return -1;
  return addr;
}

/// Get the base address of the data array variable with the given index.
static int getCgraArrVarBaseAddress(MemoryInterface &memInterface, int idx) {
  if (idx == 0)
    return memInterface.arrHeadAddr;
  return memInterface.activeArrTail[idx - 1];
}

LogicalResult
CgraLowering::addMemoryInterface(ConversionPatternRewriter &rewriter) {
  // temporal memory interface hard coded
  MemoryInterface memInterface;
  memInterface.activeIntTail = 2;
  mlir::IntegerAttr base_addr;

  auto funcOp = region.getParentOfType<cgra::FuncOp>();
  DenseMap<Value, Operation *> arrayBaseAddrs;

  for (auto [argIndex, arg] : llvm::enumerate(funcOp.getArguments())) {
    // identify the data type of the argument
    rewriter.setInsertionPointToStart(&region.front());
    if (arg.getType().isIntOrIndex()) {
      // do nothing if the argument is not in use
      if (arg.use_empty())
        continue;
      // get the address of the integer variable
      int addr = getCgraIntVarAddress(memInterface, argIndex);
      LLVM::ConstantOp constOp = rewriter.create<LLVM::ConstantOp>(
          arg.getLoc(), rewriter.getI32Type(),
          rewriter.getIntegerAttr(rewriter.getI32Type(), addr));
      arrayBaseAddrs[arg] = constOp;
      arrayBaseAddrs[arg]->setAttr("arg" + std::to_string(argIndex),
                                   rewriter.getStringAttr("int"));
      // insert lwi operation to load the integer variable
      cgra::LwiOp lwiOp = rewriter.create<cgra::LwiOp>(
          arg.getLoc(), arg.getType(), constOp.getResult());
      // replace the argument with the result of the lwi operation
      arg.replaceAllUsesWith(lwiOp.getResult());
    } else if (arg.getType().isa<LLVM::LLVMPointerType>()) {
      // do nothing if the argument is not in use
      if (arg.use_empty())
        continue;
      // get the base address of the array variable
      int baseAddr = getCgraArrVarBaseAddress(
          memInterface, argIndex - memInterface.activeIntTail);
      // create a constant operation for the base address
      LLVM::ConstantOp constOp = rewriter.create<LLVM::ConstantOp>(
          arg.getLoc(), rewriter.getI32Type(),
          rewriter.getIntegerAttr(rewriter.getI32Type(), baseAddr));
      arrayBaseAddrs[arg] = constOp;
      arrayBaseAddrs[arg]->setAttr("arg" + std::to_string(argIndex),
                                   rewriter.getStringAttr("base"));
    }
  }

  SmallVector<Operation *> gepOps;
  SmallVector<Operation *> loadOps;

  for (LLVM::GEPOp op : region.getOps<LLVM::GEPOp>()) {
    // Validate the result is used by a load operation
    auto users = op->getResult(0).getUsers();
    if (std::distance(users.begin(), users.end()) != 1) {
      llvm::errs() << "LLVM::GEPOp is not used by one load/store operation\n";
      return failure();
    }

    auto sucOp = *users.begin();
    gepOps.push_back(op);
    loadOps.push_back(sucOp);

    rewriter.setInsertionPoint(getEntryBlockTerminator());
    // insert base address and offset
    Value loadArg = op.getOperand(0);
    auto constOp = arrayBaseAddrs[loadArg];
    LLVM::ConstantOp constOffset = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(), 4);
    rewriter.setInsertionPoint(op);
    LLVM::MulOp mulOp =
        rewriter.create<LLVM::MulOp>(op.getLoc(), rewriter.getI32Type(),
                                     constOffset.getResult(), op.getOperand(1));
    // calculate the address
    LLVM::AddOp addOp = rewriter.create<LLVM::AddOp>(
        op.getLoc(), constOp->getResult(0).getType(), constOp->getResult(0),
        mulOp.getResult());
    // insert load operation
    cgra::LwiOp loadOp = rewriter.create<cgra::LwiOp>(
        op.getLoc(), sucOp->getResult(0).getType(), addOp.getResult());

    // replace the llvm.load operation with the result of the cgra.lwi
    // operation
    sucOp->getResult(0).replaceAllUsesWith(loadOp.getResult());
  }

  for (Operation *op : gepOps)
    rewriter.eraseOp(op);
  for (Operation *op : loadOps)
    rewriter.eraseOp(op);
  return success();
}

LogicalResult
CgraLowering::removeUnusedOps(ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> eraseOps;
  for (auto &op : region.getOps()) {
    if (isa<cgra::BeqOp>(op) || isa<cgra::BneOp>(op) || isa<LLVM::ReturnOp>(op))
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
  if (failed(runPartialLowering(cl, &CgraLowering::addMemoryInterface)))
    return failure();

  if (failed(runPartialLowering(cl, &CgraLowering::addMergeOps)))
    return failure();

  if (failed(runPartialLowering(cl, &CgraLowering::replaceCmpOps)))
    return failure();

  if (failed(runPartialLowering(cl, &CgraLowering::createSATMapItDAG)))
    return failure();

  if (failed(runPartialLowering(cl, &CgraLowering::removeUnusedOps)))
    return failure();

  return success();
}

/// Lower a func::FuncOp to a cgra::FuncOp.
static LogicalResult lowerFuncOp(LLVM::LLVMFuncOp funcOp, MLIRContext *ctx) {

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
    auto noneType = rewriter.getNoneType();
    if (resTypes.empty())
      resTypes.push_back(noneType);
    if (!argTypes.empty())
      for (auto argAttr : funcOp.getAllArgAttrs())
        argAttrs.push_back(argAttr);
    argTypes.push_back(noneType);

    auto funcType = rewriter.getFunctionType(argTypes, resTypes);
    newFuncOp = rewriter.create<cgra::FuncOp>(funcOp.getLoc(), funcOp.getName(),
                                              funcType, attributes);

    auto &entryBlock = funcOp.getRegion().front();
    entryBlock.addArgument(rewriter.getNoneType(), newFuncOp.getLoc());

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (!newFuncOp.isExternal()) {
      argAttrs.push_back(rewriter.getDictionaryAttr(
          rewriter.getNamedAttr("cgra.None", rewriter.getUnitAttr())));
      newFuncOp.resolveArgAndResNames();
      newFuncOp.setArgAttrsAttr(ArrayAttr::get(ctx, argAttrs));
    }

    return success();
  };

  if (failed(partiallyLowerOp(funcLowering, funcOp)))
    return failure();

  funcOp.erase();
  if (!newFuncOp.isExternal()) {
    CgraLowering cl(newFuncOp.getBody());
    return lowerRegion(cl);
  }
  return success();
};

static bool isBackEdge(Value edge, Operation *dstOp) {
  // If the destination operation is not used by any operation, it is not a
  // backedge
  if (dstOp->getResults().use_empty())
    return false;
  Operation *srcOp = edge.getDefiningOp();

  // check whether a propagation path exists from dstOp to srcOp, if yes, it
  // is a backedge.
  SmallVector<Operation *> opVec = {dstOp};
  SmallVector<Operation *> visited = {dstOp};
  while (!opVec.empty()) {
    Operation *op = opVec.back();
    opVec.pop_back();
    if (op == srcOp)
      return true;
    for (Value operand : op->getOperands())
      for (Operation *sucOp : operand.getUsers()) {
        if (llvm::is_contained(visited, sucOp))
          continue;
        visited.push_back(sucOp);
        opVec.push_back(sucOp);
      }
  }
  return false;
}

void LLVMToCgraConversionPass::runOnOperation() {
  ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());

  // rewrite funcOp
  for (auto funcOp :
       llvm::make_early_inc_range(modOp.getOps<LLVM::LLVMFuncOp>())) {
    if (failed(lowerFuncOp(funcOp, &getContext())))
      return signalPassFailure();
  }

  for (auto funcOp : llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>()))
    if (outputDAG == funcOp.getName() && failed(outputDATE2023DAG(funcOp)))
      return signalPassFailure();
};

LogicalResult LLVMToCgraConversionPass::outputDATE2023DAG(cgra::FuncOp funcOp) {
  SmallVector<Operation *> nodes;
  // define the edge by the srcOp and dstOp
  using Edge = std::pair<Operation *, Operation *>;

  std::vector<Edge> edges;
  for (Operation &op : funcOp.getOps()) {
    StringRef stage = dyn_cast<StringAttr>(op.getAttr("stage")).getValue();
    if (stage != StringRef("loop"))
      continue;

    nodes.push_back(&op);

  }
  return success();
}

namespace compigra {
std::unique_ptr<mlir::Pass> createLLVMToCgraConversion(StringRef outputDAG) {
  return std::make_unique<LLVMToCgraConversionPass>(outputDAG);
}
} // namespace compigra