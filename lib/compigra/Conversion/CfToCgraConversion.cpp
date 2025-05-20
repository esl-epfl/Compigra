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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

/// Remove unused operations
static LogicalResult removeUnusedOps(func::FuncOp funcOp) {
  SmallVector<Operation *> eraseOps;
  for (auto &op : funcOp.getOps()) {
    if (op.getBlock()->getTerminator() == &op || isa<cgra::SwiOp>(op) ||
        isa<cgra::BlasGemmOp>(op))
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
    op->erase();

  return success();
}

static LogicalResult raiseConstOpToTop(func::FuncOp funcOp) {
  for (auto cstOp :
       llvm::make_early_inc_range(funcOp.getOps<arith::ConstantOp>())) {
    cstOp->moveBefore(&funcOp.getBlocks().front().front());
  }
  return success();
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

/// Lower arith::SelectOp to cgra::BzfaOp
struct ArithSelectOpConversion : OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp selectOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto trueVal = selectOp.getTrueValue();
    auto falseVal = selectOp.getFalseValue();
    auto cond = selectOp.getCondition();

    // replace arith.select with cgra.bzfa
    rewriter.replaceOpWithNewOp<cgra::BzfaOp>(
        selectOp, selectOp->getResult(0).getType(), cond,
        SmallVector<Value>({falseVal, trueVal}));

    return success();
  }
};

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

    bool switchSuccs = condBrBlock == condBrOp.getFalseDest();
    if (switchSuccs)
      predicate = reverseCmpFlag(predicate);

    auto condBrArgs = switchSuccs ? condBrOp.getFalseDestOperands()
                                  : condBrOp.getTrueDestOperands();
    auto jumpArgs = switchSuccs ? condBrOp.getTrueDestOperands()
                                : condBrOp.getFalseDestOperands();
    Block *jumpBlock =
        switchSuccs ? condBrOp.getTrueDest() : condBrOp.getFalseDest();

    // replace cf.condbr with cgra.condbr
    cgra::CondBrPredicate cgraPred =
        getCgraBrPredicate(predicate, cmpOpr0, cmpOpr1);
    if (forceJump) {
      auto newCondBr = rewriter.replaceOpWithNewOp<cgra::ConditionalBranchOp>(
          condBrOp, cgraPred, cmpOpr0, cmpOpr1, condBrBlock, condBrArgs,
          falseBlk, SmallVector<Value>());
      rewriter.setInsertionPointToStart(falseBlk);
      auto defaultBr = rewriter.create<cf::BranchOp>(newCondBr->getLoc(),
                                                     jumpArgs, jumpBlock);
      defaultBr->moveAfter(&falseBlk->getOperations().front());
    } else {
      rewriter.replaceOpWithNewOp<cgra::ConditionalBranchOp>(
          condBrOp, cgraPred, cmpOpr0, cmpOpr1, condBrBlock, condBrArgs,
          falseBlk, jumpArgs);
    }

    return success();
  }
};

/// Rewrite arith::CmpIOp to corresponding operation in cgra dialect
struct ArithCmpIOpConversion : OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp cmpOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto predicate = cmpOp.getPredicate();

    // Use substraction to compare the two operands
    rewriter.setInsertionPoint(cmpOp);
    arith::SubIOp subOp = nullptr;
    // Reverse the operands order for greater than or equal to and greater
    if (predicate == arith::CmpIPredicate::uge ||
        predicate == arith::CmpIPredicate::sge ||
        predicate == arith::CmpIPredicate::ugt ||
        predicate == arith::CmpIPredicate::sgt) {
      subOp = rewriter.create<arith::SubIOp>(
          cmpOp.getLoc(), cmpOp.getOperand(1), cmpOp.getOperand(0));
    } else
      subOp = rewriter.create<arith::SubIOp>(
          cmpOp.getLoc(), cmpOp.getOperand(0), cmpOp.getOperand(1));

    // insert additional bzfa operation to conclude equal case of the
    // comparison.
    auto selectFlag = subOp.getResult();
    if (predicate == arith::CmpIPredicate::uge ||
        predicate == arith::CmpIPredicate::uge ||
        predicate == arith::CmpIPredicate::ule ||
        predicate == arith::CmpIPredicate::ule) {
      // create constant -1 to indicate the equal case
      auto resType = subOp.getResult().getType();
      arith::ConstantIntOp constOp = rewriter.create<arith::ConstantIntOp>(
          cmpOp.getLoc(), -1, resType.getIntOrFloatBitWidth());

      // create bzfa operation to include the equal case
      cgra::BzfaOp bzfaOp = rewriter.create<cgra::BzfaOp>(
          cmpOp.getLoc(), subOp.getResult().getType(), subOp.getResult(),
          SmallVector<Value>({constOp.getResult(), subOp.getResult()}));
      selectFlag = bzfaOp.getResult();
    }

    // Replace the select operation with bsfa/bzfa operation.
    SmallVector<Operation *> selOps;
    for (auto user : cmpOp->getUsers()) {
      if (auto selOp = dyn_cast_or_null<arith::SelectOp>(user))
        selOps.push_back(selOp);
    }
    for (auto selOp : selOps) {
      rewriter.setInsertionPoint(selOp);
      if (predicate == arith::CmpIPredicate::eq) {
        rewriter.replaceOpWithNewOp<cgra::BzfaOp>(
            selOp, selOp->getResult(0).getType(), selectFlag,
            SmallVector<Value>({selOp->getOperand(1), selOp->getOperand(2)}));
      } else if (predicate == arith::CmpIPredicate::ne) {
        rewriter.replaceOpWithNewOp<cgra::BzfaOp>(
            selOp, selOp->getResult(0).getType(), selectFlag,
            SmallVector<Value>({selOp->getOperand(2), selOp->getOperand(1)}));
      } else {
        rewriter.replaceOpWithNewOp<cgra::BsfaOp>(
            selOp, selOp->getResult(0).getType(), selectFlag,
            SmallVector<Value>({selOp->getOperand(1), selOp->getOperand(2)}));
      }
    }

    // check whether the cmpOp has non-select and non-branch users
    bool existNonSelAndBrUser = false;
    for (auto user : cmpOp->getUsers()) {
      if (!isa<cf::CondBranchOp, arith::SelectOp>(user)) {
        existNonSelAndBrUser = true;
        break;
      }
    }
    if (existNonSelAndBrUser) {
      // insert additional bsfa operation to conclude the comparison
      arith::ConstantIntOp constOp0 =
          rewriter.create<arith::ConstantIntOp>(cmpOp.getLoc(), 0, 1);
      arith::ConstantIntOp constOp1 =
          rewriter.create<arith::ConstantIntOp>(cmpOp.getLoc(), 1, 1);
      rewriter.setInsertionPoint(cmpOp);
      Operation *binSelOp = nullptr;
      if (predicate == arith::CmpIPredicate::eq) {
        // insert bzfa %selFlag, 1, 0;
        binSelOp = rewriter.create<cgra::BzfaOp>(
            cmpOp.getLoc(), rewriter.getI1Type(), selectFlag,
            SmallVector<Value>({constOp1.getResult(), constOp0.getResult()}));
      } else if (predicate == arith::CmpIPredicate::ne) {
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
      rewriter.replaceOp(cmpOp, binSelOp->getResult(0));
    } else {
      rewriter.eraseOp(cmpOp);
    }
    return success();
  }
};

template <typename MemRefOp>
Operation *computeOffSet(MemRefOp memOp, Operation *baseAddr,
                         SmallVector<Operation *> strideVals,
                         ConversionPatternRewriter &rewriter) {
  Operation *offSet = nullptr;

  for (auto [dim, indice] : llvm::enumerate(memOp.getIndices())) {
    // if indice is a constant 0, skip
    if (auto constantOp =
            dyn_cast_or_null<arith::ConstantIndexOp>(indice.getDefiningOp())) {
      if (constantOp.value() == 0) {
        continue;
      }
    }

    auto castOp = rewriter.create<arith::IndexCastOp>(
        memOp.getLoc(), rewriter.getIntegerType(32), indice);
    if (dim == memOp.getIndices().size() - 1) {
      if (offSet)
        offSet = rewriter.create<arith::AddIOp>(
            memOp.getLoc(), rewriter.getI32Type(), offSet->getResult(0),
            castOp.getResult());
      else
        offSet = castOp;
      break;
    }

    Value stride = strideVals[dim]->getResult(0);
    auto dimStride = rewriter.create<arith::MulIOp>(
        memOp.getLoc(), rewriter.getI32Type(), castOp.getResult(), stride);
    if (offSet)
      offSet = rewriter.create<arith::AddIOp>(
          memOp.getLoc(), rewriter.getI32Type(), offSet->getResult(0),
          dimStride->getResult(0));
    else
      offSet = dimStride;
  }

  return offSet;
}

static int preComputeOffset(Operation *offSetOp, Operation *byteOp) {

  if (arith::IndexCastOp cstOffset =
          dyn_cast_or_null<arith::IndexCastOp>(offSetOp)) {
    // directly compute the offset
    auto srcOffset = dyn_cast_or_null<arith::ConstantIndexOp>(
        cstOffset.getOperand().getDefiningOp());
    if (!srcOffset)
      return -1;

    long byteWidth = 4;
    if (auto cstByte = dyn_cast_or_null<arith::ConstantIntOp>((byteOp)))
      byteWidth = cstByte.value();
    return byteWidth * srcOffset.value();
  }
  return -1;
}

// Rewrite memref.get_global to constant operation
struct MemRefGetGlobalOpConversion : OpConversionPattern<memref::GetGlobalOp> {
  MemRefGetGlobalOpConversion(MLIRContext *ctx,
                              std::map<llvm::StringRef, Operation *> baseAddrs)
      : OpConversionPattern<memref::GetGlobalOp>(ctx), baseAddrs(baseAddrs) {}

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp getGlobalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto varName = getGlobalOp.getName();
    rewriter.replaceOp(getGlobalOp, baseAddrs.at(varName));
    return success();
  }

  std::map<llvm::StringRef, Operation *> baseAddrs;
};

/// Lower memref.load/ memref.store to cgra.lwi/cgra.swi
template <typename MemRefOp>
struct MemRefRWOpConversion : OpConversionPattern<MemRefOp> {
  MemRefRWOpConversion(
      MLIRContext *ctx, SmallVector<Operation *> &baseAddrs,
      std::map<llvm::StringRef, Operation *> &globalConstAddrs,
      DenseMap<Operation *, SmallVector<Operation *>> &strideValMap)
      : OpConversionPattern<MemRefOp>(ctx), baseAddrs(baseAddrs),
        globalConstAddrs(globalConstAddrs), strideValMap(strideValMap) {}

  LogicalResult
  matchAndRewrite(MemRefOp op, typename MemRefOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *byteOp = baseAddrs.back();
    Location loc = op.getLoc();

    // compute the base address of op
    Value ref = op.getMemRef();
    Operation *baseOp;
    if (auto arg = dyn_cast_or_null<BlockArgument>(ref)) {
      unsigned argIndex = arg.getArgNumber();
      baseOp = baseAddrs[argIndex];
    } else if (isa<memref::GetGlobalOp>(ref.getDefiningOp())) {
      auto getOp = dyn_cast<memref::GetGlobalOp>(ref.getDefiningOp());
      baseOp = globalConstAddrs.at(getOp.getName());
    } else {
      return failure();
    }

    // get the index of the memory reference
    rewriter.setInsertionPoint(op);
    auto offSetOp =
        computeOffSet<MemRefOp>(op, baseOp, strideValMap.at(baseOp), rewriter);

    Operation *addrOp = nullptr;
    if (offSetOp) {
      Operation *byteOffset;
      int byteWidth = preComputeOffset(offSetOp, byteOp);
      if (byteWidth >= 0) {
        byteOffset = rewriter.create<arith::ConstantIntOp>(loc, byteWidth, 32);
      } else {
        byteOffset = rewriter.create<arith::MulIOp>(loc, rewriter.getI32Type(),
                                                    offSetOp->getResult(0),
                                                    byteOp->getResult(0));
      }
      addrOp = rewriter.create<arith::AddIOp>(loc, rewriter.getI32Type(),
                                              baseOp->getResult(0),
                                              byteOffset->getResult(0));
    } else {
      addrOp = baseOp;
    }

    if constexpr (std::is_same_v<MemRefOp, memref::LoadOp>) {
      rewriter.replaceOpWithNewOp<cgra::LwiOp>(op, op.getResult().getType(),
                                               addrOp->getResult(0));
    } else if constexpr (std::is_same_v<MemRefOp, memref::StoreOp>) {
      rewriter.replaceOpWithNewOp<cgra::SwiOp>(op, op.getValue(),
                                               addrOp->getResult(0));
    }

    return success();
  }

  SmallVector<Operation *> baseAddrs;
  std::map<llvm::StringRef, Operation *> globalConstAddrs;
  DenseMap<Operation *, SmallVector<Operation *>> strideValMap;
};

} // namespace

static LogicalResult
assignMemoryToArg(mlir::Type typeAttr, unsigned &lastPtr,
                  std::vector<int> &memAlloc,
                  std::vector<std::vector<int>> &memRefDims) {
  if (!typeAttr.isa<MemRefType>())
    return failure();
  auto memrefType = typeAttr.cast<MemRefType>();
  memRefDims.push_back(std::vector<int>());
  // arg.getOperation()

  // Allocate memory based on the default size
  memAlloc.push_back(lastPtr);
  int memRefSize = 1;
  for (int i = memrefType.getRank() - 1; i >= 0; i--) {
    memRefSize *= memrefType.getDimSize(i);
    memRefDims.back().insert(memRefDims.back().begin(), memRefSize);
  }
  lastPtr += memRefSize * 4;

  return success();
}

LogicalResult
allocateMemory(ModuleOp &modOp, SmallVector<Operation *> &constAddr,
               std::map<llvm::StringRef, Operation *> &globalConstAddrs,
               DenseMap<Operation *, SmallVector<Operation *>> &offValMap,
               OpBuilder &builder, Pass::ListOption<int> &startAddr) {
  if (modOp.getOps<func::FuncOp>().empty())
    return success();

  auto funcOp = *modOp.getOps<func::FuncOp>().begin();
  std::vector<int> memAlloc;
  std::vector<std::vector<int>> memRefDims;

  unsigned lastPtr = 128;
  if (!startAddr.empty()) {
    lastPtr = startAddr[0];
  }
  std::map<int, memref::GlobalOp> globalArgs;
  // assign memory for global arguments
  for (auto [ind, arg] : llvm::enumerate(modOp.getOps<memref::GlobalOp>())) {
    globalArgs[ind] = arg;
    assignMemoryToArg(arg.getType(), lastPtr, memAlloc, memRefDims);
  }

  // assign memory for function arguments
  for (auto [ind, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (!arg.getType().isa<MemRefType>())
      return failure();
    auto memrefType = arg.getType().cast<MemRefType>();
    memRefDims.push_back(std::vector<int>());

    if (memrefType.getDimSize(0) > 0) {
      // Allocate memory based on the default size
      memAlloc.push_back(lastPtr);
      int memRefSize = 1;
      for (int i = memrefType.getRank() - 1; i >= 0; i--) {
        memRefSize *= memrefType.getDimSize(i);
        memRefDims[ind].insert(memRefDims[ind].begin(), memRefSize);
      }
      lastPtr += memRefSize * 4;
    } else {
      // Fail if the size is not specified
      return failure();
    }
  }

  // insert constant operation to initialize the offset and base address
  builder.setInsertionPointToStart(&funcOp.getBlocks().front());
  // print the memory allocation
  for (unsigned i = 0; i < memAlloc.size(); i++) {
    auto baseOp = builder.create<arith::ConstantIntOp>(
        funcOp.getLoc(), memAlloc[i], builder.getI32Type());
    std::string prefix = i < globalArgs.size() ? "global" : "arg";
    unsigned argInd = i < globalArgs.size() ? i : i - globalArgs.size();

    baseOp->setAttr("BaseAddr",
                    builder.getStringAttr(prefix + std::to_string(argInd)));

    SmallVector<Operation *> dimOps;
    for (unsigned j = 1; j < memRefDims[i].size(); j++) {
      auto dimOp = builder.create<arith::ConstantIntOp>(
          funcOp.getLoc(), memRefDims[i][j], builder.getI32Type());
      dimOp->setAttr(prefix,
                     builder.getIntegerAttr(builder.getI32Type(), argInd));
      dimOp->setAttr("DimProd",
                     builder.getIntegerAttr(builder.getI32Type(), j));
      dimOps.push_back(dimOp);
    }
    offValMap[baseOp] = dimOps;
    if (i < globalArgs.size()) {
      globalConstAddrs[globalArgs[i].getName()] = baseOp;
    } else {
      constAddr.push_back(baseOp);
    }
  }

  // create a constant operation to initialize the offset
  auto offset = builder.create<arith::ConstantIntOp>(funcOp.getLoc(), 4,
                                                     builder.getI32Type());
  constAddr.push_back(offset);

  return success();
}

void compigra::populateCfToCgraConversionPatterns(
    RewritePatternSet &patterns, SmallVector<Operation *> &constAddr,
    std::map<llvm::StringRef, Operation *> &globalConstAddrs,
    DenseMap<Operation *, SmallVector<Operation *>> &offValMap) {
  patterns.add<CfCondBrOpConversion>(patterns.getContext());
  patterns.add<ArithCmpIOpConversion>(patterns.getContext());
  patterns.add<ArithSelectOpConversion>(patterns.getContext());
  patterns.add<MemRefRWOpConversion<memref::LoadOp>,
               MemRefRWOpConversion<memref::StoreOp>>(
      patterns.getContext(), constAddr, globalConstAddrs, offValMap);
  patterns.add<MemRefGetGlobalOpConversion>(patterns.getContext(),
                                            globalConstAddrs);
}

void CfToCgraConversionPass::runOnOperation() {
  ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
  OpBuilder builder(modOp);
  std::map<llvm::StringRef, Operation *> globalConstAddrs;
  SmallVector<Operation *> constAddrs;

  // Map to store the stride information for each memory reference, where
  // the key is the constant value of the base address.
  // For a multidimensional array (e.g., c[M][N][K]), the stored value
  // is a vector of products of the inner dimensions (e.g., [N*K, K]).
  DenseMap<Operation *, SmallVector<Operation *>> offValMap;
  if (failed(allocateMemory(modOp, constAddrs, globalConstAddrs, offValMap,
                            builder, startAddr)))
    signalPassFailure();

  ConversionTarget target(getContext());

  target.addIllegalOp<arith::CmpIOp>();
  target.addIllegalOp<memref::LoadOp>();
  target.addIllegalOp<cf::CondBranchOp>();
  target.addIllegalOp<memref::StoreOp>();
  target.addIllegalOp<memref::GetGlobalOp>();
  target.addIllegalOp<arith::SelectOp>();

  target.addLegalOp<cgra::ConditionalBranchOp>();
  target.addLegalOp<cgra::LwiOp>();
  target.addLegalOp<cgra::SwiOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  RewritePatternSet patterns(&getContext());
  populateCfToCgraConversionPatterns(patterns, constAddrs, globalConstAddrs,
                                     offValMap);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

  // raise the constant operation to the top level
  auto funcOps = modOp.getOps<func::FuncOp>();
  if (!funcOps.empty()) {
    auto funcOp = *funcOps.begin();
    if (failed(removeUnusedOps(funcOp)) || failed(raiseConstOpToTop(funcOp)))
      signalPassFailure();
  }
}

namespace compigra {
std::unique_ptr<mlir::Pass> createCfToCgraConversion() {
  return std::make_unique<CfToCgraConversionPass>();
}
} // namespace compigra