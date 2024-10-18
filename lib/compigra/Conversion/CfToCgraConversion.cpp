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
      // rewriter.setInsertionPoint(condBrOp);
      auto newCondBr = rewriter.replaceOpWithNewOp<cgra::ConditionalBranchOp>(
          condBrOp, cgraPred, cmpOpr0, cmpOpr1, condBrBlock, condBrArgs,
          falseBlk, SmallVector<Value>());
      rewriter.setInsertionPointToStart(falseBlk);
      auto defaultBr = rewriter.create<cf::BranchOp>(newCondBr->getLoc(),
                                                     jumpArgs, jumpBlock);
      defaultBr->moveAfter(&falseBlk->getOperations().front());
    } else {
      // rewriter.setInsertionPoint(condBrOp);
      rewriter.replaceOpWithNewOp<cgra::ConditionalBranchOp>(
          condBrOp, cgraPred, cmpOpr0, cmpOpr1, condBrBlock, condBrArgs,
          falseBlk, jumpArgs);
    }

    if (cmpOp->hasOneUse()) {
      rewriter.eraseOp(cmpOp);
    }

    return success();
  }
};

template <typename MemRefOp>
Operation *computeOffSet(MemRefOp memOp, Operation *baseAddr,
                         SmallVector<Operation *> strideVals,
                         Operation *eleStride,
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

    // llvm::errs() << "     come to dim: " << dim << "\n";
    auto castOp = rewriter.create<arith::IndexCastOp>(
        memOp.getLoc(), rewriter.getIntegerType(32), indice);
    if (indice == memOp.getIndices().back()) {
      if (offSet)
        offSet = rewriter.create<arith::AddIOp>(
            memOp.getLoc(), rewriter.getI32Type(), offSet->getResult(0),
            castOp.getResult());
      else
        offSet = castOp;
      break;
    }

    // llvm::errs() << "    stride: " << strideVals.size() << "\n";
    Value stride = strideVals[dim]->getResult(0);
    // llvm::errs() << "    compute stride: \n";
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

/// Lower memref.load/ memref.store to cgra.lwi/cgra.swi
template <typename MemRefOp>
struct CfMemRefOpConversion : OpConversionPattern<MemRefOp> {
  CfMemRefOpConversion(
      MLIRContext *ctx, SmallVector<Operation *> &baseAddrs,
      DenseMap<Operation *, SmallVector<Operation *>> &strideValMap)
      : OpConversionPattern<MemRefOp>(ctx), baseAddrs(baseAddrs),
        strideValMap(strideValMap) {}

  LogicalResult
  matchAndRewrite(MemRefOp op, typename MemRefOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *byteOp = baseAddrs.back();
    Location loc = op.getLoc();

    // compute the base address of op
    Value ref = op.getMemRef();
    auto arg = dyn_cast<BlockArgument>(ref);
    if (!arg)
      return failure();
    unsigned argIndex = arg.getArgNumber();
    Operation *baseOp = baseAddrs[argIndex];
    // llvm::errs() << op << "\n";
    // llvm::errs() << "     ref: " << ref << "\n";

    // get the index of the memory reference
    rewriter.setInsertionPoint(op);
    auto offSetOp = computeOffSet<MemRefOp>(op, baseOp, strideValMap.at(baseOp),
                                            byteOp, rewriter);

    Operation *addrOp = nullptr;
    if (offSetOp) {
      Operation *byteOffset = rewriter.create<arith::MulIOp>(
          loc, rewriter.getI32Type(), offSetOp->getResult(0),
          byteOp->getResult(0));
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
  DenseMap<Operation *, SmallVector<Operation *>> strideValMap;
};

} // namespace

LogicalResult
allocateMemory(ModuleOp &modOp, SmallVector<Operation *> &constAddr,
               DenseMap<Operation *, SmallVector<Operation *>> &offValMap,
               OpBuilder &builder, Pass::ListOption<int> &startAddr) {
  if (modOp.getOps<func::FuncOp>().empty())
    return success();

  auto funcOp = *modOp.getOps<func::FuncOp>().begin();
  std::vector<int> memAlloc;
  std::vector<std::vector<int>> memRefDims;

  int lastPtr = 0;
  for (auto [ind, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (!arg.getType().isa<MemRefType>())
      return failure();
    auto memrefType = arg.getType().cast<MemRefType>();
    memRefDims.push_back(std::vector<int>());

    // Priorize the startAddr from the command line
    if (ind < startAddr.size()) {
      if (startAddr[ind] < lastPtr || memrefType.getRank() > 1)
        return failure();

      memAlloc.push_back(startAddr[ind]);
      lastPtr = startAddr[ind];
      continue;
    }

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
    baseOp->setAttr("BaseAddr",
                    builder.getStringAttr("arg" + std::to_string(i)));
    constAddr.push_back(baseOp);
    SmallVector<Operation *> dimOps;
    for (unsigned j = 1; j < memRefDims[i].size(); j++) {
      auto dimOp = builder.create<arith::ConstantIntOp>(
          funcOp.getLoc(), memRefDims[i][j], builder.getI32Type());
      dimOp->setAttr("Arg", builder.getIntegerAttr(builder.getI32Type(), i));
      dimOp->setAttr("DimProd",
                     builder.getIntegerAttr(builder.getI32Type(), j));
      dimOps.push_back(dimOp);
    }
    offValMap[baseOp] = dimOps;
  }
  // create a constant operation to initialize the offset
  auto offset = builder.create<arith::ConstantIntOp>(funcOp.getLoc(), 4,
                                                     builder.getI32Type());
  constAddr.push_back(offset);

  return success();
}

void compigra::populateCfToCgraConversionPatterns(
    RewritePatternSet &patterns, SmallVector<Operation *> &constAddr,
    DenseMap<Operation *, SmallVector<Operation *>> &offValMap) {
  patterns.add<CfCondBrOpConversion>(patterns.getContext());
  patterns.add<CfMemRefOpConversion<memref::LoadOp>,
               CfMemRefOpConversion<memref::StoreOp>>(patterns.getContext(),
                                                      constAddr, offValMap);
}

void CfToCgraConversionPass::runOnOperation() {
  ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
  OpBuilder builder(modOp);
  SmallVector<Operation *> constAddrs;

  // Map to store the stride information for each memory reference, where
  // the key is the constant value of the base address.
  // For a multidimensional array (e.g., c[M][N][K]), the stored value
  // is a vector of products of the inner dimensions (e.g., [N*K, K]).
  DenseMap<Operation *, SmallVector<Operation *>> offValMap;
  if (failed(allocateMemory(modOp, constAddrs, offValMap, builder, startAddr)))
    signalPassFailure();

  llvm::errs() << "Memory allocation done\n";
  ConversionTarget target(getContext());
  target.addIllegalOp<cf::CondBranchOp>();
  target.addLegalOp<cgra::ConditionalBranchOp>();
  target.addIllegalOp<memref::LoadOp>();
  target.addLegalOp<cgra::LwiOp>();
  target.addIllegalOp<memref::StoreOp>();
  target.addLegalOp<cgra::SwiOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  RewritePatternSet patterns(&getContext());
  populateCfToCgraConversionPatterns(patterns, constAddrs, offValMap);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
  // llvm::errs() << modOp << "\n";
}

namespace compigra {
std::unique_ptr<mlir::Pass> createCfToCgraConversion() {
  return std::make_unique<CfToCgraConversionPass>();
}
} // namespace compigra