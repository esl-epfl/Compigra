//===- CgraFitToOpenEdge.cpp -Rewrite ops to fit in HW ISA -*- C++ -----*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --fit-openedge pass, which rewrite the operations to fit into
// the openedge CGRA. For example, beq does not support immediate values, so the
// constant value must be loaded into an opeartion.
//
//===----------------------------------------------------------------------===//

#include "compigra/Transforms/CgraFitToOpenEdge.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

// Debugging support
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace compigra;

static bool isValidDataType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    if (intType.getWidth() == 32) {
      return true;
    }
  if (auto decType = dyn_cast<Float32Type>(type))
    return true;

  return false;
}

namespace {
/// Driver for the fit-openedge pass.
struct CgraFitToOpenEdgePass
    : public compigra::impl::CgraFitToOpenEdgeBase<CgraFitToOpenEdgePass> {

  explicit CgraFitToOpenEdgePass(StringRef outputDAG) {}
  void runOnOperation() override;
};

/// Rewrite constant operation to make all the immediate field in hardware ISA
/// valid.
struct ConstantOpRewrite : public OpRewritePattern<arith::ConstantOp> {
  ConstantOpRewrite(MLIRContext *ctx, SmallVector<Operation *> &insertedOps,
                    Operation *frontOp)
      : OpRewritePattern(ctx), insertedOps(insertedOps), frontOp(frontOp) {}

  LogicalResult matchAndRewrite(arith::ConstantOp constOp,
                                PatternRewriter &rewriter) const override;

private:
  SmallVector<Operation *> &insertedOps;
  Operation *frontOp;
};

/// Rewrite cgra::cond_br operation to make it branch to phi node with differnt
/// operands source.
struct CondBrOpRewrite : public OpRewritePattern<cgra::ConditionalBranchOp> {
  CondBrOpRewrite(MLIRContext *ctx, Operation *zeroOp)
      : OpRewritePattern(ctx), zeroOp(zeroOp) {}

  LogicalResult matchAndRewrite(cgra::ConditionalBranchOp condBrOp,
                                PatternRewriter &rewriter) const override;

private:
  Operation *zeroOp;
};

/// Rewrite cf::br operation to make it branch to phi node with differnt
/// operands source.
struct BranchOpRewrite : public OpRewritePattern<cf::BranchOp> {
  BranchOpRewrite(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(cf::BranchOp brOp,
                                PatternRewriter &rewriter) const override;

private:
  Operation *zeroOp;
};

/// Fix the load operators to be I32, as the Load&Store interface is fixed in
/// OpenEdge.
struct LwiOpRewrite : public OpRewritePattern<cgra::LwiOp> {
  LwiOpRewrite(MLIRContext *ctx) : OpRewritePattern(ctx) {}
  LogicalResult matchAndRewrite(cgra::LwiOp lwiOp,
                                PatternRewriter &rewriter) const override;
};

struct SAddOpRewrite : public OpRewritePattern<arith::AddIOp> {
  SAddOpRewrite(MLIRContext *ctx) : OpRewritePattern(ctx) {}
  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rewriter) const override;
};

OpenEdgeISATarget::OpenEdgeISATarget(MLIRContext *ctx)
    : ConversionTarget(*ctx) {
  addLegalDialect<cgra::CgraDialect>();
  addDynamicallyLegalDialect<arith::ArithDialect>(
      [&](Operation *op) { return !isa<arith::ConstantOp>(op); });

  // add the operation to the target
  addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp constOp) {
    if (!dyn_cast_or_null<IntegerAttr>(constOp->getAttr("value")))
      return true;

    int value = constOp.getValue().cast<IntegerAttr>().getInt();
    bool isAddr = isAddrConstOp(constOp);
    bool validAddr = isValidImmAddr(constOp);
    if (isAddr && !validAddr)
      return false;

    // if the value exceed the Imm range
    if (!isAddr || (isAddr && !constOp->hasOneUse()))
      if (value < -4097 || value > 4096)
        return false;

    for (auto &use : constOp->getUses()) {
      auto user = use.getOwner();
      if (isa<cf::BranchOp>(user))
        // The branch operation cannot use immediate values (Imm) for
        // computation, nor can it propagate constants.
        return false;
      if (isa<cgra::ConditionalBranchOp>(user)) {
        if (value != 0 || use.getOperandNumber() != 1)
          return false;
      }

      // if the constant is used by swi for imm store, it is not legal
      if (isa<cgra::SwiOp>(user) &&
          user->getOperand(0).getDefiningOp() == constOp)
        return false;
    }

    return true;
  });

  addDynamicallyLegalOp<cgra::LwiOp>([&](cgra::LwiOp lwiOp) {
    return isValidDataType(lwiOp.getResult().getType());
  });

  addDynamicallyLegalOp<arith::AddIOp>([&](arith::AddIOp addOp) {
    auto opA = addOp.getOperand(0).getDefiningOp();
    auto opB = addOp.getOperand(1).getDefiningOp();
    // if one of the operands is argument, it is legal
    if (!opA || !opB)
      return true;

    if (isa<arith::ConstantOp>(opA) && isa<arith::ConstantOp>(opB)) {
      auto valueA = opA->getAttr("value").cast<IntegerAttr>().getInt();
      auto valueB = opB->getAttr("value").cast<IntegerAttr>().getInt();
      if (valueA == 0 || valueB == 0)
        return true;
      if (opA->getAttr("value") != opB->getAttr("value"))
        return false;
    }
    return true;
  });

  addDynamicallyLegalOp<cgra::ConditionalBranchOp>([&](Operation *op) {
    DenseSet<Value> sources;
    for (auto &opr : op->getOpOperands()) {
      if (opr.getOperandNumber() < 2)
        continue;
      if (!sources.insert(opr.get()).second) {
        return false;
      }
    }
    return true;
  });

  addDynamicallyLegalOp<cf::BranchOp>([&](Operation *op) {
    DenseSet<Value> sources;
    for (auto &opr : op->getOpOperands()) {
      if (!sources.insert(opr.get()).second) {
        return false;
      }
    }
    return true;
  });
}
} // namespace

bool isValidImmAddr(arith::ConstantOp constOp) {
  // Address can not exceeds 13 bits
  int maxAddr = 0b1111111111111;
  auto value = constOp.getValue().cast<IntegerAttr>().getInt();
  return value <= maxAddr;
};

Operation *existsConstant(int intVal, SmallVector<Operation *> &insertedOps) {
  for (auto &op : insertedOps) {
    if (!op->getAttrDictionary().contains("constant"))
      continue;
    auto produceVal = op->getAttr("constant").dyn_cast<IntegerAttr>().getInt();
    if (produceVal == intVal) {
      return op;
    }
  }
  return nullptr;
}

bool isAddrConstOp(arith::ConstantOp constOp) {
  for (auto &use : constOp->getUses()) {
    if (isa<cgra::LwiOp>(use.getOwner()))
      return true;
    // Only address can be used as Imm field of swi operation
    if (use.getOperandNumber() == 1 && isa<cgra::SwiOp>(use.getOwner()))
      return true;
  }

  return false;
}

template <typename FuncOp> LogicalResult raiseConstOperation(FuncOp funcOp) {
  Operation &beginOp = *funcOp.getOps().begin();
  // raise the constant operation to the top level
  auto constantOps = funcOp.template getOps<arith::ConstantIntOp>();
  for (auto op : llvm::make_early_inc_range(constantOps))
    if (op->getBlock()->isEntryBlock())
      op->moveBefore(&beginOp);

  return success();
}

/// remove the constant operation if it is not used
template <typename FuncOp> LogicalResult removeUnusedConstOp(FuncOp funcOp) {
  auto constantOps = funcOp.template getOps<arith::ConstantIntOp>();
  for (auto op : llvm::make_early_inc_range(constantOps)) {
    if (op->use_empty())
      op->erase();
  }

  return success();
}

LogicalResult removeEqualWidthBWOp(cgra::FuncOp funcOp) {
  auto sextOps = funcOp.template getOps<LLVM::SExtOp>();
  for (auto op : llvm::make_early_inc_range(sextOps)) {
    if (op.getOperand().getType() == op.getResult().getType()) {
      if (op.getOperand().getDefiningOp())
        op.replaceAllUsesWith(op.getOperand().getDefiningOp());
      op->erase();
    }
  }

  return success();
}

static bool isLoopBlock(Block *blk) {
  for (auto sucBlk : blk->getSuccessors())
    if (sucBlk == blk)
      return true;
  return false;
}

Operation *generateValidConstant(arith::ConstantOp constOp,
                                 PatternRewriter &rewriter) {
  int32_t valAttr =
      (int32_t)constOp->getAttr("value").dyn_cast<IntegerAttr>().getInt();

  Location loc = constOp->getLoc();
  // check whether all the user of the consOp is in exit block, if true, set
  // the insertion location to the exit block
  bool allExit = true;
  for (auto user : constOp->getUsers())
    if (!isa<func::ReturnOp>(user->getBlock()->getTerminator())) {
      allExit = false;
      break;
    }
  if (allExit) {
    auto firstOp = *constOp->getUsers().begin();
    rewriter.setInsertionPoint(firstOp);
    loc = firstOp->getLoc();
  } else {
    rewriter.setInsertionPoint(constOp);
  }

  // get the lowest 0-11 bits
  const int32_t mask0 = 0b111111111111;
  const int32_t mask1 = 0b111111111111000000000000;
  const int32_t mask2 = 0b11111111000000000000000000000000;
  if (!(mask1 & valAttr))
    return nullptr;

  int32_t part0 = mask0 & valAttr;

  auto lowerBits = rewriter.create<arith::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(part0));
  rewriter.updateRootInPlace(constOp, [&] {
    constOp->setAttr("value", rewriter.getI32IntegerAttr(part0));
  });

  int32_t part1 = mask1 & valAttr;
  part1 = part1 >> 12;
  auto midBitVal = rewriter.create<arith::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(part1));
  auto zeroOp = rewriter.create<arith::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(0));
  // Generate midBits through add operation
  auto midBits =
      rewriter.create<arith::AddIOp>(loc, constOp.getResult().getType(),
                                     midBitVal.getResult(), zeroOp.getResult());

  auto shiftImm1 = rewriter.create<arith::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(12));
  auto shiftOp1 = rewriter.create<arith::ShLIOp>(
      loc, constOp.getResult().getType(), midBits.getResult(),
      shiftImm1.getResult());
  auto addOp = rewriter.create<arith::AddIOp>(
      loc, constOp.getResult().getType(), lowerBits.getResult(),
      shiftOp1.getResult());

  // If the higher bits are zero
  if (!(mask2 & valAttr)) {
    addOp->setAttr("constant", rewriter.getI32IntegerAttr(valAttr));
    rewriter.replaceUsesWithIf(
        constOp, addOp.getResult(),
        [&](OpOperand &operand) { return operand.getOwner() != addOp; });
    return addOp;
  }

  int32_t part2 = mask2 & valAttr;
  part2 = part2 >> 24;
  auto highBitVal = rewriter.create<arith::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(part2));
  auto zeroOp2 = rewriter.create<arith::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(0));
  // Generate highBits through add operation
  auto highBits = rewriter.create<arith::AddIOp>(
      loc, constOp.getResult().getType(), highBitVal.getResult(),
      zeroOp2.getResult());
  auto shiftImm2 = rewriter.create<arith::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(24));
  auto shiftOp2 = rewriter.create<arith::ShLIOp>(
      loc, constOp.getResult().getType(), highBits.getResult(),
      shiftImm2.getResult());
  auto sumOp =
      rewriter.create<arith::AddIOp>(loc, constOp.getResult().getType(),
                                     addOp.getResult(), shiftOp2.getResult());

  sumOp->setAttr("constant", rewriter.getI32IntegerAttr(valAttr));
  rewriter.replaceUsesWithIf(
      constOp, sumOp.getResult(),
      [&](OpOperand &operand) { return operand.getOwner() != sumOp; });
  return sumOp;
}

static LogicalResult outputDATE2023DAG(cgra::FuncOp funcOp,
                                       std::string outputDAG) {

  Block *loopBlk = nullptr;
  // Find the loop block
  for (auto &blk : funcOp.getBlocks())
    if (isLoopBlock(&blk)) {
      loopBlk = &blk;
      break;
    }
  // Get the oeprations in the loop block
  SmallVector<Operation *> nodes;
  for (Operation &op : loopBlk->getOperations()) {
    nodes.push_back(&op);
  }

  // initialize print function
  satmapit::PrintSatMapItDAG printer(loopBlk->getTerminator(), nodes);
  printer.init();
  if (failed(printer.printDAG(outputDAG)))
    return failure();

  return success();
}

arith::AddIOp generateImmAddOp(arith::ConstantOp constOp, Operation *user,
                               PatternRewriter &rewriter) {
  auto intAttr = constOp->getAttr("value").dyn_cast<IntegerAttr>();

  // insert before the first operation
  Location loc = user->getLoc();

  // insert a new zero constant operation
  // rewriter.setInsertionPoint(constOp);
  rewriter.setInsertionPoint(constOp);
  auto zeroConst = rewriter.create<arith::ConstantOp>(
      loc, constOp.getType(), rewriter.getI32IntegerAttr(0));
  // replicate the constant operation in case it is used by multiple
  // operations
  auto immConst = rewriter.create<arith::ConstantOp>(
      loc, constOp.getType(), rewriter.getI32IntegerAttr(intAttr.getInt()));
  rewriter.setInsertionPoint(user);
  auto addOp = rewriter.create<arith::AddIOp>(
      loc, constOp.getType(), zeroConst.getResult(), immConst.getResult());
  // set the imm attribute of the operation
  addOp->setAttr("constant", intAttr);
  return addOp;
}

LogicalResult SAddOpRewrite::matchAndRewrite(arith::AddIOp addOp,
                                             PatternRewriter &rewriter) const {
  rewriter.setInsertionPoint(addOp);
  arith::ConstantOp zeroCst = rewriter.create<arith::ConstantOp>(
      addOp.getLoc(), addOp.getType(), rewriter.getI32IntegerAttr(0));
  auto newOpA = rewriter.create<arith::AddIOp>(
      addOp.getLoc(), addOp.getOperand(0), zeroCst.getResult());
  rewriter.updateRootInPlace(addOp,
                             [&] { addOp->setOperand(0, newOpA.getResult()); });
  return success();
}

LogicalResult
ConstantOpRewrite::matchAndRewrite(arith::ConstantOp constOp,
                                   PatternRewriter &rewriter) const {

  auto value = constOp.getValue().cast<IntegerAttr>().getInt();

  // indirectly load & store cannot exceed reserved address range
  // if the constant is used as immediate for address
  if (isAddrConstOp(constOp) && !isValidImmAddr(constOp)) {
    if (auto validOp = existsConstant(value, insertedOps)) {
      // set it to valid range, to be removed later on
      constOp.replaceAllUsesWith(validOp);
    } else {
      auto addOp = generateValidConstant(constOp, rewriter);
      insertedOps.push_back(addOp);
    }
    // set the value to 0, to be removed later on
    rewriter.updateRootInPlace(constOp, [&] {
      constOp->setAttr("value", rewriter.getI32IntegerAttr(0));
    });
    return success();
  }

  // rewrite the constant operation if the value exceed the range
  if (value < -4097 || value > 4096) {
    auto addOp = generateValidConstant(constOp, rewriter);
    insertedOps.push_back(addOp);
    // set the value to 0, to be removed later on
    rewriter.updateRootInPlace(constOp, [&] {
      constOp->setAttr("value", rewriter.getI32IntegerAttr(0));
    });
    return success();
  }

  for (auto &use : llvm::make_early_inc_range(constOp->getUses())) {
    Operation *user = use.getOwner();
    // Always create new operation if the constant is used by branch ops,
    // which would be propagated to multiple operations in the successor
    // blocks.
    if (isa<cf::BranchOp, cgra::ConditionalBranchOp, cgra::SwiOp>(user)) {
      auto addOp = generateImmAddOp(constOp, user, rewriter);
      insertedOps.push_back(addOp);
      user->setOperand(use.getOperandNumber(), addOp.getResult());
    }
  }

  rewriter.updateRootInPlace(
      constOp, [&] { constOp->setAttr("value", constOp->getAttr("value")); });

  return success();
}

LogicalResult
CondBrOpRewrite::matchAndRewrite(cgra::ConditionalBranchOp condBrOp,
                                 PatternRewriter &rewriter) const {
  DenseSet<Value> sources;
  for (auto &opr : condBrOp->getOpOperands()) {
    if (opr.getOperandNumber() < 2)
      continue;
    auto val = opr.get();
    if (!sources.insert(opr.get()).second) {
      // generate a mov op to make the operands unique
      auto movOp = rewriter.create<arith::AddIOp>(val.getLoc(), val,
                                                  zeroOp->getResult(0));
      rewriter.updateRootInPlace(condBrOp, [&] {
        condBrOp->setOperand(opr.getOperandNumber(), movOp.getResult());
      });
    }
  }
  return success();
}

LogicalResult
BranchOpRewrite::matchAndRewrite(cf::BranchOp brOp,
                                 PatternRewriter &rewriter) const {
  DenseSet<Value> sources;
  for (auto &opr : brOp->getOpOperands()) {
    auto val = opr.get();
    if (!sources.insert(opr.get()).second) {
      // generate a mov op to make the operands unique
      auto movOp = rewriter.create<arith::AddIOp>(val.getLoc(), val,
                                                  zeroOp->getResult(0));
      rewriter.updateRootInPlace(brOp, [&] {
        brOp->setOperand(opr.getOperandNumber(), movOp.getResult());
      });
    }
  }
  return success();
}

LogicalResult LwiOpRewrite::matchAndRewrite(cgra::LwiOp lwiOp,
                                            PatternRewriter &rewriter) const {
  auto origType = lwiOp.getOperand().getType(); // valid I32 address type

  rewriter.updateRootInPlace(
      lwiOp, [&] { lwiOp.getResult().setType(lwiOp.getResult().getType()); });
  // set the type of corresponding successor
  std::stack<Value> dstOprs;
  dstOprs.push(lwiOp.getResult());
  while (!dstOprs.empty()) {
    auto opr = dstOprs.top();
    dstOprs.pop();

    // change opr to origType
    opr.setType(origType);

    // push the predecessors if its type is not the same as result type
    for (auto user : opr.getUsers())
      if (user->getNumResults() == 1 &&
          !isValidDataType(user->getResult(0).getType()))
        dstOprs.push(user->getResult(0));
  }
  return success();
}

void CgraFitToOpenEdgePass::runOnOperation() {
  ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns{ctx};
  OpenEdgeISATarget target(ctx);
  OpBuilder builder(ctx);

  SmallVector<Operation *> insertedOps;
  // get the front operation
  auto funcOp = *modOp.getOps<func::FuncOp>().begin();
  Operation *frontOp = funcOp.getBody().front().getTerminator();
  Operation *zeroOp = nullptr;
  for (auto &op : frontOp->getBlock()->getOperations()) {
    if (isa<arith::ConstantOp>(op) &&
        op.getAttr("value").cast<IntegerAttr>().getValue() == 0) {
      zeroOp = &op;
      break;
    }
  }
  if (!zeroOp)
    zeroOp = builder.create<arith::ConstantOp>(
        frontOp->getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
  patterns.add<ConstantOpRewrite>(ctx, insertedOps, frontOp);
  patterns.add<CondBrOpRewrite>(ctx, zeroOp);
  patterns.add<BranchOpRewrite>(ctx);
  patterns.add<LwiOpRewrite>(ctx);
  patterns.add<SAddOpRewrite>(ctx);

  // adapt the constant operation to meet the requirement of Imm field of
  // openedge CGRA
  if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
    signalPassFailure();

  // raise the constant operation to the top level
  auto cgraFuncOps = modOp.getOps<cgra::FuncOp>();
  if (!cgraFuncOps.empty()) {
    auto funcOp = *cgraFuncOps.begin();
    if (failed(raiseConstOperation<cgra::FuncOp>(funcOp)) ||
        failed(removeUnusedConstOp<cgra::FuncOp>(funcOp)) ||
        failed(removeEqualWidthBWOp(funcOp)))
      signalPassFailure();
  } else {
    auto funcOps = modOp.getOps<func::FuncOp>();
    if (!funcOps.empty()) {
      auto funcOp = *funcOps.begin();
      if (failed(removeUnusedConstOp<func::FuncOp>(funcOp)))
        signalPassFailure();
    }
  }

  // print the DAG of the specified function
  if (!outputDAG.empty()) {
    size_t lastSlashPos = outputDAG.find_last_of("/");
    bool isPath = lastSlashPos != StringRef::npos;
    StringRef funcName =
        isPath ? outputDAG.substr(lastSlashPos + 1) : outputDAG;

    for (auto funcOp :
         llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>())) {
      if (funcName == funcOp.getName() &&
          failed(outputDATE2023DAG(funcOp, outputDAG)))
        return signalPassFailure();
    }
  }
}

namespace compigra {
std::unique_ptr<mlir::Pass> createCgraFitToOpenEdge(StringRef outputDAG) {
  return std::make_unique<CgraFitToOpenEdgePass>(outputDAG);
}
} // namespace compigra