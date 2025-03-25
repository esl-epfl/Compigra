//===-  CfFixIndexWidth.cpp - Fix index to CGRA PE bitwidth -----*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the fix-index-width pass, which rewrite the index of cf
// operation to match the bitwidth of the CGRA PE.
//
//===----------------------------------------------------------------------===//

#include "compigra/Transforms/CfFixIndexWidth.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

// for printing debug informations
// #include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace compigra;

LogicalResult fixIndexWidth(Operation &op, OpBuilder &builder,
                            unsigned bitwidth) {
  // revise the operand type to desired bitwidth
  for (auto opr : op.getOperands()) {
    if (opr.getType().isIntOrIndex())
      opr.setType(IntegerType::get(op.getContext(), bitwidth));
  }

  // revise the result type to desired bitwidth
  for (auto res : op.getResults()) {
    if (res.getType().isIntOrIndex())
      res.setType(IntegerType::get(op.getContext(), bitwidth));
  }

  // Check if the operation is a constant operation, set the value attribute to
  // int32
  if (isa<arith::ConstantOp>(op))
    if (auto intAttr = op.getAttr("value").dyn_cast_or_null<IntegerAttr>()) {
      int maxValue = (1 << (bitwidth - 1)) - 1;
      int minValue = -(1 << (bitwidth - 1));
      if (intAttr.getInt() > maxValue || intAttr.getInt() < minValue) {
        return failure();
      }
      // convert the value attributes  to int32
      op.setAttr(
          "value",
          IntegerAttr::get(IntegerType::get(op.getContext(), bitwidth),
                           op.getAttr("value").cast<IntegerAttr>().getInt()));
    }

  return success();
}

namespace {
struct OpenEdgeHWTarget : public ConversionTarget {
  OpenEdgeHWTarget(MLIRContext *ctx) : ConversionTarget(*ctx) {
    addIllegalOp<arith::IndexCastOp>();
    addIllegalOp<arith::ExtSIOp>();
    addIllegalOp<arith::TruncIOp>();
  }
};

struct IndexCastOpRewrite : public OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern<arith::IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp indexCastOp,
                                PatternRewriter &rewriter) const override {
    // replace the index_cast operation with its operand
    indexCastOp.getResult().replaceAllUsesWith(indexCastOp.getOperand());
    rewriter.eraseOp(indexCastOp);
    return success();
  }
};

struct ExtSIOpRewrite : public OpRewritePattern<arith::ExtSIOp> {
  using OpRewritePattern<arith::ExtSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtSIOp extSIOp,
                                PatternRewriter &rewriter) const override {
    // replace the extsi operation with its operand
    rewriter.replaceOp(extSIOp, extSIOp.getOperand());
    extSIOp.getResult().replaceAllUsesWith(extSIOp.getOperand());
    rewriter.eraseOp(extSIOp);
    return success();
  }
};

struct TruncIOpRewrite : public OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern<arith::TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncIOp truncIOp,
                                PatternRewriter &rewriter) const override {
    // replace the trunc operation with its operand
    rewriter.replaceOp(truncIOp, truncIOp.getOperand());
    truncIOp.getResult().replaceAllUsesWith(truncIOp.getOperand());
    rewriter.eraseOp(truncIOp);
    return success();
  }
};

/// Driver for the index bitwidth fix pass.
struct CfFixIndexWidthPass
    : public compigra::impl::CfFixIndexWidthBase<CfFixIndexWidthPass> {
  explicit CfFixIndexWidthPass(unsigned bitwidth) { this->bitwidth = bitwidth; }

  void runOnOperation() override {
    OpBuilder builder(&getContext());
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    auto funcOp = *modOp.getOps<func::FuncOp>().begin();

    // set every operation type to int32
    std::vector<Operation *> ops;
    getOperation()->walk([&](Operation *op) { ops.push_back(op); });
    for (auto *op : ops) {
      if (failed(fixIndexWidth(*op, builder, bitwidth)))
        return signalPassFailure();
    }

    // revise the block argument type to desired bitwidth
    for (auto &blk : funcOp.getBlocks()) {
      for (auto arg : blk.getArguments())
        if (arg.getType().isIntOrIndex())
          arg.setType(IntegerType::get(arg.getContext(), bitwidth));
    }

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns{ctx};
    OpenEdgeHWTarget target(ctx);

    patterns.add<IndexCastOpRewrite>(ctx);
    patterns.add<ExtSIOpRewrite>(ctx);
    patterns.add<TruncIOpRewrite>(ctx);

    if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createCfFixIndexWidth(unsigned bitwidth) {
  return std::make_unique<CfFixIndexWidthPass>(bitwidth);
}
} // namespace compigra