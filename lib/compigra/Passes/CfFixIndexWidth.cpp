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

#include "compigra/Passes/CfFixIndexWidth.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

// for printing debug informations
// #include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace compigra;

LogicalResult fixIndexWidth(Operation &op, OpBuilder &builder) {
  // Make sure the operator is valid for memref load and store
  if (isa<memref::LoadOp>(op)) {
    // ensure the memref pointer is index
    if (!isa<IndexType>(op.getOperand(1).getType()))
      // insert index_cast operation
      builder.setInsertionPoint(&op);
    auto index = builder.create<arith::IndexCastOp>(
        op.getLoc(), builder.getIndexType(), op.getOperand(1));
    op.setOperand(1, index);
    return success();
  } else if (isa<memref::StoreOp>(op)) {
    // ensure the memref pointer is index
    if (!isa<IndexType>(op.getOperand(2).getType()))
      // insert index_cast operation
      builder.setInsertionPoint(&op);
    auto index = builder.create<arith::IndexCastOp>(
        op.getLoc(), builder.getIndexType(), op.getOperand(2));
    op.setOperand(2, index);
    return success();
  }

  // revise the operand type to int32
  if (op.getNumOperands() > 0)
    for (auto opr : op.getOperands())
      if (opr.getType().isIndex())
        opr.setType(IntegerType::get(op.getContext(), 32));

  // revise the result type to int32
  if (op.getNumResults() > 0)
    for (auto res : op.getResults()) {
      // if (res.getUse())
      if (res.getType().isIndex())
        res.setType(IntegerType::get(op.getContext(), 32));
    }

  // erase index_cast operation
  if (isa<arith::IndexCastOp>(op)) {
    // replace the index_cast operation with its operand
    op.getResult(0).replaceAllUsesWith(op.getOperand(0));
    op.erase();
    return success();
  }

  // Check if the operation is a constant operation, set the value attribute to
  // int32
  if (isa<arith::ConstantOp>(op))
    if (auto attrType = op.getAttr("value").cast<IntegerAttr>().getType())
      if (attrType.isa<IndexType>()) {
        // convert the value attributes  to int32
        op.setAttr(
            "value",
            IntegerAttr::get(IntegerType::get(op.getContext(), 32),
                             op.getAttr("value").cast<IntegerAttr>().getInt()));
      }

  return success();
}

namespace {
/// Driver for the index bitwidth fix pass.
struct CfFixIndexWidthPass
    : public compigra::impl::CfFixIndexWidthBase<CfFixIndexWidthPass> {
  void runOnOperation() override {
    OpBuilder builder(&getContext());

    std::vector<Operation *> ops;
    getOperation()->walk([&](Operation *op) { ops.push_back(op); });
    for (auto *op : ops) {
      if (failed(fixIndexWidth(*op, builder)))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace compigra {
std::unique_ptr<mlir::Pass> createCfFixIndexWidth() {
  return std::make_unique<CfFixIndexWidthPass>();
}
} // namespace compigra
} // namespace mlir