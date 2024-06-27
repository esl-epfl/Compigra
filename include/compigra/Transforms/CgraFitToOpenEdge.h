//===- CgraFitToHW.h - Rewrite ops to allow it fit in HW ISA *- C++ -----*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --fit-openedge pass.
//
//===----------------------------------------------------------------------===//

#ifndef CGRA_FIT_TO_OPENEDGE_H
#define CGRA_FIT_TO_OPENEDGE_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Check whether a constant operation is used as Imm field for an memory
/// address.
bool isAddrConstOp(LLVM::ConstantOp constOp);

/// Move all constant operations to the entry block
LogicalResult raiseConstOperation(cgra::FuncOp funcOp);

/// Erased constant operations if they are not in used
LogicalResult removeUnusedConstOp(cgra::FuncOp funcOp);

/// Use add operation to generate a value if Imm field is not allowed.
LLVM::AddOp generateImmAddOp(LLVM::ConstantOp constOp,
                             PatternRewriter &rewriter);

/// Check whether the constant operation has been adapted to generate by
/// computation. Avoid generate multiple
Operation *existsConstant(int intVal, SmallVector<Operation *> &insertedOps);

/// Rewrite constant operations can not fit into Imm field
Operation *generateValidConstant(LLVM::ConstantOp constOp,
                                 PatternRewriter &rewriter);

/// OpenEdge address Imm field
bool isValidImmAddr(LLVM::ConstantOp constOp);

namespace {
// Initialze the constant target that can not be deployed in the openedge
// CGRA
struct ConstTarget : public ConversionTarget {
  ConstTarget(MLIRContext *ctx);
};

} // namespace

namespace compigra {
#define GEN_PASS_DEF_CGRAFITTOOPENEDGE
#define GEN_PASS_DECL_CGRAFITTOOPENEDGE
#include "compigra/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCgraFitToOpenEdge(StringRef outputDAG = "");
} // end namespace compigra

namespace {
/// Driver for the fit-openedge pass.
struct CgraFitToOpenEdgePass
    : public compigra::impl::CgraFitToOpenEdgeBase<CgraFitToOpenEdgePass> {

  explicit CgraFitToOpenEdgePass(StringRef outputDAG) {}
  void runOnOperation() override;
};

struct ConstantOpRewrite : public OpRewritePattern<LLVM::ConstantOp> {
  ConstantOpRewrite(MLIRContext *ctx, int *constBase,
                    SmallVector<Operation *> &insertedOps)
      : OpRewritePattern(ctx), constBase(constBase), insertedOps(insertedOps) {}

  LogicalResult matchAndRewrite(LLVM::ConstantOp constOp,
                                PatternRewriter &rewriter) const override;

protected:
  int *constBase;
  SmallVector<Operation *> &insertedOps;
};
} // namespace

#endif // CGRA_FIT_TO_OPENEDGE_H