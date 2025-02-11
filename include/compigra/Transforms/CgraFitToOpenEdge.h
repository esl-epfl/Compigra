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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include <stack>

using namespace mlir;

/// Check whether a constant operation is used as Imm field for an memory
/// address.
bool isAddrConstOp(arith::ConstantOp constOp);

/// Move all constant operations to the entry block
template <typename FuncOp> LogicalResult raiseConstOperation(FuncOp funcOp);

/// Erase constant operations if they are not in used
template <typename FuncOp> LogicalResult removeUnusedConstOp(FuncOp funcOp);

/// Erase bitwidth related operations if they have been rewritten and have equal
/// bitwidths for input and output.
LogicalResult removeEqualWidthBWOp(cgra::FuncOp funcOp);

/// Use add operation to generate a value if Imm field is not allowed.
arith::AddIOp generateImmAddOp(arith::ConstantOp constOp, Operation *user,
                               PatternRewriter &rewriter);

/// Check whether the constant operation has been adapted to generate by
/// computation. Avoid generate multiple
Operation *existsConstant(int intVal, SmallVector<Operation *> &insertedOps);

/// Rewrite constant operations can not fit into Imm field
Operation *generateValidConstant(arith::ConstantOp constOp,
                                 PatternRewriter &rewriter);

/// OpenEdge address Imm field
bool isValidImmAddr(arith::ConstantOp constOp);

namespace {
// Initialze the IR target that can not be deployed in the openedge
// CGRA
struct OpenEdgeISATarget : public ConversionTarget {
  OpenEdgeISATarget(MLIRContext *ctx);
};

} // namespace

namespace compigra {
#define GEN_PASS_DEF_CGRAFITTOOPENEDGE
#define GEN_PASS_DECL_CGRAFITTOOPENEDGE
#include "compigra/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCgraFitToOpenEdge(StringRef outputDAG = "");
} // end namespace compigra

#endif // CGRA_FIT_TO_OPENEDGE_H