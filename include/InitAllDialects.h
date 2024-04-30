//===- InitAllDialects.h - Compigra dialects registration --------*- C++-*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects that
// Compigra users may care about.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_INITALLDIALECTS_H
#define COMPIGRA_INITALLDIALECTS_H

#include "compigra/CgraDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace compigra {

inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::LLVM::LLVMDialect, mlir::affine::AffineDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect,
                  cgra::CgraDialect>();
}

} // namespace compigra
} // namespace mlir

#endif // COMPIGRA_INITALLDIALECTS_H