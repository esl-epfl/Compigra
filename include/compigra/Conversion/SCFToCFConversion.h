//===- SCFToCFConversion.h - SCF to ControlFlow Conversion --*- C++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCF_TO_CF_CONVERSION_H
#define SCF_TO_CF_CONVERSION_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>
using namespace mlir;

namespace mlir {
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace compigra {
#define GEN_PASS_DEF_SCFTOCONTROLFLOW
#define GEN_PASS_DECL_SCFTOCONTROLFLOW
#include "compigra/Conversion/Passes.h.inc"
/// Collect a set of patterns to convert SCF operations to CFG branch-based
/// operations within the ControlFlow dialect.
void populateSCFToControlFlowConversionPatterns(
    mlir::RewritePatternSet &patterns);

/// Creates a pass to convert SCF operations to CFG branch-based operation in
/// the ControlFlow dialect.
std::unique_ptr<Pass> createSCFToControlFlowPass();
} // namespace compigra

#endif // SCF_TO_CF_CONVERSION_H
