//===- CfToCgraConversion.h - Convert part ops to Cgra dialect *--- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --convert-cf-to-cgra pass.
//
//===----------------------------------------------------------------------===//

#ifndef CF_TO_CGRA_CONVERSION_H
#define CF_TO_CGRA_CONVERSION_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace compigra {
#define GEN_PASS_DEF_CFTOCGRACONVERSION
#define GEN_PASS_DECL_CFTOCGRACONVERSION
#include "compigra/Conversion/Passes.h.inc"

void populateCfToCgraConversionPatterns(
    RewritePatternSet &patterns, SmallVector<Operation *> &baseAddrs,
    std::map<llvm::StringRef, Operation *> &globalConstAddrs,
    DenseMap<Operation *, SmallVector<Operation *>> &strideValMap);

std::unique_ptr<mlir::Pass> createCfToCgraConversion();
} // namespace compigra

namespace {
struct CfToCgraConversionPass
    : public compigra::impl::CfToCgraConversionBase<CfToCgraConversionPass> {
  CfToCgraConversionPass() {}

  void runOnOperation() override;
};
} // namespace

#endif // CF_TO_CGRA_CONVERSION_H
