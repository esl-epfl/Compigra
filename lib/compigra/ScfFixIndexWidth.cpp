//===-  ScfFixIndexWidth.cpp - Fix index to CGRA PE bitwidth ----*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the scf-fix-index-width pass, which rewrite the index of scf
// operation to match the bitwidth of the CGRA PE.
//
//===----------------------------------------------------------------------===//

#include "compigra/ScfFixIndexWidth.h"

using namespace mlir;
using namespace compigra;

namespace {
/// Driver for the index bitwidth fix pass.
struct ScfFixIndexWidthPass
    : public compigra::impl::ScfFixIndexWidthBase<ScfFixIndexWidthPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::compigra::createScfFixIndexWidth() {
  return std::make_unique<ScfFixIndexWidthPass>();
}