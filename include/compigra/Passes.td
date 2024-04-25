//===-  Passes.h - Passes definition ----------------------------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_PASSES_TD
#define COMPIGRA_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Cf passes
//===----------------------------------------------------------------------===//

def CfFixIndexWidth : Pass<"fix-index-width"> {
  let summary = "Fix the index width to CGRA processing element width";
  let constructor = "mlir::compigra::createCfFixIndexWidth()";
}

#endif // COMPIGRA_PASSES_TD
