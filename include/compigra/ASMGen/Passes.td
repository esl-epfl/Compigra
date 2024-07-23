//===-  Passes.h - Passes definition ----------------------------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_PASSES_TD
#define COMPIGRA_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LLVM passes
//===----------------------------------------------------------------------===//

def OpenEdgeASMGen : Pass<"gen-openedge-asm"> {
  let summary = "Generate open-edge assembly code";
  let constructor = "compigra::createOpenEdgeASMGen()";
  let dependentDialects = ["cgra::CgraDialect"];
  let options = [
    Option<"funcName", "func-name", "std::string", "",
           "  Specify the name of the fucntion to convert. ">,
    Option<"mapResult", "map-result", "std::string", "",
           "  Specify the map result by the SAT-MapIt">,
    Option<"nGrid", "grid", "int", "",
           "  Number of grid blocks">,
  ];
}

#endif // COMPIGRA_PASSES_TD
