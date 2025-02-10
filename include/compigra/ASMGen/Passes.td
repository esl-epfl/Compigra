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
// Cgra passes
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
    Option<"nGrid", "grid", "int", "", "  Number of grid blocks">,
  ];
}

def ASMGenTemporalCGRA : Pass<"gen-temporal-cgra-asm"> {
  let summary = "Generate temporal cgra code";
  let constructor = "compigra::createASMGenTemporalCGRA()";
  let dependentDialects = ["cgra::CgraDialect"];
  let options = [
    Option<"nRow", "row", "int", "", "  Number of PE in one row">,
    Option<"nCol", "col", "int", "", "  Number of PE in one column">,
    Option<"mem", "mem", "int", "",
           "  Start address of the memory for the evicted value">,
    Option<"asmOutDir", "asm-out", "std::string", "",
           "  Specify the directory of the assembly file">,
  ];
}

#endif // COMPIGRA_PASSES_TD
