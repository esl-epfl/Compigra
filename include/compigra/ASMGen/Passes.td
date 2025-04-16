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

def FastASMGenTemporalCGRA : Pass<"gen-openedge-asm-fast"> {
  let summary = "Generate open-edge assembly code";
  let constructor = "compigra::createFastASMGenTemporalCGRA()";
  let dependentDialects = ["cgra::CgraDialect", "cf::ControlFlowDialect"];
  let options = [
    Option<"nRow", "row", "int", "", "  Number of PE in one row">,
    Option<"nCol", "col", "int", "", "  Number of PE in one column">,
    Option<"mem", "mem", "int", "",
           "  Start address of the memory for the evicted value">,
    Option<"msOpt", "ms-opt", "std::string", "",
           "  Executable of modulo scheduler">,
    Option<"asmOutDir", "asm-out", "std::string", "",
           "  Specify the directory of the assembly file">,
  ];
}

def ASMGenTemporalCGRA : Pass<"gen-temporal-cgra-asm"> {
  let summary = "Generate temporal cgra code";
  let constructor = "compigra::createASMGenTemporalCGRA()";
  let dependentDialects = ["cgra::CgraDialect", "cf::ControlFlowDialect"];
  let options = [
    Option<"nRow", "row", "int", "", "  Number of PE in one row">,
    Option<"nCol", "col", "int", "", "  Number of PE in one column">,
    Option<"mem", "mem", "int", "",
           "  Start address of the memory for the evicted value">,
    Option<"msOpt", "ms-opt", "std::string", "",
           "  Executable of modulo scheduler">,
    Option<"asmOutDir", "asm-out", "std::string", "",
           "  Specify the directory of the assembly file">,
  ];
}

#endif // COMPIGRA_PASSES_TD
