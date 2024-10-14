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

def LLVMToCgraConversion : Pass<"convert-llvm-to-cgra"> {
  let summary = "Convert LLVM to CGRA dialect";
  let constructor = "compigra::createLLVMToCgraConversion()";
  let dependentDialects = ["cgra::CgraDialect"];
  let options = [
    Option<"funcName", "func-name", "std::string", "",
           "    Specify the name of the fucntion to convert. ">,
    Option<
        "memAlloc", "mem-json", "std::string", "",
        "    Specify the memory allocation policy for data interface between "
        "CGRA and host processor ">
  ];
}

//===----------------------------------------------------------------------===//
// CF to CGRA passes
//===----------------------------------------------------------------------===//
def CfToCgraConversion : Pass<"convert-cf-to-cgra"> {
  let summary = "Convert CF to CGRA dialect";
  let constructor = "compigra::createCfToCgraConversion()";
  let dependentDialects = ["cgra::CgraDialect"];
  let options = [
    ListOption<"startAddr", "start-addr", "int", 
               "    Specify the start address of the parameters">
  ];
}

#endif // COMPIGRA_PASSES_TD
