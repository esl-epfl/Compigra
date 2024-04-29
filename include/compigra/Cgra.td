//===- Cgra.td - Cgra dialect definition ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file originates from the CIRCT project (https://github.com/llvm/circt).
// It includes modifications made as part of Dynamatic.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Cgra dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CGRA_DIALECT_TD
#define CGRA_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "compigra/CgraInterfaces.td"

def Cgra_Dialect : Dialect {
  let name = "cgra";
  let cppNamespace = "::cgra";

  let summary = "Types and operations for the Cgra dialect";
  let description = [{This dialect defined the `cgra` dialect, providing branch,
                      select, and merge Op in CGRA ASM.}];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  // Opt-out of properties for now, must migrate by LLVM 19.  #5273.
  let usePropertiesForAttributes = 0;
}

include "compigra/CgraOps.td"
#endif // CGRA_DIALECT_TD