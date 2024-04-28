//===- Ops.td - Cgra operation definitions -----------------*- tablegen -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//
//
// This file define Cgra ops in tablegen.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "compigra/CgraDialect.td"


/// Base class for Cgra dialect operations.
class Cgra_Op<string mnemonic, list<Trait> traits = []>
    : Op<Cgra_Dialect, mnemonic, traits> {
}

/// Base class for Cgra dialect types.
class Cgra_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Cgra_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MergeOp : Cgra_Op<"merge", [
  Pure, SOSTInterface, SameOperandsAndResultType
]> {
  let summary = "merge operation";
  let description = [{
    The merge operation represents a (nondeterministic)
    merge operation. Any input is propagated to the single output. The
    number of inputs corresponds to the number of predecessor
    blocks.

    Example:
    ```
    %0 = merge %a, %b, %c : i32
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$dataOperands);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}