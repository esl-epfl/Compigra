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

/// Base class for Cgra dialect operations.
class Cgra_Op<string mnemonic, list<Trait> traits = []>
    : Op<Cgra_Dialect, mnemonic, traits> {
}

/// Base class for Cgra dialect types.
class Cgra_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Cgra_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// This is almost exactly like a standard FuncOp, except that it defines a
// Dominance-Free Scope
def FuncOp : Cgra_Op<"func", [
  IsolatedFromAbove, FunctionOpInterface, OpAsmOpInterface, Symbol,
  RegionKindInterface
]> {
  let summary = "Cgra dialect function.";
  let description = [{The func operation represents a cgra function.This is
                          almost exactly like a standard FuncOp,
                      except that it has some extra verification conditions.}];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs);
  let regions = (region AnyRegion : $body);

  let skipDefaultBuilders = 1;

  let builders =
       [OpBuilder<(ins "llvm::StringRef":$name, "mlir::FunctionType":$type,
                      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    ::mlir::Block *addEntryBlock();

    /// Implement RegionKindInterface.
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }

    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    /// Returns the argument types of this function.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Returns the names of the arguments to this function.
    mlir::ArrayAttr getArgNames() {
      return (*this)->getAttrOfType<mlir::ArrayAttr>("argNames");
    }

    /// Returns the names of the results from this function.
    mlir::ArrayAttr getResNames() {
      return (*this)->getAttrOfType<mlir::ArrayAttr>("resNames");
    }

    /// Returns the argument name at the given index.
    mlir::StringAttr getArgName(unsigned idx) {
      return getArgNames()[idx].cast<mlir::StringAttr>();
    }

    /// Returns the result name at the given index.
    mlir::StringAttr getResName(unsigned idx) {
      return getResNames()[idx].cast<mlir::StringAttr>();
    }

    /// Resolve argument and result names. This can be used during building of
    /// a cgra.func operation to ensure that names provided by an incoming
    /// operation fulfills the name requirements of the cgra function.
    void resolveArgAndResNames();

    /// Hook for FunctionOpInterface, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    mlir::LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<mlir::FunctionType>())
        return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                           "' attribute of function type");
      return mlir::success();
    }

    /// Returns the body block of the function.
    mlir::Block *getBodyBlock() { return &getBody().front(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() { return "cgra"; }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def MergeOp : Cgra_Op<"merge", [
  Pure, MergeLikeOpInterface, SOSTInterface, SameOperandsAndResultType
]> {
  let summary = "merge operation";
  let description = [{
    The merge operation represents a(nondeterministic) merge operation
        .Any input is propagated to the single output
        .The number of inputs corresponds to the number of predecessor blocks.

    Example:
    ``` % 0 = merge % a,
    % b,
    %
    c: i32
    ```
  }];

  let arguments = (ins Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def BranchOp : Cgra_Op<"br", [
  Pure, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["sostIsControl"]>,
  // DeclareOpInterfaceMethods<ExecutableOpInterface>,
  // DeclareOpInterfaceMethods<GeneralOpInterface>,
  AllTypesMatch<["dataOperand", "dataResult"]>
]> {
  let summary = "branch operation";
  let description = [{
    The branch operation represents an unconditional
        branch.The single data input is propagated to the single successor
            .The input must be triggered by some predecessor to avoid continous
                triggering of a successor block.

    Example:
      ```mlir % 1 = br % 0: i32
      ```
  }];
  let arguments = (ins AnyType : $dataOperand);
  let results = (outs AnyType : $dataResult);
  let hasCustomAssemblyFormat = 1;
}

def ConditionalBranchOp : Cgra_Op<"cond_br", [
  Pure,
  // DeclareOpInterfaceMethods<ExecutableOpInterface>,
  // DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>,
  TypesMatchWith<"data operand type matches true branch result type",
                 "dataOperand", "trueResult", "$_self">,
  TypesMatchWith<"data operand type matches false branch result type",
                 "dataOperand", "falseResult", "$_self">
]> {
  let summary = "conditional branch operation";
  let description = [{
    The cbranch operation represents a conditional branch
        .The data input is propagated to one of the two outputs based on the
            condition input.

    Example:
     ```mlir % true,
    % false = conditional_branch % cond,
    %
    data: i32
     ```
  }];

  let arguments = (ins I1 : $conditionOperand, AnyType : $dataOperand);
  let results = (outs AnyType : $trueResult, AnyType : $falseResult);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 }
    ;
  }];
}

def BeqOp : Cgra_Op<"beq", [
  Pure, SOSTInterface,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>
]> {
  let summary = "branch if equal operation";
  let description = [{
    The branch operation represents a conditional branch to the operation which
        produces jumpOperands.

    Example:
      ```% 0 = beq[% operand1, % operand2] % jumpOperand{attributes}: i32
      ```
  }];
  let arguments = (ins Variadic<AnyType> : $dataOperands, AnyType : $jumpOperand);
  let results = (outs AnyType : $dataResult);
  let hasCustomAssemblyFormat = 1;
}

def BneOp : Cgra_Op<"bne", [
  Pure, SOSTInterface,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>
]> {
  let summary = "branch if not equal operation";
  let description = [{
    The branch operation represents a conditional branch to the operation which
        produces jumpOperands.

    Example:
      ```% 0 = bne[% operand1, % operand2] % jumpOperand{attributes}: i32
      ```
  }];
  let arguments = (ins Variadic<AnyType> : $dataOperands, AnyType : $jumpOperand);
  let results = (outs AnyType : $dataResult);
  let hasCustomAssemblyFormat = 1;
}

def BsfzOp : Cgra_Op<"bsfz", [
  Pure
  // SameOperandsAndResultType,
  // DeclareOpInterfaceMethods<ExecutableOpInterface>,
  // DeclareOpInterfaceMethods<GeneralOpInterface>,
]> {
  let summary = "bsfz operation";
  let description = [{
    The bsfz operation represents a select operation based on the flag input
        .The data input is propagated to the single output based on the flag
            input.

    Example:
      ``` mlir % 1 = bsfz % flag,
    [% trueOperand, % falseOperand]: i32
      ```
  }];
  let arguments = (ins  AnyType : $flagOperand, Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $dataResult);
  let hasCustomAssemblyFormat = 1;
}

def BsfaOp : Cgra_Op<"bsfa", [
  Pure
  // SameOperandsAndResultType,
  // DeclareOpInterfaceMethods<ExecutableOpInterface>,
  // DeclareOpInterfaceMethods<GeneralOpInterface>,
]> {
  let summary = "bsfa operation";
  let description = [{
    The bsfz operation represents a select operation based on the flag input
        .The data input is propagated to the single output based on the flag
            input.

    Example:
      ``` mlir % 1 = bsfa % flag,
    [% trueOperand, % falseOperand]: i32
      ```
  }];
  let arguments = (ins  AnyType : $flagOperand, Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $dataResult);
  let hasCustomAssemblyFormat = 1;
}

def LwiOp : Cgra_Op<"lwi", [Pure]> {
  let summary = "load word indirectly operation";
  let description = [{
    The lwi operations represents a load operation of the cgra,
    where it loads the data in the addressOperand to the result operand.

    Example:
      ``` mlir % 1 = lwi %
    addressOperand: i32
      ```
  }];
  let arguments = (ins AnyType : $addressOperand);
  let results = (outs AnyType : $dataResult);
  let hasCustomAssemblyFormat = 1;
}