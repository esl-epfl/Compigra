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
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

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
      return mlir::RegionKind::SSACFG;
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

def BlasGemmOp : Cgra_Op<"gemm_blas",
                         [Pure, MemoryEffectsOpInterface, IsolatedFromAbove]> {
  let summary = "specialized gemm kernel for assembly generation";
  let description = [{
    The gemm_blas operation represents pre -
    compiled assembly code would be generated with the input.

    Example:
      ``` gemm_blas(% inner_bound, % middle_bound, % outer_bound, % in_mat1,
                     % in_mat2, % out_mat)      ```
  }];

  let arguments = (ins AnyTypeOf<[Index, AnyInteger]>
                   : $innerBound, AnyTypeOf<[Index, AnyInteger]>
                   : $middleBound, AnyTypeOf<[Index, AnyInteger]>
                   : $outerBound,
                   AnyRankedOrUnrankedMemRef:$matIn1,
                       AnyRankedOrUnrankedMemRef:$matIn2,
                       AnyRankedOrUnrankedMemRef:$matOut);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
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

def JumpOp : Cgra_Op<"jump", [
  Pure, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["sostIsControl"]>,
  // DeclareOpInterfaceMethods<ExecutableOpInterface>,
  // DeclareOpInterfaceMethods<GeneralOpInterface>,
  AllTypesMatch<["dataOperand", "dataResult"]>
]> {
  let summary = "Jump operation";
  let description = [{
    The jump operation represents an unconditional
        branch.The single data input is propagated to the single successor
            .The input must be triggered by some predecessor to avoid continous
                triggering of a successor block.

    Example:
      ```mlir % 1 = jump % 0: i32
      ```
  }];
  let arguments = (ins AnyType : $dataOperand);
  let results = (outs AnyType : $dataResult);
  let hasCustomAssemblyFormat = 1;
}

def ConditionalBranchOp : Cgra_Op<"cond_br", [
  Pure, Terminator, AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>
]> {
  let summary = "branch under specific condition";
  let description = [{
    The branch operation represents a conditional branch to the true basic block

    Example:
      ```% 0 = beq[% operand1, % operand2] ^ BBi(% dataoperands),
    ^BBj{
        attributes}
      ```
  }];
  let arguments = (ins Cgra_CondBrIPredicateAttr: $predicate,
                  AnyInteger : $cmpOperand1, AnyInteger : $cmpOperand2, 
                   Variadic<AnyType> : $trueDestOperands, 
                   Variadic<AnyType> : $falseDestOperands);
  let successors = (successor AnySuccessor
                    : $trueDest, AnySuccessor
                    : $falseDest);

  let builders = [
    OpBuilder<(ins "cgra::CondBrPredicate": $predicate, "::mlir::Value":$cmpOperand1, "::mlir::Value":$cmpOperand2,
       "::mlir::Block *":$trueDest,
      "::mlir::ValueRange":$trueOperands, "::mlir::Block *":$falseDest,
      "::mlir::ValueRange":$falseOperands), [{
      build($_builder, $_state, predicate, cmpOperand1, cmpOperand2, 
      trueOperands, falseOperands, trueDest,
            falseDest);
    }]>,
    OpBuilder<(ins "cgra::CondBrPredicate": $predicate, "::mlir::Value":$cmpOperand1, "::mlir::Value":$cmpOperand2,
      "::mlir::Block *":$trueDest, "::mlir::Block *":$falseDest, 
      CArg<"::mlir::ValueRange", "{}">:$falseOperands), [{
      build($_builder, $_state, predicate, cmpOperand1, cmpOperand2, 
      trueDest, ValueRange(), falseDest,
            falseOperands);
    }]>];

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 }
    ;

    // static cgra::CondBrPredicate getPredicateByName(::mlir::StringRef name);

    size_t getNumTrueDestOperands() { return getTrueDestOperands().size(); }

    // Accessors for operands to the 'true' destination.
    ::mlir::Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueDestOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, ::mlir::Value value) {
      assert(idx < getNumTrueDestOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    void setTrueDest(::mlir::Block * block) { setSuccessor(block, trueIndex); }

    void setFalseDest(::mlir::Block * block) {
      setSuccessor(block, falseIndex);
    }

    unsigned getNumTrueOperands() { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index) {
      getTrueDestOperandsMutable().erase(index);
    }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      getFalseDestOperandsMutable().erase(index);
    }

    void eraseOperand(unsigned index) {
      if (index >= getFalseDestOperandIndex())
        eraseFalseOperand(index - getFalseDestOperandIndex());
      else if (index >= getTrueDestOperandIndex())
        eraseTrueOperand(index - getTrueDestOperandIndex());
      else
        llvm_unreachable("invalid operand index");
    }

    void eraseOperands(unsigned start, unsigned length) {
      for (unsigned i = 0; i < length; ++i)
        eraseOperand(start);
    }

    // Accessors for operands to the 'false' destination.
    ::mlir::Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, ::mlir::Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_range getTrueOperands() { return getTrueDestOperands(); }
    operand_range getFalseOperands() { return getFalseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 2; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let assemblyFormat = [{
    `<`$predicate`>`
      ` ` `[` $cmpOperand1 `:` type($cmpOperand1) `,` 
      $cmpOperand2 `:` type($cmpOperand2) `]` `,`
      $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
      $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
      attr-dict
  }];
  let hasVerifier = 1;
}

def BzfaOp : Cgra_Op<"bzfa", [
  Pure
  // SameOperandsAndResultType,
  // DeclareOpInterfaceMethods<ExecutableOpInterface>,
  // DeclareOpInterfaceMethods<GeneralOpInterface>,
]> {
  let summary = "bzfa operation";
  let description = [{
    The bzfa operation represents a select operation based on the flag input
        .The data input is propagated to the single output based on the flag
            input.

    Example:
      ``` mlir % 1 = bzfa % flag,
    [% trueOperand, % falseOperand]: i32
      ```
  }];
  let arguments = (ins  AnyType : $flagOperand, Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $dataResult);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    ::mlir::Value getPredicate() { return getOperand(0); };
  }];
  let hasVerifier = 1;
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
  let hasVerifier = 1;
}

def LwdOp : Cgra_Op<"lwd", [Pure]> {
  let summary = "load argument address";
  let description = [{
    The lwd operations represents a direct load operation of the cgra,
    where it loads the address of the argument operand.

    Example:
      ``` mlir % 1 = lwd %
    Operand: i32
      ```
  }];

  let arguments = (ins AnyType : $addressOperand);
  let results = (outs AnyType : $dataResult);

  let hasCustomAssemblyFormat = 1;
 let assemblyFormat = [{
  $addressOperand `:` type($addressOperand) `->` type($dataResult) `,` attr-dict
}];

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

def SwiOp : Cgra_Op<"swi", [Pure]> {
  let summary = "store word indirectly operation";
  let description = [{
    The swi operations represents a store operation of the cgra,
    where it stores the dataOperand to the address specified by the
        addressOperand.

    Example:
      ```swi %
    dataOperand: i32, % addressOperand: i32
      ```
  }];
  let arguments = (ins AnyType : $dataOperand, AnyType : $addressOperand);
  let hasCustomAssemblyFormat = 1;
}