//===- CompigraInterfaces.td - cgra interfaces -------------*- tablegen -*-===//
//
// Compigra under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the structured interface for Handshake ops.
//
//===----------------------------------------------------------------------===//

#ifndef CGRA_OP_INTERFACES
#define CGRA_OP_INTERFACES

include "mlir/IR/OpBase.td"

def SOSTInterface : OpInterface<"SOSTInterface"> {
  let cppNamespace = "::compigra::cgra";
  let description = [{
      Sized Operation with Single Type (SOST).
      
      These are operations whose operands all have the same type and which have
      an integer size property, be it the number of operation operands (e.g.,
      for a merge) or the number of operation results (e.g., for a fork).
  }];

  let methods = [
    InterfaceMethod<[{
        Get the data type associated to the operation.
        The default implementation of this method simply returns the type of
        the first operation operand.
      }],
      "mlir::Type", "getDataType", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp->getOperands().front().getType();
      }]
    >,
    InterfaceMethod<[{
        Get the size associated to the operation.
        The default implementation of this method simply returns the number of
        operation operands.
      }],
      "unsigned", "getSize", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp->getNumOperands();
      }]
    >,
    InterfaceMethod<[{
        Determine whether the operation is a control operation.
        The default implementation of this method assumes that the operation
        is a control operation if and only if its associated data type is a
        NoneType.
      }],
      "bool", "sostIsControl", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        // The operation is a control operation if its single data type is a 
        // NoneType.
        return concreteOp.getDataType().template isa<mlir::NoneType>();
      }]
    >,
    InterfaceMethod<[{
        Print the "SOST characteristics" of an operation.
        If the `explicitSize` parameter is set to true, then the method prints
        the operation's size (in the SOST sense) between square brackets before
        printing the operation's operands, attributes, and data type.
      }],
      "void", "sostPrint", (ins 
        "mlir::OpAsmPrinter &": $printer, "bool": $explicitSize
      ), "", 
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);

        if (explicitSize) {
          printer << " [" << concreteOp.getSize() << "]";
        }
        printer << " " << concreteOp->getOperands();
        printer.printOptionalAttrDict(concreteOp->getAttrs());
        printer << " : " << concreteOp.getDataType();
      }]
    >
  ];

  let verify = [{
    auto concreteOp = mlir::cast<ConcreteOp>($_op);

    // SOST operation's size must be at least one
    if (concreteOp.getSize() < 1) {
      return concreteOp.emitOpError(
        "SOST operation's size must be at least 1, but has size ") 
        << concreteOp.getSize();
    }

    // SOST operation's operands must all have the same type
    auto dataType = concreteOp.getDataType();
    for (auto operand : concreteOp->getOperands())
      if (operand.getType() != dataType) 
        return concreteOp.emitOpError("SOST operation reports having data type ")
          << dataType << ", but one operand has type " << operand.getType();
    
    return mlir::success();
  }];
}

def MergeLikeOpInterface : OpInterface<"MergeLikeOpInterface"> {
  let cppNamespace = "::compigra::cgra";
  let description = [{
     Some handshake operations can have predecessors in other
     blocks. This is primarily useful for verification purposes during
     lowering from other dialect, such as the standard CDFG dialect.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns an operand range over the data signals being merged.
      }],
      "mlir::OperandRange", "getDataOperands", (ins)
    >,
  ];

  let verify = [{
    auto concreteOp = mlir::cast<ConcreteOp>($_op);

    auto operands = concreteOp.getDataOperands();

    if (!operands.size())
      return concreteOp.emitOpError("must have at least one data operand");

    mlir::Type resultType = $_op->getResult(0).getType();

    for (auto operand : operands)
      if (operand.getType() != resultType)
        return concreteOp.emitOpError("operand has type ") << operand.getType()
            << ", but result has type " << resultType;

    return mlir::success();
  }];
}

def GeneralOpInterface : OpInterface<"GeneralOpInterface"> {
  let cppNamespace = "::compigra::cgra";
  let description =
      [{"Simulate the Execution of ops. The op takes a set of input values and "
        "returns the corresponding outputs assuming the precondition to "
        "execute holds."}];

  let methods = [
    InterfaceMethod<
        "Simulate the Execution of the general op with given inputs", "void",
        "execute",
        (ins "std::vector<llvm::Any> &" : $ins,
           "std::vector<llvm::Any> &" : $outs)>,
  ];
}

#endif // CGRA_OP_INTERFACES