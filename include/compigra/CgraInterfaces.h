//===- CgraInterfaces.h - Cgra op interfaces --------------------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces of the Cgra dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_CGRA_INTERFACES_H
#define COMPIGRA_CGRA_INTERFACES_H

#include "compigra/CgraDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/Any.h"


// namespace cgra {

// struct MemLoadInterface {
//   unsigned index;
//   mlir::Value addressIn;
//   mlir::Value dataOut;
//   mlir::Value doneOut;
// };

// struct MemStoreInterface {
//   unsigned index;
//   mlir::Value addressIn;
//   mlir::Value dataIn;
//   mlir::Value doneOut;
// };

// /// Default implementation for checking whether an operation is a control
// /// operation. This function cannot be defined within ControlInterface
// /// because its implementation attempts to cast the operation to an
// /// SOSTInterface, which may not be declared at the point where the default
// /// trait's method is defined. Therefore, the default implementation of
// /// ControlInterface's isControl method simply calls this function.
// bool isControlOpImpl(Operation *op);
// } // end namespace cgra

#include "compigra/CgraInterfaces.h.inc"

#endif // COMPIGRA_CGRA_INTERFACES_H
