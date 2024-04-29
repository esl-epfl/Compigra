//===- Dialect.cpp - Implement the Handshake dialect ----------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#include "compigra/CgraDialect.h"
#include "compigra/CgraOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace cgra;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void CgraDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "compigra/Cgra.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.

#include "compigra/CgraDialect.cpp.inc"
#include "compigra/CgraEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "compigra/CgraAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "compigra/CgraTypes.cpp.inc"
// #include "compigra/CgraInterfaces.cpp.inc"