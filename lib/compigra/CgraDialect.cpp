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
// #include "compigra/Ops.h"
// #include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace compigra;
using namespace compigra::cgra;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void CgraDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "compigra/CompigraOps.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#define GET_ATTRDEF_CLASSES
// #include "circt/Dialect/Handshake/HandshakeAttributes.cpp.inc"
// #include "circt/Dialect/Handshake/HandshakeDialect.cpp.inc"
// #include "circt/Dialect/Handshake/HandshakeEnums.cpp.inc"
// #include "circt/Dialect/Handshake/HandshakeInterfaces.cpp.inc"