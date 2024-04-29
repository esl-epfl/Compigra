//===- Dialect.h - compigra dialect declaration -----------------*- C++ -*-===//
//
// Compigra under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Handshake MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CGRA_DIALECT_H
#define CGRA_DIALECT_H

#include "mlir/IR/Dialect.h"

// Pull in the Dialect definition.
#include "compigra/CgraDialect.h.inc"

// Pull in all enum type definitions, attributes,
// and utility function declarations.
#include "compigra/CgraEnums.h.inc"

#endif // CGRA_DIALECT_H