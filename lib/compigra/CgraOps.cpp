//===- CgraOps.cpp - Handshake MLIR Operations ----------------0-----------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Cgra operations struct.
//
//===----------------------------------------------------------------------===//

#include "compigra/CgraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace cgra;

static ParseResult parseIntInSquareBrackets(OpAsmParser &parser, int &v) {
  if (parser.parseLSquare() || parser.parseInteger(v) || parser.parseRSquare())
    return failure();
  return success();
}

static ParseResult
parseSostOperation(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                   OperationState &result, int &size, Type &type,
                   bool explicitSize) {
  if (explicitSize)
    if (parseIntInSquareBrackets(parser, size))
      return failure();

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (!explicitSize)
    size = operands.size();
  return success();
}

ParseResult MergeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> resultTypes, dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (parseSostOperation(parser, allOperands, result, size, type, false))
    return failure();

  dataOperandsTypes.assign(size, type);
  resultTypes.push_back(type);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void MergeOp::print(OpAsmPrinter &p) { sostPrint(p, false); }
