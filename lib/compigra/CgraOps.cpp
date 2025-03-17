//===- CgraOps.cpp - Cgra MLIR Operations ---------------------------------===//
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

namespace mlir {
class AsmParser;
} // namespace mlir

using namespace mlir;
using namespace cgra;

static ParseResult parseIntInSquareBrackets(OpAsmParser &parser, int &v) {
  if (parser.parseLSquare() || parser.parseInteger(v) || parser.parseRSquare())
    return failure();
  return success();
}

/// Parse the sized operation with single type
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

/// Generates names for a cgra.func input and output arguments, based on
/// the number of args as well as a prefix.
static SmallVector<Attribute> getFuncOpNames(Builder &builder, unsigned cnt,
                                             StringRef prefix) {
  SmallVector<Attribute> resNames;
  for (unsigned i = 0; i < cnt; ++i)
    resNames.push_back(builder.getStringAttr(prefix + std::to_string(i)));
  return resNames;
}

/// Helper function for appending input/output attributes for cgra FuncOp.
void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(FuncOp::getFunctionTypeAttrName(state.name),
                     TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());

  if (const auto *argNamesAttrIt = llvm::find_if(
          attrs, [&](auto attr) { return attr.getName() == "argNames"; });
      argNamesAttrIt == attrs.end())
    state.addAttribute("argNames", builder.getArrayAttr({}));

  if (llvm::find_if(attrs, [&](auto attr) {
        return attr.getName() == "resNames";
      }) == attrs.end())
    state.addAttribute("resNames", builder.getArrayAttr({}));

  state.addRegion();
}

/// Helper function for appending a string to an array attribute, and
/// rewriting the attribute back to the operation.
static void addStringToStringArrayAttr(Builder &builder, Operation *op,
                                       StringRef attrName, StringAttr str) {
  llvm::SmallVector<Attribute> attrs;
  llvm::copy(op->getAttrOfType<ArrayAttr>(attrName).getValue(),
             std::back_inserter(attrs));
  attrs.push_back(str);
  op->setAttr(attrName, builder.getArrayAttr(attrs));
}

void FuncOp::resolveArgAndResNames() {
  Builder builder(getContext());

  /// Generate a set of fallback names. These are used in case names are
  /// missing from the currently set arg- and res name attributes.
  auto fallbackArgNames = getFuncOpNames(builder, getNumArguments(), "in");
  auto fallbackResNames = getFuncOpNames(builder, getNumResults(), "out");
  auto argNames = getArgNames().getValue();
  auto resNames = getResNames().getValue();

  /// Use fallback names where actual names are missing.
  auto resolveNames = [&](auto &fallbackNames, auto &actualNames,
                          StringRef attrName) {
    for (auto fallbackName : llvm::enumerate(fallbackNames)) {
      if (actualNames.size() <= fallbackName.index())
        addStringToStringArrayAttr(
            builder, this->getOperation(), attrName,
            fallbackName.value().template cast<StringAttr>());
    }
  };
  resolveNames(fallbackArgNames, argNames, "argNames");
  resolveNames(fallbackResNames, resNames, "resNames");
}

LogicalResult FuncOp::verify() {
  // If this function is external there is nothing to do.
  if (isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the
  // entry block line up.  The trait already verified that the number of
  // arguments is the same between the signature and the block.
  auto fnInputTypes = getArgumentTypes();
  Block &entryBlock = front();

  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  // Verify that we have a name for each argument and result of this
  // function.
  auto verifyPortNameAttr = [&](StringRef attrName,
                                unsigned numIOs) -> LogicalResult {
    auto portNamesAttr = (*this)->getAttrOfType<ArrayAttr>(attrName);

    if (!portNamesAttr)
      return emitOpError() << "expected attribute '" << attrName << "'.";

    auto portNames = portNamesAttr.getValue();
    if (portNames.size() != numIOs)
      return emitOpError() << "attribute '" << attrName << "' has "
                           << portNames.size()
                           << " entries but is expected to have " << numIOs
                           << ".";

    if (llvm::any_of(portNames,
                     [&](Attribute attr) { return !attr.isa<StringAttr>(); }))
      return emitOpError() << "expected all entries in attribute '" << attrName
                           << "' to be strings.";

    return success();
  };
  if (failed(verifyPortNameAttr("argNames", getNumArguments())))
    return failure();
  if (failed(verifyPortNameAttr("resNames", getNumResults())))
    return failure();

  return success();
}

/// Parses a FuncOp signature using
/// mlir::function_interface_impl::parseFunctionSignature while getting access
/// to the parsed SSA names to store as attributes.
static ParseResult
parseFuncOpArgs(OpAsmParser &parser,
                SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
                SmallVectorImpl<Type> &resTypes,
                SmallVectorImpl<DictionaryAttr> &resAttrs) {
  bool isVariadic;
  if (mlir::function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/true, entryArgs, isVariadic, resTypes,
          resAttrs)
          .failed())
    return failure();

  return success();
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  StringAttr nameAttr;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> resTypes;
  SmallVector<DictionaryAttr> resAttributes;
  SmallVector<Attribute> argNames;

  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse signature
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parseFuncOpArgs(parser, args, resTypes, resAttributes))
    return failure();
  mlir::function_interface_impl::addArgAndResultAttrs(
      builder, result, args, resAttributes,
      cgra::FuncOp::getArgAttrsAttrName(result.name),
      cgra::FuncOp::getResAttrsAttrName(result.name));

  // Set function type
  SmallVector<Type> argTypes;
  for (auto arg : args)
    argTypes.push_back(arg.type);

  result.addAttribute(
      cgra::FuncOp::getFunctionTypeAttrName(result.name),
      TypeAttr::get(builder.getFunctionType(argTypes, resTypes)));

  // Determine the names of the arguments. If no SSA values are present, use
  // fallback names.
  bool noSSANames =
      llvm::any_of(args, [](auto arg) { return arg.ssaName.name.empty(); });
  if (noSSANames) {
    argNames = getFuncOpNames(builder, args.size(), "in");
  } else {
    llvm::transform(args, std::back_inserter(argNames), [&](auto arg) {
      return builder.getStringAttr(arg.ssaName.name.drop_front());
    });
  }

  // Parse attributes
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // If argNames and resNames wasn't provided manually, infer argNames attribute
  // from the parsed SSA names and resNames from our naming convention.
  if (!result.attributes.get("argNames"))
    result.addAttribute("argNames", builder.getArrayAttr(argNames));
  if (!result.attributes.get("resNames")) {
    auto resNames = getFuncOpNames(builder, resTypes.size(), "out");
    result.addAttribute("resNames", builder.getArrayAttr(resNames));
  }

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  llvm::SMLoc loc = parser.getCurrentLocation();
  auto parseResult = parser.parseOptionalRegion(*body, args,
                                                /*enableNameShadowing=*/false);
  if (!parseResult.has_value())
    return success();

  if (failed(*parseResult))
    return failure();
  // Function body was parsed, make sure its not empty.
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  // If a body was parsed, the arg and res names need to be resolved
  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/true, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
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

LogicalResult MergeOp::verify() {
  auto operands = getOperands();
  if (operands.empty())
    return emitOpError("operation must have at least one operand");
  return success();
}

ParseResult JumpOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (parseSostOperation(parser, allOperands, result, size, type, false))
    return failure();

  dataOperandsTypes.assign(size, type);
  result.addTypes({type});
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void JumpOp::print(OpAsmPrinter &p) { sostPrint(p, false); }

bool ConditionalBranchOp::isControl() { return true; }

LogicalResult ConditionalBranchOp::verify() {
  // check whether true destination block hase the same arugment size
  auto trueDest = getTrueDest();
  if (trueDest->getNumArguments() != getNumTrueDestOperands())
    return emitOpError() << "Successor #0 expect " << getNumTrueDestOperands()
                         << "arguments but have "
                         << trueDest->getNumArguments();
  auto falseDest = getFalseDest();
  if (falseDest->getNumArguments() != getNumFalseOperands())
    return emitOpError() << "Successor #1 expect " << getNumFalseOperands()
                         << " arguments but have "
                         << falseDest->getNumArguments();
  return success();
}

/// Parse the cgra select-like operation such as bzfa and bsfa.
static ParseResult
parseSelLikeOp(OpAsmParser &parser, OpAsmParser::UnresolvedOperand &jumpOperand,
               Type &dataType, Type &flagType,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
               OperationState &result) {
  if (parser.parseOperand(jumpOperand) || parser.parseColon() ||
      parser.parseType(flagType) || parser.parseLSquare() ||
      parser.parseOperandList(operands) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(dataType))
    return failure();

  return success();
}

/// Operands should share the same type
static bool hasSameType(SmallVector<Value> operands) {
  if (operands.size() <= 1)
    return true;
  auto type = operands.front().getType();
  for (int i = 1; i < operands.size(); ++i)
    if (operands[i].getType() != type)
      return false;

  return true;
}

LogicalResult BzfaOp::verify() {
  auto operands = getDataOperands();
  auto resultOpr = getDataResult();
  SmallVector<Value> allOpr(operands.begin(), operands.end());
  allOpr.push_back(resultOpr);

  if (!hasSameType(allOpr))
    return emitOpError() << "expected all data operands have the same type";
  return success();
}

LogicalResult BsfaOp::verify() {
  auto operands = getDataOperands();
  auto resultOpr = getDataResult();
  SmallVector<Value> allOpr(operands.begin(), operands.end());
  allOpr.push_back(resultOpr);

  if (!hasSameType(operands))
    return emitOpError() << "expected all data operands have the same type";
  return success();
}

ParseResult BzfaOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand flagOperand;
  Type dataType, flagType;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  if (failed(parseSelLikeOp(parser, flagOperand, dataType, flagType,
                            allOperands, result)))
    return failure();
  // push flagOperand in front of allOperands
  allOperands.insert(allOperands.begin(), flagOperand);

  SmallVector<Type, 3> dataOperandsTypes;
  // assign the same type to all operands
  dataOperandsTypes.push_back(flagType);
  dataOperandsTypes.push_back(dataType);
  dataOperandsTypes.push_back(dataType);

  // int size = allOperands.size();
  // dataOperandsTypes.assign(size, dataType);
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  result.addTypes(dataType);

  if (parser.resolveOperands(allOperands, ArrayRef<Type>(dataOperandsTypes),
                             allOperandLoc, result.operands))
    return failure();

  return success();
}

void BzfaOp::print(OpAsmPrinter &p) {
  auto ops = getOperands();
  Type flagType = getFlagOperand().getType();
  Type type = getDataOperands().front().getType();
  p << " " << ops.front() << " : ";
  p.printType(flagType);
  p << " [";
  for (size_t i = 1; i < ops.size(); i++) {
    p.printOperand(ops[i]);
    if (i == 1)
      p << ", ";
  }
  p << "] ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << type;
}

ParseResult BsfaOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand flagOperand;
  Type selectType, dataType;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  if (failed(parseSelLikeOp(parser, flagOperand, dataType, selectType,
                            allOperands, result)))
    return failure();

  allOperands.insert(allOperands.begin(), flagOperand);

  // assign the same type to all operands
  SmallVector<Type, 3> dataOperandsTypes;
  dataOperandsTypes.assign(2, dataType);
  dataOperandsTypes.insert(dataOperandsTypes.begin(), selectType);

  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  result.addTypes(dataType);

  if (parser.resolveOperands(allOperands, ArrayRef<Type>(dataOperandsTypes),
                             allOperandLoc, result.operands))
    return failure();

  return success();
}

void BsfaOp::print(OpAsmPrinter &p) {
  auto ops = getOperands();
  Type flagType = getFlagOperand().getType();
  Type type = getDataOperands().front().getType();
  p << " " << ops.front() << " : ";
  p.printType(flagType);
  p << " [";
  for (size_t i = 1; i < ops.size(); i++) {
    p.printOperand(ops[i]);
    if (i == 1)
      p << ", ";
  }
  p << "] ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << type;
}

ParseResult LwiOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 1> allOperands;
  OpAsmParser::UnresolvedOperand addressOperand, dataResult;
  Type addrType, dataType;
  SmallVector<Type, 1> dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addressOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(addrType) || parser.parseArrow() ||
      parser.parseType(dataType))
    return failure();

  allOperands.push_back(addressOperand);
  dataOperandsTypes.assign(1, addrType);
  result.addTypes(dataType);
  if (parser.resolveOperands(allOperands, ArrayRef<Type>(dataOperandsTypes),
                             allOperandLoc, result.operands))
    return failure();
  return success();
}

void LwiOp::print(OpAsmPrinter &p) {
  Type addrType = getAddressOperand().getType();
  Type resType = getResult().getType();
  p << " " << getAddressOperand();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << addrType << "->" << resType;
}

ParseResult SwiOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  OpAsmParser::UnresolvedOperand dataOperand, addressOperand;

  SmallVector<Type, 2> dataOperandsTypes;
  Type dataType, addrType;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addressOperand) || parser.parseComma() ||
      parser.parseOperand(dataOperand) || parser.parseColon() ||
      parser.parseType(dataType) || parser.parseComma() ||
      parser.parseType(addrType) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  dataOperandsTypes.push_back(dataType);
  dataOperandsTypes.push_back(addrType);
  if (parser.resolveOperands(
          SmallVector<OpAsmParser::UnresolvedOperand, 4>{addressOperand,
                                                         dataOperand},
          ArrayRef<Type>(dataOperandsTypes), allOperandLoc, result.operands))
    return failure();
  return success();
}

void SwiOp::print(OpAsmPrinter &p) {
  Type type = getDataOperand().getType();
  p << " " << getDataOperand() << ", " << getAddressOperand();
  p << " : " << type << ", " << getAddressOperand().getType();
  p.printOptionalAttrDict((*this)->getAttrs());
}

#define GET_ATTRDEF_CLASSES
#include "compigra/CgraInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "compigra/Cgra.cpp.inc"
