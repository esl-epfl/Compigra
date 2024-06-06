//===- PrintSatMapItDAG.h - print text file for SatMapIt --------*- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the text printout functions for Sat-MapIt code base.
//
//===----------------------------------------------------------------------===//

#ifndef PRINT_SAT_MAP_IT_DAG_H
#define PRINT_SAT_MAP_IT_DAG_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "llvm/ADT/SmallSet.h"
// #include "llvm/ADT/ilist.h"
// #include "llvm/Support/Format.h"

using namespace mlir;

namespace compigra {
namespace satmapit {
size_t getNodeIndex(Operation *op, SmallVector<Operation *> &nodes,
                    SmallVector<LLVM::ConstantOp> constants = {},
                    SmallVector<Operation *> liveIns = {});

class PrintSatMapItDAG {
public:
  // initialization
  PrintSatMapItDAG(SmallVector<Operation *> nodes,
                   SmallVector<LLVM::ConstantOp> constants,
                   SmallVector<Operation *> liveIns)
      : nodes(nodes), constants(constants), liveIns(liveIns) {}

  // print the DAG into multiple text files
  LogicalResult printDAG(std::string fileName);
  LogicalResult printNodes(std::string fileName);
  LogicalResult printConsts(std::string fileName);
  LogicalResult printEdges(std::string fileName);

private:
  SmallVector<Operation *> nodes;
  SmallVector<LLVM::ConstantOp> constants;
  SmallVector<Operation *> liveIns;

  llvm::DenseMap<llvm::StringRef, int> CgraInsts = {
      {"EXIT", 0},     {"add", 1},      {"sub", 2},      {"mul", 3},
      {"div", 4},      {"UADD", 5},     {"USUB", 6},     {"UMUL", 7},
      {"UDIV", 8},     {"shl", 9},      {"lshr", 10},    {"and", 11},
      {"or", 12},      {"xor", 13},     {"and", 14},     {"nor", 15},
      {"LXNOR", 16},   {"bsfa", 17},    {"bzfa", 43},    {"INA", 18},
      {"INB", 19},     {"FXP_ADD", 20}, {"FXP_SUB", 21}, {"FXP_MUL", 22},
      {"FXP_DIV", 23}, {"beq", 24},     {"bne", 25},     {"blt", 26},
      {"bge", 27},     {"lwd", 28},     {"lwi", 29},     {"LWIPI", 30},
      {"swd", 31},     {"swi", 32},     {"SWIPI", 33},   {"NOP", 34},
      {"merge", 40},   {"ble", 41},     {"bge", 42},     {"ashr", 44},
      {"mv", 45}};
};

} // namespace satmapit
} // namespace compigra

#endif // PRINT_SAT_MAP_IT_DAG_H