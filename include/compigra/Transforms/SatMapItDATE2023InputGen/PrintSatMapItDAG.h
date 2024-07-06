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
#include "unordered_set"

using namespace mlir;

namespace compigra {
// Struct to hold the data for each instruction
struct Instruction {
  std::string name;
  int time;
  int pe;
  // register to store the result, R0:0, R1:1,..., Rout: maxReg
  int Rout = -1;
  std::string opA = "Unknown";
  std::string opB = "Unknown";
  // int immediate;
};

namespace satmapit {
class PrintSatMapItDAG {
public:
  // initialization
  PrintSatMapItDAG(Operation *terminator, SmallVector<Operation *> nodes,
                   SmallVector<LLVM::ConstantOp> constants,
                   SmallVector<BlockArgument> &BlockArgs,
                   SmallVector<Value> &liveOutArgs)
      : terminator(terminator), nodes(nodes), constants(constants),
        BlockArgs(BlockArgs), liveOutArgs(liveOutArgs) {}

  PrintSatMapItDAG(Operation *terminator, SmallVector<Operation *> nodes)
      : terminator(terminator), nodes(nodes) {}

  // print the DAG into multiple text files
  LogicalResult printDAG(std::string fileName);
  LogicalResult printNodes(std::string fileName);
  LogicalResult printConsts(std::string fileName);
  LogicalResult printEdges(std::string fileName);
  LogicalResult printLiveIns(std::string fileName);
  LogicalResult printLiveOuts(std::string fileName);

  // init the BlockArgs and liveOuts according to the terminator
  LogicalResult init();
  void initLoopBlock() { loopBlock = terminator->getBlock(); }
  void initPredBlock() {
    for (auto pred : loopBlock->getPredecessors())
      if (pred != loopBlock) {
        initBlock = pred;
        break;
      }
  }

  // If the operation is constant, add it to constants; if the operation belongs
  // to initBlock, add it to liveIns; if the operation is propated to finiBlock,
  // add the corresponding operation finiBlock to liveOuts.
  void addNodes(Operation *op);
  int getNodeIndex(Operation *op);
  int getNodeIndex(Value val);

private:
  SmallVector<Operation *> nodes = {};
  Operation *terminator = nullptr;
  Block *loopBlock, *initBlock, *finiBlock;
  unsigned blockArg = 0;
  SmallVector<BlockArgument> BlockArgs = {};
  SmallVector<Value> liveOutArgs = {};

  // operation out of SAT-MapIt schedule block
  SmallVector<Operation *> liveIns = {};
  SmallVector<Operation *> liveOuts = {};
  SmallVector<LLVM::ConstantOp> constants = {};

  // Argument with corresponding definition operations
  using selectOps = SmallVector<Value, 2>;
  std::map<int, selectOps> argMaps;

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
      {"phi", 40},     {"ble", 41},     {"bge", 42},     {"ashr", 44},
      {"mv", 45}};
};

/// Parse the produced map and register allocation result produced by Sat-MapIt.
void parseLine(const std::string &line, std::map<int, Instruction> &instMap,
               const unsigned maxReg);

/// Parse the module schedule result which include the prolog, kernel and epilog
void parsePKE(const std::string &line, unsigned termId,
              std::vector<std::unordered_set<int>> &bbTimeMap,
              std::map<int, std::unordered_set<int>> &opTimeMap);

} // namespace satmapit
} // namespace compigra

#endif // PRINT_SAT_MAP_IT_DAG_H