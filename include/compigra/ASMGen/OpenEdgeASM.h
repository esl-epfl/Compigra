//===- OpenEdge.h - Declare the functions for gen OpenEdge ASM --*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines assembly generation functions for OpenEdge.
//
//===----------------------------------------------------------------------===//

#ifndef OPEN_EDGE_ASM_H
#define OPEN_EDGE_ASM_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

using namespace mlir;

namespace compigra {
#define GEN_PASS_DEF_OPENEDGEASMGEN
#define GEN_PASS_DECL_OPENEDGEASMGEN
#include "compigra/ASMGen/Passes.h.inc"

// Schedule unit is a pair of time and PE, and the register to store the result
struct ScheduleUnit {
  int time;
  int pe;
  int reg;
};

class OpenEdgeASMGen {
public:
  // initialize the region and the maximum number of PEs
  OpenEdgeASMGen(Region &region, unsigned maxReg, unsigned grid)
      : region(region), maxReg(maxReg), nRow(grid), nCol(grid) {}

  /// Get the execution time of the operation. If the operation is
  /// executed multiple times, return the first execution time. If the
  /// execution time is not set, return INT_MAX
  int getEarliestExecutionTime(Operation *op);

  /// Get the execution time using the result value.
  int getEarliestExecutionTime(Value val);

  /// Get the earliest execution time of operations in the block
  int getEarliestExecutionTime(Block *block);

  int getConnectedBlock(int block, std::string direction);

  /// Assign the schedule results from SAT-MapIt printout to the operations
  LogicalResult assignSchedule(mlir::Block::OpListType &ops,
                               std::map<int, Instruction> instructions);

  void writeOpResult(Operation *op, int time, int pe, int reg) {
    ScheduleUnit unit = {time, pe, reg};
    solution[op] = unit;
  }

  const std::map<Operation *, ScheduleUnit> getCurrSolution() {
    return solution;
  }

  /// Struct to hold the data for each instruction
#ifdef HAVE_GUROBI
  LogicalResult createSchedulerAndSolve();

  void initObjectiveFunction(GRBModel &model, GRBVar &funcStartT,
                             GRBVar &funcEndT,
                             std::map<Operation *, GRBVar> &timeOpVar,
                             std::map<Block *, GRBVar> &timeBlkEntry,
                             std::map<Block *, GRBVar> &timeBlkExit);
  void initVariables(GRBModel &model, std::map<Block *, GRBVar> &timeBlkEntry,
                     std::map<Block *, GRBVar> &timeBlkExit,
                     std::map<Operation *, GRBVar> &timeOpVar,
                     std::map<Operation *, GRBVar> &spaceOpVar);
  void initKnownSchedule(GRBModel &model,
                         std::map<Operation *, GRBVar> &timeOpVar,
                         std::map<Operation *, GRBVar> &spaceOpVar);
  void initOpTimeConstraints(GRBModel &model,
                             std::map<Operation *, GRBVar> &timeOpVar,
                             std::map<Block *, GRBVar> &timeBlkEntry,
                             std::map<Block *, GRBVar> &timeBlkExit);
  void initOpSpaceConstraints(GRBModel &model,
                              std::map<Operation *, GRBVar> &spaceOpVar);
  void initOpTimeSpaceConstraints(GRBModel &model,
                                  std::map<Operation *, GRBVar> &timeOpVar,
                                  std::map<Operation *, GRBVar> &spaceOpVar);

  // void printKownSchedule();
  std::map<Operation *, Instruction> knownRes;

#endif

protected:
  Region &region;
  unsigned maxReg;
  unsigned nRow, nCol;
  //   data structure to store the schedule results of operations

  std::map<Operation *, ScheduleUnit> solution;
};

std::unique_ptr<mlir::Pass> createOpenEdgeASMGen(StringRef funcName = "",
                                                 StringRef mapResult = "");
} // end namespace compigra

#endif // OPEN_EDGE_ASM_H