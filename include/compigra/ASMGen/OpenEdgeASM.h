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

/// Struct to hold the data for each instruction
#ifdef HAVE_GUROBI
// class BasicBlockScheduler {
// public:
//   BasicBlockScheduler(unsigned bbId) : bbId(bbId) {}
//   void initVariables(mlir::Block::OpListType &ops, GRBModel &model);
//   void initObjective();
//   /// existedOps contains the existed schedule results for a range of
//   /// operations, if the operation is connected with the existed schedule,
//   /// its  must be written into the constraints
//   void initConstraints(const std::map<Operation *, ScheduleUnit> existedOps);

// protected:
//   std::map<Operation *, GRBVar> timeVarMap;
//   std::map<Operation *, GRBVar> peVarMap;
//   unsigned bbId = 0;
// };
void initVariables(unsigned bbId, mlir::Block::OpListType &ops, GRBModel &model,
                   std::map<Operation *, GRBVar> &timeVarMap,
                   std::map<Operation *, GRBVar> &peVarMap);
void initConstraints(GRBModel &model,
                     const std::map<Operation *, ScheduleUnit> &existedOps,
                     std::map<Operation *, GRBVar> &timeVarMap,
                     std::map<Operation *, GRBVar> &peVarMap);
void initBBScheduler(unsigned bbId, mlir::Block::OpListType &ops);
#endif

class OpenEdgeASMGen {
public:
  // initialize the region and the maximum number of PEs
  OpenEdgeASMGen(Region &region, unsigned maxReg)
      : region(region), maxReg(maxReg) {}

  /// Get the execution time of the operation. If the operation is
  /// executed multiple times, return the first execution time. If the
  /// execution time is not set, return INT_MAX
  int getEarliestExecutionTime(Operation *op);

  /// Get the execution time using the result value.
  int getEarliestExecutionTime(Value val);

  /// Get the earliest execution time of operations in the block
  int getEarliestExecutionTime(Block *block);

  /// Assign the schedule results from SAT-MapIt printout to the operations
  LogicalResult assignSchedule(mlir::Block::OpListType &ops,
                               std::map<int, Instruction> instructions);

  const std::map<Operation *, ScheduleUnit> getCurrResult() { return result; }

protected:
  Region &region;
  unsigned maxReg;
  //   data structure to store the schedule results of operations
  std::map<Operation *, ScheduleUnit> result;
};

std::unique_ptr<mlir::Pass> createOpenEdgeASMGen(StringRef funcName = "",
                                                 StringRef mapResult = "");
} // end namespace compigra

#endif // OPEN_EDGE_ASM_H