//===- KernelSchedule.h - Declare the class for ops schedule ----*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines class for schedule functions.
//
//===----------------------------------------------------------------------===//

#ifndef KERNEL_SCHEDULE_H
#define KERNEL_SCHEDULE_H

#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
using namespace mlir;

#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

/// CGRA base scheduler class, where T is the struct that define the placement
/// of the operation.
namespace compigra {

/// Functions to get the operation that is connected to the user operation via
/// branch.
Operation *getCntOpIndirectly(Operation *userOp, Operation *op);

/// Functions to get the operation that is connected to the value via branch.
/// If the value has definition, return the operation that defines the value.
/// Otherwise, return the producer operations that propagate to the value.
SmallVector<Operation *, 4> getCntOpIndirectly(Value val);

template <typename T> class CGRAKernelScheduler {
public:
  CGRAKernelScheduler(unsigned maxReg, unsigned nRow, unsigned nCol)
      : maxReg(maxReg), nRow(nRow), nCol(nCol) {}

  virtual LogicalResult createSchedulerAndSolve() { return success(); };

  /// Get the schedule result
  std::map<Operation *, T> getSolution() { return solution; }

  /// Get the schedule result for the operation
  T getSolution(Operation *op) { return solution[op]; }

protected:
  // maximum number of registers in the PE;
  unsigned maxReg;

  // number of rows and columns in the CGRA
  unsigned nRow, nCol;

  std::map<Operation *, T> solution;
};

// Schedule unit is a pair of time and PE, and the register to store the result
struct ScheduleUnit {
  int time;
  int pe;
  int reg;
};

/// OpenEdge kernel scheduler class
class OpenEdgeKernelScheduler : public CGRAKernelScheduler<ScheduleUnit> {
public:
  /// Initialize the region which contains the operations and the maximum number
  /// of PEs
  OpenEdgeKernelScheduler(Region &region, unsigned maxReg, unsigned grid)
      : CGRAKernelScheduler<ScheduleUnit>(maxReg, grid, grid), region(region) {}

  LogicalResult createSchedulerAndSolve() override;

protected:
  Region &region;

public:
#ifdef HAVE_GUROBI

  void initObjectiveFunction(GRBModel &model, GRBVar &funcStartT,
                             GRBVar &funcEndT,
                             std::map<Operation *, GRBVar> &timeOpVar,
                             std::map<Block *, GRBVar> &timeBlkEntry,
                             std::map<Block *, GRBVar> &timeBlkExit,
                             GRBLinExpr &objExpr);
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
                              std::map<Operation *, GRBVar> &spaceOpVar,
                              std::map<Operation *, GRBVar> &timeOpVar);
  void initOpTimeSpaceConstraints(GRBModel &model,
                                  std::map<Operation *, GRBVar> &timeOpVar,
                                  std::map<Operation *, GRBVar> &spaceOpVar);
#endif

private:
  int getConnectedBlock(int block, std::string direction);
  void writeOpResult(Operation *op, int time, int pe, int reg) {
    ScheduleUnit unit = {time, pe, reg};
    solution[op] = unit;
  }

public:
  /// Assign the schedule results from SAT-MapIt printout to the operations
  void assignSchedule(mlir::Block::OpListType &ops,
                      std::map<int, Instruction> instructions);

  // void printKownSchedule();
  std::map<Operation *, Instruction> knownRes;
};
} // namespace compigra

#endif // KERNEL_SCHEDULE_H
