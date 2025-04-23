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

// #include "compigra/Scheduler/ModuloScheduleAdapter.h"
#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
using namespace mlir;

#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

/// CGRA base scheduler class, where T is the struct that define the placement
/// of the operation.
namespace compigra {
// Schedule unit is a pair of time and PE, and the register to store the result
struct ScheduleUnit {
  int time;
  int pe;
  int reg;
};

/// Function to get the operation that is connected to the user operation via
/// branch(br/cond_br). This function only returns the first operation that uses
/// the block argument, under the assumption that an add zero operation is added
/// for each block argument.
Operation *getCntUseOpIndirectly(OpOperand &useOpr);

/// Function to get all the user operations including directly user and user use
/// the value propagated through branch (br/cond_br).
SmallPtrSet<Operation *, 4> getCntUserIndirectly(Value val);

/// Functions to get the operation that is connected to the value via branch.
/// If the value has definition, return the operation that defines the value.
/// Otherwise, return the producer operations that propagate to the value.
SmallVector<Operation *, 4> getCntDefOpIndirectly(Value val);

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

} // namespace compigra

#endif // KERNEL_SCHEDULE_H
