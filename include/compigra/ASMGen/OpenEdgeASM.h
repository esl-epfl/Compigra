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
#include "compigra/Scheduler/KernelSchedule.h"
#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

using namespace mlir;

namespace compigra {
#define GEN_PASS_DEF_OPENEDGEASMGEN
#define GEN_PASS_DECL_OPENEDGEASMGEN
#include "compigra/ASMGen/Passes.h.inc"

class OpenEdgeASMGen {
public:
  // initialize the region and the maximum number of PEs
  OpenEdgeASMGen(Region &region, unsigned maxReg, unsigned grid)
      : region(region), maxReg(maxReg), nRow(grid), nCol(grid) {}

  /// Get the execution time of the operation. If the operation is
  /// executed multiple times, return the first execution time. If the
  /// execution time is not set, return INT_MAX
  int getEarliestExecutionTime(Operation *op);

  /// Get the earliest execution time of the kernel execution
  int getKernelStart();

  /// Get the latest execution time of the kernel execution, which is the
  /// distance of last PC of the kernel to the first PC of the kernel. The loop
  /// execution time is not considered.
  int getKernelEnd();

  /// Get the earliest execution time of operations in the block
  int getEarliestExecutionTime(Block *block);

  const std::map<Operation *, ScheduleUnit> getCurrSolution() {
    return solution;
  }

  void setSolution(const std::map<Operation *, ScheduleUnit> sol) {
    solution = sol;
  }

  // Output the ASM code to the file, if gridLike is true, the output is in the
  // format of CGRA grid, otherwise, the output is put sequentially.
  void printKnownSchedule(bool GridLIke = false, int startPC = 0);

  // Schedule result with register allocation for ISA format adaptation
  std::map<Operation *, Instruction> instSolution;

  // The start PC of the kernel
  int startPC = 0;

  // set the start PC of the kernel
  void initBaseTime(int time) {
    startPC = time;
    baseTime = time - getKernelStart();
  }

protected:
  Region &region;
  unsigned maxReg;
  unsigned nRow, nCol;

  // The shedule result might not start from 0, the baseTime is the additional
  // time step to ensure the schedule kernel starts from startPC;
  int baseTime = 0;

  //   data structure to store the schedule results of operations
  std::map<Operation *, ScheduleUnit> solution;

public:
  /// Get all operations scheduled at a specific time
  std::map<int, Operation *> getOperationsAtTime(int time);
  /// Get all operations scheduled at a specific PE
  std::map<int, Operation *> getOperationsAtPE(int pe);

  /// Function to allocate registers for the operations within each PE. The
  /// allocation result does not the pre-allocated registers in solution and
  /// allocate registers for other operations.
  LogicalResult
  allocateRegisters(std::map<Operation *, Instruction> restriction = {});

private:
  /// Convert solution with register allocation result to knownRes which
  /// specifies the operand in corresponding register in its producing PE.
  LogicalResult convertToInstructionMap();

  // Print op to OpenEdge ISA format. If dropNeighbourBr is set, if branch
  // destination is the next PC below it, it is removed to be NOP;
  std::string printInstructionToISA(Operation *op, bool dropNeighbourBr = true);
};

std::unique_ptr<mlir::Pass> createOpenEdgeASMGen(StringRef funcName = "",
                                                 StringRef mapResult = "");
} // end namespace compigra

#endif // OPEN_EDGE_ASM_H