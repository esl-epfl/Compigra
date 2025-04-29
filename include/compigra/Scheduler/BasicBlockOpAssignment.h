//===- BasicBlockOpAssignment.h - Declares the class/functions to place
// operations of a basic block *- C++-* ----------------------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares class for BasicBlockOpAsisgnment functions.
//
//===----------------------------------------------------------------------===//

#ifndef BASIC_BLOCK_OP_ASSIGNMENT_H
#define BASIC_BLOCK_OP_ASSIGNMENT_H

#include "compigra/Scheduler/KernelSchedule.h"
#include "compigra/Support/Utils.h"

using namespace mlir;

namespace compigra {
enum ScheduleStrategy {
  // The schedule strategy is used to determine the order of the operations
  // inside a basic block.
  // ASAP schedules operations as soon as possible(when all of its producers
  // are scheduled).
  // ALAP schedules them as late as possible(scheduled until its consumer
  // require its result).
  // DYNAMIC determines the operation scheduling order during the compilation
  // time.
  ASAP = 0,
  ALAP = 1,
  DYNAMIC = 2
};

enum RegAttr { NK = -1, IN = 0, EX = 1, IE = 2 };

/// Describes the CGRA attributes through the number of rows, columns and the
/// internal registers.
struct GridAttribute {
  unsigned nRow;
  unsigned nCol;
  unsigned maxReg;
};

/// Describes the spatial placement of a value in the CGRA, which includes the
/// pe and the register attribute.
struct ValuePlacement {
  Value val;
  unsigned pe;
  RegAttr regAttr;
};

struct PERegUse {
  int inNum;
  bool exAvail;
};

// class BasicBlockOpAsisgnment {
// public:
//   BasicBlockOpAsisgnment(unsigned maxReg, unsigned nRow, unsigned nCol,
//                          Block *block, unsigned bbId)
//       : maxReg(maxReg), nRow(nRow), nCol(nCol), block(block), bbId(bbId),
//         builder(builder) {}

//   void searchCriticalPath();

// private:
//   unsigned maxReg;
//   unsigned nRow;
//   unsigned nCol;
//   // Interface for the the global schduler if the ILP model does not have
//   // solution
//   unsigned storeAddr;
//   Value spill = nullptr;
//   Operation *failUser = nullptr;
//   OpBuilder builder;

//   Block *block;
//   unsigned bbId;
// };

void mappingBBdataflowToCGRA(
    Block *block, std::map<Block *, SetVector<Value>> &liveIns,
    std::map<Block *, SetVector<Value>> &liveOuts,
    std::map<Operation *, ScheduleUnit> &subSolution,
    std::vector<ValuePlacement> &initGraph,
    std::vector<ValuePlacement> &finiGraph, GridAttribute &attr,
    ScheduleStrategy strategy = ScheduleStrategy::ASAP);
} // namespace compigra

#endif
