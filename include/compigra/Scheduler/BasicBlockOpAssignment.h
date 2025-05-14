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
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
void logMessage(const std::string &message, bool overwrite = false);

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

using placeunit = std::pair<unsigned, RegAttr>;

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

class BasicBlockOpAsisgnment {
public:
  BasicBlockOpAsisgnment(Block *block, unsigned maxReg, unsigned nRow,
                         unsigned nCol, OpBuilder builder)
      : builder(builder), curBlock(block) {
    attr = {nRow, nCol, maxReg};
  }

private:
  Block *curBlock;
  GridAttribute attr;
  OpBuilder builder;
  arith::ConstantOp zeroIntOp;
  arith::ConstantOp zeroFloatOp;

  std::vector<ValuePlacement> startEmbeddingGraph;
  std::vector<ValuePlacement> finiEmbeddingGraph;

  SetVector<Value> liveout;
  SetVector<Value> livein;

  std::map<Operation *, std::pair<int, int>> schedulePriority;
  std::map<Operation *, ScheduleUnit> solution;
  SetVector<Operation *> scheduledOps;

  // The operations and their corresponding spill operations
  std::vector<Value> spilledVals;

  void updateSchedulePriority(int timeSlot,
                              std::map<Block *, SetVector<Value>> liveIns,
                              std::map<Block *, SetVector<Value>> liveOuts);

  void updateCDFG(Block *scheduleBB, std::vector<ValuePlacement> initGraph,
                  std::vector<ValuePlacement> finiGraph);

  bool createRoutePath(Operation *failOp,
                       std::vector<ValuePlacement> &producers,
                       std::vector<unsigned> &movs,
                       std::vector<ValuePlacement> curGraph,
                       SmallVector<mlir::Operation *, 4> otherFailureOps = {},
                       unsigned threshold = 2);

  SmallVector<Operation *, 4>
  routeOperation(std::vector<ValuePlacement> producers,
                 std::vector<unsigned> movs, Operation *failedOp);

  double stepSA(int height, SmallVector<Operation *, 4> &schedulingOps,
                std::map<Operation *, ScheduleUnit> &tmpScheduleResult,
                std::vector<ValuePlacement> &tmpGraph,
                std::map<Operation *, std::vector<placeunit>> &existSpace,
                SetVector<Value> liveOut,
                std::vector<ValuePlacement> &finiGraph, GridAttribute attr,
                Operation *shuffleOp = nullptr);

  void setUpLiveness(std::map<Block *, SetVector<Value>> &liveIns,
                     std::map<Block *, SetVector<Value>> &liveOuts) {
    this->livein = liveIns[curBlock];
    this->liveout = liveOuts[curBlock];
  }

  std::vector<placeunit> searchOpPlacementSpace(
      Operation *op, std::vector<ValuePlacement> &curGraph,
      std::vector<ValuePlacement> &finiGraph,
      std::map<Operation *, std::pair<unsigned, RegAttr>> tmpResult);

  int placeOperations(int timeSlot, SmallVector<Operation *, 4> &schedulingOps,
                      std::map<Operation *, ScheduleUnit> &scheduleResult,
                      std::vector<ValuePlacement> &curGraph,
                      std::map<Operation *, std::vector<placeunit>> &space,
                      std::vector<ValuePlacement> &finiGraph,
                      Operation *shuffleOp = nullptr);

public:
  void setPrerequisiteToStartGraph(std::vector<ValuePlacement> initGraph) {
    this->startEmbeddingGraph = initGraph;
  }

  void setPrerequisiteToFinishGraph(std::vector<ValuePlacement> finiGraph) {
    this->finiEmbeddingGraph = finiGraph;
  }

  std::vector<ValuePlacement> getStartEmbeddingGraph() {
    return startEmbeddingGraph;
  }
  std::vector<ValuePlacement> getFiniEmbeddingGraph() {
    return finiEmbeddingGraph;
  }

  LogicalResult
  mappingBBdataflowToCGRA(std::map<Block *, SetVector<Value>> &liveIns,
                          std::map<Block *, SetVector<Value>> &liveOuts,
                          ScheduleStrategy strategy = ScheduleStrategy::ASAP);

  void setUpZeroOp(arith::ConstantOp zeroIntOp, arith::ConstantOp zeroFloatOp) {
    this->zeroIntOp = zeroIntOp;
    this->zeroFloatOp = zeroFloatOp;
  }
};

} // namespace compigra

#endif
