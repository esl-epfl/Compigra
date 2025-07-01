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

void computeLiveValue(Region &region,
                      std::map<Block *, SetVector<Value>> &liveIns,
                      std::map<Block *, SetVector<Value>> &liveOuts);

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

using placeunit = std::pair<int, RegAttr>;

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

class BasicBlockOpAssignment {
public:
  BasicBlockOpAssignment(Block *block, unsigned maxReg, unsigned nRow,
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

  /// The operations that are blocked from being pop out,
  // WRITE-ONLY by createRoutePath function
  SetVector<unsigned> blockedProdPEs;

  /// Initialize the embedding graph, where the key is the [time slot, PE], the
  /// value indicates the value placed in the graph and its register attribute.
  /// It is noticed that graph[int,int] = <nullptr, nullptr> which indicates the
  /// PE is occupied by an operation does not produce any value at the time
  /// slot.
  void
  initEmbeddingGraphWithLiveIn(std::map<Block *, SetVector<Value>> liveIns,
                               std::map<Block *, SetVector<Value>> liveOuts,
                               std::vector<ValuePlacement> &initGraph,
                               OpBuilder &builder, GridAttribute attr);

  Operation *createAtomicMovOp(Value val, bool replaceCurBlkUse,
                               bool customLoc);

  void updateSchedulePriority(int timeSlot,
                              std::map<Block *, SetVector<Value>> liveIns,
                              std::map<Block *, SetVector<Value>> liveOuts);

  void updateCDFG(Block *scheduleBB, std::vector<ValuePlacement> initGraph,
                  std::vector<ValuePlacement> finiGraph);

  /// Create route path to accommodate the failed operation.
  /// The function  returns
  /// 0: route path is created from its producers where the number of
  /// route steps for each producer is stored in `movs`.
  /// 1: route path is created from the failed operation itself for it to access
  /// its consumers. One step routing is employed.
  ///  -1: failed to create route path as routing does not solve the problem.
  int createRoutePath(Operation *failOp, std::vector<ValuePlacement> &producers,
                      std::vector<unsigned> &movs,
                      std::vector<ValuePlacement> curGraph,
                      std::vector<ValuePlacement> finiGraph,
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
                int shuffleOpIdx = -1);

  void setUpLiveness(std::map<Block *, SetVector<Value>> &liveIns,
                     std::map<Block *, SetVector<Value>> &liveOuts) {
    this->livein = liveIns[curBlock];
    this->liveout = liveOuts[curBlock];
  }

  std::vector<placeunit> searchOpPlacementSpace(
      Operation *op, std::vector<ValuePlacement> &curGraph,
      const std::vector<ValuePlacement> &finiGraph,
      std::map<Operation *, std::pair<unsigned, RegAttr>> tmpResult);

  int placeOperations(int timeSlot, SmallVector<Operation *, 4> &schedulingOps,
                      std::map<Operation *, ScheduleUnit> &scheduleResult,
                      std::vector<ValuePlacement> &curGraph,
                      std::map<Operation *, std::vector<placeunit>> &space,
                      std::vector<ValuePlacement> &finiGraph,
                      int shuffleOpIdx = -1);

  LogicalResult
  finiEmbeddingGraphWithLiveOut(std::vector<ValuePlacement> &finiGraph,
                                std::vector<ValuePlacement> &scheduleGraph,
                                OpBuilder &builder, GridAttribute attr);

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

  std::map<Operation *, ScheduleUnit> getSolution() { return solution; }
};

} // namespace compigra

#endif
