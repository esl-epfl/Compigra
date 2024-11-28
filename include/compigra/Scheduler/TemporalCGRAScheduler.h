//===- TemporalCGRAScheduler.h - Declare the class/functions for 2D temporal-
// spatial schedule for temporal CGRAs *- C++-* --------------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares class for temporal CGRA schedule functions.
//
//===----------------------------------------------------------------------===//

#ifndef TEMPORAL_CGRA_SCHEDULER_H
#define TEMPORAL_CGRA_SCHEDULER_H

#include "compigra/ASMGen/InterferenceGraphCreation.h"
#include "compigra/Scheduler/BasicBlockILPModel.h"
#include "compigra/Scheduler/KernelSchedule.h"

using namespace mlir;

namespace compigra {
/// Class for temporal CGRA scheduler. TemporalCGRAScheduler decomposes the
/// mapping in the function region into basic blocks and schedules the operation
/// with ILP model built in BasicBlockILPScheduler.
class TemporalCGRAScheduler : public CGRAKernelScheduler<ScheduleUnit> {
public:
  TemporalCGRAScheduler(Region &region, unsigned maxReg, unsigned nRow,
                        unsigned nCol, OpBuilder builder)
      : CGRAKernelScheduler(maxReg, nRow, nCol), region(region),
        builder(builder) {}

  LogicalResult createSchedulerAndSolve() override;

private:
  Region &region;

  std::map<Block *, int> blockStartT;
  std::map<Block *, int> blockEndT;

public:
  void printBlockLiveValue(std::string fileName);

  int getBlockStartT(Block *block) { return blockStartT[block]; }

  void setBlockExecutionTime(Block *block, int timeStart, int timeEnd) {
    blockStartT[block] = timeStart;
    blockEndT[block] = timeEnd;
  }

  LogicalResult readScheduleResult(const std::string fileName);

private:
  // std::map<Operation *, ScheduleUnit> globalConstrs;
  std::vector<std::pair<unsigned, Value>> memStack;

  std::map<Block *, SetVector<Value>> liveIns;
  std::map<Block *, SetVector<Value>> liveOuts;

  std::map<Operation *, ScheduleUnit> getBlockSubSolution(Block *block);

  SetVector<std::pair<Operation *, Operation *>> opRAWs;

  liveVec liveValInterPlaces;
  liveVec liveValExterPlaces;

  void saveSubILPModelResult(const std::map<Operation *, ScheduleUnitBB> res);

  void computeLiveValue();

  void writeLiveOutResult(const liveVec liveOutExter,
                          const liveVec liveOutInter);

  liveVec getExternalLiveIn(Block *block);
  liveVec getInternalLiveIn(Block *block);
  liveVec getExternalLiveOut(Block *block);
  liveVec getInternalLiveOut(Block *block);

  OpBuilder builder;
  // TODO[@YW]: add the function for rollback the useless Mov
  void insertMovOp(Value origVal, Operation *user);

  void rollBackMovOp(Value failVal);
  cgra::LwiOp insertLoadOp(Operation *refOp, unsigned addr, Value origVal,
                           unsigned opIndex = -1);

  LogicalResult splitDFGWithLSOps(Value saveVal, Operation *failUser = nullptr,
                                  unsigned memLoc = UINT_MAX,
                                  bool processCntPhi = false);
  void insertInternalLSOps(Operation *srcOp, Operation *dstOp);
  void placeLwiOpToBlock(Block *block, Operation *refOp, unsigned opIndex,
                         cgra::LwiOp lwiOp);
  LogicalResult placeLwiOpToBlock(Block *block, BlockArgument arg,
                                  cgra::LwiOp lwiOp);
  void placeSwiOpToBlock(Block *block, cgra::SwiOp swiOp);

  // sequence of blocks to be scheduled
  std::vector<Block *> scheduleSeq;

  void calculateTemporalSpatialSchedule(const std::string fileName);

  void makeScheduleSeq();
  unsigned scheduleIdx = 0;

public:
  std::map<Operation *, Instruction> knownRes;
};
} // namespace compigra

#endif // TEMPORAL_CGRA_SCHEDULER_H
