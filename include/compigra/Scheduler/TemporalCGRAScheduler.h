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
#include "compigra/Support/Utils.h"

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
  OpBuilder builder;

  std::map<Block *, int> blockStartT;
  std::map<Block *, int> blockEndT;

public:
  void printBlockLiveValue(std::string fileName);

  int getBlockStartT(Block *block) { return blockStartT[block]; }

  void setBlockExecutionTime(Block *block, int timeStart) {
    blockStartT[block] = timeStart;
  }

  // Read the schedule result from the written file, usually avoid to call
  // `createSchedulerAndSolve` function for runtime efficiency.
  LogicalResult readScheduleResult(const std::string fileName);

  void setMaxLivePath(unsigned maxLivePath) { this->maxLivePath = maxLivePath; }

  void setReserveMem(unsigned reserveMem) { this->reserveMem = reserveMem; }

  // The schedule result comes from external scheduler, which does not support
  // DFG split for the blocks.
  void blockBBSchedule(const std::map<Operation *, ScheduleUnit> res);

private:
  // ======================== Liveness Data Structures =======================
  // Corresponding livein and liveout values of each block
  std::map<Block *, SetVector<Value>> liveIns;
  std::map<Block *, SetVector<Value>> liveOuts;

  // All live value and its located PE.
  liveVec liveValAndPEs;
  // Blocked Basic Blocks which does not allow DFG split
  std::set<Block *> blockedBBs;

  // Start address of the memory for the evicted value
  unsigned reserveMem = 0;

  // ====================== Liveness Analysis Functions ======================
  /// Get the livein and liveout values for each block.
  void computeLiveValue();

  void makeScheduleSeq();

  /// Rules to determine whether the value is internal or external live value.
  unsigned maxLivePath = 5;

  bool isExternalLive(Value val);

  /// Get the internal and external livein and liveout values for each block
  /// based on the rules defined in isExternalLive.
  liveVec getExternalLiveIn(Block *block);
  liveVec getInternalLiveIn(Block *block);
  liveVec getExternalLiveOut(Block *block);
  liveVec getInternalLiveOut(Block *block);

  // ===================== Result Read & Write Functions ======================
  /// Get the solution for one block
  std::map<Operation *, ScheduleUnit> getBlockSubSolution(Block *block);

  /// Save the ILP model result for corresponding operations.
  void saveSubILPModelResult(const std::map<Operation *, ScheduleUnitBB> res);

  void storeLocalResult(const liveVec localVec);

  /// Read the live value placement result to liveValInterPlaces and
  /// liveValExterPlaces.
  void writeLiveOutResult(const liveVec liveOutExter,
                          const liveVec liveOutInter, const liveVec liveInExter,
                          const liveVec liveInInter);

  /// Each basic block has its own ILP model to schedule the operations. which
  /// is assumed to start from T = 0. To fullfill the temporal spatial schedule,
  /// the basic block start time should be placed to ensure the basic block
  /// execution does not overlap.
  /// The function adopts sequential placement of the blocks and writes the
  /// final temporal spatial schedule to the file.
  void calculateTemporalSpatialSchedule(const std::string fileName);

  // ================ ILP Model Failure Handling Data Structures ===============
  // stack to store the value evicted to memory, the first element is the
  // address, the second element is the corresponding value.
  std::vector<std::pair<unsigned, Value>> memStack;

  // ================== ILP Model Failure Handling Functions ==================
  /// Insert an mov operation to extend the value propagation range.
  void insertMovOp(Value origVal, Operation *user);

  /// Remove all new inserted mov operations.
  void rollBackMovOp(Value failVal, int maxIter);

  /// Insert load and store operations to split the producer and consumer.
  LogicalResult splitDFGWithLSOps(Value saveVal, Operation *failUser = nullptr,
                                  unsigned memLoc = UINT_MAX,
                                  bool processCntPhi = false, bool load = true,
                                  bool store = true);

  /// Insert load and store operations to split the producer and consumer which
  /// are in the same basic blocks.
  void insertInternalLSOps(Operation *srcOp, Operation *dstOp);

  /// Insert a load operation from `addr`. This load operation is placed after
  /// refOp
  cgra::LwiOp insertLoadOp(Operation *refOp, unsigned addr, Value origVal,
                           unsigned opIndex = -1);

  /// If the block has been scheduled, place the lwi/swi operation to the
  /// block's original schedule result without rerun its ILP model.

  // block: the block to place the lwi/swi operation
  // refOp and opIndex specify the value to be replaced by the lwi operation
  void placeLwiOpToBlock(Block *block, Operation *refOp, unsigned opIndex,
                         cgra::LwiOp lwiOp);

  LogicalResult placeLwiOpToBlock(Block *block, BlockArgument arg,
                                  cgra::LwiOp lwiOp);

  void placeSwiOpToBlock(Block *block, cgra::SwiOp swiOp);

  // ======================= Schedule Sequence of BBs =======================
  // Sequence of blocks to be scheduled
  std::vector<Block *> scheduleSeq;
  // Index of the current block in the scheduleSeq
  unsigned scheduleIdx = 0;

public:
  std::map<Operation *, Instruction> knownRes;
};
} // namespace compigra

#endif // TEMPORAL_CGRA_SCHEDULER_H
