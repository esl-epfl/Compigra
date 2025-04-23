//===- ModuloScheduleAdapter.h - Declare adapter for MS ---------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Modulo schduling can change both DFG and CFG of the program. This adapter
// declares functions that rewrite the IR to match the schedule result.
//
//===----------------------------------------------------------------------===//

#ifndef MODULO_SCHEDULE_ADAPTER_H
#define MODULO_SCHEDULE_ADAPTER_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraOps.h"
#include "compigra/Scheduler/BasicBlockOpAssignment.h"
#include "compigra/Scheduler/KernelSchedule.h"
#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include <set>
#include <stack>
#include <unordered_set>

using namespace mlir;

namespace compigra {
unsigned getOpId(Block::OpListType &opList, Operation *search);

/// Data structure to map the operation id to the operation

using opWithId = std::pair<Operation *, int>;
/// Data structure to store the operations index, each set contains the index of
/// different interations.
// using opIdInIter = std::vector<std::set<int>>;

/// The modulo scheduler might generate efficient schedule result by overlapping
/// loop with prolog and epilog that does not exist in current CFG. This
/// function adapts the CFG to the schedule result for further whole kernel
/// function scheduling.
class ModuloScheduleAdapter {
  enum loopStage { prolog, loop, epilog };

private:
  // Block *getLoopBlock(Region &region);
  Block *getInitBlock(Block *loopBlk);

  Block *getCondBrFalseDest(Block *blk);

public:
  /// The modulo scheduling adapter adapts a loop in the templateBlock to be
  /// overlapped, initiated another iteration by the initial interval(II).
  /// The adaptation is driven by the MS result, which is described by two maps:
  /// opTimeMap, and timeSlotsOfBBs. The opTimeMap use time(sequentially in PC)
  /// as key, where values indicates at what time the operations to be execute.
  /// The timeSlotsOfBBs is a vector of basic blocks, where each basic block
  /// contains the operations in the block indicated by their index in
  /// templateBlock.
  ModuloScheduleAdapter(Region &region, Block *templateBlock,
                        OpBuilder &builder, unsigned II,
                        std::map<int, int> execTime,
                        const std::map<int, std::set<int>> opTimeMap,
                        const std::vector<std::set<int>> timeSlotsOfBBs)
      : region(region), templateBlock(templateBlock), builder(builder), II(II),
        execTime(execTime), opTimeMap(opTimeMap),
        timeSlotsOfBBs(timeSlotsOfBBs),
        loopOpList(templateBlock->getOperations()) {}

  // Initialize the adapter, it would return success if the block is adaptable
  // according to the set up.
  LogicalResult init();

private:
  Region &region;
  OpBuilder &builder;
  int II = -1;
  std::map<int, int> execTime;
  // key: PC, value: set of operation Ids
  std::map<int, std::set<int>> opTimeMap;
  // Vector of basic blocks, each block contains the operations Id in the block
  std::vector<std::set<int>> timeSlotsOfBBs;
  Block::OpListType &loopOpList;

  // Blocks to describe the CFG in the region, where templateBlock is the
  // original loop block that will be replaced, initBlock and finiBlock are the
  // connected to the loop block to initiate and finalize the loop.
  Block *templateBlock = nullptr;
  // startBlock is the new created block to start the loop execution.
  Block *startBlock = nullptr;
  SmallVector<Block *, 4> newBlocks;
  // Number of operations in the templateBlock
  unsigned loopOpNum = 0;
  // The loop block id in the timeSlotsOfBBs
  int loopBlkId = -1;
  // The first iteration id of the terminator of the new created loop kernel
  // block.
  int loopIterId = -1;

  /// Compare flag for loop continuation
  int loopCmpOprId1, loopCmpOprId2;
  cgra::CondBrPredicate cmpFlag;

public:
  // The initialization and the finalization block for the loop kernel
  Block *initBlock = nullptr;
  Block *finiBlock = nullptr;
  /// Adapt the CFG with the modulo scheduling result.
  LogicalResult adaptCFGWithLoopMS();

  SmallVector<Block *, 4> getNewBlocks() { return newBlocks; }

  SmallVector<Block *, 4> getPrologAndKernelBlocks();

  // Write the schedule result of the modulo scheduler to solutions, the
  // execution time is calculated by its basic block start time and the
  // operation's execution time in the block, the PE is the operation PE, and
  // the register is only assigned if Rout==maxReg, while the internal register
  // allocation from the modulo scheduler is dropped because it is useless for
  // the global register allocation.
  LogicalResult
  assignScheduleResult(const std::map<int, Instruction> instructions,
                       int maxReg, int maxPE);

  std::map<Operation *, compigra::ScheduleUnit> getSolutions() {
    return solution;
  }

  std::map<Operation *, compigra::ScheduleUnit> getPrologAndKernelSolutions();

  std::vector<std::pair<Value, int>> getPrerequisites() {
    return prerequisites;
  }

  /// Support functions to adapt the CFG and create the DFG within new created
  /// basic blocks.
private:
  // Data structure to store the operations Id in the block
  std::map<int, std::map<int, Operation *>> prologOps;
  std::map<Block *, std::map<int, std::map<int, Operation *>>> epilogOpsBB;

  std::map<Operation *, compigra::ScheduleUnit> solution;

  std::vector<std::pair<Value, int>> prerequisites;

  LogicalResult
  updateOperands(int bbId, int iterId, int opId, Operation *adaptOp,
                 const std::map<int, std::map<int, Operation *>> prologOpMap,
                 const std::map<int, std::map<int, Operation *>> epilogOpMap,
                 std::vector<std::pair<int, int>> &propagatedOps,
                 bool isEpilog = false);

  // Create operations in bbId's block(prolog or kernel) of the modulo
  // scheduled loop. `bbTimeId` is PC specified for the block, `propOpSets` is
  // the set of operations that need to be propagated to the next block.
  LogicalResult
  initOperationsInBB(Block *blk, int bbId, const std::set<int> bbTimeId,
                     std::vector<std::pair<int, int>> &propOpSets);

  // Complete the epilog for the end of execution of bbId's block. `termIter`
  // is the iteration id which should be executed. All operations belong to
  // the later iterations should be dropped.
  LogicalResult completeUnexecutedOperationsInBB(
      Block *blk, int bbId, unsigned termIter,
      const std::map<int, std::map<int, Operation *>> executedOpMap,
      std::vector<std::pair<int, int>> &propOpSets);

  /// Remove the block arguments from the terminator of the term specified by
  /// argId, dest is the block to be connected.
  void removeBlockArgs(Operation *term, std::vector<unsigned> argId,
                       Block *dest = nullptr);

  /// Remove the original loop block (templateblock) and its operations
  void removeTempletBlock();

  /// Remove the useless block arguments in the CFG for the prolog stage which
  /// takes the operands from initBlock unconditionally.
  void removeUselessBlockArg();
};

} // namespace compigra

#endif // MODULO_SCHEDULE_ADAPTER_H