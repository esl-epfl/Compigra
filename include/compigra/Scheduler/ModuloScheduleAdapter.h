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
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include <set>
#include <stack>
#include <unordered_set>

using namespace mlir;
namespace compigra {
/// Data structure to map the operation id to the operation
using mapId2Op = std::map<int, Operation *>;
///
using opWithId = std::pair<Operation *, int>;
/// Data structure to store the operations index, each set contains the index of
/// different interations.
using opIdInIter = std::vector<std::set<int>>;

/// The modulo scheduler might generate efficient schedule result by overlapping
/// loop with prolog and epilog that does not exist in current CFG. This
/// function adapts the CFG to the schedule result for further whole kernel
/// function scheduling.
class ModuloScheduleAdapter {
  enum loopStage { prolog, loop, epilog };

private:
  Block *getLoopBlock(Region &region);
  Block *getInitBlock(Block *loopBlk);

  Block *getCondBrFalseDest(Block *blk);

public:
  /// The modulo scheduling result is described by two maps: opTimeMap and
  /// bbTimeMap. The opTimeMap use time(sequentially in PC) as key, where values
  /// indicates the the index of the operations to be executed at that time. The
  /// bbTimeMap is a vector of basic blocks, where each basic block contains the
  /// opearations in the block specified by the opTimeMap.
  ModuloScheduleAdapter(Region &region, OpBuilder &builder,
                        Block::OpListType &loopOpList, unsigned II,
                        std::map<int, int> execTime,
                        const std::map<int, std::unordered_set<int>> opTimeMap,
                        const std::vector<std::unordered_set<int>> bbTimeMap)
      : region(region), builder(builder), II(II), execTime(execTime),
        opTimeMap(opTimeMap), bbTimeMap(bbTimeMap), loopOpList(loopOpList) {
    // Get related basic blocks
    templateBlock = getLoopBlock(region);
    loopOpNum = loopOpList.size();
    loopFalseBlk = getCondBrFalseDest(templateBlock);
    initBlock = getInitBlock(templateBlock);
    if (loopFalseBlk->hasNoSuccessors())
      finiBlock = loopFalseBlk;
    else
      finiBlock = loopFalseBlk->getSuccessor(0);
  }

private:
  Region &region;
  OpBuilder &builder;
  unsigned II = -1;
  std::map<int, int> execTime;
  std::map<int, std::unordered_set<int>> opTimeMap;
  std::vector<std::unordered_set<int>> bbTimeMap;
  Block::OpListType &loopOpList;

  Block *templateBlock = nullptr;
  Block *loopFalseBlk = nullptr;
  Block *initBlock = nullptr;
  Block *finiBlock = nullptr;
  unsigned loopOpNum = 0;

public:
  /// Adapt the CFG with the modulo scheduling result.
  LogicalResult adaptCFGWithLoopMS();

  /// Support functions to adapt the CFG and create the DFG within new created
  /// basic blocks.
private:
  /// Remove the block arguments from the terminator of the term specified by
  /// argId, dest is the block to be connected.
  void removeBlockArgs(Operation *term, std::vector<unsigned> argId,
                       Block *dest = nullptr);
  /// add producer result as the branch argument for the dest block on the
  /// terminator(term) of predesesor block, insertBefore indicates whether
  /// insert the argument before the original arguments or not
  void addBranchArgument(Operation *term, Operation *producer, Block *dest,
                         bool insertBefore = false);
  /// Initialize the basic block with the operations in it specified by
  /// opSets. `opSets` contains multiple operation groups belong to
  /// different loop iterations, and the operations within the same group is
  /// stored in the set. `preGenOps` stores the operations from last
  /// iteration which would be used as the operands in the current
  /// iteration. `isKernel` indicates whether the block is the kernel block.
  /// The kernel could take operators from prolog and the loop kernel block
  /// itself, which must be processed with block arguments.
  LogicalResult initDFGBB(Block *blk, std::vector<std::set<int>> &opSets,
                          mapId2Op &preGenOps, bool isKernel = false,
                          bool exit = false);
  ;

  /// Generate operations in a basic that if the loop is terminated in the
  /// prolog phase.
  /// `existOps` contains the operations have been generated in the prolog, so
  /// the exit block should contains the complement of the operations w.r.t the
  /// fullSet(loopOpList).
  Block *createExitBlock(cgra::ConditionalBranchOp condBr,
                         std::vector<std::set<int>> &existOps,
                         mapId2Op &preGenOps,
                         cgra::ConditionalBranchOp loopTerm = nullptr);

  /// Remove the original loop block (templateblock) and its operations
  void removeTempletBlock();

  /// Replace liveOut arguments from original loop(templateBlock) to
  /// corresponding value in the CFG.
  // `insertOpsList` stores the operations which CFG could have multiple path to
  // propagate to the fini block,
  LogicalResult replaceLiveOutWithNewPath(std::vector<mapId2Op> insertOpsList);

  /// Remove the useless block arguments in the CFG for the prolog stage which
  /// takes the operands from initBlock unconditionally.
  void removeUselessBlockArg();

private:
  // Data structure to store the operations Id in the block
  // std::map<Block *, std::vector<opWithId>> enterBlks;
  // std::map<Block *, std::vector<opWithId>> exitBlks;
  std::map<Block *, std::vector<opWithId>> blkOpIds;

  LogicalResult searchInitArg(std::stack<Block *> &blkSt, int id,
                              Block *curBlk);

public:
  /// Return created DFG in the CFG, this is the
  /// aggregation of enterDFGs and exitDFGs
  std::vector<std::vector<opWithId>> getCreatedDFGs() {
    std::vector<std::vector<opWithId>> createdDFGs;
    createdDFGs.insert(createdDFGs.end(), enterDFGs.begin(), enterDFGs.end());
    createdDFGs.insert(createdDFGs.end(), exitDFGs.begin(), exitDFGs.end());
    return createdDFGs;
  }

public:
  /// Data structure to store the enterDFGs(prolog, loop,epilog) and
  /// exitDFGs(complement of prolog)
  std::vector<std::vector<opWithId>> enterDFGs = {};
  std::vector<std::vector<opWithId>> exitDFGs = {};

private:
  /// Store the operations in preBlocks to enterDFGs and postBlocks to exitDFGs
  LogicalResult saveDFGs(SmallVector<Block *> preBlocks,
                         SmallVector<Block *> postBlocks);
};

} // namespace compigra

#endif // MODULO_SCHEDULE_ADAPTER_H