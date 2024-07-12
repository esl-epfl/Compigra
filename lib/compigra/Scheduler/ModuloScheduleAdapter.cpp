//===- ModuloScheduleAdapter.cpp - Declare adapter for MS -------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Modulo schduling can change both DFG and CFG of the program. This file
// implements functions that rewrite the IR to match the schedule result.
//
//===----------------------------------------------------------------------===//

#include "compigra/Scheduler/ModuloScheduleAdapter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace compigra;

/// Get the loop block of the region
Block *ModuloScheduleAdapter::getLoopBlock(Region &region) {
  for (auto &block : region)
    for (auto suc : block.getSuccessors())
      if (suc == &block)
        return &block;
  return nullptr;
}

/// Get the init block of the region
Block *ModuloScheduleAdapter::getInitBlock(Block *loopBlk) {
  for (auto pred : loopBlk->getPredecessors())
    if (pred != loopBlk)
      return pred;
  return nullptr;
}

/// the conditional branch block has two successors, for the loop the true
/// points to the entry of the loop by default, get the false block
Block *ModuloScheduleAdapter::getCondBrFalseDest(Block *blk) {
  for (auto succ : blk->getSuccessors())
    if (succ != blk)
      return succ;
  return nullptr;
}

/// Determine whether a  basic block is the loop kernel by counting the
/// operation Id is equal to the number of operations in the block.
static bool isLoopKernel(unsigned endId, const std::set<int> bb) {
  for (size_t i = 0; i < endId; i++)
    if (std::find(bb.begin(), bb.end(), i) == bb.end())
      return false;
  return true;
}

/// Get i'th op in opList
static Operation *getOp(Block::OpListType &opList, unsigned i) {
  // Check if the index is within the bounds of the container
  if (i < std::distance(opList.begin(), opList.end())) {
    auto it = opList.begin();
    std::advance(it, i);
    return &(*it);
  }
  return nullptr;
}

/// Get the operation index in the opList
static unsigned getOpId(Block::OpListType &opList, Operation *search) {
  for (auto [ind, op] : llvm::enumerate(opList))
    if (&op == search)
      return ind;
  return -1;
}

/// Return the difference set
static std::set<int> getDiffSet(std::set<int> set1, std::set<int> set2) {
  std::set<int> diffSet;
  for (auto val : set1)
    if (set2.find(val) == set2.end())
      diffSet.insert(val);
  return diffSet;
}

/// Return the union set specified by key in keys
static std::set<int> getUnionSet(std::map<int, std::unordered_set<int>> mapSet,
                                 std::unordered_set<int> keys) {
  // Get the index in ascending order
  std::set<int> unionSet;
  if (keys.empty())
    return unionSet;

  for (auto [key, vals] : mapSet) {
    if (keys.count(key))
      for (auto val : vals)
        unionSet.insert(val);
  }
  return unionSet;
}

/// Return the union set specified by key in keys
static std::set<int> getUnionSet(std::vector<std::set<int>> sets) {
  if (sets.empty())
    return {};
  if (sets.size() == 1)
    return sets[0];

  // Get the index in ascending order
  std::set<int> unionSet;
  for (auto set : sets)
    for (auto val : set)
      unionSet.insert(val);
  return unionSet;
}

/// In modulo scheduling, multiple operations might be executed in a same basic
/// block with different operantors. This function is used to get the
/// aggregation of the operation sets belongs to different loop iterations.
static std::vector<std::set<int>>
getOperationSet(std::map<int, std::unordered_set<int>> timeMap,
                std::unordered_set<int> keys,
                std::vector<std::set<int>> &prevSet, bool epilog = false) {

  if (keys.empty())
    return {};

  if (epilog) {
    std::vector<std::set<int>> opSets = {{}};
    for (auto [key, vals] : timeMap)
      if (keys.count(key))
        for (auto val : vals) {
          // if the value shows twice, it belongs to different loop iterations
          if (opSets.back().count(val) == 0)
            opSets.back().insert(val);
          else {
            opSets.push_back({});
            opSets.back().insert(val);
          }
        }

    // sort opSets according to its smallest element
    std::sort(opSets.begin(), opSets.end(),
              [](const std::set<int> &a, const std::set<int> &b) {
                return *a.begin() > *b.begin();
              });
    return opSets;
  }

  // if it is prolog or loop kernel, it always add new ops based on the previous
  // operation set
  std::vector<std::set<int>> opSets = prevSet;
  auto unionSet = getUnionSet(timeMap, keys);
  auto prevUnionSet = getUnionSet(prevSet);
  opSets.push_back(getDiffSet(unionSet, prevUnionSet));
  // sort opSets according to its smallest element
  std::sort(opSets.begin(), opSets.end(),
            [](const std::set<int> &a, const std::set<int> &b) {
              return *a.begin() > *b.begin();
            });
  return opSets;
}

/// Reverse the conditional branch flag if the true and false block is reversed
static void reverseCondBrFlag(cgra::ConditionalBranchOp condBr,
                              bool reverseBB = false) {
  switch (condBr.getPredicate()) {
  case cgra::CondBrPredicate::ne:
    condBr.setPredicate(cgra::CondBrPredicate::eq);
    break;
  case cgra::CondBrPredicate::eq:
    condBr.setPredicate(cgra::CondBrPredicate::ne);
    break;
  case cgra::CondBrPredicate::lt: {
    condBr.setPredicate(cgra::CondBrPredicate::ge);
    Value tmp = condBr.getOperand(0);
    // reverse the first operands order
    condBr.setOperand(0, condBr.getOperand(1));
    condBr.setOperand(1, tmp);
    break;
  }
  case cgra::CondBrPredicate::ge: {
    condBr.setPredicate(cgra::CondBrPredicate::lt);
    Value tmp = condBr.getOperand(0);
    // reverse the first operands order
    condBr.setOperand(0, condBr.getOperand(1));
    condBr.setOperand(1, tmp);
  }
  }
  if (!reverseBB)
    return;
  // reverse the true and false block
  auto tmp = condBr.getTrueDest();
  condBr.setTrueDest(condBr.getFalseDest());
  condBr.setFalseDest(tmp);
}

/// add producer result as the branch argument for the dest block on the
/// terminator(term) of predesesor block, insertBefore indicates whether insert
/// the argument before the original arguments or not
void ModuloScheduleAdapter::addBranchArgument(Operation *term,
                                              Operation *producer, Block *dest,
                                              bool insertBefore) {
  // need to create a new terminator to replace the old one
  builder.setInsertionPoint(term);
  if (auto br = dyn_cast<LLVM::BrOp>(term)) {
    SmallVector<Value> operands{br.getOperands().begin(),
                                br.getOperands().end()};
    if (insertBefore)
      operands.insert(operands.begin(), producer->getResult(0));
    else
      operands.push_back(producer->getResult(0));
    builder.create<LLVM::BrOp>(term->getLoc(), operands, dest);
    term->erase();
    return;
  }

  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(term)) {
    SmallVector<Value> trueOperands{condBr.getTrueDestOperands().begin(),
                                    condBr.getTrueDestOperands().end()};

    SmallVector<Value> falseOperands{condBr.getFalseDestOperands().begin(),
                                     condBr.getFalseDestOperands().end()};

    // if (producer->getResults().size() == 1) {
    //   llvm::errs() << "add branch argument for " << *term << "\n";
    // } else
    //   llvm::errs() << "invalid " << *producer << "\n";

    if (condBr.getTrueDest() == dest) {
      if (insertBefore)
        trueOperands.insert(trueOperands.begin(), producer->getResult(0));
      else
        trueOperands.push_back(producer->getResult(0));
      builder.create<cgra::ConditionalBranchOp>(
          term->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
          condBr.getOperand(1), dest, trueOperands, condBr.getFalseDest(),
          falseOperands);

    } else if (condBr.getFalseDest() == dest) {
      if (insertBefore)
        falseOperands.insert(falseOperands.begin(), producer->getResult(0));
      else
        falseOperands.push_back(producer->getResult(0));
      builder.create<cgra::ConditionalBranchOp>(
          term->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
          condBr.getOperand(1), condBr.getTrueDest(), trueOperands, dest,
          falseOperands);
    } else {
      llvm::errs() << "\nfailed on " << *term << "\n";
    }
    term->erase();
    return;
  }
}

/// Remove the block arguments from the terminator of the term specified by
/// argId, dest is the block to be connected.
void ModuloScheduleAdapter::removeBlockArgs(Operation *term,
                                            std::vector<unsigned> argId,
                                            Block *dest = nullptr) {
  builder.setInsertionPoint(term);
  if (auto br = dyn_cast<LLVM::BrOp>(term)) {
    SmallVector<Value> operands;
    for (auto [ind, opr] : llvm::enumerate(br->getOperands()))
      if (std::find(argId.begin(), argId.end(), ind) == argId.end())
        operands.push_back(opr);
    builder.create<LLVM::BrOp>(term->getLoc(), operands, br.getSuccessor());
    term->erase();
    return;
  }

  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(term)) {
    SmallVector<Value> trueOperands;
    for (auto [ind, opr] : llvm::enumerate(condBr.getTrueDestOperands()))
      if (std::find(argId.begin(), argId.end(), ind) == argId.end())
        trueOperands.push_back(opr);
    SmallVector<Value> falseOperands;
    for (auto [ind, opr] : llvm::enumerate(condBr.getFalseDestOperands()))
      if (std::find(argId.begin(), argId.end(), ind) == argId.end())
        falseOperands.push_back(opr);
    builder.create<cgra::ConditionalBranchOp>(
        term->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
        condBr.getOperand(1), condBr.getTrueDest(), trueOperands,
        condBr.getFalseDest(), falseOperands);
    term->erase();
    return;
  }
}

void ModuloScheduleAdapter::removeUselessBlockArg() {
  for (auto &block : region) {
    if (block.isEntryBlock())
      continue;
    if (block.getArguments().size() == 0 ||
        std::distance(block.getPredecessors().begin(),
                      block.getPredecessors().end()) > 1)
      continue;

    // the block has only one predecessor and have arguments
    // get the corresponding value in the predecessor
    auto prevTerm = (*block.getPredecessors().begin())->getTerminator();
    auto oprIndBase = 0;
    if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(prevTerm)) {
      // the block is the false dest
      oprIndBase = 2;
      if (condBr.getFalseDest() == &block)
        oprIndBase += condBr.getTrueDestOperands().size();
    }

    // find the corresponding value in the predecessor
    std::vector<unsigned> argId;
    unsigned oprInd = 0;
    for (auto arg : llvm::make_early_inc_range(block.getArguments())) {
      arg.replaceAllUsesWith(prevTerm->getOperand(oprIndBase + oprInd));
      argId.push_back(oprInd);
      oprInd++;
    }
    // remove all block arguments
    llvm::BitVector bitVec(argId.size(), true);
    block.eraseArguments(bitVec);
    removeBlockArgs(prevTerm, argId, &block);
  }
}

static int existBlockArgument(std::vector<int> argIds, int id) {
  for (auto [blkArg, opId] : llvm::enumerate(argIds)) {
    if (opId == id)
      return blkArg;
  }
  return -1;
}

/// Initialize the DFG within the blk with the operations in the opSet
LogicalResult
ModuloScheduleAdapter::initDFGBB(Block *blk, std::vector<std::set<int>> &opSets,
                                 mapId2Op &preGenOps, bool isKernel,
                                 bool exit) {
  auto &opList = templateBlock->getOperations();
  unsigned totalOpNum = opList.size();
  auto loopBlkArgs =
      dyn_cast<cgra::ConditionalBranchOp>(templateBlock->getTerminator())
          .getTrueDestOperands();
  mapId2Op curGenOps;
  std::vector<opWithId> createdDFG;

  std::vector<int> argIds;
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;

  Operation *lastOp = nullptr;

  for (auto [_, opSet] : llvm::enumerate(opSets))
    for (auto ind : opSet) {
      // Get the operation from the index
      auto op = getOp(opList, ind);

      if (lastOp)
        builder.setInsertionPointAfter(lastOp);
      else
        builder.setInsertionPointToStart(blk);

      Operation *repOp;

      if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(op)) {
        repOp = builder.create<cgra::ConditionalBranchOp>(
            lastOp->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
            condBr.getOperand(1), condBr.getTrueDest(), condBr.getFalseDest());
      } else
        repOp = builder.clone(*op);

      for (auto [oprId, opr] : llvm::enumerate(repOp->getOperands())) {
        if (auto blockArg = dyn_cast<BlockArgument>(opr)) {
          auto corOp = loopBlkArgs[blockArg.getArgNumber()].getDefiningOp();
          int corOpId = getOpId(opList, corOp);
          // use current generated operations
          if (curGenOps.count(corOpId) > 0 && preGenOps.count(corOpId) == 0) {
            repOp->setOperand(oprId, curGenOps[corOpId]->getResult(0));
            continue;
          }
          // insert block argument for this op
          // check whether opr exists in the block arguments
          // int argId = existBlockArgument(argIds, corOpId);
          // if (argId >= 0) {
          //   repOp->setOperand(oprId, blk->getArgument(argId));

          // } 
          else {
            blk->addArgument(opr.getType(),
                             blk->getOperations().front().getLoc());
            repOp->setOperand(oprId, blk->getArguments().back());
            argIds.push_back(getOpId(opList, corOp));
          }
          continue;
        } else if (opr.getDefiningOp()) {
          // Determine whether defined by operation produced in the loop
          auto defOp = opr.getDefiningOp();

          if (defOp->getBlock() == templateBlock) {
            unsigned opId = getOpId(opList, defOp);

            // use current generated operations
            if (curGenOps.count(opId) > 0) {
              auto corOp = curGenOps[opId];
              repOp->setOperand(oprId, corOp->getResult(0));
              continue;
            }

            // if the predecessor has generated the result, consume it
            if (preGenOps.count(opId) > 0) {
              auto corOp = preGenOps[opId];
              // if the current block is kernel, it has to take arguments from
              // the prolog or itself
              if (isKernel) {
                // int argId = existBlockArgument(argIds, opId);
                // if (argId >= 0) {
                //   repOp->setOperand(oprId, blk->getArgument(argId));
                //   continue;
                // }
                argTypes.push_back(corOp->getResult(0).getType());
                blk->addArgument(corOp->getResult(0).getType(),
                                 blk->getOperations().front().getLoc());
                repOp->setOperand(oprId, blk->getArguments().back());
                // the loop takes preGenOps[opId] and curGenOps[opId] as
                // operands
                argIds.push_back(opId);
              } else {
                repOp->setOperand(oprId, corOp->getResult(0));
              }
              continue;
            }
            return failure();
          }
        }

        // set the same operand with op
        repOp->setOperand(oprId, opr.getDefiningOp()->getResult(0));
      }

      curGenOps[ind] = repOp;
      createdDFG.push_back({repOp, ind});
      lastOp = repOp;
    }

  // add branch arguments for the kernel block
  // the loop block must have a conditional terminator
  if (isKernel) {
    if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(blk->getTerminator()))
      condBr.setTrueDest(blk);
    else
      return failure();
  }

  // add block arguments for all predecessors
  for (auto pred :
       llvm::make_early_inc_range(lastOp->getBlock()->getPredecessors()))
    for (auto argId : argIds) {
      if (pred == blk)
        addBranchArgument(pred->getTerminator(), curGenOps[argId], blk);
      else {
        if (preGenOps.count(argId) > 0)
          addBranchArgument(pred->getTerminator(), preGenOps[argId], blk);
      }
    }

  // update preGenOps with curGenOps
  for (auto [ind, op] : curGenOps)
    preGenOps[ind] = op;

  // store curGenOps
  if (exit)
    exitDFGs.push_back(createdDFG);
  else
    enterDFGs.push_back(createdDFG);

  // createdDFGs.push_back(curGenOps);
  return success();
}

/// Generate operations in a basic that if the loop is terminated in the prolog
/// phase.
Block *
ModuloScheduleAdapter::createExitBlock(cgra::ConditionalBranchOp condBr,
                                       std::vector<std::set<int>> &existOps,
                                       mapId2Op &preGenOps, bool isKernel) {
  std::vector<mlir::Location> loc1(condBr.getFalseDestOperands().size(),
                                   condBr->getLoc());

  // full Set is the operation index from 0 to
  // templateBlk->getOperations().size() - 1
  std::set<int> fullSet;
  for (int i = 0; i < loopOpNum; ++i) {
    fullSet.insert(i);
  }

  auto connBB = builder.createBlock(
      finiBlock, condBr.getFalseDestOperands().getTypes(), loc1);
  condBr.setTrueDest(connBB);
  Location loc = condBr->getLoc();
  std::vector<std::set<int>> exitOps;
  if (!isKernel) {
    // get the diff of existedOps[] and the templateBlk
    for (int i = 0; i < existOps.size(); i++) {
      // get the union set from first to the end - i of the existedOps
      auto unionSet = getUnionSet(
          std::vector<std::set<int>>(existOps.begin() + i, existOps.end()));
      // get the difference between unionSet and full set;
      auto exitLoopOps = getDiffSet(fullSet, unionSet);
      if (!exitLoopOps.empty())
        exitOps.push_back(exitLoopOps);
    }
    //   create DFG within the basic block
    if (!exitOps.empty())
      if (failed(initDFGBB(connBB, exitOps, preGenOps, false, true)))
        return nullptr;

    // insert an unconditional branch to terminate the block
    if (connBB->getOperations().empty())
      loc = connBB->getOperations().back().getLoc();
  }

  // insert an unconditional branch to the start of the block
  builder.setInsertionPointToEnd(connBB);
  builder.create<LLVM::BrOp>(loc, finiBlock);

  return connBB;
}

LogicalResult ModuloScheduleAdapter::replaceLiveOutWithNewPath(
    std::vector<mapId2Op> insertOpsList) {
  // check whether operations in the fini phase use the operations in the loop
  for (auto &op : finiBlock->getOperations()) {
    for (auto opr : op.getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(opr)) {
        // TODO[@Yuxuan]: general method to handle block arguments
        return failure();
      }
      if (opr.getDefiningOp()) {
        auto defOp = opr.getDefiningOp();
        if (defOp->getBlock() != templateBlock)
          continue;
        unsigned opId = getOpId(loopOpList, defOp);
        // insert branch arguments for the terminators of the predecessors
        // stored in insertOpsList
        auto blockArg = finiBlock->addArgument(
            opr.getType(), finiBlock->getOperations().front().getLoc());
        // replace the use of opr to arg;
        op.replaceUsesOfWith(opr, blockArg);
        for (auto opList : insertOpsList)
          addBranchArgument(opList[opId]->getBlock()->getTerminator(),
                            opList[opId], finiBlock);
      }
    }
  }
  return success();
}

LogicalResult ModuloScheduleAdapter::saveDFGs(SmallVector<Block *> preBlocks,
                                              SmallVector<Block *> postBlocks) {
  // save the enter DFGs
  for (auto [i, blk] : llvm::enumerate(preBlocks)) {
    if (enterDFGs[i].size() != blk->getOperations().size()) {
      return failure();
    }
    for (auto [j, op] : llvm::enumerate(blk->getOperations())) {
      auto &corrDFG = enterDFGs[i];
      corrDFG[j].first = &op;
    }
  }

  // save the exit DFGs
  for (auto [i, blk] : llvm::enumerate(postBlocks)) {
    if (exitDFGs[i].size() != blk->getOperations().size()) {
      return failure();
    }
    for (auto [j, op] : llvm::enumerate(blk->getOperations())) {
      auto &corrDFG = exitDFGs[i];
      corrDFG[j].first = &op;
    }
  }
  return success();
}

void ModuloScheduleAdapter::removeTempletBlock() {
  // delete the origianl loop block
  // collect all operations in reverse order in a temporary vector.

  std::vector<Operation *> toErase;
  for (auto &op : llvm::reverse(loopOpList)) {
    toErase.push_back(&op);
  }
  // use llvm::make_early_inc_range to erase safely.
  for (auto *op : llvm::make_early_inc_range(toErase))
    op->erase();

  templateBlock->erase();
  // erase loopFalseBlk
  loopFalseBlk->erase();
}

LogicalResult ModuloScheduleAdapter::adaptCFGWithLoopMS() {
  /// Data structure to store the liveOut arguments in different loop iterations
  std::vector<mapId2Op> insertOpsList = {};
  SmallVector<Block *> newBlks = {initBlock};
  SmallVector<Block *> preParts;
  SmallVector<Block *> postParts;
  //   init has been pushed into newBlks, step to prolog
  loopStage phase = prolog;

  // get the union of operations within a basic block
  mapId2Op insertOps = {};
  std::vector<std::set<int>> opSet = {};
  std::vector<std::set<int>> opSetPrev = {};
  for (auto [ind, s] : llvm::enumerate(bbTimeMap)) {
    // epilog is after the loop block
    if (phase == loop)
      phase = epilog;

    opSetPrev = opSet;
    opSet = getOperationSet(opTimeMap, s, opSet, phase == epilog);

    if (isLoopKernel(loopOpNum, getUnionSet(opSet))) {
      phase = loop;
    }

    // insert a new block before the template block
    auto newBlk = builder.createBlock(templateBlock);
    newBlks.push_back(newBlk);
    if (phase == epilog)
      postParts.push_back(newBlk);
    else
      preParts.push_back(newBlk);

    // connect the current block to the CFG
    switch (phase) {
    case prolog:
    case loop: {
      // if it is prolog, connect the block with block before it
      auto predBlk = newBlks.rbegin()[1];
      auto termOp = predBlk->getTerminator();
      if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(termOp)) {
        // create exit block for executing the rest of operations and jump to
        // the fini block
        std::vector<mlir::Location> loc1(condBr.getFalseDestOperands().size(),
                                         condBr->getLoc());
        // generate true and false successors for last block,
        // suppose last block is BB0, the terminator decides to re-enter the
        // loop(newBlk) or quit the loop(connBB)
        //  ^BB0:
        //  %flag, if true, go to ^newBlk, else go to ^connBB
        //   `opSetPrev` is the operations in the BB0, and phase == epilog
        //   indicates whether BB0 is loop
        auto preGenOps = insertOps;
        auto connBB =
            createExitBlock(condBr, opSetPrev, preGenOps, phase == epilog);
        //   put the generated operations on different paths into the list
        if (!connBB)
          return failure();
        insertOpsList.push_back(preGenOps);
        postParts.push_back(connBB);

        // create a new condBr op to switch the false and true dest
        builder.setInsertionPoint(termOp);
        auto newTerm = builder.create<cgra::ConditionalBranchOp>(
            termOp->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
            condBr.getOperand(1), connBB, condBr.getFalseDestOperands(), newBlk,
            condBr.getTrueDestOperands());

        // reverse the flag of the conditional branch
        reverseCondBrFlag(newTerm);
        termOp->erase();
      } else if (auto br = dyn_cast<LLVM::BrOp>(termOp)) {
        br.setSuccessor(newBlk);
      }
      break;
    }
    case epilog: {
      auto predBlk = newBlks.rbegin()[1];
      auto termOp = predBlk->getTerminator();
      if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(termOp)) {
        // set the true dest to be loop
        condBr.setTrueDest(predBlk);
        // set false dest to be epilog block
        condBr.setFalseDest(newBlk);
      } else {
        // the terminator of the loop block must be conditional branch
        return failure();
      }
      break;
    }
    default:
      break;
    }

    // init DFG in the new created basic block
    if (failed(initDFGBB(newBlk, opSet, insertOps, phase == loop,
                         phase == epilog)))
      return failure();

  } // end of the prolog-loop-epilog creation

  // create a jump to the fini block
  auto epilogBlk = newBlks.rbegin()[0];
  builder.setInsertionPointToEnd(epilogBlk);
  builder.create<LLVM::BrOp>(epilogBlk->getOperations().back().getLoc(),
                             finiBlock);

  // add the operatons generated in the loop path
  insertOpsList.push_back(insertOps);

  if (failed(replaceLiveOutWithNewPath(insertOpsList)))
    return failure();
  // remove the template block in the region
  removeTempletBlock();
  // remove the block arguments if it is not used
  removeUselessBlockArg();

  llvm::errs() << "save DFG\n";
  if (failed(saveDFGs(preParts, postParts)))
    return failure();

  llvm::errs() << "FINISH SUCCESSFULLY\n";

  return success();
}