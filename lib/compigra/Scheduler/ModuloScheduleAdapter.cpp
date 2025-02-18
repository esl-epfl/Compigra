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

static std::vector<std::set<int>> getDiffSet(std::vector<std::set<int>> set1,
                                             std::vector<std::set<int>> set2) {
  for (auto it1 = set1.begin(); it1 != set1.end();) {
    bool foundEqual = false;
    for (const auto &s2 : set2) {
      if (*it1 == s2) {
        foundEqual = true;
        break;
      }
    }
    if (foundEqual) {
      it1 = set1.erase(it1); // erase returns the next iterator
    } else {
      ++it1;
    }
  }
  return set1;
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

/// Return the union set for two sets
static std::set<int> getUnionSet(std::set<int> set1, std::set<int> set2) {
  std::set<int> unionSet;
  for (auto val : set1)
    unionSet.insert(val);
  for (auto val : set2)
    unionSet.insert(val);
  return unionSet;
}

/// insert the op index to the right loop iteration
static void insertOpToIter(std::vector<std::set<int>> &opSets, int val) {
  for (auto &s : opSets) {
    if (s.count(val) == 0) {
      s.insert(val);
      return;
    }
  }
  // all existed iteration have executed the op, init a loop iter
  opSets.push_back({});
  opSets.back().insert(val);
}

/// if repeteted index appears, attach them to new set
static std::vector<std::set<int>>
getCombinationSet(std::map<int, std::unordered_set<int>> timeMap,
                  std::unordered_set<int> keys) {
  std::vector<std::set<int>> opSets = {{}};
  for (auto [key, vals] : timeMap)
    if (keys.count(key))
      for (auto val : vals)
        insertOpToIter(opSets, val);

  return opSets;
}

/// In modulo scheduling, multiple operations might be executed in a same basic
/// block with different operantors. This function is used to get the
/// aggregation of the operation sets belongs to different loop iterations.
static std::vector<std::set<int>>
getOperationSet(std::map<int, std::unordered_set<int>> timeMap,
                std::unordered_set<int> keys,
                std::vector<std::set<int>> prevSet,
                const std::map<int, int> execTime, bool epilog = false) {

  if (keys.empty())
    return {};

  if (epilog) {
    auto opSets = getCombinationSet(timeMap, keys);

    // sort opSets according to its latest element meaning the iteration is
    // initialized in prior order
    std::sort(opSets.begin(), opSets.end(),
              [&](const std::set<int> &a, const std::set<int> &b) {
                return execTime.at(*a.begin()) > execTime.at(*b.begin());
              });
    return opSets;
  }

  if (prevSet.empty())
    return (getCombinationSet(timeMap, keys));

  // another loop iteration is initialized in prolog
  std::vector<std::set<int>> opSets(prevSet.size() + 1);
  for (auto [t, vals] : timeMap)
    if (keys.count(t) > 0)
      for (auto val : vals) {
        // if hase not been executed, add to cur set
        bool newIter = true;
        for (auto [ind, prev] : llvm::enumerate(prevSet)) {
          // prev = prevSet[ind];
          auto executed = getUnionSet(
              std::vector<std::set<int>>(prevSet.begin() + ind, prevSet.end()));
          executed = getUnionSet(opSets[ind], executed);
          if (executed.count(val) == 0) {
            opSets[ind].insert(val);
            // inserted, update prevSet to avoid add multiple times
            newIter = false;
            break;
          }
        }
        // if all previous iteration contains such op, meaning it is a new iter
        if (newIter)
          opSets.back().insert(val);
      }

  std::sort(opSets.begin(), opSets.end(),
            [&](const std::set<int> &a, const std::set<int> &b) {
              return execTime.at(*a.begin()) > execTime.at(*b.begin());
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

void ModuloScheduleAdapter::removeBlockArgs(Operation *term,
                                            std::vector<unsigned> argId,
                                            Block *dest) {
  builder.setInsertionPoint(term);
  if (auto br = dyn_cast<LLVM::BrOp>(term)) {
    SmallVector<Value> operands;
    for (auto [ind, opr] : llvm::enumerate(br->getOperands())) {
      if (std::find(argId.begin(), argId.end(), (unsigned)ind) == argId.end()) {
        operands.push_back(opr);
        llvm::errs() << "NOT FOUND " << ind;
      }
    }

    builder.create<LLVM::BrOp>(term->getLoc(), operands, br.getSuccessor());
    term->erase();
    return;
  }

  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(term)) {
    SmallVector<Value> trueOperands;
    for (auto [ind, opr] : llvm::enumerate(condBr.getTrueDestOperands()))
      if (condBr.getTrueDest() != dest ||
          std::find(argId.begin(), argId.end(), ind) == argId.end())
        trueOperands.push_back(opr);
    SmallVector<Value> falseOperands;
    for (auto [ind, opr] : llvm::enumerate(condBr.getFalseDestOperands()))
      if (condBr.getFalseDest() != dest ||
          std::find(argId.begin(), argId.end(), ind) == argId.end())
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

static Operation *hasOpId(std::vector<opWithId> opIds, int id) {
  for (auto [op, opId] : opIds)
    if (opId == id)
      return op;
  return nullptr;
}

LogicalResult ModuloScheduleAdapter::searchInitArg(std::stack<Block *> &blkSt,
                                                   int id, Block *curBlk) {
  std::stack<Block *> trackSt;
  Block *trackBlk = curBlk;
  while (!blkSt.empty()) {
    auto blk = blkSt.top();
    blkSt.pop();
    if (blkOpIds.count(blk) > 0) {
      auto opWId = blkOpIds.at(blk);
      if (hasOpId(opWId, id)) {
        // add argument to the block
        addBranchArgument(blk->getTerminator(), hasOpId(opWId, id), trackBlk);
        // add block arguments for all predecessors in trackSt
        while (!trackSt.empty()) {
          auto pred = trackSt.top();
          trackSt.pop();

          pred->addArgument(hasOpId(opWId, id)->getResult(0).getType(),
                            pred->getOperations().front().getLoc());
          if (!trackSt.empty())
            addBranchArgument(pred->getTerminator(), hasOpId(opWId, id),
                              trackSt.top());
          else
            addBranchArgument(pred->getTerminator(), hasOpId(opWId, id),
                              curBlk);
          blkOpIds[pred].push_back({hasOpId(opWId, id), id});
        }
        return success();
      } else
        trackSt.push(blk);
    }
    for (auto pred : blk->getPredecessors())
      if (blkOpIds.count(pred) > 0) {
        // llvm::errs() << pred->getOperations().front() << "\n";
        blkSt.push(pred);
      }
    trackBlk = blk;
  }
  llvm::errs() << "FAILED\n";
  return failure();
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
  Operation *termOp = nullptr;
  unsigned termOpId = -1;

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
        // one BB can only have one terminator
        if (termOp)
          return failure();
        termOp = repOp;
        termOpId = createdDFG.size();

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
          } else {
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

  // always let the terminator terminates current bb;
  if (termOp != lastOp && termOp != nullptr) {
    termOp->moveAfter(lastOp);
    // move termOp to the back of createdDFG in order to maintain consistency
    // with IR
    std::rotate(createdDFG.begin() + termOpId,
                createdDFG.begin() + termOpId + 1, createdDFG.end());
  }

  // add branch arguments for the kernel block
  // the loop block must have a conditional terminator
  if (isKernel && !exit) {
    if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(blk->getTerminator()))
      condBr.setTrueDest(blk);
    else
      return failure();
  }

  // add block arguments for all predecessors
  for (auto pred :
       llvm::make_early_inc_range(lastOp->getBlock()->getPredecessors())) {
    for (auto argId : argIds) {
      if (pred == blk)
        addBranchArgument(pred->getTerminator(), curGenOps[argId], blk);
      else {
        if (preGenOps.count(argId) > 0)
          addBranchArgument(pred->getTerminator(), preGenOps[argId], blk);
        else {
          std::stack<Block *> blkSt;
          blkSt.push(pred);
          if (failed(searchInitArg(blkSt, argId, blk)))
            return failure();
        }
      }
    }
  }

  // update preGenOps with curGenOps
  std::vector<int> genIds;
  for (auto [ind, op] : curGenOps) {
    preGenOps[ind] = op;
    genIds.push_back(ind);
  }

  // store curGenOps
  if (exit) {
    exitDFGs.push_back(createdDFG);
  } else {
    enterDFGs.push_back(createdDFG);
  }
  blkOpIds[blk] = createdDFG;

  return success();
}

/// Generate operations in a basic that if the loop is terminated in the
/// prolog phase.
Block *ModuloScheduleAdapter::createExitBlock(
    cgra::ConditionalBranchOp condBr, std::vector<std::set<int>> &existOps,
    mapId2Op &preGenOps, cgra::ConditionalBranchOp loopTerm) {

  // full Set is the operation index from 0 to
  // templateBlk->getOperations().size() - 1
  std::set<int> fullSet;
  for (int i = 0; i < loopOpNum; ++i) {
    fullSet.insert(i);
  }

  auto connBB = builder.createBlock(finiBlock);
  if (loopTerm)
    loopTerm.setFalseDest(connBB);
  condBr.setTrueDest(connBB);
  Location loc = condBr->getLoc();
  std::vector<std::set<int>> exitOps;

  // get the diff of existedOps[] and the templateBlk
  for (int i = 0; i < existOps.size(); i++) {
    // get the union set from first to the end - i of the existedOps
    auto unionSet = getUnionSet(
        std::vector<std::set<int>>(existOps.begin() + i, existOps.end()));

    // if the executed ops in the prolog does not contain the terminator, they
    // are redundant ops.
    if (!unionSet.count(loopOpNum - 1))
      continue;
    // get the difference between unionSet and full set;
    auto exitLoopOps = getDiffSet(fullSet, unionSet);
    if (!exitLoopOps.empty())
      exitOps.push_back(exitLoopOps);
  }

  // print exitOps
  for (auto opSet : exitOps) {
    for (auto opId : opSet) {
      llvm::errs() << opId << " ";
    }
    llvm::errs() << "\n";
  }

  //   create DFG within the basic block
  if (!exitOps.empty())
    if (failed(
            initDFGBB(connBB, exitOps, preGenOps, loopTerm != nullptr, true)))
      return nullptr;

  // insert an unconditional branch to terminate the block
  if (connBB->getOperations().empty())
    loc = connBB->getOperations().back().getLoc();

  // insert an unconditional branch to the start of the block
  builder.setInsertionPointToEnd(connBB);
  builder.create<LLVM::BrOp>(loc, finiBlock);

  return connBB;
}

LogicalResult ModuloScheduleAdapter::replaceLiveOutWithNewPath(
    std::vector<mapId2Op> insertOpsList) {
  // check whether operations in the fini phase use the operations in the loop
  SmallVector<BlockArgument> finiArgs;
  for (auto &op : finiBlock->getOperations()) {
    for (auto opr : op.getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(opr)) {
        llvm::errs() << op << ": " << opr << "\n";
        // if it is a block argument from other block, use the original value
        if (blockArg.getOwner() != finiBlock)
          continue;
        // already added the argument propogated from the successors
        if (std::find(finiArgs.begin(), finiArgs.end(), blockArg) !=
            finiArgs.end())
          continue;
        finiArgs.push_back(blockArg);
        // still use block argument, the parameter propagation is handled by the
        // predecessor
        unsigned argId = blockArg.getArgNumber();
        // if (loopFalseBlk->getSuccessor(0) != finiBlock)
        //   return failure();
        auto cBr =
            dyn_cast<cgra::ConditionalBranchOp>(templateBlock->getTerminator());
        Operation *defOp = cBr.getFalseOperand(argId).getDefiningOp();
        unsigned opId = getOpId(templateBlock->getOperations(), defOp);
        for (auto opList : insertOpsList)
          addBranchArgument(opList[opId]->getBlock()->getTerminator(),
                            opList[opId], finiBlock);
        continue;
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
    // need to update the terminator in prolog as it is updated during epilog
    // DFG generation
    for (auto [j, op] : llvm::enumerate(blk->getOperations())) {
      auto &corrDFG = enterDFGs[i];
      corrDFG[j].first = &op;
    }
  }

  // save the exit DFGs
  for (auto [i, blk] : llvm::enumerate(postBlocks)) {
    // additionally jump operation is added to the block to control the DFG
    // exit behavior.
    if (exitDFGs[i].size() != blk->getOperations().size() - 1) {
      return failure();
    }
  }
  return success();
}

void ModuloScheduleAdapter::removeTempletBlock() {
  // delete the origianl loop block
  // collect all operations in reverse order in a temporary vector.

  // erase loopFalseBlk terminator in case it is used for parameter propagation.
  // loopFalseBlk->getTerminator()->erase();

  std::vector<Operation *> toErase;
  for (auto &op : llvm::reverse(loopOpList)) {
    toErase.push_back(&op);
  }

  for (auto *op : llvm::make_early_inc_range(toErase))
    op->erase();

  templateBlock->erase();
  // loopFalseBlk->erase();
}

/// get the corresponding operand in the init block, suppose in loop block,
/// the block argument receives the operand from the loop, get the
/// corresponding id for the loop operation. For example br ^bb1 (%0, %1)
/// ^bb1: (%2, %3)
///  %4 = add %2, %3
/// cond_br [%4, 0], ^bb2(%3, %4), ^bb3
/// the result operands in ^bb1 are indexed by %2->0, %3->1,%4->3
/// then %0 corresponds to %3(id=1), %1 corresponds to %4(id=2)
static std::vector<opWithId> getInitOperandId(Block *initBlk, Block *loopBlk) {
  std::vector<opWithId> opIds;
  Operation *termOp = initBlk->getTerminator();
  unsigned opIdBase = 0;
  std::vector<int> removedIds = {};
  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(termOp)) {
    if (condBr.getTrueDest() == loopBlk)
      opIdBase = 2;
    else
      opIdBase = 2 + condBr.getTrueDestOperands().size();
  }

  if (auto condBr =
          dyn_cast<cgra::ConditionalBranchOp>(loopBlk->getTerminator()))
    for (auto [ind, opr] : llvm::enumerate(condBr.getTrueDestOperands())) {
      auto corrOpr = termOp->getOperand(opIdBase + ind);
      removedIds.push_back(opIdBase + ind);
      opIds.push_back({corrOpr.getDefiningOp(),
                       getOpId(loopBlk->getOperations(), opr.getDefiningOp())});
    }

  // remove the operands in the terminator
  termOp->eraseOperands(removedIds[0], removedIds.size());

  return opIds;
}

LogicalResult ModuloScheduleAdapter::adaptCFGWithLoopMS() {
  // remove the operators to the loop block, add it later in the CFG
  // generation to solve the data dependency issue by storing the branch
  // operator
  auto cntOpr = getInitOperandId(initBlock, templateBlock);
  blkOpIds[initBlock] = cntOpr;

  /// Data structure to store the liveOut arguments in different loop
  /// iterations
  SmallVector<Block *> newBlks = {initBlock};

  SmallVector<Block *> preParts;
  SmallVector<Block *> postParts;
  // init has been pushed into newBlks, step to prolog
  loopStage phase = prolog;

  // get the union of operations within a basic block
  mapId2Op insertOps = {};
  Block *loopBlock = nullptr;
  opIdInIter opSet = {};
  std::vector<opIdInIter> existIds = {};
  std::vector<mapId2Op> insertOpsList = {};
  for (auto [ind, s] : llvm::enumerate(bbTimeMap)) {
    // does not process epilog in the prolog-loop basic block generation
    if (phase == loop)
      break;

    opSet = getOperationSet(opTimeMap, s, opSet, execTime, false);
    existIds.push_back(opSet);

    // print opSet
    llvm::errs() << "init DFG for " << ind << " th block\n";
    for (auto set : opSet) {
      llvm::errs() << "{";
      for (auto opId : set) {
        llvm::errs() << opId << " ";
      }
      llvm::errs() << "}";
    }
    llvm::errs() << "\n";

    // insert a new block before the template block
    auto newBlk = builder.createBlock(templateBlock);
    newBlks.push_back(newBlk);
    preParts.push_back(newBlk);

    if (isLoopKernel(loopOpNum, getUnionSet(opSet))) {
      phase = loop;
      loopBlock = newBlk;
    }
    // connect the current block to the CFG
    auto predBlk = newBlks.rbegin()[1];
    auto preTermOp = predBlk->getTerminator();
    if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(preTermOp)) {
      // create a new condBr op to switch the false and true dest, temporarily
      // point the true dest to newBlk.
      builder.setInsertionPoint(preTermOp);
      auto newTerm = builder.create<cgra::ConditionalBranchOp>(
          preTermOp->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
          condBr.getOperand(1), finiBlock, condBr.getFalseDestOperands(),
          newBlk, condBr.getTrueDestOperands());

      // reverse the flag of the conditional branch
      reverseCondBrFlag(newTerm);
      preTermOp->erase();
    } else if (auto br = dyn_cast<LLVM::BrOp>(preTermOp)) {
      br.setSuccessor(newBlk);
    }

    // init DFG in the new created basic block
    if (failed(initDFGBB(newBlk, opSet, insertOps, phase == loop, false)))
      return failure();
    insertOpsList.push_back(insertOps);

  } // end of the prolog-loop creation

  std::vector<mapId2Op> liveOutOpList = {};
  // for (size_t ind = 1; ind < newBlks.size() - 1; ++ind) {
  for (int ind = newBlks.size() - 2; ind >= 1; --ind) {
    auto s = bbTimeMap[ind - 1];
    auto preGenOps = insertOpsList[ind - 1];
    opSet = existIds[ind - 1];

    // connect the current block to the CFG

    // if it is prolog, connect the block with block before it
    auto firstHalf = newBlks[ind];
    auto termOp = firstHalf->getTerminator();
    if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(termOp)) {
      // generate the quit loop operations corresponding to the prolog phase,
      // suppose last block is BB0, the terminator decides to re-enter the
      // loop or quit the loop(quitBB)
      //  ^BB0:
      //  %flag, if true, go to ^loop, else go to ^quitBB
      cgra::ConditionalBranchOp optionalTerm = nullptr;
      // epilog DFG is the same with complement of last prolog, suppose the
      // loop is prolog1->prolog2 ... prolog (k-1)
      //  -> kernel
      //  -> epilog (k-1) ... -> epilog 1
      // porlog (k-1) shares the same epilog DFG with kernel, as they all have
      // (L-1) iteration unfinished.
      if (ind + 1 == newBlks.size() - 1) {
        if (auto loopTerm =
                dyn_cast<cgra::ConditionalBranchOp>(loopBlock->getTerminator()))
          optionalTerm = loopTerm;
        else
          return failure();
      }
      llvm::errs() << "create exit block for " << ind - 1 << " th prolog\n";
      auto quitBB = createExitBlock(condBr, opSet, preGenOps, optionalTerm);
      //   put the generated operations on different paths into the list
      liveOutOpList.push_back(preGenOps);
      if (!quitBB)
        return failure();
      postParts.push_back(quitBB);
    }

  } // end of the epilog creation
  llvm::errs() << "finish bb creation\n";

  // add the operatons generated in the loop path
  if (failed(replaceLiveOutWithNewPath(liveOutOpList)))
    return failure();
  llvm::errs() << "finish arg replacement\n";

  // remove the template block in the region
  removeTempletBlock();
  llvm::errs() << "finish remove block\n";

  // remove the block arguments if it is not used
  removeUselessBlockArg();
  llvm::errs() << "finish remove block arg\n";

  if (failed(saveDFGs(preParts, postParts)))
    return failure();
  llvm::errs() << "finish save DFGs\n";

  llvm::errs() << "MS adapter finish\n";
  return success();
}