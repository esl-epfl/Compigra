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
#include "compigra/Support/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

using namespace mlir;
using namespace compigra;

/// Get the init block of the region
Block *ModuloScheduleAdapter::getInitBlock(Block *loopBlk) {
  // if there is only one predecessor different from the loop block, return it,
  // otherwise, create an empty block and connect it to the loop block.
  if (std::distance(loopBlk->getPredecessors().begin(),
                    loopBlk->getPredecessors().end()) == 2) {
    for (auto pred : loopBlk->getPredecessors())
      if (pred != loopBlk)
        return pred;
  } else {
    // TODO[@YYY]: test with corresponding cases
    OpBuilder builder(loopBlk->getParentOp());
    auto newBlock = builder.createBlock(loopBlk->getParent());
    builder.setInsertionPointToEnd(newBlock);
    // change the precessor to the new block
    SmallVector<Value, 4> mergeArgs;
    for (auto arg : templateBlock->getArguments()) {
      auto newArg = newBlock->addArgument(arg.getType(), arg.getLoc());
      mergeArgs.push_back(newArg);
    }
    builder.create<cf::BranchOp>(loopBlk->getOperations().front().getLoc(),
                                 loopBlk, mergeArgs);
    for (auto pred : loopBlk->getPredecessors()) {
      if (pred == loopBlk)
        continue;
      // replace the branch to the loop block with the new block
      auto termOp = pred->getTerminator();
      if (auto branchOp = dyn_cast_or_null<cf::BranchOp>(termOp)) {
        branchOp.setDest(newBlock);
      } else if (auto condBrOp =
                     dyn_cast_or_null<cgra::ConditionalBranchOp>(termOp)) {
        if (condBrOp.getTrueDest() == loopBlk)
          condBrOp.setTrueDest(newBlock);
        else
          condBrOp.setFalseDest(newBlock);
      }
    }
    // remove all block argument s
    return newBlock;
  }
}

/// the conditional branch block has two successors, for the loop the true
/// points to the entry of the loop by default, get the false block
Block *ModuloScheduleAdapter::getCondBrFalseDest(Block *blk) {
  for (auto succ : blk->getSuccessors())
    if (succ != blk)
      return succ;
  return nullptr;
}

std::map<Operation *, ScheduleUnit>
ModuloScheduleAdapter::getPrologAndKernelSolutions() {
  std::map<Operation *, ScheduleUnit> prologAndKernelSolution;
  for (auto [op, su] : solution) {
    if (epilogOpsBB.count(op->getBlock()))
      continue;
    prologAndKernelSolution[op] = su;
  }
  return prologAndKernelSolution;
}

SmallVector<Block *, 4> ModuloScheduleAdapter::getPrologAndKernelBlocks() {
  SmallVector<Block *, 4> prologAndKernelBlocks;
  for (auto blk : newBlocks) {
    if (epilogOpsBB.count(blk))
      continue;
    prologAndKernelBlocks.push_back(blk);
  }
  return prologAndKernelBlocks;
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
unsigned compigra::getOpId(Block::OpListType &opList, Operation *search) {
  for (auto [ind, op] : llvm::enumerate(opList))
    if (&op == search)
      return ind;
  return -1;
}

/// Return the union set specified by key in keys
static std::set<int> getUnionSet(std::map<int, std::set<int>> mapSet,
                                 std::set<int> keys) {
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

LogicalResult ModuloScheduleAdapter::init() {
  // Get related basic blocks
  loopOpNum = loopOpList.size();
  if (loopOpNum == 0)
    return failure();

  finiBlock = getCondBrFalseDest(templateBlock);
  if (!finiBlock)
    return failure();

  initBlock = getInitBlock(templateBlock);
  if (!initBlock)
    return failure();

  // finiBlock = loopFalseBlk;
  // The last block in the timeSlotsOfBBs is the epilog block, and the second
  // to last block is the loop block
  unsigned termCount = 0;
  for (auto [ind, s] : llvm::enumerate(timeSlotsOfBBs)) {
    // timeStep += s.size();
    auto opSet = getUnionSet(opTimeMap, s);
    if (opSet.count(loopOpNum - 1) > 0)
      termCount++;
    if (isLoopKernel(loopOpNum, opSet)) {
      loopBlkId = ind;
      break;
    }
  }
  loopIterId = termCount - 1;

  auto termOp = templateBlock->getTerminator();
  if (auto condBr = dyn_cast_or_null<cgra::ConditionalBranchOp>(termOp))
    cmpFlag = condBr.getPredicate();
  else
    return failure();
  loopCmpOprId1 = getOpId(loopOpList, termOp->getOperand(0).getDefiningOp());
  loopCmpOprId2 = getOpId(loopOpList, termOp->getOperand(1).getDefiningOp());
  return success();
}

static Value getPropagatedValue(Operation *terminator, Block *dstBlk,
                                unsigned argId) {
  if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
    return br.getOperand(argId);
  }
  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(terminator)) {
    if (dstBlk == condBr.getTrueDest())
      return condBr.getTrueOperand(argId);
    return condBr.getFalseOperand(argId);
  }
  return nullptr;
}

/// Return the latest iteration id of an operation.
/// opMap stores all the executed operations with their iteration id and op
/// id, given a opId, get the latest iteration id for the op
static int
getLatestIterId(const std::map<int, std::map<int, Operation *>> opMap, int opId,
                bool getExisted = true) {
  int iterId = 0;
  while (opMap.count(iterId) && opMap.at(iterId).count(opId)) {
    iterId++;
  }
  return getExisted ? iterId - 1 : iterId;
}

Operation *getInitialValue(Value opr, Block *initBlock = nullptr) {
  if (opr.getDefiningOp())
    return opr.getDefiningOp();

  if (auto arg = dyn_cast<BlockArgument>(opr)) {

    auto succ = arg.getOwner()->getSuccessors().front();
    if (initBlock) {
      succ = initBlock;
    }
    auto term = succ->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(term))
      return getInitialValue(br.getOperand(arg.getArgNumber()));
    if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(term)) {
      if (succ == condBr.getTrueDest())
        return getInitialValue(condBr.getTrueOperand(arg.getArgNumber()));
      return getInitialValue(condBr.getFalseOperand(arg.getArgNumber()));
    }
  }
}

LogicalResult ModuloScheduleAdapter::updateOperands(
    int bbId, int iterId, int opId, Operation *adaptOp,
    const std::map<int, std::map<int, Operation *>> prologOpMap,
    const std::map<int, std::map<int, Operation *>> newCreatedOps,
    std::vector<std::pair<int, int>> &propagatedOps, bool isEpilog) {
  // first update the operands
  auto blk = adaptOp->getBlock();
  // auto opMap = prologOpMap merge epilogOpMap;
  std::map<int, std::map<int, Operation *>> opMap = prologOpMap;
  bool kernel = (bbId == loopBlkId) && (!isEpilog);

  auto mergedOpMap = prologOpMap;
  if (isEpilog)
    for (auto [iterId, opMap] : newCreatedOps) {
      for (auto [opId, op] : opMap) {
        mergedOpMap[iterId][opId] = op;
      }
    }

  // Helper function to add block argument and record propagation
  auto addBlockArgAndPropagate = [&](Type argType, int operandId,
                                     int propOpId) {
    auto src1 = getLatestIterId(mergedOpMap, propOpId);

    // auto it = std::find(propagatedOps.begin(), propagatedOps.end(),
    //                     std::make_pair(src1, propOpId));
    // if (it != propagatedOps.end()) {
    //   int index = std::distance(propagatedOps.begin(), it);
    //   adaptOp->setOperand(operandId, blk->getArguments()[index]);
    //   return;
    // }
    Location loc = blk->getOperations().empty()
                       ? blk->getPrevNode()->getTerminator()->getLoc()
                       : blk->getOperations().front().getLoc();

    blk->addArgument(argType, loc);
    adaptOp->setOperand(operandId, blk->getArguments().back());

    propagatedOps.push_back({src1, propOpId});
    propagatedOps.push_back({src1 + 1, propOpId});
  };

  auto getTemplateTrueOperand = [&](Operation *op, unsigned argId) {
    if (auto condBr = dyn_cast_or_null<cgra::ConditionalBranchOp>(op))
      return condBr.getTrueOperand(argId);
    return Value();
  };

  for (auto [adaptId, opr] : llvm::enumerate(adaptOp->getOperands())) {
    if (auto blockArg = dyn_cast<BlockArgument>(opr)) {
      if (blockArg.getParentBlock() != templateBlock)
        continue;
      // if iterId is in prologe, the argument is from previous iteration
      auto argInd = blockArg.getArgNumber();
      if (bbId < loopBlkId) {
        if (iterId == 0) {
          adaptOp->setOperand(adaptId, startBlock->getArgument(argInd));
          continue;
        }
        // get the last iteration id
        auto propagatedVal =
            getTemplateTrueOperand(templateBlock->getTerminator(), argInd);
        auto op = opMap.at(iterId - 1)
                      .at(getOpId(loopOpList, propagatedVal.getDefiningOp()));
        adaptOp->setOperand(adaptId, op->getResult(0));
        continue;
      }

      // it would not only receive operands from the previous iteration of the
      // successor block, but also receive operands from the loop iteration
      if (bbId == loopBlkId) {
        auto propagatedVal =
            getTemplateTrueOperand(templateBlock->getTerminator(), argInd);
        auto propOpId = getOpId(loopOpList, propagatedVal.getDefiningOp());
        if (newCreatedOps.count(iterId - 1) &&
            newCreatedOps.at(iterId - 1).count(propOpId)) {
          auto curDefOp = newCreatedOps.at(iterId - 1).at(propOpId);
          adaptOp->setOperand(adaptId, curDefOp->getResult(0));
          continue;
        }
        //  add initial value to the block
        if (getLatestIterId(mergedOpMap, propOpId) == -1) {
          auto corrArg = isa<BlockArgument>(opr)
                             ? startBlock->getArgument(
                                   opr.cast<BlockArgument>().getArgNumber())
                             : opr;
          prologOps[-1][propOpId] = getInitialValue(corrArg, initBlock);
        }
        addBlockArgAndPropagate(blockArg.getType(), adaptId, propOpId);
        continue;
      }
    }

    if (auto defOp = opr.getDefiningOp()) {
      // produced outside the loop, no need to update
      if (defOp->getBlock() != templateBlock)
        continue;
      unsigned defOpId = getOpId(loopOpList, defOp);

      if (bbId < loopBlkId) {
        // determine whether the defOp is calculated in the loop, if not, no
        // need to update
        int latestProducerIterId = getLatestIterId(opMap, defOpId);
        if (latestProducerIterId == -1) {
          if (isEpilog && newCreatedOps.count(iterId) &&
              newCreatedOps.at(iterId).count(defOpId)) {
            adaptOp->setOperand(
                adaptId, newCreatedOps.at(iterId).at(defOpId)->getResult(0));
          } else {
            return failure();
          }
        } else {
          adaptOp->setOperand(
              adaptId,
              opMap.at(latestProducerIterId).at(defOpId)->getResult(0));
        }
        continue;
      }

      // the loop block receives multiple operands
      if (bbId == loopBlkId) {
        if (newCreatedOps.count(iterId) &&
            newCreatedOps.at(iterId).count(defOpId)) {
          auto curDefOp = newCreatedOps.at(iterId).at(defOpId);
          adaptOp->setOperand(adaptId, curDefOp->getResult(0));
          continue;
        }
        addBlockArgAndPropagate(opr.getType(), adaptId, defOpId);
        continue;
        // record the argument id for the block
      }
    }
    return failure();
  }
  return success();
}

/// Reverse the conditional branch flag if the true and false block is
/// reversed
static void reverseCondBrFlag(cgra::ConditionalBranchOp condBr) {
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
    break;
  }
  case cgra::CondBrPredicate::ge: {
    condBr.setPredicate(cgra::CondBrPredicate::lt);
    Value tmp = condBr.getOperand(0);
  }
  }
  // reverse the true and false block
  auto tmp = condBr.getTrueDest();
  condBr.setTrueDest(condBr.getFalseDest());
  condBr.setFalseDest(tmp);
}

void ModuloScheduleAdapter::removeBlockArgs(Operation *term,
                                            std::vector<unsigned> argId,
                                            Block *dest) {
  builder.setInsertionPoint(term);
  if (auto br = dyn_cast<cf::BranchOp>(term)) {
    SmallVector<Value> operands;
    for (auto [ind, opr] : llvm::enumerate(br->getOperands())) {
      if (std::find(argId.begin(), argId.end(), (unsigned)ind) == argId.end()) {
        operands.push_back(opr);
        llvm::errs() << "NOT FOUND " << ind;
      }
    }

    builder.create<cf::BranchOp>(term->getLoc(), operands, br.getSuccessor());
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

void ModuloScheduleAdapter::removeTempletBlock() {
  // delete the origianl loop block
  // collect all operations in reverse order in a temporary vector.

  // erase loopFalseBlk terminator in case it is used for parameter
  // propagation. loopFalseBlk->getTerminator()->erase();

  std::vector<Operation *> toErase;
  for (auto &op : llvm::reverse(loopOpList)) {
    toErase.push_back(&op);
  }

  for (auto *op : llvm::make_early_inc_range(toErase))
    op->erase();

  templateBlock->erase();

  // loopFalseBlk->erase();
}

std::vector<int> sortElementsByIteration(
    std::set<int> s,
    const std::map<int, std::map<int, Operation *>> prologOps) {
  std::vector<int> sortedOpIds;
  // if second key is the same, sort by the first key
  std::vector<std::pair<int, int>> iterOpPairs;
  for (const auto &opId : s) {
    iterOpPairs.emplace_back(getLatestIterId(prologOps, opId), opId);
  }

  std::sort(iterOpPairs.begin(), iterOpPairs.end());
  for (const auto &[iterId, opId] : iterOpPairs) {
    sortedOpIds.push_back(opId);
  }

  return sortedOpIds;
}

LogicalResult ModuloScheduleAdapter::initOperationsInBB(
    Block *blk, int bbId, const std::set<int> bbTimeId,
    std::vector<std::pair<int, int>> &propOpSets) {
  std::map<int, std::map<int, Operation *>> prologBBOps;
  for (auto t : bbTimeId) {
    std::set<int> opIds = opTimeMap.at(t);
    // order elements in opIds according to their iteration, basicly the
    // number of showing up times of the opId in prologOps
    auto sortedOpIds = sortElementsByIteration(opIds, prologOps);
    for (auto opId : sortedOpIds) {
      // initialize new iteraions
      int iterId = getLatestIterId(prologOps, opId, false);
      int execTimeInBB = execTime.at(opId);
      // create terminator at the end of the block
      if (opId >= loopOpNum - 1)
        continue;

      // if not the terminator, clone the operation from template block with
      // opId
      builder.setInsertionPointToEnd(blk);
      Operation *op = builder.clone(*getOp(loopOpList, opId));
      std::vector<std::pair<int, int>> propagatedOps;
      // For creating prolog block, the epilogOps is empty
      if (failed(updateOperands(bbId, iterId, opId, op, prologOps, prologBBOps,
                                propagatedOps)))
        return failure();
      // merge propagatedOps to the propOpSets
      propOpSets.insert(propOpSets.end(), propagatedOps.begin(),
                        propagatedOps.end());
      // connect the operation with its corresponding operands
      prologOps[iterId][opId] = op;
      prologBBOps[iterId][opId] = op;
    }
  }
  return success();
}

LogicalResult ModuloScheduleAdapter::completeUnexecutedOperationsInBB(
    Block *blk, int bbId, unsigned termIter,
    const std::map<int, std::map<int, Operation *>> executedOpMap,
    std::vector<std::pair<int, int>> &propOpSets) {
  // if executedOpMap first key <= termIterId, get all the second key(executed
  // op Id), if the second key does not include 0-loopOpNum-2, add the
  // operation to the epilog block
  std::map<int, std::map<int, Operation *>> epilogOps;
  for (auto [iterId, opMap] : executedOpMap) {
    if (iterId > termIter)
      continue;

    for (unsigned opId = 0; opId < loopOpNum - 1; ++opId) {
      if (opMap.count(opId) != 0)
        continue;

      // create the operation in the block
      builder.setInsertionPointToEnd(blk);
      Operation *op = builder.clone(*getOp(loopOpList, opId));
      std::vector<std::pair<int, int>> propagatedOps;
      if (failed(updateOperands(bbId, iterId, opId, op, executedOpMap,
                                epilogOps, propagatedOps, true)))
        return failure();
      epilogOps[iterId][opId] = op;
      // merge propagatedOps to the propOpSets
      propOpSets.insert(propOpSets.end(), propagatedOps.begin(),
                        propagatedOps.end());
    }
  }

  // the epilog block would directly jump to the fini block
  // insert an unconditional branch to the start of the block
  builder.setInsertionPointToEnd(blk);
  SmallVector<Value> jumpArgs;
  for (auto arg : finiBlock->getArguments()) {
    // get the opid from the template terminator
    auto opId =
        getOpId(loopOpList, getPropagatedValue(templateBlock->getTerminator(),
                                               finiBlock, arg.getArgNumber())
                                .getDefiningOp());
    auto iterId = getLatestIterId(epilogOps, opId);
    if (epilogOps.count(iterId) && epilogOps[iterId].count(opId)) {
      jumpArgs.push_back(epilogOps[iterId][opId]->getResult(0));
    } else {
      // seek the latest iteration id from the prolog block
      iterId = getLatestIterId(prologOps, opId);
      jumpArgs.push_back(prologOps[iterId][opId]->getResult(0));
    }
  }
  builder.create<cf::BranchOp>(blk->getOperations().back().getLoc(), finiBlock,
                               jumpArgs);
  epilogOpsBB[blk] = epilogOps;
  return success();
}

void replaceSuccessor(Block *blk, Block *oldBlk, Block *newBlk) {
  for (auto succ : blk->getSuccessors()) {
    if (auto br = dyn_cast<cf::BranchOp>(blk->getTerminator())) {
      br.setSuccessor(newBlk);
    } else if (auto condBr =
                   dyn_cast<cgra::ConditionalBranchOp>(blk->getTerminator())) {
      if (condBr.getTrueDest() == oldBlk) {
        condBr.setTrueDest(newBlk);
      } else if (condBr.getFalseDest() == oldBlk) {
        condBr.setFalseDest(newBlk);
      }
    }
  }
}

void addPropagateValue(Operation *sucTermOp, Value propVal, Block *targetBB) {
  if (auto br = dyn_cast<cf::BranchOp>(sucTermOp)) {
    if (br.getSuccessor() == targetBB) {
      SmallVector<Value, 4> newOperands(br.getOperands().begin(),
                                        br.getOperands().end());
      newOperands.push_back(propVal);
      br.getOperation()->setOperands(newOperands);
    }
  } else if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(sucTermOp)) {
    if (condBr.getTrueDest() == targetBB)
      condBr.getTrueDestOperandsMutable().append(propVal);
    else if (condBr.getFalseDest() == targetBB)
      condBr.getFalseDestOperandsMutable().append(propVal);
  }
}

LogicalResult ModuloScheduleAdapter::adaptCFGWithLoopMS() {
  unsigned iterInd = 0;
  Block *curBlk;
  Block *loopCond = nullptr;
  Block *loopQuit = finiBlock;

  startBlock = builder.createBlock(templateBlock);
  curBlk = startBlock;
  // adapt the initBlock to the start block
  replaceSuccessor(initBlock, templateBlock, startBlock);

  // Initialize the block arguments to match the init block value
  // propagation
  for (auto arg : templateBlock->getArguments())
    startBlock->addArgument(arg.getType(), arg.getLoc());
  std::vector<std::pair<int, int>> emptySet;
  // startBlock = startBlk;
  newBlocks.push_back(startBlock);
  initOperationsInBB(startBlock, 0, timeSlotsOfBBs[0], emptySet);

  std::vector<std::pair<int, int>> propOpsToEpilog;
  std::vector<std::pair<int, int>> propOpsToProlog;

  for (auto [bbId, s] : llvm::enumerate(timeSlotsOfBBs)) {
    // already created the start block
    if (bbId == 0)
      continue;

    auto termIterId = getLatestIterId(prologOps, loopOpNum - 1, false);

    // get condition to loopCond and loopQuit
    auto origTerm = templateBlock->getTerminator();
    Value cmpOpr1 = origTerm->getOperand(0);
    Value cmpOpr2 = origTerm->getOperand(1);

    auto updateTermCmpOperand = [&](int cmpOprId, Value &cmpOpr) {
      if (cmpOprId >= 0) {
        cmpOpr = prologOps[termIterId][cmpOprId]->getResult(0);
        if (cmpOpr.getParentBlock() != curBlk && bbId == loopBlkId + 1) {
          curBlk->addArgument(cmpOpr.getType(), cmpOpr.getLoc());
          cmpOpr = curBlk->getArguments().back();
          propOpsToProlog.push_back({termIterId, cmpOprId});
          propOpsToProlog.push_back({termIterId + 1, cmpOprId});
        }
      }
    };

    updateTermCmpOperand(loopCmpOprId1, cmpOpr1);
    updateTermCmpOperand(loopCmpOprId2, cmpOpr2);

    if (bbId <= loopBlkId) {
      // create the epilog block for bbId-1's block for loop exit
      auto epiBlk = builder.createBlock(loopQuit);
      builder.setInsertionPointToStart(epiBlk);
      if (failed(completeUnexecutedOperationsInBB(epiBlk, bbId, termIterId,
                                                  prologOps, propOpsToEpilog)))
        return failure();
      newBlocks.push_back(epiBlk);

      // create the prolog block for bbId's block for loop continuation
      auto proBlk = builder.createBlock(templateBlock);
      builder.setInsertionPointToStart(proBlk);
      if (failed(initOperationsInBB(proBlk, bbId, s, propOpsToProlog)))
        return failure();
      newBlocks.push_back(proBlk);

      loopCond = proBlk;
      loopQuit = epiBlk;
    }

    // create terminator for bbId-1's block to connect the epilog block for
    // loop exit and corresponding prolog(kernel) block for loop continuation
    builder.setInsertionPointToEnd(curBlk);
    SmallVector<Value> contArgs;
    SmallVector<Value> finiArgs;
    // add arguments to finiBlock for the last iteration
    if (loopQuit == finiBlock) {
      // add the arguments to the finiBlock
      for (auto arg : finiBlock->getArguments()) {
        // get the opid from the template terminator
        auto opId = getOpId(loopOpList, getPropagatedValue(origTerm, finiBlock,
                                                           arg.getArgNumber())
                                            .getDefiningOp());
        auto iterId = getLatestIterId(prologOps, opId);
        finiArgs.push_back(prologOps[iterId][opId]->getResult(0));
      }
    }
    auto termOp = builder.create<cgra::ConditionalBranchOp>(
        curBlk->getOperations().back().getLoc(), cmpFlag, cmpOpr1, cmpOpr2,
        loopCond, contArgs, loopQuit, finiArgs);

    if (bbId != loopBlkId + 1 &&
        termOp.getFalseDest() != termOp->getBlock()->getNextNode())
      reverseCondBrFlag(termOp);

    prologOps[termIterId][loopOpNum - 1] = termOp;
    curBlk = loopCond;

    if (bbId == loopBlkId + 1)
      break;
  }

  // Propagate the values for the exit block of the kernel block
  for (auto [propKey1, propKey2] : propOpsToEpilog) {
    Operation *propOp;
    propOp = prologOps[propKey1][propKey2];
    Operation *controlOp = propOp->getBlock()->getTerminator();
    if (propKey1 == -1)
      controlOp = newBlocks.end()[-3]->getTerminator();
    addPropagateValue(controlOp, propOp->getResult(0), loopQuit);
  }

  // Propagate the values for the kernel block
  for (auto [propKey1, propKey2] : propOpsToProlog) {
    // Operation *propOp;
    Operation *propOp = prologOps[propKey1][propKey2];
    Operation *controlOp = propOp->getBlock()->getTerminator();
    if (propKey1 == -1)
      controlOp = newBlocks.end()[-3]->getTerminator();
    addPropagateValue(controlOp, propOp->getResult(0), loopCond);
  }

  removeTempletBlock();
  removeUselessBlockArg();
  return success();
}

/// Copy a block argument for another user use
void copyAnotherBlockArgument(BlockArgument arg, BlockArgument newArg,
                              SetVector<Block *> &visited, OpBuilder &builder) {
  Block *block = arg.getOwner();
  if (visited.count(block))
    return;
  visited.insert(block);

  // relatedVals.insert(val);
  unsigned ind = arg.getArgNumber();
  for (Block *pred : block->getPredecessors()) {
    Operation *branchOp = pred->getTerminator();
    Value operand = nullptr;
    if (auto br = dyn_cast<cf::BranchOp>(branchOp)) {
      operand = br.getOperand(ind);
      if (operand.isa<BlockArgument>()) {
        auto newArg = pred->addArgument(arg.getType(), arg.getLoc());
        copyAnotherBlockArgument(operand.cast<BlockArgument>(), newArg, visited,
                                 builder);
        continue;
      }
      auto defOp = operand.getDefiningOp();
      // copy the defOp and add it to the branch operands
      builder.setInsertionPointAfter(defOp);
      auto newOp = builder.clone(*defOp);
      // if the newOp use the arg for computation, replace it with the new
      // block argument
      for (auto &use : arg.getUses()) {
        if (use.getOwner() == newOp) {
          use.set(newArg);
        }
      }
      // attach the result of newOp to the branch operand
      SmallVector<Value, 4> newOperands(br.getOperands().begin(),
                                        br.getOperands().end());
      newOperands.push_back(newOp->getResult(0));
      branchOp->setOperands(newOperands);
    } else if (auto cbr = dyn_cast<cgra::ConditionalBranchOp>(branchOp)) {
      if (block == cbr.getTrueDest()) {
        operand = cbr.getTrueOperand(ind);
        if (operand.isa<BlockArgument>()) {
          auto newArg = pred->addArgument(arg.getType(), arg.getLoc());
          copyAnotherBlockArgument(operand.cast<BlockArgument>(), newArg,
                                   visited, builder);
          continue;
        }
        auto defOp = operand.getDefiningOp();
        // copy the defOp and add it to the branch operands
        builder.setInsertionPointAfter(defOp);
        auto newOp = builder.clone(*defOp);
        for (auto &use : arg.getUses()) {
          if (use.getOwner() == newOp) {
            use.set(newArg);
          }
        }
        cbr.getTrueDestOperandsMutable().append(newOp->getResult(0));

      } else {
        operand = cbr.getFalseOperand(ind);
        if (operand.isa<BlockArgument>()) {
          auto newArg = pred->addArgument(arg.getType(), arg.getLoc());
          copyAnotherBlockArgument(operand.cast<BlockArgument>(), newArg,
                                   visited, builder);
          continue;
        }
        auto defOp = operand.getDefiningOp();
        // copy the defOp and add it to the branch operands
        builder.setInsertionPointAfter(defOp);
        auto newOp = builder.clone(*defOp);
        for (auto &use : arg.getUses()) {
          if (use.getOwner() == newOp) {
            use.set(newArg);
          }
        }
        cbr.getFalseDestOperandsMutable().append(newOp->getResult(0));
      }
    }
  }
  return;
}

static void assignPrerequisite(Value val, Operation *consumerOp,
                               std::vector<std::pair<Value, int>> &existArgs,
                               OpBuilder &builder, ScheduleUnit schedule) {
  // if val is a block argument
  if (auto bbArg = dyn_cast_or_null<BlockArgument>(val)) {
    for (auto [arg, prereqPE] : existArgs) {
      if (prereqPE == schedule.pe) {
        val.replaceUsesWithIf(arg, [&](OpOperand &operand) {
          return operand.getOwner() == consumerOp;
        });
        return;
      }
    }
    auto origInd = bbArg.getArgNumber();
    // if cannot find reused value, add another block argument
    auto newArg =
        val.getParentBlock()->addArgument(val.getType(), val.getLoc());
    existArgs.push_back({newArg, schedule.pe});
    // copy the corresponding propagated value
    // TODO[@YYY26/Feb], code refactor
    SetVector<Block *> visited;
    copyAnotherBlockArgument(bbArg, newArg, visited, builder);
    val.replaceUsesWithIf(newArg, [&](OpOperand &operand) {
      return operand.getOwner() == consumerOp;
    });
    return;
  }

  auto defOp = val.getDefiningOp();
  for (auto [arg, prereqPE] : existArgs) {
    if (prereqPE == schedule.pe) {
      defOp->replaceUsesWithIf(arg.getDefiningOp(), [&](OpOperand &operand) {
        return operand.getOwner() == consumerOp;
      });
      return;
    }
  }

  // copy the value
  builder.setInsertionPointAfter(defOp);
  auto newOp = builder.clone(*defOp);
  defOp->replaceUsesWithIf(newOp, [&](OpOperand &operand) {
    return operand.getOwner() == consumerOp;
  });
  existArgs.push_back({newOp->getResult(0), schedule.pe});
}

static int seekAvailableSlot(
    const std::map<mlir::Operation *, compigra::ScheduleUnit> solution,
    Operation *assginOp, int preferPE, int maxPE, int time) {
  bool usePreferPE = true;
  std::vector<int> usedPEs;
  for (auto [op, schedule] : solution) {
    if (op->getBlock() == assginOp->getBlock() && schedule.time == time) {
      usedPEs.push_back(schedule.pe);
      if (schedule.pe == preferPE)
        usePreferPE = false;
    }
  }
  if (usePreferPE)
    return preferPE;

  for (int i = 0; i < maxPE; ++i)
    if (std::find(usedPEs.begin(), usedPEs.end(), i) == usedPEs.end())
      return i;

  return -1;
}

LogicalResult ModuloScheduleAdapter::assignScheduleResult(
    const std::map<int, Instruction> instructions, int maxReg, int maxPE) {
  // for operation in prologOps, its execution time is exectTime[opId] +
  // iterId * II
  int termPE = -1;

  // key to sort the live in arguments
  std::vector<Value> origLiveInArgs;

  // the first vector records the original live in arguments, and the second
  // vector records the value split from the original live in arguments and
  // their placement.
  std::vector<std::vector<std::pair<Value, int>>> liveInArgs;
  for (auto [iterId, opMap] : prologOps) {
    if (iterId == -1)
      continue;
    for (auto [opId, op] : opMap) {
      auto execTimeInBB = execTime.at(opId) + iterId * II;
      int reg = instructions.at(opId).Rout == maxReg ? maxReg : -1;
      ScheduleUnit schedule = {execTimeInBB, instructions.at(opId).pe, reg};
      solution[op] = schedule;
      if (opId == loopOpNum - 1)
        termPE = instructions.at(opId).pe;
      // assign the corresponding consumer with the schedule
      for (auto opr : op->getOperands()) {
        // if the operand has been resolved by the schedule solution, no need
        // to write them to the prerequisites
        if (std::find(newBlocks.begin(), newBlocks.end(),
                      opr.getParentBlock()) != newBlocks.end())
          continue;

        if (auto defOp = opr.getDefiningOp())
          if (isa<arith::ConstantOp>(defOp))
            continue;

        unsigned liveInArgId = origLiveInArgs.size();

        // whether the livein value has already been initialized
        if (std::find(origLiveInArgs.begin(), origLiveInArgs.end(), opr) ==
            origLiveInArgs.end()) {
          origLiveInArgs.push_back(opr);
          liveInArgs.push_back({});
        } else {
          liveInArgId = std::distance(
              origLiveInArgs.begin(),
              std::find(origLiveInArgs.begin(), origLiveInArgs.end(), opr));
        }
        if (liveInArgs[liveInArgId].size() == 0)
          liveInArgs[liveInArgId].push_back({opr, instructions.at(opId).pe});
        assignPrerequisite(opr, op, liveInArgs[liveInArgId], builder, schedule);
      }
    }
  }

  // for operations in epilogOpsBB, its execution time is execTime[opId] +
  // iterStartTime, where iterStartTime is the earliest time among
  // prologOps[iterId]
  for (auto [blk, opMap] : epilogOpsBB) {
    int endBBTime;

    for (auto [iterId, ops] : opMap) {
      if (iterId == -1)
        continue;
      // get earliest time of ops in solution
      int iterStartTime = INT_MAX;
      for (auto [opId, op] : prologOps.at(iterId)) {
        if (solution.find(op) != solution.end())
          iterStartTime = std::min(iterStartTime, solution[op].time);
      }
      endBBTime = iterStartTime;
      for (auto [opId, op] : ops) {
        auto execTimeInBB = execTime.at(opId) + iterStartTime;
        ScheduleUnit schedule = {execTimeInBB, instructions.at(opId).pe, -1};
        solution[op] = schedule;
        endBBTime = std::max(endBBTime, execTimeInBB);

        // assign the live-in value
        for (auto opr : op->getOperands()) {
          // if the operand has been resolved by the schedule solution, no need
          // to write them to the prerequisites
          if (std::find(newBlocks.begin(), newBlocks.end(),
                        opr.getParentBlock()) != newBlocks.end())
            continue;

          // if it is the blockArgument of this block, no need to assign
          if (std::find(blk->getArguments().begin(), blk->getArguments().end(),
                        opr) != blk->getArguments().end())
            continue;

          if (auto defOp = opr.getDefiningOp())
            if (isa<arith::ConstantOp>(defOp))
              continue;

          unsigned liveInArgId = origLiveInArgs.size();

          // whether the livein value has already been initialized
          if (std::find(origLiveInArgs.begin(), origLiveInArgs.end(), opr) ==
              origLiveInArgs.end()) {
            origLiveInArgs.push_back(opr);
            liveInArgs.push_back({});
          } else {
            liveInArgId = std::distance(
                origLiveInArgs.begin(),
                std::find(origLiveInArgs.begin(), origLiveInArgs.end(), opr));
          }
          if (liveInArgs[liveInArgId].size() == 0)
            liveInArgs[liveInArgId].push_back({opr, instructions.at(opId).pe});

          assignPrerequisite(opr, op, liveInArgs[liveInArgId], builder,
                             schedule);
        }
      }
    }
    // the last jump operation does not have a schedule, assign it with the
    // same conditional branch operation
    // if at endBBTime, termPE is free, assign it to termPE, otherwise, assign a
    // free PE
    int assignPE = seekAvailableSlot(solution, blk->getTerminator(), termPE,
                                     maxPE, endBBTime);
    ScheduleUnit schedule = {endBBTime, assignPE, -1};
    solution[blk->getTerminator()] = schedule;
  }

  // for element pair in liveInArgs, write them to prerequisites
  for (auto liveInArg : liveInArgs) {
    for (auto splitVal : liveInArg)
      prerequisites.push_back(splitVal);
  }
  return success();
}
