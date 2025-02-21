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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace compigra;

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
unsigned compigra::getOpId(Block::OpListType &opList, Operation *search) {
  for (auto [ind, op] : llvm::enumerate(opList))
    if (&op == search)
      return ind;
  return -1;
}

/// Return the difference set
// static std::set<int> getDiffSet(std::set<int> set1, std::set<int> set2) {
//   std::set<int> diffSet;
//   for (auto val : set1)
//     if (set2.find(val) == set2.end())
//       diffSet.insert(val);
//   return diffSet;
// }

// static std::vector<std::set<int>> getDiffSet(std::vector<std::set<int>> set1,
//                                              std::vector<std::set<int>> set2)
//                                              {
//   for (auto it1 = set1.begin(); it1 != set1.end();) {
//     bool foundEqual = false;
//     for (const auto &s2 : set2) {
//       if (*it1 == s2) {
//         foundEqual = true;
//         break;
//       }
//     }
//     if (foundEqual) {
//       it1 = set1.erase(it1); // erase returns the next iterator
//     } else {
//       ++it1;
//     }
//   }
//   return set1;
// }

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
  // The last block in the timeSlotsOfBBs is the epilog block, and the second to
  // last block is the loop block
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
/// opMap stores all the executed operations with their iteration id and op id,
/// given a opId, get the latest iteration id for the op
static int
getLatestIterId(const std::map<int, std::map<int, Operation *>> opMap, int opId,
                bool getExisted = true) {
  int iterId = 0;
  while (opMap.count(iterId) && opMap.at(iterId).count(opId)) {
    iterId++;
  }
  return getExisted ? iterId - 1 : iterId;
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
    Location loc = blk->getOperations().empty()
                       ? blk->getPrevNode()->getTerminator()->getLoc()
                       : blk->getOperations().front().getLoc();

    blk->addArgument(argType, loc);
    adaptOp->setOperand(operandId, blk->getArguments().back());

    auto src1 = getLatestIterId(mergedOpMap, propOpId);
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
      break;
    for (unsigned opId = 0; opId < loopOpNum - 1; ++opId) {
      if (opMap.count(opId) == 0) {
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
  }

  // the epilog block would directly jump to the fini block
  // insert an unconditional branch to the start of the block
  builder.setInsertionPointToEnd(blk);
  builder.create<cf::BranchOp>(blk->getOperations().back().getLoc(), finiBlock);
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

void addPropagateValue(Operation *sucTermOp, Operation *propOp,
                       Block *targetBB) {
  if (auto br = dyn_cast<cf::BranchOp>(sucTermOp)) {
    if (br.getSuccessor() == targetBB) {
      SmallVector<Value, 4> newOperands(br.getOperands().begin(),
                                        br.getOperands().end());
      newOperands.push_back(propOp->getResult(0));
      br.getOperation()->setOperands(newOperands);
    }
  } else if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(sucTermOp)) {
    if (condBr.getTrueDest() == targetBB)
      condBr.getTrueDestOperandsMutable().append(propOp->getResult(0));
    else if (condBr.getFalseDest() == targetBB)
      condBr.getFalseDestOperandsMutable().append(propOp->getResult(0));
  }
}

LogicalResult ModuloScheduleAdapter::adaptCFGWithLoopMS() {
  unsigned iterInd = 0;
  Block *curBlk;
  Block *loopCond = nullptr;
  Block *loopQuit;

  auto startBlk = builder.createBlock(templateBlock);
  curBlk = startBlk;
  // adapt the initBlock to the start block
  replaceSuccessor(initBlock, templateBlock, startBlk);

  // Initialize the block arguments to match the init block value
  // propagation
  for (auto arg : templateBlock->getArguments())
    startBlk->addArgument(arg.getType(), arg.getLoc());
  std::vector<std::pair<int, int>> emptySet;
  startBlock = startBlk;
  initOperationsInBB(startBlk, 0, timeSlotsOfBBs[0], emptySet);

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
    if (loopCmpOprId1 > 0)
      cmpOpr1 = prologOps[termIterId][loopCmpOprId1]->getResult(0);

    if (loopCmpOprId2 > 0)
      cmpOpr2 = prologOps[termIterId][loopCmpOprId2]->getResult(0);

    if (bbId <= loopBlkId) {
      // create the epilog block for bbId-1's block for loop exit
      auto epiBlk = builder.createBlock(finiBlock);
      builder.setInsertionPointToStart(epiBlk);
      if (failed(completeUnexecutedOperationsInBB(epiBlk, bbId, termIterId,
                                                  prologOps, propOpsToEpilog)))
        return failure();

      // create the prolog block for bbId's block for loop continuation
      auto proBlk = builder.createBlock(templateBlock);

      builder.setInsertionPointToStart(proBlk);
      if (failed(initOperationsInBB(proBlk, bbId, s, propOpsToProlog)))
        return failure();

      loopCond = proBlk;
      loopQuit = epiBlk;
    }

    // create terminator for bbId-1's block to connect the epilog block for
    // loop exit and corresponding prolog(kernel) block for loop continuation
    builder.setInsertionPointToEnd(curBlk);
    auto termOp = builder.create<cgra::ConditionalBranchOp>(
        curBlk->getOperations().back().getLoc(), cmpFlag, cmpOpr1, cmpOpr2,
        loopCond, loopQuit);
    if (termOp.getFalseDest() != termOp->getBlock()->getNextNode())
      reverseCondBrFlag(termOp, true);
    prologOps[termIterId][loopOpNum - 1] = termOp;
    curBlk = loopCond;

    if (bbId == loopBlkId + 1)
      break;
  }

  // Propagate the values for the exit block of the kernel block
  for (auto [propKey1, propKey2] : propOpsToEpilog) {
    Operation *propOp;
    propOp = prologOps[propKey1][propKey2];
    addPropagateValue(propOp->getBlock()->getTerminator(), propOp, loopQuit);
  }

  // Propagate the values for the kernel block
  for (auto [propKey1, propKey2] : propOpsToProlog) {
    auto propOp = prologOps[propKey1][propKey2];
    addPropagateValue(propOp->getBlock()->getTerminator(), propOp, loopCond);
  }

  removeTempletBlock();
  removeUselessBlockArg();
  return success();
}

LogicalResult ModuloScheduleAdapter::assignScheduleResult(
    const std::map<int, Instruction> instructions) {
  // for operation in prologOps, its execution time is exectTime[opId] +
  // iterId * II
  for (auto [iterId, opMap] : prologOps) {
    for (auto [opId, op] : opMap) {
      auto execTimeInBB = execTime.at(opId) + iterId * II;
      ScheduleUnit schedule = {execTimeInBB, instructions.at(opId).pe, -1};
      solution[op] = schedule;
    }
  }

  // for operations in epilogOpsBB, its execution time is execTime[opId] +
  // iterStartTime, where iterStartTime is the earliest time among
  // prologOps[iterId]
  for (auto [_, opMap] : epilogOpsBB) {
    for (auto [iterId, ops] : opMap) {
      // get earliest time of ops in solution
      int iterStartTime = INT_MAX;
      for (auto [opId, op] : prologOps.at(iterId)) {
        if (solution.find(op) != solution.end())
          iterStartTime = std::min(iterStartTime, solution[op].time);
      }
      for (auto [opId, op] : ops) {
        auto execTimeInBB = execTime.at(opId) + iterStartTime;
        ScheduleUnit schedule = {execTimeInBB, instructions.at(opId).pe, -1};
        solution[op] = schedule;
      }
    }
  }
  return success();
}
