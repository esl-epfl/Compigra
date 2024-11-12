//===- TemporalCGRAScheduler.cpp - Implement the class/functions for 2D temporal
// spatial schedule for temporal CGRAs *- C++-* --------------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements class for temporal CGRA schedule functions.
//
//===----------------------------------------------------------------------===//

#include "compigra/Scheduler/TemporalCGRAScheduler.h"
#include "fstream"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace compigra;

/// Insert a value to a set if it is not a constant. The constant value is not
/// considered as a live value.
static void insertToSet(Value val, SetVector<Value> &vec) {
  if (dyn_cast_or_null<arith::ConstantIntOp>(val.getDefiningOp()) ||
      dyn_cast_or_null<arith::ConstantFloatOp>(val.getDefiningOp()))
    return;
  vec.insert(val);
}

/// Update the liveOut set of `blk` by adding the liveIn value of its
/// successor blocks. If the liveIn value is a block argument (phi node), add
/// the corresponding value in the predecessor block.
static void updateLiveOutBySuccessorLiveIn(Value val, Block *blk,
                                           SetVector<Value> &liveOut) {
  if (auto arg = dyn_cast_or_null<BlockArgument>(val)) {
    Block *argBlk = arg.getOwner();

    auto termOp = blk->getTerminator();
    if (auto branchOp = dyn_cast_or_null<cf::BranchOp>(termOp)) {
      if (argBlk == branchOp.getSuccessor()) {
        unsigned argIndex = arg.getArgNumber();
        liveOut.insert(branchOp.getOperand(argIndex));
        return;
      }
    } else if (auto branchOp =
                   dyn_cast_or_null<cgra::ConditionalBranchOp>(termOp)) {
      if (argBlk == branchOp.getSuccessor(0)) {
        unsigned argIndex = arg.getArgNumber();

        liveOut.insert(branchOp.getTrueOperand(argIndex));
        return;
      } else if (argBlk == branchOp.getSuccessor(1)) {
        unsigned argIndex = arg.getArgNumber();
        liveOut.insert(branchOp.getFalseOperand(argIndex));
        return;
      }
    }
  }

  liveOut.insert(val);
}

static SmallVector<Block *, 4> getCntBlocksThroughPhi(Value val) {
  SmallVector<Block *, 4> cntBlocks;
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    if (isa<cf::BranchOp>(user)) {
      Block *succBlk = user->getSuccessor(0);
      cntBlocks.push_back(succBlk);
    }
    if (isa<cgra::ConditionalBranchOp>(user)) {
      unsigned argIndex = use.getOperandNumber();
      if (argIndex < 2)
        continue;

      if (argIndex >=
          2 + dyn_cast<cgra::ConditionalBranchOp>(user).getNumTrueOperands())
        cntBlocks.push_back(user->getSuccessor(1));
      else
        cntBlocks.push_back(user->getSuccessor(0));
    }
  }
  return cntBlocks;
}

static BlockArgument getCntBlockArgument(Value val, Block *succBlk) {
  // search for the connected block argument
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    if (isa<cf::BranchOp>(user)) {
      unsigned argIndex = use.getOperandNumber();
      if (user->getSuccessor(0) == succBlk)
        return succBlk->getArgument(argIndex);
    }
    if (isa<cgra::ConditionalBranchOp>(user)) {
      unsigned argIndex = use.getOperandNumber();
      if (argIndex < 2)
        continue;
      // true successor
      if (user->getSuccessor(0) == succBlk)
        return succBlk->getArgument(argIndex - 2);
      // false successor
      if (user->getSuccessor(1) == succBlk)
        return succBlk->getArgument(
            argIndex - 2 -
            dyn_cast<cgra::ConditionalBranchOp>(user).getNumTrueOperands());
    }
  }
  // no matching block argument
  return nullptr;
}

static SmallVector<Value, 2> getSrcOprandsOfPhi(BlockArgument arg) {
  SmallVector<Value, 2> srcOprands;
  Block *blk = arg.getOwner();
  unsigned argIndex = arg.getArgNumber();
  for (auto predBlk : blk->getPredecessors()) {
    Operation *termOp = predBlk->getTerminator();
    if (auto branchOp = dyn_cast_or_null<cf::BranchOp>(termOp)) {
      srcOprands.push_back(branchOp.getOperand(argIndex));
    } else if (auto branchOp =
                   dyn_cast_or_null<cgra::ConditionalBranchOp>(termOp)) {
      if (predBlk == branchOp.getSuccessor(0)) {
        srcOprands.push_back(branchOp.getTrueOperand(argIndex));
      } else {
        srcOprands.push_back(branchOp.getFalseOperand(argIndex));
      }
    }
  }
  return srcOprands;
}

void TemporalCGRAScheduler::computeLiveValue() {
  // compute def and use for each block
  std::map<Block *, SetVector<Value>> defMap;
  std::map<Block *, SetVector<Value>> useMap;

  for (auto &block : region) {
    SetVector<Value> def;
    SetVector<Value> use;
    // push all block arguments to use
    for (auto arg : block.getArguments())
      insertToSet(arg, use);

    for (auto &op : block.getOperations()) {
      for (auto res : op.getResults())
        insertToSet(res, def);

      for (auto opr : op.getOperands())
        //   branch argument is not a use
        insertToSet(opr, use);
    }
    defMap[&block] = def;
    useMap[&block] = use;
  }

  // calculate (use - def)
  std::map<Block *, SetVector<Value>> outBBUse;
  for (auto &block : region) {
    SetVector<Value> outUse;
    for (auto V : useMap[&block]) {
      if (!defMap[&block].count(V)) {
        outUse.insert(V);
      }
    }
    outBBUse[&block] = outUse;
  }

  // clear liveIn and liveOut
  liveIns.clear();
  liveOuts.clear();

  // compute liveIn and liveOut for each block
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &block : region) {
      SetVector<Value> liveIn = outBBUse[&block];
      SetVector<Value> liveOut = liveOuts[&block];

      // liveIn = outBBUse + (liveOut - def)
      for (auto val : liveOut)
        if (!defMap[&block].count(val))
          insertToSet(val, liveIn);

      for (auto succ : block.getSuccessors()) {
        // ====================TBRM====================
        // auto it = region.getBlocks().begin();
        // auto it_ptr = it;
        // std::advance(it_ptr, 4);
        // auto it_suc_ptr = it;
        // std::advance(it_suc_ptr, 5);
        // auto &dst_blk = *it_ptr;
        // auto &succ_blk = *it_suc_ptr;
        // ====================TBRM====================

        // add to succesor's liveOut
        for (auto val : liveIns[succ]) {
          // ====================TBRM====================
          //   if (&block == &dst_blk && (succ == &succ_blk) &&
          //       val.isa<BlockArgument>()) {
          //     for (auto [ind, bb] : llvm::enumerate(region))
          //       if (&bb == val.getParentBlock()) {
          //         llvm::errs() << "~~~~liveIn: " << ind << "\n";
          //         break;
          //       }
          //   }
          // ====================TBRM====================

          updateLiveOutBySuccessorLiveIn(val, &block, liveOut);
        }

        // ====================TBRM====================
        // if (&block == &dst_blk && (succ == &succ_blk)) {
        //   for (auto val : liveOut) {
        //     if (val.isa<BlockArgument>()) {
        //       for (auto [ind, bb] : llvm::enumerate(region))
        //         if (&bb == val.getParentBlock()) {
        //           llvm::errs() << "~~~ " << ind << " ";
        //           break;
        //         }
        //     }
        //     llvm::errs() << "~~~ " << val << "\n";
        //   }
        // }
        // ====================TBRM====================
      }
      if (liveIn != liveIns[&block] || liveOut != liveOuts[&block]) {
        liveIns[&block] = liveIn;
        liveOuts[&block] = liveOut;
        changed = true;
      }
    }
  }

  printBlockLiveValue("liveValue.txt");
}

void TemporalCGRAScheduler::printBlockLiveValue(std::string fileName) {
  // Open output file stream
  std::ofstream outFile(fileName);
  if (!outFile.is_open()) {
    llvm::errs() << "Error: Could not open file " << fileName << "\n";
    return;
  }

  unsigned blockNum = 0;
  // print liveIn and liveOut
  for (auto &block : region) {
    outFile << "Block: " << blockNum << "\n";
    outFile << "LiveIn: ";
    for (auto val : liveIns[&block]) {
      if (val.isa<BlockArgument>()) {
        for (auto [ind, bb] : llvm::enumerate(region))
          if (&bb == val.getParentBlock()) {
            outFile << ind << " ";
            break;
          }
      }
      std::string str;
      llvm::raw_string_ostream rso(str);
      rso << val;
      outFile << rso.str() << "\n";
    }
    outFile << "LiveOut: ";
    for (auto val : liveOuts[&block]) {
      if (val.isa<BlockArgument>()) {
        for (auto [ind, bb] : llvm::enumerate(region))
          if (&bb == val.getParentBlock()) {
            outFile << ind << " ";
            break;
          }
      }
      std::string str;
      llvm::raw_string_ostream rso(str);
      rso << val;
      outFile << rso.str() << "\n";
    }
    outFile << "\n";
    blockNum++;
  }

  outFile.close();
}

static void getAllPhiRelatedValues(Value val, SetVector<Value> &relatedVals) {
  if (relatedVals.count(val))
    return;

  if (val.isa<BlockArgument>()) {
    relatedVals.insert(val);
    unsigned ind = val.cast<BlockArgument>().getArgNumber();
    Block *block = val.getParentBlock();
    for (Block *pred : block->getPredecessors()) {
      Operation *branchOp = pred->getTerminator();
      Value operand = nullptr;
      if (auto br = dyn_cast<cf::BranchOp>(branchOp)) {
        operand = br.getOperand(ind);
      } else if (auto cbr = dyn_cast<cgra::ConditionalBranchOp>(branchOp)) {
        if (block == cbr.getTrueDest())
          operand = cbr.getTrueOperand(ind);
        else
          operand = cbr.getFalseOperand(ind);
      }
      relatedVals.insert(operand);
      // recursively get the related values
      getAllPhiRelatedValues(operand, relatedVals);
    }
    return;
  }

  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    llvm::errs() << "Can not track " << val << " for phi related chain\n";
    return;
  }
  relatedVals.insert(val);
  for (auto &use : val.getUses()) {
    unsigned index = use.getOperandNumber();
    if (auto br = dyn_cast_or_null<cf::BranchOp>(use.getOwner())) {
      getAllPhiRelatedValues(br->getSuccessor(0)->getArgument(index),
                             relatedVals);
    }
    if (auto cbr =
            dyn_cast_or_null<cgra::ConditionalBranchOp>(use.getOwner())) {
      bool isTrueOpr = index >= 2 && index < 2 + cbr.getNumTrueDestOperands();
      if (isTrueOpr) {
        getAllPhiRelatedValues(cbr.getTrueDest()->getArgument(index - 2),
                               relatedVals);
      } else {
        getAllPhiRelatedValues(cbr.getFalseDest()->getArgument(
                                   index - cbr.getNumTrueDestOperands() - 2),
                               relatedVals);
      }
    }
  }

  // return relatedVals;
}

bool compigra::isPhiRelatedValue(Value val) {
  if (val.isa<BlockArgument>())
    return true;

  for (auto &use : val.getUses()) {
    // cf.br carries the block argument
    if (isa<cf::BranchOp>(use.getOwner()))
      return true;

    // cgra.cond_br carries the block argument after the condition
    if (isa<cgra::ConditionalBranchOp>(use.getOwner()) &&
        use.getOperandNumber() > 1)
      return true;
  }
  return false;
}

void TemporalCGRAScheduler::writeLiveOutResult(const liveVec liveOutExter,
                                               const liveVec liveOutInter) {
  for (auto [val, ind] : liveOutExter) {
    if (isPhiRelatedValue(val)) {
      SetVector<Value> relatedVals;
      getAllPhiRelatedValues(val, relatedVals);
      for (auto relatedVal : relatedVals)
        liveValExterPlaces.push_back({relatedVal, ind});

    } else {
      liveValExterPlaces.push_back({val, ind});
    }
  }

  for (auto [val, ind] : liveOutInter)
    liveValInterPlaces.push_back({val, ind});
}

liveVec TemporalCGRAScheduler::getExternalLiveIn(Block *block) {
  auto bbLiveIn = liveIns[block];
  liveVec liveInExter;

  for (auto val : bbLiveIn) {
    // search whether the val is in the liveOutExterPlaces
    auto it = std::find_if(
        liveValExterPlaces.begin(), liveValExterPlaces.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValExterPlaces.end()) {
      auto index = it->second;
      liveInExter.push_back({val, index});
    }
  }
  return liveInExter;
}

liveVec TemporalCGRAScheduler::getInternalLiveIn(Block *block) {
  auto bbLiveIn = liveIns[block];
  liveVec liveInInter;

  for (auto val : bbLiveIn) {
    // search whether the val is in the liveOutInterPlaces
    auto it = std::find_if(
        liveValInterPlaces.begin(), liveValInterPlaces.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValInterPlaces.end()) {
      auto index = it->second;
      liveInInter.push_back({val, index});
    }
  }
  return liveInInter;
}

void TemporalCGRAScheduler::saveSubILPModelResult(
    const std::map<Operation *, ScheduleUnitBB> res) {
  for (auto [op, su] : res) {
    ScheduleUnit res = {su.time, su.pe, -1};
    solution[op] = res;
  }
}

void TemporalCGRAScheduler::insertMovOp(Value origVal, Operation *user) {
  builder.setInsertionPoint(user);
  auto zero = builder.create<arith::ConstantIntOp>(user->getLoc(), 0,
                                                   builder.getIntegerType(32));
  auto movOp = builder.create<arith::AddIOp>(user->getLoc(), origVal, zero);
  user->replaceUsesOfWith(origVal, movOp->getResult(0));
}

void TemporalCGRAScheduler::placeLSOpsToBlock(Block *block) {
  llvm::errs() << "Not Implemented Error change pre-scheduled result\n";
}

void TemporalCGRAScheduler::insertLSOps(Value origVal, unsigned memLoc,
                                        bool processCntPhi) {

  unsigned lastPtr = memLoc == UINT_MAX ? 0 : memLoc;
  if (!memStack.empty() && memLoc == UINT_MAX) {
    lastPtr = memStack.back().first + 4;
  }

  if (processCntPhi) {
    SmallVector<Block *, 4> succBlocks = getCntBlocksThroughPhi(origVal);
    for (auto suc : succBlocks) {
      // get the phi value
      auto phiVal = getCntBlockArgument(origVal, suc);
      // insert lwi to replace the phi value
      builder.setInsertionPoint(&suc->getOperations().front());
      auto constOp = builder.create<arith::ConstantIntOp>(
          suc->getOperations().front().getLoc(), lastPtr,
          builder.getIntegerType(32));
      auto loadOp =
          builder.create<cgra::LwiOp>(suc->getOperations().front().getLoc(),
                                      origVal.getType(), constOp->getResult(0));
      phiVal.replaceAllUsesWith(loadOp->getResult(0));
      // insert swi op for all source operands
      SmallVector<Value, 2> srcVals = getSrcOprandsOfPhi(phiVal);
      for (auto src : srcVals) {
        insertLSOps(src, lastPtr, false);
        // erase BlockArgument from the IR
      }
    }
    return;
  }

  auto refOp = origVal.getDefiningOp();
  if (!refOp)
    refOp = &(origVal.cast<BlockArgument>()
                  .getParentBlock()
                  ->getOperations()
                  .front());

  // insert lwi op before the user block
  std::map<Block *, cgra::LwiOp> lwiOps;
  // If user is not specified, insert lwi op to all users
  for (auto &use : llvm::make_early_inc_range(origVal.getUses())) {
    Operation *user = use.getOwner();

    llvm::errs() << "User: " << *user << "\n";
    Block *userBlock = user->getBlock();
    unsigned blockIndex = std::distance(
        scheduleSeq.begin(),
        std::find(scheduleSeq.begin(), scheduleSeq.end(), userBlock));
    // TODO[@YX]: avoid insert unnecessary lwi ops
    if (userBlock == origVal.getParentBlock())
      continue;
    if (blockIndex < scheduleIdx) {
      placeLSOpsToBlock(userBlock);
      // return;
    }

    if (lwiOps.count(userBlock) != 0) {
      user->setOperand(use.getOperandNumber(), lwiOps[userBlock]);
      continue;
    }
    builder.setInsertionPoint(&userBlock->getOperations().front());
    auto addr = builder.create<arith::ConstantIntOp>(
        refOp->getLoc(), lastPtr, builder.getIntegerType(32));
    auto loadOp =
        builder.create<cgra::LwiOp>(userBlock->getOperations().front().getLoc(),
                                    origVal.getType(), addr->getResult(0));
    user->setOperand(use.getOperandNumber(), loadOp->getResult(0));
    lwiOps[userBlock] = loadOp;
  }

  builder.setInsertionPointAfter(refOp);
  // TODO: Seek whether swi op is already inserted
  auto addr = builder.create<arith::ConstantIntOp>(refOp->getLoc(), lastPtr,
                                                   builder.getIntegerType(32));
  builder.create<cgra::SwiOp>(refOp->getLoc(), origVal, addr->getResult(0));
  memStack.push_back({lastPtr, origVal});
}

static void handleMovAttempFailure(Operation *preFailOp, Value preSpillVal,
                                   BasicBlockILPModel &scheduler, int &movNum,
                                   unsigned maxTry) {
  Operation *failUser = scheduler.getFailUser();
  if (failUser != preFailOp) {
    scheduler.setFailureStrategy(FailureStrategy::Mov);
    movNum = maxTry;
    return;
  }

  Value spill = scheduler.getSpillVal();
  auto definingOp = spill.getDefiningOp();
  Operation *movOp = dyn_cast_or_null<arith::AddIOp>(definingOp);
  if (!movOp) {
    movOp = dyn_cast_or_null<arith::AddFOp>(definingOp);
  }

  // not the same failure, reset the movNum
  if (!movOp || movOp->getOperand(0) != preSpillVal) {
    scheduler.setFailureStrategy(FailureStrategy::Mov);
    movNum = maxTry;
    return;
  }

  auto cstOp = movOp->getOperand(1).getDefiningOp();
  if (!cstOp || !dyn_cast_or_null<arith::ConstantOp>(cstOp) ||
      dyn_cast<arith::ConstantOp>(cstOp)
              .getValue()
              .dyn_cast<IntegerAttr>()
              .getInt() != 0) {
    scheduler.setFailureStrategy(FailureStrategy::Mov);
    movNum = maxTry;
    return;
  }

  // the same failure, try again and decrease the movNum
  movNum--;
}

void TemporalCGRAScheduler::makeScheduleSeq() {
  scheduleSeq.clear();
  scheduleIdx = 0;
  for (auto &bb : region.getBlocks())
    scheduleSeq.push_back(&bb);
}

LogicalResult TemporalCGRAScheduler::createSchedulerAndSolve() {
  // Create scheduler for each block
  makeScheduleSeq();
  computeLiveValue();
  for (auto [bb, block] : llvm::enumerate(scheduleSeq)) {
    llvm::errs() << "\nBlock " << bb << " is scheduling\n";

    BasicBlockILPModel scheduler(maxReg, nRow, nCol, block, bb, builder);
    scheduler.setLiveValue(liveIns[block], liveOuts[block]);
    scheduler.setLiveInPrerequisite(getExternalLiveIn(block),
                                    getInternalLiveIn(block));

    int maxIter = 3;
    Operation *failUser = nullptr;
    Value spill = nullptr;
    int movNum = 1;
    scheduler.setFailureStrategy(FailureStrategy::Mov);
    while (maxIter > 0) {
      if (movNum < 0) {
        // step back, remove the additional sadd zero ops
        scheduler.setFailureStrategy(FailureStrategy::Split);
      }

      if (failed(scheduler.createSchedulerAndSolve())) {
        if (scheduler.getFailureStrategy() == FailureStrategy::Mov) {
          handleMovAttempFailure(failUser, spill, scheduler, movNum, 3);
          spill = scheduler.getSpillVal();
          failUser = scheduler.getFailUser();
          insertMovOp(spill, failUser);
          llvm::errs() << "Spill " << spill << " for "
                       << *(scheduler.getFailUser()) << "\n";
        }
        if (scheduler.getFailureStrategy() == FailureStrategy::Split) {
          // split the liveOut value
          spill = scheduler.getSpillVal();
          bool pushPhiToMem =
              isPhiRelatedValue(spill) && !spill.isa<BlockArgument>();
          insertLSOps(spill, UINT_MAX, pushPhiToMem);
          if (pushPhiToMem) {
            // observe the asm
            return success();
          }
          computeLiveValue();
          scheduler.setLiveValue(liveIns[block], liveOuts[block]);
          scheduler.setLiveInPrerequisite(getExternalLiveIn(block),
                                          getInternalLiveIn(block));
        }

        maxIter--;
        continue;
      }
      break;
    }

    if (maxIter == 0) {
      llvm::errs() << "Failed to schedule block " << bb << "\n";
      return failure();
    }
    saveSubILPModelResult(scheduler.getSolution());

    writeLiveOutResult(scheduler.getExternalLiveOutResult(),
                       scheduler.getInternalLiveOutResult());
    if (bb == 10)
      break;

    scheduleIdx++;
  }
  // print solution
  llvm::errs() << "\n====================\n";
  for (auto [op, su] : solution) {
    llvm::errs() << "Operation: " << *op << " Time: " << su.time
                 << " PE: " << su.pe << "\n";
  }
  return success();
}
