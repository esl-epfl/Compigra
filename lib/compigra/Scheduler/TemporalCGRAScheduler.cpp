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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <iomanip> // For std::setw

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

static SmallVector<Value, 2> getSrcOprandsOfPhi(BlockArgument arg,
                                                bool eraseUse = false) {
  SmallVector<Value, 2> srcOprands;
  Block *blk = arg.getOwner();
  unsigned argIndex = arg.getArgNumber();
  for (auto predBlk : blk->getPredecessors()) {
    Operation *termOp = predBlk->getTerminator();
    if (auto branchOp = dyn_cast_or_null<cf::BranchOp>(termOp)) {
      srcOprands.push_back(branchOp.getOperand(argIndex));
      if (eraseUse)
        branchOp.eraseOperand(argIndex);
    } else if (auto branchOp =
                   dyn_cast_or_null<cgra::ConditionalBranchOp>(termOp)) {
      if (predBlk == branchOp.getSuccessor(0)) {
        srcOprands.push_back(branchOp.getTrueOperand(argIndex));
        // remove argIndex from the false operand
        if (eraseUse)
          branchOp.eraseOperand(argIndex + 2);
      } else {
        srcOprands.push_back(branchOp.getFalseOperand(argIndex));
        if (eraseUse)
          branchOp.eraseOperand(argIndex + 2 + branchOp.getNumTrueOperands());
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

  for (auto [val, ind] : liveOutInter) {
    if (isPhiRelatedValue(val)) {
      SetVector<Value> relatedVals;
      getAllPhiRelatedValues(val, relatedVals);
      for (auto relatedVal : relatedVals)
        liveValInterPlaces.push_back({relatedVal, ind});
    } else {
      liveValInterPlaces.push_back({val, ind});
    }
  }
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
  // determine whether origVal is fixed point type or integer type
  Type valType = origVal.getType();

  Operation *movOp;
  if (isa<IntegerType>(valType)) {
    auto zero = builder.create<arith::ConstantIntOp>(user->getLoc(), 0,
                                                     origVal.getType());
    movOp = builder.create<arith::AddIOp>(user->getLoc(), origVal, zero);
  } else if (isa<Float32Type>(valType)) {
    auto zero = builder.create<arith::ConstantFloatOp>(
        user->getLoc(), APFloat(0.0f), origVal.getType().cast<Float32Type>());
    movOp = builder.create<arith::AddFOp>(user->getLoc(), origVal, zero);
  }

  user->replaceUsesOfWith(origVal, movOp->getResult(0));
}

void TemporalCGRAScheduler::insertInternalLSOps(Operation *srcOp,
                                                Operation *dstOp) {
  unsigned lastPtr = 0;
  auto &refOp = srcOp->getBlock()->getOperations().front();
  lastPtr = memStack.back().first + 4;

  // insert swi op after the srcOp
  builder.setInsertionPointAfter(srcOp);
  auto addr = builder.create<arith::ConstantIntOp>(refOp.getLoc(), lastPtr,
                                                   builder.getIntegerType(32));
  auto swi = builder.create<cgra::SwiOp>(srcOp->getLoc(), srcOp->getResult(0),
                                         addr->getResult(0));
  swi->setAttr("memLoc", builder.getI32IntegerAttr(lastPtr));

  // insert lwi op before the dstOp
  // builder.setInsertionPoint(dstOp);
  auto loadOp = builder.create<cgra::LwiOp>(
      swi->getLoc(), dstOp->getResult(0).getType(), addr->getResult(0));
  for (auto &use : llvm::make_early_inc_range(srcOp->getResult(0).getUses())) {
    Operation *user = use.getOwner();
    if (user == dstOp)
      user->setOperand(use.getOperandNumber(), loadOp->getResult(0));
  }
  opRAWs.insert({loadOp, swi});
}

std::map<Operation *, ScheduleUnit>
TemporalCGRAScheduler::getBlockSubSolution(Block *block) {
  std::map<Operation *, ScheduleUnit> subResult;
  for (auto [op, su] : solution) {
    if (op->getBlock() == block)
      subResult[op] = su;
  }
  return subResult;
}

void TemporalCGRAScheduler::placeLwiOpToBlock(Block *block, Operation *refOp,
                                              unsigned opIndex,
                                              cgra::LwiOp lwiOp) {
  llvm::errs() << "Not Implemented Error change pre-scheduled result\n";

  // bool isPhi =
  // print the existing schedule result
  std::map<Operation *, ScheduleUnit> subResult;
  std::vector<ScheduleUnit> subSchedule;
  for (auto [op, su] : solution) {
    if (op->getBlock() == block) {
      subResult[op] = su;
      subSchedule.push_back(su);
    }
  }

  // print the subResult according to su.time
  std::sort(subSchedule.begin(), subSchedule.end(),
            [](ScheduleUnit a, ScheduleUnit b) {
              return (a.time < b.time) || (a.time == b.time && a.pe < b.pe);
            });
  unsigned timeStart = 0;
  llvm::errs() << timeStart << "\n";
  // if su.time != timeStart, print \n
  for (auto su : subSchedule) {
    if (su.time != timeStart) {
      timeStart = su.time;
      llvm::errs() << "\n";
      llvm::errs() << timeStart << "\n";
    }
    llvm::errs() << su.pe << " ";
  }
  llvm::errs() << "\n";
  llvm::errs() << solution.at(refOp).time << " " << solution.at(refOp).pe
               << "\n";
  llvm::errs() << "===========================\n";
}

LogicalResult TemporalCGRAScheduler::placeLwiOpToBlock(Block *block,
                                                       BlockArgument arg,
                                                       cgra::LwiOp lwiOp) {
  // to load phi value, it must be the start of the block
  for (auto &[op, su] : solution) {
    if (op->getBlock() == block)
      su.time++;
  }
  // check where the arg located live
  auto it = std::find_if(
      liveValExterPlaces.begin(), liveValExterPlaces.end(),
      [&](std::pair<Value, unsigned> p) { return p.first == arg; });
  if (it != liveValExterPlaces.end()) {
    solution[lwiOp] = {0, (int)it->second, -1};
    return success();
  }

  it = std::find_if(
      liveValInterPlaces.begin(), liveValInterPlaces.end(),
      [&](std::pair<Value, unsigned> p) { return p.first == arg; });
  if (it != liveValInterPlaces.end()) {
    solution[lwiOp] = {0, (int)it->second, -1};
    return success();
  }
  return failure();
}

void TemporalCGRAScheduler::placeSwiOpToBlock(Block *block, Operation *refOp,
                                              cgra::SwiOp swiOp) {
  // get refOp's schedule result
  unsigned nextCycle = solution.at(refOp).time + 1;
  unsigned pe = solution.at(refOp).pe;

  unsigned left_pe = (pe - nCol + nRow * nCol) % (nRow * nCol);
  unsigned right_pe = (pe + nCol) % (nRow * nCol);
  unsigned top_pe = (pe - nCol) % (nRow * nCol);
  unsigned bottom_pe = (pe + nCol) % (nRow * nCol);

  std::vector<unsigned> peList = {pe, left_pe, right_pe, top_pe, bottom_pe};

  std::map<Operation *, ScheduleUnit> subResult;
  std::vector<ScheduleUnit> subSchedule;
  for (auto [op, su] : solution) {
    if (op->getBlock() == block) {
      subResult[op] = su;
      subSchedule.push_back(su);
    }
  }
  llvm::errs() << solution.at(refOp).time << " " << solution.at(refOp).pe
               << "\n";
  // check whether pe, left_pe, right_pe, top_pe and bottom_pe are available
  // during nextCycle.
  bool isAvailable = false;
  for (auto p : peList) {
    auto it = std::find_if(
        subSchedule.begin(), subSchedule.end(),
        [&](ScheduleUnit su) { return su.time == nextCycle && su.pe == p; });
    // if the pe is not occupied, place the swiOp to the pe
    if (it == subSchedule.end()) {
      solution[swiOp] = {(int)nextCycle, (int)p, -1};
      isAvailable = true;
      break;
    }
  }
  if (!isAvailable) {
    // place swiOp to the next available pe, delay all the schedule after
    // nextCycle
    for (auto &[op, su] : solution) {
      llvm::errs() << "delay " << op << "1 CC\n";
      if (op->getBlock() == block && su.time >= nextCycle)
        su.time++;
    }
    solution[swiOp] = {(int)nextCycle, (int)pe, -1};
  }
}

cgra::LwiOp TemporalCGRAScheduler::insertLoadOp(Operation *refOp, unsigned addr,
                                                Value origVal,
                                                unsigned opIndex) {
  llvm::errs() << "Create inserted load op\n";
  Block *userBlock = refOp->getBlock();
  unsigned blockIndex = std::distance(
      scheduleSeq.begin(),
      std::find(scheduleSeq.begin(), scheduleSeq.end(), userBlock));

  builder.setInsertionPoint(refOp);
  auto constOp = builder.create<arith::ConstantIntOp>(
      refOp->getLoc(), addr, builder.getIntegerType(32));
  Type valType = origVal.getType();
  auto loadOp = builder.create<cgra::LwiOp>(refOp->getLoc(), valType,
                                            constOp->getResult(0));
  if (blockIndex < scheduleIdx) {
    if (auto arg = dyn_cast_or_null<BlockArgument>(origVal)) {
      if (failed(placeLwiOpToBlock(userBlock, arg, loadOp)))
        return nullptr;
    } else {
      placeLwiOpToBlock(userBlock, refOp, opIndex, loadOp);
    }
  }
  return loadOp;
}

LogicalResult TemporalCGRAScheduler::splitDFGWithLSOps(Value origVal,
                                                       Operation *failUser,
                                                       unsigned memLoc,
                                                       bool processCntPhi) {
  llvm::errs() << origVal << " Conduct phi in memory: " << processCntPhi
               << "\n";

  auto refOp = origVal.getDefiningOp();
  if (refOp && failUser) {
    // process intra-bb split
    if (refOp->getBlock() == failUser->getBlock()) {
      insertInternalLSOps(refOp, failUser);
      llvm::errs() << "Intra-bb split\n";
    }
    return success();
  }

  if (!refOp)
    refOp = &(origVal.cast<BlockArgument>()
                  .getParentBlock()
                  ->getOperations()
                  .front());

  //  Seek whether swi op is already inserted
  auto memWriteOp = std::find_if(
      memStack.begin(), memStack.end(),
      [&](std::pair<unsigned, Value> p) { return p.second == origVal; });
  if (memWriteOp != memStack.end()) {
    llvm::errs() << "WARNING: Already inserted " << origVal << " at " << memLoc
                 << "\n";
    memLoc = memWriteOp->first;
  }

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
      auto loadOp =
          insertLoadOp(&suc->getOperations().front(), lastPtr, phiVal, true);
      // cannot find position to insert lwi op
      if (!loadOp)
        return failure();
      phiVal.replaceAllUsesWith(loadOp->getResult(0));
      suc->eraseArgument(phiVal.getArgNumber());

      // insert swi op for all source operands
      SmallVector<Value, 2> srcVals = getSrcOprandsOfPhi(phiVal, true);
      for (auto src : srcVals)
        if (failed(splitDFGWithLSOps(src, nullptr, lastPtr, false)))
          return failure();
    }
    return success();
  } else {
    // insert lwi op before the user block
    std::map<Block *, cgra::LwiOp> lwiOps;
    // If user is not specified, insert lwi op to all users
    for (auto &use : llvm::make_early_inc_range(origVal.getUses())) {
      Operation *user = use.getOwner();
      Block *userBlock = user->getBlock();

      llvm::errs() << "User: " << *user << "\n";
      // not process intra-bb split here
      if (userBlock == refOp->getBlock())
        continue;
      if (lwiOps.count(userBlock) != 0) {
        user->setOperand(use.getOperandNumber(), lwiOps[userBlock]);
        continue;
      }
      auto loadOp =
          insertLoadOp(user, lastPtr, origVal, use.getOperandNumber());
      user->setOperand(use.getOperandNumber(), loadOp->getResult(0));
      lwiOps[userBlock] = loadOp;
      llvm::errs() << "split " << loadOp << "\n";
    }
  }

  if (memWriteOp == memStack.end()) {
    if (isa<cf::BranchOp, cgra::ConditionalBranchOp, func::ReturnOp>(refOp))
      builder.setInsertionPoint(refOp);
    else
      builder.setInsertionPointAfter(refOp);

    unsigned blockIndex = std::distance(
        scheduleSeq.begin(),
        std::find(scheduleSeq.begin(), scheduleSeq.end(), refOp->getBlock()));

    auto addr = builder.create<arith::ConstantIntOp>(
        refOp->getLoc(), lastPtr, builder.getIntegerType(32));
    auto swi = builder.create<cgra::SwiOp>(refOp->getLoc(), origVal,
                                           addr->getResult(0));
    swi->setAttr("memLoc", builder.getI32IntegerAttr(lastPtr));
    if (blockIndex < scheduleIdx) {
      placeSwiOpToBlock(refOp->getBlock(), refOp, swi);
    }

    llvm::errs() << "split " << swi << "\n";
    memStack.push_back({lastPtr, origVal});
  }
  return success();
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

    int maxIter = 10;
    Operation *failUser = nullptr;
    Operation *checkptr = nullptr;
    Value spill = nullptr;
    int movNum = 1;
    scheduler.setFailureStrategy(FailureStrategy::Mov);
    while (maxIter > 0) {
      if (movNum < 0) {
        // step back, remove the additional sadd zero ops
        scheduler.setFailureStrategy(FailureStrategy::Split);
        llvm::errs()
            << "\n\n==================================strategy: split\n";
      }

      if (succeeded(scheduler.createSchedulerAndSolve()))
        break;

      if (scheduler.getFailureStrategy() == FailureStrategy::Mov) {
        handleMovAttempFailure(failUser, spill, scheduler, movNum, 3);
        spill = scheduler.getSpillVal();
        failUser = scheduler.getFailUser();
        scheduler.setCheckPoint(failUser);
        insertMovOp(spill, failUser);
      }
      if (scheduler.getFailureStrategy() == FailureStrategy::Split) {
        // split the liveOut value
        spill = scheduler.getSpillVal();
        failUser = scheduler.getFailUser();
        bool pushPhiToMem =
            isPhiRelatedValue(spill) && !spill.isa<BlockArgument>();
        if (failed(splitDFGWithLSOps(spill, failUser, UINT_MAX, pushPhiToMem)))
          return failure();
        computeLiveValue();
        scheduler.setLiveValue(liveIns[block], liveOuts[block]);
        scheduler.setLiveInPrerequisite(getExternalLiveIn(block),
                                        getInternalLiveIn(block));
        scheduler.setRAWPair(opRAWs);
      }
      if (scheduler.getFailureStrategy() == FailureStrategy::Abort) {
        llvm::errs() << "Optimization abort\n";
        return failure();
      }

      maxIter--;
    }

    if (maxIter == 0) {
      llvm::errs() << "Failed to schedule block " << bb << "\n";
      return failure();
    }
    saveSubILPModelResult(scheduler.getSolution());

    writeLiveOutResult(scheduler.getExternalLiveOutResult(),
                       scheduler.getInternalLiveOutResult());
    // if (bb == 10)
    //   break;

    scheduleIdx++;
  }

  calculateTemporalSpatialSchedule("temporalSpatialSchedule.csv");
  return success();
}

void TemporalCGRAScheduler::calculateTemporalSpatialSchedule(
    const std::string fileName) {
  unsigned kernelTime = 0;
  for (auto &block : region.getBlocks()) {
    int startTime = kernelTime;
    int endTime = kernelTime;
    blockStartT[&block] = startTime;
    for (auto &op : block.getOperations()) {
      if (solution.find(&op) == solution.end()) {
        if (!isa<arith::ConstantIntOp, arith::ConstantFloatOp>(op)) {
          llvm::errs() << "Operation " << op << " is not scheduled\n";
        }
        continue;
      }
      auto &su = solution[&op];
      su.time += kernelTime;
      endTime = std::max(endTime, su.time);
    }
    blockEndT[&block] = endTime + 1;
    kernelTime = endTime + 1;
    llvm::errs() << "start: " << startTime << " end: " << endTime << "\n";
  }

  std::ofstream csvFile(fileName);
  for (auto [op, su] : solution) {
    std::string str;
    llvm::raw_string_ostream rso(str);
    rso << *op;
    csvFile << rso.str() << "&" << su.time << "&" << su.pe << "\n";
  }
}

// Function to split a string by a delimiter
static std::vector<std::string> split(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

LogicalResult
TemporalCGRAScheduler::readScheduleResult(const std::string filename) {
  std::vector<std::vector<std::string>> data;
  std::map<std::string, ScheduleUnitBB> scheduleResult;
  std::ifstream file(filename);

  if (!file.is_open()) {
    llvm::errs() << "Error: Could not open the file " << filename << "\n";
    return failure();
  }

  std::string line;
  while (std::getline(file, line)) {
    // Split the line by commas and store the result
    std::vector<std::string> row = split(line, '&');
    scheduleResult[row[0]] = {std::stoi(row[1]), std::stoi(row[2])};
    data.push_back(row);
  }

  file.close();

  for (auto &op : region.getOps()) {
    std::string opName;
    llvm::raw_string_ostream rso(opName);
    rso << op;
    if (scheduleResult.find(opName) != scheduleResult.end()) {
      auto su = scheduleResult[opName];
      solution[&op] = {su.time, su.pe, -1};
      std::ostringstream oss;
      oss << std::left << std::setw(70) << opName << std::setw(10) << su.time
          << std::setw(10) << su.pe;
      llvm::errs() << oss.str() << "\n";
    }
  }
}
