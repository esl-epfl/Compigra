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
        // add to succesor's liveOut
        for (auto val : liveIns[succ])
          updateLiveOutBySuccessorLiveIn(val, &block, liveOut);
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

static void pushResultToLiveVec(liveVec &liveVec, Value val, unsigned index) {
  // if find the value in the liveVec, update the index
  auto it = std::find_if(
      liveVec.begin(), liveVec.end(),
      [&](std::pair<Value, unsigned> p) { return p.first == val; });
  if (it != liveVec.end()) {
    it->second = index;
  } else {
    liveVec.push_back({val, index});
  }
}

void TemporalCGRAScheduler::writeLiveOutResult(const liveVec liveOutExter,
                                               const liveVec liveOutInter) {
  for (auto [val, ind] : liveOutExter) {
    if (isPhiRelatedValue(val)) {
      SetVector<Value> relatedVals;
      getAllPhiRelatedValues(val, relatedVals);
      for (auto relatedVal : relatedVals)
        pushResultToLiveVec(liveValExterPlaces, relatedVal, ind);
    } else {
      pushResultToLiveVec(liveValExterPlaces, val, ind);
    }
  }

  for (auto [val, ind] : liveOutInter) {
    if (isPhiRelatedValue(val)) {
      SetVector<Value> relatedVals;
      getAllPhiRelatedValues(val, relatedVals);
      for (auto relatedVal : relatedVals)
        pushResultToLiveVec(liveValInterPlaces, relatedVal, ind);
    } else {
      pushResultToLiveVec(liveValInterPlaces, val, ind);
    }
  }
}

liveVec TemporalCGRAScheduler::getExternalLiveIn(Block *block) {
  auto bbLiveIn = liveIns[block];
  liveVec liveInExter;
  // TODO[@YW]: add the function for determine whether the val should be
  // external live in

  // for (auto val : bbLiveIn) {
  //   // search whether the val is in the liveOutExterPlaces
  //   auto it = std::find_if(
  //       liveValExterPlaces.begin(), liveValExterPlaces.end(),
  //       [&](std::pair<Value, unsigned> p) { return p.first == val; });

  //   if (it != liveValExterPlaces.end()) {
  //     auto index = it->second;
  //     liveInExter.push_back({val, index});
  //   }
  // }
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
    } else {
      liveInInter.push_back({val, UINT32_MAX});
    }
  }
  return liveInInter;
}

liveVec TemporalCGRAScheduler::getExternalLiveOut(Block *block) {
  auto bbLiveOut = liveOuts[block];
  liveVec liveOutExter;
  // TODO[@YW]: add the function for determine whether the val should be
  // external live out
  return liveOutExter;
}

liveVec TemporalCGRAScheduler::getInternalLiveOut(Block *block) {
  auto bbLiveOut = liveOuts[block];
  liveVec liveOutInter;

  for (auto val : bbLiveOut) {
    // search whether the val is in the liveOutInterPlaces
    auto it = std::find_if(
        liveValInterPlaces.begin(), liveValInterPlaces.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValInterPlaces.end()) {
      auto index = it->second;
      liveOutInter.push_back({val, index});
    } else {
      liveOutInter.push_back({val, UINT32_MAX});
    }
  }
  return liveOutInter;
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
  if (!memStack.empty())
    lastPtr = memStack.back().first + 4;
  memStack.push_back({lastPtr, srcOp->getResult(0)});

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
    // TODO[@YW]: determine whether inter or exter
    pushResultToLiveVec(liveValExterPlaces, lwiOp->getResult(0),
                        (int)it->second);
    return success();
  }

  it = std::find_if(
      liveValInterPlaces.begin(), liveValInterPlaces.end(),
      [&](std::pair<Value, unsigned> p) { return p.first == arg; });
  if (it != liveValInterPlaces.end()) {
    solution[lwiOp] = {0, (int)it->second, -1};
    // TODO[@YW]: determine whether inter or exter
    pushResultToLiveVec(liveValInterPlaces, lwiOp->getResult(0),
                        (int)it->second);
    return success();
  }
  return failure();
}

void TemporalCGRAScheduler::placeSwiOpToBlock(Block *block, cgra::SwiOp swiOp) {
  // get refOp's schedule result
  auto storeOpr = swiOp->getOperand(0);
  Operation *refOp = storeOpr.getDefiningOp();
  unsigned nextCycle, pe;
  if (isa<BlockArgument>(storeOpr)) {
    nextCycle = 0;
    auto it = std::find_if(
        liveValExterPlaces.begin(), liveValExterPlaces.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == storeOpr; });
    if (it != liveValExterPlaces.end())
      pe = it->second;

    it = std::find_if(
        liveValInterPlaces.begin(), liveValInterPlaces.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == storeOpr; });
    if (it != liveValInterPlaces.end())
      pe = it->second;
  }
  if (refOp) {
    nextCycle = solution.at(refOp).time + 1;
    pe = solution.at(refOp).pe;
  }

  unsigned leftPE = (pe - nCol + nRow * nCol) % (nRow * nCol);
  unsigned rightPE = (pe + nCol) % (nRow * nCol);
  unsigned topPE = (pe - nCol) % (nRow * nCol);
  unsigned bottomPE = (pe + nCol) % (nRow * nCol);

  std::vector<unsigned> peList = {pe, leftPE, rightPE, topPE, bottomPE};

  std::map<Operation *, ScheduleUnit> subResult;
  std::vector<ScheduleUnit> subSchedule;
  for (auto [op, su] : solution) {
    if (op->getBlock() == block) {
      subResult[op] = su;
      subSchedule.push_back(su);
    }
  }

  // check whether pe, left_pe, right_pe, top_pe and bottom_pe are available
  // during nextCycle.
  bool isAvailable = false;
  // the termintor must be placed no earlier than swiOp
  Operation *termOp = block->getTerminator();
  auto &termSu = solution[termOp];
  unsigned swiPE = pe;
  if (termSu.time < nextCycle) {
    termSu.time++;
    if (termSu.pe == pe)
      swiPE = rightPE;
  }

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
    llvm::errs() << "Failed to find available pe for " << swiOp << "\n";
    // place swiOp to the next available pe, delay all the schedule after
    // nextCycle
    for (auto &[op, su] : solution) {
      llvm::errs() << "delay " << op << "1 CC\n";
      if (op->getBlock() == block && su.time >= nextCycle)
        su.time++;
    }
    solution[swiOp] = {(int)nextCycle, (int)swiPE, -1};
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

  auto refOp = origVal.getDefiningOp();
  if (refOp && failUser) {
    // process intra-bb split
    if (refOp->getBlock() == failUser->getBlock())
      insertInternalLSOps(refOp, failUser);

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
      placeSwiOpToBlock(refOp->getBlock(), swi);
    }

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

void TemporalCGRAScheduler::rollBackMovOp(Value failVal) {
  // check whether failVal is produced by val + 0
  auto definingOp = failVal.getDefiningOp();
  if (!definingOp || !isa<arith::AddIOp, arith::AddFOp>(definingOp))
    return;

  Operation *zeroOp = definingOp->getOperand(1).getDefiningOp();
  // if zeroOp is not a constant zero, return
  if (!zeroOp || !dyn_cast_or_null<arith::ConstantIntOp>(zeroOp) ||
      dyn_cast<arith::ConstantIntOp>(zeroOp)
              .getValue()
              .dyn_cast<IntegerAttr>()
              .getInt() != 0) {
    return;
  }
  failVal.replaceAllUsesWith(definingOp->getOperand(0));
  // remove the movOp and constant zero Op
  definingOp->erase();
  zeroOp->erase();
  rollBackMovOp(definingOp->getOperand(0));
}

LogicalResult TemporalCGRAScheduler::createSchedulerAndSolve() {
  // Create scheduler for each block
  makeScheduleSeq();
  computeLiveValue();
  for (auto [bb, block] : llvm::enumerate(scheduleSeq)) {
    llvm::errs() << "\nBlock " << bb << " is scheduling\n";

    BasicBlockILPModel bbILPModel(maxReg, nRow, nCol, block, bb, builder);
    bbILPModel.setLiveOutPrerequisite(getExternalLiveOut(block),
                                      getInternalLiveOut(block));
    bbILPModel.setLiveInPrerequisite(getExternalLiveIn(block),
                                     getInternalLiveIn(block));

    int maxIter = 15;
    Operation *failUser = nullptr;
    Operation *checkptr = nullptr;
    Value spill = nullptr;
    int movNum = 3;
    bbILPModel.setFailureStrategy(FailureStrategy::Mov);

    // try to schedule the block
    bool findSolution = false;
    while (maxIter > 0) {
      // Run the ILP model
      if (succeeded(bbILPModel.createSchedulerAndSolve())) {
        findSolution = true;
        break;
      }

      // handle the failure according to the failure strategy
      if (bbILPModel.getFailureStrategy() == FailureStrategy::Mov) {
        spill = bbILPModel.getSpillVal();
        failUser = bbILPModel.getFailUser();
        // Determine whether still adopt the mov strategy
        handleMovAttempFailure(failUser, spill, bbILPModel, movNum, 3);
        if (movNum > 0) {
          bbILPModel.setCheckPoint(nullptr);
          // bbILPModel.setCheckPoint(failUser);
          insertMovOp(spill, failUser);
        } else {
          // step back, remove the additional sadd zero ops
          bbILPModel.setFailureStrategy(FailureStrategy::Split);
          rollBackMovOp(bbILPModel.getSpillVal());
          maxIter += 3;
        }
      }
      if (bbILPModel.getFailureStrategy() == FailureStrategy::Split) {
        // split the liveOut value
        spill = bbILPModel.getSpillVal();
        failUser = bbILPModel.getFailUser();
        bool pushPhiToMem =
            isPhiRelatedValue(spill) && !spill.isa<BlockArgument>();
        if (failed(splitDFGWithLSOps(spill, failUser, UINT_MAX, pushPhiToMem)))
          return failure();
        // Rerun the liveness analysis
        computeLiveValue();
        bbILPModel.setLiveOutPrerequisite(getExternalLiveOut(block),
                                          getInternalLiveOut(block));
        bbILPModel.setLiveInPrerequisite(getExternalLiveIn(block),
                                         getInternalLiveIn(block));
        bbILPModel.setRAWPair(opRAWs);
        movNum = 3;
        bbILPModel.setFailureStrategy(FailureStrategy::Mov);
      }
      if (bbILPModel.getFailureStrategy() == FailureStrategy::Abort) {
        llvm::errs() << "Optimization abort\n";
        return failure();
      }
      maxIter--;
    }

    if (!findSolution) {
      llvm::errs() << "Failed to schedule block " << bb << "\n";
      return failure();
    }
    writeLiveOutResult(bbILPModel.getExternalLiveOutResult(),
                       bbILPModel.getInternalLiveOutResult());
    saveSubILPModelResult(bbILPModel.getSolution());
    scheduleIdx++;
    // if (bb == 10)
    //   break;
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
      if (solution.find(&op) == solution.end())
        continue;

      auto &su = solution[&op];
      su.time += kernelTime;
      endTime = std::max(endTime, su.time);
    }
    blockEndT[&block] = endTime + 1;
    kernelTime = endTime + 1;
  }

  std::ofstream csvFile(fileName);
  for (auto [bbInd, bb] : llvm::enumerate(region.getBlocks())) {
    for (auto &op : bb.getOperations()) {
      if (solution.find(&op) == solution.end())
        continue;
      std::string str;
      llvm::raw_string_ostream rso(str);
      rso << op;
      auto su = solution[&op];
      csvFile << rso.str() << "&" << su.time << "&" << su.pe << "&" << bbInd
              << "\n";
    }
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
  // std::map<std::string, ScheduleUnitBB> scheduleResult;
  std::ifstream file(filename);

  if (!file.is_open()) {
    llvm::errs() << "Error: Could not open the file " << filename << "\n";
    return failure();
  }

  std::string line;
  while (std::getline(file, line)) {
    // Split the line by commas and store the result
    std::vector<std::string> row = split(line, '&');
    data.push_back(row);
  }

  file.close();

  for (auto [bbInd, bb] : llvm::enumerate(region.getBlocks())) {
    for (auto &op : bb.getOperations()) {
      std::string opName;
      llvm::raw_string_ostream rso(opName);
      rso << op;
      // test whether opName is in the data first and bbInd is in the last
      auto it = std::find_if(
          data.begin(), data.end(), [&](std::vector<std::string> row) {
            return row[0] == opName && std::stoi(row[3]) == bbInd;
          });
      if (it == data.end())
        continue;
      ScheduleUnit su = {std::stoi((*it)[1]), std::stoi((*it)[2]), -1};
      solution[&op] = su;
      std::ostringstream oss;
      oss << std::left << std::setw(70) << opName << std::setw(10) << su.time
          << std::setw(10) << su.pe;
      llvm::errs() << oss.str() << "\n";
    }
  }
}
