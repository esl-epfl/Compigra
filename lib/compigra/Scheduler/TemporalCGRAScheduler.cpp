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
      std::string attr = isExternalLive(val) ? "External" : "Internal";
      outFile << rso.str() << " " << attr << "\n";
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
      std::string attr = isExternalLive(val) ? "External" : "Internal";
      outFile << rso.str() << " " << attr << "\n";
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
      // Only for flag comparison not for phi value propagation
      if (index < 2)
        continue;
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
      for (auto relatedVal : relatedVals) {
        llvm::errs() << "LiveOutExter: " << relatedVal << " " << ind << "\n";
        pushResultToLiveVec(liveValAndPEs, relatedVal, ind);
      }
    } else {
      pushResultToLiveVec(liveValAndPEs, val, ind);
    }
  }

  for (auto [val, ind] : liveOutInter) {
    if (isPhiRelatedValue(val)) {
      SetVector<Value> relatedVals;
      getAllPhiRelatedValues(val, relatedVals);
      for (auto relatedVal : relatedVals)
        pushResultToLiveVec(liveValAndPEs, relatedVal, ind);
    } else {
      pushResultToLiveVec(liveValAndPEs, val, ind);
    }
  }
}

bool TemporalCGRAScheduler::isExternalLive(Value val) {
  // DFG rewrite should not affect the attributes, strictly follow the previous
  // decision
  // if (std::find_if(liveValExterPlaces.begin(), liveValExterPlaces.end(),
  //                  [&](std::pair<Value, unsigned> p) {
  //                    return p.first == val;
  //                  }) != liveValExterPlaces.end())
  //   return true;

  // if (std::find_if(liveValInterPlaces.begin(), liveValInterPlaces.end(),
  //                  [&](std::pair<Value, unsigned> p) {
  //                    return p.first == val;
  //                  }) != liveValInterPlaces.end())
  //   return false;

  if (isPhiRelatedValue(val)) {
    SetVector<Value> relatedVals;
    getAllPhiRelatedValues(val, relatedVals);
    unsigned totalLength = 0;
    for (auto rVal : relatedVals) {
      for (auto &block : region) {
        if (liveIns[&block].count(rVal) && liveOuts[&block].count(rVal))
          totalLength += getCriticalPath(&block).size();
      }
      if (isa<BlockArgument>(rVal)) {
        for (auto user : rVal.getUsers()) {
          if (getEarliestStartTime(user) > maxLivePath)
            return false;
        }
      } else {
        if (getLatestEndTime(rVal.getDefiningOp()) > maxLivePath)
          return false;
      }
    }
    return totalLength < maxLivePath;
  }

  // // if val is live in over three blocks, return false;
  unsigned blockNum = 0;
  for (auto &block : region) {
    if (liveIns[&block].count(val) || liveOuts[&block].count(val))
      blockNum++;
  }
  if (blockNum > 3)
    return false;

  Operation *defOp = val.getDefiningOp();
  if (defOp) {
    unsigned maxHop = 0;
    for (auto user : defOp->getUsers()) {
      // calculate the theoretical live path length
      maxHop = std::max(maxHop, getShortestLiveHops(defOp, user));
    }

    // For all the values in relatedVals, their theoretical live path cannot
    // exceed certain hops.
    if (maxHop < maxLivePath)
      return true;
  }

  return false;
}

liveVec TemporalCGRAScheduler::getExternalLiveIn(Block *block) {
  auto bbLiveIn = liveIns[block];
  liveVec liveInExter;
  for (auto val : bbLiveIn) {
    bool isExternal = isExternalLive(val);
    if (!isExternal)
      continue;
    // search whether the val is in the liveOutExterPlaces
    auto it = std::find_if(
        liveValAndPEs.begin(), liveValAndPEs.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValAndPEs.end()) {
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
    bool isInternal = !isExternalLive(val);
    if (!isInternal)
      continue;
    // search whether the val is in the liveOutInterPlaces
    auto it = std::find_if(
        liveValAndPEs.begin(), liveValAndPEs.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValAndPEs.end()) {
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

  for (auto val : bbLiveOut) {
    bool isExternal = isExternalLive(val);
    if (!isExternal)
      continue;

    auto it = std::find_if(
        liveValAndPEs.begin(), liveValAndPEs.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValAndPEs.end()) {
      auto index = it->second;
      liveOutExter.push_back({val, index});
    } else {
      liveOutExter.push_back({val, UINT32_MAX});
    }
  }
  return liveOutExter;
}

liveVec TemporalCGRAScheduler::getInternalLiveOut(Block *block) {
  auto bbLiveOut = liveOuts[block];
  liveVec liveOutInter;

  for (auto val : bbLiveOut) {
    bool isInternal = !isExternalLive(val);
    if (!isInternal)
      continue;
    // search whether the val is in the liveOutInterPlaces
    auto it = std::find_if(
        liveValAndPEs.begin(), liveValAndPEs.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValAndPEs.end()) {
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
    bool existZero = false;
    Operation *zeroOp;
    for (auto op : region.getOps<arith::ConstantIntOp>()) {
      if (op.value() == 0) {
        zeroOp = op;
        existZero = true;
        zeroOp->moveBefore(&region.front().getOperations().front());
        break;
      }
    }
    if (!existZero) {
      zeroOp = builder.create<arith::ConstantIntOp>(user->getLoc(), 0,
                                                    origVal.getType());
    }
    movOp = builder.create<arith::AddIOp>(user->getLoc(), origVal,
                                          zeroOp->getResult(0));

  } else if (isa<Float32Type>(valType)) {
    Operation *zeroOp;
    bool existZero = false;
    for (auto op : region.getOps<arith::ConstantFloatOp>()) {
      if (op.value().convertToFloat() == 0.0f) {
        zeroOp = op;
        existZero = true;
        // move zeroOp to the front of the module
        zeroOp->moveBefore(&region.front().getOperations().front());
        break;
      }
    }
    if (!existZero) {
      zeroOp = builder.create<arith::ConstantFloatOp>(
          user->getLoc(), APFloat(0.0f), origVal.getType().cast<Float32Type>());
    }
    movOp = builder.create<arith::AddFOp>(user->getLoc(), origVal,
                                          zeroOp->getResult(0));
  }

  user->replaceUsesOfWith(origVal, movOp->getResult(0));
}

void TemporalCGRAScheduler::insertInternalLSOps(Operation *srcOp,
                                                Operation *dstOp) {
  unsigned assignAddr = 0;

  auto &refOp = srcOp->getBlock()->getOperations().front();
  if (!memStack.empty())
    assignAddr = memStack.back().first + 4;
  memStack.push_back({assignAddr, srcOp->getResult(0)});

  // insert swi op after the srcOp
  builder.setInsertionPointAfter(srcOp);
  auto addr = builder.create<arith::ConstantIntOp>(refOp.getLoc(), assignAddr,
                                                   builder.getIntegerType(32));
  auto swi = builder.create<cgra::SwiOp>(srcOp->getLoc(), srcOp->getResult(0),
                                         addr->getResult(0));
  swi->setAttr("memLoc", builder.getI32IntegerAttr(assignAddr));

  // insert lwi op before the dstOp
  builder.setInsertionPoint(dstOp);
  auto loadOp = builder.create<cgra::LwiOp>(
      dstOp->getLoc(), srcOp->getResult(0).getType(), addr->getResult(0));
  for (auto &use : llvm::make_early_inc_range(srcOp->getResult(0).getUses())) {
    Operation *user = use.getOwner();
    if (user == dstOp)
      user->setOperand(use.getOperandNumber(), loadOp->getResult(0));
  }
  // opRAWs.insert({loadOp, swi});
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
  // print the existing schedule result
  std::map<Operation *, ScheduleUnit> subResult;
  std::vector<ScheduleUnit> subSchedule;
  for (auto [op, su] : solution) {
    if (op->getBlock() == block) {
      subResult[op] = su;
      subSchedule.push_back(su);
    }
  }

  int prevCycle = solution.at(refOp).time - 1;
  unsigned pe = solution.at(refOp).pe;

  unsigned leftPE = (pe - nCol + nRow * nCol) % (nRow * nCol);
  unsigned rightPE = (pe + nCol) % (nRow * nCol);
  unsigned topPE = (pe - nCol) % (nRow * nCol);
  unsigned bottomPE = (pe + nCol) % (nRow * nCol);
  std::vector<unsigned> peList = {pe, leftPE, rightPE, topPE, bottomPE};

  // check whether pe, left_pe, right_pe, top_pe and bottom_pe are available
  // during nextCycle.
  unsigned lwiPE = pe;
  if (prevCycle < 0) {
    prevCycle = 0;
    for (auto &[op, su] : solution)
      if (op->getBlock() == block)
        su.time++;
  }
  // seek whether there slot inside the bb execution to accommodate the lwiOp
  for (auto p : peList) {
    auto it = std::find_if(
        subSchedule.begin(), subSchedule.end(),
        [&](ScheduleUnit su) { return su.time == prevCycle && su.pe == p; });
    // if the pe is not occupied, place the swiOp to the pe
    if (it == subSchedule.end()) {
      solution[lwiOp] = {(int)prevCycle, (int)p, -1};
      // accomodate the lwiOp execution time and unit
      return;
    }
  }

  solution[lwiOp] = {(int)prevCycle, (int)lwiPE, -1};
}

LogicalResult TemporalCGRAScheduler::placeLwiOpToBlock(Block *block,
                                                       BlockArgument arg,
                                                       cgra::LwiOp lwiOp) {
  // to load phi value, it must be the start of the block
  for (auto &[op, su] : solution) {
    if (op->getBlock() == block)
      su.time++;
  }

  computeLiveValue();
  bool isExternal = isExternalLive(lwiOp->getResult(0));

  // Get the original arg position
  unsigned assignPE = UINT_MAX;
  auto predicate = [&](const std::pair<Value, unsigned> &p) {
    return p.first == arg;
  };

  // Search in liveValAndPEs
  auto it = std::find_if(liveValAndPEs.begin(), liveValAndPEs.end(), predicate);
  if (it == liveValAndPEs.end())
    return failure();

  assignPE = it->second;
  solution[lwiOp] = {0, (int)it->second, -1};

  return success();
}

void TemporalCGRAScheduler::placeSwiOpToBlock(Block *block, cgra::SwiOp swiOp) {
  // get refOp's schedule result
  auto storeOpr = swiOp->getOperand(0);
  Operation *refOp = storeOpr.getDefiningOp();
  unsigned nextCycle, pe;
  if (isa<BlockArgument>(storeOpr)) {
    nextCycle = 0;
    auto it = std::find_if(
        liveValAndPEs.begin(), liveValAndPEs.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == storeOpr; });
    if (it != liveValAndPEs.end())
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
  Operation *termOp = block->getTerminator();
  auto &termSu = solution[termOp];
  // the terminator must be executed at the end of the block, if nextCycle
  // exceed the terminator time, delay the terminator.
  // If the terminator is executed at the PE which produce the value to be
  // stored, let the store operation executed to its neighbour PE.
  unsigned swiPE = pe;
  if (termSu.time < nextCycle) {
    termSu.time++;
    if (termSu.pe == pe)
      swiPE = rightPE;
  }
  // seek whether there slot inside the bb execution to accommodate the swiOp
  for (auto p : peList) {
    auto it = std::find_if(
        subSchedule.begin(), subSchedule.end(),
        [&](ScheduleUnit su) { return su.time == nextCycle && su.pe == p; });
    // if the pe is not occupied, place the swiOp to the pe
    if (it == subSchedule.end()) {
      solution[swiOp] = {(int)nextCycle, (int)p, -1};
      return;
    }
  }

  // place swiOp to the next available pe, delay all the schedule after
  // nextCycle
  for (auto &[op, su] : solution) {
    if (op->getBlock() == block && su.time >= nextCycle)
      su.time++;
  }
  solution[swiOp] = {(int)nextCycle, (int)swiPE, -1};
}

cgra::LwiOp TemporalCGRAScheduler::insertLoadOp(Operation *refOp, unsigned addr,
                                                Value origVal,
                                                unsigned opIndex) {
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
  llvm::errs() << "split DFG \n";
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
    memLoc = memWriteOp->first;
  }

  unsigned assignAddr = memLoc;
  if (memLoc == UINT_MAX) {
    if (!memStack.empty())
      assignAddr = memStack.back().first + 4;
    else
      assignAddr = 0;
  }

  if (processCntPhi) {
    if (auto arg = dyn_cast_or_null<BlockArgument>(origVal)) {
      auto loadOp = insertLoadOp(refOp, assignAddr, arg, true);
      // set the prerequisite
      if (!loadOp)
        return failure();
      pushResultToLiveVec(liveValAndPEs, loadOp->getResult(0),
                          solution[loadOp].pe);
      arg.replaceAllUsesWith(loadOp->getResult(0));
      arg.getParentBlock()->eraseArgument(arg.getArgNumber());
      // TODO[@YW]: re-implment the logic here.
      llvm::errs() << "load inserted\n";
      SmallVector<Value, 2> srcVals = getSrcOprandsOfPhi(arg, true);
      for (auto src : srcVals) {
        if (!src.getDefiningOp() ||
            isa<cf::BranchOp, cgra::ConditionalBranchOp, func::ReturnOp>(
                src.getDefiningOp()))
          builder.setInsertionPoint(src.getParentBlock()->getTerminator());
        else
          builder.setInsertionPointAfter(src.getDefiningOp());
        unsigned blockIndex =
            std::distance(scheduleSeq.begin(),
                          std::find(scheduleSeq.begin(), scheduleSeq.end(),
                                    src.getParentBlock()));

        auto addr = builder.create<arith::ConstantIntOp>(
            refOp->getLoc(), assignAddr, builder.getIntegerType(32));
        auto swi = builder.create<cgra::SwiOp>(refOp->getLoc(), src,
                                               addr->getResult(0));
        swi->setAttr("memLoc", builder.getI32IntegerAttr(assignAddr));
        if (blockIndex < scheduleIdx) {
          placeSwiOpToBlock(src.getParentBlock(), swi);
        }

        memStack.push_back({assignAddr, origVal});
      }
      return success();
    } else {
      SmallVector<Block *, 4> succBlocks = getCntBlocksThroughPhi(origVal);

      for (auto suc : succBlocks) {
        // get the phi value
        auto phiVal = getCntBlockArgument(origVal, suc);
        // insert lwi to replace the phi value
        auto loadOp = insertLoadOp(&suc->getOperations().front(), assignAddr,
                                   phiVal, true);
        // cannot find position to insert lwi op
        if (!loadOp)
          return failure();

        // set the prerequisite
        pushResultToLiveVec(liveValAndPEs, loadOp->getResult(0),
                            solution[loadOp].pe);

        // insert swi op for all source operands
        SmallVector<Value, 2> srcVals = getSrcOprandsOfPhi(phiVal, true);

        phiVal.replaceAllUsesWith(loadOp->getResult(0));
        suc->eraseArgument(phiVal.getArgNumber());

        for (auto src : srcVals)
          if (failed(splitDFGWithLSOps(src, nullptr, assignAddr, false)))
            return failure();
      }
      return success();
    }
  } else {
    // insert lwi op before the user block
    std::map<Block *, cgra::LwiOp> lwiOps;
    // If user is not specified, insert lwi op to all users
    for (auto &use : llvm::make_early_inc_range(origVal.getUses())) {
      Operation *user = use.getOwner();
      Block *userBlock = user->getBlock();
      // not process intra-bb split here
      if (userBlock == refOp->getBlock())
        continue;
      if (lwiOps.count(userBlock) != 0) {
        // if used by multiple operations, ensure dominance by moving it to the
        // front of the block
        auto existOp = lwiOps[userBlock];
        auto memAddr = existOp->getOperand(0).getDefiningOp();
        user->setOperand(use.getOperandNumber(), existOp->getResult(0));
        memAddr->moveBefore(region.getBlocks().front().getTerminator());
        existOp->moveBefore(&userBlock->getOperations().front());
        continue;
      }
      auto loadOp =
          insertLoadOp(user, assignAddr, origVal, use.getOperandNumber());
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
        refOp->getLoc(), assignAddr, builder.getIntegerType(32));
    auto swi = builder.create<cgra::SwiOp>(refOp->getLoc(), origVal,
                                           addr->getResult(0));
    swi->setAttr("memLoc", builder.getI32IntegerAttr(assignAddr));
    if (blockIndex < scheduleIdx) {
      placeSwiOpToBlock(refOp->getBlock(), swi);
    }

    memStack.push_back({assignAddr, origVal});
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
  printBlockLiveValue("liveValue.txt");

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
        // Determine whether still adopt the mov strategy
        handleMovAttempFailure(failUser, spill, bbILPModel, movNum, 3);
        spill = bbILPModel.getSpillVal();
        failUser = bbILPModel.getFailUser();
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
        bool pushPhiToMem = isPhiRelatedValue(spill);
        if (failed(splitDFGWithLSOps(spill, failUser, UINT_MAX, pushPhiToMem)))
          return failure();
        // Rerun the liveness analysis
        computeLiveValue();
        printBlockLiveValue("liveValue.txt");

        bbILPModel.setLiveOutPrerequisite(getExternalLiveOut(block),
                                          getInternalLiveOut(block));
        bbILPModel.setLiveInPrerequisite(getExternalLiveIn(block),
                                         getInternalLiveIn(block));
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
      // TODO: handle the failure
      return success();
    }
    llvm::errs() << scheduleIdx << "th block is scheduled\n";
    writeLiveOutResult(bbILPModel.getExternalLiveOutResult(),
                       bbILPModel.getInternalLiveOutResult());
    saveSubILPModelResult(bbILPModel.getSolution());
    scheduleIdx++;
  }

  calculateTemporalSpatialSchedule("temporalSpatialSchedule.csv");
  // printBlockLiveValue("liveValue.txt");
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
