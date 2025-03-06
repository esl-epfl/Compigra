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
  if (dyn_cast_or_null<arith::ConstantOp>(val.getDefiningOp()) ||
      dyn_cast_or_null<arith::ConstantIntOp>(val.getDefiningOp()) ||
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

static void pushResultToLiveVec(liveVec &liveVec, Value val, unsigned index) {
  // if find the value in the liveVec, update the index
  auto it = std::find_if(
      liveVec.begin(), liveVec.end(),
      [&](std::pair<Value, unsigned> p) { return p.first == val; });
  if (it != liveVec.end()) {
    if (it->second != UINT32_MAX && it->second != index)
      llvm::errs() << "Warning: " << val << " has multiple live places {"
                   << it->second << " " << index << "}\n";
    if (it->second == UINT32_MAX) {
      it->second = index;
      // llvm::errs() << "Store " << val << " at " << index << "\n";
    }
  } else {
    // llvm::errs() << "Store " << val << " at " << index << "\n";
    liveVec.push_back({val, index});
  }
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
        // branch argument is not a use
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

void TemporalCGRAScheduler::setupPrerequisite(
    std::vector<std::pair<Value, int>> valPlacement) {
  for (auto [val, index] : valPlacement) {
    if (isPhiRelatedValue(val)) {
      SetVector<Value> relatedVals;
      getAllPhiRelatedValues(val, relatedVals);
      for (auto relatedVal : relatedVals)
        pushResultToLiveVec(liveValAndPEs, relatedVal, index);
    } else {
      pushResultToLiveVec(liveValAndPEs, val, index);
    }
  }
}

void TemporalCGRAScheduler::storeLocalResult(const liveVec localVec) {
  for (auto [val, ind] : localVec) {
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

void TemporalCGRAScheduler::writeLiveOutResult(const liveVec liveOutExter,
                                               const liveVec liveOutInter,
                                               const liveVec liveInExter,
                                               const liveVec liveInInter) {
  storeLocalResult(liveOutExter);
  storeLocalResult(liveOutInter);
  storeLocalResult(liveInExter);
  storeLocalResult(liveInInter);

  // compute liveValAndPEs
  // TODO[@YYY]for all the live-in of the restricted block, it must match with
  // the solution
  // for (auto block : restrictedBBs) {
  //   for (auto val : liveIns[block]) {
  //     if (std::find(restrictedBBs.begin(), restrictedBBs.end(),
  //                   val.getParentBlock()) != restrictedBBs.end())
  //       continue;
  //     int restrictPE = UINT32_MAX;
  //     for (auto user : val.getUsers()) {
  //       if (user->getBlock() == block) {
  //         restrictPE = solution[user].pe;
  //         // llvm::errs() << val << " has to be placed at " << restrictPE
  //         //              << " for " << *user << "\n";
  //         break;
  //       }
  //     }
  //     if (restrictPE == UINT32_MAX)
  //       continue;
  //     if (isPhiRelatedValue(val)) {
  //       SetVector<Value> relatedVals;
  //       getAllPhiRelatedValues(val, relatedVals);
  //       for (auto relatedVal : relatedVals)
  //         pushResultToLiveVec(liveValAndPEs, relatedVal, restrictPE);
  //     } else {
  //       pushResultToLiveVec(liveValAndPEs, val, restrictPE);
  //     }
  //   }
  // }
}

bool TemporalCGRAScheduler::isExternalLive(Value val) {
  return false;
  // first check the liveness according to the pre-defined solution
  if (isPhiRelatedValue(val)) {
    SetVector<Value> relatedVals;
    getAllPhiRelatedValues(val, relatedVals);
    for (auto relatedVal : relatedVals) {
      if (auto defOp = relatedVal.getDefiningOp()) {
        if (restrictedBBs.count(defOp->getBlock()) && solution.count(defOp) > 0)
          return solution[defOp].reg == maxReg;
      }
    }
  } else if (auto defOp = val.getDefiningOp()) {
    if (restrictedBBs.count(defOp->getBlock()) && solution.count(defOp) > 0)
      return solution[defOp].reg == maxReg;
  }

  // if used by the blocked block, return false
  for (auto user : val.getUsers()) {
    if (restrictedBBs.count(user->getBlock()))
      return false;

    // if the user is br / cond_br, and the corresponding block is blocked,
    // return false
    if (auto branchOp = dyn_cast_or_null<cf::BranchOp>(user)) {
      if (restrictedBBs.count(branchOp.getSuccessor()))
        return false;
    } else if (auto branchOp =
                   dyn_cast_or_null<cgra::ConditionalBranchOp>(user)) {
      if (restrictedBBs.count(branchOp.getSuccessor(0)) ||
          restrictedBBs.count(branchOp.getSuccessor(1)))
        return false;
    }
  }

  if (val.use_empty())
    return false;

  // if val is phi related, ensure it produced the same attributes for all
  // related values.
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

    llvm::errs() << "get external livein: " << val << "\n";
    // search whether the val is in the liveOutExterPlaces
    auto it = std::find_if(
        liveValAndPEs.begin(), liveValAndPEs.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValAndPEs.end()) {
      auto index = it->second;
      liveInExter.push_back({val, index});
    } else {
      liveInExter.push_back({val, UINT32_MAX});
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
    llvm::errs() << "get internal livein: " << val << "\n";
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
    llvm::errs() << "get external liveout: " << val << "\n";
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

    llvm::errs() << "get internal liveout: " << val << "\n";
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
  int blockStart = INT_MAX;
  for (auto [op, su] : res) {
    int reg = solution.count(op) > 0 ? solution[op].reg : -1;
    ScheduleUnit res = {su.time, su.pe, reg};
    solution[op] = res;
    // if (getBlockStartT(op->getBlock()) > su.time)
    //   setBlockExecutionTime(op->getBlock(), su.time);
  }
}

void TemporalCGRAScheduler::resctrictBBSchedule(
    const std::map<Operation *, ScheduleUnit> res) {
  for (auto [op, su] : res) {
    restrictedBBs.insert(op->getBlock());
    // ScheduleUnit res = {su.time, su.pe, -1};
    solution[op] = su;
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

void TemporalCGRAScheduler::placeLwiOpToBlock(Block *block, unsigned seekPE,
                                              int seekTime, cgra::LwiOp lwiOp,
                                              bool onlyPE) {
  // print the existing schedule result
  std::map<Operation *, ScheduleUnit> subResult;
  std::vector<ScheduleUnit> subSchedule;
  // if lwiOp already scheduled, erase the original schedule result
  // auto it = solution.find(lwiOp);
  // if (it != solution.end()) {
  //   solution.erase(it);
  // }

  // int blockStart = INT32_MAX;
  // // if (time == 0)
  // for (auto &[op, su] : solution) {
  //   if (op->getBlock() == block) {
  //     if (su.time < blockStart)
  //       blockStart = su.time;
  //   }
  // }

  for (auto [op, su] : solution) {
    if (op->getBlock() == block) {
      subResult[op] = su;
      subSchedule.push_back(su);
    }
  }

  int prevCycle = seekTime - 1;
  unsigned pe = seekPE;

  unsigned leftPE = (pe - nCol + nRow * nCol) % (nRow * nCol);
  unsigned rightPE = (pe + nCol) % (nRow * nCol);
  unsigned topPE = (pe - nCol) % (nRow * nCol);
  unsigned bottomPE = (pe + nCol) % (nRow * nCol);
  std::vector<unsigned> peList = {pe};
  if (!onlyPE)
    peList = {pe, leftPE, rightPE, topPE, bottomPE};

  // check whether pe, left_pe, right_pe, top_pe and bottom_pe are available
  // during nextCycle.
  unsigned lwiPE = pe;
  // if (prevCycle < blockStart) {
  //   prevCycle = blockStart;
  //   for (auto &[op, su] : solution)
  //     if (op->getBlock() == block)
  //       su.time++;
  // }
  // seek whether there slot inside the bb execution to accommodate the lwiOp
  for (auto p : peList) {
    auto it = std::find_if(
        subSchedule.begin(), subSchedule.end(),
        [&](ScheduleUnit su) { return su.time == prevCycle && su.pe == p; });
    // if the pe is not occupied, place the lwi to the pe
    if (it == subSchedule.end()) {
      solution[lwiOp] = {(int)prevCycle, (int)p, -1};
      // accomodate the lwiOp execution time and unit
      return;
    }
  }

  // delay all the operations after the prevCycle
  for (auto &[op, su] : solution) {
    if (op->getBlock() == block && su.time >= prevCycle)
      su.time++;
  }
  solution[lwiOp] = {(int)prevCycle, (int)lwiPE, -1};

  int blockStart = INT32_MAX;
  for (auto &[op, su] : solution) {
    if (op->getBlock() == block) {
      if (su.time < blockStart)
        blockStart = su.time;
    }
  }

  if (blockStart < 0)
    // increment the blockStart time
    for (auto &[op, su] : solution)
      if (op->getBlock() == block)
        su.time = su.time - blockStart;
}

void TemporalCGRAScheduler::placeLwiOpToBlock(Block *block, BlockArgument arg,
                                              cgra::LwiOp lwiOp,
                                              unsigned seekPE) {
  // to load phi value, it must be the start of the block
  int blockStart = INT32_MAX;
  std::map<Operation *, ScheduleUnit> subResult;
  for (auto &[op, su] : solution) {
    if (op->getBlock() == block) {
      if (su.time < blockStart)
        blockStart = su.time;
      subResult[op] = su;
    }
  }

  int nextCycle = blockStart;
  bool loadPhiInStart = true;
  // if in blockStart time, all executed operations are phi operations, delay
  // the blockStart time
  for (auto [op, su] : subResult) {
    if (su.time == blockStart && !isa<cgra::LwiOp>(op)) {
      loadPhiInStart = false;
      break;
    }
  }
  if (loadPhiInStart)
    nextCycle++;

  placeLwiOpToBlock(block, seekPE, nextCycle, lwiOp, true);

  // assignPE = it->second;
  // solution[lwiOp] = {blockStart, (int)it->second, -1};
  // llvm::errs() << "place " << lwiOp << " to " << blockStart << " "
  //              << (int)it->second << "\n";
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
                                                Value origVal, unsigned opIndex,
                                                bool forcePlacement) {
  Block *userBlock = refOp->getBlock();
  Type valType = origVal.getType();

  builder.setInsertionPoint(refOp);

  unsigned blockIndex = std::distance(
      scheduleSeq.begin(),
      std::find(scheduleSeq.begin(), scheduleSeq.end(), userBlock));

  auto constOp = builder.create<arith::ConstantIntOp>(
      refOp->getLoc(), addr, builder.getIntegerType(32));
  auto loadOp = builder.create<cgra::LwiOp>(refOp->getLoc(), valType,
                                            constOp->getResult(0));
  llvm::errs() << "insert load op: " << loadOp << " for \n";
  llvm::errs() << *refOp << "\n";
  bool placeLoad = forcePlacement || (blockIndex < scheduleIdx);

  if (placeLoad) {
    llvm::errs() << "place load: " << forcePlacement << " [" << blockIndex
                 << " " << scheduleIdx << "]\n";
    if (auto arg = dyn_cast_or_null<BlockArgument>(origVal)) {
      computeLiveValue();
      // Search in liveValAndPEs
      unsigned pe = UINT32_MAX;
      auto it = std::find_if(
          liveValAndPEs.begin(), liveValAndPEs.end(),
          [&](const std::pair<Value, unsigned> &p) { return p.first == arg; });
      // if not found in liveValAndPEs, assign it to the user PE
      if (it != liveValAndPEs.end()) {
        pe = it->second;
      } else {
        pe = solution.at(refOp).pe;
      }
      if (pe == UINT32_MAX)
        return nullptr;

      placeLwiOpToBlock(userBlock, arg, loadOp, pe);
    } else {
      int time = solution.at(refOp).time;
      unsigned pe = solution.at(refOp).pe;
      placeLwiOpToBlock(userBlock, pe, time, loadOp);
    }
  }

  return loadOp;
}

// std::optional<cgra::LwiOp>
// TemporalCGRAScheduler::existAvailableLoad(unsigned memLoc, Operation *failUser,
//                                           Block *loadBlk) {
//   for (auto loadOp : loadBlk->getOps<cgra::LwiOp>()) {
//     // if loadOp get address is not constant, or the constant not equal to
//     // memLoc, continue
//     auto addrOp = dyn_cast_or_null<arith::ConstantIntOp>(
//         loadOp.getOperand().getDefiningOp());
//     if (!addrOp || addrOp.value() != memLoc)
//       continue;

//     // evaluate whether locate in the same pe
//     auto su = solution.at(failUser);

//     if (solution.count(loadOp) && solution[loadOp].pe == su.pe) {
//       // revise the op's time
//       llvm::errs() << "exist available load " << loadOp << " at " << su.pe
//                    << "\n";
//       llvm::errs() << "Replaced : "
//                    << (loadOp->getBlock() == failUser->getBlock()) << "\n";
//       if (loadOp->getBlock() == failUser->getBlock())
//         placeLwiOpToBlock(loadBlk, su.pe,
//                           std::min(su.time, solution[loadOp].time), loadOp,
//                           true);
//       return loadOp;
//     }
//   }
//   return std::nullopt;
// }

LogicalResult TemporalCGRAScheduler::splitDFGWithLSOps(Value origVal,
                                                       Operation *failUser,
                                                       unsigned memLoc,
                                                       bool processCntPhi,
                                                       bool load, bool store) {
  llvm::errs() << "split DFG for" << origVal << "\n";
  auto refOp = origVal.getDefiningOp();
  if (refOp && failUser) {
    // process intra-bb split
    if (refOp->getBlock() == failUser->getBlock()) {
      insertInternalLSOps(refOp, failUser);
      return success();
    }
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
      assignAddr = reserveMem;
  }

  if (processCntPhi) {
    SmallVector<Block *, 4> succBlocks;
    if (auto arg = dyn_cast_or_null<BlockArgument>(origVal))
      succBlocks.push_back(arg.getParentBlock());
    else
      succBlocks = getCntBlocksThroughPhi(origVal);

    for (auto suc : succBlocks) {
      // get the phi value
      BlockArgument phiVal;
      if (auto arg = dyn_cast_or_null<BlockArgument>(origVal))
        phiVal = arg;
      else
        phiVal = getCntBlockArgument(origVal, suc);

      // replace the phi value with lwi op
      llvm::errs() << "phi insert load: " << assignAddr << "\n";
      if (failed(splitDFGWithLSOps(phiVal, nullptr, assignAddr, false,
                                   load = true, store = false)))
        return failure();

      SmallVector<Value, 2> srcVals = getSrcOprandsOfPhi(phiVal, true);
      suc->eraseArgument(phiVal.getArgNumber());

      // insert swi op for all source operands
      for (auto src : srcVals)
        if (failed(splitDFGWithLSOps(src, nullptr, assignAddr, load = false,
                                     store = true)))
          return failure();
    }
    return success();
  }

  if (load) {
    // insert lwi op before the user block
    std::map<Block *, cgra::LwiOp> lwiOps;
    // If user is not specified, insert lwi op to all users
    for (auto &use : llvm::make_early_inc_range(origVal.getUses())) {
      Operation *user = use.getOwner();
      Block *userBlock = user->getBlock();
      llvm::errs() << "user : " << *user << "\n";

      if (lwiOps.count(userBlock) == 0) {
        // auto existLoad = existAvailableLoad(assignAddr, user, userBlock);
        // // if loadOp has value, lwiOps[userBlock] = loadOp.value
        // if (existLoad.has_value()) {
        //   lwiOps[userBlock] = existLoad.value();
        //   user->setOperand(use.getOperandNumber(),
        //                    existLoad.value()->getResult(0));
        //   continue;
        // }
        cgra::LwiOp loadOp =
            insertLoadOp(user, assignAddr, origVal, use.getOperandNumber(),
                         restrictedBBs.count(userBlock));
        if (!loadOp)
          return failure();

        user->setOperand(use.getOperandNumber(), loadOp->getResult(0));
        lwiOps[userBlock] = loadOp;
        llvm::errs() << "insert lwi op: " << loadOp << "\n";
        continue;
      }

      // if used by multiple operations, ensure dominance by moving it to the
      // front of the block
      auto existOp = lwiOps[userBlock];
      auto memAddr = existOp->getOperand(0).getDefiningOp();
      user->setOperand(use.getOperandNumber(), existOp->getResult(0));
      memAddr->moveBefore(region.getBlocks().front().getTerminator());
      // CFG dominance, no need to adjust the internal DFG dominance
      if (existOp->getBlock() != userBlock)
        continue;
      // get the first user in the block
      Operation *locOp;
      auto userSet = existOp->getResult(0).getUsers();
      for (auto &op : existOp->getBlock()->getOperations())
        // if op is the user of existOp
        if (std::find(userSet.begin(), userSet.end(), &op) != userSet.end()) {
          locOp = &op;
          break;
        }
      existOp->moveBefore(locOp);
      llvm::errs() << "use existing lwi op: " << existOp << "\n";
    }
  }

  // if the value has no been written, insert swi op
  if (store && memWriteOp == memStack.end()) {
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
    llvm::errs() << "insert swi op: " << swi << "\n";
    bool placeStore =
        restrictedBBs.count(refOp->getBlock()) || (blockIndex < scheduleIdx);
    if (placeStore)
      placeSwiOpToBlock(refOp->getBlock(), swi);

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

static bool dfsBlock(Block *point, Block *block, std::vector<Block *> &cycle,
                     std::unordered_set<Block *> &visited) {
  // Mark the current block as visited.
  visited.insert(block);
  cycle.push_back(block);

  for (auto succ : block->getSuccessors()) {
    if (succ == point) {
      // cycle.push_back(point);
      return true;
    }

    // If `succ` has not been visited, perform DFS on it.
    if (visited.find(succ) == visited.end()) {
      if (dfsBlock(point, succ, cycle, visited)) {
        return true; // Cycle detected.
      }
    }
  }

  // Backtrack: remove the current block from the cycle.
  cycle.pop_back();
  return false;
}

static bool existInCycle(std::vector<Block *> &newCycle,
                         std::vector<std::vector<Block *>> &cycles) {
  for (auto &cycle : cycles) {
    if (cycle.size() != newCycle.size())
      continue;

    // Check if `newCycle` is a rotation of `cycle`.
    auto start = std::find(cycle.begin(), cycle.end(), newCycle[0]);
    if (start != cycle.end()) {
      bool isMatch = true;
      for (size_t i = 0; i < cycle.size(); ++i) {
        // Compare elements taking rotation into account.
        if (cycle[(start - cycle.begin() + i) % cycle.size()] != newCycle[i]) {
          isMatch = false;
          break;
        }
      }
      if (isMatch)
        return true;
    }
  }
  return false;
}

void TemporalCGRAScheduler::makeScheduleSeq() {
  scheduleSeq.clear();
  scheduleIdx = 0;
  // first assign the blocked BBs
  for (auto &bb : region.getBlocks())
    if (restrictedBBs.find(&bb) != restrictedBBs.end())
      scheduleSeq.push_back(&bb);
  for (auto &bb : region.getBlocks()) {
    if (std::find(scheduleSeq.begin(), scheduleSeq.end(), &bb) ==
        scheduleSeq.end())
      scheduleSeq.push_back(&bb);
  }

  // // init the schedule sequence
  // scheduleSeq.clear();
  // scheduleIdx = 0;

  // std::map<Block *, int> blockOrder;
  // std::vector<Block *> workList;
  // for (auto &block : region.getBlocks())
  //   workList.push_back(&block);

  // // calculate the block order
  // std::vector<std::vector<Block *>> visitedCycle;
  // for (auto &block : region.getBlocks()) {
  //   if (!blockOrder.count(&block))
  //     blockOrder[&block] = 0;
  //   // dfs the block, if it returns to itself, it is a cycle
  //   std::vector<Block *> cycle;
  //   std::unordered_set<Block *> visited;
  //   if (dfsBlock(&block, &block, cycle, visited) &&
  //       !existInCycle(cycle, visitedCycle)) {
  //     visitedCycle.push_back(cycle);
  //     for (auto &b : cycle)
  //       blockOrder[b] += 1;
  //   }
  // }

  // for (auto &block : region.getBlocks()) {
  //   // calculate the number of constant ops
  //   unsigned cstOpNum =
  //   std::distance(block.getOps<arith::ConstantOp>().begin(),
  //                                     block.getOps<arith::ConstantOp>().end());
  //   cstOpNum += std::distance(block.getOps<arith::ConstantFloatOp>().begin(),
  //                             block.getOps<arith::ConstantFloatOp>().end());
  //   cstOpNum += std::distance(block.getOps<arith::ConstantIntOp>().begin(),
  //                             block.getOps<arith::ConstantIntOp>().end());
  //   auto weight = block.getOperations().size() - cstOpNum;
  //   blockOrder[&block] += weight * 0.1;
  // }

  // // sort the block according to the order
  // std::sort(workList.begin(), workList.end(), [&](Block *a, Block *b) {
  //   if (blockOrder[a] == blockOrder[b])
  //     return a->getOperations().size() > b->getOperations().size();
  //   return blockOrder[a] > blockOrder[b];
  // });
  // scheduleSeq = workList;
}

void TemporalCGRAScheduler::rollBackMovOp(Value failVal, int maxIter) {
  // check whether failVal is produced by val + 0
  if (maxIter <= 0)
    return;
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
  // remove the movOp and constant zero Op, if they are not used
  definingOp->erase();
  if (zeroOp->use_empty())
    zeroOp->erase();
  rollBackMovOp(definingOp->getOperand(0), maxIter - 1);
}

LogicalResult TemporalCGRAScheduler::createSchedulerAndSolve() {
  // Create scheduler for each block
  makeScheduleSeq();
  computeLiveValue();
  printBlockLiveValue("liveValue.txt");

  for (auto [bb, block] : llvm::enumerate(scheduleSeq)) {

    // llvm::errs() << "\nBlock " << bb << " is scheduling\n";

    BasicBlockILPModel bbILPModel(maxReg, nRow, nCol, block, bb, builder);

    bool findSolution = false;
    //  if the block has been scheduled by other scheduler, skip it
    if (restrictedBBs.find(block) != restrictedBBs.end()) {
      // bbILPModel.saveSubILPModelResult("sub_ilp_" + std::to_string(bb) +
      //                                  ".csv");
      bbILPModel.writeKnownResult();
      findSolution = true;
      bbILPModel.setupPreScheduleResult(getBlockSubSolution(block));

      bbILPModel.setSolverMode(1);
    }

    bbILPModel.setLiveOutPrerequisite(getExternalLiveOut(block),
                                      getInternalLiveOut(block));
    bbILPModel.setLiveInPrerequisite(getExternalLiveIn(block),
                                     getInternalLiveIn(block));

    // bbILPModel.readScheduleResult("sub_ilp_" + std::to_string(bb) + ".csv");
    // saveSubILPModelResult(bbILPModel.getSolution());
    // llvm::errs() << "Block " << bb << " is scheduling\n";
    // continue;
    // return success();

    int maxIter = 15;
    Operation *failUser = nullptr;
    Operation *checkptr = nullptr;
    Value spill = nullptr;
    int movNum = 3;
    bbILPModel.setFailureStrategy(FailureStrategy::Mov);

    // try to schedule the block
    while (!findSolution && maxIter > 0) {
      // Run the ILP model
      if (succeeded(bbILPModel.createSchedulerAndSolve())) {
        findSolution = true;
        break;
      }

      // handle the failure according to the failure strategy
      if (bbILPModel.getFailureStrategy() == FailureStrategy::Mov) {
        llvm::errs() << "Strategy: Mov\n";
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
          rollBackMovOp(bbILPModel.getSpillVal(), movNum);
          maxIter += 3;
        }
      }
      if (bbILPModel.getFailureStrategy() == FailureStrategy::Split) {
        llvm::errs() << "Strategy: split\n";
        // split the liveOut value
        spill = bbILPModel.getSpillVal();
        failUser = bbILPModel.getFailUser();
        bool pushPhiToMem = isPhiRelatedValue(spill);
        if (failed(splitDFGWithLSOps(spill, failUser, UINT_MAX, pushPhiToMem)))
          return failure();
        // Rerun the liveness analysis
        computeLiveValue();
        bbILPModel.setupPreScheduleResult(getBlockSubSolution(block));
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
      return failure();
    }
    llvm::errs() << scheduleIdx << " block is scheduled\n\n";
    writeLiveOutResult(bbILPModel.getExternalLiveOutResult(),
                       bbILPModel.getInternalLiveOutResult(),
                       bbILPModel.getExternalLiveInResult(),
                       bbILPModel.getInternalLiveInResult());

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
    int alignStartTime = kernelTime;
    int endTime = kernelTime;
    auto bbStart = getBlockStartT(&block);
    auto gap = kernelTime - bbStart;
    // blockStartT[&block] = alignStartTime;
    for (auto &op : block.getOperations()) {
      if (solution.find(&op) == solution.end())
        continue;

      auto &su = solution[&op];
      su.time += gap;
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
              << "\r\n";
    }
  }
  csvFile.close();
  llvm::errs() << "Temporal spatial schedule is saved to " << fileName << "\n";
}
