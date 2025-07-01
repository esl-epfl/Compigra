//===- BasicBlockOpAssignment.cpp - Implements the class/functions to place
// operations of a basic block *- C++-* ----------------------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements class for BasicBlockILPModel functions.
//
//===----------------------------------------------------------------------===//

#include "compigra/Scheduler/BasicBlockOpAssignment.h"
#include "compigra/CgraOps.h"
#include <fstream>
#include <numeric>
#include <queue>

using namespace mlir;
using namespace compigra;

// Function to log messages to a file
void logMessage(const std::string &message, bool overwrite) {
  // if overwrite is true, clear the log file
  if (overwrite) {
    std::ofstream logFile("compigra_mapping.log",
                          std::ios::out | std::ios::trunc);
    logFile.close();
  }

  std::ofstream logFile("compigra_mapping.log", std::ios::out | std::ios::app);
  if (!logFile.is_open()) {
    llvm::errs() << "ERROR: Unable to open log file for writing.\n";
    return;
  }
  logFile << message << std::endl;
  logFile.close();
}

template <typename T>
static void getSubSet(SetVector<T> &vec1, SetVector<T> &vec2) {
  for (auto it = vec1.begin(); it != vec1.end();) {
    if (vec2.count(*it) == 0)
      it = vec1.erase(it);
    else
      ++it;
  }
}

template <typename T>
static SetVector<T> getInterSection(SetVector<T> &vec1, SetVector<T> &vec2) {
  SetVector<T> result;
  for (auto it = vec1.begin(); it != vec1.end();) {
    if (vec2.count(*it) > 0)
      result.insert(*it);
    ++it;
  }
  return result;
}

static bool isTerminatorOp(Operation *op) {
  return op->getBlock()->getTerminator() == op;
}

static unsigned getNonCstOpSize(Block *block) {
  unsigned constantOpSize =
      std::distance(block->getOps<arith::ConstantOp>().begin(),
                    block->getOps<arith::ConstantOp>().end());
  unsigned scheduledOpSize = block->getOperations().size() - constantOpSize;
  return scheduledOpSize;
}

/// Insert a value to a set if it is not a constant. The constant value is not
/// considered as a live value.
static void insertNonConst(Value val, SetVector<Value> &vec) {
  if (dyn_cast_or_null<arith::ConstantOp>(val.getDefiningOp()) ||
      dyn_cast_or_null<arith::ConstantIntOp>(val.getDefiningOp()) ||
      dyn_cast_or_null<arith::ConstantFloatOp>(val.getDefiningOp()))
    return;
  vec.insert(val);
}

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

void computeLiveValue(Region &region,
                      std::map<Block *, SetVector<Value>> &liveIns,
                      std::map<Block *, SetVector<Value>> &liveOuts) {
  // compute def and use for each block
  std::map<Block *, SetVector<Value>> defMap;
  std::map<Block *, SetVector<Value>> useMap;

  for (auto &block : region) {
    SetVector<Value> def;
    SetVector<Value> use;
    // push all block arguments to use
    for (auto arg : block.getArguments()) {
      // the entry block argument is IN/OUT of the function
      if (!block.isEntryBlock())
        insertNonConst(arg, use);
    }

    for (auto &op : block.getOperations()) {
      for (auto res : op.getResults())
        insertNonConst(res, def);

      for (auto opr : op.getOperands())
        // branch argument is not a use
        insertNonConst(opr, use);
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
          insertNonConst(val, liveIn);

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

static SmallVector<Operation *, 4>
getNextLayerOps(Block *block, SetVector<Value> &liveIn,
                std::set<Operation *> &visitedOps) {
  SmallVector<Operation *, 4> schedulingOps;
  auto termOp = block->getTerminator();

  for (auto &op : block->getOperations()) {
    if (isa<arith::ConstantOp>(op) || visitedOps.count(&op) ||
        isTerminatorOp(&op)) {
      continue;
    }

    // operation can be scheduled if all of its operands exist
    bool canSchedule = true;
    for (auto operand : op.getOperands()) {
      // check whether the operand is a liveIn value or belongs to the
      // scheduled operations
      bool isLiveIn =
          std::find(liveIn.begin(), liveIn.end(), operand) != liveIn.end();
      bool producedByScheduledOp =
          std::find(visitedOps.begin(), visitedOps.end(),
                    operand.getDefiningOp()) != visitedOps.end();
      bool isConstant = operand.getDefiningOp() &&
                        isa<arith::ConstantOp>(operand.getDefiningOp());
      if (!isLiveIn && !producedByScheduledOp && !isConstant) {
        canSchedule = false;
        break;
      }
    }

    // if the operation can be scheduled, add it to the scheduled operations
    bool loadCstAddr = isa<cgra::LwiOp>(op) &&
                       op.getOperand(0).getDefiningOp() &&
                       isa<arith::ConstantOp>(op.getOperand(0).getDefiningOp());
    if (canSchedule || loadCstAddr)
      schedulingOps.push_back(&op);
  }
  return schedulingOps;
}

static bool isLive(Value val, Block *curBlk, SetVector<Value> liveOut,
                   SetVector<Operation *> scheduledOps) {
  // check whether the value is used by the scheduledOp
  if (liveOut.contains(val))
    return true;

  for (auto user : val.getUsers()) {
    if (user->getBlock() != curBlk)
      continue;
    if (!scheduledOps.contains(user)) {
      return true;
    }
  }
  return false;
}

static bool isLiveExcept(Value val, Block *curBlk, Operation *user,
                         SetVector<Value> liveOut,
                         SetVector<Operation *> scheduledOps) {
  // check whether the value is used by the scheduledOp
  scheduledOps.insert(user);
  return isLive(val, curBlk, liveOut, scheduledOps);
}

static SmallVector<Operation *, 4>
getPreviousLayerOps(Block *block, SetVector<Value> &liveout,
                    std::set<Operation *> &ancestors) {
  SmallVector<Operation *, 4> visitedOps;

  for (auto op : llvm::make_early_inc_range(ancestors)) {
    // swi op and terminator op can be scheduled at the last stage
    if (op->getResults().empty()) {
      visitedOps.push_back(op);
      continue;
    }

    // if the operation produce live-out or its produced value is not used by
    // the ancestors, add it to the scheduling operations
    auto result = op->getResult(0);
    // bool canSchedule =
    //     std::find(liveout.begin(), liveout.end(), result) != liveout.end() ||
    //     result.use_empty();
    bool canSchedule = true;

    // any users does not belong to the ancestors
    for (auto &use : result.getUses()) {
      auto user = use.getOwner();
      bool nonInstUser =
          isa<cf::BranchOp>(user) ||
          (isa<cgra::ConditionalBranchOp>(user) && use.getOperandNumber() > 1);
      if (ancestors.count(user) && !nonInstUser) {
        canSchedule = false;
        break;
      }
    }

    if (canSchedule) {
      visitedOps.push_back(op);
    }
  }
  return visitedOps;
}

/// Compute the schedule priority of the operations in the block. The earliest
/// schedule time is traversed through the block liveIn, and the latest
/// schedule time is traversed through the block liveOut. This function
/// returns a map of the operations and their schedule priority [earliest,
/// latest].
static std::map<Operation *, std::pair<int, int>> getSchedulePriority(
    Block *block, SetVector<Value> &liveIn, SetVector<Value> &liveOut,
    int curDepth = 0,
    std::map<Operation *, std::pair<int, int>> prevPriority = {}) {
  std::map<Operation *, std::pair<int, int>> schedulePriority;

  std::set<Operation *> visitedOps;
  // the live-in height is zero
  int earliest = 1;
  // scheduledOp size = block->getOperations().size() - constantOp size - 1
  int totalOpSize = getNonCstOpSize(block) - 1;

  while (visitedOps.size() < totalOpSize && earliest < 100) {
    // get all operations that its operands are in the liveIn set or belong to
    // the scheduledOps
    SmallVector<Operation *, 4> schedulingOps =
        getNextLayerOps(block, liveIn, visitedOps);

    if (schedulingOps.empty()) {
      // print non scheduled ops
      llvm::errs() << earliest << " Non scheduled ops: \n";
      for (auto &op : block->getOperations()) {
        if (!isa<arith::ConstantOp>(op) && visitedOps.count(&op) == 0)
          llvm::errs() << op << "\n";
      }
      break;
    }
    for (auto op : schedulingOps) {
      if (prevPriority.count(op) == 0 && earliest < curDepth)
        continue;
      if (prevPriority.count(op) && earliest < prevPriority[op].first)
        continue;
      schedulePriority[op] = {earliest, INT_MAX};
      visitedOps.insert(op);
    }
    if (visitedOps.size() < totalOpSize)
      earliest++;
  }

  int latest = earliest;
  // accomandate the terminator op
  auto termOp = block->getTerminator();
  bool delayOneCC = false;
  // delay One clock cycle of the terminator execution if its first two operands
  // earliest should be the same as the last scheduled operation
  if (isa<cgra::ConditionalBranchOp>(termOp)) {
    delayOneCC =
        llvm::any_of(termOp->getOperands().take_front(2), [&](Value arg) {
          return arg.getDefiningOp() &&
                 schedulePriority.count(arg.getDefiningOp()) &&
                 schedulePriority[arg.getDefiningOp()].first == latest;
        });
  }
  latest += delayOneCC;
  schedulePriority[termOp] = {latest, latest};
  visitedOps.insert(termOp);

  while (!visitedOps.empty()) {
    // get all operations that its operands are in the liveOut set or belong
    // to the scheduledOps
    SmallVector<Operation *, 4> schedulingOps =
        getPreviousLayerOps(block, liveOut, visitedOps);

    for (auto op : schedulingOps) {
      schedulePriority[op].second = latest;
      visitedOps.erase(op);
    }
    latest--;
  }

  return schedulePriority;
}

static SmallVector<Operation *, 4> getScheduleOps(
    Block *block, int height,
    const std::map<Operation *, std::pair<int, int>> schedulePriority,
    SetVector<Operation *> scheduledOps,
    ScheduleStrategy strategy = ScheduleStrategy::ASAP) {
  SmallVector<Operation *, 4> schedulingOps;
  for (auto [op, priority] : schedulePriority) {
    if (scheduledOps.count(op))
      continue;

    // if the operation is in the scheduling height, add it to the scheduling
    // operations
    if (strategy == ScheduleStrategy::ASAP && priority.first <= height)
      schedulingOps.push_back(op);

    if (strategy == ScheduleStrategy::ALAP && priority.second >= height)
      schedulingOps.push_back(op);

    if (strategy == ScheduleStrategy::DYNAMIC && priority.first <= height &&
        priority.second >= height) {
      schedulingOps.push_back(op);
    }
  }

  // sort the scheduling operations according to the schedule priority
  std::sort(schedulingOps.begin(), schedulingOps.end(),
            [&](Operation *op1, Operation *op2) {
              return (schedulePriority.at(op1).first +
                      schedulePriority.at(op1).second) <
                     (schedulePriority.at(op2).first +
                      schedulePriority.at(op2).second);
            });
  return schedulingOps;
}

static int getDistance(int pe1, int pe2, int row, int col) {
  int x1 = pe1 / 4;
  int y1 = pe1 % 4;

  int x2 = pe2 / 4;
  int y2 = pe2 % 4;

  int dx = std::min(abs(x1 - x2), row - abs(x1 - x2));
  int dy = std::min(abs(y1 - y2), col - abs(y1 - y2));

  // dx = dx == 1 ? 0 : dx;
  // dy = dy == 1 ? 0 : dy;

  return dx + dy;
}

// Builds a child tree of operations starting from a given value.
// This function traverses the users of a given value (`val`) and constructs a
// tree of operations (`childTree`) that are reachable through the user chain
// within the same block (`blk`).
void buildChildTree(Value val, SetVector<Operation *> &childTree, Block *blk) {
  std::queue<Operation *> userQueue;
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    bool isInstantChild =
        !isa<cf::BranchOp>(user) &&
        !(isa<cgra::ConditionalBranchOp>(user) && use.getOperandNumber() > 1);
    if (isInstantChild)
      childTree.insert(user);

    if (user->getBlock() == blk)
      userQueue.push(user);
  }

  while (!userQueue.empty()) {
    auto currentUser = userQueue.front();
    userQueue.pop();

    for (auto &useChild : currentUser->getUses()) {
      auto userChild = useChild.getOwner();

      bool isInstantChild = !isa<cf::BranchOp>(userChild) &&
                            !(isa<cgra::ConditionalBranchOp>(userChild) &&
                              useChild.getOperandNumber() > 1);
      if (isInstantChild)
        childTree.insert(userChild);
      if (userChild->getBlock() == blk)
        userQueue.push(userChild);
    }
  }
}

static double getSpatialAffinityCost(
    Value val1, ScheduleUnit unit1, Value val2, ScheduleUnit unit2,
    const std::map<Operation *, std::pair<int, int>> heightMap,
    std::vector<ValuePlacement> &finiGraph, GridAttribute grid, Block *blk) {

  auto pe1 = unit1.pe;
  auto pe2 = unit2.pe;

  int maxHeight = 0;
  for (const auto &entry : heightMap)
    maxHeight = std::max(maxHeight, entry.second.second);

  SetVector<Operation *> childTree1;
  SetVector<Operation *> childTree2;
  buildChildTree(val1, childTree1, blk);

  if (!childTree1.contains(val2.getDefiningOp()))
    buildChildTree(val2, childTree2, blk);

  auto children = getInterSection<Operation *>(childTree1, childTree2);

  double affinity = 0;
  double maxRouting = grid.nRow / 2 + grid.nCol / 2;

  double bias = 0;
  for (auto child : children) {
    auto childHeight =
        heightMap.count(child) ? heightMap.at(child).first : maxHeight + 1;
    auto val1Height =
        (val1.getDefiningOp() && heightMap.count(val1.getDefiningOp()))
            ? heightMap.at(val1.getDefiningOp()).first
            : 0;
    auto val2Height =
        (val1.getDefiningOp() && heightMap.count(val1.getDefiningOp()))
            ? heightMap.at(val2.getDefiningOp()).first
            : 0;

    int depth1 = childHeight - val1Height;
    int depth2 = childHeight - val2Height;
    int depth = (depth1 + depth2) / 2;

    affinity += std::pow(2, -depth);

    // if child is liveout,the affinity must include the spatial distance to the
    // liveout space
    auto liveOutIt = std::find_if(
        finiGraph.begin(), finiGraph.end(), [&](const ValuePlacement &place) {
          return child->getNumResults() > 0 && place.val == child->getResult(0);
        });
    if (liveOutIt != finiGraph.end()) {
      auto childPE = liveOutIt->pe;
      // evaluate the pe1 and pe2's distance to the childPE
      int childDistance1 = getDistance(pe1, childPE, grid.nRow, grid.nCol);
      int childDistance2 = getDistance(pe2, childPE, grid.nRow, grid.nCol);
      bias = std::pow(2, -depth) * (childDistance1 + childDistance2) / 2.0 /
             maxRouting;
    }
  }

  return affinity * getDistance(pe1, pe2, grid.nRow, grid.nCol) / maxRouting +
         bias;
}

// get the torus routing PEs
static std::vector<unsigned>
getTorusRoutingPEs(unsigned pe, GridAttribute &attr, bool includeSelf = true) {
  std::vector<unsigned> routingPEs;
  if (includeSelf)
    routingPEs.push_back(pe);
  int nRow = attr.nRow;
  int nCol = attr.nCol;

  int row = pe / nCol;
  int col = pe % nCol;

  int left_col = (col - 1 + nCol) % nCol;
  int right_col = (col + 1) % nCol;
  int up_row = (row - 1 + nRow) % nRow;
  int bottom_row = (row + 1) % nRow;

  routingPEs.push_back(row * nCol + left_col);
  routingPEs.push_back(row * nCol + right_col);
  routingPEs.push_back(up_row * nCol + col);
  routingPEs.push_back(bottom_row * nCol + col);

  return routingPEs;
}

bool isOccupied(std::vector<ValuePlacement> curGraph,
                std::vector<ValuePlacement> finiGraph, unsigned pe) {
  // if find any curGraph that is occupied by the PE, return true
  for (const auto &place : curGraph) {
    bool taken = place.pe == pe && (place.regAttr == RegAttr::EX);
    auto val = place.val;
    // Check if the value exists in finiGraph with regAttr=EX|IE
    auto it = std::find_if(finiGraph.begin(), finiGraph.end(),
                           [&](const ValuePlacement &finiPlace) {
                             return finiPlace.val == val &&
                                    (finiPlace.regAttr == RegAttr::EX ||
                                     finiPlace.regAttr == RegAttr::IE);
                           });

    if (taken || it != finiGraph.end()) {
      return true;
    }
  }
  return false;
};

ValuePlacement *getOccupiedValue(std::vector<ValuePlacement> curGraph,
                                 std::vector<ValuePlacement> finiGraph,
                                 SetVector<mlir::Operation *> scheduledOps,
                                 Block *curBlk, unsigned pe) {
  for (auto &place : curGraph) {
    bool taken = place.pe == pe && (place.regAttr == RegAttr::EX);
    // check whether the value is still in use
    bool inUse = false;
    for (auto user : place.val.getUsers()) {
      if (user->getBlock() != curBlk || isa<cf::BranchOp>(user) ||
          (isa<cgra::ConditionalBranchOp>(user) &&
           user->getOperand(0) != place.val &&
           user->getOperand(1) != place.val))
        continue;
      if (scheduledOps.count(user) == 0)
        inUse = true;
    }
    if (taken && inUse) {
      return &place;
    }
    auto val = place.val;
    // Check if the value exists in finiGraph with regAttr=EX|IE
    auto it = std::find_if(finiGraph.begin(), finiGraph.end(),
                           [&](const ValuePlacement &finiPlace) {
                             return finiPlace.val == val &&
                                    (finiPlace.regAttr == RegAttr::EX ||
                                     finiPlace.regAttr == RegAttr::IE);
                           });
    if (it != finiGraph.end()) {
      llvm::errs() << "LiveOut\n";
      return &(*it);
    }
  }
  return nullptr;
}

// static double
// getAccessCost(Block *curBlk, std::vector<ValuePlacement> curGraph,
//               std::map<Operation *, ScheduleUnit> scheduleResult,
//               SetVector<Operation *> scheduledOps, SetVector<Value> liveOut,
//               std::vector<ValuePlacement> finiGraph, GridAttribute &attr,
//               size_t opNum,
//               std::map<mlir::Operation *, std::pair<int, int>>
//               schedulePriority, int height = 0) {
//   double cost = 0;

//   for (auto valPlace : curGraph) {
//     auto val = valPlace.val;
//     auto pe = valPlace.pe;
//     auto regAttr = valPlace.regAttr;
//     // if (val.getDefiningOp() && scheduledOps.count(val.getDefiningOp()))
//     //   continue;

//     // search available mobility range
//     std::vector<unsigned> mobilityRange;
//     if (regAttr == RegAttr::EX || regAttr == RegAttr::IE) {
//       for (auto routingPE : getTorusRoutingPEs(pe, attr))
//         if (!isOccupied(curGraph, finiGraph, routingPE))
//           mobilityRange.push_back(routingPE);
//     } else if (regAttr == RegAttr::IN) {
//       if (!isOccupied(curGraph, finiGraph, pe))
//         mobilityRange.push_back(pe);
//     }

//     // get non-scheduled users
//     SetVector<Operation *> nonScheduledUsers;
//     for (auto user : val.getUsers()) {
//       if (user->getBlock() != curBlk || isa<cf::BranchOp>(user) ||
//           (isa<cgra::ConditionalBranchOp>(user) && user->getOperand(0) != val
//           &&
//            user->getOperand(1) != val))
//         continue;
//       if (scheduledOps.count(user) == 0 && scheduleResult.count(user) == 0)
//         nonScheduledUsers.insert(user);
//     }
//     if (mobilityRange.size() == 0 && nonScheduledUsers.size() != 0) {
//       std::string message;
//       llvm::raw_string_ostream rso(message);
//       rso << "Value: " << val << " has no mobility range.\n";
//       logMessage(rso.str());
//     }

//     cost +=
//         nonScheduledUsers.size() / std::max(0.01,
//         (double)mobilityRange.size());
//   }

//   // add the blocked PE;
//   double blockPenalty = 0;
//   // for (auto i = 0; i < attr.nRow * attr.nCol; i++)
//   //   if (isOccupied(curGraph, finiGraph, i))
//   //     blockPenalty += 0.1;

//   // add the used register
//   return opNum == 0 ? blockPenalty : blockPenalty + cost / opNum;
// }
static double getAccessCost(
    Block *curBlk, std::vector<ValuePlacement> curGraph,
    std::map<Operation *, ScheduleUnit> scheduleResult,
    SetVector<Operation *> scheduledOps, SetVector<Value> liveOut,
    std::vector<ValuePlacement> finiGraph, GridAttribute &attr, size_t opNum,
    const std::map<mlir::Operation *, std::pair<int, int>> schedulePriority,
    int height = 0) {

  double cost = 0;
  for (auto i = 0; i < attr.nRow * attr.nCol; i++) {
    // get the aggregation of the value placement in current PE
    SmallVector<ValuePlacement> liveVals;
    bool isCritical = false;
    ValuePlacement criticalOp;
    for (auto valPlace : curGraph) {
      if (valPlace.pe == i) {
        auto defOp = valPlace.val.getDefiningOp();
        isCritical = defOp && schedulePriority.count(defOp) &&
                     (schedulePriority.at(defOp).first ==
                      schedulePriority.at(defOp).second);
        if (isCritical) {
          criticalOp = valPlace;
          isCritical = true;
        } else {
          liveVals.push_back(valPlace);
        }
      }
    }

    // if criticalOp exist, insert it to the begin of liveVals
    if (isCritical)
      liveVals.insert(liveVals.begin(), criticalOp);

    // Evaluate the access cost in current PE
    for (auto valPlace : liveVals) {
      auto val = valPlace.val;
      auto regAttr = valPlace.regAttr;
      auto pe = valPlace.pe;
      SetVector<Operation *> nonScheduledUsers;
      for (auto user : val.getUsers()) {
        if (user->getBlock() != curBlk || isa<cf::BranchOp>(user) ||
            (isa<cgra::ConditionalBranchOp>(user) &&
             user->getOperand(0) != val && user->getOperand(1) != val))
          continue;
        if (scheduledOps.count(user) == 0 && scheduleResult.count(user) == 0)
          nonScheduledUsers.insert(user);
      }

      // search available mobility range
      std::vector<unsigned> mobilityRange;
      if (regAttr == RegAttr::EX || regAttr == RegAttr::IE) {
        for (auto routingPE : getTorusRoutingPEs(pe, attr))
          if (!isOccupied(curGraph, finiGraph, routingPE))
            mobilityRange.push_back(routingPE);
      } else if (regAttr == RegAttr::IN) {
        if (!isOccupied(curGraph, finiGraph, pe))
          mobilityRange.push_back(pe);
      }

      if (mobilityRange.size() == 0 && nonScheduledUsers.size() != 0) {
        std::string message;
        llvm::raw_string_ostream rso(message);
        rso << "Value: " << val << " has no mobility range.\n";
        logMessage(rso.str());
      }

      cost += nonScheduledUsers.size() /
              std::max(0.01, (double)mobilityRange.size());
      // only evaluate cost for the critical operation
      if (isCritical && valPlace.val == criticalOp.val)
        break;
    }
  }
  return opNum == 0 ? 0 : 0 + cost / opNum;
}

static double getSuccessCost(
    std::map<Operation *, ScheduleUnit> scheduleResult,
    const std::map<Operation *, std::pair<int, int>> schedulePriority,
    SmallVector<mlir::Operation *, 4> totalOps) {
  double cost = 0;
  double normRatio = 0;
  for (auto op : totalOps) {
    auto earliest = schedulePriority.at(op).first;
    auto latest = schedulePriority.at(op).second;
    double weight = std::pow(2, earliest - latest);
    if (scheduleResult.count(op) != 0) {
      cost += weight;
    }
    normRatio += weight;
  }
  double failAll = cost == 0 ? 1e2 : 0;
  return normRatio == 0 ? failAll : failAll + (normRatio - cost) / normRatio;
}

static double getSpatialAffinityTotalCost(
    Block *curBlk, std::map<Operation *, ScheduleUnit> scheduleResult,
    std::vector<ValuePlacement> initGraph,
    std::vector<ValuePlacement> finiGraph,
    std::map<Operation *, std::pair<int, int>> heightMap, GridAttribute grid,
    SetVector<Value> liveOut, SetVector<Operation *> scheduledOps) {
  double cost = 0;
  int count = 0;
  for (auto op : scheduleResult) {
    if (op.first->getNumResults() == 0)
      continue;
    // check the affinity cost with livein values
    for (auto place : initGraph) {
      if (place.val == op.first->getResult(0) ||
          !isLive(place.val, curBlk, liveOut, scheduledOps))
        continue;

      cost += getSpatialAffinityCost(
          place.val, {0, (int)place.pe, -1}, op.first->getResult(0), op.second,
          heightMap, finiGraph, grid, op.first->getBlock());
    }
    count++;
  }

  return count == 0 ? 0 : cost / count;
}

static SetVector<unsigned> getAvailablePEs(std::map<int, PERegUse> &freeReg,
                                           RegAttr reg) {
  SetVector<unsigned> peList;
  for (auto [pe, use] : freeReg) {
    if (reg == RegAttr::IE && use.inNum > 0 && use.exAvail) {
      peList.insert(pe);
      continue;
    }
    if (reg == RegAttr::IN && use.inNum > 0) {
      peList.insert(pe);
      continue;
    }
    if (reg == RegAttr::EX && use.exAvail) {
      peList.insert(pe);
      continue;
    }
  }
  return peList;
}
static std::map<int, PERegUse>
getResourceGraph(std::vector<ValuePlacement> curGraph, GridAttribute &attr) {
  // get available slots
  std::map<int, PERegUse> freeReg;
  for (int i = 0; i < attr.nRow * attr.nCol; i++)
    freeReg[i] = {(int)attr.maxReg, true};

  // remove resources occupied by values in initGraph
  for (auto place : curGraph) {
    auto pe = place.pe;
    auto reg = place.regAttr;
    // check whether
    // if regAttr is IN, remove the resource from the freeReg
    if (reg == RegAttr::IN || reg == RegAttr::IE)
      freeReg[pe].inNum--;
    // if regAttr is EX, remove the resource from the freeReg
    if (reg == RegAttr::EX || reg == RegAttr::IE)
      freeReg[pe].exAvail = false;
  }
  return freeReg;
}

static void updateEmbeddingGraph(std::vector<ValuePlacement> &curGraph,
                                 Block *curBlk, GridAttribute attr,
                                 Operation *op, unsigned pe, RegAttr reg,
                                 SetVector<Operation *> &scheduledOps,
                                 SetVector<Value> &liveout) {
  // op takes place of pe, invalidate the Rout of the pe
  for (auto it = curGraph.begin(); it != curGraph.end();) {
    if (it->pe != pe) {
      ++it;
      continue;
    }

    if (it->regAttr == RegAttr::EX) {
      // remove it from the curGraph
      it = curGraph.erase(it);
      continue;
    }

    if (it->regAttr == RegAttr::IE) {
      it->regAttr = RegAttr::IN;
    }
    ++it;

    // if (it->regAttr == RegAttr::IN) {
    //   // check whether the value should be kept anymore
    //   auto val = it->val;
    //   if (!isLive(val, liveout, scheduledOps)) {
    //     std::string message;
    //     llvm::raw_string_ostream rso(message);
    //     rso << val << " [" << pe << " " << static_cast<int>(reg)
    //         << "] ERASED\n";
    //     if (isa<BlockArgument>(val))
    //       logMessage(rso.str());
    //     it = curGraph.erase(it);
    //   } else {
    //     // log the value and its register
    //     std::string message;
    //     llvm::raw_string_ostream rso(message);
    //     rso << val << " [" << pe << " " << static_cast<int>(reg) << "]
    //     LIVE\n"; if (isa<BlockArgument>(val))
    //       logMessage(rso.str());
    //     ++it;
    //   }
    // }
  }

  // update internal register
  for (auto p = 0; p < attr.nCol * attr.nRow; p++) {
    for (auto it = curGraph.begin(); it != curGraph.end();) {
      if (it->pe != p || it->regAttr != RegAttr::IN) {
        ++it;
        continue;
      }
      // check whether the value should be kept anymore
      auto val = it->val;
      if (!isLive(val, curBlk, liveout, scheduledOps)) {
        it = curGraph.erase(it);
      } else {
        ++it;
      }
    }
  }

  // add operation to curGraph
  if (op->getNumResults() > 0)
    curGraph.push_back({op->getResult(0), pe, reg});
}

static std::map<int, PERegUse> getAvailableResourceGraph(
    std::vector<ValuePlacement> curGraph, SetVector<Operation *> scheduledOps,
    GridAttribute &attr, Operation *scheduleOp, SetVector<Value> liveout,
    const std::vector<ValuePlacement> finiGraph,
    std::map<Operation *, std::pair<unsigned, RegAttr>> tmpResult) {
  // get available slots
  SetVector<Operation *> layerScheduledOps;
  std::vector<bool> used(attr.nRow * attr.nCol, false);
  for (auto [op, unit] : tmpResult) {
    layerScheduledOps.insert(op);
    used[unit.first] = true;
  }

  // remove resources occupied by liveout values
  for (auto valP : curGraph) {
    auto val = valP.val;
    auto regAttr = valP.regAttr;
    // if find val should be external live in finiGraph
    if (std::find_if(finiGraph.begin(), finiGraph.end(),
                     [&](const ValuePlacement &place) {
                       return place.val == val &&
                              (place.regAttr == RegAttr::EX ||
                               place.regAttr == RegAttr::IE);
                     }) != finiGraph.end()) {
      logMessage(std::to_string(valP.pe) + " for live out\n");
      used[valP.pe] = true;
    }
  }

  std::map<int, PERegUse> freeReg;
  for (int i = 0; i < attr.nRow * attr.nCol; i++)
    freeReg[i] = {(int)attr.maxReg, !used[i]};

  // remove resources occupied by values in initGraph
  for (auto place : curGraph) {
    auto pe = place.pe;
    auto reg = place.regAttr;
    // if the place.val is not live, it does not occupy the resource
    bool deadVal =
        !isLive(place.val, scheduleOp->getBlock(), liveout, scheduledOps);
    if (deadVal)
      continue;

    // if regAttr is IN, remove the resource from the freeReg
    if (reg == RegAttr::IN || reg == RegAttr::IE)
      freeReg[pe].inNum--;
    // if regAttr is EX, remove the resource from the freeReg
    if (reg == RegAttr::EX) {
      SetVector<Operation *> totalScheduledOp = scheduledOps;
      totalScheduledOp.insert(layerScheduledOps.begin(),
                              layerScheduledOps.end());
      freeReg[pe].exAvail &=
          !isLiveExcept(place.val, scheduleOp->getBlock(), scheduleOp, liveout,
                        totalScheduledOp);
    }
  }
  return freeReg;
}

static void printResourceGraph(std::map<int, PERegUse> freeReg,
                               GridAttribute attr) {
  // print nRow and nCol
  std::string message;
  for (int i = 0; i < attr.nCol; i++) {
    for (int j = 0; j < attr.nRow; j++) {
      auto pe = i * attr.nRow + j;
      auto availPE = freeReg[pe].exAvail;
      auto availRegs = freeReg[pe].inNum;
      if (availPE)
        message += "[? | " + std::to_string(availRegs) + "] ";
      else
        message += "[X | " + std::to_string(availRegs) + "] ";
    }
    message += "\n";
  }
  logMessage(message);
}

static int getInitialLiveInPlacement(
    Value in, Block *block, std::vector<ValuePlacement> &initGraph,
    const std::map<Block *, SetVector<Value>> liveIns,
    const std::map<Block *, SetVector<Value>> liveOuts, GridAttribute &attr) {
  // if the value has more than 3 users, it should be external register
  int userNum = 0;
  for (auto user : in.getUsers())
    // if (user->getBlock() == block)
    userNum++;

  // stored in external if the value has more than 3 users or is live in a
  // limited number of blocks
  bool isExternal =
      userNum > 3 || getValueLiveLength(in, liveIns, liveOuts) < 3;
  // if the value exists in a long path, it should be internal register
  bool isInternal =
      !isExternal || getValueLiveLength(in, liveIns, liveOuts) >= 3;

  RegAttr regAttr;
  if (isExternal && isInternal)
    regAttr = RegAttr::IE;
  else if (isExternal)
    regAttr = RegAttr::EX;
  else
    regAttr = RegAttr::IN;
  // get available slots
  std::map<int, PERegUse> freeReg = getResourceGraph(initGraph, attr);
  auto peList = getAvailablePEs(freeReg, regAttr);

  // pe is randomly assigned from the available PE list
  unsigned pe = peList[rand() % peList.size()];
  initGraph.push_back({in, pe, regAttr});
  // update the freeReg
  if (regAttr == RegAttr::IN || regAttr == RegAttr::IE)
    freeReg[pe].inNum--;
  if (regAttr == RegAttr::EX || regAttr == RegAttr::IE)
    freeReg[pe].exAvail = false;

  std::string inStr;
  llvm::raw_string_ostream rso(inStr);
  rso << in;

  logMessage("INIT GRAPH: " + rso.str() + " " + std::to_string(pe) + " " +
             std::to_string(static_cast<int>(regAttr)));
}

void BasicBlockOpAssignment::initEmbeddingGraphWithLiveIn(
    std::map<Block *, SetVector<Value>> liveIns,
    std::map<Block *, SetVector<Value>> liveOuts,
    std::vector<ValuePlacement> &initGraph, OpBuilder &builder,
    GridAttribute attr) {

  auto liveIn = liveIns[curBlock];
  for (auto val : liveIn) {
    auto it = std::find_if(initGraph.begin(), initGraph.end(),
                           [&](ValuePlacement p) { return p.val == val; });
    // randomly assign the PE to the liveIn value
    if (it == initGraph.end()) {
      getInitialLiveInPlacement(val, curBlock, initGraph, liveIns, liveOuts,
                                attr);
    }

    // if find in the internal register in the liveIn graph, pop it out
    int userCount = 0;
    for (auto user : val.getUsers())
      if (user->getBlock() == curBlock)
        userCount++;
    if (userCount >= 3) {
      // create pop out operation
      builder.setInsertionPoint(&curBlock->getOperations().front());
      Operation *movOp = createAtomicMovOp(val, true, true);
    }
  }
  // if val not in initGraph, assign the liveIn value with the lowest cost
}

LogicalResult
finiEmbeddingGraphWithLiveOut(std::vector<ValuePlacement> &finiGraph,
                              std::vector<ValuePlacement> &scheduleGraph,
                              OpBuilder &builder, GridAttribute attr) {
  for (auto &req : finiGraph) {
    auto place =
        std::find_if(scheduleGraph.begin(), scheduleGraph.end(),
                     [&](ValuePlacement p) { return p.val == req.val; });
    if (place == scheduleGraph.end())
      return failure();

    if (req.pe == place->pe)
      continue;

    auto srcPE = place->pe;
    auto dstPE = req.pe;

    // find a path from srcPE to dstPE
    // if req.pe != scheduleGraph.pe
  }
}

static void removeElement(SetVector<unsigned> &vec, unsigned pe) {
  auto it = std::find(vec.begin(), vec.end(), pe);
  if (it != vec.end())
    vec.erase(it);
}

ValuePlacement getSrcValuePlacement(Value src, Operation *op,
                                    std::vector<ValuePlacement> curGraph) {

  auto it = std::find_if(curGraph.begin(), curGraph.end(),
                         [&](ValuePlacement p) { return p.val == src; });

  if (it != curGraph.end())
    return *it;
  return {nullptr, UINT_MAX, RegAttr::NK};
}

bool isCstZero(Operation *op) {
  auto cstOp = dyn_cast_or_null<arith::ConstantOp>(op);
  if (cstOp == nullptr)
    return false;

  auto attr = cstOp.getValue();
  if ((attr.isa<mlir::IntegerAttr>() &&
       attr.cast<mlir::IntegerAttr>().getValue().isZero()) ||
      (attr.isa<mlir::FloatAttr>() &&
       attr.cast<mlir::FloatAttr>().getValue().isZero()))
    return true;
  return false;
}

Value getRouteSrcOp(Operation *op, std::vector<Value> spilledVals,
                    unsigned &step) {
  // op = add srcOp, zero, get SrcOp value until the value is not in
  // spilledVals first check whether op match the pattern;

  if (isa<arith::AddIOp>(op) || isa<arith::AddFOp>(op))
    if (std::find(spilledVals.begin(), spilledVals.end(), op->getOperand(0)) !=
        spilledVals.end())
      if (isCstZero(op->getOperand(1).getDefiningOp()))
        return op->getOperand(0);

  return op->getResult(0);
}

SmallVector<Operation *, 4> getRouteOpStep1(Value val) {
  SmallVector<Operation *, 4> routeOps;
  for (auto user : val.getUsers()) {
    if ((isa<arith::AddIOp>(user) || isa<arith::AddFOp>(user)) &&
        user->getOperand(0) == val && user->getOperand(1).getDefiningOp() &&
        isCstZero(user->getOperand(1).getDefiningOp())) {
      routeOps.push_back(user);
    }
  }
  return routeOps;
}

/// Get the valid placement space of the operation.
std::vector<placeunit> BasicBlockOpAssignment::searchOpPlacementSpace(
    Operation *scheduleOp, std::vector<ValuePlacement> &curGraph,
    const std::vector<ValuePlacement> &finiGraph,
    std::map<Operation *, std::pair<unsigned, RegAttr>> tmpResult) {
  int nRow = attr.nRow;
  int nCol = attr.nCol;
  std::vector<std::pair<int, RegAttr>> placementSpace;

  // get the available hardware resources
  auto regUse = getAvailableResourceGraph(
      curGraph, scheduledOps, attr, scheduleOp, liveout, finiGraph, tmpResult);
  printResourceGraph(regUse, attr);

  SetVector<unsigned> availablePEs;
  for (int i = 0; i < nRow * nCol; i++) {
    if (!regUse[i].exAvail)
      continue;

    availablePEs.insert(i);
    if (isRouteOp(scheduleOp)) {
      placementSpace.push_back({i, RegAttr::EX});
      continue;
    }

    if (regUse[i].inNum > 0)
      placementSpace.push_back({i, RegAttr::IE});
    if (regUse[i].inNum == 0)
      placementSpace.push_back({i, RegAttr::EX});
  }

  // limit the placement space according to its consumer and producer
  for (auto &opVal : scheduleOp->getOpOperands()) {
    if (isa<cf::BranchOp>(scheduleOp) ||
        (isa<cgra::ConditionalBranchOp>(scheduleOp) &&
         opVal.getOperandNumber() > 1))
      break;
    SetVector<unsigned> routingPEs;
    auto opr = opVal.get();
    auto oprPlace = getSrcValuePlacement(opr, scheduleOp, curGraph);
    // not find existing value, no need to limit the placement space
    if (oprPlace.val == nullptr)
      continue;

    auto pe = oprPlace.pe;
    auto regAttr = oprPlace.regAttr;
    if (regAttr == RegAttr::IN)
      routingPEs.insert(pe);
    else if (regAttr == RegAttr::EX || regAttr == RegAttr::IE) {
      bool selfRoute =
          isRouteOp(scheduleOp) && isRouteOp(oprPlace.val.getDefiningOp());
      for (auto routingPE : getTorusRoutingPEs(pe, attr, !selfRoute))
        routingPEs.insert(routingPE);
    }
    // placementSpace intersect routingPEs
    getSubSet<unsigned>(availablePEs, routingPEs);
  }

  // Filter placementSpace to keep only PEs that are in routing scope
  placementSpace.erase(
      std::remove_if(placementSpace.begin(), placementSpace.end(),
                     [&](const std::pair<unsigned, RegAttr> &p) {
                       return availablePEs.count(p.first) == 0;
                     }),
      placementSpace.end());

  if (placementSpace.empty())
    return placementSpace;

  // if the operation result is restricted by the finiGraph, it is restricted
  // to the PE that is used by the finiGraph
  for (auto res : scheduleOp->getResults()) {
    auto it = std::find_if(finiGraph.begin(), finiGraph.end(),
                           [&](ValuePlacement p) { return p.val == res; });
    if (it != finiGraph.end()) {
      auto pe = it->pe;
      SetVector<unsigned> tempSet;
      tempSet.insert(it->pe);
      // print availablePEs

      std::string message;
      llvm::raw_string_ostream rso(message);
      for (auto p : availablePEs) {
        rso << p << " ";
      }
      rso << "\n";

      getSubSet<unsigned>(availablePEs, tempSet);

      if (availablePEs.empty()) {
        rso << "SEARCH SPACE: Operation: " << *scheduleOp << " xx " << it->pe
            << " xx has no available placement space.\n";
        logMessage(rso.str());
        return {};
      }

      // auto tmpAvailablePEs = availablePEs;
      // getSubSet<unsigned>(tmpAvailablePEs, tempSet);
      // if (!tmpAvailablePEs.empty())
      //   availablePEs = tmpAvailablePEs;
    }
  }

  placementSpace.erase(
      std::remove_if(placementSpace.begin(), placementSpace.end(),
                     [&](const std::pair<unsigned, RegAttr> &p) {
                       return availablePEs.count(p.first) == 0;
                     }),
      placementSpace.end());

  return placementSpace;
}

static std::pair<int, compigra::RegAttr>
getSample(std::vector<std::pair<int, compigra::RegAttr>> opSpace,
          bool allowNull = true) {
  std::pair<int, compigra::RegAttr> assignPE;
  double randomValue = static_cast<double>(rand()) / RAND_MAX;
  if (randomValue < 0.2 && allowNull) {
    assignPE = {-1, RegAttr::NK};
  } else {
    assignPE = *(opSpace.begin() + rand() % opSpace.size());
  }
  return assignPE;
}

int BasicBlockOpAssignment::placeOperations(
    int timeSlot, SmallVector<Operation *, 4> &schedulingOps,
    std::map<Operation *, ScheduleUnit> &scheduleResult,
    std::vector<ValuePlacement> &curGraph,
    std::map<Operation *, std::vector<placeunit>> &space,
    std::vector<ValuePlacement> &finiGraph, int shuffleOpIdx) {

  unsigned suc = 0;
  std::map<Operation *, std::pair<unsigned, RegAttr>> tmpResult;
  SetVector<Operation *> tmpScheduledOps;

  for (auto [idx, op] : llvm::enumerate(schedulingOps)) {
    std::pair<int, compigra::RegAttr> assignPE;
    std::string message;
    llvm::raw_string_ostream rso(message);
    rso << "Operation: " << *op;
    // simulated annealing to get a random placement
    if ((int)idx <= shuffleOpIdx) {
      // if (scheduleResult.count(op) == 0 && shuffleOpIdx > 0)
      //   continue;
      if (space.count(op) == 0) {
        // logMessage(rso.str() + " -> Not scheduled quit");
        continue;
      }
      auto opSpace = space.at(op);
      if (idx == shuffleOpIdx) {
        // opSpace.insert(opSpace.begin(), {-1, RegAttr::NK});
        assignPE = getSample(opSpace);
        if (assignPE.first == -1) {
          // remove result from the scheduleResult
          scheduleResult.erase(op);
          // rso << " choose to not assigned";
          // logMessage(rso.str());
          continue;
        }
      } else {
        if (!scheduleResult.count(op)) {
          continue;
        }
        assignPE = (scheduleResult.at(op).reg == attr.maxReg)
                       ? std::pair{scheduleResult.at(op).pe, RegAttr::EX}
                       : std::pair{scheduleResult.at(op).pe, RegAttr::IE};
        // rso << "not flip : ";
      }
    } else {
      // search new placement space
      auto placementSpace =
          searchOpPlacementSpace(op, curGraph, finiGraph, tmpResult);
      // randomly choose a placement space
      if (placementSpace.empty()) {
        rso << " has empty placement space.\n";
        logMessage(rso.str());
        continue;
      }

      // insert a non-scheduled decision
      space[op] = placementSpace;
      assignPE = getSample(placementSpace);
      // placementSpace.insert(placementSpace.begin(), {-1, RegAttr::NK});
      // assignPE = *(placementSpace.begin() + rand() %
      // placementSpace.size()); not assign operations at this time slot
      if (assignPE.first == -1) {
        // rso << " choose to not assigned\n";
        // logMessage(rso.str());
        continue;
      }
      // rso << " flip ";
    }

    // rso << " Assigned: " << assignPE.first
    //     << " RegAttr: " << static_cast<int>(assignPE.second) << "\n";
    // logMessage(rso.str());
    suc++;
    tmpScheduledOps.insert(op);
    tmpResult[op] = assignPE;
  }

  for (auto op : tmpScheduledOps) {
    auto pe = tmpResult[op].first;
    RegAttr regAttr = tmpResult[op].second;
    SetVector<Operation *> totalScheduledOp = scheduledOps;
    totalScheduledOp.insert(tmpScheduledOps.begin(), tmpScheduledOps.end());
    updateEmbeddingGraph(curGraph, curBlock, attr, op, pe, regAttr,
                         totalScheduledOp, liveout);
  }

  // write tmpScheduleResult to scheduleResult
  for (auto [op, unit] : tmpResult) {
    if (unit.second == RegAttr::EX)
      scheduleResult[op] = {timeSlot, (int)unit.first, (int)attr.maxReg};
    else
      scheduleResult[op] = {timeSlot, (int)unit.first, -1};
  }
  return suc;
}

int shuffleSearchSpace(
    std::map<Operation *, std::vector<placeunit>> &searchSpace,
    std::map<Operation *, ScheduleUnit> &scheduleResult,
    SmallVector<mlir::Operation *, 4> schedulingOps) {
  // randomly choose an operation where its searchSpace has multiple elements
  std::vector<int> shuffleIds;
  // for (int i = 0; i < schedulingOps.size(); i++) {
  //   auto it = searchSpace.find(schedulingOps[i]);
  //   if (it != searchSpace.end() && it->second.size() > 1)
  //     shuffleIds.push_back(i);
  // }
  // // if no operation can be shuffled, return nullptr
  // if (shuffleIds.empty())
  //   return -1;
  // unsigned shuffleOpIdx = shuffleIds[rand() % shuffleIds.size()];

  unsigned shuffleOpIdx = rand() % schedulingOps.size();

  // keep all elements before the shuffleOpIdx in the searchSpace, and remove
  // others
  for (auto ind = shuffleOpIdx + 1; ind < schedulingOps.size(); ind++) {
    auto it = searchSpace.find(schedulingOps[ind]);
    if (it != searchSpace.end())
      searchSpace.erase(it);

    auto it2 = scheduleResult.find(schedulingOps[ind]);
    if (it2 != scheduleResult.end())
      scheduleResult.erase(it2);
  }
  // logMessage("shuffleOpIdx:" + std::to_string(shuffleOpIdx));
  return shuffleOpIdx;
}

static void sortProducersByWeight(std::vector<ValuePlacement> &producers,
                                  const std::vector<unsigned> &weight) {
  std::vector<size_t> indices(producers.size());
  // Initialize indices 0, 1, 2, ..., n-1
  for (size_t i = 0; i < indices.size(); ++i)
    indices[i] = i;

  // Sort indices based on weight
  std::sort(indices.begin(), indices.end(),
            [&weight](size_t a, size_t b) { return weight[a] > weight[b]; });

  // Apply the sorted indices to reorder producers
  std::vector<ValuePlacement> sortedProducers;
  sortedProducers.reserve(producers.size());
  for (size_t i : indices)
    sortedProducers.push_back(producers[i]);

  producers = std::move(sortedProducers);
}

int BasicBlockOpAssignment::createRoutePath(
    Operation *failOp, std::vector<ValuePlacement> &producers,
    std::vector<unsigned> &movs, std::vector<ValuePlacement> curGraph,
    std::vector<ValuePlacement> finiGraph,
    SmallVector<mlir::Operation *, 4> otherFailureOps, unsigned threshold) {
  // get all the producers of the failOp
  for (auto &opr : failOp->getOpOperands()) {
    if (isa<cf::BranchOp>(failOp) ||
        (isa<cgra::ConditionalBranchOp>(failOp) && opr.getOperandNumber() > 1))
      continue;
    auto opValue = opr.get();
    auto prodPtr =
        std::find_if(curGraph.begin(), curGraph.end(),
                     [&](ValuePlacement p) { return p.val == opValue; });
    bool notInclude =
        std::find_if(producers.begin(), producers.end(), [&](ValuePlacement p) {
          return p.val == prodPtr->val;
        }) == producers.end();
    if (prodPtr != curGraph.end() && notInclude) {
      producers.push_back(*prodPtr);
      movs.push_back(0);
    }
  }

  // sort the producer with their usage by other failure operations
  std::vector<unsigned> weight(producers.size(), 0);
  std::set<unsigned> prodPEs;
  blockedProdPEs.clear();
  for (auto [ind, prod] : llvm::enumerate(producers)) {
    auto val = prod.val;
    prodPEs.insert(prod.pe);
    for (auto user : val.getUsers())
      if (std::find(otherFailureOps.begin(), otherFailureOps.end(), user) !=
          otherFailureOps.end())
        weight[ind]++;
  }
  sortProducersByWeight(producers, weight);
  // print producer and their weight

  // get the available PEs
  SetVector<unsigned> avaiPEs;
  for (auto i = 0; i < attr.nRow * attr.nCol; i++) {
    auto occupied =
        getOccupiedValue(curGraph, finiGraph, scheduledOps, curBlock, i);
    // if the pe is occupied but not by the producer, return false
    if (occupied && prodPEs.count(i) > 0) {
      std::string message;
      llvm::raw_string_ostream rso(message);
      rso << "Occupied PE: " << occupied->val << " at " << occupied->pe
          << " by producer\n";
      logMessage(rso.str());
      //  if the producer PE is occupied by other value, we need to route the
      //  other value
      bool blockPop = false;
      for (auto prod : producers) {
        if (prod.val == occupied->val) {
          blockPop = true;
          break;
        }
      }
      if (!blockPop) {
        blockedProdPEs.insert(i);
      }
    }
    if (!occupied)
      avaiPEs.insert(i);
  }
  if (!blockedProdPEs.empty()) {
    logMessage("Blocked PE :" + std::to_string(blockedProdPEs[0]));
    return -1;
  }

  // define map population function of step 1
  auto populateRoutingPEs = [&](unsigned pe, SetVector<unsigned> &map) {
    for (auto routingPE : getTorusRoutingPEs(pe, attr))
      if (avaiPEs.count(routingPE) > 0)
        map.insert(routingPE);
  };

  // initialize pop map for each producer
  std::map<unsigned, SetVector<unsigned>> popMap;
  // init popMap
  for (auto [ind, prod] : llvm::enumerate(producers)) {
    popMap[ind].insert(prod.pe);
    auto regAttr = prod.regAttr;
    if (regAttr == RegAttr::EX || regAttr == RegAttr::IE) {
      populateRoutingPEs(prod.pe, popMap[ind]);
    }
    auto spilVal = prod.val;
    if (std::find(spilledVals.begin(), spilledVals.end(), spilVal) !=
        spilledVals.end()) {
      populateRoutingPEs(prod.pe, popMap[ind]);
      movs[ind] += 1;
    }
  }

  //  check whether the intersection exists
  auto intersection = popMap[0];
  for (size_t i = 1; i < producers.size(); ++i)
    intersection = getInterSection<unsigned>(intersection, popMap[i]);
  if (!intersection.empty()) {
    // check whether movs[ind] are all 0, if yes, meaning that there are
    // available spots to accomandate the consumers, but is less than the
    // total number of placed operations. Let movs[0]++ for routing
    // for (size_t i = 1; i < producers.size(); ++i) {
    //   if (movs[i] > 0)
    //     return true;
    // }
    // movs[0]++;

    // If the producers need to be routed, we can return 0, otherwise, meaning
    // that the failedOp can access its producers directly, it is restricted by
    // its consumer access range.
    for (size_t i = 1; i < producers.size(); ++i) {
      if (movs[i] > 0)
        return 0;
    }
    return 1;
  }

  int population = 0;
  while (population < threshold) {
    population++;
    // check the intersection of the popMap
    SetVector<unsigned> intersection;
    for (auto [ind, prod] : llvm::enumerate(producers)) {
      // populate the popMap
      auto pe = prod.pe;
      auto regAttr = prod.regAttr;
      auto prevPop = popMap[ind];
      for (auto routPE : prevPop)
        populateRoutingPEs(routPE, popMap[ind]);
      movs[ind] += 1;

      //  check whether the intersection exists
      auto intersection = popMap[0];
      for (size_t i = 1; i < producers.size(); ++i)
        intersection = getInterSection<unsigned>(intersection, popMap[i]);
      if (!intersection.empty())
        return 0;
    }
  }
  logMessage("failed to find path");
  return -1;
}

Operation *BasicBlockOpAssignment::createAtomicMovOp(Value val,
                                                     bool replaceCurBlkUse,
                                                     bool customLoc) {
  if (!customLoc) {
    if (isa<BlockArgument>(val) || val.getParentBlock() != curBlock)
      builder.setInsertionPoint(&curBlock->getOperations().front());
    else
      builder.setInsertionPointAfter(val.getDefiningOp());
  }
  Operation *movOp;
  if (isa<IntegerType>(val.getType())) {
    movOp =
        builder.create<arith::AddIOp>(val.getLoc(), val, zeroIntOp.getResult());
  } else if (isa<FloatType>(val.getType())) {
    movOp = builder.create<arith::AddFOp>(val.getLoc(), val,
                                          zeroFloatOp.getResult());
  }

  if (replaceCurBlkUse) {
    // replace the use of val to movOp
    val.replaceUsesWithIf(movOp->getResult(0), [&](OpOperand &use) {
      auto owner = use.getOwner();
      return owner->getBlock() == curBlock && owner != movOp &&
             !(isa<cf::BranchOp>(owner) ||
               (isa<cgra::ConditionalBranchOp>(owner) &&
                use.getOperandNumber() > 1));
    });
  }
  return movOp;
}

SmallVector<Operation *, 4>
BasicBlockOpAssignment::routeOperation(std::vector<ValuePlacement> producers,
                                       std::vector<unsigned> movs,
                                       Operation *failedOp) {
  SmallVector<Operation *, 4> routeOps;
  llvm::errs() << "failed op: " << *failedOp << "\n";
  for (size_t i = 0; i < producers.size(); ++i) {
    auto producer = producers[i];
    auto movNum = movs[i];
    if (movNum == 0)
      continue;
    // check whether the mov operation exists
    unsigned movStep = 0;
    auto origVal = producer.val;
    llvm::errs() << "producer: " << origVal << "\n";

    while (std::find(spilledVals.begin(), spilledVals.end(), origVal) !=
               spilledVals.end() &&
           movStep < movNum) {
      origVal = getRouteOpStep1(origVal)[0]->getResult(0);
      movStep++;
    }

    if (movStep == movNum) {
      // replace the producer use with the origVal
      // failedOp->replaceUsesOfWith(producer.val, origVal);
      // test whether replace
      llvm::errs() << "Replace use in " << *failedOp << " with " << origVal
                   << "\n";
      producer.val.replaceUsesWithIf(origVal, [&](OpOperand &opr) {
        auto owner = opr.getOwner();
        if ((isa<cgra::ConditionalBranchOp>(owner) &&
             opr.getOperandNumber() > 1))
          return false;
        return opr.getOwner() == failedOp;
      });
      continue;
    }
    // if found producer.val in spilledVals, meaning the value is already
    // spilled, then find the next one

    // Route producer movNum times
    auto routeVal = origVal;
    Operation *finalRouteOp = nullptr;
    for (unsigned j = movStep; j < movNum; ++j) {
      Operation *movOp = createAtomicMovOp(routeVal, false, false);
      finalRouteOp = movOp;
      spilledVals.push_back(routeVal);
      routeOps.push_back(movOp);
      routeVal = movOp->getResult(0);
    }
    // replace the use of origVal with the finalRouteOp if the use does not
    // belongs to scheduledOps
    // failedOp->replaceUsesOfWith(origVal, finalRouteOp->getResult(0));
    origVal.replaceUsesWithIf(finalRouteOp->getResult(0), [&](OpOperand &opr) {
      auto owner = opr.getOwner();
      if ((isa<cgra::ConditionalBranchOp>(owner) && opr.getOperandNumber() > 1))
        return false;
      return opr.getOwner() == failedOp;
    });
  }
  return routeOps;
}

void BasicBlockOpAssignment::updateSchedulePriority(
    int timeSlot, std::map<Block *, SetVector<Value>> liveIns,
    std::map<Block *, SetVector<Value>> liveOuts) {

  auto newSchedulePriority =
      getSchedulePriority(curBlock, liveIns[curBlock], liveOuts[curBlock],
                          timeSlot, schedulePriority);

  // keep the delayed schedule priority
  for (auto [op, priority] : newSchedulePriority) {
    if (schedulePriority.count(op) == 0)
      schedulePriority[op] = priority;
    else {
      auto oldPriority = schedulePriority[op];
      schedulePriority[op] = {std::max(oldPriority.first, priority.first),
                              std::max(oldPriority.second, priority.second)};
    }
  }

  std::string message;
  llvm::raw_string_ostream rso(message);
  for (auto [op, priority] : schedulePriority) {
    rso << "Schedule Priority: " << *op << " [" << priority.first << " "
        << priority.second << "]";
    if (solution.count(op))
      rso << " : " << solution.at(op).time << " " << solution.at(op).pe;
    rso << "\n";
  }
  logMessage(rso.str());
}

void BasicBlockOpAssignment::updateCDFG(Block *scheduleBB,
                                        std::vector<ValuePlacement> initGraph,
                                        std::vector<ValuePlacement> finiGraph) {
  // update the initGraph and finiGraph
  startEmbeddingGraph = initGraph;

  // the register in the finiGraph are considered
  for (auto &place : finiGraph) {
    if (place.regAttr == RegAttr::IN)
      continue;

    if (place.val.getParentBlock() == scheduleBB) {
      // the all user locates in the successors of the scheduleBB, it can be
      // external live
      bool allUserInSucBB =
          llvm::all_of(place.val.getUsers(), [&](Operation *user) {
            return llvm::is_contained(scheduleBB->getSuccessors(),
                                      user->getBlock());
          });
      if (!allUserInSucBB)
        place.regAttr = RegAttr::IN;
    }
  }
  finiEmbeddingGraph = finiGraph;
}

double BasicBlockOpAssignment::stepSA(
    int height, SmallVector<Operation *, 4> &schedulingOps,
    std::map<Operation *, ScheduleUnit> &tmpScheduleResult,
    std::vector<ValuePlacement> &tmpGraph,
    std::map<Operation *, std::vector<placeunit>> &existSpace,
    SetVector<Value> liveOut, std::vector<ValuePlacement> &finiGraph,
    GridAttribute attr, int shuffleOpIdx) {
  // Initial placement
  int suc = placeOperations(height, schedulingOps, tmpScheduleResult, tmpGraph,
                            existSpace, finiGraph, shuffleOpIdx);
  double sucCost =
      getSuccessCost(tmpScheduleResult, schedulePriority, schedulingOps);
  double affinityCost = getSpatialAffinityTotalCost(
      curBlock, tmpScheduleResult, tmpGraph, finiGraph, schedulePriority, attr,
      liveOut, scheduledOps);
  double accessCost = getAccessCost(
      curBlock, tmpGraph, tmpScheduleResult, scheduledOps, liveOut, finiGraph,
      attr, schedulingOps.size(), schedulePriority, height);
  // get the total cost
  double currentCost = sucCost + affinityCost + accessCost;

  logMessage("cost: " + std::to_string(sucCost) + " + " +
             std::to_string(affinityCost) + " + " + std::to_string(accessCost) +
             " = " + std::to_string(currentCost) + "\n");
  return currentCost;
}

LogicalResult BasicBlockOpAssignment::mappingBBdataflowToCGRA(
    std::map<Block *, SetVector<Value>> &liveIns,
    std::map<Block *, SetVector<Value>> &liveOuts, ScheduleStrategy strategy) {

  auto blockIn = liveIns[curBlock];
  auto blockOut = liveOuts[curBlock];
  setUpLiveness(liveIns, liveOuts);

  auto initGraph = startEmbeddingGraph;
  auto finiGraph = finiEmbeddingGraph;
  // init all livein in initGraph
  initEmbeddingGraphWithLiveIn(liveIns, liveOuts, initGraph, builder, attr);

  // get the schedule priority of the operations in the block
  schedulePriority = getSchedulePriority(curBlock, blockIn, blockOut);

  int height = 1;

  int totalOpNum = getNonCstOpSize(curBlock);
  int maxTry = 0;
  auto graphScheduleBefore = initGraph;
  while (scheduledOps.size() < totalOpNum && maxTry < 35) {
    maxTry++;
    logMessage("----height: " + std::to_string(height) + "----\n");

    auto schedulingOps = getScheduleOps(curBlock, height, schedulePriority,
                                        scheduledOps, strategy);
    // log schedulingOps
    std::string message;
    llvm::raw_string_ostream totalRso(message);
    totalRso << "Scheduling Operations at height " << height << ": ";
    for (auto op : schedulingOps) {
      totalRso << *op << "\n";
    }
    logMessage(totalRso.str());

    std::map<Operation *, ScheduleUnit> tmpScheduleResult;
    std::map<Operation *, std::vector<placeunit>> searchSpace;
    auto tmpGraph = graphScheduleBefore;

    // Initial placement
    double currentCost =
        stepSA(height, schedulingOps, tmpScheduleResult, tmpGraph, searchSpace,
               blockOut, finiGraph, attr);
    auto layerScheduleResult = tmpScheduleResult;
    auto graphScheduleAfter = tmpGraph;

    double bestCost = currentCost;

    // simulated annealing to create a loop that get random
    // placement, record status, cost and determine the final placement
    int iterSA = 200;
    std::vector<double> lastThreeCosts;
    double previousCost = bestCost;

    for (int iter = 0; iter < iterSA; iter++) {
      // get a random placement
      tmpGraph = graphScheduleBefore;
      int shuffleOpIdx =
          shuffleSearchSpace(searchSpace, tmpScheduleResult, schedulingOps);
      if (shuffleOpIdx < 0) {
        break;
      }
      logMessage("----SA step: " + std::to_string(iter) + "----\n");

      double currentCost =
          stepSA(height, schedulingOps, tmpScheduleResult, tmpGraph,
                 searchSpace, blockOut, finiGraph, attr, shuffleOpIdx);

      // Generate a random number between 0 and 1
      double accept = static_cast<double>(rand()) / RAND_MAX;
      if (currentCost < bestCost && accept <= 0.9) {
        bestCost = currentCost;
        graphScheduleAfter = tmpGraph;
        layerScheduleResult = tmpScheduleResult;
      }

      // Track the last three costs
      lastThreeCosts.push_back(currentCost);
      if (lastThreeCosts.size() > 3)
        lastThreeCosts.erase(lastThreeCosts.begin());

      // Check if the last three states are stable
      if (iter > 10 &&
          std::abs(bestCost - std::accumulate(lastThreeCosts.begin(),
                                              lastThreeCosts.end(), 0.0) /
                                  lastThreeCosts.size()) < 1e-3) {
        break; // Step out if the state is stable
      }

      previousCost = bestCost;
    }
    logMessage("Best cost: " + std::to_string(bestCost) + "\n");

    // post simulated annealing, check whether graph transformation is needed
    SmallVector<Operation *, 4> graphTransformedOps;
    for (auto op : schedulingOps) {
      if (layerScheduleResult.count(op) == 0 &&
          schedulePriority.at(op).second <= height)
        graphTransformedOps.push_back(op);
    }
    if (!graphTransformedOps.empty()) {
      bool transformed = false;
      for (auto op : graphTransformedOps) {
        logMessage("Try to solve the graph transformation");
        unsigned producerNum = op->getNumOperands();
        std::vector<ValuePlacement> producers;
        std::vector<unsigned> movs;
        // detect whether the operation is routable
        bool routable =
            createRoutePath(op, producers, movs, graphScheduleBefore, finiGraph,
                            graphTransformedOps);
        logMessage("routable: " + std::to_string(routable) + "\n");
        if (routable >= 0) {
          std::string message;
          llvm::raw_string_ostream rso(message);
          if (routable == 0 && !isRouteOp(op)) {
            logMessage("Route producers\n");
            unsigned newRouteOpsNum = 0;
            auto newRouteOps = routeOperation(producers, movs, op);
            newRouteOpsNum = newRouteOps.size();
            rso << "Warning: Route for: " << *op << "\n";
            transformed = true;
            // if the op is already a route operation, but not scheduled, don't
            // create new route ops
            updateSchedulePriority(height, liveIns, liveOuts);
            for (size_t i = 0; i < producers.size(); ++i) {
              auto producer = producers[i];
              auto movNum = movs[i];
              rso << "    route: " << producer.val << " :" << movNum << "\n";
            }
            totalOpNum += newRouteOpsNum;
            logMessage(rso.str());
          }
          // otherwise, route the operation itself
          if (routable == 1) {
            logMessage("Route to consumer\n");
            if (liveout.count(op->getResult(0)) > 0) {
              auto newLiveOut =
                  createAtomicMovOp(op->getResult(0), false, false);
              // replace the use if it is used outside the block
              op->getResult(0).replaceUsesWithIf(
                  newLiveOut->getResult(0), [&](OpOperand &use) {
                    auto owner = use.getOwner();
                    return owner->getBlock() != curBlock ||
                           (owner->getBlock() == curBlock &&
                            (isa<cf::BranchOp>(owner) ||
                             (isa<cgra::ConditionalBranchOp>(owner) &&
                              use.getOperandNumber() > 1)));
                  });
              // replace the value in finiGraph and liveout
              for (auto &place : finiGraph) {
                if (op->getResult(0) == place.val) {
                  place.val = newLiveOut->getResult(0);
                  // place.regAttr = RegAttr::EX;
                  // place.pe = attr.maxReg; // use the maxReg to indicate EX
                }
              }
              liveout.remove(op->getResult(0));
              liveout.insert(newLiveOut->getResult(0));
              auto &parentRegion = *(curBlock->getParent());
              computeLiveValue(parentRegion, liveIns, liveOuts);
              transformed = true;
              updateSchedulePriority(height, liveIns, liveOuts);
              totalOpNum++;
            }
          }
        } else {
          std::string message;
          llvm::raw_string_ostream rso(message);
          // check whether pop the blocked PE can solve the problem
          rso << "Warning: Cannot route for: " << *op
              << ", the graph transformation is needed.\n";
          logMessage(rso.str());

          SetVector<unsigned> avaiPEs;
          for (auto i = 0; i < attr.nRow * attr.nCol; i++) {
            auto occupied = getOccupiedValue(graphScheduleBefore, finiGraph,
                                             scheduledOps, curBlock, i);
            if (!occupied)
              avaiPEs.insert(i);
          }
          bool success = true;
          unsigned newRouteOpsNum = 0;
          for (auto blockProd : blockedProdPEs) {
            // try to route the occupied value to other PEs;
            // if it is not routable, return false to indicate it is
            // impossible to route for the failOp
            auto cntPEs = getTorusRoutingPEs(blockProd, attr, false);
            // if none of the cntPEs in the avaiPEs return false
            SetVector<unsigned> cntPESet(cntPEs.begin(), cntPEs.end());
            auto intersection = getInterSection<unsigned>(cntPESet, avaiPEs);
            if (intersection.empty())
              success = false;

            // otherwise, route the blocked PE
            auto blockPlace =
                getOccupiedValue(graphScheduleBefore, finiGraph, scheduledOps,
                                 curBlock, blockProd);

            // not create double route
            if (blockPlace->val.getDefiningOp() &&
                isRouteOp(blockPlace->val.getDefiningOp())) {
              continue;
            }

            transformed = true;
            // log the block Place
            std::string message;
            llvm::raw_string_ostream rso1(message);
            rso << "Blocked PE: " << blockPlace->val << " ^^^ "
                << blockPlace->pe << "\n";
            logMessage(rso1.str());
            // builder.setInsertionPoint
            auto freePEOp = createAtomicMovOp(blockPlace->val, false, false);
            newRouteOpsNum++;
            // replace the use of blockPlace->val with freePEOp
            blockPlace->val.replaceUsesWithIf(
                freePEOp->getResult(0), [&](OpOperand &use) {
                  auto owner = use.getOwner();
                  return owner->getBlock() == curBlock && owner != freePEOp &&
                         !scheduledOps.count(owner) &&
                         !(isa<cf::BranchOp>(owner) ||
                           (isa<cgra::ConditionalBranchOp>(owner) &&
                            use.getOperandNumber() > 1));
                });
          }
          if (!success) {
            rso << "Error: Failed to route the blocked PEs, graph "
                   "transformation is needed.\n";
            // TODO[@Y]: split the graph via DFG
            return failure();
          }
          updateSchedulePriority(height, liveIns, liveOuts);
          totalOpNum += newRouteOpsNum;
        }
      }
      // logMessage(rso.str());

      // re-schedule
      // TODO[@YY]: official rollback
      if (transformed)
        continue;
    }

    // prepare scheduling for the next layer
    for (auto [op, res] : layerScheduleResult) {
      // remove it from the schedulingOps
      auto it = std::find(schedulingOps.begin(), schedulingOps.end(), op);
      schedulingOps.erase(it);
      scheduledOps.insert(op);
      solution[op] = res;

      // log the final placement info
      std::string message;
      llvm::raw_string_ostream rso(message);
      rso << "SCHEDULE RESULT: " << *op << " ---> " << std::to_string(res.pe)
          << "\n";
      logMessage(rso.str());
    }
    graphScheduleBefore = graphScheduleAfter;

    for (auto op : schedulingOps) {
      // delay the scheduling
      schedulePriority[op].first += 1;
      std::string message;
      llvm::raw_string_ostream rso(message);
      rso << "DELAY SCHEDULE: " << *op << "\n";
      // SetVector<Operation *> delayedOps;
      // if (op->getNumResults() == 0)
      //   continue;
      // buildChildTree(op->getResult(0), delayedOps, curBlock);
      // for (auto child : delayedOps) {
      //   if (schedulePriority.count(child))
      //     schedulePriority[child].first += 1;
      // }
    }
    updateSchedulePriority(height, liveIns, liveOuts);
    height++;
  }

  finiGraph = graphScheduleBefore;

  if (scheduledOps.size() < totalOpNum)
    return failure();

  updateCDFG(curBlock, initGraph, finiGraph);
  return success();
}
