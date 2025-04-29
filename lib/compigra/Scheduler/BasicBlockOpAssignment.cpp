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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <queue>

using namespace mlir;
using namespace compigra;

template <typename T>
static void getDiffSet(SetVector<T> &vec1, SetVector<T> &vec2) {
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

static SmallVector<Operation *, 4>
getNextLayerOps(Block *block, SetVector<Value> &liveIn,
                std::set<Operation *> &scheduledOps) {
  SmallVector<Operation *, 4> schedulingOps;
  auto termOp = block->getTerminator();

  for (auto &op : block->getOperations()) {
    if (isa<arith::ConstantOp>(op) || scheduledOps.count(&op) ||
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
          std::find(scheduledOps.begin(), scheduledOps.end(),
                    operand.getDefiningOp()) != scheduledOps.end();
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

static bool isLive(Value val, SetVector<Value> liveOut,
                   SetVector<Operation *> scheduledOps) {
  // check whether the value is used by the scheduledOp
  return liveOut.contains(val) ||
         llvm::any_of(val.getUsers(), [&](Operation *user) {
           return !scheduledOps.contains(user);
         });
}

static bool isLiveExcept(Value val, Operation *user, SetVector<Value> liveOut,
                         SetVector<Operation *> scheduledOps) {
  // check whether the value is used by the scheduledOp
  scheduledOps.insert(user);
  return isLive(val, liveOut, scheduledOps);
}

static SmallVector<Operation *, 4>
getPreviousLayerOps(Block *block, SetVector<Value> &liveout,
                    std::set<Operation *> &ancestors) {
  SmallVector<Operation *, 4> schedulingOps;

  for (auto op : llvm::make_early_inc_range(ancestors)) {
    // swi op and terminator op can be scheduled at the last stage
    if (op->getResults().empty()) {
      schedulingOps.push_back(op);
      continue;
    }

    // if the operation produce live-out or its produced value is not used by
    // the ancestors, add it to the scheduling operations
    auto result = op->getResult(0);
    bool canSchedule =
        std::find(liveout.begin(), liveout.end(), result) != liveout.end();

    // any users does not belong to the ancestors
    for (auto user : result.getUsers())
      if (ancestors.count(user) == 0) {
        canSchedule = true;
        break;
      }

    if (canSchedule) {
      schedulingOps.push_back(op);
    }
  }
  return schedulingOps;
}

/// Compute the schedule priority of the operations in the block. The earliest
/// schedule time is traversed through the block liveIn, and the latest
/// schedule time is traversed through the block liveOut. This function
/// returns a map of the operations and their schedule priority [earliest,
/// latest].
static std::map<Operation *, std::pair<int, int>>
getSchedulePriority(Block *block, SetVector<Value> &liveIn,
                    SetVector<Value> &liveOut) {
  std::map<Operation *, std::pair<int, int>> schedulePriority;

  std::set<Operation *> scheduledOps;
  // the live-in height is zero
  int earliest = 0;
  // scheduledOp size = block->getOperations().size() - constantOp size - 1
  int scheduledOpSize = getNonCstOpSize(block) - 1;

  while (scheduledOps.size() < scheduledOpSize) {
    earliest++;
    // get all operations that its operands are in the liveIn set or belong to
    // the scheduledOps
    SmallVector<Operation *, 4> schedulingOps =
        getNextLayerOps(block, liveIn, scheduledOps);

    for (auto op : schedulingOps) {
      schedulePriority[op] = {earliest, INT_MAX};
      scheduledOps.insert(op);
    }
  }

  int latest = earliest;
  while (!scheduledOps.empty()) {
    // get all operations that its operands are in the liveOut set or belong
    // to the scheduledOps
    SmallVector<Operation *, 4> schedulingOps =
        getPreviousLayerOps(block, liveOut, scheduledOps);

    for (auto op : schedulingOps) {
      schedulePriority[op].second = latest;
      scheduledOps.erase(op);
    }
    latest--;
  }
  return schedulePriority;
}

static SmallVector<Operation *, 4>
getScheduleOps(Block *block, int height,
               std::map<Operation *, std::pair<int, int>> schedulePriority,
               SetVector<Operation *> scheduledOps,
               ScheduleStrategy strategy = ScheduleStrategy::ASAP) {
  SmallVector<Operation *, 4> schedulingOps;
  for (auto [op, priority] : schedulePriority) {
    if (scheduledOps.count(op))
      continue;

    // if the operation is in the scheduling height, add it to the scheduling
    // operations
    if (strategy == ScheduleStrategy::ASAP && priority.first == height)
      schedulingOps.push_back(op);

    if (strategy == ScheduleStrategy::ALAP && priority.second == height)
      schedulingOps.push_back(op);

    if (strategy == ScheduleStrategy::DYNAMIC && priority.first <= height &&
        priority.second >= height) {
      schedulingOps.push_back(op);
    }
  }

  // sort the scheduling operations according to the schedule priority
  std::sort(
      schedulingOps.begin(), schedulingOps.end(),
      [&](Operation *op1, Operation *op2) {
        return (schedulePriority[op1].first + schedulePriority[op1].second) <
               (schedulePriority[op2].first + schedulePriority[op2].second);
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

  return dx + dy;
}

static double
getSpatialAffinityCost(Value val1, ScheduleUnit unit1, Value val2,
                       ScheduleUnit unit2,
                       std::map<Operation *, std::pair<int, int>> heightMap,
                       GridAttribute grid, Block *blk) {

  auto pe1 = unit1.pe;
  auto pe2 = unit2.pe;

  int maxHeight = 0;
  for (const auto &entry : heightMap)
    maxHeight = std::max(maxHeight, entry.second.second);

  // Builds a child tree of operations starting from a given value.
  //
  // This function traverses the users of a given value (`val`) and constructs a
  // tree of operations (`childTree`) that are reachable through the user chain
  // within the same block (`blk`).
  auto buildChildTree = [](Value val, SetVector<Operation *> &childTree,
                           Block *blk) {
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
  };

  SetVector<Operation *> childTree1;
  SetVector<Operation *> childTree2;
  buildChildTree(val1, childTree1, blk);

  if (!childTree1.contains(val2.getDefiningOp()))
    buildChildTree(val2, childTree2, blk);

  auto children = getInterSection<Operation *>(childTree1, childTree2);

  double affinity = 0;
  double maxRouting = grid.nRow / 2 + grid.nCol / 2;

  for (auto child : children) {
    auto childHeight = heightMap[child].first;
    auto val1Height =
        val1.getDefiningOp() ? heightMap[val1.getDefiningOp()].first : 0;
    auto val2Height =
        val2.getDefiningOp() ? heightMap[val2.getDefiningOp()].first : 0;

    double depth1 = childHeight - val1Height;
    double depth2 = childHeight - val2Height;
    double depth = (depth1 + depth2) / 2;

    // get the distance between the two PEs
    // if (depth < maxRouting)
    affinity += std::pow(2, -(depth1 + depth2) / 2);
  }

  // if (!children.empty()) {
  //   llvm::errs() << "AFF [" << val1 << " && " << val2 << "]:";
  //   llvm::errs() << "  "
  //                << affinity * getDistance(pe1, pe2, grid.nRow, grid.nCol) /
  //                       maxRouting
  //                << "\n";
  // }

  return affinity * getDistance(pe1, pe2, grid.nRow, grid.nCol) / maxRouting;
}

static double getSpatialAffinityTotalCost(
    std::map<Operation *, ScheduleUnit> scheduleResult,
    std::vector<ValuePlacement> initGraph,
    std::map<Operation *, std::pair<int, int>> heightMap, GridAttribute grid) {
  double cost = 0;
  int count = 0;
  for (auto place : initGraph)
    for (auto op : scheduleResult) {
      if (op.first->getNumResults() == 0)
        continue;
      cost += getSpatialAffinityCost(place.val, {0, (int)place.pe, -1},
                                     op.first->getResult(0), op.second,
                                     heightMap, grid, op.first->getBlock());
      count++;
    }

  for (auto op1 = scheduleResult.begin();
       op1 != std::prev(scheduleResult.end()); op1++) {
    // if op does not have a result, skip it
    if (op1->first->getNumResults() == 0)
      continue;
    for (auto op2 = std::next(op1); op2 != scheduleResult.end(); op2++) {
      if (op2->first->getNumResults() == 0)
        continue;
      cost += getSpatialAffinityCost(op1->first->getResult(0), op1->second,
                                     op2->first->getResult(0), op2->second,
                                     heightMap, grid, op1->first->getBlock());
      count++;
    }
  }

  return cost / count;
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

static void updateResourceGraph(std::vector<ValuePlacement> &curGraph,
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
      continue;
    }
    if (it->regAttr == RegAttr::IN) {
      // check whether the value should be kept anymore
      auto val = it->val;
      if (!isLive(val, liveout, scheduledOps))
        it = curGraph.erase(it);
      else
        ++it;
    }
  }

  // add operation to curGraph
  if (op->getNumResults() > 0)
    curGraph.push_back({op->getResult(0), pe, reg});
}

static std::map<int, PERegUse> getAvailableResourceGraph(
    std::vector<ValuePlacement> curGraph, GridAttribute &attr, Operation *op,
    SetVector<Value> liveout, SetVector<Operation *> scheduledOps) {
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
      freeReg[pe].exAvail = !isLiveExcept(place.val, op, liveout, scheduledOps);
  }
  return freeReg;
}

static void printResourceGraph(std::map<int, PERegUse> freeReg,
                               GridAttribute attr) {
  // print nRow and nCol
  for (int i = 0; i < attr.nCol; i++) {
    for (int j = 0; j < attr.nRow; j++) {
      auto pe = i * attr.nRow + j;
      auto availPE = freeReg[pe].exAvail;
      auto availRegs = freeReg[pe].inNum;
      if (availPE)
        llvm::errs() << "[? | " << availRegs << "] ";
      else
        llvm::errs() << "[X | " << availRegs << "] ";
    }
    llvm::errs() << "\n";
  }
}

static int getInitialLiveInPlacement(
    Value in, Block *block, std::vector<ValuePlacement> &initGraph,
    std::map<Block *, SetVector<Value>> &liveIns,
    std::map<Block *, SetVector<Value>> &liveOuts, GridAttribute &attr) {
  // if the value has more than 3 users, it should be external register
  int userNum = 0;
  for (auto user : in.getUsers())
    if (user->getBlock() == block)
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
  llvm::errs() << "INIT GRAPH: " << in << " " << pe << " " << regAttr << "\n";
}

/// Initialize the embedding graph, where the key is the [time slot, PE], the
/// value indicates the value placed in the graph and its register attribute.
/// It is noticed that graph[int,int] = <nullptr, nullptr> which indicates the
/// PE is occupied by an operation does not produce any value at the time slot.
static void initEmbeddingGraph(Block *blk,
                               std::map<Block *, SetVector<Value>> liveIns,
                               std::map<Block *, SetVector<Value>> liveOuts,
                               std::vector<ValuePlacement> &initGraph,
                               GridAttribute attr) {

  auto liveIn = liveIns[blk];
  for (auto val : liveIn) {
    auto it = std::find_if(initGraph.begin(), initGraph.end(),
                           [&](ValuePlacement p) { return p.val == val; });
    // randomly assign the PE to the liveIn value
    if (it == initGraph.end()) {
      getInitialLiveInPlacement(val, blk, initGraph, liveIns, liveOuts, attr);
    }
  }
  // if val not in initGraph, assign the liveIn value with the lowest cost
}

// get the torus routing PEs
static std::vector<unsigned> getTorusRoutingPEs(unsigned pe,
                                                GridAttribute &attr) {
  std::vector<unsigned> routingPEs{pe};
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

/// Get the valid placement space of the operation.
SetVector<unsigned> searchOpPlacementSpace(
    Operation *op, std::vector<ValuePlacement> &curGraph,
    std::vector<ValuePlacement> &finiGraph, SetVector<Value> liveout,
    SetVector<Operation *> scheduledOps, GridAttribute &attr) {
  int nRow = attr.nRow;
  int nCol = attr.nCol;
  SetVector<unsigned> placementSpace;

  // get the available hardware resources
  auto regUse =
      getAvailableResourceGraph(curGraph, attr, op, liveout, scheduledOps);
  printResourceGraph(regUse, attr);

  for (int i = 0; i < nRow * nCol; i++) {
    if (regUse[i].inNum > 0 && regUse[i].exAvail)
      placementSpace.insert(i);
  }

  // limit the placement space according to its consumer and producer
  for (auto opr : op->getOperands()) {
    SetVector<unsigned> routingPEs;
    auto oprPlace = getSrcValuePlacement(opr, op, curGraph);
    // not find existing value, no need to limit the placement space
    if (oprPlace.val == nullptr)
      continue;

    auto pe = oprPlace.pe;
    auto regAttr = oprPlace.regAttr;
    if (regAttr == RegAttr::IN)
      routingPEs.insert(pe);
    else if (regAttr == RegAttr::EX || regAttr == RegAttr::IE) {
      for (auto routingPE : getTorusRoutingPEs(pe, attr))
        routingPEs.insert(routingPE);
    }
    // placementSpace intersect routingPEs
    getDiffSet<unsigned>(placementSpace, routingPEs);
  }

  if (placementSpace.empty()) {
    llvm::errs() << "SRC Error: No available placement space for the operation "
                 << *op << "\n";
    return placementSpace;
  }

  // if the operation result is restricted by the finiGraph, it is restricted
  // to the PE that is used by the finiGraph
  for (auto res : op->getResults()) {
    auto it = std::find_if(finiGraph.begin(), finiGraph.end(),
                           [&](ValuePlacement p) { return p.val == res; });
    if (it != finiGraph.end()) {
      auto pe = it->pe;
      SetVector<unsigned> tempSet;
      tempSet.insert(it->pe);
      getDiffSet<unsigned>(placementSpace, tempSet);
    }
  }
  // print the placement space
  llvm::errs() << "[ ";
  for (auto pe : placementSpace)
    llvm::errs() << pe << " ";
  llvm::errs() << "]\n";

  return placementSpace;
}

/// Primary cost function for operation placement: No. operation can be placed
/// in the same time slot.
static int getSuccessRatio(std::map<Operation *, ScheduleUnit> &scheduleResult,
                           Operation *op, int time) {
  int cost = 0;
  // TODO
  return cost;
}

int placeOperations(int timeSlot, SmallVector<Operation *, 4> &schedulingOps,
                    std::map<Operation *, ScheduleUnit> &scheduleResult,
                    SetVector<Value> &liveIn, SetVector<Value> &liveOut,
                    std::vector<ValuePlacement> &curGraph,
                    std::vector<ValuePlacement> &finiGraph,
                    SetVector<Operation *> &scheduledOps, GridAttribute &attr) {

  int maxTry = 0;
  unsigned suc = 0;
  llvm::errs() << "timeSlot: " << timeSlot << " " << schedulingOps.size()
               << "\n";
  std::map<Operation *, ScheduleUnit> tmpResult;
  SetVector<Operation *> tmpScheduledOps;
  for (auto op : schedulingOps) {
    // get search space
    auto placementSpace = searchOpPlacementSpace(
        op, curGraph, finiGraph, liveOut, tmpScheduledOps, attr);
    // randomly choose a placement space
    if (placementSpace.empty())
      continue;

    unsigned pe = *(placementSpace.begin() + rand() % placementSpace.size());
    // TODO: determine the register attribute
    RegAttr regAttr = RegAttr::IE;
    suc++;
    updateResourceGraph(curGraph, op, pe, regAttr, tmpScheduledOps, liveOut);
    tmpScheduledOps.insert(op);
    tmpResult[op] = {timeSlot, (int)pe, -1};
  }
  // calculate affinity cost according to tmpResult

  // erased from schedulingOps
  for (auto op : tmpScheduledOps) {
    // remove it from the schedulingOps
    auto it = std::find(schedulingOps.begin(), schedulingOps.end(), op);
    schedulingOps.erase(it);
    scheduledOps.insert(op);
  }
  // write tmpScheduleResult to scheduleResult
  for (auto [op, unit] : tmpResult) {
    scheduleResult[op] = tmpResult[op];
  }
  return suc;
}

void compigra::mappingBBdataflowToCGRA(
    Block *block, std::map<Block *, SetVector<Value>> &liveIns,
    std::map<Block *, SetVector<Value>> &liveOuts,
    std::map<Operation *, ScheduleUnit> &scheduleResult,
    std::vector<ValuePlacement> &initGraph,
    std::vector<ValuePlacement> &finiGraph, GridAttribute &attr,
    ScheduleStrategy strategy) {

  auto blockIn = liveIns[block];
  auto blockOut = liveOuts[block];
  // init all livein in initGraph
  initEmbeddingGraph(block, liveIns, liveOuts, initGraph, attr);

  // get the schedule priority of the operations in the block
  std::map<Operation *, std::pair<int, int>> schedulePriority =
      getSchedulePriority(block, blockIn, blockOut);

  int height = 1;
  SetVector<Operation *> scheduledOps;
  int scheduledOpSize = getNonCstOpSize(block) - 1; // -1 for the terminator op
  int maxTry = 0;
  auto scheduleGraph = initGraph;
  while (scheduledOps.size() < scheduledOpSize && maxTry < 100) {
    maxTry++;
    std::map<Operation *, ScheduleUnit> tmpScheduleResult;
    auto schedulingOps =
        getScheduleOps(block, height, schedulePriority, scheduledOps, strategy);
    int totalOpNum = schedulingOps.size();

    auto tmpGraph = scheduleGraph;
    // TODO[@30th/April]: simulated annealing to create a loop that get random
    // placement, record status, cost and determine the final placement
    int suc =
        placeOperations(height, schedulingOps, tmpScheduleResult, blockIn,
                        blockOut, tmpGraph, finiGraph, scheduledOps, attr);
    auto cost = (double)(1 - suc) / totalOpNum +
                getSpatialAffinityTotalCost(tmpScheduleResult, initGraph,
                                            schedulePriority, attr);

    scheduleGraph = tmpGraph;
    for (auto op : schedulingOps) {
      // delay the scheduling if schedulePriority[op].second > height
      if (schedulePriority[op].second > height)
        schedulePriority[op].first = height + 1;
      else
        llvm::errs() << "ERROR: " << *op << " must be scheduled now\n";
    }
    height++;
    // TODO: what if ops in the same height are not scheduled?
  }

  // for (auto [op, priority] : schedulePriority) {
  //   llvm::errs() << "Schedule priority: " << *op << " [" << priority.first
  //                << ", " << priority.second << "]\n";
  // }
}
