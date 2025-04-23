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

using namespace mlir;
using namespace compigra;

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
  llvm::errs() << "scheduledOpSize: " << scheduledOpSize << "\n";

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

static SetVector<Operation *>
getScheduleOps(Block *block, int height,
               std::map<Operation *, std::pair<int, int>> schedulePriority,
               SetVector<Operation *> &scheduledOps,
               ScheduleStrategy strategy = ScheduleStrategy::ASAP) {
  SetVector<Operation *> schedulingOps;
  for (auto [op, priority] : schedulePriority) {
    if (scheduledOps.count(op))
      continue;

    // if the operation is in the scheduling height, add it to the scheduling
    // operations
    if (strategy == ScheduleStrategy::ASAP && priority.first == height)
      schedulingOps.insert(op);

    if (strategy == ScheduleStrategy::ALAP && priority.second == height)
      schedulingOps.insert(op);

    if (strategy == ScheduleStrategy::DYNAMIC && priority.first <= height &&
        priority.second >= height) {
      schedulingOps.insert(op);
    }
  }
  return schedulingOps;
}

void placeOperations(SetVector<Operation *> &schedulingOps,
                     std::map<Operation *, ScheduleUnit> &scheduleResult,
                     std::vector<std::pair<Value, ScheduleUnit>> &solution,
                     SetVector<Operation *> &scheduledOps) {
  // put scheduled operations to scheduledOps
  for (auto op : schedulingOps) {
    scheduledOps.insert(op);
  }
}

void compigra::mappingBBdataflowToCGRA(
    Block *block, SetVector<Value> &liveIn, SetVector<Value> &liveOut,
    std::map<Operation *, ScheduleUnit> &scheduleResult,
    std::vector<std::pair<Value, ScheduleUnit>> &solution, CGRAAttribute &attr,
    ScheduleStrategy strategy) {
  // get the schedule priority of the operations in the block
  std::map<Operation *, std::pair<int, int>> schedulePriority =
      getSchedulePriority(block, liveIn, liveOut);

  int height = 1;
  SetVector<Operation *> scheduledOps;
  int scheduledOpSize = getNonCstOpSize(block) - 1; // -1 for the terminator op
  int maxTry = 0;
  while (scheduledOps.size() < scheduledOpSize && maxTry < 100) {
    maxTry++;
    auto schedulingOps =
        getScheduleOps(block, height, schedulePriority, scheduledOps);
  }

  // for (auto [op, priority] : schedulePriority) {
  //   llvm::errs() << "Schedule priority: " << *op << " [" << priority.first
  //                << ", " << priority.second << "]\n";
  // }
}
