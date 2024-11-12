//===- TemporalCGRAScheduler.h - Declares the class/functions of basic block
// ILP model to schedule the executions of operations*- C++-* -------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares class for BasicBlockILPModel functions.
//
//===----------------------------------------------------------------------===//

#ifndef BASIC_BLOCK_ILP_MODEL_H
#define BASIC_BLOCK_ILP_MODEL_H

#include "compigra/Scheduler/KernelSchedule.h"
#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

using namespace mlir;

namespace compigra {
/// Class for basic block ILP scheduler. BasicBlockILPModel builds an ILP
/// model to schedule the operation execution time and PE assignment for each
/// operation in the basic block.
struct ScheduleUnitBB {
  int time;
  int pe;
};

// Vector to store the live value and its corresponding PE.
using liveVec = std::vector<std::pair<Value, unsigned>>;

bool isPhiRelatedValue(Value val);

/// Strategy to handle BasicBlockILPModel failure, if Abort, abort the
/// schedule, if Mov, insert an add operation which can extend the value
/// propagation range, if Split, split the producer and consumer operations with
/// lwi/swi operations.
enum class FailureStrategy { Abort, Mov, Split };

/// BasicBlockILPModel class to build an ILP model to schedule the
/// placement and execution time for each operation in the basic block.
class BasicBlockILPModel : public CGRAKernelScheduler<ScheduleUnitBB> {
public:
  BasicBlockILPModel(unsigned maxReg, unsigned nRow, unsigned nCol,
                         Block *block, unsigned bbId, OpBuilder builder)
      : CGRAKernelScheduler(maxReg, nRow, nCol), block(block), bbId(bbId),
        builder(builder) {}

  LogicalResult createSchedulerAndSolve() override;

  std::map<Operation *, ScheduleUnitBB> getSolution() { return solution; }

  void setLiveValue(SetVector<Value> liveIn, SetVector<Value> liveOut);
  void setStoreAddr(unsigned addr) { storeAddr = addr; }

  // void removeSpillOut(Value val) { liveOut.remove(val); }

  liveVec getExternalLiveOutResult() { return liveOutExter; }
  liveVec getInternalLiveOutResult() { return liveOutInter; }

  void setLiveInPrerequisite(const liveVec liveInExter,
                             const liveVec liveInInter) {
    // print liveInExter and liveInInter
    for (auto [val, ind] : liveInExter) {
      llvm::errs() << "LiveInExter: " << val << " " << ind << "\n";
    }
    for (auto [val, ind] : liveInInter) {
      llvm::errs() << "LiveInInter: " << val << " " << ind << "\n";
    }
    this->liveInInter = liveInInter;
    this->liveInExter = liveInExter;
  }
  void setFailureStrategy(FailureStrategy strategy) {
    this->strategy = strategy;
  }

  FailureStrategy getFailureStrategy() { return strategy; }
  Value getSpillVal() { return spill; }
  Operation *getFailUser() { return failUser; }

private:
  // Interface for the the global schduler if the ILP model does not have
  // solution
  unsigned storeAddr;
  FailureStrategy strategy = FailureStrategy::Abort;
  // Value spillIn = nullptr;
  Value spill = nullptr;
  Operation *failUser = nullptr;
  OpBuilder builder;

  Block *block;
  unsigned bbId;

  // SetVector<Value> liveIn;
  // SetVector<Value> liveOut;

  // Hash map to store the live value for each PE
  liveVec liveInInter;
  liveVec liveInExter;
  liveVec liveOutInter;
  liveVec liveOutExter;

#ifdef HAVE_GUROBI
  /// Initialize the mapping variables for the block operations.
  LogicalResult initVariablesForBlock(GRBModel &model,
                                      std::map<Operation *, GRBVar> &opTimeVar,
                                      std::map<Operation *, GRBVar> &opPeVar);

  LogicalResult createLocalDominanceConstraints(
      GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar);

  LogicalResult createLocalLivenessConstraints(
      GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
      const std::map<Operation *, GRBVar> opPeVar,
      const std::map<Operation *, std::string> varName);

  LogicalResult
  createHardwareConstraints(GRBModel &model,
                            const std::map<Operation *, GRBVar> opTimeVar,
                            const std::map<Operation *, GRBVar> opPeVar,
                            const std::map<Operation *, std::string> varName);

  LogicalResult createGlobalLiveInInterConstraints(
      GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
      const std::map<Operation *, GRBVar> opPeVar);

  LogicalResult createGlobalLiveInExterConstraints(
      GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
      const std::map<Operation *, GRBVar> opPeVar);

  LogicalResult createGlobalLiveOutInterConstraints(
      GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
      const std::map<Operation *, GRBVar> opPeVar,
      const std::map<Operation *, std::string> varName);

  LogicalResult createGlobalLiveOutExterConstraints(
      GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
      const std::map<Operation *, GRBVar> opPeVar);

  LogicalResult
  createObjetiveFunction(GRBModel &model,
                         const std::map<Operation *, GRBVar> opTimeVar);

  void writeLiveOutResult(const std::map<Operation *, GRBVar> opPeVar);

  void writeILPResult(const std::map<Operation *, GRBVar> opTimeVar,
                      const std::map<Operation *, GRBVar> opPeVar);
#endif
public:
  std::map<Operation *, std::string> varName;
};
} // namespace compigra
#endif // BASIC_BLOCK_ILP_MODEL_H