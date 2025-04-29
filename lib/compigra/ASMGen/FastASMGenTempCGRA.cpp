//===- FastASMGenOpenEdge.cpp - Implements the functions for temporal CGRA ASM
// fast generation *- C++ -*-----------------------------------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements assembly generation functions for OpenEdge.
//
//===----------------------------------------------------------------------===//

#include "compigra/ASMGen/FastASMGenTempCGRA.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraOps.h"
#include "compigra/Scheduler/BasicBlockOpAssignment.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <fstream>
#include <set>

using namespace mlir;
using namespace compigra;

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

void printBlockLiveValue(Region &region,
                         std::map<Block *, SetVector<Value>> &liveIns,
                         std::map<Block *, SetVector<Value>> &liveOuts) {

  unsigned blockNum = 0;
  // print liveIn and liveOut
  for (auto &block : region) {
    llvm::errs() << "Block: " << blockNum << "\n";
    llvm::errs() << "LiveIn: ";
    for (auto val : liveIns[&block]) {
      if (val.isa<BlockArgument>()) {
        for (auto [ind, bb] : llvm::enumerate(region))
          if (&bb == val.getParentBlock()) {
            llvm::errs() << ind << " ";
            break;
          }
      }
      llvm::errs() << val << "\n";
    }
    llvm::errs() << "LiveOut: ";
    for (auto val : liveOuts[&block]) {
      if (val.isa<BlockArgument>()) {
        for (auto [ind, bb] : llvm::enumerate(region))
          if (&bb == val.getParentBlock()) {
            llvm::errs() << ind << " ";
            break;
          }
      }

      llvm::errs() << val << "\n";
    }
    llvm::errs() << "\n";
    blockNum++;
  }
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

void maxIndependentSubGraphs(Block *block, SetVector<Value> liveIn) {}

namespace {
struct FastASMGenTemporalCGRAPass
    : public compigra::impl::FastASMGenTemporalCGRABase<
          FastASMGenTemporalCGRAPass> {

  explicit FastASMGenTemporalCGRAPass(int nRow, int nCol, int mem,
                                      StringRef msOpt, StringRef asmOutDir) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    auto funcOp = *modOp.getOps<func::FuncOp>().begin();
    if (asmOutDir.empty())
      asmOutDir = "out";
    std::string outDir = asmOutDir;

    Region &region = funcOp.getBody();

    std::map<Block *, SetVector<Value>> liveIns;
    std::map<Block *, SetVector<Value>> liveOuts;
    computeLiveValue(region, liveIns, liveOuts);
    printBlockLiveValue(region, liveIns, liveOuts);
    std::map<Operation *, ScheduleUnit> solution;

    GridAttribute attr{(unsigned)nRow, (unsigned)nCol, 3};

    std::map<Block *, std::vector<ValuePlacement>> initEmbeddingGraphs;
    std::map<Block *, std::vector<ValuePlacement>> finiEmbeddingGraphs;

    int bbId = 0;
    for (auto &bb : region.getBlocks()) {
      // create the schedule for each block
      std::map<Operation *, ScheduleUnit> subSolution;
      auto initGraph =
          initEmbeddingGraphs.find(&bb) != initEmbeddingGraphs.end()
              ? initEmbeddingGraphs[&bb]
              : std::vector<ValuePlacement>{};
      auto finiGraph =
          finiEmbeddingGraphs.find(&bb) != finiEmbeddingGraphs.end()
              ? finiEmbeddingGraphs[&bb]
              : std::vector<ValuePlacement>{};
      mappingBBdataflowToCGRA(&bb, liveIns, liveOuts, subSolution, initGraph,
                              finiGraph, attr);
      // update the initGraph and finiGraph
      initEmbeddingGraphs[&bb] = initGraph;
      finiEmbeddingGraphs[&bb] = finiGraph;
      llvm::errs() << "==============================\n\n";
      // add the subSolution to the global solution
      for (auto [op, unit] : subSolution) {
        if (solution.count(op)) {
          llvm::errs() << "Error: Operation " << op
                       << " is already in the "
                          "solution\n";
          return signalPassFailure();
        }
        solution[op] = unit;
      }
      // if (bbId == 1)
      //   break;
      bbId++;
    }
  };
};
} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createFastASMGenTemporalCGRA(int nRow, int nCol,
                                                         int mem,
                                                         StringRef msOpt,
                                                         StringRef asmOutDir) {
  return std::make_unique<FastASMGenTemporalCGRAPass>(nRow, nCol, mem, msOpt,
                                                      asmOutDir);
}
} // namespace compigra