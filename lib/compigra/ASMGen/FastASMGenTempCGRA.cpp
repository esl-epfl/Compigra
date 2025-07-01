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
#include "compigra/Support/OpenEdgeASM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <fstream>
#include <set>

using namespace mlir;
using namespace compigra;

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

void printLiveGraph(std::map<Block *, std::vector<ValuePlacement>> graph) {
  std::string message;
  llvm::raw_string_ostream rso(message);
  for (auto [blk, graph] : graph) {
    rso << "------------------\n";
    rso << *blk->getTerminator() << "\n";
    for (auto val : graph) {
      rso << val.val << " [" << val.pe << " " << static_cast<int>(val.regAttr)
          << "]\n";
    }
    rso << "------------------\n";
  }
  logMessage(rso.str());
}

void maxIndependentSubGraphs(Block *block, SetVector<Value> liveIn) {}

arith::ConstantOp getZeroConstant(Region &region, OpBuilder &builder,
                                  bool isFloat = false) {
  arith::ConstantOp zeroOp;
  for (auto &op : region.getOps()) {
    auto zeroCst = dyn_cast_or_null<arith::ConstantOp>(op);
    if (!zeroCst)
      continue;

    if (auto intAttr = zeroCst.getValue().dyn_cast<IntegerAttr>()) {
      if (!isFloat && intAttr.getValue().isZero()) {
        zeroOp = zeroCst;
        break;
      }
    } else if (auto floatAttr = zeroCst.getValue().dyn_cast<FloatAttr>()) {
      if (isFloat && floatAttr.getValue().isZero()) {
        zeroOp = zeroCst;
        break;
      }
    }
  }

  // if zeroOp is not found, create a new one
  if (!zeroOp && !isFloat) {
    zeroOp = builder.create<arith::ConstantOp>(
        region.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
  } else if (!zeroOp && isFloat) {
    zeroOp = builder.create<arith::ConstantOp>(
        region.getLoc(), builder.getF32Type(), builder.getF32FloatAttr(0.0));
  }
  return zeroOp;
}

static Value getArgumentOperand(Operation *termOp, Block *sucBlk,
                                unsigned argInd) {
  if (auto branchOp = dyn_cast_or_null<cf::BranchOp>(termOp)) {
    return branchOp.getOperand(argInd);
  } else if (auto branchOp =
                 dyn_cast_or_null<cgra::ConditionalBranchOp>(termOp)) {
    if (sucBlk == branchOp.getTrueDest()) {
      return branchOp.getOperand(argInd + 2);
    } else if (sucBlk == branchOp.getFalseDest()) {
      return branchOp.getOperand(argInd + 2 +
                                 branchOp.getNumTrueDestOperands());
    }
  }
  return nullptr;
}

Value resolveInjectedValue(
    Value val, Block *curBlk, Block *prevBlk,
    const std::map<Block *, SetVector<Value>> &liveOuts) {
  if (liveOuts.at(prevBlk).count(val))
    return val;
  for (auto arg : curBlk->getArguments()) {
    if (arg == val)
      return getArgumentOperand(prevBlk->getTerminator(), curBlk,
                                arg.getArgNumber());
  }
  llvm::errs() << "Error: cannot resolve injected value " << val << " in "
               << *prevBlk->getTerminator() << "\n";
};

Value resolvePropagatedValue(
    Value val, Block *curBlk, Block *sucBlk,
    const std::map<Block *, SetVector<Value>> &liveIns) {
  if (liveIns.at(sucBlk).count(val))
    return val;

  BlockArgument arg;
  auto termOp = curBlk->getTerminator();
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    if (user != termOp)
      continue;

    auto argInd = use.getOperandNumber();
    if (auto br = dyn_cast<cf::BranchOp>(use.getOwner())) {
      arg = sucBlk->getArgument(argInd);
      break;
    }

    if (auto cbr = dyn_cast<cgra::ConditionalBranchOp>(use.getOwner())) {
      if (sucBlk == cbr.getTrueDest()) {
        if (argInd >= 2 && argInd < 2 + cbr.getNumTrueDestOperands()) {
          arg = sucBlk->getArgument(argInd - 2);
          break;
        }
      } else if (sucBlk == cbr.getFalseDest()) {
        if (argInd >= 2 + cbr.getNumTrueDestOperands()) {
          arg = sucBlk->getArgument(argInd - cbr.getNumTrueDestOperands() - 2);
          break;
        }
      }
    }
  }

  if (liveIns.at(sucBlk).count(arg))
    return arg;
  return NULL;
};

void updateLiveGraph(std::vector<ValuePlacement> &graph,
                     ValuePlacement prequisite) {
  auto it = std::find_if(graph.begin(), graph.end(), [&](ValuePlacement p) {
    return p.val == prequisite.val;
  });
  if (it != graph.end()) {
    it->pe = prequisite.pe;
    it->regAttr = prequisite.regAttr;
  } else {
    graph.push_back(prequisite);
  }
};

void updatePredecessorPlacement(
    ValuePlacement valPlace, Block *curBlk,
    std::map<Block *, SetVector<Value>> &liveIns,
    std::map<Block *, SetVector<Value>> &liveOuts,
    std::map<Block *, std::vector<ValuePlacement>> &bbInitGraphs,
    std::map<Block *, std::vector<ValuePlacement>> &bbFiniGraphs) {

  auto val = valPlace.val;
  DenseSet<Block *> visited;
  visited.insert(curBlk);
  // recursively propagate the value to the successors
  for (auto *pred : curBlk->getPredecessors()) {
    // if (pred == curBlk)
    //   continue;

    visited.insert(pred);
    Value propVal = resolveInjectedValue(val, curBlk, pred, liveOuts);
    updateLiveGraph(bbFiniGraphs[pred],
                    {propVal, valPlace.pe, valPlace.regAttr});

    std::vector<std::pair<Block *, Value>> stack;
    if (liveIns[pred].count(propVal)) {
      // update the initGraph
      updateLiveGraph(bbInitGraphs[pred],
                      {propVal, valPlace.pe, valPlace.regAttr});
      stack.emplace_back(pred, propVal);
    }

    // init an visited set to avoid infinite loop
    while (!stack.empty()) {
      auto [cur, curVal] = stack.back();
      stack.pop_back();
      visited.insert(cur);

      for (auto *prevPred : cur->getPredecessors()) {
        // if (prevPred == cur)
        //   continue;

        Value nextPropVal =
            resolveInjectedValue(curVal, cur, prevPred, liveOuts);
        updateLiveGraph(bbFiniGraphs[prevPred],
                        {nextPropVal, valPlace.pe, valPlace.regAttr});

        if (liveIns[prevPred].count(nextPropVal)) {
          // update the initGraph
          updateLiveGraph(bbInitGraphs[prevPred],
                          {nextPropVal, valPlace.pe, valPlace.regAttr});
          if (!visited.count(prevPred))
            stack.emplace_back(prevPred, nextPropVal);
        }
      }
    }
  }
}

void updateSuccessorPlacement(
    ValuePlacement valPlace, Block *curBlk,
    std::map<Block *, SetVector<Value>> &liveIns,
    std::map<Block *, SetVector<Value>> &liveOuts,
    std::map<Block *, std::vector<ValuePlacement>> &bbInitGraphs,
    std::map<Block *, std::vector<ValuePlacement>> &bbFiniGraphs) {
  auto val = valPlace.val;
  DenseSet<Block *> visited;
  visited.insert(curBlk);

  for (auto *suc : curBlk->getSuccessors()) {
    if (suc == curBlk)
      continue;
    visited.insert(suc);
    Value propVal = resolvePropagatedValue(val, curBlk, suc, liveIns);
    if (propVal == NULL)
      continue;
    if (propVal != val)
      updatePredecessorPlacement({propVal, valPlace.pe, valPlace.regAttr}, suc,
                                 liveIns, liveOuts, bbInitGraphs, bbFiniGraphs);

    updateLiveGraph(bbInitGraphs[suc],
                    {propVal, valPlace.pe, valPlace.regAttr});

    std::vector<std::pair<Block *, Value>> stack;
    if (liveOuts[suc].count(propVal)) {
      // update the finiGraph
      updateLiveGraph(bbFiniGraphs[suc],
                      {propVal, valPlace.pe, valPlace.regAttr});
      stack.emplace_back(suc, propVal);
    }

    while (!stack.empty()) {
      auto [cur, curVal] = stack.back();
      stack.pop_back();
      visited.insert(cur);

      for (auto *nextSuc : cur->getSuccessors()) {
        if (nextSuc == cur)
          continue;

        Value nextPropVal =
            resolvePropagatedValue(curVal, curBlk, nextSuc, liveIns);
        if (nextPropVal == NULL)
          continue;

        updateLiveGraph(bbInitGraphs[nextSuc],
                        {nextPropVal, valPlace.pe, valPlace.regAttr});

        if (liveOuts[nextSuc].count(nextPropVal)) {
          updateLiveGraph(bbFiniGraphs[nextSuc],
                          {nextPropVal, valPlace.pe, valPlace.regAttr});
          if (!visited.count(nextSuc))
            stack.emplace_back(nextSuc, nextPropVal);
        }
      }
    }
  }
}

// Update the global value placement if the initGraph and finiGraph of the
// updateBlk changed. All the other placement of other blocks are changed to
// maintain consistency of the liveness graph.
void updateGlobalValPlacement(
    Block *updateBlk, Region &region,
    std::map<Block *, SetVector<Value>> &liveIns,
    std::map<Block *, SetVector<Value>> &liveOuts,
    std::map<Block *, std::vector<ValuePlacement>> &bbInitGraphs,
    std::map<Block *, std::vector<ValuePlacement>> &bbFiniGraphs) {
  // update liveness graph if graph transformation is performed
  computeLiveValue(region, liveIns, liveOuts);

  auto &initGraph = bbInitGraphs[updateBlk];
  auto &finiGraph = bbFiniGraphs[updateBlk];

  // only value that are live in the graph
  auto removeDeadValue = [](std::vector<ValuePlacement> &graph,
                            SetVector<Value> &liveSet) {
    graph.erase(std::remove_if(graph.begin(), graph.end(),
                               [&liveSet](const ValuePlacement &val) {
                                 return liveSet.count(val.val) == 0;
                               }),
                graph.end());
  };

  removeDeadValue(initGraph, liveIns[updateBlk]);
  removeDeadValue(finiGraph, liveOuts[updateBlk]);

  llvm::errs() << "InitGraph: \n";
  for (auto val : initGraph) {
    llvm::errs() << val.val << " " << val.pe << " "
                 << static_cast<int>(val.regAttr) << "\n";
  }
  llvm::errs() << "FiniGraph: \n";
  for (auto val : finiGraph) {
    llvm::errs() << val.val << " " << val.pe << " "
                 << static_cast<int>(val.regAttr) << "\n";
  }

  // update the global value placement with the updated initGraph
  for (auto valPlace : initGraph) {
    auto val = valPlace.val;
    auto pe = valPlace.pe;
    auto regAttr = valPlace.regAttr;
    // find the corresponding value if it is live in other bb's initGraph or
    // finiGraph
    auto curBlk = updateBlk;
    updatePredecessorPlacement(valPlace, updateBlk, liveIns, liveOuts,
                               bbInitGraphs, bbFiniGraphs);
  }

  // update the global value placement with the updated finiGraph
  for (auto valPlace : finiGraph) {
    auto val = valPlace.val;
    auto pe = valPlace.pe;
    auto regAttr = valPlace.regAttr;
    // find the corresponding value if it is live in other bb's initGraph or
    // finiGraph
    auto curBlk = updateBlk;
    updateSuccessorPlacement(valPlace, updateBlk, liveIns, liveOuts,
                             bbInitGraphs, bbFiniGraphs);
  }

  // finish update
}

void calculateTemporalSpatialSchedule(
    Region &region,
    std::map<mlir::Operation *, compigra::ScheduleUnit> &solution,
    const std::string fileName) {
  unsigned kernelTime = 0;
  for (auto &block : region.getBlocks()) {
    int alignStartTime = kernelTime;
    int endTime = kernelTime;
    auto bbStart = 1;
    auto gap = kernelTime - bbStart;
    // blockStartT[&block] = alignStartTime;
    for (auto &op : block.getOperations()) {
      if (solution.find(&op) == solution.end())
        continue;

      auto &su = solution[&op];
      su.time += gap;
      endTime = std::max(endTime, su.time);
    }
    // blockEndT[&block] = endTime + 1;
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

namespace {
struct FastASMGenTemporalCGRAPass
    : public compigra::impl::FastASMGenTemporalCGRABase<
          FastASMGenTemporalCGRAPass> {

  explicit FastASMGenTemporalCGRAPass(int nRow, int nCol, int mem,
                                      StringRef msOpt, StringRef asmOutDir) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    auto funcOp = *modOp.getOps<func::FuncOp>().begin();
    OpBuilder builder(funcOp.getContext());
    if (asmOutDir.empty())
      asmOutDir = "out";
    std::string outDir = asmOutDir;

    Region &region = funcOp.getBody();

    std::map<Block *, SetVector<Value>> liveIns;
    std::map<Block *, SetVector<Value>> liveOuts;

    std::map<Block *, std::vector<ValuePlacement>> bbInitGraphs;
    std::map<Block *, std::vector<ValuePlacement>> bbFiniGraphs;

    computeLiveValue(region, liveIns, liveOuts);
    printBlockLiveValue(region, liveIns, liveOuts);

    int bbId = 0;

    logMessage("BasicBlock op assignment\n", true);
    std::map<mlir::Operation *, compigra::ScheduleUnit> rawSolution;

    for (auto &bb : region.getBlocks()) {
      llvm::errs() << "\n";
      logMessage("\nBBId: " + std::to_string(bbId) +
                 "==============================\n");

      llvm::errs() << "BBId: " + std::to_string(bbId) +
                          "==============================\n";
      // Init operation assginer
      BasicBlockOpAssignment bbOpAssignment(&bb, 3, nRow, nCol, builder);
      auto zeroIntOp = getZeroConstant(region, builder);
      auto zeroFloatOp = getZeroConstant(region, builder, true);
      bbOpAssignment.setUpZeroOp(zeroIntOp, zeroFloatOp);

      // set up liveness prerequisite
      bbOpAssignment.setPrerequisiteToStartGraph(bbInitGraphs[&bb]);
      bbOpAssignment.setPrerequisiteToFinishGraph(bbFiniGraphs[&bb]);
      if (failed(bbOpAssignment.mappingBBdataflowToCGRA(liveIns, liveOuts))) {
        // DEBUG, print the liveIn and liveOut and their placement
        llvm::errs() << "Failed to map BB dataflow to CGRA\n";
        llvm::errs() << "LiveIn: ";
        for (auto val : liveIns[&bb]) {
          if (val.isa<BlockArgument>()) {
            for (auto [ind, bb] : llvm::enumerate(region.getBlocks()))
              if (&bb == val.getParentBlock()) {
                llvm::errs() << ind << " ";
                break;
              }
          }
          llvm::errs() << val << ": ";
          if (bbInitGraphs[&bb].empty()) {
            llvm::errs() << "No placement\n";
          } else {
            for (auto valPlace : bbInitGraphs[&bb]) {
              if (valPlace.val == val) {
                llvm::errs() << "[" << valPlace.pe << " "
                             << static_cast<int>(valPlace.regAttr) << "]\n";
              }
            }
          }
        }
        llvm::errs() << "LiveOut: ";
        for (auto val : liveOuts[&bb]) {
          if (val.isa<BlockArgument>()) {
            for (auto [ind, bb] : llvm::enumerate(region.getBlocks()))
              if (&bb == val.getParentBlock()) {
                llvm::errs() << ind << " ";
                break;
              }
          }
          llvm::errs() << val << ": ";
          if (bbFiniGraphs[&bb].empty()) {
            llvm::errs() << "No placement\n";
          } else {
            for (auto valPlace : bbFiniGraphs[&bb]) {
              if (valPlace.val == val) {
                llvm::errs() << "[" << valPlace.pe << " "
                             << static_cast<int>(valPlace.regAttr) << "]\n";
              }
            }
          }
        }

        return;
        return signalPassFailure();
      }

      auto soluBB = bbOpAssignment.getSolution();
      // write soluBB into rawSolution
      for (auto [op, unit] : soluBB) {
        rawSolution[op] = unit;
      }

      // print the initGraph and finiGraph of the block
      auto initGraph = bbOpAssignment.getStartEmbeddingGraph();
      auto finiGraph = bbOpAssignment.getFiniEmbeddingGraph();

      bbInitGraphs[&bb] = initGraph;
      bbFiniGraphs[&bb] = finiGraph;
      // update the liveIn and liveOut with the initGraph and finiGraph
      computeLiveValue(region, liveIns, liveOuts);
      updateGlobalValPlacement(&bb, region, liveIns, liveOuts, bbInitGraphs,
                               bbFiniGraphs);
      logMessage("InitGraph: ");
      printLiveGraph(bbInitGraphs);
      logMessage("FiniGraph:");
      printLiveGraph(bbFiniGraphs);

      // if (bbId == 2)
      //   break;
      bbId++;
    }

    // organize the rawSolution to a final solution
    calculateTemporalSpatialSchedule(region, rawSolution,
                                     "space_temporal_assignment.csv");
    // perform register allocation
    OpenEdgeASMGen asmGen(region, 3, nRow);
    asmGen.setSolution(rawSolution);
    if (failed(asmGen.allocateRegisters())) {
      llvm::errs() << "Failed to allocate registers\n";
      // return signalPassFailure();
    }
    asmGen.printKnownSchedule(true, 0, outDir);
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