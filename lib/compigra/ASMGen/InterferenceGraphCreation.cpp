//===- InterferenceGraphCreation.cpp -  Funcs for IG gen *- C++ ---------*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions for interference graph generation in CGRA PE.
//
//===----------------------------------------------------------------------===//

#include "compigra/ASMGen/InterferenceGraphCreation.h"
#include "compigra/Scheduler/KernelSchedule.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace compigra;

bool doesADominateB(Operation *opA, Operation *opB, Operation *topLevelOp) {
  // Create an instance of DominanceInfo
  DominanceInfo dominanceInfo(topLevelOp);

  // Check if defA dominates defB
  return dominanceInfo.dominates(opA, opB);
}

static int getValueIndex(Value val, std::map<int, Value> opResult) {
  for (auto [ind, res] : opResult)
    if (res.getDefiningOp() == val.getDefiningOp())
      return ind;
  return -1;
}

static bool equalValueSet(std::unordered_set<int> set1,
                          std::unordered_set<int> set2) {
  if (set1.size() != set2.size())
    return false;
  for (auto val : set1)
    if (set2.find(val) == set2.end())
      return false;
  return true;
}

// static bool phiSrc

namespace compigra {
InterferenceGraph<int>
createInterferenceGraph(std::map<int, mlir::Operation *> &opList,
                        std::map<int, Value> &opMap) {
  std::map<Operation *, std::unordered_set<int>> def;
  std::map<Operation *, std::unordered_set<int>> use;

  InterferenceGraph<int> graph;
  unsigned ind = 0;
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    // Branch and constant operation is not interfered with other
    // operations
    Operation *op = it->second;
    llvm::errs() << "---" << *op << "\n";
    if (isa<LLVM::BrOp, LLVM::ConstantOp>(op))
      continue;
    if (op->getNumResults() > 0)
      if (getValueIndex(op->getResult(0), opMap) == -1) {
        opMap[ind] = op->getResult(0);
        // init a key in the adjList to represent the operator
        graph.addVertex(ind);
        // init the vertex that belongs to this PE which need to be colored
        graph.initVertex(ind);
        ind++;
        def[op].insert(getValueIndex(op->getResult(0), opMap));
      }
  }

  // Add operands to the graph
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    Operation *op = it->second;
    if (isa<LLVM::BrOp, LLVM::ConstantOp>(op))
      continue;
    for (auto [opInd, operand] : llvm::enumerate(op->getOperands())) {
      auto defOp = getCntDefOpIndirectly(operand, op->getBlock())[0];
      if (isa<LLVM::ConstantOp>(defOp))
        continue;
      // Skip the branch operator
      if (isa<cgra::ConditionalBranchOp>(op) && opInd >= 2)
        break;

      // TODO[@Yuxuan]: handle the block argument(phi node) case
      if (getValueIndex(operand, opMap) == -1) {
        opMap[ind] = operand;
        graph.addVertex(ind);
        // if the operand is a block argument and its source operation is
        // executed inside the PE, add it to the vertex set
        if (isa<BlockArgument>(operand)) {
          auto producer = getCntDefOpIndirectly(operand, op->getBlock())[0];
          // if find producer in opList
          if (std::any_of(opList.begin(), opList.end(),
                          [producer](const auto &entry) {
                            return entry.second == producer;
                          }))
            graph.initVertex(ind);
        }
        ind++;
      }
      use[op].insert(getValueIndex(operand, opMap));
      llvm::errs() << "USE(" << *op << ")" << operand << " "
                   << getValueIndex(operand, opMap) << "\n";
    }
  }

  std::vector<Operation *> sortedOps;
  for (auto [t, op] : opList)
    sortedOps.push_back(op);

  // print sortedOps
  std::map<Operation *, std::unordered_set<int>> liveIn;
  std::map<Operation *, std::unordered_set<int>> liveOut;
  while (true) {
    bool changed = false;
    for (int i = sortedOps.size() - 1; i >= 0; i--) {
      auto op = sortedOps[i];
      if (i < sortedOps.size() - 1) {
        // TODO[@Yuxuan]: define operation that seek the real successor
        auto succ = sortedOps[i + 1];
        // Calculate liveOut
        // if liveIn is empty, continue
        if (liveIn.find(succ) == liveIn.end())
          continue;
        for (auto live : liveIn[succ]) {
          // don't add the result operators from other PE to liveOut
          if (!graph.needColor(live))
            continue;
          if (liveOut[op].find(live) == liveOut[op].end()) {
            changed = true;
            liveOut[op].insert(live);
          }
        }
      }

      // Calculate liveIn
      std::unordered_set<int> newLiveIn = use[op];
      for (auto v : liveOut[op])
        if (def[op].find(v) == def[op].end())
          newLiveIn.insert(v);

      // check whether liveIn is changed
      if (!equalValueSet(newLiveIn, liveIn[op])) {
        changed = true;
        liveIn[op] = newLiveIn;
      }
    }
    if (!changed)
      break;
  }

  // create interference graph with defOp and liveOut
  for (auto op : sortedOps) {
    if (op->getNumResults() == 0)
      continue;
    auto defOp = getValueIndex(op->getResult(0), opMap);
    if (defOp == -1)
      continue;
    for (auto liveOp : liveOut[op]) {
      if (defOp != liveOp)
        graph.addEdge(defOp, liveOp);
    }
  }
  return graph;
}
} // namespace compigra
