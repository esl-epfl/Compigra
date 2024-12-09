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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace compigra;

bool doesADominateB(Operation *opA, Operation *opB, Operation *topLevelOp) {
  // Create an instance of DominanceInfo
  DominanceInfo dominanceInfo(topLevelOp);

  // Check if defA dominates defB
  return dominanceInfo.dominates(opA, opB);
}

static int
getOperationIndex(Operation *op,
                  const std::map<int, std::pair<Operation *, Value>> opMap) {
  for (auto [ind, pair] : opMap)
    if (pair.first == op)
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

namespace compigra {
int getValueIndex(Value val,
                  const std::map<int, std::pair<Operation *, Value>> opMap) {
  for (auto [ind, pair] : opMap)
    if (pair.second == val)
      return ind;
  return -1;
}

SmallVector<Value, 2> getSrcOprandsOfPhi(BlockArgument arg, bool eraseUse) {
  SmallVector<Value, 2> srcOprands;
  Block *blk = arg.getOwner();
  unsigned argIndex = arg.getArgNumber();
  for (auto predBlk : blk->getPredecessors()) {
    Operation *termOp = predBlk->getTerminator();
    if (auto branchOp = dyn_cast_or_null<cf::BranchOp>(termOp)) {
      srcOprands.push_back(branchOp.getOperand(argIndex));
      if (eraseUse)
        branchOp.eraseOperand(argIndex);
    } else if (auto branchOp =
                   dyn_cast_or_null<cgra::ConditionalBranchOp>(termOp)) {
      if (blk == branchOp.getSuccessor(0)) {
        srcOprands.push_back(branchOp.getTrueOperand(argIndex));
        // remove argIndex from the false operand
        if (eraseUse)
          branchOp.eraseOperand(argIndex + 2);
      } else {
        srcOprands.push_back(branchOp.getFalseOperand(argIndex));
        if (eraseUse)
          branchOp.eraseOperand(argIndex + 2 + branchOp.getNumTrueOperands());
      }
    }
  }
  return srcOprands;
}

bool isPhiRelatedValue(Value val) {
  if (val.isa<BlockArgument>())
    return true;

  for (auto &use : val.getUses()) {
    // cf.br carries the block argument
    if (isa<cf::BranchOp>(use.getOwner()))
      return true;

    // cgra.cond_br carries the block argument after the condition
    if (isa<cgra::ConditionalBranchOp>(use.getOwner()) &&
        use.getOperandNumber() > 1)
      return true;
  }
  return false;
}

SmallVector<Block *, 4> getCntBlocksThroughPhi(Value val) {
  SmallVector<Block *, 4> cntBlocks;
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    if (isa<cf::BranchOp>(user)) {
      Block *succBlk = user->getSuccessor(0);
      cntBlocks.push_back(succBlk);
    }
    if (isa<cgra::ConditionalBranchOp>(user)) {
      unsigned argIndex = use.getOperandNumber();
      if (argIndex < 2)
        continue;

      if (argIndex >=
          2 + dyn_cast<cgra::ConditionalBranchOp>(user).getNumTrueOperands())
        cntBlocks.push_back(user->getSuccessor(1));
      else
        cntBlocks.push_back(user->getSuccessor(0));
    }
  }
  return cntBlocks;
}

BlockArgument getCntBlockArgument(Value val, Block *succBlk) {
  // search for the connected block argument
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    if (isa<cf::BranchOp>(user)) {
      unsigned argIndex = use.getOperandNumber();
      if (user->getSuccessor(0) == succBlk)
        return succBlk->getArgument(argIndex);
    }
    if (isa<cgra::ConditionalBranchOp>(user)) {
      unsigned argIndex = use.getOperandNumber();
      if (argIndex < 2)
        continue;
      // true successor
      if (user->getSuccessor(0) == succBlk)
        return succBlk->getArgument(argIndex - 2);
      // false successor
      if (user->getSuccessor(1) == succBlk)
        return succBlk->getArgument(
            argIndex - 2 -
            dyn_cast<cgra::ConditionalBranchOp>(user).getNumTrueOperands());
    }
  }
  // no matching block argument
  return nullptr;
}

InterferenceGraph<int>
createInterferenceGraph(std::map<int, mlir::Operation *> &opList,
                        std::map<int, std::pair<Operation *, Value>> &defMap,
                        std::map<int, std::unordered_set<int>> ctrlFlow) {
  std::map<int, std::unordered_set<int>> use;

  InterferenceGraph<int> graph;
  unsigned ind = 0;
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    // Branch and constant operation is not interfered with other
    // operations
    Operation *op = it->second;
    if (isa<LLVM::BrOp, LLVM::ConstantOp>(op))
      continue;
    if (op->getNumResults() == 0) {
      // def is empty for the operation without result
      defMap[ind] = {op, nullptr};
      ind++;
      continue;
    }

    if (getValueIndex(op->getResult(0), defMap) == -1) {
      defMap[ind] = {op, op->getResult(0)};
      // init a key in the adjList to represent the operator
      graph.addVertex(ind);
      ind++;

      // init the vertex that belongs to this PE which need to be colored
      // graph.initVertex(ind);
      // def[ind].insert(op->getResult(0));
      // def[op].insert(getValueIndex(op->getResult(0), opMap));
    }
    // if the value is phi related, also add the block argument to the graph
    if (!isPhiRelatedValue(op->getResult(0)))
      continue;

    llvm::errs() << "Phi related value: " << op->getResult(0) << "\n";
    for (auto suc : getCntBlocksThroughPhi(op->getResult(0))) {
      auto arg = getCntBlockArgument(op->getResult(0), suc);
      llvm::errs() << "Block argument: " << arg << " "
                   << suc->getOperations().front() << "\n";
      defMap[ind] = {nullptr, arg};
      ind++;
    }
  }

  // Add operands to the graph
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    Operation *op = it->second;
    if (isa<LLVM::BrOp, LLVM::ConstantOp>(op))
      continue;
    for (auto [opInd, operand] : llvm::enumerate(op->getOperands())) {
      // Skip the branch operator
      if (isa<cgra::ConditionalBranchOp>(op) && opInd >= 2)
        break;

      auto defOp = getCntDefOpIndirectly(operand)[0];
      if (isa<LLVM::ConstantOp>(defOp))
        continue;

      int useInd = getValueIndex(operand, defMap);
      if (useInd == -1)
        continue;

      if (op->getNumResults() == 0)
        use[getOperationIndex(op, defMap)].insert(useInd);
      else
        use[getValueIndex(op->getResult(0), defMap)].insert(useInd);

      // TODO[@YYY]: handle the block argument(phi node) case
      // For operand produced in other PE, consider it as a constant and don't
      // add it to the graph.

      // llvm::errs() << operand << " not found in opMap\n";
      // opMap[ind] = operand;
      // graph.addVertex(ind);
      // if the operand is a block argument and its source operation is
      // executed inside the PE, add it to the vertex set

      // Only consider if the operand is a block argument
      // if (!isa<BlockArgument>(operand))
      //   continue;

      // auto producer = getCntDefOpIndirectly(operand)[0];
      // auto it = std::find_if(
      //     opList.begin(), opList.end(),
      //     [producer](const auto &entry) { return entry.second == producer;
      //     });
      // if (it == opList.end())
      //   continue;

      // graph.addVertex(ind);
      // ind++;

      // for (auto srcOperand :
      //      getSrcOprandsOfPhi(dyn_cast<BlockArgument>(operand))) {
      //   int useInd = getValueIndex(srcOperand, opMap);
      //   if (useInd == -1)
      //     break;
      //   def[ind].insert(operand);
      //   ind++;
      // }

      // // if find producer in opList
      // use[ind].insert(operand);
      // ind++;
    }
    // use[op].insert(getValueIndex(operand, opMap));
  }

  // print def and use
  for (auto &[ind, val] : defMap) {
    llvm::errs() << "def[" << ind << "]: ";
    if (val.first)
      llvm::errs() << *(val.first) << "\n";
    else
      llvm::errs() << val.second << "\n";
    // print its use
    llvm::errs() << "   use: ";
    for (auto useInd : use[ind])
      llvm::errs() << useInd << " ";
    llvm::errs() << "\n";
  }

  std::vector<Operation *> sortedOps;
  for (auto [t, op] : opList)
    sortedOps.push_back(op);

  // print sortedOps
  std::map<Operation *, std::unordered_set<int>> liveIn;
  std::map<Operation *, std::unordered_set<int>> liveOut;

  // std::map<int, std::unordered_set<int>> liveIn;
  // std::map<int, std::unordered_set<int>> liveOut;
  // TODO[@YYY]: consider the effect of the control flow
  auto succMap = getSuccessorMap(opList, ctrlFlow);
  while (true) {
    bool changed = false;
    for (int i = sortedOps.size() - 1; i >= 0; i--) {
      auto op = sortedOps[i];
      if (i < sortedOps.size()) {
        auto succOps = succMap[op];
        for (auto succ : succOps) {
          // Calculate liveOut
          // if liveIn is empty, continue
          if (liveIn.find(succ) == liveIn.end())
            continue;
          for (auto live : liveIn[succ]) {
            // don't add the result operators from other PE to liveOut
            // if (!graph.needColor(live))
            //   continue;
            if (liveOut[op].find(live) == liveOut[op].end()) {
              changed = true;
              liveOut[op].insert(live);
            }
          }
        }
      }

      // Calculate liveIn
      int opInd = -1;
      if (op->getNumResults() == 0)
        opInd = getOperationIndex(op, defMap);
      else
        opInd = getValueIndex(op->getResult(0), defMap);
      std::unordered_set<int> newLiveIn = use[opInd];
      for (auto v : liveOut[op])
        // if (def[op].find(v) == def[op].end())
        //   newLiveIn.insert(v);
        if (v != opInd)
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
    auto defOp = getValueIndex(op->getResult(0), defMap);
    if (defOp == -1)
      continue;
    for (auto liveOp : liveOut[op]) {
      if (defOp != liveOp)
        graph.addEdge(defOp, liveOp);
    }
  }
  return graph;
}

SmallVector<Operation *>
getSuccOps(Operation *op, const std::map<int, mlir::Operation *> &opList,
           std::map<int, std::unordered_set<int>> ctrlFlow) {
  SmallVector<Operation *> succOps;
  // find the dst PC of op
  int srcPC = -1;
  for (auto [ind, opTest] : opList) {
    if (opTest == op) {
      srcPC = ind;
      break;
    }
  }

  std::stack<int> pcStack;
  std::set<int> visited;
  std::set<int> inStack;
  pcStack.push(srcPC);

  while (!pcStack.empty()) {
    int curPC = pcStack.top();

    if (visited.find(curPC) != visited.end()) {
      pcStack.pop();
      continue;
    }

    if (inStack.find(curPC) != inStack.end()) {
      // node is already in the stack, meaning visited it before but not
      // fully processed it; Pop it now as all its children have been visited
      pcStack.pop();
      visited.insert(curPC);
      continue;
    }

    // First time seeing this node, mark it as in the stack
    inStack.insert(curPC);

    bool allSuccVisited = true;
    for (int cntPC : ctrlFlow[curPC]) {
      if (visited.find(cntPC) != visited.end())
        continue;

      if (inStack.find(cntPC) == inStack.end()) {
        allSuccVisited = false;
      }

      if (opList.find(cntPC) != opList.end())
        succOps.push_back(opList.at(cntPC));
      else
        pcStack.push(cntPC);
    }

    if (allSuccVisited) {
      // All children of this node have been visited, mark it as visited
      visited.insert(curPC);
      pcStack.pop();
    }
  }
  return succOps;
}

std::map<Operation *, std::unordered_set<Operation *>>
getSuccessorMap(const std::map<int, mlir::Operation *> &opList,
                const std::map<int, std::unordered_set<int>> ctrlFlow) {
  std::map<Operation *, std::unordered_set<Operation *>> succMap;
  for (auto [ind, op] : opList) {
    auto succOps = getSuccOps(op, opList, ctrlFlow);
    succMap[op] =
        std::unordered_set<Operation *>(succOps.begin(), succOps.end());
  }
  return succMap;
}
} // namespace compigra
