//===- OpenEdge.cpp - Declare the functions for gen OpenEdge ASM*- C++ -*-===//
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

#include "compigra/ASMGen/OpenEdgeASM.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraOps.h"
#include "compigra/Scheduler/KernelSchedule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <set>

#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

#include <iomanip>

using namespace mlir;
using namespace compigra;

static bool equalValueSet(std::unordered_set<int> set1,
                          std::unordered_set<int> set2) {
  if (set1.size() != set2.size())
    return false;
  for (auto val : set1)
    if (set2.find(val) == set2.end())
      return false;
  return true;
}

/// Return the union set specified by key in keys
static std::set<int> getUnionSet(std::map<int, std::unordered_set<int>> mapSet,
                                 std::unordered_set<int> keys) {
  // Get the index in ascending order
  std::set<int> unionSet;
  if (keys.empty())
    return unionSet;

  for (auto [key, vals] : mapSet) {
    if (keys.count(key))
      for (auto val : vals)
        unionSet.insert(val);
  }
  return unionSet;
}

/// Return the union set specified by key in keys
static std::set<int> getUnionSet(std::vector<std::set<int>> sets) {
  // Get the index in ascending order
  std::set<int> unionSet;
  for (auto set : sets)
    for (auto val : set)
      unionSet.insert(val);
  return unionSet;
}

/// Return the difference set
static std::set<int> getDiffSet(std::set<int> set1, std::set<int> set2) {
  std::set<int> diffSet;
  for (auto val : set1)
    if (set2.find(val) == set2.end())
      diffSet.insert(val);
  return diffSet;
}

/// In modulo scheduling, multiple operations might be executed in a same basic
/// block with different operantors. This function is used to get the
/// aggregation of the operation sets belongs to different loop iterations.
static std::vector<std::set<int>>
getOperationSet(std::map<int, std::unordered_set<int>> timeMap,
                std::unordered_set<int> keys,
                std::vector<std::set<int>> &prevSet, bool epilog = false) {

  if (keys.empty())
    return {};

  if (epilog) {
    std::vector<std::set<int>> opSets = {{}};
    for (auto [key, vals] : timeMap)
      if (keys.count(key))
        for (auto val : vals) {
          // if the value shows twice, it belongs to different loop iterations
          if (opSets.back().count(val) == 0)
            opSets.back().insert(val);
          else {
            opSets.push_back({});
            opSets.back().insert(val);
          }
        }

    // sort opSets according to its smallest element
    std::sort(opSets.begin(), opSets.end(),
              [](const std::set<int> &a, const std::set<int> &b) {
                return *a.begin() > *b.begin();
              });
    return opSets;
  }

  // if it is prolog or loop kernel, it always add new ops based on the previous
  // operation set
  std::vector<std::set<int>> opSets = prevSet;
  auto unionSet = getUnionSet(timeMap, keys);
  auto prevUnionSet = getUnionSet(prevSet);
  opSets.push_back(getDiffSet(unionSet, prevUnionSet));
  // sort opSets according to its smallest element
  std::sort(opSets.begin(), opSets.end(),
            [](const std::set<int> &a, const std::set<int> &b) {
              return *a.begin() > *b.begin();
            });
  return opSets;
}

static int getValueIndex(Value val, std::map<int, Value> opResult) {
  for (auto [ind, res] : opResult)
    if (res.getDefiningOp() == val.getDefiningOp())
      return ind;
  return -1;
}

/// Get the loop block of the region
static Block *getLoopBlock(Region &region) {
  for (auto &block : region)
    for (auto suc : block.getSuccessors())
      if (suc == &block)
        return &block;
  return nullptr;
}

/// Get the init block of the region
static Block *getInitBlock(Block *loopBlk) {
  for (auto pred : loopBlk->getPredecessors())
    if (pred != loopBlk)
      return pred;
  return nullptr;
}

static Block *getCondBrFalseDest(Block *blk) {
  for (auto succ : blk->getSuccessors())
    if (succ != blk)
      return succ;
  return nullptr;
}

static void getReplicatedOp(Operation *op, OpBuilder &builder) {}

/// Determine whether a  basic block is the loop kernel by counting the
/// operation Id is equal to the number of operations in the block.
static bool isKernel(unsigned endId, const std::set<int> bb) {
  for (size_t i = 0; i < endId; i++)
    if (std::find(bb.begin(), bb.end(), i) == bb.end())
      return false;
  return true;
}

static LogicalResult replaceSucBlock(Operation *term, Block *newBlock,
                                     Block *prevBlock = nullptr) {
  if (auto br = dyn_cast<LLVM::BrOp>(term)) {
    br.setSuccessor(newBlock);
    return success();
  }

  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(term)) {
    for (auto [ind, suc] : llvm::enumerate(condBr->getSuccessors()))
      if (suc == prevBlock) {
        condBr.setSuccessor(newBlock, ind);
        return success();
      }
  }
  return failure();
}

/// Get i'th op in opList
static Operation *getOp(Block::OpListType &opList, unsigned i) {
  // Check if the index is within the bounds of the container
  if (i < std::distance(opList.begin(), opList.end())) {
    auto it = opList.begin();
    std::advance(it, i);
    return &(*it);
  }
  return nullptr;
}

static unsigned getOpId(Block::OpListType &opList, Operation *search) {
  for (auto [ind, op] : llvm::enumerate(opList))
    if (&op == search)
      return ind;
  return -1;
}

static LogicalResult setCloneOpOperands(Operation *cloneOp, Operation *op) {
  for (auto [ind, opr] : llvm::enumerate(op->getOperands()))
    // Determine whether the operands is block arguments

    return success();
}

static void addBranchArgument(Operation *term, Operation *producer, Block *dest,
                              OpBuilder *builder) {
  // need to create a new terminator to replace the old one
  builder->setInsertionPoint(term);
  if (auto br = dyn_cast<LLVM::BrOp>(term)) {
    SmallVector<Value> operands;
    for (auto opr : br->getOperands())
      operands.push_back(opr);
    operands.push_back(producer->getResult(0));
    builder->create<LLVM::BrOp>(term->getLoc(), operands, dest);
    term->erase();
    return;
  }

  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(term)) {
    SmallVector<Value> trueOperands;
    for (auto opr : condBr.getTrueDestOperands())
      trueOperands.push_back(opr);
    SmallVector<Value> falseOperands;
    for (auto opr : condBr.getFalseDestOperands())
      falseOperands.push_back(opr);
    if (condBr.getTrueDest() == dest) {
      trueOperands.push_back(producer->getResult(0));
      builder->create<cgra::ConditionalBranchOp>(
          term->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
          condBr.getOperand(1), dest, trueOperands, condBr.getFalseDest(),
          falseOperands);
    } else {
      falseOperands.push_back(producer->getResult(0));
      builder->create<cgra::ConditionalBranchOp>(
          term->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
          condBr.getOperand(1), condBr.getTrueDest(), trueOperands, dest,
          falseOperands);
    }
    term->erase();
    return;
  }
}

static void removeBlockArgs(Operation *term, std::vector<unsigned> argId,
                            OpBuilder &builder, Block *dest = nullptr) {
  builder.setInsertionPoint(term);
  if (auto br = dyn_cast<LLVM::BrOp>(term)) {
    SmallVector<Value> operands;
    for (auto [ind, opr] : llvm::enumerate(br->getOperands()))
      if (std::find(argId.begin(), argId.end(), ind) == argId.end())
        operands.push_back(opr);
    builder.create<LLVM::BrOp>(term->getLoc(), operands, br.getSuccessor());
    term->erase();
    return;
  }

  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(term)) {
    SmallVector<Value> trueOperands;
    for (auto [ind, opr] : llvm::enumerate(condBr.getTrueDestOperands()))
      if (std::find(argId.begin(), argId.end(), ind) == argId.end())
        trueOperands.push_back(opr);
    SmallVector<Value> falseOperands;
    for (auto [ind, opr] : llvm::enumerate(condBr.getFalseDestOperands()))
      if (std::find(argId.begin(), argId.end(), ind) == argId.end())
        falseOperands.push_back(opr);
    builder.create<cgra::ConditionalBranchOp>(
        term->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
        condBr.getOperand(1), condBr.getTrueDest(), trueOperands,
        condBr.getFalseDest(), falseOperands);
    term->erase();
    return;
  }
}

/// Initialize the DFG within the blk with the operations in the opSet
static LogicalResult initDFGBB(Block *blk, Block *templateBlk, Block *prevNode,
                               std::vector<std::set<int>> &opSets,
                               std::map<int, Operation *> &preGenOps,
                               OpBuilder &builder, bool isKernel = false) {
  auto &opList = templateBlk->getOperations();
  unsigned totalOpNum = opList.size();
  std::map<int, Operation *> curGenOps;

  std::vector<int> argIds;
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;

  Operation *lastOp = nullptr;

  for (auto [_, opSet] : llvm::enumerate(opSets))
    for (auto ind : opSet) {
      // Get the operation from the index
      auto op = getOp(opList, ind);

      if (lastOp)
        builder.setInsertionPointAfter(lastOp);
      else
        builder.setInsertionPointToStart(blk);

      Operation *repOp = builder.clone(*op);

      for (auto [oprId, opr] : llvm::enumerate(op->getOperands())) {
        if (isa<BlockArgument>(opr)) {
          // insert block argument for this op
          auto arg = blk->addArgument(
              opr.getType(), prevNode->getOperations().back().getLoc());
          repOp->setOperand(oprId, arg);
          continue;
        }

        // Determine whether defined by operation produced in the loop
        auto defOp = opr.getDefiningOp();
        if (defOp)
          if (defOp->getBlock() == templateBlk) {
            unsigned opId = getOpId(opList, defOp);
            // use current generated operations
            if (curGenOps.count(opId) > 0) {
              auto corOp = curGenOps[opId];
              repOp->setOperand(oprId, corOp->getResult(0));
              continue;
            }

            // if the predecessor has generated the result, consume it
            if (preGenOps.count(opId) > 0) {
              auto corOp = preGenOps[opId];
              // if the current block is kernel, it has to take arguments from
              // the prolog or itself
              if (isKernel) {
                argTypes.push_back(corOp->getResult(0).getType());
                auto arg =
                    blk->addArgument(corOp->getResult(0).getType(),
                                     prevNode->getOperations().back().getLoc());
                repOp->setOperand(oprId, arg);
                // revise the predecessor terminator to propagate corOp result
                addBranchArgument(corOp->getBlock()->getTerminator(), corOp,
                                  blk, &builder);
                // the loop takes preGenOps[opId] and curGenOps[opId] as
                // operands
                argIds.push_back(opId);
              } else {
                repOp->setOperand(oprId, corOp->getResult(0));
              }
              continue;
            }
            llvm::errs() << "opId: " << opId << " not in opSet\n";
            return failure();
          }

        // set the same operand with op
        repOp->setOperand(oprId, opr.getDefiningOp()->getResult(0));
      }

      curGenOps[ind] = repOp;
      lastOp = repOp;
    }
  // replace curGenOps with preGenOps
  preGenOps = curGenOps;

  // add branch arguments for the kernel block
  if (isKernel) {
    // the loop block must have a conditional terminator
    if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(blk->getTerminator()))
      condBr.setTrueDest(blk);
    else
      return failure();
    for (auto argId : argIds)
      addBranchArgument(blk->getTerminator(), curGenOps[argId], blk, &builder);
  }

  return success();
}

static void reverseCondBrFlag(cgra::ConditionalBranchOp condBr,
                              bool reverseBB = false) {
  switch (condBr.getPredicate()) {
  case cgra::CondBrPredicate::ne:
    condBr.setPredicate(cgra::CondBrPredicate::eq);
    break;
  case cgra::CondBrPredicate::eq:
    condBr.setPredicate(cgra::CondBrPredicate::ne);
    break;
  case cgra::CondBrPredicate::lt: {
    condBr.setPredicate(cgra::CondBrPredicate::ge);
    Value tmp = condBr.getOperand(0);
    // reverse the first operands order
    condBr.setOperand(0, condBr.getOperand(1));
    condBr.setOperand(1, tmp);
    break;
  }
  case cgra::CondBrPredicate::ge: {
    condBr.setPredicate(cgra::CondBrPredicate::lt);
    Value tmp = condBr.getOperand(0);
    // reverse the first operands order
    condBr.setOperand(0, condBr.getOperand(1));
    condBr.setOperand(1, tmp);
  }
  }
  if (!reverseBB)
    return;
  // reverse the true and false block
  auto tmp = condBr.getTrueDest();
  condBr.setTrueDest(condBr.getFalseDest());
  condBr.setFalseDest(tmp);
}

static LogicalResult removeUselessBlockArg(Region &region, OpBuilder &builder) {
  unsigned num = 0;
  for (auto &block : region) {
    num++;
    if (block.isEntryBlock())
      continue;
    if (block.getArguments().size() == 0 ||
        std::distance(block.getPredecessors().begin(),
                      block.getPredecessors().end()) > 1)
      continue;
    // the block has only one predecessor and have arguments
    // get the corresponding value in the predecessor
    auto prevTerm = (*block.getPredecessors().begin())->getTerminator();
    auto oprIndBase = 0;
    if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(prevTerm)) {
      // the block is the false dest
      oprIndBase = 2;
      if (condBr.getFalseDest() == &block)
        oprIndBase += condBr.getTrueDestOperands().size();
    }

    // find the corresponding value in the predecessor
    std::vector<unsigned> argId;
    unsigned oprInd = 0;
    for (auto arg : llvm::make_early_inc_range(block.getArguments())) {
      arg.replaceAllUsesWith(prevTerm->getOperand(oprIndBase + oprInd));
      argId.push_back(oprInd);
      oprInd++;
    }
    // remove all block arguments
    llvm::BitVector bitVec(argId.size(), true);
    block.eraseArguments(bitVec);
    removeBlockArgs(prevTerm, argId, builder, &block);
  }
  return success();
}

LogicalResult
compigra::adaptCFGWithLoopMS(Region &region, OpBuilder &builder,
                             std::map<int, std::unordered_set<int>> &opTimeMap,
                             std::vector<std::unordered_set<int>> &bbTimeMap) {

  // Get related basic blocks
  Block *loopBlock = getLoopBlock(region);
  auto &loopOpList = loopBlock->getOperations();
  unsigned numOp = loopOpList.size();
  Block *initBlock = getInitBlock(loopBlock);
  Block *loopFalseBlk = getCondBrFalseDest(loopBlock);
  Block *finiBlock = loopFalseBlk->getSuccessor(0);

  SmallVector<Block *> newBlks = {initBlock};
  // init basic blocks first
  Block *prev = initBlock;
  Block *beforeNode = loopBlock;

  // specify the phase for CFG generation, phase :0(init), 1(prolog),
  // 2(loop),3(epilog),4(fini)
  int phase = 1;

  // Get the union of operations within a basic block
  std::map<int, Operation *> insertOps = {};
  std::vector<std::set<int>> opSet = {};
  for (auto [ind, s] : llvm::enumerate(bbTimeMap)) {
    // epilog is after the loop block
    if (phase == 2)
      phase = 3;

    opSet = getOperationSet(opTimeMap, s, opSet, phase == 3);
    for (auto u : opSet) {
      llvm::errs() << "{";
      for (auto i : u)
        llvm::errs() << i << " ";
      llvm::errs() << "} ";
    }
    llvm::errs() << "\n";

    prev = newBlks.back();
    if (isKernel(numOp, getUnionSet(opSet))) {
      llvm::errs() << ind << " is kernel\n";
      // beforeNode = finiBlock;
      phase = 2;
    }

    auto newBlk = builder.createBlock(loopBlock);
    newBlks.push_back(newBlk);

    // connect the current block to the CFG
    switch (phase) {
    case 1:
    case 2: {
      // if it is prolog, connect the block with block before it
      auto predBlk = newBlks.rbegin()[1];
      auto termOp = predBlk->getTerminator();
      if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(termOp)) {
        // execute the rest of operations and jump to the fini block
        std::vector<mlir::Location> loc1(
            condBr.getFalseDestOperands().size(),
            loopFalseBlk->getTerminator()->getLoc());
        auto connBB = builder.createBlock(
            finiBlock, condBr.getFalseDestOperands().getTypes(), loc1);
        builder.setInsertionPointToStart(connBB);
        builder.create<LLVM::BrOp>(
            connBB->getPrevNode()->getTerminator()->getLoc(), finiBlock);

        // create a new condBr op to switch the false and true dest
        builder.setInsertionPoint(termOp);
        auto newTerm = builder.create<cgra::ConditionalBranchOp>(
            termOp->getLoc(), condBr.getPredicate(), condBr.getOperand(0),
            condBr.getOperand(1), connBB, condBr.getFalseDestOperands(), newBlk,
            condBr.getTrueDestOperands());

        // reverse the flag of the conditional branch
        reverseCondBrFlag(newTerm);
        termOp->erase();
      } else if (auto br = dyn_cast<LLVM::BrOp>(termOp)) {
        br.setSuccessor(newBlk);
      }
      break;
    }
    case 3: {
      auto predBlk = newBlks.rbegin()[1];
      auto termOp = predBlk->getTerminator();
      if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(termOp)) {
        // set the true dest to be loop
        condBr.setTrueDest(predBlk);
        // set false dest to be epilog block
        condBr.setFalseDest(newBlk);
      } else {
        // the terminator of the loop block must be conditional branch
        return failure();
      }
      break;
    }
    default:
      break;
    }

    // init DFG in the new created basic block
    initDFGBB(newBlks.back(), loopBlock, prev, opSet, insertOps, builder,
              phase == 2);
  }
  // create a jump to the fini block
  auto epilog = newBlks.rbegin()[0];
  builder.setInsertionPointToEnd(epilog);
  builder.create<LLVM::BrOp>(epilog->getOperations().back().getLoc(),
                             finiBlock);

  // delete the origianl loop block
  // collect all operations in reverse order in a temporary vector.
  std::vector<Operation *> toErase;
  for (auto &op : llvm::reverse(loopOpList)) {
    toErase.push_back(&op);
  }
  // use llvm::make_early_inc_range to erase safely.
  for (auto *op : llvm::make_early_inc_range(toErase)) {
    op->erase();
  }
  loopBlock->erase();

  // erase loopFalseBlk
  loopFalseBlk->getTerminator()->erase();
  loopFalseBlk->erase();

  // remove the block arguments if it is not used
  removeUselessBlockArg(region, builder);

  return success();
}

/// Create the interference graph for the operations in the PE, the
/// corresponding relations with the abstract operands are stored in the
/// opMap.
static InterferenceGraph<int>
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
    if (isa<LLVM::BrOp, LLVM::ConstantOp>(op))
      continue;
    if (op->getNumResults() > 0)
      if (getValueIndex(op->getResult(0), opMap) == -1) {
        opMap[ind] = op->getResult(0);
        graph.addVertex(ind);

        graph.initVertex(ind);
        ind++;
        def[op].insert(getValueIndex(op->getResult(0), opMap));
      }
  }

  // Add operands to the graph
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    Operation *op = it->second;
    for (auto [opInd, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<LLVM::ConstantOp>(getCntOpIndirectly(operand)[0]))
        continue;
      // Skip the branch operator
      if (isa<cgra::ConditionalBranchOp>(op) && opInd == 2)
        break;

      if (getValueIndex(operand, opMap) == -1) {
        opMap[ind] = operand;
        graph.addVertex(ind);
        use[op].insert(getValueIndex(operand, opMap));
        ind++;
      }
    }
  }

  std::map<Operation *, std::unordered_set<int>> liveIn;
  std::map<Operation *, std::unordered_set<int>> liveOut;
  while (true) {
    bool changed = false;
    for (auto [t, op] : opList) {
      // Calculate liveOut
      for (auto succ : op->getUsers())
        for (auto live : liveIn[succ])
          if (liveOut[op].find(live) == liveOut[op].end()) {
            changed = true;
            liveOut[op].insert(live);
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
  for (auto [t, op] : opList) {
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

/// Function to perform Lexicographical Breadth-First Search and obtain
/// PEO
static std::vector<int>
lexBFS(const std::map<int, std::unordered_set<int>> &adjList) {
  int n = adjList.size();
  if (n == 0)
    return {};
  std::vector<int> peo;
  std::vector<std::unordered_set<int>> partition = {std::unordered_set<int>()};

  // Initialize the partition with all vertices
  for (const auto &entry : adjList) {
    partition[0].insert(entry.first);
  }

  while (!partition.empty()) {
    // Get the first set from partition and pick an arbitrary vertex
    std::unordered_set<int> currentSet = partition.front();
    partition.erase(partition.begin());
    int v = *currentSet.begin();
    currentSet.erase(currentSet.begin());
    peo.push_back(v);

    // If there are more vertices in the current set, reinsert it into the
    // partition
    if (!currentSet.empty()) {
      partition.insert(partition.begin(), currentSet);
    }

    // Update partition by moving neighbors of v to the front of their
    // respective sets
    std::vector<std::unordered_set<int>> newPartition;
    for (auto &part : partition) {
      std::unordered_set<int> newPart1, newPart2;
      for (int u : part) {
        if (adjList.at(v).find(u) != adjList.at(v).end()) {
          newPart1.insert(u);
        } else {
          newPart2.insert(u);
        }
      }
      if (!newPart1.empty())
        newPartition.push_back(newPart1);
      if (!newPart2.empty())
        newPartition.push_back(newPart2);
    }
    partition = newPartition;
  }

  return peo;
}

LogicalResult
compigra::allocateOutRegInPE(std::map<int, mlir::Operation *> opList,
                             std::map<Operation *, ScheduleUnit> &solution,
                             unsigned maxReg) {
  // init Operation result to integer
  std::map<int, Value> opMap;
  auto graph = createInterferenceGraph(opList, opMap);

  // print opMap and interference graph
  llvm::errs() << "OpMap:\n";
  // for (auto [ind, val] : opMap) {
  //   llvm::errs() << ind << " " << val << " "
  //                << (std::find(graph.vertices.begin(),
  //                graph.vertices.end(),
  //                              ind) == graph.vertices.end())
  //                << "\n";
  // }
  // graph.printGraph();

  // allocate register using graph coloring
  // TODO[@Yuxuan]: Spill the graph if the number of registers is not
  // enough
  auto peo = lexBFS(graph.adjList);
  if (peo.empty())
    return success();
  llvm::errs() << "PEO: [";
  // First check whether v has been pre-colored
  for (auto v : peo) {

    Value val = opMap[v];
    llvm::errs() << " " << v;

    auto defOp = val.getDefiningOp();
    if (!defOp)
      continue;

    llvm::errs() << "(" << *defOp << " NEED?: " << graph.needColor(v) << ")";
    if (graph.needColor(v)) {
      if (solution[defOp].reg >= 0)
        graph.colorMap[v] = solution[defOp].reg;
    } else {
      // Use produced pe as the color of the vertex
      graph.colorMap[v] = solution[defOp].pe;
    }
    llvm::errs() << " " << graph.colorMap[v];
  }
  llvm::errs() << "]\n";

  // Color the vertices in the order of PEO
  for (auto v : peo) {
    Value val = opMap[v];
    auto defOp = val.getDefiningOp();
    if (!defOp)
      continue;
    if (graph.needColor(v))
      llvm::errs() << "RA: " << v << " : " << solution[defOp].reg << "\n";
    if (graph.needColor(v) && solution[defOp].reg < 0) {
      std::unordered_set<int> usedColors;
      for (auto u : graph.adjList[v]) {
        if (graph.colorMap.find(u) != graph.colorMap.end())
          usedColors.insert(graph.colorMap[u]);
      }
      int color = 0;
      while (usedColors.find(color) != usedColors.end()) {
        color++;
        if (color >= maxReg)
          llvm::errs() << "FAILED ALLOCATE REGISTER\n";
        // maxReg is Rout. In principle, the register should not be
        // assigned to Rout if it is used internally.
        if (color >= maxReg)
          return failure();
      }
      graph.colorMap[v] = color;
      llvm::errs() << "ALLOCATE REGISTER " << color << " TO " << val << "\n";
      solution[defOp].reg = color;
    }
  }
  return success();
}

/// Function to print the map of instructions
static void
printInstructions(const std::map<Operation *, Instruction> &instructions) {
  for (const auto &pair : instructions) {
    const Instruction &inst = pair.second;
    llvm::errs() << "Op: " << *pair.first << "\n"
                 << "Name: " << inst.name << " "
                 << "Time: " << inst.time << " "
                 << "PE: " << inst.pe << " "
                 << "Rout: " << inst.Rout << " "
                 << "OpA: " << inst.opA << " "
                 << "OpB: " << inst.opB << "\n\n";
  }
}

static bool isRegisterDigit(const std::string &reg, unsigned maxReg) {
  if (reg.empty() || reg[0] != 'R')
    return false;
  for (size_t i = 1; i < reg.size(); i++) {
    if (!isdigit(reg[i]))
      return false;
  }
  return std::stoi(reg.substr(1)) <= maxReg;
}

LogicalResult OpenEdgeASMGen::allocateRegisters(
    std::map<Operation *, Instruction> restriction) {

  // First write restriction to the solution
  for (auto [op, inst] : restriction) {
    instSolution[op] = inst;
    solution[op].reg = inst.Rout;
  }

  for (size_t pe = 0; pe < nRow * nCol; ++pe) {
    auto ops = getOperationsAtPE(pe);

    bool update = false;
    // Allocate registers if it can be inferred from the known result.
    for (auto [time, op] : ops) {
      // Check whether the PE of the operation is known, if yes, skip
      // allocation
      if (solution[op].reg != -1)
        continue;

      for (auto &use : llvm::make_early_inc_range(op->getUses())) {
        Operation *user = getCntOpIndirectly(use.getOwner(), op);
        // if the user PE is not restricted, don't allocate register for
        // now
        int userPE = solution[user].pe;
        // If the user is executed in neighbour PE, the result should be
        // stored in Rout.
        if (userPE != pe) {
          solution[op].reg = maxReg;
          // Found the assigned register, stop seeking from other users
          break;
        }

        if (solution[user].reg != -1) {
          // The result operand should match with use operand
          unsigned operandIndex = use.getOperandNumber();
          auto regStr = operandIndex == 0 ? instSolution[user].opA
                                          : instSolution[user].opB;
          if (regStr == "ROUT")
            solution[op].reg = maxReg;
          else {
            if (isRegisterDigit(regStr, maxReg))
              solution[op].reg = std::stoi(regStr.substr(1));
            else
              return failure();
          }
          // If the correspond PE is set, stop seeking from its other
          // users
          break;
        }
      }
    }

    // allocate register for the operations in the PE
    llvm::errs() << "\nPE = " << pe << "\n";
    if (failed(allocateOutRegInPE(ops, solution, maxReg)))
      return failure();
  }

  for (auto [op, sol] : solution) {
    if (op->getNumResults() > 0 && sol.reg == -1)
      llvm::errs() << "Failed" << *op << " " << sol.pe << "\n";
    // return failure();
    instSolution[op].Rout = sol.reg;
  }

  llvm::errs() << "Register allocation done\n";

  // write register allocation results to instructions
  convertToInstructionMap();
  return success();
}

int OpenEdgeASMGen::getEarliestExecutionTime(Operation *op) {
  if (solution.find(op) != solution.end())
    return solution[op].time;
  return INT_MAX;
};

int OpenEdgeASMGen::getKernelStart() {
  int time = INT_MAX;
  for (auto [op, inst] : solution)
    if (inst.time < time)
      time = inst.time;
  return time;
};

int OpenEdgeASMGen::getKernelEnd() {
  int time = getKernelStart();
  for (auto [op, inst] : solution)
    if (inst.time > time)
      time = inst.time;
  return time;
};

int OpenEdgeASMGen::getEarliestExecutionTime(Block *block) {
  int time = INT_MAX;
  // Get the earliest execution of operations in the block
  for (Operation &op : block->getOperations()) {
    time = std::min(time, getEarliestExecutionTime(&op));
  }
  return time;
}

std::map<int, Operation *> OpenEdgeASMGen::getOperationsAtTime(int time) {
  std::map<int, Operation *> ops;
  for (auto [op, inst] : solution)
    if (inst.time == time)
      ops[inst.pe] = op;

  return ops;
}

std::map<int, Operation *> OpenEdgeASMGen::getOperationsAtPE(int pe) {
  std::map<int, Operation *> ops;
  for (auto [op, inst] : solution)
    if (inst.pe == pe)
      ops[inst.time] = op;

  return ops;
}

// Helper function to pad strings to a fixed width
static std::string padString(const std::string &str, size_t width) {
  if (str.length() >= width) {
    return str;
  }
  return str + std::string(width - str.length(),
                           ' '); // Pad with spaces on the right
}

static int getOpIndex(Operation *op) {
  auto block = op->getBlock();
  int index = 0;
  for (auto &iterOp : block->getOperations()) {
    if (op == &iterOp)
      return index;
    index++;
  }
  return -1;
}

/// Function for retrieve the operand knowning the src operation and user
/// operand's definition operand's PES.
/// e.g. Suppose %a = op %b, ..., return the string of the %b for %a, such
/// as R0, R1,... or RCT, RCB, RCR, RCL.
static std::string getOperandSrcReg(int peA, int peB, int srcReg, int nRow,
                                    int nCol, unsigned maxReg) {
  if (peA == peB)
    if (srcReg == maxReg)
      return "Rout";
    else
      return "R" + std::to_string(srcReg);

  // check userPE in the neighbour of srcPE
  // check whether peB on peA right
  if ((peA + 1) % nCol == peB % nCol && (peA / nCol == peB / nCol))
    return "RCR";

  // check whether peB on peA left
  if ((peB + 1) % nCol == peA % nCol && (peA / nCol == peB / nCol))
    return "RCL";

  // check whether peB on peA top
  if (((peA - nCol + (nRow * nCol)) % (nRow * nCol)) == peB)
    return "RCT";

  // check whether peB on peA bottom
  if (((peA + nCol) % (nRow * nCol)) == peB)
    return "RCB";

  return "ERROR";
}

LogicalResult OpenEdgeASMGen::convertToInstructionMap() {
  for (auto [op, unit] : solution) {
    if (isa<LLVM::BrOp>(op)) {
      continue;
    }

    Instruction inst;
    if (instSolution.find(op) != instSolution.end())
      inst = instSolution[op];
    else
      inst = Instruction{op->getName().getStringRef().str(),
                         unit.time,
                         unit.pe,
                         unit.reg,
                         "Unknown",
                         "Unknown"};

    // Get defition operation
    if (op->getNumOperands() > 0 && inst.opA == "Unknown") {
      auto producerA = op->getOperand(0).getDefiningOp();
      if (isa<LLVM::ConstantOp>(producerA))
        // assign opA to be Imm
        inst.opA = std::to_string(
            producerA->getAttrOfType<IntegerAttr>("value").getInt());
      else if (solution.find(producerA) != solution.end())
        inst.opA =
            getOperandSrcReg(unit.pe, solution[producerA].pe,
                             solution[producerA].reg, nRow, nCol, maxReg);
      else
        return failure();
    }

    if (op->getNumOperands() > 1 && inst.opB == "Unknown") {
      auto producerB = op->getOperand(1).getDefiningOp();
      if (isa<LLVM::ConstantOp>(producerB))
        // assign opA to be Imm
        inst.opB = std::to_string(
            producerB->getAttrOfType<IntegerAttr>("value").getInt());
      else if (solution.find(producerB) != solution.end())
        inst.opB =
            getOperandSrcReg(unit.pe, solution[producerB].pe,
                             solution[producerB].reg, nRow, nCol, maxReg);
      else
        return failure();
    }

    instSolution[op] = inst;
  }
  return success();
}

std::string OpenEdgeASMGen::printInstructionToISA(Operation *op,
                                                  bool dropNeighbourBr) {
  // If it is return, return EXIT
  if (isa<LLVM::ReturnOp>(op))
    return "EXIT";

  // Drop the dialect prefix
  size_t pos = op->getName().getStringRef().find(".");
  std::string opName = op->getName().getStringRef().substr(pos + 1).str();
  if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(op)) {
    switch (condBr.getPredicate()) {
    case cgra::CondBrPredicate::eq:
      opName = "beq";
      break;
    case cgra::CondBrPredicate::ne:
      opName = "bne";
      break;
    case cgra::CondBrPredicate::ge:
      opName = "bge";
      break;
    case cgra::CondBrPredicate::lt:
      opName = "blt";
      break;
    }
  }
  // make opName capital
  for (auto &c : opName) {
    c = std::toupper(c);
  }
  if (isaMap.count(opName) > 0)
    opName = isaMap[opName];

  // If the operation is branchOp, find the branch target
  if (auto brOp = dyn_cast<LLVM::BrOp>(op)) {
    auto block = brOp->getSuccessor(0);
    int sucTime = getEarliestExecutionTime(block);
    int curTime = getEarliestExecutionTime(op);
    if (sucTime == curTime + 1 && dropNeighbourBr)
      return "NOP";
    return "JUMP " + std::to_string(sucTime + baseTime);
  }

  std::string ROUT = "";
  if (op->getNumResults() > 0)
    ROUT = instSolution[op].Rout == maxReg
               ? " ROUT,"
               : " R" + std::to_string(instSolution[op].Rout) + ",";

  std::string opA = "";
  if (op->getNumOperands() > 0) {
    // check whether the operand is immediate
    auto cntOp = getCntOpIndirectly(op->getOperand(0))[0];
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(cntOp)) {
      int imm = constOp.getValueAttr().dyn_cast<IntegerAttr>().getInt();
      opA = imm ? " " + std::to_string(imm) : " ZERO";
    } else
      opA = " " + instSolution[op].opA;
  }

  std::string opB = "";
  if (op->getNumOperands() > 1) {
    auto cntOp = getCntOpIndirectly(op->getOperand(1))[0];
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(cntOp)) {
      int imm = constOp.getValueAttr().dyn_cast<IntegerAttr>().getInt();
      opB = imm ? "," + std::to_string(imm) : ",ZERO";
    } else
      opB = "," + instSolution[op].opB;
  }

  std::string addition = "";
  if (isa<cgra::ConditionalBranchOp>(op)) {
    auto block = op->getSuccessor(0);
    int sucTime = getEarliestExecutionTime(block);
    addition = " " + std::to_string(sucTime + baseTime);
  }

  return opName + ROUT + opA + opB + addition;
}

void OpenEdgeASMGen::dropTimeFrame(int time) {
  SmallVector<Operation *> removeOps;
  for (auto &it : llvm::make_early_inc_range(solution)) {
    if (it.second.time == time)
      removeOps.push_back(it.first);
    else if (it.second.time > time)
      it.second.time--;
  }
  for (auto op : removeOps)
    solution.erase(op);
}

/// Print the known schedule
void OpenEdgeASMGen::printKnownSchedule(bool GridLIke, int startPC,
                                        std::string outDir) {
  // For each time step
  initBaseTime(startPC);
  std::vector<std::vector<std::string>> asmCode;
  int endTime = getKernelEnd();
  for (int t = getKernelStart(); t <= endTime; t++) {
    // Get the operations scheduled at the time step
    auto ops = getOperationsAtTime(t);
    // print ops
    // for (auto [pe, op] : ops) {
    //   llvm::errs() << "Time = " << t << " PE = " << pe << " ";
    //   llvm::errs() << *op << "\n";
    // }
    std::vector<std::string> asmCodeLine;
    bool isNOP = true;
    for (int i = 0; i < nRow; i++) {
      for (int j = 0; j < nCol; j++) {
        if (ops.find(i * nCol + j) != ops.end()) {
          std::string isa = printInstructionToISA(ops[i * nCol + j]);
          isNOP = (isa != "NOP") ? false : isNOP;
          asmCodeLine.push_back(isa);
        } else {
          asmCodeLine.push_back("NOP");
        }
      }
    }
    if (!isNOP)
      asmCode.push_back(asmCodeLine);
    // If the time step is empty, skip it and revise the base time
    else {
      dropTimeFrame(t);
      endTime = getKernelEnd();
      t--;
    }
  }

  // Write the schedule to the file
  if (outDir.empty())
    return;

  std::string gridOut = outDir + "_grid.sat";
  outDir = outDir + ".sat";

  std::ofstream file(outDir);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file\n";
    return;
  }

  // Print the instruction for each time step
  for (int t = 0; t < asmCode.size(); t++) {
    file << "T = " << startPC + t << "\n";
    for (int i = 0; i < asmCode[t].size(); i++) {
      file << padString(asmCode[t][i], 25);
      file << "\n";
    }
  }

  // Write the grid-like schedule
  if (!GridLIke)
    return;

  std::ofstream gridFile(gridOut);
  for (int t = 0; t < asmCode.size(); t++) {
    gridFile << "Time = " << startPC + t << "\n";
    for (int i = 0; i < asmCode[t].size(); i++) {
      gridFile << padString(asmCode[t][i], 25);
      if (i % nCol == nCol - 1)
        gridFile << "\n";
    }
  }
}

/// Function to parse the scheduled results produced by SAT-MapIt line by
/// line and store the instruction in the map.
static LogicalResult
readMapFile(std::string mapResult, unsigned maxReg, unsigned numOps,
            std::map<int, std::unordered_set<int>> &opTimeMap,
            std::vector<std::unordered_set<int>> &bbTimeMap,
            std::map<int, Instruction> &instructions) {
  std::ifstream file(mapResult);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open file\n";
    return failure();
  }

  std::string line;

  bool parsing = false;
  bool cfgParse = false;

  // Read each line and parse it into the map
  while (std::getline(file, line)) {
    parsing = false;
    // find PKE, parse the modulo schedule
    if (line.find("PKE") != std::string::npos) {
      cfgParse = true;
      continue;
    }
    // end of the modulo schedule result
    if (cfgParse)
      if (line.find("t:") == std::string::npos)
        cfgParse = false;

    if (line.find("Id:") != std::string::npos) {
      parsing = true;
    }

    if (cfgParse)
      satmapit::parsePKE(line, numOps, bbTimeMap, opTimeMap);

    if (parsing)
      satmapit::parseLine(line, instructions, maxReg);
  }

  return success();
}

/// Initialize the block arguments to be SADD
static LogicalResult initBlockArgs(Block *block,
                                   std::map<int, Instruction> &instructions,
                                   OpBuilder &builder) {
  // Get the phi nodes in the block
  int nPhi = 0;
  for (auto [index, inst] : instructions) {
    if (inst.name != "phi")
      break;
    nPhi++;
  }

  if (nPhi != block->getNumArguments())
    return failure();

  Block *prevNode = block->getPrevNode();
  // Initialize the block arguments to be SADD, from the last to the first
  // to keep the index consistent
  for (int i = nPhi - 1; i >= 0; i--) {
    // insert constant Zero and SADD
    builder.setInsertionPoint(prevNode->getTerminator());
    auto zeroOp = builder.create<LLVM::ConstantOp>(
        prevNode->getTerminator()->getLoc(), builder.getIntegerType(32),
        builder.getI32IntegerAttr(0));

    builder.setInsertionPointToStart(block);
    auto addOp = builder.create<LLVM::AddOp>(
        block->getArgument(i).getLoc(), builder.getIntegerType(32),
        block->getArgument(i), zeroOp.getResult());
    // Replace the block argument with the add operation
    block->getArgument(i).replaceUsesWithIf(
        addOp.getResult(),
        [&](OpOperand &operand) { return operand.getOwner() != addOp; });
    // Revise the instruction map
    instructions[i].name = "add";
  }
  return success();
}

/// Determine whether the loop execution time of operations belonging to
/// different loop overlapped. `bbTimeMap` records the time of prolog,
/// loop kernel, and epilog. If the end of the `bbTimeMap` which is the
/// epilog are empty, the loop execution time does not overlap.
static bool kernelOverlap(std::vector<std::unordered_set<int>> bbTimeMap) {
  if (bbTimeMap.empty())
    return false;
  return !bbTimeMap.back().empty();
}

namespace {
struct OpenEdgeASMGenPass
    : public compigra::impl::OpenEdgeASMGenBase<OpenEdgeASMGenPass> {

  explicit OpenEdgeASMGenPass(StringRef funcName, StringRef mapResult) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    unsigned maxReg = 3;

    llvm::errs() << mapResult << "\n";
    size_t pos = mapResult.find_last_of("/");
    // Default output file directory
    std::string outDir = mapResult.substr(0, pos + 1) + "out";

    for (auto funcOp :
         llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>())) {
      if (funcOp.getName() != funcName)
        continue;
      Region &r = funcOp.getBody();
      auto loopBlock = getLoopBlock(r);
      int opSize = loopBlock->getOperations().size();

      // read the loop basic block modulo schedule result
      std::map<int, Instruction> instructions;
      std::map<int, std::unordered_set<int>> opTimeMap;
      std::vector<std::unordered_set<int>> bbTimeMap = {{}};
      if (failed(readMapFile(mapResult, maxReg,
                             opSize + loopBlock->getNumArguments() - 1,
                             opTimeMap, bbTimeMap, instructions)))
        return signalPassFailure();
      // init block arguments to be SADD
      if (failed(initBlockArgs(loopBlock, instructions, builder)))
        return signalPassFailure();
      // init modulo schedule result
      if (kernelOverlap(bbTimeMap))
        adaptCFGWithLoopMS(r, builder, opTimeMap, bbTimeMap);
      llvm::errs() << funcOp << "\n";

      // init OpenEdgeASMGen
      // OpenEdgeASMGen asmGen(r, maxReg, 4);
      // // init scheduler
      // OpenEdgeKernelScheduler scheduler(r, maxReg, 4);
      // scheduler.assignSchedule(loopBlock->getOperations(),
      // instructions); if (failed(scheduler.createSchedulerAndSolve())) {
      //   llvm::errs() << "Failed to create scheduler and solve\n";
      //   return signalPassFailure();
      // }

      // // assign schedule results and produce assembly
      // asmGen.setSolution(scheduler.getSolution());
      // llvm::errs() << "Allocate Register...\n";
      // asmGen.allocateRegisters(scheduler.knownRes);
      // asmGen.printKnownSchedule(true, 0, outDir);
    }
    // Assign operations in the init phase
  }
};
} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createOpenEdgeASMGen(StringRef funcName,
                                                 StringRef mapResult) {
  return std::make_unique<OpenEdgeASMGenPass>(funcName, mapResult);
}
} // namespace compigra