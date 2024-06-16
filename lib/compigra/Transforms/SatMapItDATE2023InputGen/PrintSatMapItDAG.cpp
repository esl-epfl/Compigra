//===- PrintSatMapItDAG.cpp - print text file for SatMapIt ------*- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the text printout functions for Sat-MapIt code base.
//
//===----------------------------------------------------------------------===//

#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

#define DEBUG_TYPE "PRINT_SATMAPIT_DAG"

using namespace mlir;
using namespace compigra;
using namespace compigra::satmapit;

static unsigned getPredecessorCount(Block *blk) {
  return std::distance(blk->getPredecessors().begin(),
                       blk->getPredecessors().end());
}

// The edge in the DAG is a backedge if the destination operation is a merge,
// and both the srcOp and dstOp are in the loop stage. The dstcOp receives its
// opearnds from the latter execution of srcOp.
static bool isBackEdge(Operation *srcOp, Operation *dstOp) {
  //   return isa<cgra::MergeLikeOpInterface>(dstOp);
  return false;
}

template <typename T>
static void getFalseDestOperands(T branchOp,
                                 SmallVector<Value> &falseDestOperands) {
  // The true branch arguments are the operands of the true branch
  falseDestOperands.clear();
  falseDestOperands.append(branchOp.getFalseDestOperands().begin(),
                           branchOp.getFalseDestOperands().end());
}

template <typename T>
static void getTrueDestOperands(T branchOp,
                                SmallVector<Value> &trueDestOperands) {
  // The true branch arguments are the operands of the true branch
  trueDestOperands.clear();
  trueDestOperands.append(branchOp.getTrueDestOperands().begin(),
                          branchOp.getTrueDestOperands().end());
}

template <typename T>
static Operation *getDefinitionOpOfArg(unsigned ind, T *term, Block *block) {
  bool targetBlk = term->getTrueDest() == block;

  if (targetBlk)
    return term->getTrueDestOperands()[ind].getDefiningOp();
  else
    return term->getFalseDestOperands()[ind].getDefiningOp();
}

int PrintSatMapItDAG::getNodeIndex(Operation *op) {
  // seek the constant value
  size_t constBase = blockArg + nodes.size() + 10;
  for (auto [ind, constOp] : llvm::enumerate(constants))
    if (op == constOp)
      return ind + constBase;

  // seek the live-in operation
  size_t liveInBase = constBase + constants.size() + 10;
  for (auto [ind, liveIn] : llvm::enumerate(liveIns))
    if (op == liveIn)
      return ind * 10 + liveInBase + 1;

  // seek the live-out operation
  size_t liveOutBase = liveInBase + liveIns.size() + 10;
  for (auto [ind, liveOut] : llvm::enumerate(liveOuts))
    if (op == liveOut)
      return ind * 10 + liveOutBase + 1;

  // seek the operation
  for (auto [ind, node] : llvm::enumerate(nodes)) {
    if (op == node)
      return ind + blockArg;
  }
  return -1;
}

int PrintSatMapItDAG::getNodeIndex(Value val) {
  if (auto op = val.getDefiningOp())
    return getNodeIndex(op);
  // check whether the value is a block argument
  for (auto [ind, arg] : llvm::enumerate(BlockArgs))
    if (val == arg)
      return ind;

  return -1;
}

void PrintSatMapItDAG::addNodes(Operation *op) {
  for (auto node : nodes)
    if (op == node)
      return;

  for (auto node : constants)
    if (op == node)
      return;

  for (auto node : liveIns)
    if (op == node)
      return;

  // not find in existed node sets
  // if it is a constant operaiton, add it into constant
  if (auto constOp = dyn_cast<LLVM::ConstantOp>(op)) {
    constants.push_back(constOp);
    return;
  }

  // not find in existed node sets, add it into liveIn
  if (op->getBlock() == initBlock) {
    liveIns.push_back(op);
    return;
  }
}

static Operation *getDefinitionOpOfBlockArg(unsigned ind, Block *pred,
                                            Block *block) {
  auto termOp = pred->getTerminator();
  if (auto beqOp = dyn_cast<cgra::BeqOp>(termOp))
    return getDefinitionOpOfArg<cgra::BeqOp>(ind, &beqOp, block);
  if (auto bneOp = dyn_cast<cgra::BneOp>(termOp))
    return getDefinitionOpOfArg<cgra::BneOp>(ind, &bneOp, block);
  if (auto bneOp = dyn_cast<cgra::BgeOp>(termOp))
    return getDefinitionOpOfArg<cgra::BgeOp>(ind, &bneOp, block);
  if (auto bneOp = dyn_cast<cgra::BltOp>(termOp))
    return getDefinitionOpOfArg<cgra::BltOp>(ind, &bneOp, block);
  return nullptr;
}

LogicalResult PrintSatMapItDAG::init() {
  // init the loop block and the predecessor block
  initLoopBlock();
  initPredBlock();
  blockArg = loopBlock->getNumArguments();
  //  Get the LiveIn and LiveOut arguments

  BlockArgs.append(loopBlock->getArguments().begin(),
                   loopBlock->getArguments().end());
  llvm::errs() << "The number of liveIn arguments: " << BlockArgs.size()
               << "\n";
  // The liveOut arguments are the true branch arguments

  if (auto beqOp = dyn_cast<cgra::BeqOp>(terminator)) {
    getFalseDestOperands<cgra::BeqOp>(beqOp, liveOutArgs);
  } else if (auto bneOp = dyn_cast<cgra::BneOp>(terminator)) {
    getFalseDestOperands<cgra::BneOp>(bneOp, liveOutArgs);
  } else if (auto bltOp = dyn_cast<cgra::BltOp>(terminator)) {
    getFalseDestOperands<cgra::BltOp>(bltOp, liveOutArgs);
  } else if (auto bgeOp = dyn_cast<cgra::BgeOp>(terminator)) {
    getFalseDestOperands<cgra::BgeOp>(bgeOp, liveOutArgs);
  }
  llvm::errs() << "The number of liveOut arguments: " << liveOutArgs.size()
               << "\n";

  // init constant, liveIn, and liveOut operations
  for (auto [ind, arg] : llvm::enumerate(BlockArgs)) {
    // Loop block should have two predecessors
    if (getPredecessorCount(loopBlock) != 2)
      return failure();
    SmallVector<Operation *, 2> parameters;
    for (auto pred : loopBlock->getPredecessors()) {
      // Get the value that is passed to the loop block
      if (pred->getSuccessors().size() == 1) {
        if (auto brOp = dyn_cast<LLVM::BrOp>(pred->getTerminator())) {
          auto defOp = brOp->getOperand(ind).getDefiningOp();
          llvm::errs() << *defOp << "\n";
          parameters.push_back(defOp);
          addNodes(defOp);
        }
      } else {
        auto defOp = getDefinitionOpOfBlockArg(ind, loopBlock, pred);
        llvm::errs() << *defOp << "\n";
        parameters.push_back(defOp);
        addNodes(defOp);
      }
    }
    argMaps[ind] = parameters;
  }

  for (auto [ind, node] : llvm::enumerate(nodes)) {
    for (auto Operand : node->getOperands()) {
      if (auto defOp = Operand.getDefiningOp())
        addNodes(defOp);
    }
  }
  return success();
}

// static

LogicalResult PrintSatMapItDAG::printNodes(std::string fileName) {
  std::ofstream dotFile;
  dotFile.open(fileName.c_str());

  // SmallVector
  // print the block arguments to be merge node
  for (auto [ind, argPair] : llvm::enumerate(argMaps)) {
    std::string nodeName = "phi";
    auto ops = argPair.second;

    int predicateSel = -1;
    int leftOpInd = getNodeIndex(ops[0]);
    int rightOpInd = getNodeIndex(ops[1]);

    dotFile << std::to_string(ind) << " " << nodeName << " " << leftOpInd << " "
            << rightOpInd << " " << std::to_string(predicateSel);

    dotFile << " " << std::to_string(CgraInsts[nodeName]) << "\n";
  }

  for (auto [ind, node] : llvm::enumerate(nodes)) {
    size_t namePos = node->getName().getStringRef().str().find(".");
    std::string nodeName =
        node->getName().getStringRef().str().substr(namePos + 1);

    int predicateSel = -1;
    int leftOpInd = -1;
    int rightOpInd = -1;
    if (isa<cgra::BzfaOp, cgra::BsfaOp>(node)) {
      // get the predicate for selection
      predicateSel = getNodeIndex(node->getOperand(0));

      leftOpInd = getNodeIndex(node->getOperand(1));
      rightOpInd = getNodeIndex(node->getOperand(2));
    } else {
      // get the operator source
      // check whether the operation is an left operand and right operand
      if (node->getNumOperands() > 2) {
        LLVM_DEBUG(llvm::dbgs()
                   << node->getName() << " has more than two operands\n");
      }

      for (auto [ind, opr] : llvm::enumerate(node->getOperands())) {
        if (ind == 0)
          leftOpInd = getNodeIndex(opr);
        if (ind == 1)
          rightOpInd = getNodeIndex(opr);
      }
    }

    dotFile << std::to_string(ind + blockArg) << " " << nodeName << " "
            << leftOpInd << " " << rightOpInd << " "
            << std::to_string(predicateSel);

    dotFile << " " << std::to_string(CgraInsts[nodeName]) << "\n";
  }
  dotFile.close();

  return success();
}

LogicalResult PrintSatMapItDAG::printConsts(std::string fileName) {
  std::ofstream dotFile;
  dotFile.open(fileName.c_str());
  for (auto [ind, constOp] : llvm::enumerate(constants)) {
    // The constant should only have one user
    if (std::distance(constOp->getUsers().begin(), constOp->getUsers().end()) >
        1) {
      llvm::errs() << "const " << constOp << " has more than one user\n";
      return failure();
    }

    // get the one and only one user of the constant
    auto user = *constOp->getUsers().begin();
    unsigned posLR = user->getOperand(0).getDefiningOp() == constOp ? 0 : 1;

    size_t userInd = getNodeIndex(user);

    if (userInd == uint32_t(-1))
      return failure();

    // TODO: check the type of the constant
    // currently only support integer
    int constVal = constOp.getValueAttr().cast<IntegerAttr>().getInt();

    dotFile << getNodeIndex(constOp.getOperation()) << " " << userInd;
    dotFile << " " << constVal << " " << posLR << "\n";
  }
  dotFile.close();

  return success();
}

LogicalResult PrintSatMapItDAG::printEdges(std::string fileName) {
  std::ofstream dotFile;
  dotFile.open(fileName.c_str());
  for (auto arg : BlockArgs) {
    for (auto user : arg.getUsers()) {
      dotFile << getNodeIndex(arg) << " ";
      dotFile << getNodeIndex(user) << " 0 1\n";
    }
  }

  for (auto *node : nodes) {
    if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(node))
      continue;
    for (auto &use : node->getUses()) {
      // ignore the cgra branch operation's destination
      auto user = use.getOwner();
      // if user does not belong to loop stage, it should be live-out
      if (user->getBlock() != loopBlock)
        continue;
      if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(user))
        // If it is for operand propagation through, where the two operands of
        // branch operations are for comparison flag generation
        if (use.getOperandNumber() > 1) {
          dotFile << getNodeIndex(node) << " ";
          dotFile << getNodeIndex(user->getOperand(use.getOperandNumber()))
                  << " 1 1\n";
          continue;
        }

      dotFile << getNodeIndex(node) << " ";
      dotFile << getNodeIndex(user) << " 0 1\n";
    }
  }

  dotFile.close();
  return success();
}

LogicalResult PrintSatMapItDAG::printLiveIns(std::string fileName) {

  std::string inNodeFile = fileName + "nodes";
  std::string inEdgeFile = fileName + "_edges";

  // print the live-in nodes to text file
  std::ofstream dotFile;
  dotFile.open(inNodeFile.c_str());
  for (auto liveIn : liveIns)
    dotFile << getNodeIndex(liveIn) << "\n";
  dotFile.close();

  // print live-in edges to the text file
  dotFile.open(inEdgeFile.c_str());
  for (auto liveIn : liveIns) {
    for (auto user : liveIn->getUsers()) {
      if (user->getBlock() != loopBlock)
        continue;
      dotFile << getNodeIndex(liveIn) << " ";
      dotFile << getNodeIndex(user) << " 0 1\n";
    }
  }
  dotFile.close();

  return success();
}

LogicalResult PrintSatMapItDAG::printLiveOuts(std::string fileName) {
  std::string outNodeFile = fileName + "nodes";
  std::string outEdgeFile = fileName + "_edges";

  // print the live-out nodes to text file
  std::ofstream dotFile;
  dotFile.open(outNodeFile.c_str());
  // for (auto liveOut : liveOuts)
  //   dotFile << getNodeIndex(liveOut, nodes, constants, liveIns, liveOuts)
  //           << "\n";
  // dotFile.close();

  // // print live-out edges to the text file
  // dotFile.open(outEdgeFile.c_str());
  // for (Operation *liveOut : liveOuts) {
  //   for (Value operand : liveOut->getOperands()) {
  //     auto defOp = operand.getDefiningOp();
  //     if (getOpStage(defOp) != "loop")
  //       continue;
  //     dotFile << getNodeIndex(defOp, nodes, constants, liveIns) << " "
  //             << getNodeIndex(liveOut, nodes, constants, liveIns, liveOuts)
  //             << " 0 1\n";
  //   }
  // }
  dotFile.close();

  return success();
}

LogicalResult PrintSatMapItDAG::printDAG(std::string fileName) {
  std::string nodeFile = fileName + "_nodes";
  std::string edgeFile = fileName + "_edges";
  std::string constFile = fileName + "_constants";

  std::string liveInFile = fileName + "_livein";
  std::string liveOutFile = fileName + "_liveout";
  if (failed(printNodes(nodeFile)) || failed(printConsts(constFile)) ||
      failed(printEdges(edgeFile)) || failed(printLiveIns(liveInFile)) ||
      failed(printLiveOuts(liveOutFile)))
    return failure();

  return success();
}