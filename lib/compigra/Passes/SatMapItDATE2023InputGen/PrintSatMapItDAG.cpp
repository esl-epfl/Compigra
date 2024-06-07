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

#include "compigra/Passes/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

using namespace mlir;
using namespace compigra;
using namespace compigra::satmapit;

// Get the stage of the operation for scheduling
static std::string getOpStage(Operation *op) {
  if (auto stageAttr = op->getAttrOfType<StringAttr>("stage"))
    return stageAttr.getValue().str();
  return "";
}

// The edge in the DAG is a backedge if the destination operation is a merge,
// and both the srcOp and dstOp are in the loop stage. The dstcOp receives its
// opearnds from the latter execution of srcOp.
static bool isBackEdge(Operation *srcOp, Operation *dstOp) {
  if (getOpStage(srcOp) == "loop" && (getOpStage(dstOp) == "loop"))
    return isa<cgra::MergeLikeOpInterface>(dstOp);
  return false;
}

size_t satmapit::getNodeIndex(Operation *op, SmallVector<Operation *> &nodes,
                              SmallVector<LLVM::ConstantOp> constants,
                              SmallVector<Operation *> liveIns,
                              SmallVector<Operation *> liveOuts) {
  // first seek the constant value
  size_t constBase = nodes.size() + 10;

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
      return ind;
  }
  return -1;
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

LogicalResult PrintSatMapItDAG::printNodes(std::string fileName) {

  unsigned nodeCount = nodes.size();

  std::ofstream dotFile;
  dotFile.open(fileName.c_str());
  for (auto [ind, node] : llvm::enumerate(nodes)) {
    size_t namePos = node->getName().getStringRef().str().find(".");
    std::string nodeName =
        node->getName().getStringRef().str().substr(namePos + 1);

    // get the operator source
    // check whether the operation is an left operand and right operand
    if (node->getNumOperands() < 2) {
      llvm::errs() << node->getName() << " has less than two operands\n";
      return failure();
    }

    int predicateSel = -1;
    Operation *leftOp, *rightOp;
    if (auto bzfaOp = dyn_cast<cgra::BzfaOp>(node)) {
      predicateSel = getNodeIndex(bzfaOp.getPredicate().getDefiningOp(), nodes);
      dotFile << " " << predicateSel;
      leftOp = node->getOperand(1).getDefiningOp();
      rightOp = node->getOperand(2).getDefiningOp();
    } else {
      leftOp = node->getOperand(0).getDefiningOp();
      rightOp = node->getOperand(1).getDefiningOp();
    }

    dotFile << std::to_string(ind) << " " << nodeName << " "
            << getNodeIndex(leftOp, nodes, constants, liveIns) << " "
            << getNodeIndex(rightOp, nodes, constants, liveIns) << " "
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

    size_t userInd = getNodeIndex(user, nodes);

    if (userInd == uint32_t(-1))
      return failure();

    // TODO: check the type of the constant
    // currently only support integer
    int constVal = constOp.getValueAttr().cast<IntegerAttr>().getInt();

    dotFile << getNodeIndex(constOp, nodes, constants) << " " << userInd;
    dotFile << " " << constVal << " " << posLR << "\n";
  }
  dotFile.close();

  return success();
}

LogicalResult PrintSatMapItDAG::printEdges(std::string fileName) {
  std::ofstream dotFile;
  dotFile.open(fileName.c_str());
  for (auto *node : nodes) {
    for (auto user : node->getUsers()) {
      if (getOpStage(user) != "loop")
        continue;

      // ignore the cgra branch operation's destination
      if (isa<cgra::BeqOp, cgra::BneOp>(user) &&
          user->getOperand(2).getDefiningOp() == node)
        continue;

      unsigned distance = isBackEdge(node, user) ? 1 : 0;
      dotFile << " " << getNodeIndex(node, nodes, constants, liveIns) << " ";
      dotFile << getNodeIndex(user, nodes, constants, liveIns) << " "
              << distance << " 1\n";
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
    dotFile << " " << getNodeIndex(liveIn, nodes, constants, liveIns) << "\n";
  dotFile.close();

  // print live-in edges to the text file
  dotFile.open(inEdgeFile.c_str());
  for (auto liveIn : liveIns) {
    for (auto user : liveIn->getUsers()) {
      if (getOpStage(user) != "loop")
        continue;
      dotFile << " " << getNodeIndex(liveIn, nodes, constants, liveIns) << " ";
      dotFile << getNodeIndex(user, nodes, constants, liveIns) << " 0 1\n";
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
  for (auto liveOut : liveOuts)
    dotFile << " " << getNodeIndex(liveOut, nodes, constants, liveIns, liveOuts)
            << "\n";
  dotFile.close();

  // print live-out edges to the text file
  dotFile.open(outEdgeFile.c_str());
  for (Operation *liveOut : liveOuts) {
    for (Value operand : liveOut->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (getOpStage(defOp) != "loop")
        continue;
      dotFile << " " << getNodeIndex(defOp, nodes, constants, liveIns) << " "
              << getNodeIndex(liveOut, nodes, constants, liveIns, liveOuts)
              << " 0 1\n";
    }
  }
  dotFile.close();

  return success();
}
