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

size_t satmapit::getNodeIndex(Operation *op, SmallVector<Operation *> &nodes,
                              SmallVector<LLVM::ConstantOp> constants,
                              SmallVector<Operation *> liveIns) {
  // first seek the constant value
  size_t constBase = nodes.size() + 10;

  for (auto [ind, constOp] : llvm::enumerate(constants))
    if (op == constOp)
      return ind + constBase;

  // seek the live-in operation
  size_t liveInBase = constBase + constants.size();
  for (auto [ind, liveIn] : llvm::enumerate(liveIns))
    if (op == liveIn)
      return ind + liveInBase;

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

  std::string liveInNode = fileName + "_liveinnodes";
  std::string liveInEdge = fileName + "_livein_edges";

  std::string liveOutNode = fileName + "_liveoutnodes";
  std::string liveOutEdge = fileName + "_liveout_edges";
  if (failed(printNodes(nodeFile)) || failed(printConsts(constFile)) ||
      failed(printEdges(edgeFile)))
    return failure();

  return success();
}

LogicalResult PrintSatMapItDAG::printNodes(std::string nodeFile) {

  unsigned nodeCount = nodes.size();

  std::ofstream dotFile;
  dotFile.open(nodeFile.c_str());
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

LogicalResult PrintSatMapItDAG::printConsts(std::string constFile) {
  std::ofstream dotFile;
  dotFile.open(constFile.c_str());
  for (auto [ind, constOp] : llvm::enumerate(constants)) {
    // The constant should only have one user
    if (std::distance(constOp->getUsers().begin(), constOp->getUsers().end()) >
        1) {
      llvm::errs() << "const " << constOp << " has more than one user\n";
      return failure();
    }

    llvm::errs() << "const " << constOp;
    // get the one and only one user of the constant
    auto user = *constOp->getUsers().begin();
    size_t userInd = getNodeIndex(user, nodes);
    llvm::errs() << "---->user " << *user << "\n";

    if (userInd == unsigned(-1))
      return failure();

    // TODO: check the type of the constant
    // currently only support integer
    int constVal = constOp.getValueAttr().cast<IntegerAttr>().getInt();
    llvm::errs() << getNodeIndex(constOp, nodes, constants) << " " << userInd;
    llvm::errs() << " " << constVal << " 1 \n";
    // dotFile << getNodeIndex(op, nodes, constants) << " " << userInd;
    // dotFile << " " << constVal << "1 \n";
  }
  dotFile.close();

  return success();
}

LogicalResult PrintSatMapItDAG::printEdges(std::string edgeFile) {
  return success();
}
