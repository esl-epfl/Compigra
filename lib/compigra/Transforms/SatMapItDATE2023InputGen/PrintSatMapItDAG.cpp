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

/// Get the specific operand of the conditional branch operation. `block` is the
/// one of the successor blocks to get connected operand of the conditional
/// branch operation. If block is the true branch, then the operand is the
/// index-th of trueDestOperands. Otherwise, the operand is the index-th of the
/// falseDestOperands.
template <typename T>
static Value getCondBranchOperand(unsigned ind, T *termOp, Block *block) {
  bool targetBlk = termOp->getTrueDest() == block;

  if (targetBlk)
    return termOp->getTrueDestOperands()[ind];
  else
    return termOp->getFalseDestOperands()[ind];
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
  for (auto [ind, node] : llvm::enumerate(nodes))
    if (op == node)
      return ind + blockArg;

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

/// Get the corresponding value of the block argument.
static Value getCntBlockArgInPredcessor(unsigned ind, Block *pred,
                                        Block *block) {
  auto termOp = pred->getTerminator();
  if (auto brOp = dyn_cast<LLVM::BrOp>(termOp))
    return brOp->getOperand(ind);
  if (auto beqOp = dyn_cast<cgra::BeqOp>(termOp))
    return getCondBranchOperand<cgra::BeqOp>(ind, &beqOp, block);
  if (auto bneOp = dyn_cast<cgra::BneOp>(termOp))
    return getCondBranchOperand<cgra::BneOp>(ind, &bneOp, block);
  if (auto bneOp = dyn_cast<cgra::BgeOp>(termOp))
    return getCondBranchOperand<cgra::BgeOp>(ind, &bneOp, block);
  if (auto bneOp = dyn_cast<cgra::BltOp>(termOp))
    return getCondBranchOperand<cgra::BltOp>(ind, &bneOp, block);
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
  // The liveOut arguments are the false branch arguments

  if (auto beqOp = dyn_cast<cgra::BeqOp>(terminator)) {
    getFalseDestOperands<cgra::BeqOp>(beqOp, liveOutArgs);
  } else if (auto bneOp = dyn_cast<cgra::BneOp>(terminator)) {
    getFalseDestOperands<cgra::BneOp>(bneOp, liveOutArgs);
  } else if (auto bltOp = dyn_cast<cgra::BltOp>(terminator)) {
    getFalseDestOperands<cgra::BltOp>(bltOp, liveOutArgs);
  } else if (auto bgeOp = dyn_cast<cgra::BgeOp>(terminator)) {
    getFalseDestOperands<cgra::BgeOp>(bgeOp, liveOutArgs);
  }

  // init constant, liveIn, and liveOut operations
  for (auto [ind, arg] : llvm::enumerate(BlockArgs)) {
    // Loop block should have two predecessors
    if (getPredecessorCount(loopBlock) != 2)
      return failure();
    SmallVector<Value, 2> parameters;

    for (auto pred : loopBlock->getPredecessors()) {
      // Get the value that is passed to the loop block
      Value corrArg = pred->getTerminator()->getOperand(ind);
      if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(
              pred->getTerminator()))
        corrArg = getCntBlockArgInPredcessor(ind, pred, loopBlock);
      parameters.push_back(corrArg);
      auto defOp = corrArg.getDefiningOp();
      if (defOp) {
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
    int leftOpInd, rightOpInd;
    Value leftOp = ops[0];
    if (auto defOp = leftOp.getDefiningOp()) {
      leftOpInd = getNodeIndex(defOp);
    } else if (auto blockArg = leftOp.dyn_cast<BlockArgument>()) {
      // The leftOp could be a block argument
      leftOpInd = blockArg.getArgNumber();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "The left operand is not defined\n");
      return failure();
    }

    Value rightOp = ops[1];
    if (auto defOp = rightOp.getDefiningOp()) {
      rightOpInd = getNodeIndex(defOp);
    } else if (auto blockArg = rightOp.dyn_cast<BlockArgument>()) {
      // The rightOp could be a block argument
      rightOpInd = blockArg.getArgNumber();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "The right operand is not defined\n");
      return failure();
    }

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
    for (auto &use : arg.getUses()) {

      auto user = use.getOwner();
      // no need to handle the liveOut arguments
      if (user->getBlock() != loopBlock)
        continue;
      int userInd = -1;
      // branch operator is propagated to the successor block
      if (auto brOp = dyn_cast<LLVM::BrOp>(user)) {
        if (brOp->getBlock()->getSuccessor(0) == loopBlock)
          userInd = use.getOperandNumber();
      } else if (use.getOperandNumber() > 1 &&
                 isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(
                     user)) {
        // if it is propagated to the successor block through bne, beq, blt, bge
        if (user->getBlock()->getSuccessor(0) == loopBlock) {
          // LLVM to CGRA conversion should adapt the loopblock to be the first
          // block successor
          userInd = use.getOperandNumber() - 2;
        }
      } else {
        userInd = getNodeIndex(user);
      }

      // if it can seek the user index
      if (userInd == -1)
        return failure();

      dotFile << getNodeIndex(arg) << " ";
      dotFile << userInd << " 0 1\n";
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
          dotFile << use.getOperandNumber() - 2 << " 1 1\n";
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
    for (auto &use : liveIn->getUses()) {
      auto user = use.getOwner();
      // the use could be used in the loop block in two ways:
      // 1. the user belongs to the loop block
      // 2. the user is propagated to the successor block, where the user
      // receives the value from the basic block argument
      bool inUse = user->getBlock() == loopBlock;
      inUse = inUse || (isa<LLVM::BrOp>(user) &&
                        user->getBlock()->getSuccessor(0) == loopBlock);
      if (!inUse)
        continue;
      dotFile << getNodeIndex(liveIn) << " ";

      int userInd = -1;
      if (auto brOp = dyn_cast<LLVM::BrOp>(user)) {
        if (brOp->getBlock()->getSuccessor(0) == loopBlock)
          userInd = use.getOperandNumber();
      } else if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(
                     user) &&
                 use.getOperandNumber() > 1) {
        // if it is propagated to the successor block through bne, beq, blt, bge
        if (user->getBlock()->getSuccessor(0) == loopBlock) {
          // LLVM to CGRA conversion should adapt the loopblock to be the first
          // block successor
          Value arg = getCntBlockArgInPredcessor(use.getOperandNumber() - 2,
                                                 loopBlock, loopBlock);
          userInd = use.getOperandNumber() - 2;
        }
      } else {
        userInd = getNodeIndex(user);
      }

      if (userInd == -1)
        return failure();
      dotFile << userInd << " 0 1\n";
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
  dotFile.close();

  // // print live-out edges to the text file
  dotFile.open(outEdgeFile.c_str());
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

/// Function to split a string by whitespace
static std::unordered_set<int> getOpIdSplit(std::istringstream &stream) {
  std::unordered_set<int> tokens;

  int number;
  while (stream >> number) {
    llvm::errs() << number << " ";
    tokens.insert(number);
  }
  return tokens;
}

void satmapit::parsePKE(const std::string &line, unsigned termId,
                        std::vector<std::unordered_set<int>> &bbTimeMap,
                        std::map<int, std::unordered_set<int>> &opTimeMap) {
  std::istringstream lineStream(line);
  // first parse t: time
  std::string token, tStr;
  // Read the first part (t: t)
  std::getline(lineStream, token, ' ');
  std::getline(lineStream, token, ' ');
  int tVal = std::stoi(token.substr(token.find(":") + 1));
  bbTimeMap.back().insert(tVal);

  // Initialize the set for the values
  std::unordered_set<int> values;

  // Read the remaining parts (values)
  while (std::getline(lineStream, token, ' ')) {
    if (!token.empty()) {
      values.insert(std::stoi(token));
      if (std::stoi(token) == termId)
        // push back a new set for the new basic block
        bbTimeMap.push_back({});
    }
  }
  opTimeMap[tVal] = values;
}

void satmapit::parseLine(const std::string &line,
                         std::map<int, Instruction> &instMap,
                         const unsigned maxReg) {

  std::istringstream lineStream(line);
  std::string idStr, nameStr, timeStr, peStr, RoutStr, opAStr, opBStr, immStr;
  std::string idVal, nameVal, timeVal, peVal, RoutVal, opAVal, opBVal, immVal;
  // Read Id token
  std::getline(lineStream, idStr, ' ');
  std::getline(lineStream, idVal, ' ');
  int id = std::stoi(idVal.substr(idVal.find(":") + 1));

  // Read name
  std::getline(lineStream, nameStr, ' ');
  std::getline(lineStream, nameVal, ' ');
  nameVal = nameVal.substr(nameVal.find(":") + 1);

  // Read time
  std::getline(lineStream, timeStr, ' ');
  std::getline(lineStream, timeVal, ' ');
  timeVal = timeVal.substr(timeVal.find(":") + 1);

  // Read pe
  std::getline(lineStream, peStr, ' ');
  std::getline(lineStream, peVal, ' ');
  peVal = peVal.substr(peVal.find(":") + 1);

  // Read Rout
  std::getline(lineStream, RoutStr, ' ');
  std::getline(lineStream, RoutVal, ' ');
  RoutVal = RoutVal.substr(RoutVal.find(":") + 1);
  // if RoutVal = Ri:i, else RoutVal = Rout(mexReg)
  // Find substr after R, if it is out, then it is Rout, else it is Ri
  auto reg = RoutVal.substr(RoutVal.find("R") + 1);
  RoutVal = reg == "OUT" ? std::to_string(maxReg) : reg;

  // Read opA
  std::getline(lineStream, opAStr, ' ');
  std::getline(lineStream, opAVal, ' ');
  opAVal = opAVal.substr(opAVal.find(":") + 1);

  // Read opB
  std::getline(lineStream, opBStr, ' ');
  std::getline(lineStream, opBVal, ' ');
  opBVal = opBVal.substr(opBVal.find(":") + 1);

  // Read immediate
  std::getline(lineStream, immStr, ' ');
  std::getline(lineStream, immVal, ' ');
  immVal = immVal.substr(immVal.find(":") + 1);

  // Create and insert the Inst object
  Instruction inst;
  inst.name = nameVal;
  inst.time = std::stoi(timeVal);
  inst.pe = std::stoi(peVal);
  inst.Rout = std::stoi(RoutVal);
  inst.opA = opAVal;
  inst.opB = opBVal;

  instMap[id] = inst;
}