//===- TemporalCGRAScheduler.cpp - Implement the class/functions for 2D temporal
// spatial schedule for temporal CGRAs *- C++-*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements class for temporal CGRA schedule functions.
//
//===----------------------------------------------------------------------===//

#include "compigra/Scheduler/TemporalCGRAScheduler.h"
#include "fstream"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace compigra;

/// Insert a value to a set if it is not a constant. The constant value is not
/// considered as a live value.
static void insertToSet(Value val, SetVector<Value> &vec) {
  if (dyn_cast_or_null<arith::ConstantIntOp>(val.getDefiningOp()) ||
      dyn_cast_or_null<arith::ConstantFloatOp>(val.getDefiningOp()))
    return;
  vec.insert(val);
}

/// Update the liveOut set of `blk` by adding the liveIn value of its
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

static SmallVector<Block *, 4> getCntBlocksThroughPhi(Value val) {
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

static BlockArgument getCntBlockArgument(Value val, Block *succBlk) {
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

static SmallVector<Value, 2> getSrcOprandsOfPhi(BlockArgument arg) {
  SmallVector<Value, 2> srcOprands;
  Block *blk = arg.getOwner();
  unsigned argIndex = arg.getArgNumber();
  for (auto predBlk : blk->getPredecessors()) {
    Operation *termOp = predBlk->getTerminator();
    if (auto branchOp = dyn_cast_or_null<cf::BranchOp>(termOp)) {
      srcOprands.push_back(branchOp.getOperand(argIndex));
    } else if (auto branchOp =
                   dyn_cast_or_null<cgra::ConditionalBranchOp>(termOp)) {
      if (predBlk == branchOp.getSuccessor(0)) {
        srcOprands.push_back(branchOp.getTrueOperand(argIndex));
      } else {
        srcOprands.push_back(branchOp.getFalseOperand(argIndex));
      }
    }
  }
  return srcOprands;
}

void TemporalCGRAScheduler::computeLiveValue() {
  // compute def and use for each block
  std::map<Block *, SetVector<Value>> defMap;
  std::map<Block *, SetVector<Value>> useMap;

  for (auto &block : region) {
    SetVector<Value> def;
    SetVector<Value> use;
    // push all block arguments to use
    for (auto arg : block.getArguments())
      insertToSet(arg, use);

    for (auto &op : block.getOperations()) {
      for (auto res : op.getResults())
        insertToSet(res, def);

      for (auto opr : op.getOperands())
        //   branch argument is not a use
        insertToSet(opr, use);
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
          insertToSet(val, liveIn);

      for (auto succ : block.getSuccessors()) {
        // ====================TBRM====================
        // auto it = region.getBlocks().begin();
        // auto it_ptr = it;
        // std::advance(it_ptr, 4);
        // auto it_suc_ptr = it;
        // std::advance(it_suc_ptr, 5);
        // auto &dst_blk = *it_ptr;
        // auto &succ_blk = *it_suc_ptr;
        // ====================TBRM====================

        // add to succesor's liveOut
        for (auto val : liveIns[succ]) {
          // ====================TBRM====================
          //   if (&block == &dst_blk && (succ == &succ_blk) &&
          //       val.isa<BlockArgument>()) {
          //     for (auto [ind, bb] : llvm::enumerate(region))
          //       if (&bb == val.getParentBlock()) {
          //         llvm::errs() << "~~~~liveIn: " << ind << "\n";
          //         break;
          //       }
          //   }
          // ====================TBRM====================

          updateLiveOutBySuccessorLiveIn(val, &block, liveOut);
        }

        // ====================TBRM====================
        // if (&block == &dst_blk && (succ == &succ_blk)) {
        //   for (auto val : liveOut) {
        //     if (val.isa<BlockArgument>()) {
        //       for (auto [ind, bb] : llvm::enumerate(region))
        //         if (&bb == val.getParentBlock()) {
        //           llvm::errs() << "~~~ " << ind << " ";
        //           break;
        //         }
        //     }
        //     llvm::errs() << "~~~ " << val << "\n";
        //   }
        // }
        // ====================TBRM====================
      }
      if (liveIn != liveIns[&block] || liveOut != liveOuts[&block]) {
        liveIns[&block] = liveIn;
        liveOuts[&block] = liveOut;
        changed = true;
      }
    }
  }

  printBlockLiveValue("liveValue.txt");
}

void TemporalCGRAScheduler::printBlockLiveValue(std::string fileName) {
  // Open output file stream
  std::ofstream outFile(fileName);
  if (!outFile.is_open()) {
    llvm::errs() << "Error: Could not open file " << fileName << "\n";
    return;
  }

  unsigned blockNum = 0;
  // print liveIn and liveOut
  for (auto &block : region) {
    outFile << "Block: " << blockNum << "\n";
    outFile << "LiveIn: ";
    for (auto val : liveIns[&block]) {
      if (val.isa<BlockArgument>()) {
        for (auto [ind, bb] : llvm::enumerate(region))
          if (&bb == val.getParentBlock()) {
            outFile << ind << " ";
            break;
          }
      }
      std::string str;
      llvm::raw_string_ostream rso(str);
      rso << val;
      outFile << rso.str() << "\n";
    }
    outFile << "LiveOut: ";
    for (auto val : liveOuts[&block]) {
      if (val.isa<BlockArgument>()) {
        for (auto [ind, bb] : llvm::enumerate(region))
          if (&bb == val.getParentBlock()) {
            outFile << ind << " ";
            break;
          }
      }
      std::string str;
      llvm::raw_string_ostream rso(str);
      rso << val;
      outFile << rso.str() << "\n";
    }
    outFile << "\n";
    blockNum++;
  }

  outFile.close();
}

static bool visibleOutside(Value val) {
  if (!isPhiRelatedValue(val)) {
    std::vector<Operation *> users;
    Block *defBlock = val.getDefiningOp()->getBlock();
    for (auto user : val.getUsers()) {
      if (user->getBlock() == defBlock)
        continue;
      if (std::find(users.begin(), users.end(), user) == users.end())
        users.push_back(user);
    }
    if (users.size() <= 2)
      return false;
  }

  // if not phi related, check whether it is used outside the block only once
  return true;
}

void BasicBlockILPScheduler::setLiveValue(SetVector<Value> liveIn,
                                          SetVector<Value> liveOut) {
  // this->liveIn = liveIn;
  // this->liveOut = liveOut;
  liveOutExter.clear();
  liveOutInter.clear();

  // split liveOut to be liveOutInter and liveOutExter
  for (auto val : liveOut) {
    if (visibleOutside(val))
      liveOutExter.push_back({val, UINT32_MAX});
    else
      liveOutInter.push_back({val, UINT32_MAX});
  }
}

#ifdef HAVE_GUROBI
static void addUnequalVarConstr(GRBModel &model, GRBVar x, GRBVar y) {
  GRBVar diff = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
  GRBVar diffAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);

  model.addConstr(diff == x - y);
  model.addGenConstrAbs(diffAbs, diff);
  model.addConstr(diffAbs >= 1E-3);
}

static void addUnequalCstConstr(GRBModel &model, GRBVar x, int y) {
  GRBVar diff = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
  GRBVar diffAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);

  model.addConstr(diff == x - y);
  model.addGenConstrAbs(diffAbs, diff);
  model.addConstr(diffAbs >= 1E-3);
}

/// Block the PE use until it gets consumed by dstOp. PE can be specified by
/// variablesI(PE of srcOp) or unsigned constant `pe`. If in strict mode, any
/// operation cannot be assigned the PE of srcOp (including the dstOp).
///  PREREQUISITE: srcOp and dstOp should be in the same block.
static LogicalResult
blockPeAssignment(GRBModel &model, Operation *srcOp, Operation *dstOp,
                  const std::map<Operation *, GRBVar> opTimeVar,
                  const std::map<Operation *, GRBVar> opPeVar,
                  const std::map<Operation *, std::string> varName,
                  bool strict = false, int constPE = -1) {
  GRBVar x, startT;
  if (srcOp) {
    x = opPeVar.at(srcOp);
    startT = opTimeVar.at(srcOp);
  } else {
    GRBVar x_ = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
    GRBVar startT_ = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
    x = x_;
    startT = startT_;
    model.addConstr(x_ == constPE);
    model.addConstr(startT_ == 0);
  }

  auto &y = opPeVar.at(dstOp);
  auto endT = opTimeVar.at(dstOp);

  // if x ! = y, consumer and producer are not in the same PE
  // GRBVar diffXY = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
  // model.addConstr(diffXY == y - x);
  // GRBVar diffXYAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
  // model.addGenConstrAbs(diffXYAbs, diffXY);
  // GRBVar ifPEEq = model.addVar(0, 1, 0, GRB_BINARY);
  // model.addConstr(ifPEEq >= diffXYAbs / 1e6);

  for (auto [op, tVar] : opTimeVar) {
    if (op == srcOp)
      continue;

    if (!strict && op == dstOp)
      continue;

    GRBVar pe = opPeVar.at(op);
    // A helper variable to indicate the time gap between the producer and
    // the consumer, where helper = 1 means startT <= t <= endT.
    GRBVar h1 = model.addVar(0, 1, 0, GRB_BINARY);
    model.addConstr(tVar >= startT - 1e3 * (1 - h1));
    model.addConstr(tVar <= startT + 1e3 * h1 - 1e-2);

    GRBVar h2 = model.addVar(0, 1, 0, GRB_BINARY);
    model.addConstr(endT >= tVar - 1e3 * (1 - h2));

    if (strict) {
      model.addConstr(endT <= tVar + 1e3 * h2 - 1e-2);
    } else {
      model.addConstr(endT <= tVar + 1e3 * h2);
    }

    GRBVar h = model.addVar(0, 1, 0, GRB_BINARY);

    // if t_start<=t<=t_end && x!=y

    // if (strict) {
    //   GRBVar xvars[2] = {h1, h2};
    //   model.addGenConstrAnd(h, xvars, 2);
    // } else {
    //   GRBVar xvars[3] = {h1, h2, ifPEEq};
    //   model.addGenConstrAnd(h, xvars, 3);
    // }
    GRBVar xvars[2] = {h1, h2};
    model.addGenConstrAnd(h, xvars, 2);

    // Set constraints for pe != x if helper == 1
    GRBVar diff = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
    GRBVar diffAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);

    std::string srcOpName =
        srcOp ? varName.at(srcOp) : "PE" + std::to_string(constPE);
    model.addConstr(diff == pe - x, "diff" + varName.at(op) + "_" +
                                        varName.at(dstOp) + "_" + srcOpName);
    model.addGenConstrAbs(diffAbs, diff);

    model.addGenConstrIndicator(h, 1, diffAbs, GRB_GREATER_EQUAL, 1e-4,
                                "indicator_" + varName.at(dstOp) + "with" +
                                    srcOpName);
  }
  // check whether the model is feasible
  model.optimize();
  if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL ||
      model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL)
    return success();

  return failure();
}

LogicalResult BasicBlockILPScheduler::createLocalDominanceConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar) {

  // Consumer must be executed after the producer
  Operation *termOp = block->getTerminator();
  for (auto [op, var] : opTimeVar) {
    model.addConstr(var <= opTimeVar.at(termOp));

    for (auto userOp : op->getUsers()) {
      if (opTimeVar.count(userOp) > 0)
        model.addConstr(var + 1 <= opTimeVar.at(userOp));
    }
  }
  return success();
}

static LogicalResult placeToCntPe(GRBModel &model,
                                  SmallVector<GRBVar, 5> allowPe, GRBVar cntPe,
                                  std::string op1Name, std::string op2Name,
                                  std::string prefix = "") {
  GRBVar self = allowPe[0];
  GRBVar left = allowPe[1];
  GRBVar right = allowPe[2];
  GRBVar top = allowPe[3];
  GRBVar bottom = allowPe[4];

  // Add constraints that y must be one of the neighbors
  GRBVar fLeft =
      model.addVar(0, 1, 0, GRB_BINARY, "L" + op1Name + "X" + op2Name);
  GRBVar fRight =
      model.addVar(0, 1, 0, GRB_BINARY, "R" + op1Name + "X" + op2Name);
  GRBVar fTop =
      model.addVar(0, 1, 0, GRB_BINARY, "T" + op1Name + "X" + op2Name);
  GRBVar fBottom =
      model.addVar(0, 1, 0, GRB_BINARY, "B" + op1Name + "X" + op2Name);
  GRBVar fSelf =
      model.addVar(0, 1, 0, GRB_BINARY, "S" + op1Name + "X" + op2Name);

  if (prefix != "") {
    model.addQConstr(cntPe == left * fLeft + right * fRight + top * fTop +
                                  bottom * fBottom + self * fSelf,
                     "N_" + prefix);

    model.addConstr(fLeft + fRight + fTop + fBottom + fSelf == 1,
                    "N_" + prefix);
  } else {

    model.addQConstr(cntPe == left * fLeft + right * fRight + top * fTop +
                                  bottom * fBottom + self * fSelf);

    model.addConstr(fLeft + fRight + fTop + fBottom + fSelf == 1);
  }
  return success();
}

LogicalResult BasicBlockILPScheduler::createHardwareConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar,
    const std::map<Operation *, std::string> varName) {
  unsigned constrId = 0;
  for (auto [op, x] : opPeVar) {
    SetVector<Operation *> producers;
    for (auto [ind, opVal] : llvm::enumerate(op->getOperands())) {
      auto defOp = opVal.getDefiningOp();
      if (defOp && opPeVar.count(defOp) > 0)
        producers.insert(opVal.getDefiningOp());
    }

    // the consumer should able to access the producer's output
    // Create helper variables for the possible neighbors
    GRBVar left = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
    GRBVar right = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
    GRBVar top = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
    GRBVar bottom = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);

    // Auxiliary variables for calculations
    GRBVar xRow = model.addVar(0, nRow - 1, 0, GRB_INTEGER);
    GRBVar xCol = model.addVar(0, nCol - 1, 0, GRB_INTEGER);

    // Constraints to calculate row and column indices of x
    // xRow == x / nCol
    model.addConstr(xRow == (x - xCol) / nCol);
    // xCol == x % nCol
    GRBVar u = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
    model.addConstr(x == u * nCol + xCol);

    for (auto prodOp : producers) {
      auto y = opPeVar.at(prodOp);

      // Calculate left neighbor (wrap around if needed)
      GRBVar leftCol = model.addVar(0, nCol - 1, 0, GRB_INTEGER);
      GRBVar uLeft = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(xCol - 1 == nCol * uLeft + leftCol);
      model.addConstr(left == xRow * nCol + leftCol);

      // Calculate right neighbor (wrap around if needed)
      GRBVar rightCol = model.addVar(0, nCol - 1, 0, GRB_INTEGER);
      GRBVar uRight = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(xCol + 1 == nCol * uRight + rightCol);
      model.addConstr(right == xRow * nCol + rightCol);

      // Calculate top neighbor (wrap around if needed)
      GRBVar topRow = model.addVar(0, nRow - 1, 0, GRB_INTEGER);
      GRBVar uTop = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(xRow - 1 == nRow * uTop + topRow);
      model.addConstr(top == topRow * nCol + xCol);

      // Calculate bottom neighbor (wrap around if needed)
      GRBVar bottomRow = model.addVar(0, nRow - 1, 0, GRB_INTEGER);
      GRBVar uBottom =
          model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(xRow + 1 == nRow * uBottom + bottomRow);
      model.addConstr(bottom == bottomRow * nCol + xCol);

      placeToCntPe(model, SmallVector<GRBVar, 5>{x, left, right, top, bottom},
                   y, varName.at(op), varName.at(prodOp),
                   "H_" + std::to_string(constrId));
      constrId++;
    }
  }

  for (size_t i = 0; i < opTimeVar.size(); i++) {
    auto timePair = *std::next(opTimeVar.begin(), i);
    auto spacePair = *std::next(opPeVar.begin(), i);

    GRBVar t1 = timePair.second;
    GRBVar s1 = spacePair.second;

    // Block *block = timePair.first->getBlock();
    for (size_t j = i + 1; j < opTimeVar.size(); j++) {
      auto timePair2 = *std::next(opTimeVar.begin(), j);
      auto spacePair2 = *std::next(opPeVar.begin(), j);

      // If the operations are in different blocks, skip
      //   if (block != timePair2.first->getBlock())
      //     continue;

      GRBVar t2 = timePair2.second;
      GRBVar s2 = spacePair2.second;

      GRBVar t_eq = model.addVar(0, 1, 0, GRB_BINARY);
      GRBVar s_eq = model.addVar(0, 1, 0, GRB_BINARY);

      GRBVar diffT = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
      GRBVar diffTAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(diffT == t1 - t2);
      model.addGenConstrAbs(diffTAbs, diffT);

      GRBVar diffS = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
      GRBVar diffSAbs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(diffS == s1 - s2);
      model.addGenConstrAbs(diffSAbs, diffS);

      // If diffAbs is zero, t_eq and s_eq should be one,
      model.addConstr(t_eq >= 1 - diffTAbs);
      model.addConstr(s_eq >= 1 - diffSAbs);
      model.addConstr(t_eq + s_eq <= 1);
    }
  }
  return success();
}

LogicalResult BasicBlockILPScheduler::createGlobalLiveInInterConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar) {
  unsigned constrId = 0;
  for (auto [val, pe] : liveInInter) {
    for (auto user : val.getUsers()) {
      if (opPeVar.count(user) == 0)
        continue;

      // the user operation should be assigned to the same PE
      model.addConstr(opPeVar.at(user) == pe);
      model.optimize();
      if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL &&
          model.get(GRB_IntAttr_Status) != GRB_SUBOPTIMAL)
        return failure();
    }
  }
  return success();
}

LogicalResult BasicBlockILPScheduler::createGlobalLiveInExterConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar) {
  // Before the complete comsumption of the liveIn value, the block which stores
  // the liveIn Value should be blocked.
  unsigned constrId = 0;
  for (auto [val, pe] : liveInExter) {
    for (auto user : val.getUsers()) {
      if (opPeVar.count(user) == 0)
        continue;

      // block the PE until the liveIn value is consumed
      if (failed(blockPeAssignment(model, nullptr, user, opTimeVar, opPeVar,
                                   varName, false, pe))) {
        llvm::errs() << "Failed to create global live in for " << val << "\n";
        // store val, replace val with swi value
        return failure();
      }

      // the user operation should be assigned to the neighbor or itself
      GRBVar self = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
      model.addConstr(self == pe);
      GRBVar left = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
      model.addConstr(left == (pe - 1 + nCol) % nCol + (int)(pe / nCol) * nCol);
      GRBVar right = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
      model.addConstr(right == (pe + 1) % nCol + (int)(pe / nCol) * nCol);
      GRBVar top = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
      model.addConstr(top == (pe - nCol + nRow * nCol) % (nRow * nCol));
      GRBVar bottom = model.addVar(0, nRow * nCol - 1, 0, GRB_INTEGER);
      model.addConstr(bottom == (pe + nCol) % (nRow * nCol));

      placeToCntPe(model,
                   SmallVector<GRBVar, 5>{self, left, right, top, bottom},
                   opPeVar.at(user), std::to_string(pe), varName.at(user),
                   "GIn_" + std::to_string(constrId));
      constrId++;
      model.optimize();

      if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL &&
          model.get(GRB_IntAttr_Status) != GRB_SUBOPTIMAL) {
        llvm::errs() << "Failed to place " << *user << "\n";

        spill = val;
        failUser = user;
        return failure();
      }
    }
  }
  return success();
}

LogicalResult BasicBlockILPScheduler::createLocalLivenessConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar,
    const std::map<Operation *, std::string> varName) {
  for (auto [consumer, x] : opPeVar) {
    std::vector<Operation *> producers;
    for (auto [ind, opVal] : llvm::enumerate(consumer->getOperands())) {
      auto defOp = opVal.getDefiningOp();
      if (defOp && opPeVar.count(defOp) > 0)
        producers.push_back(opVal.getDefiningOp());
    }

    for (auto prodOp : producers) {
      if (failed(blockPeAssignment(model, prodOp, consumer, opTimeVar, opPeVar,
                                   varName))) {
        llvm::errs() << "Failed to create local liveness for " << *prodOp
                     << "\n";
        return failure();
      }
    }
  }
  return success();
}

LogicalResult BasicBlockILPScheduler::initVariablesForBlock(
    GRBModel &model, std::map<Operation *, GRBVar> &opTimeVar,
    std::map<Operation *, GRBVar> &opPeVar) {
  for (auto [opId, op] : llvm::enumerate(block->getOperations())) {
    // Skip the constant operations which is mapped to Imm field
    if (isa<arith::ConstantOp>(op))
      continue;

    // Create the variable for the operation
    opTimeVar[&op] =
        model.addVar(0, GRB_INFINITY, 0.0, GRB_INTEGER,
                     "t_" + std::to_string(bbId) + "_" + std::to_string(opId));
    opPeVar[&op] =
        model.addVar(0.0, nCol * nRow - 1, 0.0, GRB_INTEGER,
                     "pe_" + std::to_string(bbId) + "_" + std::to_string(opId));
    varName[&op] = std::to_string(bbId) + "_" + std::to_string(opId);
  }
  return success();
}

LogicalResult BasicBlockILPScheduler::createGlobalLiveOutInterConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar,
    const std::map<Operation *, std::string> varName) {
  // the inter value should be distributed among the PEs
  std::vector<int> availUse(nRow * nCol, 2);
  for (auto &[val, index] : liveInInter) {
    availUse[index] -= 1;
  }
  // CANCEL PRINT
  llvm::errs() << "AvailUse: [";
  for (auto i : availUse) {
    llvm::errs() << i << " ";
  }
  llvm::errs() << "]\n";

  for (unsigned p = 0; p < nRow * nCol; p++) {
    // GRBVar useSum = model.addVar(0, availUse[p], 0, GRB_INTEGER);
    GRBLinExpr accumulatedSum = 0;
    for (auto [val, index] : liveOutInter) {
      if (p == 0)
        llvm::errs() << "LiveOutInter: " << val << "\n";
      Operation *defOp = val.getDefiningOp();
      if (!defOp || opTimeVar.count(defOp) == 0)
        continue;

      auto peVar = opPeVar.at(defOp);

      GRBVar flag = model.addVar(0, 1, 0, GRB_BINARY);
      GRBVar diff = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_INTEGER);
      GRBVar abs = model.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);
      model.addConstr(diff == peVar - p);
      model.addGenConstrAbs(abs, diff);
      model.addConstr(flag >= (1 - abs), "liveOutInter_" + varName.at(defOp) +
                                             "_" + std::to_string(p));

      accumulatedSum += flag;
      model.addConstr(accumulatedSum <= availUse[p]);
      model.optimize();
      if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL &&
          model.get(GRB_IntAttr_Status) != GRB_SUBOPTIMAL) {
        llvm::errs() << "Failed to assign live Out at " << p << " for " << val
                     << "\n";
        strategy = FailureStrategy::Split;
        spill = val;
        return failure();
      }
    }
    // model.addConstr(useSum == accumulatedSum, "LiveUse_" +
    // std::to_string(p));
  }
  return success();
}

LogicalResult BasicBlockILPScheduler::createGlobalLiveOutExterConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar) {
  Operation *termOp = block->getTerminator();
  for (auto [val, _] : liveOutExter) {
    // first check can external liveness survive
    Operation *defOp = val.getDefiningOp();
    llvm::errs() << "LiveOutExter: " << val << "\n";

    if (defOp && opTimeVar.count(defOp) > 0) {
      if (failed(blockPeAssignment(model, defOp, termOp, opTimeVar, opPeVar,
                                   varName, true))) {
        strategy = FailureStrategy::Split;
        spill = val;
        llvm::errs() << "Failed to create global live out for " << val << "\n";
        return failure();
      }
      continue;
    }

    // keep the liveOut value if it is get from the external block

    auto it = std::find_if(
        liveInExter.begin(), liveInExter.end(),
        [val](std::pair<Value, unsigned> p) { return p.first == val; });
    if (it == liveInExter.end()) {
      llvm::errs() << "Failed to find the live out value " << val << "\n";
      strategy = FailureStrategy::Abort;
      return failure();
    }
    unsigned pe = it->second;
    if (failed(blockPeAssignment(model, nullptr, termOp, opTimeVar, opPeVar,
                                 varName, true, pe))) {
      strategy = FailureStrategy::Split;
      spill = val;
      llvm::errs() << "Failed to create global live out for " << val << "\n";
      return failure();
    }
  }

  return success();
}

LogicalResult BasicBlockILPScheduler::createObjetiveFunction(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar) {
  Operation *termOp = block->getTerminator();
  //   Assuming the start execution time of the block is 0, minimize the
  //   terminator execution time as the objective function.
  GRBLinExpr obj = opTimeVar.at(termOp);
  model.setObjective(obj, GRB_MINIMIZE);
  return success();
}

LogicalResult BasicBlockILPScheduler::createSchedulerAndSolve() {
  // Create scheduler for each operation
  GRBEnv env = GRBEnv("./gurobi.log");
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel model = GRBModel(env);
  int time_limit = 60;
  model.set(GRB_DoubleParam_TimeLimit, time_limit);

  // Objective function
  GRBLinExpr obj = 0;

  std::map<Operation *, GRBVar> timeVarMap;
  std::map<Operation *, GRBVar> peVarMap;

  initVariablesForBlock(model, timeVarMap, peVarMap);

  createObjetiveFunction(model, timeVarMap);

  if (failed(createGlobalLiveInInterConstraints(model, timeVarMap, peVarMap)))
    return failure();

  llvm::errs() << "Created global live in inter constraints\n";

  if (failed(createGlobalLiveInExterConstraints(model, timeVarMap, peVarMap)))
    return failure();

  llvm::errs() << "Created global live in exter constraints\n";

  createLocalDominanceConstraints(model, timeVarMap);

  createLocalLivenessConstraints(model, timeVarMap, peVarMap, varName);

  createHardwareConstraints(model, timeVarMap, peVarMap, varName);

  if (failed(createGlobalLiveOutInterConstraints(model, timeVarMap, peVarMap,
                                                 varName))) {
    return failure();
  }
  llvm::errs() << "Create global live out inter constraints\n";

  if (failed(createGlobalLiveOutExterConstraints(model, timeVarMap, peVarMap)))
    return failure();

  // Optimize the model

  model.write("model_" + std::to_string(bbId) + ".lp");
  model.optimize();

  if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL ||
      model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL) {
  } else if (model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT) {
    int solCount = model.get(GRB_IntAttr_SolCount);
    if (solCount == 0) {
      llvm::errs() << "No solution found within " << time_limit << "s\n";
      return failure();
    }
    // If solCount > 0, continue with partial solution
  } else {
    llvm::errs() << "Model is infeasible\n";
    return failure();
  }

  std::ofstream csvFile("output_" + std::to_string(bbId) + ".csv");
  model.write("solution_" + std::to_string(bbId) + ".sol");

  for (auto [op, var] : timeVarMap) {
    std::string str;
    llvm::raw_string_ostream rso(str);
    rso << *op;
    csvFile << rso.str() << "&" << var.get(GRB_StringAttr_VarName) << "&"
            << var.get(GRB_DoubleAttr_X) << "&"
            << peVarMap[op].get(GRB_StringAttr_VarName) << "&"
            << peVarMap[op].get(GRB_DoubleAttr_X) << "\n";
  }

  writeLiveOutResult(peVarMap);
  writeILPResult(timeVarMap, peVarMap);

  return success();
}

void BasicBlockILPScheduler::writeILPResult(
    const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar) {
  for (auto [op, var] : opPeVar) {
    int pe = (int)var.get(GRB_DoubleAttr_X);
    int time = (int)opTimeVar.at(op).get(GRB_DoubleAttr_X);
    ScheduleUnitBB su = {time, pe};
    solution[op] = su;
  }
  return;
}

void BasicBlockILPScheduler::writeLiveOutResult(
    const std::map<Operation *, GRBVar> opPeVar) {
  for (auto &[val, index] : liveOutExter) {
    llvm::errs() << "LiveOutExter: " << val << "\n";
    Operation *defOp = val.getDefiningOp();
    if (defOp && opPeVar.count(defOp) > 0) {
      int pe = (int)opPeVar.at(defOp).get(GRB_DoubleAttr_X);
      index = pe;
      continue;
    }
    auto it = std::find_if(
        liveInExter.begin(), liveInExter.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });
    if (it != liveInExter.end()) {
      index = it->second;
    }
  }

  for (auto &[val, index] : liveOutInter) {
    Operation *defOp = val.getDefiningOp();
    if (defOp && opPeVar.count(defOp) > 0) {
      int pe = (int)opPeVar.at(defOp).get(GRB_DoubleAttr_X);
      index = pe;
      continue;
    }
    auto it = std::find_if(
        liveInInter.begin(), liveInInter.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });
    if (it != liveInInter.end()) {
      index = it->second;
    }
  }
}
#endif

static void getAllPhiRelatedValues(Value val, SetVector<Value> &relatedVals) {
  if (relatedVals.count(val))
    return;

  if (val.isa<BlockArgument>()) {
    relatedVals.insert(val);
    unsigned ind = val.cast<BlockArgument>().getArgNumber();
    Block *block = val.getParentBlock();
    for (Block *pred : block->getPredecessors()) {
      Operation *branchOp = pred->getTerminator();
      Value operand = nullptr;
      if (auto br = dyn_cast<cf::BranchOp>(branchOp)) {
        operand = br.getOperand(ind);
      } else if (auto cbr = dyn_cast<cgra::ConditionalBranchOp>(branchOp)) {
        if (block == cbr.getTrueDest())
          operand = cbr.getTrueOperand(ind);
        else
          operand = cbr.getFalseOperand(ind);
      }
      relatedVals.insert(operand);
      // recursively get the related values
      getAllPhiRelatedValues(operand, relatedVals);
    }
    return;
  }

  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    llvm::errs() << "Can not track " << val << " for phi related chain\n";
    return;
  }
  relatedVals.insert(val);
  for (auto &use : val.getUses()) {
    unsigned index = use.getOperandNumber();
    if (auto br = dyn_cast_or_null<cf::BranchOp>(use.getOwner())) {
      getAllPhiRelatedValues(br->getSuccessor(0)->getArgument(index),
                             relatedVals);
    }
    if (auto cbr =
            dyn_cast_or_null<cgra::ConditionalBranchOp>(use.getOwner())) {
      bool isTrueOpr = index >= 2 && index < 2 + cbr.getNumTrueDestOperands();
      if (isTrueOpr) {
        getAllPhiRelatedValues(cbr.getTrueDest()->getArgument(index - 2),
                               relatedVals);
      } else {
        getAllPhiRelatedValues(cbr.getFalseDest()->getArgument(
                                   index - cbr.getNumTrueDestOperands() - 2),
                               relatedVals);
      }
    }
  }

  // return relatedVals;
}

bool compigra::isPhiRelatedValue(Value val) {
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

void TemporalCGRAScheduler::writeLiveOutResult(const liveVec liveOutExter,
                                               const liveVec liveOutInter) {
  for (auto [val, ind] : liveOutExter) {
    if (isPhiRelatedValue(val)) {
      SetVector<Value> relatedVals;
      getAllPhiRelatedValues(val, relatedVals);
      for (auto relatedVal : relatedVals)
        liveValExterPlaces.push_back({relatedVal, ind});

    } else {
      liveValExterPlaces.push_back({val, ind});
    }
  }

  for (auto [val, ind] : liveOutInter)
    liveValInterPlaces.push_back({val, ind});
}

liveVec TemporalCGRAScheduler::getExternalLiveIn(Block *block) {
  auto bbLiveIn = liveIns[block];
  liveVec liveInExter;

  for (auto val : bbLiveIn) {
    // search whether the val is in the liveOutExterPlaces
    auto it = std::find_if(
        liveValExterPlaces.begin(), liveValExterPlaces.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValExterPlaces.end()) {
      auto index = it->second;
      liveInExter.push_back({val, index});
    }
  }
  return liveInExter;
}

liveVec TemporalCGRAScheduler::getInternalLiveIn(Block *block) {
  auto bbLiveIn = liveIns[block];
  liveVec liveInInter;

  for (auto val : bbLiveIn) {
    // search whether the val is in the liveOutInterPlaces
    auto it = std::find_if(
        liveValInterPlaces.begin(), liveValInterPlaces.end(),
        [&](std::pair<Value, unsigned> p) { return p.first == val; });

    if (it != liveValInterPlaces.end()) {
      auto index = it->second;
      liveInInter.push_back({val, index});
    }
  }
  return liveInInter;
}

void TemporalCGRAScheduler::saveSubILPModelResult(
    const std::map<Operation *, ScheduleUnitBB> res) {
  for (auto [op, su] : res) {
    ScheduleUnit res = {su.time, su.pe, -1};
    solution[op] = res;
  }
}

void TemporalCGRAScheduler::insertMovOp(Value origVal, Operation *user) {
  builder.setInsertionPoint(user);
  auto zero = builder.create<arith::ConstantIntOp>(user->getLoc(), 0,
                                                   builder.getIntegerType(32));
  auto movOp = builder.create<arith::AddIOp>(user->getLoc(), origVal, zero);
  user->replaceUsesOfWith(origVal, movOp->getResult(0));
}

void TemporalCGRAScheduler::placeLSOpsToBlock(Block *block) {
  llvm::errs() << "Not Implemented Error change pre-scheduled result\n";
}

void TemporalCGRAScheduler::insertLSOps(Value origVal, unsigned memLoc,
                                        bool processCntPhi) {

  unsigned lastPtr = memLoc == UINT_MAX ? 0 : memLoc;
  if (!memStack.empty() && memLoc == UINT_MAX) {
    lastPtr = memStack.back().first + 4;
  }

  if (processCntPhi) {
    SmallVector<Block *, 4> succBlocks = getCntBlocksThroughPhi(origVal);
    for (auto suc : succBlocks) {
      // get the phi value
      auto phiVal = getCntBlockArgument(origVal, suc);
      // insert lwi to replace the phi value
      builder.setInsertionPoint(&suc->getOperations().front());
      auto constOp = builder.create<arith::ConstantIntOp>(
          suc->getOperations().front().getLoc(), lastPtr,
          builder.getIntegerType(32));
      auto loadOp =
          builder.create<cgra::LwiOp>(suc->getOperations().front().getLoc(),
                                      origVal.getType(), constOp->getResult(0));
      phiVal.replaceAllUsesWith(loadOp->getResult(0));
      // insert swi op for all source operands
      SmallVector<Value, 2> srcVals = getSrcOprandsOfPhi(phiVal);
      for (auto src : srcVals) {
        insertLSOps(src, lastPtr, false);
        // erase BlockArgument from the IR
      }
    }
    return;
  }

  auto refOp = origVal.getDefiningOp();
  if (!refOp)
    refOp = &(origVal.cast<BlockArgument>()
                  .getParentBlock()
                  ->getOperations()
                  .front());

  // insert lwi op before the user block
  std::map<Block *, cgra::LwiOp> lwiOps;
  // If user is not specified, insert lwi op to all users
  for (auto &use : llvm::make_early_inc_range(origVal.getUses())) {
    Operation *user = use.getOwner();

    llvm::errs() << "User: " << *user << "\n";
    Block *userBlock = user->getBlock();
    unsigned blockIndex = std::distance(
        scheduleSeq.begin(),
        std::find(scheduleSeq.begin(), scheduleSeq.end(), userBlock));
    // TODO[@YX]: avoid insert unnecessary lwi ops
    if (userBlock == origVal.getParentBlock())
      continue;
    if (blockIndex < scheduleIdx) {
      placeLSOpsToBlock(userBlock);
      // return;
    }

    if (lwiOps.count(userBlock) != 0) {
      user->setOperand(use.getOperandNumber(), lwiOps[userBlock]);
      continue;
    }
    builder.setInsertionPoint(&userBlock->getOperations().front());
    auto addr = builder.create<arith::ConstantIntOp>(
        refOp->getLoc(), lastPtr, builder.getIntegerType(32));
    auto loadOp =
        builder.create<cgra::LwiOp>(userBlock->getOperations().front().getLoc(),
                                    origVal.getType(), addr->getResult(0));
    user->setOperand(use.getOperandNumber(), loadOp->getResult(0));
    lwiOps[userBlock] = loadOp;
  }

  builder.setInsertionPointAfter(refOp);
  // TODO: Seek whether swi op is already inserted
  auto addr = builder.create<arith::ConstantIntOp>(refOp->getLoc(), lastPtr,
                                                   builder.getIntegerType(32));
  builder.create<cgra::SwiOp>(refOp->getLoc(), origVal, addr->getResult(0));
  memStack.push_back({lastPtr, origVal});
}

static void handleMovAttempFailure(Operation *preFailOp, Value preSpillVal,
                                   BasicBlockILPScheduler &scheduler,
                                   int &movNum, unsigned maxTry) {
  Operation *failUser = scheduler.getFailUser();
  if (failUser != preFailOp) {
    scheduler.setFailureStrategy(FailureStrategy::Mov);
    movNum = maxTry;
    return;
  }

  Value spill = scheduler.getSpillVal();
  auto definingOp = spill.getDefiningOp();
  Operation *movOp = dyn_cast_or_null<arith::AddIOp>(definingOp);
  if (!movOp) {
    movOp = dyn_cast_or_null<arith::AddFOp>(definingOp);
  }

  // not the same failure, reset the movNum
  if (!movOp || movOp->getOperand(0) != preSpillVal) {
    scheduler.setFailureStrategy(FailureStrategy::Mov);
    movNum = maxTry;
    return;
  }

  auto cstOp = movOp->getOperand(1).getDefiningOp();
  if (!cstOp || !dyn_cast_or_null<arith::ConstantOp>(cstOp) ||
      dyn_cast<arith::ConstantOp>(cstOp)
              .getValue()
              .dyn_cast<IntegerAttr>()
              .getInt() != 0) {
    scheduler.setFailureStrategy(FailureStrategy::Mov);
    movNum = maxTry;
    return;
  }

  // the same failure, try again and decrease the movNum
  movNum--;
}

void TemporalCGRAScheduler::makeScheduleSeq() {
  scheduleSeq.clear();
  scheduleIdx = 0;
  for (auto &bb : region.getBlocks())
    scheduleSeq.push_back(&bb);
}

LogicalResult TemporalCGRAScheduler::createSchedulerAndSolve() {
  // Create scheduler for each block
  makeScheduleSeq();
  computeLiveValue();
  for (auto [bb, block] : llvm::enumerate(scheduleSeq)) {
    llvm::errs() << "\nBlock " << bb << " is scheduling\n";

    BasicBlockILPScheduler scheduler(maxReg, nRow, nCol, block, bb, builder);
    scheduler.setLiveValue(liveIns[block], liveOuts[block]);
    scheduler.setLiveInPrerequisite(getExternalLiveIn(block),
                                    getInternalLiveIn(block));

    int maxIter = 3;
    Operation *failUser = nullptr;
    Value spill = nullptr;
    int movNum = 1;
    scheduler.setFailureStrategy(FailureStrategy::Mov);
    while (maxIter > 0) {
      if (movNum < 0) {
        // step back, remove the additional sadd zero ops
        scheduler.setFailureStrategy(FailureStrategy::Split);
      }

      if (failed(scheduler.createSchedulerAndSolve())) {
        if (scheduler.getFailureStrategy() == FailureStrategy::Mov) {
          handleMovAttempFailure(failUser, spill, scheduler, movNum, 3);
          spill = scheduler.getSpillVal();
          failUser = scheduler.getFailUser();
          insertMovOp(spill, failUser);
          llvm::errs() << "Spill " << spill << " for "
                       << *(scheduler.getFailUser()) << "\n";
        }
        if (scheduler.getFailureStrategy() == FailureStrategy::Split) {
          // split the liveOut value
          spill = scheduler.getSpillVal();
          bool pushPhiToMem =
              isPhiRelatedValue(spill) && !spill.isa<BlockArgument>();
          insertLSOps(spill, UINT_MAX, pushPhiToMem);
          if (pushPhiToMem) {
            // observe the asm
            return success();
          }
          computeLiveValue();
          scheduler.setLiveValue(liveIns[block], liveOuts[block]);
          scheduler.setLiveInPrerequisite(getExternalLiveIn(block),
                                          getInternalLiveIn(block));
        }

        maxIter--;
        continue;
      }
      break;
    }

    if (maxIter == 0) {
      llvm::errs() << "Failed to schedule block " << bb << "\n";
      return failure();
    }
    saveSubILPModelResult(scheduler.getSolution());

    writeLiveOutResult(scheduler.getExternalLiveOutResult(),
                       scheduler.getInternalLiveOutResult());
    if (bb == 10)
      break;

    scheduleIdx++;
  }
  // print solution
  llvm::errs() << "\n====================\n";
  for (auto [op, su] : solution) {
    llvm::errs() << "Operation: " << *op << " Time: " << su.time
                 << " PE: " << su.pe << "\n";
  }
  return success();
}
