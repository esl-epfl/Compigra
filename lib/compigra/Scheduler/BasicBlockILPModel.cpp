//===- BasicBlockILPModel.cpp - Implement the class/functions of basic block
// ILP model to schedule the executions of operations*- C++-* -------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements class for BasicBlockILPModel functions.
//
//===----------------------------------------------------------------------===//

#include "compigra/Scheduler/BasicBlockILPModel.h"
#include "fstream"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace compigra;

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
                  bool strict = false, int constPE = -1, bool check = true) {
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
  if (check) {
    model.optimize();
    if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL ||
        model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL)
      return success();
    return failure();
  }
  return success();
}

void BasicBlockILPModel::createRAWConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar) {
  for (auto pair : opRAWs) {
    auto [dstOp, srcOp] = pair;
    if (opTimeVar.count(srcOp) == 0 || opTimeVar.count(dstOp) == 0)
      continue;
    GRBVar x = opTimeVar.at(dstOp);
    GRBVar y = opTimeVar.at(srcOp);
    model.addConstr(x <= y + 1);
    llvm::errs() << "RAW: " << *dstOp << " << " << *srcOp << "\n";
  }
}

LogicalResult BasicBlockILPModel::createLocalDominanceConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar) {

  // Consumer must be executed after the producer
  Operation *termOp = block->getTerminator();
  for (auto op : scheduleOps) {
    GRBVar var = opTimeVar.at(op);
    model.addConstr(var <= opTimeVar.at(termOp));

    for (auto &use : op->getUses()) {
      auto userOp = use.getOwner();
      if (opTimeVar.count(userOp) == 0)
        continue;
      if (isa<cf::BranchOp>(userOp))
        continue;
      if (isa<cgra::ConditionalBranchOp>(userOp) && use.getOperandNumber() >= 2)
        continue;
      model.addConstr(var + 1 <= opTimeVar.at(userOp));
    }
  }
  return success();
}

static LogicalResult placeToCntPe(GRBModel &model,
                                  SmallVector<GRBVar, 5> allowPe, GRBVar cntPe,
                                  std::string op1Name, std::string op2Name,
                                  std::string prefix = "", bool check = true) {
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
                     "Nbr_" + prefix);
    model.addConstr(fLeft + fRight + fTop + fBottom + fSelf == 1,
                    "NbrF_" + prefix);
  } else {
    model.addQConstr(cntPe == left * fLeft + right * fRight + top * fTop +
                                  bottom * fBottom + self * fSelf);
    model.addConstr(fLeft + fRight + fTop + fBottom + fSelf == 1);
  }
  if (check) {
    model.optimize();
    if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL &&
        model.get(GRB_IntAttr_Status) != GRB_SUBOPTIMAL) {
      return failure();
    }
  }
  return success();
}

LogicalResult BasicBlockILPModel::createHardwareConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar,
    const std::map<Operation *, std::string> varName) {
  unsigned constrId = 0;
  for (auto op : scheduleOps) {
    GRBVar x = opPeVar.at(op);
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

      if (failed(placeToCntPe(
              model, SmallVector<GRBVar, 5>{x, left, right, top, bottom}, y,
              varName.at(op), varName.at(prodOp),
              "H_" + std::to_string(constrId), true))) {

        spill = prodOp->getResult(0);
        failUser = op;
        checkptr = op;
        llvm::errs() << "internal split for " << *prodOp << " -> " << *op
                     << "\n";
        if (strategy == FailureStrategy::Mov)
          llvm::errs() << "strategy : mov"
                       << "\n";
        if (strategy == FailureStrategy::Split)
          llvm::errs() << "strategy : split"
                       << "\n";
        if (strategy == FailureStrategy::Abort)
          llvm::errs() << "strategy : abort"
                       << "\n";
        return failure();
      }

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

LogicalResult BasicBlockILPModel::createGlobalLiveInInterConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar) {
  unsigned constrId = 0;
  for (auto [val, pe] : liveInInter) {
    for (auto user : val.getUsers()) {
      if (opPeVar.count(user) == 0 || pe == UINT32_MAX)
        continue;

      // the user operation should be assigned to the same PE
      model.addConstr(opPeVar.at(user) == pe);
      model.optimize();
      if (model.get(GRB_IntAttr_Status) != GRB_OPTIMAL &&
          model.get(GRB_IntAttr_Status) != GRB_SUBOPTIMAL) {
        spill = val;
        failUser = user;
        return failure();
      }
    }
  }
  return success();
}

LogicalResult BasicBlockILPModel::createGlobalLiveInExterConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar) {
  // Before the complete comsumption of the liveIn value, the block which stores
  // the liveIn Value should be blocked.
  unsigned constrId = 0;
  for (auto [val, pe] : liveInExter) {
    for (auto user : val.getUsers()) {
      if (opPeVar.count(user) == 0 || pe == UINT32_MAX)
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

LogicalResult BasicBlockILPModel::createLocalLivenessConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar,
    const std::map<Operation *, std::string> varName) {
  bool check = false;
  for (auto consumer : scheduleOps) {
    std::vector<Operation *> producers;
    // llvm::errs() << "Create liveness constraints for: " << *consumer << "\n";
    for (auto [ind, opVal] : llvm::enumerate(consumer->getOperands())) {
      auto defOp = opVal.getDefiningOp();
      if (defOp && opPeVar.count(defOp) > 0)
        producers.push_back(opVal.getDefiningOp());
    }
    if (consumer == checkptr)
      check = true;

    for (auto prodOp : producers) {
      if (failed(blockPeAssignment(model, prodOp, consumer, opTimeVar, opPeVar,
                                   varName, check))) {
        llvm::errs() << "Failed to create local liveness for " << *prodOp
                     << "\n";
        spill = prodOp->getResult(0);
        failUser = consumer;
        checkptr = consumer;
        return failure();
      }
    }
  }
  return success();
}

LogicalResult BasicBlockILPModel::initVariablesForBlock(
    GRBModel &model, std::map<Operation *, GRBVar> &opTimeVar,
    std::map<Operation *, GRBVar> &opPeVar) {
  scheduleOps.clear();
  unsigned opId = 0;
  for (auto &op : block->getOperations()) {
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
    scheduleOps.push_back(&op);
    opId++;
  }
  return success();
}

LogicalResult BasicBlockILPModel::createGlobalLiveOutInterConstraints(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar,
    const std::map<Operation *, GRBVar> opPeVar,
    const std::map<Operation *, std::string> varName) {
  // the inter value should be distributed among the PEs
  std::vector<int> availUse(nRow * nCol, 2);
  for (auto &[val, index] : liveInInter) {
    if (index == UINT32_MAX)
      continue;
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
    for (auto [val, _] : liveOutInter) {
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
        failUser = nullptr;
        return failure();
      }
    }
    // model.addConstr(useSum == accumulatedSum, "LiveUse_" +
    // std::to_string(p));
  }
  return success();
}

LogicalResult BasicBlockILPModel::createGlobalLiveOutExterConstraints(
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
    if (pe == UINT32_MAX)
      continue;
    if (failed(blockPeAssignment(model, nullptr, termOp, opTimeVar, opPeVar,
                                 varName, true, pe))) {
      strategy = FailureStrategy::Split;
      spill = val;
      llvm::errs() << "Failed to create global live out for " << val << " ";
      if (isa<BlockArgument>(val))
        for (auto [ind, bb] : llvm::enumerate(block->getParent()->getBlocks()))
          if (&bb == val.getParentBlock()) {
            llvm::errs() << ind << "\n";
            break;
          }
      return failure();
    }
  }

  return success();
}

LogicalResult BasicBlockILPModel::createObjetiveFunction(
    GRBModel &model, const std::map<Operation *, GRBVar> opTimeVar) {
  Operation *termOp = block->getTerminator();
  //   Assuming the start execution time of the block is 0, minimize the
  //   terminator execution time as the objective function.
  GRBLinExpr obj = opTimeVar.at(termOp);
  model.setObjective(obj, GRB_MINIMIZE);
  return success();
}

void BasicBlockILPModel::saveSubILPModelResult(std::string filename) {
  std::ofstream csvFile(filename);
  // If the model is infeasible, write the model to solution
  for (auto [op, res] : solution) {
    std::string str;
    llvm::raw_string_ostream rso(str);
    rso << *op;
    csvFile << rso.str() << "&" << res.time << "&" << res.pe << "\n";
  }
  csvFile.close();
}

LogicalResult BasicBlockILPModel::createSchedulerAndSolve() {
  // Create scheduler for each operation
  GRBEnv env = GRBEnv("./gurobi.log");
  env.set(GRB_IntParam_OutputFlag, 0);
  env.start();
  GRBModel model = GRBModel(env);
  int time_limit = 600;
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

  createRAWConstraints(model, timeVarMap);

  createLocalDominanceConstraints(model, timeVarMap);

  if (failed(createHardwareConstraints(model, timeVarMap, peVarMap, varName)))
    return failure();

  llvm::errs() << "Created hardware constraints\n";

  if (failed(createLocalLivenessConstraints(model, timeVarMap, peVarMap,
                                            varName))) {
    // strategy = FailureStrategy::Abort;
    return failure();
  }
  llvm::errs() << "Created local liveness constraints\n";

  if (failed(createGlobalLiveOutInterConstraints(model, timeVarMap, peVarMap,
                                                 varName))) {
    return failure();
  }
  llvm::errs() << "Create global live out inter constraints\n";

  if (failed(createGlobalLiveOutExterConstraints(model, timeVarMap, peVarMap)))
    return failure();
  llvm::errs() << "Created global live out exter constraints\n";

  // Optimize the model
  model.write("model_" + std::to_string(bbId) + ".lp");
  model.optimize();

  if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL ||
      model.get(GRB_IntAttr_Status) == GRB_SUBOPTIMAL) {
  } else if (model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT) {
    int solCount = model.get(GRB_IntAttr_SolCount);
    if (solCount == 0) {
      llvm::errs() << "No solution found within " << time_limit << "s\n";
      strategy = FailureStrategy::Abort;
      return failure();
    }
    // If solCount > 0, continue with partial solution
  } else {
    llvm::errs() << "Model is infeasible\n";
    return failure();
  }

  writeLiveOutResult(peVarMap);
  writeILPResult(timeVarMap, peVarMap);
  saveSubILPModelResult("sub_ilp_" + std::to_string(bbId) + ".csv");
  return success();
}

void BasicBlockILPModel::writeILPResult(
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

void BasicBlockILPModel::writeLiveOutResult(
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
