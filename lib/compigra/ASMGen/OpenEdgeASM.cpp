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
#include "compigra/Scheduler/ModuloScheduleAdapter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <set>

#ifdef HAVE_GUROBI
#include "gurobi_c++.h"
#endif

// #include <iomanip>

using namespace mlir;
using namespace compigra;

/// Get the loop block of the region
static Block *getLoopBlock(Region &region) {
  for (auto &block : region)
    for (auto suc : block.getSuccessors())
      if (suc == &block)
        return &block;
  return nullptr;
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

/// Allocate physical register based on the interference vertices
static char allocatePhysicalRegOnIG(std::unordered_set<int> interfNodes,
                                    std::map<int, char> colorMap,
                                    std::unordered_set<int> usedColors,
                                    unsigned maxReg) {
  // std::unordered_set<int> usedColors;
  // for (auto u : interfNodes) {
  //   if (colorMap.find(u) != colorMap.end())
  //     usedColors.insert(colorMap[u]);
  // }
  char color = 0;
  while (usedColors.find(color) != usedColors.end()) {
    color++;
    if (color >= maxReg) {
      return maxReg;
    }
  }
  return color;
}

static int getValueIndex(Value val, std::map<int, Value> opResult) {
  for (auto [ind, res] : opResult)
    if (res.getDefiningOp() == val.getDefiningOp())
      return ind;
  return -1;
}

// TODO[@YYY]: allocate register for the phi node (could be other PE, e.g.
// RCL, RCR, RCT, RCB)
std::map<int, int> getLimitationUseWithPhiNode(
    const std::vector<int> phiNodes, const std::map<int, Value> opMap,
    const std::map<int, mlir::Operation *> opList,
    compigra::InterferenceGraph<int> &graph, unsigned maxReg) {
  std::map<int, int> limitedUse;
  for (int node : phiNodes) {
    // phi node should be block argument of the basic block
    // BlockArgument arg = dyn_cast<BlockArgument>(opMap.at(node));
    auto arg = opMap.at(node);

    auto defOps = getCntDefOpIndirectly(arg, arg.getParentBlock());
    // allocate register for the phi node
    std::unordered_set<int> usedColors;
    for (auto u : graph.adjList[node]) {
      if (graph.colorMap.find(u) != graph.colorMap.end())
        usedColors.insert(graph.colorMap[u]);
    }

    // check whether the phi node is pre-colored
    char color = allocatePhysicalRegOnIG(graph.adjList[node], graph.colorMap,
                                         usedColors, maxReg);
    for (auto defOp : defOps) {
      // all defining op should be executed in the same PE, if it is executed in
      // current PE, then it should be assigned to the same register.
      bool internal = false;
      for (auto [_, op] : opList)
        if (op == defOp) {
          internal = true;
          break;
        }

      if (!internal)
        break;

      // find the corresponding value in the graph
      int defNode = getValueIndex(defOp->getResult(0), opMap);
      limitedUse[defNode] = {};

      // check whether the node is pre-colored, if yes then the color should be
      // the same
      if (graph.colorMap.find(defNode) != graph.colorMap.end()) {
        color = graph.colorMap[defNode];
        llvm::errs() << "DEF NODE: " << defNode << " set color "
                     << std::to_string(color) << "\n";
      }
    }
    // limit the coloring selection of the phi node
    std::unordered_set<int> limited;
    // rewrite all value in limitedUse
    for (auto [v, _] : limitedUse) {
      limitedUse[v] = color;
      // update adjacent list
      graph.colorMap[v] = color;
    }
    limitedUse[node] = color;
    graph.colorMap[node] = color;
  }
  return limitedUse;
}

LogicalResult compigra::allocateOutRegInPE(
    std::map<int, mlir::Operation *> opList,
    std::map<Operation *, ScheduleUnit> &solution, unsigned maxReg,
    std::map<int, std::unordered_set<int>> pcCtrlFlow) {
  // init Operation result to integer
  std::map<int, Value> opMap;
  auto graph = createInterferenceGraph(opList, opMap, pcCtrlFlow);

  // print opMap and interference graph
  llvm::errs() << "--------------Interference Graph-----------------\n";
  graph.printGraph();

  // allocate register using graph coloring
  // TODO[@YYY]: Spill the graph if the number of registers is not
  // enough
  auto peo = lexBFS(graph.adjList);
  if (peo.empty())
    return success();

  // the register allocation for block arguments (phi node) must be limited to
  // the same register.
  std::vector<int> phiList;
  llvm::errs() << "PEO: [";
  for (auto v : peo) {
    Value val = opMap[v];
    llvm::errs() << " " << v;
    // first mark phi node as limited use
    if (isa<BlockArgument>(val)) {
      phiList.push_back(v);
      llvm::errs() << "(" << val << ")";
      continue;
    }

    auto defOp = val.getDefiningOp();
    // check whether v has been pre-colored
    llvm::errs() << "(" << *defOp << " NEED?: " << graph.needColor(v) << ")";
    if (!graph.needColor(v)) {
      // Use produced pe + 100 as the color of the vertex to avoid conflict
      graph.colorMap[v] = solution[defOp].pe + 100;
      continue;
    }

    if (solution[defOp].reg >= 0)
      graph.colorMap[v] = solution[defOp].reg;

    if (graph.colorMap.find(v) != graph.colorMap.end())
      llvm::errs() << "{" << std::to_string(graph.colorMap[v]) << "}";
  }
  llvm::errs() << "]\n";

  // write limitation to the graph
  std::map<int, int> limitedUse =
      getLimitationUseWithPhiNode(phiList, opMap, opList, graph, maxReg);

  // Color the vertices in the order of PEO
  for (auto v : peo) {
    Value val = opMap[v];
    auto defOp = val.getDefiningOp();
    if (!defOp)
      continue;

    if (graph.needColor(v)) {
      if (solution[defOp].reg >= 0) {
        graph.colorMap[v] = solution[defOp].reg;
        llvm::errs() << v << ": " << std::to_string(graph.colorMap[v]) << "\n";
        continue;
      }

      // allocate register according to the limited use of the phi node
      if (limitedUse.find(v) != limitedUse.end()) {
        // TODO[@YYY]: check the validity of the limitation
        graph.colorMap[v] = limitedUse[v];
        llvm::errs() << v << ": "
                     << "ALLOCATE R" << std::to_string(limitedUse[v]) << " TO "
                     << val << "\n";
        solution[defOp].reg = limitedUse[v];
        continue;
      }

      std::unordered_set<int> usedColors;
      // allocate register using the interference graph
      for (auto u : graph.adjList[v]) {
        if (graph.colorMap.find(u) != graph.colorMap.end())
          usedColors.insert(graph.colorMap[u]);
      }

      // char color = 0;
      // while (usedColors.find(color) != usedColors.end()) {
      //   color++;
      //   if (color >= maxReg) {
      //     llvm::errs() << "FAILED ALLOCATE REGISTER\n";

      //     // maxReg is Rout. In principle, the register should not be
      //     // assigned to Rout if it is used internally.
      //     return failure();
      //   }
      // }
      char color = allocatePhysicalRegOnIG(graph.adjList[v], graph.colorMap,
                                           usedColors, maxReg);
      if (color >= maxReg) {
        llvm::errs() << "FAILED ALLOCATE REGISTER for " << v << "\n";
        return failure();
      }
      graph.colorMap[v] = color;
      llvm::errs() << v << ": ALLOCATE R" << std::to_string(color) << " TO "
                   << val << "\n";
      solution[defOp].reg = color;
    }
  }
  return success();
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

std::map<int, std::unordered_set<int>> OpenEdgeASMGen::getPcCtrlFlow() {
  std::map<int, std::unordered_set<int>> pcCtrlFlow;
  int start = getKernelStart();
  int end = getKernelEnd();
  for (int i = start; i <= end; i++) {
    pcCtrlFlow[i] = {};
    auto opList = getOperationsAtTime(i);
    // if opList does not contain any branch ops, pc increment by 1
    for (auto [pe, op] : opList) {
      if (isa<LLVM::BrOp>(op) || isa<cgra::ConditionalBranchOp>(op)) {
        auto block = op->getBlock();
        int time = i;
        int sucTime = getEarliestExecutionTime(block->getSuccessor(0));
        pcCtrlFlow[time].insert(sucTime);
        if (isa<cgra::ConditionalBranchOp>(op)) {
          pcCtrlFlow[time].insert(i + 1);
        }
        break;
      }
      if (isa<LLVM::ReturnOp>(op)) {
        pcCtrlFlow[i].insert(INT_MAX);
        break;
      }
    }
    if (pcCtrlFlow[i].empty())
      pcCtrlFlow[i].insert(i + 1);
  }

  return pcCtrlFlow;
}

LogicalResult OpenEdgeASMGen::allocateRegisters(
    std::map<Operation *, Instruction> restriction) {

  // First write restriction to the solution
  for (auto [op, inst] : restriction) {
    instSolution[op] = inst;
    solution[op].reg = inst.Rout;
  }

  auto pcCtrlFlow = getPcCtrlFlow();
  // print pcCtrlFlow
  for (auto [time, suc] : pcCtrlFlow) {
    llvm::errs() << "Time: " << time << " ->{ ";
    for (auto s : suc)
      llvm::errs() << s << ", ";
    llvm::errs() << "}\n";
  }

  for (size_t pe = 0; pe < nRow * nCol; ++pe) {
    llvm::errs() << "\nPE = " << pe << "\n";
    auto ops = getOperationsAtPE(pe);
    // Allocate registers if it can be inferred from the known result.
    for (auto [time, op] : ops) {
      // Check whether the PE of the operation is known, if yes, skip
      // allocation
      if (solution[op].reg != -1)
        continue;

      // if the operation does not have result, skip
      if (op->getNumResults() == 0)
        continue;

      auto userList = getCntUserIndirectly(op->getResult(0));
      bool allUserOutside = true;
      for (auto user : userList) {
        // Operation *user = getCntUseOpIndirectly(use);
        int userPE = solution[user].pe;
        // If the user is executed in neighbour PE, the result should be
        // stored in Rout.
        if (userPE == pe) {
          allUserOutside = false;
        }

        if (solution[user].reg != -1 &&
            instSolution.find(user) != instSolution.end()) {
          // The result operand should match with use operand
          int leftOpId = 0;
          if (isa<cgra::BsfaOp, cgra::BzfaOp>(user))
            leftOpId = 1;
          std::string regStr =
              leftOpId ? instSolution[user].opA : instSolution[user].opB;
          if (regStr == "ROUT") {
            solution[op].reg = maxReg;
            break;
          } else if (isRegisterDigit(regStr, maxReg)) {
            // the user mast be executed in the same PE so that the register can
            // be internal.
            if (userPE != pe)
              return failure();
            solution[op].reg = std::stoi(regStr.substr(1));
            break;
          }
          // If the correspond PE is set, stop seeking from its other
          // users
        }

        // if the user PE is not restricted, don't allocate register for now
      }
      if (allUserOutside)
        solution[op].reg = maxReg;
    }

    // allocate register for the operations in the PE
    if (failed(allocateOutRegInPE(ops, solution, maxReg, pcCtrlFlow))) {
      llvm::errs() << "Failed to allocate register for PE " << pe << "\n";
      return failure();
    }
    llvm::errs() << "PE " << pe << " register allocation done\n";
  }

  for (auto [op, sol] : solution) {
    if (op->getNumResults() > 0 && sol.reg == -1) {
      llvm::errs() << "Failed" << *op << " " << sol.pe << "\n";
      return failure();
    }
    instSolution[op].name = op->getName().getStringRef().str();
    instSolution[op].pe = sol.pe;
    instSolution[op].time = sol.time;
    instSolution[op].Rout = sol.reg;
  }

  llvm::errs() << "Register allocation done\n";

  // write register allocation results to instructions
  if (failed(convertToInstructionMap()))
    return failure();

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

/// Function for retrieve the operand knowning the src operation and user
/// operand's definition operand's PES.
/// e.g. Suppose %a = op %b, ..., return the string of the %b for %a, such
/// as R0, R1,... or RCT, RCB, RCR, RCL.
static std::string getOperandSrcReg(int peA, int peB, int srcReg, int nRow,
                                    int nCol, unsigned maxReg) {
  if (peA == peB) {
    if (srcReg == maxReg)
      return "ROUT";
    else
      return "R" + std::to_string(srcReg);
  }

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
    int leftId = 0;
    if (isa<cgra::BzfaOp, cgra::BsfaOp>(op))
      leftId = 1;

    // Get defition operation
    if (op->getNumOperands() > leftId && inst.opA == "Unknown") {
      auto producerA =
          getCntDefOpIndirectly(op->getOperand(leftId), op->getBlock())[0];
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

    if (op->getNumOperands() > leftId + 1 && inst.opB == "Unknown") {
      auto producerB =
          getCntDefOpIndirectly(op->getOperand(leftId + 1), op->getBlock())[0];
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
    return "JUMP " + std::to_string(sucTime + baseTime) + ", " +
           std::to_string(startPC);
  }

  std::string ROUT = "";
  if (op->getNumResults() > 0)
    ROUT = instSolution[op].Rout == maxReg
               ? " ROUT,"
               : " R" + std::to_string(instSolution[op].Rout) + ",";

  std::string opA = "";

  int leftId = 0;
  if (isa<cgra::BzfaOp, cgra::BsfaOp>(op)) {
    leftId = 1;
  }

  if (op->getNumOperands() > 0) {
    // check whether the operand is immediate
    auto cntOp =
        getCntDefOpIndirectly(op->getOperand(leftId), op->getBlock())[0];
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(cntOp)) {
      int imm = constOp.getValueAttr().dyn_cast<IntegerAttr>().getInt();
      opA = imm ? " " + std::to_string(imm) : " ZERO";
    } else
      opA = " " + instSolution[op].opA;
  }

  std::string opB = "";
  if (op->getNumOperands() > 1) {
    auto cntOp =
        getCntDefOpIndirectly(op->getOperand(leftId + 1), op->getBlock())[0];
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
    addition = ", " + std::to_string(sucTime + baseTime);
  }

  if (isa<cgra::BzfaOp, cgra::BsfaOp>(op)) {
    auto cntOp = getCntDefOpIndirectly(op->getOperand(0), op->getBlock())[0];
    if (auto constOp = dyn_cast<LLVM::ConstantOp>(cntOp)) {
      int imm = constOp.getValueAttr().dyn_cast<IntegerAttr>().getInt();
      addition = imm ? " " + std::to_string(imm) : " ZERO";
    } else {
      int predicatePE = instSolution[cntOp].pe;
      llvm::errs() << "predicate(" << *cntOp << "): " << instSolution[op].pe
                   << " <- " << predicatePE << "\n";
      llvm::errs() << "opA: " << instSolution[op].opA
                   << " opB: " << instSolution[op].opB << "\n";
      addition =
          ", " + getOperandSrcReg(instSolution[op].pe, instSolution[cntOp].pe,
                                  0, nRow, nCol, maxReg);
    }
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
    llvm::errs() << "Unable to open " << outDir << "\n";
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
readMapFile(std::string mapResult, unsigned maxReg, unsigned numOps, int &II,
            std::map<int, std::unordered_set<int>> &opTimeMap,
            std::vector<std::unordered_set<int>> &bbTimeMap,
            std::map<int, Instruction> &instructions) {
  std::ifstream file(mapResult);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open " << mapResult << "\n";
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

    if (line.find("II: ") != std::string::npos) {
      II = std::stoi(line.substr(4));
      continue;
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
  unsigned nPhi = 0;
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

/// Get the execution time of operations in one loop iteration
static std::map<int, int>
getLoopOpUnfoldExeTime(const std::map<int, std::unordered_set<int>> opTimeMap) {
  std::map<int, int> opExecT;
  std::set<int> opIds;
  for (auto &timeSet : opTimeMap) {
    for (auto opId : timeSet.second)
      if (opIds.count(opId) == 0) {
        opIds.insert(opId);
        opExecT[opId] = timeSet.first;
      }
  }
  return opExecT;
}

/// Function to use the modulo schedule result to initialize existing results in
/// the region and
LogicalResult useModuloScheduleResult(const std::string mapResult, int &II,
                                      const unsigned maxReg,
                                      OpenEdgeKernelScheduler &scheduler,
                                      Block *loopBlock, Region &r,
                                      OpBuilder &builder) {
  int opSize = loopBlock->getOperations().size();

  // read the loop basic block modulo schedule result
  std::map<int, Instruction> instructions;
  std::map<int, std::unordered_set<int>> opTimeMap;
  std::vector<std::unordered_set<int>> bbTimeMap = {{}};
  if (failed(readMapFile(mapResult, maxReg,
                         opSize + loopBlock->getNumArguments() - 1, II,
                         opTimeMap, bbTimeMap, instructions)))
    return failure();

  // init block arguments to be SADD
  if (failed(initBlockArgs(loopBlock, instructions, builder)))
    return failure();

  // update the operation size if the block argument is considered
  opSize = loopBlock->getOperations().size();

  // init modulo schedule result
  if (kernelOverlap(bbTimeMap)) {
    std::map<int, int> execTime = getLoopOpUnfoldExeTime(opTimeMap);
    ModuloScheduleAdapter adapter(r, builder, loopBlock->getOperations(), II,
                                  execTime, opTimeMap, bbTimeMap);
    if (failed(adapter.adaptCFGWithLoopMS()))
      return failure();

    // hash map to store the total round of operation execution, where
    // totalRound[ind] represents ind-th operation's in the loop has been
    // executed totalRound[ind] times
    std::vector<int> totalRound(opSize, 0);

    int curPC = 0;

    auto preParts = adapter.enterDFGs;
    auto postParts = adapter.exitDFGs;

    std::vector<std::vector<int>> preOpIds;
    std::vector<int> bbStarts;
    for (size_t i = 0; i < preParts.size(); i++) {
      bbStarts.push_back(curPC);
      scheduler.assignSchedule(preParts[i], II, curPC, execTime, instructions,
                               totalRound);
      curPC++;
      preOpIds.push_back(totalRound);
    }

    for (int i = 0; i < preParts.size() - 1; i++) {
      scheduler.assignSchedule(postParts[i], II, curPC, execTime, instructions,
                               preOpIds[preParts.size() - 2 - i],
                               curPC - bbStarts[preParts.size() - 1 - i]);
      curPC++;
    }

  } else
    scheduler.assignSchedule(loopBlock->getOperations(), instructions);
  return success();
}

static Operation *getFirstOpInRegion(Region &r) {
  for (auto &block : r.getBlocks()) {
    for (auto &op : block.getOperations()) {
      if (isa<LLVM::ConstantOp>(op))
        continue;
      return &op;
    }
  }
  return nullptr;
}

void readScheduleResult(Region &r, OpenEdgeKernelScheduler &scheduler) {
  std::string mapResult = "/home/yuxuan/Projects/24S/Compigra/build/sha4_0.sol";
  std::ifstream file(mapResult);
  if (!file.is_open()) {
    llvm::errs() << "Unable to open " << mapResult << "\n";
    return;
  }
  std::map<std::string, float> timeDict;
  std::map<std::string, float> peDict;

  std::string line;
  while (std::getline(file, line)) {
    if (line.find("time_") != std::string::npos) {
      auto key = line.substr(5, line.find(" ") - 5);
      auto value = std::stof(line.substr(line.find(" ") + 1));
      timeDict[key] = value;
    }

    if (line.find("pe_") != std::string::npos) {
      auto key = line.substr(3, line.find(" ") - 3);
      auto value = std::stof(line.substr(line.find(" ") + 1));
      peDict[key] = value;
    }
  }
  // Init instruction based on the schedule result
  for (auto [ind, block] : llvm::enumerate(r.getBlocks())) {
    for (auto [idOp, op] : llvm::enumerate(block.getOperations())) {
      if (isa<LLVM::ConstantOp>(op))
        continue;
      auto name = op.getName().getStringRef().str();
      auto t = timeDict[std::to_string(ind) + "_" + std::to_string(idOp)];
      auto pe = peDict[std::to_string(ind) + "_" + std::to_string(idOp)];
      auto instruction = Instruction{name,
                                     static_cast<int>(std::round(t)),
                                     static_cast<int>(std::round(pe)),
                                     -1,
                                     "Unknown",
                                     "Unknown"};
      scheduler.knownRes[&op] = instruction;
      llvm::errs() << op << " " << instruction.time << " " << instruction.pe
                   << "\n";
    }
  }
}

namespace {
struct OpenEdgeASMGenPass
    : public compigra::impl::OpenEdgeASMGenBase<OpenEdgeASMGenPass> {

  explicit OpenEdgeASMGenPass(StringRef funcName, StringRef mapResult,
                              int nGrid) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    char maxReg = 3;
    // initial interval
    int II;

    size_t pos = mapResult.find_last_of("/");
    // Default output file directory
    std::string outDir = mapResult.substr(0, pos + 1) + "out";

    for (auto funcOp :
         llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>())) {
      if (funcOp.getName() != funcName)
        continue;
      Region &r = funcOp.getBody();
      // init scheduler
      auto grid = nGrid.getValue();
      if (!nGrid.hasValue() || nGrid.getValue() <= 0)
        grid = 4;

      OpenEdgeKernelScheduler scheduler(r, maxReg, grid);

      // assign the first operation to PC 0
      if (mapResult.hasValue()) {
        auto loopBlock = getLoopBlock(r);
        if (failed(useModuloScheduleResult(mapResult, II, maxReg, scheduler,
                                           loopBlock, r, builder)))
          return signalPassFailure();
      } else {
        int startPC = 0;
        // DEBUG: READ RESULT FROM FILE
        readScheduleResult(r, scheduler);
        scheduler.knownRes[getFirstOpInRegion(r)] =
            Instruction{"init", startPC, 0, (int)maxReg, "Unknown", "Unknown"};
      }

      // init OpenEdgeASMGen
      OpenEdgeASMGen asmGen(r, maxReg, grid);
      if (failed(scheduler.createSchedulerAndSolve())) {
        llvm::errs() << "Failed to create scheduler and solve\n";
        return signalPassFailure();
      }

      // assign schedule results and produce assembly
      asmGen.setSolution(scheduler.getSolution());
      llvm::errs() << "Allocate Register...\n";
      if (failed(asmGen.allocateRegisters(scheduler.knownRes))) {
        llvm::errs() << "Failed to allocate registers\n";
        return signalPassFailure();
      }
      asmGen.printKnownSchedule(true, 0, outDir);
      llvm::errs() << funcOp << "\n";
    }
  }
};
} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass>
createOpenEdgeASMGen(StringRef funcName, StringRef mapResult, int nGrid) {
  return std::make_unique<OpenEdgeASMGenPass>(funcName, mapResult, nGrid);
}
} // namespace compigra