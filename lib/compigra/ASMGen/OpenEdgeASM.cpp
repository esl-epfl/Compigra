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

static bool equalValueSet(std::unordered_set<int> set1,
                          std::unordered_set<int> set2) {
  if (set1.size() != set2.size())
    return false;
  for (auto val : set1)
    if (set2.find(val) == set2.end())
      return false;
  return true;
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

static void getReplicatedOp(Operation *op, OpBuilder &builder) {}

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
    if (isa<LLVM::BrOp, LLVM::ConstantOp>(op))
      continue;
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

  std::vector<Operation *> sortedOps;
  for (auto [t, op] : opList) {
    sortedOps.push_back(op);
    llvm::errs() << "put: " << *op << "\n";
  }
  llvm::errs() << "sortedOps size: " << sortedOps.size() << "\n";
  std::map<Operation *, std::unordered_set<int>> liveIn;
  std::map<Operation *, std::unordered_set<int>> liveOut;
  while (true) {
    bool changed = false;
    for (int i = sortedOps.size() - 1; i >= 0; i--) {
      auto op = sortedOps[i];
      if (i < sortedOps.size() - 1) {
        auto succ = sortedOps[i + 1];
        // Calculate liveOut
        // if liveIn is empty, continue
        if (liveIn.find(succ) == liveIn.end())
          continue;
        for (auto live : liveIn[succ])
          if (liveOut[op].find(live) == liveOut[op].end()) {
            changed = true;
            liveOut[op].insert(live);
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
  graph.printGraph();

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
readMapFile(std::string mapResult, unsigned maxReg, unsigned numOps, int &II,
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

namespace {
struct OpenEdgeASMGenPass
    : public compigra::impl::OpenEdgeASMGenBase<OpenEdgeASMGenPass> {

  explicit OpenEdgeASMGenPass(StringRef funcName, StringRef mapResult) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    OpBuilder builder(&getContext());
    unsigned maxReg = 4;
    // initial interval
    int II;

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
                             opSize + loopBlock->getNumArguments() - 1, II,
                             opTimeMap, bbTimeMap, instructions)))
        return signalPassFailure();

      // init block arguments to be SADD
      if (failed(initBlockArgs(loopBlock, instructions, builder)))
        return signalPassFailure();
      // update the operation size if the block argument is considered
      opSize = loopBlock->getOperations().size();

      // init scheduler
      OpenEdgeKernelScheduler scheduler(r, maxReg, 4);

      // init modulo schedule result
      if (kernelOverlap(bbTimeMap)) {
        ModuloScheduleAdapter adapter(r, builder, loopBlock->getOperations(),
                                      opTimeMap, bbTimeMap);
        adapter.adaptCFGWithLoopMS();
        // llvm::errs() << funcOp << "\n";

        // hash map to store the total round of operation execution, where
        // totalRound[ind] represents ind-th operation's in the loop has been
        // executed totalRound[ind] times
        std::vector<int> totalRound(opSize, 0);
        std::map<int, int> execTime = getLoopOpUnfoldExeTime(opTimeMap);

        int curPC = 0;
        llvm::errs() << "opSize: " << opSize << "\n";
        llvm::errs() << "totalRound: " << totalRound.size() << "\n";

        auto preParts = adapter.enterDFGs;
        auto postParts = adapter.exitDFGs;
        std::vector<std::vector<int>> preOpIds;
        for (size_t i = 0; i < preParts.size(); i++) {
          scheduler.assignSchedule(preParts[i], false, II, curPC, execTime,
                                   instructions, totalRound);
          curPC++;
          preOpIds.push_back(totalRound);
        }

        for (int i = postParts.size() - 1; i >= 0; i--) {
          scheduler.assignSchedule(postParts[i], true, II, curPC, execTime,
                                   instructions, preOpIds[i]);
          curPC++;
        }

      } else {
        scheduler.assignSchedule(loopBlock->getOperations(), instructions);
      }

      // init OpenEdgeASMGen
      OpenEdgeASMGen asmGen(r, maxReg, 4);
      if (failed(scheduler.createSchedulerAndSolve())) {
        llvm::errs() << "Failed to create scheduler and solve\n";
        return signalPassFailure();
      }

      // assign schedule results and produce assembly
      asmGen.setSolution(scheduler.getSolution());
      llvm::errs() << "Allocate Register...\n";
      asmGen.allocateRegisters(scheduler.knownRes);
      asmGen.printKnownSchedule(true, 0, outDir);
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