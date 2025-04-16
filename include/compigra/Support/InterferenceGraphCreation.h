//===- InterferenceGraphCreation.h - Funcs for IG gen *- C++ ------------*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for interference graph generation in CGRA PE.
//
//===----------------------------------------------------------------------===//

#include "compigra/Support/Utils.h"
#include "mlir/IR/Value.h"
#include <map>
#include <unordered_set>

#ifndef INTERFERENCE_GRAPH_CREATION_H
#define INTERFERENCE_GRAPH_CREATION_H

using namespace mlir;

bool doesADominateB(Operation *opA, Operation *opB, Operation *topLevelOp);

namespace compigra {
int getValueIndex(Value val,
                  const std::map<int, std::pair<Operation *, Value>> opMap);
SmallVector<Value, 2> getSrcOprandsOfPhi(BlockArgument arg,
                                         bool eraseUse = false);

/// Get the successor block of the user of the `val` as block argument.
SmallVector<Block *, 4> getCntBlocksThroughPhi(Value val);

/// Get the corresponding block argument of the `val` in the `succBlk`.
BlockArgument getCntBlockArgument(Value val, Block *succBlk);

/// Interference graph created for the PE in CGRA of internal register
/// allocation.
template <typename T> class InterferenceGraph {
public:
  InterferenceGraph() {}

  void addVertex(T v) {
    // Check if the vertex is already in the graph
    if (adjList.find(v) == adjList.end()) {
      adjList[v]; // add an empty set
    }
  }

  // Initialize the vertex, which is the vertex that belongs to this PE and need
  // to be considered for register allocation.
  // void initVertex(T v) { vertices.push_back(v); }

  void addEdge(T v1, T v2) {
    addVertex(v1);
    addVertex(v2);
    adjList[v1].insert(v2);
    adjList[v2].insert(v1);
  }

  void printGraph() {
    for (auto &v : adjList) {
      llvm::errs() << "Vertex " << v.first << " -> ";
      for (auto &u : v.second) {
        llvm::errs() << u << " ";
      }
      llvm::errs() << "\n";
    }
  }

  bool interference(T v1, T v2);

  // bool needColor(T v) {
  //   if (adjList.find(v) == adjList.end())
  //     return false;
  //   return true;
  // }
  // Vertices of the graph are interferring with other nodes, but does not
  // necessary belong to this PE. vertices records the vertices in the graph and
  // belong to this PE which need to be considered for register allocation.
  // std::vector<T> vertices;
  std::map<T, std::unordered_set<T>> adjList;
  std::map<T, char> colorMap;
};

/// Create the interference graph for the operations in the PE, the
/// corresponding relations with the abstract operands are stored in the
/// opMap.
InterferenceGraph<int>
createInterferenceGraph(std::map<int, mlir::Operation *> &opList,
                        std::map<int, std::pair<Operation *, Value>> &defMap,
                        std::map<int, std::unordered_set<int>> ctrlFlow);

/// Get the successor operations of the `op`. The control flow of CGRA could be
/// controlled by other PEs, which branch direction is given by ctrlFlow.
SmallVector<Operation *>
getSuccOps(Operation *op, const std::map<int, mlir::Operation *> &opList,
           std::map<int, std::unordered_set<int>> ctrlFlow);

/// Get the successor operations of the operation in the PE.
std::map<Operation *, std::unordered_set<Operation *>>
getSuccessorMap(const std::map<int, mlir::Operation *> &opList,
                const std::map<int, std::unordered_set<int>> ctrlFlow);
} // namespace compigra

#endif
