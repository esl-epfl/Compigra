//===- BasicBlockOpAssignment.cpp - Implements the class/functions to place
// operations of a basic block *- C++-* ----------------------------------===//
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

#include "compigra/Scheduler/BasicBlockOpAssignment.h"

using namespace mlir;
using namespace compigra;

void BasicBlockOpAsisgnment::searchCriticalPath() {
  // first seek root node
  SmallVector<Operation *, 8> roots;
  for (Operation &op : block->getOperations()) {
    bool hasLocalUser = true;
    for (auto user : op.getUsers())
      if (user->getBlock() == block) {
        hasLocalUser = false;
        break;
      }

    if (!hasLocalUser)
      roots.push_back(&op);
  }
  // track from the root to the leaf node

  return;
}
