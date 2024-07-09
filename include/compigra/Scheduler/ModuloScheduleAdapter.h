//===- ModuloScheduleAdapter.h - Declare adapter for MS ---------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Modulo schduling can change both DFG and CFG of the program. This adapter
// declares functions that rewrite the IR to match the schedule result.
//
//===----------------------------------------------------------------------===//

#ifndef MODULO_SCHEDULE_ADAPTER_H
#define MODULO_SCHEDULE_ADAPTER_H

#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include <unordered_set>

using namespace mlir;

/// The modulo scheduler might generate efficient schedule result by overlapping
/// loop with prolog and epilog that does not exist in current CFG. This
/// function adapts the CFG to the schedule result for further whole kernel
/// function scheduling.
namespace compigra {
LogicalResult
adaptCFGWithLoopMS(Region &region, OpBuilder &builder,
                   std::map<int, std::unordered_set<int>> &opTimeMap,
                   std::vector<std::unordered_set<int>> &bbTimeMap);
}

#endif // MODULO_SCHEDULE_ADAPTER_H