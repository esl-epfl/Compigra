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

namespace compigra {
using namespace mlir;
/// The modulo scheduler might generate efficient schedule result by overlapping
/// loop with prolog and epilog that does not exist in current CFG. This
/// function adapts the CFG to the schedule result for further whole kernel
/// function scheduling.
/// The modulo scheduling result is described by two maps: opTimeMap and
/// bbTimeMap. The opTimeMap use time(sequentially in PC) as key, where values
/// indicates the the index of the operations to be executed at that time. The
/// bbTimeMap is a vector of basic blocks, where each basic block contains the
/// opearations in the block specified by the opTimeMap.
enum loopStage { init, prolog, loop, epilog, fini };
LogicalResult
adaptCFGWithLoopMS(Region &region, OpBuilder &builder,
                   std::map<int, std::unordered_set<int>> &opTimeMap,
                   std::vector<std::unordered_set<int>> &bbTimeMap);
/// Data structure to map the operation id to the operation
using mapId2Op = std::map<int, Operation *>;
} // namespace compigra

#endif // MODULO_SCHEDULE_ADAPTER_H