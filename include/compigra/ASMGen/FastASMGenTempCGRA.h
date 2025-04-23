//===- FastASMGenOpenEdge.h - Declares the functions for temporal CGRA assembly
// fast generation *- C++ -*-----------------------------------------------===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements fast assembly generation functions for OpenEdge.
//
//===----------------------------------------------------------------------===//

#ifndef FAST_ASM_GEN_TEMPORAL_CGRA_H
#define FAST_ASM_GEN_TEMPORAL_CGRA_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "compigra/Scheduler/KernelSchedule.h"
#include "compigra/Support/InterferenceGraphCreation.h"
#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#include <unordered_set>

using namespace mlir;

namespace compigra {
#define GEN_PASS_DEF_FASTASMGENTEMPORALCGRA
#define GEN_PASS_DECL_FASTASMGENTEMPORALCGRA
#include "compigra/ASMGen/Passes.h.inc"
std::unique_ptr<mlir::Pass>
createFastASMGenTemporalCGRA(int nRow = 3, int nCol = 3, int mem = 0,
                             StringRef msOpt = "", StringRef asmOutDir = "");
} // namespace compigra

#endif // FAST_ASM_GEN_TEMPORAL_CGRA_H
