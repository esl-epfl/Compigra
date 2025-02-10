//===- ASMGenOpenEdge.h - Declares the functions for temporal CGRA ASM
// generation *- C++ -*---------------------------------------------------===//
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

#ifndef ASM_GEN_TEMPORAL_CGRA_H
#define ASM_GEN_TEMPORAL_CGRA_H

#include "compigra/ASMGen/InterferenceGraphCreation.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "compigra/Scheduler/TemporalCGRAScheduler.h"
#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#include <unordered_set>

using namespace mlir;
using namespace compigra;

namespace compigra {
#define GEN_PASS_DEF_ASMGENTEMPORALCGRA
#define GEN_PASS_DECL_ASMGENTEMPORALCGRA
#include "compigra/ASMGen/Passes.h.inc"
std::unique_ptr<mlir::Pass> createASMGenTemporalCGRA(int nRow = 3, int nCol = 3,
                                                     int mem = 0,
                                                     StringRef asmOutDir = "");
} // namespace compigra

#endif // ASM_GEN_TEMPORAL_CGRA_H
