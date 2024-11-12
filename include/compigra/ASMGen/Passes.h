//===-  Passes.h - Passes registration --------------------------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for Assembly code passes for different CGRA
// architectures.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_ASM_GEN_PASSES_H
#define COMPIGRA_ASM_GEN_PASSES_H

#include "compigra/ASMGen/ASMGenTempCGRA.h"
#include "compigra/ASMGen/OpenEdgeASM.h"

namespace compigra {

#define GEN_PASS_REGISTRATION
#include "compigra/ASMGen/Passes.h.inc"

std::unique_ptr<mlir::Pass>
createOpenEdgeASMGen(StringRef funcName, StringRef mapResult, int nGrid);
std::unique_ptr<mlir::Pass> createASMGenTemporalCGRA(int nRow, int nCol);
} // end namespace compigra
#endif // COMPIGRA_ASM_GEN_PASSES_H
