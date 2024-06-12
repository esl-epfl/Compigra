//===-  Passes.h - Passes registration --------------------------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_CONVERSION_PASSES_H
#define COMPIGRA_CONVERSION_PASSES_H

#include "compigra/Conversion/LLVMToCgraConversion.h"

namespace compigra {

#define GEN_PASS_REGISTRATION
#include "compigra/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLLVMToCgraConversion(StringRef outputDAG,
                                                       StringRef memAlloc);
} // end namespace compigra
#endif // COMPIGRA_CONVERSION_PASSES_H
