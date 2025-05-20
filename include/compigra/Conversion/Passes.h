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

#include "compigra/Conversion/CfToCgraConversion.h"
#include "compigra/Conversion/LLVMToCgraConversion.h"
#include "compigra/Conversion/SCFToCFConversion.h"

using namespace mlir;
using namespace compigra;

namespace compigra {

#define GEN_PASS_REGISTRATION
#include "compigra/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createSCFToControlFlowPass();
std::unique_ptr<mlir::Pass> createLLVMToCgraConversion(StringRef funcName,
                                                       StringRef memAlloc);
std::unique_ptr<mlir::Pass> createCfToCgraConversion();

} // end namespace compigra
#endif // COMPIGRA_CONVERSION_PASSES_H
