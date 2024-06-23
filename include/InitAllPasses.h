//===- InitAllPasses.h - All passes registration -----------------*- C++-*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes
//
//===----------------------------------------------------------------------===//

#ifndef INIT_ALL_PASSES_H
#define INIT_ALL_PASSES_H

#include "compigra/Conversion/Passes.h"
#include "compigra/Transforms/Passes.h"
#include "compigra/ASMGen/Passes.h"

namespace compigra {

inline void registerAllPasses() {
  registerConversionPasses();
  registerPasses();
  registerASMGenPasses();
}
} // namespace compigra

#endif // INIT_ALL_PASSES_H