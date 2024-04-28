//===- InitAllPasses.h - All passes registration -----------------*- C++-*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes defined
// in the Dynamatic tutorials.
//
//===----------------------------------------------------------------------===//

#ifndef INIT_ALL_PASSES_H
#define INIT_ALL_PASSES_H

// #include "compigra/Passes.h"

namespace mlir {
namespace compigra {
#define GEN_PASS_REGISTRATION
#include "compigra/Passes.h.inc"

inline void registerAllPasses() {
  registerPasses();
}
} // namespace compigra
} // namespace mlir

#endif // INIT_ALL_PASSES_H