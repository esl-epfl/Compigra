//===-  CfFuseLoopHeadBody.h - Fuse the loop head and body --*- C++ -----*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares the --fuse-loop pass, which fuses the head and body of a loop into
// one basic block.
//
//===----------------------------------------------------------------------===//
#ifndef COMPIGRA_CFFUSELOOPHEADBODY_H
#define COMPIGRA_CFFUSELOOPHEADBODY_H

#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace compigra {
#define GEN_PASS_DEF_CFFUSELOOPHEADBODY
#define GEN_PASS_DECL_CFFUSELOOPHEADBODY
#include "compigra/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCfFuseLoopHeadBody();
} // end namespace compigra

#endif // COMPIGRA_CFFUSELOOPHEADBODY_H