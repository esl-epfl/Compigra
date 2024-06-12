//===- CfMapToFullPredict.h - Fix index to CGRA PE bitwidth -----*- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --map-to-full-predict pass.
//
//===----------------------------------------------------------------------===//
#ifndef COMPIGRA_CFMAPTOFULLPREDICT_H
#define COMPIGRA_CFMAPTOFULLPREDICT_H

#include "compigra/CgraOps.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

enum class BlockStage { Init = 0, Loop = 1, Fini = 2 };

namespace compigra {
struct bbInfo {
  int index = -1;
  BlockStage stage;
  Block *block;
};

#define GEN_PASS_DEF_CFMAPTOFULLPREDICT
#define GEN_PASS_DECL_CFMAPTOFULLPREDICT
#include "compigra/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCfMapToFullPredict();

} // namespace compigra

#endif // COMPIGRA_CFMAPTOFULLPREDICT_H