//===-  CfMapToFullPredict.cpp - Fix index to CGRA PE bitwidth --*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --map-to-full-predict pass, which reduce the number of bbs
// by applying full predict on the DAG.
//
//===----------------------------------------------------------------------===//

#include "compigra/Passes/CfMapToFullPredict.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
// #include <set.h>

// for printing debug informations
// #include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace compigra;

enum class OpStage { Init, Loop, Fini };

namespace {
/// Driver for the index bitwidth fix pass.
struct CfMapToFullPredictPass
    : public compigra::impl::CfMapToFullPredictBase<CfMapToFullPredictPass> {
  void runOnOperation() override;
};

void CfMapToFullPredictPass::runOnOperation() {
  std::vector<Operation *> ops;
  std::vector<Block *> blocks;

  Block *initBlock = nullptr;

  getOperation()->walk([&](Operation *op) {
    ops.push_back(op);
  });

  for (auto op : ops) {
    if (auto block = op->getBlock()) {
      if (block->isEntryBlock())
        initBlock = block;
    }
  }
  
  // for (auto op : ops) {
  //   if (auto block = op->getBlock()) {
  //     if (block->isEntryBlock())
  //       continue;
  //     op->moveBefore(&initBlock->back());
  //   }
  // }

  getOperation()->walk([&](Operation *op) {
    llvm::errs() << "op: " << *op << "\n";
  });

  
}

} // namespace

namespace mlir {
namespace compigra {
std::unique_ptr<mlir::Pass> createCfMapToFullPredict() {
  return std::make_unique<CfMapToFullPredictPass>();
}
} // namespace compigra
} // namespace mlir