//===-  CfFuseLoopHeadBody.cpp - Fuse the loop head and body --*- C++ --*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --fuse-loop pass, which fuses the head and body of a loop into
// one basic block.
//
//===----------------------------------------------------------------------===//

#include "compigra/Transforms/CfFuseLoopHeadBody.h"

namespace {
/// Driver for the cf DAG rewrite pass.
struct CfFuseLoopHeadBodyPass
    : public compigra::impl::CfFuseLoopHeadBodyBase<CfFuseLoopHeadBodyPass> {

  void runOnOperation() override{};
};

} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createCfFuseLoopHeadBody() {
  return std::make_unique<CfFuseLoopHeadBodyPass>();
}
} // namespace compigra