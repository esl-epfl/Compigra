//===- CgraFitToHW.h - Rewrite ops to allow it fit in HW ISA *- C++ -----*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --fit-openedge pass.
//
//===----------------------------------------------------------------------===//

#ifndef CGRA_FIT_TO_OPENEDGE_H
#define CGRA_FIT_TO_OPENEDGE_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"

using namespace mlir;

namespace compigra {
#define GEN_PASS_DEF_CGRAFITTOOPENEDGE
#define GEN_PASS_DECL_CGRAFITTOOPENEDGE
#include "compigra/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCgraFitToOpenEdge(StringRef outputDAG="");
} // end namespace compigra

#endif // CGRA_FIT_TO_OPENEDGE_H