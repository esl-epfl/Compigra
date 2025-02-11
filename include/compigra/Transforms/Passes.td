//===-  Passes.h - Passes definition ----------------------------*- C++ -*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_TRANSFORMS_PASSES_TD
#define COMPIGRA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Cf passes
//===----------------------------------------------------------------------===//

def CfFixIndexWidth : Pass<"fix-index-width"> {
  let summary = "Fix the index width to CGRA processing element width";
  let constructor = "compigra::createCfFixIndexWidth()";
  let options = [Option<"bitwidth", "width", "unsigned", "32",
                        "    Specify the width of the index to fix to. ">];
}

def CfMergeIfToSelect : Pass<"merge-if-to-select"> {
  let summary = "Execute the if-then-else block as a select";
  let constructor = "compigra::createCfMergeIfToSelect()";
}

def CfFuseLoopHeadBody : Pass<"fuse-loop"> {
  let summary = "Fuse the loop head and body";
  let constructor = "compigra::createCfFuseLoopHeadBody()";
}

def CfReduceBranches : Pass<"reduce-branches"> {
  let summary = "Reduce the number of branches by applying full predict";
  let constructor = "compigra::createCfReduceBranches()";
}

//===----------------------------------------------------------------------===//
// Cgra passes
//===----------------------------------------------------------------------===//

def CgraFitToOpenEdge : Pass<"fit-openedge"> {
  let summary = " Rewrite ops to allow it fit in open edge ISA ";
  let constructor = "compigra::createCgraFitToOpenEdge()";
  let options = [
    Option<"outputDAG", "out-dag", "std::string", "",
           "    Specify the name of the fucntion to output the SATMapIt DAG in "
           "the format of DATE2023 code. ">
  ];
}

#endif // COMPIGRA_TRANSFORMS_PASSES_TD
