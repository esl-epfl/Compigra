//===- CfMergeIfToSelect.h - Fix index to CGRA PE bitwidth ------*- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --merge-if-to-select pass.
//
//===----------------------------------------------------------------------===//
#ifndef COMPIGRA_CFMERGEIFTOSELECT_H
#define COMPIGRA_CFMERGEIFTOSELECT_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

enum class BlockStage { Init = 0, Loop = 1, Fini = 2 };

namespace compigra {
/// detect one of the following topology, if src1 or src2 is the ancestor of the
/// other, return the ancestor block.
///        -------------
///       |  Ancestor |
///       --------------
///             |
///             |
///       ------ -------
///       |            |
///       |            |
///       V            V
///    --------    --------
///    | SRC1 |    | SRC2 |
///    --------    --------
///       |            |
///       |            |
///       |------------|
///             |
///             V
///        ------------
///        |  SUCCBLK |
//        ------------
/// Return the Ancester block if the topology is detected, otherwise return
/// nullopt
std::optional<Block *> getCommonAncestor(Block *src1, Block *src2);

/// check whether the topology match the one of followings
///   (A) ____________           (B)  ____________
///       |   mergeBB  |             |   mergeBB  |
///       |____(SRC1)__|             |____________|
///             |                           |
///       |------------|             |------------|
///       |            |             |            |
///       |            V             |            V
///       |        --------       --------     --------
///       V        | SRC2 |       | SRC1 |     | SRC2 |
///       |        --------       --------     --------
///       |____________|             |____________|
///             |                           |
///         ____V_____                  ____V_____
///         |         |                |         |
///         |  DstBLK |                |  DstBLK |
///         |_________|                |_________|
/// If the topology match, merge the two paths into one path with
/// arith::SelectOp
LogicalResult mergeTwoExecutePath(Block *mergeBB, Block *src1, Block *src2,
                                  OpBuilder &builder);

/// If two BBs are unconditional branches, merge them into one BB
///  ___________
///  |   BB1    |                 ______________
///  |____br1___|    adapted      | ops in      |
///       |          =======>     | BB1(w/o br1)|
///       |                       |             |
///  _____V______                 | ops in      |
///  |   BB2    |                 |  BB2        |
///  |__________|                 |_____________|

LogicalResult mergeSinglePathBBs(Block *srcBB, Block *dstBB,
                                 OpBuilder &builder);

#define GEN_PASS_DEF_CFMERGEIFTOSELECT
#define GEN_PASS_DECL_CFMERGEIFTOSELECT
#include "compigra/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createCfMergeIfToSelect();

} // namespace compigra

#endif // COMPIGRA_CFMERGEIFTOSELECT_H