//===- CgraFitToOpenEdge.cpp -Rewrite ops to fit in HW ISA -*- C++ -----*-===//
//
// Compigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the --fit-openedge pass, which rewrite the operations to fit into
// the openedge CGRA. For example, beq does not support immediate values, so the
// constant value must be loaded into an opeartion.
//
//===----------------------------------------------------------------------===//

#include "compigra/Transforms/CgraFitToOpenEdge.h"
#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "compigra/Transforms/SatMapItDATE2023InputGen/PrintSatMapItDAG.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

// Debugging support
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace compigra;

static bool hasOnlyUser(LLVM::ConstantOp constOp) {
  // Check if the constant operation is used by multiple operations
  auto distance =
      std::distance(constOp->getUsers().begin(), constOp->getUsers().end());
  return distance == 1;
}

static bool isAddrConstOp(LLVM::ConstantOp constOp) {
  // if (constOp->getAttrDictionary().contains("base"))
  //   return true;
  for (auto user : constOp->getUsers())
    if (isa<cgra::LwiOp, cgra::SwiOp>(user))
      return true;

  return false;
}

static cgra::LwiOp convertImmToLwi(LLVM::ConstantOp constOp, int *constBase,
                                   PatternRewriter &rewriter) {
  // auto userOps = constOp->getUsers();
  // rewriter.setInsertionPoint(constOp);
  // host processor value are specified in the hostValue StringAttr
  auto intAttr = constOp->getAttr("value").dyn_cast<IntegerAttr>();
  if (intAttr) {
    std::string strValue = std::to_string(intAttr.getInt());
    auto strAttr = rewriter.getStringAttr(strValue);
    rewriter.modifyOpInPlace(constOp, [&] {
      constOp->setAttr("hostValue", strAttr);
      constOp->setAttr("value", rewriter.getI32IntegerAttr(*constBase));
    });
  }
  // insert a lwi operation to load the constant value
  rewriter.setInsertionPoint(constOp->getBlock()->getTerminator());
  auto lwiOp = rewriter.create<cgra::LwiOp>(
      constOp->getBlock()->getTerminator()->getLoc(), constOp.getType(),
      constOp.getResult());

  *constBase += 4;

  return lwiOp;
}

static LLVM::AddOp generateImmAddOp(LLVM::ConstantOp constOp,
                                    PatternRewriter &rewriter) {
  auto intAttr = constOp->getAttr("value").dyn_cast<IntegerAttr>();
  // insert a new zero constant operation
  rewriter.setInsertionPoint(constOp);
  auto zeroConst = rewriter.create<LLVM::ConstantOp>(
      constOp.getLoc(), constOp.getType(), rewriter.getI32IntegerAttr(0));
  rewriter.setInsertionPoint(constOp->getBlock()->getTerminator());
  // replicate the constant operation in case it is used by multiple operations
  auto immConst = rewriter.create<LLVM::ConstantOp>(
      constOp.getLoc(), constOp.getType(),
      rewriter.getI32IntegerAttr(intAttr.getInt()));
  auto addOp = rewriter.create<LLVM::AddOp>(
      constOp->getBlock()->getTerminator()->getLoc(), constOp.getType(),
      zeroConst.getResult(), immConst.getResult());
  return addOp;
}

namespace {
// Initialze the constant target that can not be deployed in the openedge CGRA
struct ConstTarget : public ConversionTarget {
  ConstTarget(MLIRContext *ctx) : ConversionTarget(*ctx) {
    addLegalDialect<cgra::CgraDialect>();
    addDynamicallyLegalDialect<LLVM::LLVMDialect>(
        [&](Operation *op) { return !isa<LLVM::ConstantOp>(op); });

    // add the operation to the target
    addDynamicallyLegalOp<LLVM::ConstantOp>([&](LLVM::ConstantOp constOp) {
      auto value = constOp.getValue().cast<IntegerAttr>().getInt();
      bool isAddrOnly = isAddrConstOp(constOp) && hasOnlyUser(constOp);

      // if the value exceed the Imm range
      if (!isAddrOnly)
        if (value < -4097 || value > 4096)
          return false;

      // if the constant is used by beq, bne, blt, bge, it is not legal
      for (auto user : constOp->getUsers())
        if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(user))
          // if it is used for comparison, it is not legal
          if (user->getOperand(0).getDefiningOp() == constOp ||
              user->getOperand(1).getDefiningOp() == constOp)
            return false;

      return true;
    });
  }
};

struct ConstantOpRewrite : public OpRewritePattern<LLVM::ConstantOp> {

  ConstantOpRewrite(MLIRContext *ctx, int *constBase)
      : OpRewritePattern(ctx), constBase(constBase) {}

  LogicalResult matchAndRewrite(LLVM::ConstantOp constOp,
                                PatternRewriter &rewriter) const override {

    auto value = constOp.getValue().cast<IntegerAttr>().getInt();
    rewriter.modifyOpInPlace(constOp, [&] {
      constOp->setAttr("value", rewriter.getI32IntegerAttr(value));
    });

    // rewrite the constant operation if the value exceed the range
    if (value < -4097 || value > 4096) {
      auto lwiOp = convertImmToLwi(constOp, constBase, rewriter);
      // replace all the use of the constant operation except the lwi
      // operation
      llvm::errs() << "Replace the use of the constant operation\n";
      rewriter.replaceOpUsesWithIf(
          constOp, lwiOp.getResult(),
          [&](OpOperand &operand) { return operand.getOwner() != lwiOp; });
    }

    for (auto user : constOp->getUsers())
      // rewrite the constant operation if it is used by beq, bne, blt, bge,
      if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(user)) {
        // Only rewrite if the constant operation is used by the compare
        // operation, otherwise propagate the branch arguments
        auto addOp = generateImmAddOp(constOp, rewriter);
        user->replaceUsesOfWith(constOp.getResult(), addOp.getResult());
      }

    return success();
  }

protected:
  int *constBase;
};

static LogicalResult raiseConstOperation(cgra::FuncOp funcOp) {
  Operation &beginOp = *funcOp.getOps().begin();
  // raise the constant operation to the top level
  for (auto op :
       llvm::make_early_inc_range(funcOp.getOps<LLVM::ConstantOp>())) {
    if (op->getBlock()->isEntryBlock()) {
      op->moveBefore(&beginOp);
    }
  }

  return success();
}

static LogicalResult removeUnusedConstOp(cgra::FuncOp funcOp) {
  // remove the constant operation if it is not used
  for (auto op :
       llvm::make_early_inc_range(funcOp.getOps<LLVM::ConstantOp>())) {
    if (op->use_empty())
      op->erase();
  }

  return success();
}

static bool isLoopBlock(Block *blk) {
  for (auto sucBlk : blk->getSuccessors())
    if (sucBlk == blk)
      return true;
  return false;
}

static LogicalResult outputDATE2023DAG(cgra::FuncOp funcOp,
                                       std::string outputDAG) {

  Block *loopBlk = nullptr;
  Block *initBlk = nullptr;
  // Find the loop block
  for (auto &blk : funcOp.getBlocks())
    if (isLoopBlock(&blk)) {
      loopBlk = &blk;
      break;
    }
  // Get the oeprations in the loop block
  SmallVector<Operation *> nodes;
  for (Operation &op : loopBlk->getOperations()) {
    nodes.push_back(&op);
  }
  llvm::errs() << "The number of operations: " << nodes.size() << "\n";

  // initialize print function
  satmapit::PrintSatMapItDAG printer(loopBlk->getTerminator(), nodes);
  printer.init();
  if (failed(printer.printDAG(outputDAG)))
    return failure();

  return success();
}

/// Driver for the fit-openedge pass.
struct CgraFitToOpenEdgePass
    : public compigra::impl::CgraFitToOpenEdgeBase<CgraFitToOpenEdgePass> {

  explicit CgraFitToOpenEdgePass(StringRef outputDAG) {}

  void runOnOperation() override {
    ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns{ctx};
    ConstTarget target(ctx);

    int BaseAddr = 64;
    patterns.add<ConstantOpRewrite>(ctx, &BaseAddr);

    // adapt the constant operation to meet the requirement of Imm field of
    // openedge CGRA
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();

    // raise the constant operation to the top level
    for (auto funcOp : llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>()))
      if (failed(raiseConstOperation(funcOp)) ||
          failed(removeUnusedConstOp(funcOp)))
        signalPassFailure();

    // print the DAG of the specified function
    if (!outputDAG.empty()) {
      size_t lastSlashPos = outputDAG.find_last_of("/");
      bool isPath = lastSlashPos != StringRef::npos;
      StringRef funcName =
          isPath ? outputDAG.substr(lastSlashPos + 1) : outputDAG;

      for (auto funcOp :
           llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>()))
        if (funcName == funcOp.getName() &&
            failed(outputDATE2023DAG(funcOp, outputDAG))) {
          llvm::errs() << funcOp << "\n";
          return signalPassFailure();
        }
    }
  }
};

} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createCgraFitToOpenEdge(StringRef outputDAG) {
  return std::make_unique<CgraFitToOpenEdgePass>(outputDAG);
}
} // namespace compigra