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
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>

// Debugging support
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace compigra;

bool isValidImmAddr(LLVM::ConstantOp constOp) {
  // Address can not exceeds 13 bits
  int maxAddr = 0b1111111111111;
  auto value = constOp.getValue().cast<IntegerAttr>().getInt();
  return value <= maxAddr;
};

Operation *existsConstant(int intVal, SmallVector<Operation *> &insertedOps) {
  for (auto &op : insertedOps) {
    if (!op->getAttrDictionary().contains("constant"))
      continue;
    auto produceVal = op->getAttr("constant").dyn_cast<IntegerAttr>().getInt();
    if (produceVal == intVal) {
      return op;
    }
  }
  return nullptr;
}

bool isAddrConstOp(LLVM::ConstantOp constOp) {
  for (auto &use : constOp->getUses()) {
    if (isa<cgra::LwiOp>(use.getOwner()))
      return true;
    // Only address can be used as Imm field of swi operation
    if (use.getOperandNumber() == 1 && isa<cgra::SwiOp>(use.getOwner()))
      return true;
  }

  return false;
}

LogicalResult raiseConstOperation(cgra::FuncOp funcOp) {
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

LogicalResult removeUnusedConstOp(cgra::FuncOp funcOp) {
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

// TODO[@Yuxuan]: check the signess of the produced operations
Operation *generateValidConstant(LLVM::ConstantOp constOp,
                                 PatternRewriter &rewriter) {
  int32_t valAttr =
      (int32_t)constOp->getAttr("value").dyn_cast<IntegerAttr>().getInt();
  // Get the lowest 0-11 bits
  const int32_t mask0 = 0b111111111111;
  const int32_t mask1 = 0b111111111111000000000000;
  const int32_t mask2 = 0b11111111000000000000000000000000;
  if (!(mask1 & valAttr))
    return nullptr;

  int32_t part0 = mask0 & valAttr;
  rewriter.setInsertionPoint(constOp);
  auto lowerBits = rewriter.create<LLVM::ConstantOp>(
      constOp->getLoc(), constOp.getResult().getType(),
      rewriter.getI32IntegerAttr(part0));
  rewriter.modifyOpInPlace(constOp, [&] {
    constOp->setAttr("value", rewriter.getI32IntegerAttr(part0));
  });

  int32_t part1 = mask1 & valAttr;
  part1 = part1 >> 12;
  auto midBitVal = rewriter.create<LLVM::ConstantOp>(
      constOp->getLoc(), constOp.getResult().getType(),
      rewriter.getI32IntegerAttr(part1));
  auto zeroOp = rewriter.create<LLVM::ConstantOp>(
      constOp->getLoc(), constOp.getResult().getType(),
      rewriter.getI32IntegerAttr(0));
  // Generate midBits through add operation
  auto midBits = rewriter.create<LLVM::AddOp>(
      constOp->getLoc(), constOp.getResult().getType(), midBitVal.getResult(),
      zeroOp.getResult());

  auto shiftImm1 = rewriter.create<LLVM::ConstantOp>(
      constOp->getLoc(), constOp.getResult().getType(),
      rewriter.getI32IntegerAttr(12));
  auto shiftOp1 = rewriter.create<LLVM::ShlOp>(
      constOp->getLoc(), constOp.getResult().getType(), midBits.getResult(),
      shiftImm1.getResult());
  auto addOp = rewriter.create<LLVM::AddOp>(
      constOp->getLoc(), constOp.getResult().getType(), lowerBits.getResult(),
      shiftOp1.getResult());

  // If the higher bits are zero
  if (!(mask2 & valAttr)) {
    addOp->setAttr("constant", rewriter.getI32IntegerAttr(valAttr));
    rewriter.replaceOpUsesWithIf(
        constOp, addOp.getResult(),
        [&](OpOperand &operand) { return operand.getOwner() != addOp; });
    return addOp;
  }

  int32_t part2 = mask2 & valAttr;
  part2 = part2 >> 24;
  auto highBitVal = rewriter.create<LLVM::ConstantOp>(
      constOp->getLoc(), constOp.getResult().getType(),
      rewriter.getI32IntegerAttr(part2));
  auto zeroOp2 = rewriter.create<LLVM::ConstantOp>(
      constOp->getLoc(), constOp.getResult().getType(),
      rewriter.getI32IntegerAttr(0));
  // Generate highBits through add operation
  auto highBits = rewriter.create<LLVM::AddOp>(
      constOp->getLoc(), constOp.getResult().getType(), highBitVal.getResult(),
      zeroOp2.getResult());
  auto shiftImm2 = rewriter.create<LLVM::ConstantOp>(
      constOp->getLoc(), constOp.getResult().getType(),
      rewriter.getI32IntegerAttr(24));
  auto shiftOp2 = rewriter.create<LLVM::ShlOp>(
      constOp->getLoc(), constOp.getResult().getType(), highBits.getResult(),
      shiftImm2.getResult());
  auto sumOp = rewriter.create<LLVM::AddOp>(
      constOp->getLoc(), constOp.getResult().getType(), shiftOp1.getResult(),
      shiftOp2.getResult());

  sumOp->setAttr("constant", rewriter.getI32IntegerAttr(valAttr));
  rewriter.replaceOpUsesWithIf(
      constOp, sumOp.getResult(),
      [&](OpOperand &operand) { return operand.getOwner() != sumOp; });
  return sumOp;
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

LLVM::AddOp generateImmAddOp(LLVM::ConstantOp constOp,
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
  // set the imm attribute of the operation
  addOp->setAttr("constant", intAttr);
  return addOp;
}

namespace {
ConstTarget::ConstTarget(MLIRContext *ctx) : ConversionTarget(*ctx) {
  addLegalDialect<cgra::CgraDialect>();
  addDynamicallyLegalDialect<LLVM::LLVMDialect>(
      [&](Operation *op) { return !isa<LLVM::ConstantOp>(op); });

  // add the operation to the target
  addDynamicallyLegalOp<LLVM::ConstantOp>([&](LLVM::ConstantOp constOp) {
    auto valAttr = constOp.getValue().cast<IntegerAttr>();
    // if not integer attribute, mark as valid
    if (!valAttr)
      return true;

    int value = valAttr.getInt();
    bool isAddr = isAddrConstOp(constOp);
    bool validAddr = isValidImmAddr(constOp);
    if (isAddr && !validAddr)
      return false;

    // if the value exceed the Imm range
    if (!isAddr || (isAddr && !constOp->hasOneUse()))
      if (value < -4097 || value > 4096)
        return false;

    for (auto user : constOp->getUsers()) {
      if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(user))
        //  if the constant is used by beq, bne, blt, bge, and is used for
        //  comparison, it is not legal
        if (user->getOperand(0).getDefiningOp() == constOp ||
            user->getOperand(1).getDefiningOp() == constOp)
          return false;
      // if the constant is used by swi for imm store, it is not legal
      if (isa<cgra::SwiOp>(user) &&
          user->getOperand(0).getDefiningOp() == constOp)
        return false;
    }

    return true;
  });
}

LogicalResult
ConstantOpRewrite::matchAndRewrite(LLVM::ConstantOp constOp,
                                   PatternRewriter &rewriter) const {

  auto value = constOp.getValue().cast<IntegerAttr>().getInt();
  rewriter.modifyOpInPlace(constOp, [&] {
    constOp->setAttr("value", rewriter.getI32IntegerAttr(value));
  });

  // indirectly load & store cannot exceed reserved address range
  // if the constant is used as immediate for address
  if (isAddrConstOp(constOp) && !isValidImmAddr(constOp)) {
    if (auto validOp = existsConstant(value, insertedOps)) {
      // set it to valid range, to be removed later on
      constOp->setAttr("value", rewriter.getI32IntegerAttr(0));
      rewriter.replaceAllOpUsesWith(constOp, validOp);
    } else {
      auto addOp = generateValidConstant(constOp, rewriter);
      insertedOps.push_back(addOp);
    }
    return success();
  }

  // rewrite the constant operation if the value exceed the range
  if (value < -4097 || value > 4096) {
    if (auto validOp = existsConstant(value, insertedOps))
      rewriter.replaceAllOpUsesWith(constOp, validOp);
    else {
      auto addOp = generateValidConstant(constOp, rewriter);
      insertedOps.push_back(addOp);
    }
  }

  for (auto user : llvm::make_early_inc_range(constOp->getUsers()))
    if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp, cgra::SwiOp>(
            user)) {
      // First seek whether exists operation produce the same result
      Operation *reUseOp = existsConstant(value, insertedOps);
      // If exists, replace the use of the constant operation
      if (reUseOp) {
        user->replaceUsesOfWith(constOp.getResult(), reUseOp->getResult(0));
      } else {
        // If not, create a new operation
        auto addOp = generateImmAddOp(constOp, rewriter);
        user->replaceUsesOfWith(constOp.getResult(), addOp.getResult());
        insertedOps.push_back(addOp);
      }
    }

  return success();
}

void CgraFitToOpenEdgePass::runOnOperation() {
  ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns{ctx};
  ConstTarget target(ctx);

  int BaseAddr = 64;
  SmallVector<Operation *> insertedOps;
  patterns.add<ConstantOpRewrite>(ctx, &BaseAddr, insertedOps);

  // adapt the constant operation to meet the requirement of Imm field of
  // openedge CGRA
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
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
         llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>())) {
      llvm::errs() << funcOp << "\n";
      if (funcName == funcOp.getName() &&
          failed(outputDATE2023DAG(funcOp, outputDAG)))

        return signalPassFailure();
    }
  }
}

} // namespace

namespace compigra {
std::unique_ptr<mlir::Pass> createCgraFitToOpenEdge(StringRef outputDAG) {
  return std::make_unique<CgraFitToOpenEdgePass>(outputDAG);
}
} // namespace compigra