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

/// remove the constant operation if it is not used
LogicalResult removeUnusedConstOp(cgra::FuncOp funcOp) {
  for (auto op :
       llvm::make_early_inc_range(funcOp.getOps<LLVM::ConstantOp>())) {
    if (op->use_empty())
      op->erase();
  }

  return success();
}

LogicalResult removeEqualWidthBWOp(cgra::FuncOp funcOp) {
  for (auto op : llvm::make_early_inc_range(funcOp.getOps<LLVM::SExtOp>())) {
    if (op.getOperand().getType() == op.getResult().getType()) {
      if (op.getOperand().getDefiningOp())
        op.replaceAllUsesWith(op.getOperand().getDefiningOp());
      op->erase();
    }
  }

  return success();
}

static bool isLoopBlock(Block *blk) {
  for (auto sucBlk : blk->getSuccessors())
    if (sucBlk == blk)
      return true;
  return false;
}

Operation *generateValidConstant(LLVM::ConstantOp constOp,
                                 PatternRewriter &rewriter) {
  int32_t valAttr =
      (int32_t)constOp->getAttr("value").dyn_cast<IntegerAttr>().getInt();

  Location loc = constOp->getLoc();
  // check whether all the user of the consOp is in exit block, if true, set the
  // insertion location to the exit block
  bool allExit = true;
  for (auto user : constOp->getUsers())
    if (!isa<LLVM::ReturnOp>(user->getBlock()->getTerminator())) {
      allExit = false;
      break;
    }
  if (allExit) {
    auto firstOp = *constOp->getUsers().begin();
    rewriter.setInsertionPoint(firstOp);
    loc = firstOp->getLoc();
  } else {
    rewriter.setInsertionPoint(constOp);
  }

  // get the lowest 0-11 bits
  const int32_t mask0 = 0b111111111111;
  const int32_t mask1 = 0b111111111111000000000000;
  const int32_t mask2 = 0b11111111000000000000000000000000;
  if (!(mask1 & valAttr))
    return nullptr;

  int32_t part0 = mask0 & valAttr;

  auto lowerBits = rewriter.create<LLVM::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(part0));
  rewriter.updateRootInPlace(constOp, [&] {
    constOp->setAttr("value", rewriter.getI32IntegerAttr(part0));
  });

  int32_t part1 = mask1 & valAttr;
  part1 = part1 >> 12;
  auto midBitVal = rewriter.create<LLVM::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(part1));
  auto zeroOp = rewriter.create<LLVM::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(0));
  // Generate midBits through add operation
  auto midBits =
      rewriter.create<LLVM::AddOp>(loc, constOp.getResult().getType(),
                                   midBitVal.getResult(), zeroOp.getResult());

  auto shiftImm1 = rewriter.create<LLVM::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(12));
  auto shiftOp1 =
      rewriter.create<LLVM::ShlOp>(loc, constOp.getResult().getType(),
                                   midBits.getResult(), shiftImm1.getResult());
  auto addOp =
      rewriter.create<LLVM::AddOp>(loc, constOp.getResult().getType(),
                                   lowerBits.getResult(), shiftOp1.getResult());

  // If the higher bits are zero
  if (!(mask2 & valAttr)) {
    addOp->setAttr("constant", rewriter.getI32IntegerAttr(valAttr));
    rewriter.replaceUsesWithIf(
        constOp, addOp.getResult(),
        [&](OpOperand &operand) { return operand.getOwner() != addOp; });
    return addOp;
  }

  int32_t part2 = mask2 & valAttr;
  part2 = part2 >> 24;
  auto highBitVal = rewriter.create<LLVM::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(part2));
  auto zeroOp2 = rewriter.create<LLVM::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(0));
  // Generate highBits through add operation
  auto highBits =
      rewriter.create<LLVM::AddOp>(loc, constOp.getResult().getType(),
                                   highBitVal.getResult(), zeroOp2.getResult());
  auto shiftImm2 = rewriter.create<LLVM::ConstantOp>(
      loc, constOp.getResult().getType(), rewriter.getI32IntegerAttr(24));
  auto shiftOp2 =
      rewriter.create<LLVM::ShlOp>(loc, constOp.getResult().getType(),
                                   highBits.getResult(), shiftImm2.getResult());
  auto sumOp =
      rewriter.create<LLVM::AddOp>(loc, constOp.getResult().getType(),
                                   addOp.getResult(), shiftOp2.getResult());

  sumOp->setAttr("constant", rewriter.getI32IntegerAttr(valAttr));
  rewriter.replaceUsesWithIf(
      constOp, sumOp.getResult(),
      [&](OpOperand &operand) { return operand.getOwner() != sumOp; });
  return sumOp;
}

static LogicalResult outputDATE2023DAG(cgra::FuncOp funcOp,
                                       std::string outputDAG) {

  Block *loopBlk = nullptr;
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

  Location loc = constOp->getLoc();
  // check whether all the user of the consOp is in exit block, if true, set the
  // insertion location to the exit block
  bool allExit = true;
  for (auto user : constOp->getUsers())
    if (!isa<LLVM::ReturnOp>(user->getBlock()->getTerminator())) {
      allExit = false;
      break;
    }
  if (allExit) {
    auto firstOp = *constOp->getUsers().begin();
    rewriter.setInsertionPoint(firstOp);
    loc = firstOp->getLoc();
  } else {
    rewriter.setInsertionPoint(constOp);
  }
  // insert a new zero constant operation
  rewriter.setInsertionPoint(constOp);
  auto zeroConst = rewriter.create<LLVM::ConstantOp>(
      loc, constOp.getType(), rewriter.getI32IntegerAttr(0));
  rewriter.setInsertionPoint(constOp->getBlock()->getTerminator());
  // replicate the constant operation in case it is used by multiple operations
  auto immConst = rewriter.create<LLVM::ConstantOp>(
      loc, constOp.getType(), rewriter.getI32IntegerAttr(intAttr.getInt()));
  auto addOp = rewriter.create<LLVM::AddOp>(
      loc, constOp.getType(), zeroConst.getResult(), immConst.getResult());
  // set the imm attribute of the operation
  addOp->setAttr("constant", intAttr);
  return addOp;
}

static bool isI32IntegerType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    if (intType.getWidth() == 32) {
      return true;
    }
  return false;
}

namespace {
OpenEdgeISATarget::OpenEdgeISATarget(MLIRContext *ctx)
    : ConversionTarget(*ctx) {
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
      if (isa<LLVM::BrOp, cgra::ConditionalBranchOp>(user))
        // The branch operation cannot use immediate values (Imm) for
        // computation, nor can it propagate constants.
        return false;
      // if the constant is used by swi for imm store, it is not legal
      if (isa<cgra::SwiOp>(user) &&
          user->getOperand(0).getDefiningOp() == constOp)
        return false;
    }

    return true;
  });

  addDynamicallyLegalOp<LLVM::SExtOp>([&](LLVM::SExtOp extOp) {
    return extOp.getOperand().getType() == extOp.getResult().getType();
  });
  // addIllegalOp<LLVM::SExtOp>();
  // addDynamicallyLegalOp<LLVM::TruncOp>([&](LLVM::TruncOp truncOp) {
  //   return truncOp.getOperand().getType() == truncOp.getResult().getType();
  // });
  addDynamicallyLegalOp<cgra::LwiOp>([&](cgra::LwiOp lwiOp) {
    return isI32IntegerType(lwiOp.getResult().getType());
  });
}

LogicalResult
ConstantOpRewrite::matchAndRewrite(LLVM::ConstantOp constOp,
                                   PatternRewriter &rewriter) const {

  auto value = constOp.getValue().cast<IntegerAttr>().getInt();

  // indirectly load & store cannot exceed reserved address range
  // if the constant is used as immediate for address
  if (isAddrConstOp(constOp) && !isValidImmAddr(constOp)) {
    // if (auto validOp = existsConstant(value, insertedOps)) {
    //   // set it to valid range, to be removed later on
    //   rewriter.modifyOpInPlace(constOp, [&] {
    //     constOp->setAttr("value", rewriter.getI32IntegerAttr(0));
    //   });
    //   rewriter.replaceAllUsesWith(constOp, validOp);
    // } else
    {
      auto addOp = generateValidConstant(constOp, rewriter);
      insertedOps.push_back(addOp);
    }
    return success();
  }

  // rewrite the constant operation if the value exceed the range
  if (value < -4097 || value > 4096) {
    // if (auto validOp = existsConstant(value, insertedOps))
    //   rewriter.replaceAllUsesWith(constOp, validOp);
    // else
    {
      auto addOp = generateValidConstant(constOp, rewriter);
      insertedOps.push_back(addOp);
    }
  }

  for (auto user : llvm::make_early_inc_range(constOp->getUsers()))
    if (isa<LLVM::BrOp, cgra::ConditionalBranchOp, cgra::SwiOp>(user)) {
      // Always create new operation if the constant is used by branch ops,
      // which would be propagated to multiple operations in the successor
      // blocks.
      auto addOp = generateImmAddOp(constOp, rewriter);
      user->replaceUsesOfWith(constOp.getResult(), addOp.getResult());
      insertedOps.push_back(addOp);
    }

  // if the constant is not replaced by valid operation, return failure
  if (!constOp->use_empty()) {
    // print use
    for (auto user : constOp->getUsers())
      llvm::errs() << "Failed to replace Op on: " << *user << "\n";
    return failure();
  }

  // set the value to 0, to be removed later on
  rewriter.updateRootInPlace(constOp, [&] {
    constOp->setAttr("value", rewriter.getI32IntegerAttr(0));
  });

  return success();
}

SmallVector<Operation *, 4> getSrcOpIndirectly(Value val, Block *targetBlock) {
  SmallVector<Operation *, 4> cntOps;
  if (!val.isa<BlockArgument>()) {
    cntOps.push_back(val.getDefiningOp());
    return cntOps;
  }

  // if the value is not block argument, return empty vector

  Block *block = val.getParentBlock();
  Value propVal;

  int argInd = val.cast<BlockArgument>().getArgNumber();
  for (auto pred : block->getPredecessors()) {
    Operation *termOp = pred->getTerminator();
    // Return operation does not propagate block argument
    if (isa<LLVM::ReturnOp>(termOp))
      continue;

    if (isa<LLVM::BrOp>(termOp)) {
      // the corresponding block argument is the argInd'th operator
      propVal = termOp->getOperand(argInd);
      auto defOps = getSrcOpIndirectly(propVal, targetBlock);
      cntOps.append(defOps.begin(), defOps.end());
    } else if (auto condBr = dyn_cast<cgra::ConditionalBranchOp>(termOp)) {
      // The terminator would be beq, bne, blt, bge, etc, the propagated value
      // is counted from 2nd operand.
      if (targetBlock == condBr.getTrueDest())
        propVal = condBr.getTrueOperand(argInd);
      else if (targetBlock == condBr.getFalseDest()) {
        propVal = condBr.getFalseOperand(argInd);
      } else
        continue;

      auto defOps = getSrcOpIndirectly(propVal, targetBlock);
      cntOps.append(defOps.begin(), defOps.end());
    }
  }

  return cntOps;
}

LogicalResult LwiOpRewrite::matchAndRewrite(cgra::LwiOp lwiOp,
                                            PatternRewriter &rewriter) const {
  auto origType = lwiOp.getOperand().getType(); // valid I32 address type

  rewriter.updateRootInPlace(
      lwiOp, [&] { lwiOp.getResult().setType(rewriter.getI32Type()); });

  // set the type of corresponding successor
  std::stack<Value> dstOprs;
  dstOprs.push(lwiOp.getResult());
  while (!dstOprs.empty()) {
    auto opr = dstOprs.top();
    dstOprs.pop();

    // change opr to origType
    opr.setType(origType);

    // push the predecessors if its type is not the same as result type
    for (auto user : opr.getUsers())
      if (user->getNumResults() == 1 &&
          !isI32IntegerType(user->getResult(0).getType()))
        dstOprs.push(user->getResult(0));
  }
  return success();
}

LogicalResult
BitWidthExtOpRewrite::matchAndRewrite(LLVM::SExtOp extOp,
                                      PatternRewriter &rewriter) const {
  // track all the predecessor op of extOp
  auto resType = extOp.getResult().getType();
  auto *prodOp = extOp.getOperand().getDefiningOp();
  if (!isI32IntegerType(resType))
    return failure();

  std::stack<Value> srcOprs;
  srcOprs.push(extOp.getOperand());
  while (!srcOprs.empty()) {
    auto opr = srcOprs.top();
    srcOprs.pop();

    // change opr to origType
    opr.setType(resType);

    // push the predecessors if its type is not the same as result type
    auto defOps = getSrcOpIndirectly(opr, opr.getDefiningOp()->getBlock());
    for (auto op : defOps) {
      if (op->getResult(0).getType() != resType)
        srcOprs.push(op->getResult(0));
    }
  }
  if (prodOp)
    rewriter.replaceAllUsesWith(extOp, prodOp->getResult(0));

  // set all the predecessor type with result type, erase the operation
  rewriter.updateRootInPlace(extOp, [&] {
    extOp.getOperand().setType(resType);
    extOp->erase();
  });
  return success();
}

void CgraFitToOpenEdgePass::runOnOperation() {
  ModuleOp modOp = dyn_cast<ModuleOp>(getOperation());
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns{ctx};
  OpenEdgeISATarget target(ctx);

  SmallVector<Operation *> insertedOps;
  patterns.add<ConstantOpRewrite>(ctx, insertedOps);
  patterns.add<BitWidthExtOpRewrite>(ctx);
  patterns.add<LwiOpRewrite>(ctx);

  // adapt the constant operation to meet the requirement of Imm field of
  // openedge CGRA
  if (failed(applyPartialConversion(modOp, target, std::move(patterns))))
    signalPassFailure();

  // raise the constant operation to the top level
  for (auto funcOp : llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>()))
    if (failed(raiseConstOperation(funcOp)) ||
        failed(removeUnusedConstOp(funcOp)) ||
        failed(removeEqualWidthBWOp(funcOp)))
      signalPassFailure();

  // print the DAG of the specified function
  if (!outputDAG.empty()) {
    size_t lastSlashPos = outputDAG.find_last_of("/");
    bool isPath = lastSlashPos != StringRef::npos;
    StringRef funcName =
        isPath ? outputDAG.substr(lastSlashPos + 1) : outputDAG;

    for (auto funcOp :
         llvm::make_early_inc_range(modOp.getOps<cgra::FuncOp>())) {
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