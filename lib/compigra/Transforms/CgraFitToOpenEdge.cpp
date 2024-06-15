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

// Currently only consider add, sub, mul operations
static bool usedByArithOpOnly(LLVM::ConstantOp constOp) {
  for (auto user : constOp->getUsers())
    if (!isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, cgra::MergeOp>(user))
      return false;

  return true;
}

static LogicalResult raiseConstOperation(cgra::FuncOp funcOp) {
  Operation &beginOp = *funcOp.getOps().begin();
  // raise the constant operation to the top level
  for (auto op :
       llvm::make_early_inc_range(funcOp.getOps<LLVM::ConstantOp>())) {
    if (!op->hasOneUse())
      op.dump();
    else if (op->getAttr("stage") ==
             StringAttr::get(op->getContext(), "init")) {
      op->moveBefore(&beginOp);
    }
  }

  return success();
}

static LogicalResult outputDATE2023DAG(cgra::FuncOp funcOp,
                                       std::string outputDAG) {
  SmallVector<Operation *> nodes;
  SmallVector<LLVM::ConstantOp> constants;
  SmallVector<Operation *> liveIns;
  SmallVector<Operation *> liveOuts;
  // define the edge by the srcOp and dstOp
  using Edge = std::pair<Operation *, Operation *>;

  // text file to describe the DAG
  std::ofstream dotFile;

  std::vector<Edge> edges;

  // Store all the nodes (operations)
  for (Operation &op : funcOp.getOps()) {
    StringRef stage = dyn_cast<StringAttr>(op.getAttr("stage")).getValue();

    // SAT-MapIt only schedule operations in loop
    if (stage != StringRef("loop"))
      continue;

    // store operator related operations
    for (Value operand : op.getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        // only store the related nodes in init stage
        StringRef ownerStage =
            dyn_cast<StringAttr>(defOp->getAttr("stage")).getValue();
        if (ownerStage == StringRef("init")) {
          if (isa<LLVM::ConstantOp>(defOp))
            constants.push_back(cast<LLVM::ConstantOp>(defOp));
          else
            liveIns.push_back(defOp);
        }
      }
    }

    nodes.push_back(&op);

    // if the result of the operation is used by the operation in the fini
    // stage, it is a liveOut node
    for (auto userOp : op.getUsers()) {
      StringRef userStage =
          dyn_cast<StringAttr>(userOp->getAttr("stage")).getValue();
      if (userStage == StringRef("fini"))
        liveOuts.push_back(userOp);
    }
  }

  // initialize print function
  satmapit::PrintSatMapItDAG printer(nodes, constants, liveIns, liveOuts);
  if (failed(printer.printDAG(outputDAG)))
    return failure();

  return success();
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
  auto lwiOp = rewriter.create<cgra::LwiOp>(constOp.getLoc(), constOp.getType(),
                                            constOp.getResult());
  lwiOp->setAttr("stage", constOp->getAttr("stage"));

  *constBase += 4;

  return lwiOp;
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
      // adapt it to the lwi operation
      if (isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(user)) {
        if (!hasOnlyUser(constOp)) {
          // insert a new constant operation to specify the load address
          auto brAddrConst = rewriter.create<LLVM::ConstantOp>(
              constOp.getLoc(), constOp.getType(),
              rewriter.getI32IntegerAttr(*constBase));
          auto lwiOp = rewriter.create<cgra::LwiOp>(
              brAddrConst.getLoc(), constOp.getType(), brAddrConst.getResult());
          // replace use if it is beq, bne, blt, bge
          rewriter.replaceOpUsesWithIf(
              constOp, lwiOp.getResult(), [&](OpOperand &operand) {
                return isa<cgra::BeqOp, cgra::BneOp, cgra::BltOp, cgra::BgeOp>(
                    operand.getOwner());
              });
          *constBase += 4;
        } else {
          auto lwiOp = convertImmToLwi(constOp, constBase, rewriter);
          rewriter.replaceOpUsesWithIf(
              constOp, lwiOp.getResult(),
              [&](OpOperand &operand) { return operand.getOwner() != lwiOp; });
        }
      }

    return success();
  }

protected:
  int *constBase;
};

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
      if (failed(raiseConstOperation(funcOp)))
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