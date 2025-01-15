//===- LLVMToCgraConversion.h - Convert part ops to Cgra dialect *- C++ -*-===//
//
// Copmigra is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the --convert-llvm-to-cgra pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TO_CGRA_CONVERSION_H
#define LLVM_TO_CGRA_CONVERSION_H

#include "compigra/CgraDialect.h"
#include "compigra/CgraInterfaces.h"
#include "compigra/CgraOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace compigra {

/// Struct for describing the memory allocation policy of the address space
/// between the host processor and the CGRA.
struct MemoryInterface {
  MemoryInterface() {}
  MemoryInterface(std::vector<int> startAddr, std::vector<int> endAddr)
      : startAddr(startAddr), endAddr(endAddr) {}

  std::vector<int> startAddr;
  std::vector<int> endAddr;
};

// ============================================================================
// Partial lowering infrastructure
// ============================================================================
class CgraLowering {
public:
  bool replaceBranch = true;
  bool initMem = false;
  MemoryInterface memInterface;

  /// Groups information to rewire the IR around merge-like operations by owning
  /// basic block (which must still exist).

  /// Constructor simply takes the region being lowered and a reference to the
  /// top-level name analysis.
  explicit CgraLowering(Region &region) : region(region) {}

  void initMemoryInterface(MemoryInterface &memInterface) {
    this->memInterface = memInterface;
    initMem = true;
  }

  // Reorder the index of the basic blocks if it is not in the order of Init,
  // Loop, and Fini.
  LogicalResult reorderBBs(ConversionPatternRewriter &rewriter);

  /// Replace ICmp operation with substraction which zero/sign flags are used
  /// for further select operation;
  /// Replace ICmp + cond branch with corresponding branch operation in cgra,
  /// e.g. ICmp (ne) + cond_br -> bne;
  /// Insert additional select operation to select 0/1 based on the flag of the
  /// substitude subtraction for operations like add, sub, etc.
  LogicalResult replaceCmpOps(ConversionPatternRewriter &rewriter);

  /// Parse explicitly the memory address to the corresponding memory interface,
  /// and replace load and store with lwi/swi.
  /// This function returns failure if the LLVM::GEPOp is used by multiple
  /// load/store users
  LogicalResult addMemoryInterface(ConversionPatternRewriter &rewriter);

  // Ensure the constant operation is only used once and replicate the constant
  // if it is in multiple uses.
  LogicalResult raiseConstOnlyUse(ConversionPatternRewriter &rewriter);

  /// Remove unused operations in the region if it is not controlled operation
  /// and return operation. This function mainly removes the subtraction created
  /// in replaceCmpOps to flag the cond_br operation.
  /// createSATMapItDAG(optionally) replaces the cond_br with cgra branch
  /// operations which left the subtraction unused.
  LogicalResult removeUnusedOps(ConversionPatternRewriter &rewriter);

  Region &getRegion() { return region; }
  Block *getEntryBlock() { return &region.front(); }
  Operation *getEntryBlockTerminator() {
    return getEntryBlock()->getTerminator();
  }
  /// Get the first constant operation in the region
  Operation *getConstantOp();

protected:
  Region &region;
};

using RegionLoweringFunc =
    llvm::function_ref<LogicalResult(Region &, ConversionPatternRewriter &)>;

/// Partially lowers a region using a provided lowering function.
LogicalResult partiallyLowerRegion(const RegionLoweringFunc &loweringFunc,
                                   Region &region);

template <typename T, typename... TArgs1, typename... TArgs2>
static LogicalResult runPartialLowering(
    T &instance,
    LogicalResult (T::*memberFunc)(ConversionPatternRewriter &, TArgs2...),
    TArgs1 &...args) {
  return partiallyLowerRegion(
      [&](Region &, ConversionPatternRewriter &rewriter) -> LogicalResult {
        return (instance.*memberFunc)(rewriter, args...);
      },
      instance.getRegion());
}

#define GEN_PASS_DEF_LLVMTOCGRACONVERSION
#define GEN_PASS_DECL_LLVMTOCGRACONVERSION
#include "compigra/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLLVMToCgraConversion(StringRef funcName = "",
                                                       StringRef memAlloc = "");

} // namespace compigra

namespace {
struct LLVMToCgraConversionPass
    : public compigra::impl::LLVMToCgraConversionBase<
          LLVMToCgraConversionPass> {
  LLVMToCgraConversionPass(StringRef funcName, StringRef memAlloc) {}

  void runOnOperation() override;
  // LogicalResult outputDATE2023DAG(cgra::FuncOp funcOp);
};
} // namespace

#endif // LLVM_TO_CGRA_CONVERSION_H