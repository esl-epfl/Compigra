#ifndef COMPIGRA_PASSES_H
#define COMPIGRA_PASSES_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir{
namespace compigra {
std::unique_ptr<mlir::Pass> createScfFixIndexWidth();
} // end namespace compigra
} // end namespace mlir

#endif // COMPIGRA_PASSES_H
