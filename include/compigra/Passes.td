//===----------------------------------------------------------------------===//
//
// This file contains definitions for optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef COMPIGRA_PASSES_TD
#define COMPIGRA_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Scf passes
//===----------------------------------------------------------------------===//

def ScfFixIndexWidth : Pass<"fix-index-width"> {
  let summary = "Fix the index width to CGRA processing element width";
  let constructor = "mlir::compigra::createScfFixIndexWidth()";
}

#endif // COMPIGRA_PASSES_TD
