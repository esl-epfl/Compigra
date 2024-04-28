add_mlir_dialect(CgraOps cgra)
# add_mlir_doc(CgraDialect -gen-dialect-doc CgraDialect compigra/)
# add_mlir_doc(CgraOps -gen-op-doc CgraOps compigra/)

set(LLVM_TARGET_DEFINITIONS CgraDialect.td)

mlir_tablegen(CgraEnums.h.inc -gen-enum-decls)
mlir_tablegen(CgraEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCgraEnumsIncGen)
add_dependencies(compigra-headers MLIRCgraEnumsIncGen)

add_subdirectory(Passes)