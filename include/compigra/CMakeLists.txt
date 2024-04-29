add_compigra_dialect(Cgra cgra)
add_compigra_interface(CgraInterfaces)
# add_mlir_doc(CgraDialect -gen-dialect-doc CgraDialect compigra/)
# add_mlir_doc(CgraOps -gen-op-doc CgraOps compigra/)

set(LLVM_TARGET_DEFINITIONS Cgra.td)

mlir_tablegen(CgraEnums.h.inc -gen-enum-decls)
mlir_tablegen(CgraEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCgraEnumsIncGen)
add_dependencies(compigra-headers MLIRCgraEnumsIncGen)

mlir_tablegen(CgraAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=cgra)
mlir_tablegen(CgraAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=cgra)
add_public_tablegen_target(MLIRCgraAttributesIncGen)
add_dependencies(compigra-headers MLIRCgraAttributesIncGen)

add_subdirectory(Passes)