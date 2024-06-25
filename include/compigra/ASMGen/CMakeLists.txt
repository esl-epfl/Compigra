set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ASMGen)
add_public_tablegen_target(CompigraASMGenIncGen)
add_dependencies(compigra-headers CompigraASMGenIncGen)