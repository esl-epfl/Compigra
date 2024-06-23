set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ASMGen)
add_public_tablegen_target(CompigraASMGenIncGen)
add_dependencies(compigra-headers CompigraASMGenIncGen)

# GUROBI Optimizer related setup
# Check if GUROBI_INSTALL_DIR is provided as a CMake parameter
if(GUROBI_INSTALL_DIR)
    message(STATUS "GUROBI_INSTALL_DIR is set via CMake parameter: ${GUROBI_INSTALL_DIR}")
    include_directories(${GUROBI_INSTALL_DIR}/include)
    add_definitions(-DHAVE_GUROBI)
    message(STATUS "Including Gurobi include directory: ${GUROBI_INSTALL_DIR}/include")
else()
    message(WARNING "GUROBI_INSTALL_DIR is not set. Gurobi will not be included.")
endif()