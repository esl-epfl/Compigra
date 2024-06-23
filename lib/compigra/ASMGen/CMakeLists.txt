add_compigra_library(MLIRCompigraASMGen
OpenEdgeASM.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/compigra

DEPENDS
CompigraASMGenIncGen

LINK_LIBS PUBLIC
MLIRIR
)

# GUROBI Optimizer related setup
# Check if GUROBI_INSTALL_DIR is provided as a CMake parameter
if(GUROBI_INSTALL_DIR)
    message(STATUS "GUROBI_INSTALL_DIR is set via CMake parameter: ${GUROBI_INSTALL_DIR}")
    add_definitions(-DHAVE_GUROBI)
    include_directories(${GUROBI_INSTALL_DIR}/include)
    include_directories(${GUROBI_INSTALL_DIR}/src/cpp)
    target_link_libraries(MLIRCompigraASMGen PUBLIC 
        ${GUROBI_INSTALL_DIR}/lib/libgurobi_c++.a 
        ${GUROBI_INSTALL_DIR}/lib/libgurobi110.so)
    message(STATUS "Including Gurobi include directory: ${GUROBI_INSTALL_DIR}/include")
    message(STATUS "Linking Gurobi library: ${GUROBI_INSTALL_DIR}/lib")
else()
    message(WARNING "GUROBI_INSTALL_DIR is not set. Gurobi will not be included.")
endif()