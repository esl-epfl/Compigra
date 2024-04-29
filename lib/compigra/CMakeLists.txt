add_compigra_dialect_library(MLIRCgra
CgraDialect.cpp
CgraOps.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/compigra
${PROJECT_BINARY_DIR}/include/compigra

DEPENDS
MLIRCgraIncGen
MLIRCgraEnumsIncGen

LINK_LIBS PUBLIC
MLIRIR
MLIRMemRefDialect
MLIRLLVMDialect
MLIROpenMPDialect
MLIRAffineDialect
MLIRSupport
MLIRSCFTransforms
)
add_subdirectory(Passes)