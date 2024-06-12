add_compigra_library(MLIRCompigraConversion
LLVMToCgraConversion.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/compigra

DEPENDS
CompigraConversionIncGen

LINK_LIBS PUBLIC
MLIRIR
)
