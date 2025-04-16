add_compigra_library(MLIRCompigraASMGen
ASMGenTempCGRA.cpp
FastASMGenTempCGRA.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/compigra

DEPENDS
CompigraASMGenIncGen

LINK_LIBS PUBLIC
MLIRIR
)