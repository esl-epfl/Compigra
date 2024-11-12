add_compigra_library(MLIRCompigraASMGen
OpenEdgeASM.cpp
InterferenceGraphCreation.cpp
ASMGenTempCGRA.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/compigra

DEPENDS
CompigraASMGenIncGen

LINK_LIBS PUBLIC
MLIRIR
)