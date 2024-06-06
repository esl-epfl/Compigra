add_compigra_library(MLIRCompigraSatMapPrint
  PrintSatMapItDAG.cpp
  
  DEPENDS
  CompigraPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
