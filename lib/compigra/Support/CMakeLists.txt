add_compigra_library(MLIRCompigraSupport
  Utils.cpp
  OpenEdgeASM.cpp
  InterferenceGraphCreation.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  )