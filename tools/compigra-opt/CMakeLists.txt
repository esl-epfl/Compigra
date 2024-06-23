get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRCgra
        MLIRCompigraTransforms
        MLIRCompigraConversion
        MLIRCompigraSatMapPrint
        MLIRCompigraASMGen
        
        MLIRIR
        MLIROptLib
        MLIRAffineTransforms
        MLIRArithTransforms
        MLIRSCFTransforms
        MLIRMemRefTransforms
        MLIRLLVMDialect
        MLIRMemRefDialect
        MLIRParser
        MLIRFuncDialect
        MLIRSupport
        MLIRTransforms
        MLIRSCFDialect
        MLIRFunctionInterfaces
)

set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(compigra-opt
    compigra-opt.cpp
)

llvm_update_compile_flags(compigra-opt)
target_link_libraries(compigra-opt PRIVATE ${LIBS})

