get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRCompigraTransforms
        # MLIRCOMPIGRA
		# MLIRPolygeistTransforms
		# MLIRFuncAllExtensions
        )

add_llvm_tool(compigra-opt
    compigra-opt.cpp
)

# install(TARGETS compigra-opt
# EXPORT CompigraTargets
# RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
# COMPONENT compigra-opt)

llvm_update_compile_flags(compigra-opt)
target_link_libraries(compigra-opt PRIVATE ${LIBS})
