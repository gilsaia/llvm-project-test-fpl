set(LIBS
    MLIRPresburger
    MLIRAffineAnalysis
    MLIRParser
    )

add_mlir_tool(mlir-fpl-test
    mlir-fpl-test.cpp

    DEPENDS
    ${LIBS}
)
target_link_libraries(mlir-fpl-test PRIVATE ${LIBS})