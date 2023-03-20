add_benchmark(FPLBenchmark FplBench.cpp)

target_link_libraries(FPLBenchmark
    PRIVATE
    MLIRPresburger
    LLVMSupport
    MLIRAffineAnalysis
    MLIRParser
)