add_benchmark(MLIRPresburgerBenchmark Benchmark.cpp FPLBench.cpp)

target_include_directories(MLIRPresburgerBenchmark PRIVATE ${MLIR_ISL_SOURCE}/include)

target_link_directories(MLIRPresburgerBenchmark PRIVATE ${MLIR_ISL_SOURCE}/lib)

target_link_libraries(MLIRPresburgerBenchmark
    PRIVATE
    MLIRPresburger
    LLVMSupport
    MLIRAffineAnalysis
    MLIRParser  
    isl
)