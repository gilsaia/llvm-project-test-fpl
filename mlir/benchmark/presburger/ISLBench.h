#ifndef MLIR_BENCHMARK_PRESBURGER_ISL_BENCH_H_
#define MLIR_BENCHMARK_PRESBURGER_ISL_BENCH_H_

void ISLSetupUnion(const benchmark::State &state);
void ISLSetupSubtract(const benchmark::State &state);
void ISLSetupComplement(const benchmark::State &state);
void ISLSetupIntersect(const benchmark::State &state);
void ISLSetupIsEqual(const benchmark::State &state);
void ISLSetupIsEmpty(const benchmark::State &state);

void BM_ISLUnaryOperationCheck(benchmark::State &state);
void BM_ISLBinaryOperationCheck(benchmark::State &state);

#endif