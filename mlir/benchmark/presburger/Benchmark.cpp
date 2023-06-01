//===- Benchmark.cpp - MLIR Presburger Benchmark
//------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a benchmark for PresburgerRelation.
//
//===----------------------------------------------------------------------===//

#include <benchmark/benchmark.h>

#include "FPLBench.h"
#include "ISLBench.h"

BENCHMARK(BM_FPLBinaryOperationCheck<false>)
    ->Name("FPL_Union")
    ->Setup(FPLSetupUnion)
    ->Iterations(5);
BENCHMARK(BM_FPLBinaryOperationCheck<true>)
    ->Name("FPL_Union_Simplify")
    ->Setup(FPLSetupUnion)
    ->Iterations(5);
BENCHMARK(BM_ISLBinaryOperationCheck)
    ->Name("ISL_Union")
    ->Setup(ISLSetupUnion)
    ->Iterations(5);

BENCHMARK(BM_FPLBinaryOperationCheck<false>)
    ->Name("FPL_Subtract")
    ->Setup(FPLSetupSubtract)
    ->Iterations(5);
BENCHMARK(BM_FPLBinaryOperationCheck<true>)
    ->Name("FPL_Subtract_Simplify")
    ->Setup(FPLSetupSubtract)
    ->Iterations(5);
BENCHMARK(BM_ISLBinaryOperationCheck)
    ->Name("ISL_Subtract")
    ->Setup(ISLSetupSubtract)
    ->Iterations(5);

BENCHMARK(BM_FPLUnaryOperationCheck<false>)
    ->Name("FPL_Complement")
    ->Setup(FPLSetupComplement)
    ->Iterations(70);
BENCHMARK(BM_FPLUnaryOperationCheck<true>)
    ->Name("FPL_Complement_Simplify")
    ->Setup(FPLSetupComplement)
    ->Iterations(70);
BENCHMARK(BM_ISLUnaryOperationCheck)
    ->Name("ISL_Complement")
    ->Setup(ISLSetupComplement)
    ->Iterations(70);

BENCHMARK(BM_FPLBinaryOperationCheck<false>)
    ->Name("FPL_Intersect")
    ->Setup(FPLSetupIntersect)
    ->Iterations(5);
BENCHMARK(BM_FPLBinaryOperationCheck<true>)
    ->Name("FPL_Intersect_Simplify")
    ->Setup(FPLSetupIntersect)
    ->Iterations(5);
BENCHMARK(BM_ISLBinaryOperationCheck)
    ->Name("ISL_Intersect")
    ->Setup(ISLSetupIntersect)
    ->Iterations(5);

BENCHMARK(BM_FPLBinaryOperationCheck<false>)
    ->Name("FPL_IsEqual")
    ->Setup(FPLSetupIsEqual)
    ->Iterations(120);
BENCHMARK(BM_FPLBinaryOperationCheck<true>)
    ->Name("FPL_IsEqual_Simplify")
    ->Setup(FPLSetupIsEqual)
    ->Iterations(120);
BENCHMARK(BM_ISLBinaryOperationCheck)
    ->Name("ISL_IsEqual")
    ->Setup(ISLSetupIsEqual)
    ->Iterations(120);

BENCHMARK(BM_FPLUnaryOperationCheck<false>)
    ->Name("FPL_IsEmpty")
    ->Setup(FPLSetupIsEmpty)
    ->Iterations(5);
BENCHMARK(BM_FPLUnaryOperationCheck<true>)
    ->Name("FPL_IsEmpty_Simplify")
    ->Setup(FPLSetupIsEmpty)
    ->Iterations(5);
BENCHMARK(BM_ISLUnaryOperationCheck)
    ->Name("ISL_IsEmpty")
    ->Setup(ISLSetupIsEmpty)
    ->Iterations(5);

BENCHMARK_MAIN();