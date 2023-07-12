#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <benchmark/benchmark.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace mlir;
using namespace mlir::presburger;

/// This function parses a single set of integer constraints from the input
/// stream. The input format for a set is as follows:
/// - The first line contains two integers: eqs and inEqs. They represent the
/// number of equations and inequalities in the set, respectively.
/// - The following lines describe each constraint equation or inequality in the
/// set.
///
/// After reading the number of equations and inequalities from the input
/// stream, the function iteratively constructs the affine expressions
/// representing the constraints. It reads the parameters for each expression
/// from the input stream and forms the affine expressions by multiplying the
/// parameters with the corresponding affine dimensions or symbols. Finally, it
/// constructs an `IntegerSet` object by creating the affine constraint
/// expressions and indicating which constraints are equations and which are
/// inequalities.
IntegerSet FPLParseOneSet(std::istream &in, int &numDims, int &numSymbols,
                          MLIRContext &ctx);

/// This function parses a single case of a Presburger relation from the input
/// stream. The input data format for a case is as follows:
/// - The first line contains three integers: numDims, numSymbols, and
/// numUnions. They represent the number of dimensions, symbols, and sets,
/// respectively.
/// - The following lines contain the description of each set in the union.
PresburgerRelation FPLParseOneCase(std::istream &in, MLIRContext &ctx);

/// This function calculates the size of a given Preburger relation, which is
/// the total number of constraints and variables. It iterates over all
/// disjuncts in the relation and accumulates the product of the number of
/// constraints in each disjunct and (variable count + 1) into the provided size
/// variable.
void FPLCountPreburgerRelationSize(PresburgerRelation &relation,
                                   unsigned long long &size);

void FPLSetupUnion(const benchmark::State &state);
void FPLSetupSubtract(const benchmark::State &state);
void FPLSetupComplement(const benchmark::State &state);
void FPLSetupIntersect(const benchmark::State &state);
void FPLSetupIsEqual(const benchmark::State &state);
void FPLSetupIsEmpty(const benchmark::State &state);

template <bool useSimplify = false>
void BM_FPLUnaryOperationCheck(benchmark::State &state);

template <bool useSimplify = false>
void BM_FPLBinaryOperationCheck(benchmark::State &state);