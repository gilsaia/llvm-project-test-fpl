#include "mlir/IR/MLIRContext.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/IntegerSet.h"

#include "benchmark/benchmark.h"

#include <fstream>
#include <vector>

using namespace mlir;

IntegerSet parseOneSet(std::istream &in, int &numDims, int &numSymbols,
                       MLIRContext &ctx) {
  int inEqs, eqs;
  in >> eqs >> inEqs;
  SmallVector<AffineExpr, 4> constraints;
  SmallVector<bool, 4> isEqs;
  for (int t = 0; t < (eqs + inEqs); ++t) {
    AffineExpr equation = getAffineConstantExpr(0, &ctx);
    for (int i = 0; i <= (numDims + numSymbols); ++i) {
      int param;
      in >> param;
      AffineExpr param_expr = getAffineConstantExpr(param, &ctx);
      if (i < (numDims + numSymbols)) {
        AffineExpr var = (i < numDims) ? getAffineDimExpr(i, &ctx)
                                       : getAffineSymbolExpr(i - numDims, &ctx);
        param_expr = param_expr * var;
      }
      equation = equation + param_expr;
    }
    AffineExpr rhs = getAffineConstantExpr(0, &ctx);
    constraints.emplace_back(equation - rhs);
    isEqs.emplace_back(t < eqs ? true : false);
  }
  return IntegerSet::get(numDims, numSymbols, constraints, isEqs);
}

presburger::PresburgerSet parseOneCase(std::istream &in) {
  MLIRContext ctx(MLIRContext::Threading::DISABLED);
  int numDims, numSymbols, numUnions;
  in >> numDims >> numSymbols >> numUnions;
  SmallVector<IntegerSet, 4> eles;
  for (int i = 0; i < numUnions; ++i) {
    IntegerSet ele = parseOneSet(in, numDims, numSymbols, ctx);
    eles.emplace_back(ele);
  }
  presburger::IntegerPolyhedron initPoly = FlatAffineValueConstraints(eles[0]);
  presburger::PresburgerSet result(initPoly);
  for (int i = 1, e = eles.size(); i < e; ++i) {
    result.unionInPlace(FlatAffineValueConstraints(eles[i]));
  }
  return result;
}

static void BM_PresburgerSetEqual(benchmark::State &state) {
  std::ifstream file("./PresburgerSetEqual");
  int num;
  file >> num;

  std::vector<presburger::PresburgerSet> setA, setB;
  std::vector<int> res;
  int resCase;
  for (int i = 0; i < num; ++i) {
    setA.emplace_back(parseOneCase(file));
    setB.emplace_back(parseOneCase(file));
    file >> resCase;
    res.push_back(resCase);
  }
  file.close();
  for (auto _ : state) {
    for (int i = 0; i < num; ++i) {
      benchmark::DoNotOptimize(setA[i].isEqual(setB[i]));
      //   bool boolRes = setA[i].isEqual(setB[i]);
      //   assert(boolRes == res[i]);
    }
  }
}
BENCHMARK(BM_PresburgerSetEqual);

BENCHMARK_MAIN();