#include "FPLBench.h"
#include "utils.h"
#include <chrono>
#include <functional>

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
                          MLIRContext &ctx) {
  int inEqs, eqs;
  in >> eqs >> inEqs;
  SmallVector<AffineExpr, 4> constraints;
  SmallVector<bool, 4> isEqs;

  for (int t = 0; t < (eqs + inEqs); ++t) {
    AffineExpr equation = getAffineConstantExpr(0, &ctx);
    for (int i = 0; i <= (numDims + numSymbols); ++i) {
      int64_t param;
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

/// This function parses a single case of a Presburger relation from the input
/// stream. The input data format for a case is as follows:
/// - The first line contains three integers: numDims, numSymbols, and
/// numUnions. They represent the number of dimensions, symbols, and sets,
/// respectively.
/// - The following lines contain the description of each set in the union.
PresburgerRelation FPLParseOneCase(std::istream &in, MLIRContext &ctx) {
  int numDims, numSymbols, numUnions;
  in >> numDims >> numSymbols >> numUnions;
  SmallVector<IntegerSet, 4> eles;
  for (int i = 0; i < numUnions; ++i) {
    IntegerSet ele = FPLParseOneSet(in, numDims, numSymbols, ctx);
    eles.emplace_back(ele);
  }
  IntegerRelation reli = affine::FlatAffineValueConstraints(eles[0]);
  PresburgerRelation rel(reli);
  for (unsigned i = 1, e = eles.size(); i < e; ++i) {
    IntegerRelation reli = affine::FlatAffineValueConstraints(eles[i]);
    rel.unionInPlace(reli);
  }
  return rel;
}

/// This function calculates the size of a given Preburger relation, which is
/// the total number of constraints and variables. It iterates over all
/// disjuncts in the relation and accumulates the product of the number of
/// constraints in each disjunct and (variable count + 1) into the provided size
/// variable.
void FPLCountPreburgerRelationSize(PresburgerRelation &relation,
                                   unsigned long long &size) {
  unsigned var = relation.getNumVars();
  for (auto ele : relation.getAllDisjuncts()) {
    size += ele.getNumConstraints() * (var + 1);
  }
}

void FPLParseOneCaseOneInt(std::string &fileName,
                           std::vector<PresburgerRelation> &sets,
                           std::vector<PresburgerRelation> &setsNull) {
  MLIRContext ctx(MLIRContext::Threading::DISABLED);
  std::ifstream in(fileName);
  size_t num;
  in >> num;
  for (size_t i = 0; i < num; ++i) {
    int tmp;
    sets.emplace_back(FPLParseOneCase(in, ctx));
    in >> tmp;
  }
  in.close();
}

void FPLParseTwoCaseUseOneCase(std::string &fileName,
                               std::vector<PresburgerRelation> &sets,
                               std::vector<PresburgerRelation> &setsNull) {
  MLIRContext ctx(MLIRContext::Threading::DISABLED);
  std::ifstream in(fileName);
  size_t num;
  in >> num;
  for (size_t i = 0; i < num; ++i) {
    sets.emplace_back(FPLParseOneCase(in, ctx));
    FPLParseOneCase(in, ctx);
  }
  in.close();
}

void FPLParseTwoCaseOntInt(std::string &fileName,
                           std::vector<PresburgerRelation> &setsA,
                           std::vector<PresburgerRelation> &setsB) {
  MLIRContext ctx(MLIRContext::Threading::DISABLED);
  std::ifstream in(fileName);
  size_t num;
  in >> num;
  for (size_t i = 0; i < num; ++i) {
    int tmp;
    setsA.emplace_back(FPLParseOneCase(in, ctx));
    setsB.emplace_back(FPLParseOneCase(in, ctx));
    in >> tmp;
  }
  in.close();
}

void FPLParseThreeCaseUseTwoCase(std::string &fileName,
                                 std::vector<PresburgerRelation> &setsA,
                                 std::vector<PresburgerRelation> &setsB) {
  MLIRContext ctx(MLIRContext::Threading::DISABLED);
  std::ifstream in(fileName);
  size_t num;
  in >> num;
  for (size_t i = 0; i < num; ++i) {
    setsA.emplace_back(FPLParseOneCase(in, ctx));
    setsB.emplace_back(FPLParseOneCase(in, ctx));
    FPLParseOneCase(in, ctx);
  }
  in.close();
}

static std::function<void(std::string &, std::vector<PresburgerRelation> &,
                          std::vector<PresburgerRelation> &)>
    readFunc;
static std::function<void(PresburgerRelation &)> unaryExecFunc;
static std::function<void(PresburgerRelation &, PresburgerRelation &)>
    binaryExecFunc;
static std::string fileName;

void FPLSetupUnion(const benchmark::State &state) {
  fileName = "./PresburgerSetUnion";
  readFunc = FPLParseThreeCaseUseTwoCase;
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    benchmark::DoNotOptimize(a.unionSet(b));
  };
}

void FPLSetupSubtract(const benchmark::State &state) {
  fileName = "./PresburgerSetSubtract";
  readFunc = FPLParseThreeCaseUseTwoCase;
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    benchmark::DoNotOptimize(a.subtract(b));
  };
}

void FPLSetupComplement(const benchmark::State &state) {
  fileName = "./PresburgerSetComplement";
  readFunc = FPLParseTwoCaseUseOneCase;
  unaryExecFunc = [](PresburgerRelation &a) {
    benchmark::DoNotOptimize(a.complement());
  };
}

void FPLSetupIntersect(const benchmark::State &state) {
  fileName = "./PresburgerSetIntersect";
  readFunc = FPLParseThreeCaseUseTwoCase;
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    benchmark::DoNotOptimize(a.intersect(b));
  };
}

void FPLSetupIsEqual(const benchmark::State &state) {
  fileName = "./PresburgerSetEqual";
  readFunc = FPLParseTwoCaseOntInt;
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    benchmark::DoNotOptimize(a.isEqual(b));
  };
}

void FPLSetupIsEmpty(const benchmark::State &state) {
  fileName = "./PresburgerSetEmpty";
  readFunc = FPLParseOneCaseOneInt;
  unaryExecFunc = [](PresburgerRelation &a) {
    benchmark::DoNotOptimize(a.isIntegerEmpty());
  };
}

template <bool useSimplify = false>
void BM_FPLUnaryOperationCheck(benchmark::State &state) {
  std::vector<PresburgerRelation> setsA, setsB;
  readFunc(fileName, setsA, setsB);
  for (auto _ : state) {
    for (auto &rel : setsA) {
      if (useSimplify) {
        // TODO:call simplify
      }
      unaryExecFunc(rel);
    }
  }
  unsigned long long relationSize = 0;
  for (auto &rel : setsA) {
    FPLCountPreburgerRelationSize(rel, relationSize);
  }
  state.counters["Constraint Size"] = relationSize;

  // log info
  std::vector<int> consSizes;
  std::vector<double> consTimes;
  for (auto &rel : setsA) {
    unsigned long long size = 0;
    FPLCountPreburgerRelationSize(rel, size);
    consSizes.push_back(size);
    auto begin = std::chrono::steady_clock::now();
    unaryExecFunc(rel);
    auto end = std::chrono::steady_clock::now();
    consTimes.emplace_back(
        std::chrono::duration<double, std::nano>(end - begin).count());
  }
  auto logFileName =
      fileName + "_fpl" + (useSimplify ? "_simplify" : "") + "_info.csv";
  LogAllInfo(logFileName, consSizes, consTimes);
}

template void BM_FPLUnaryOperationCheck<false>(benchmark::State &state);
template void BM_FPLUnaryOperationCheck<true>(benchmark::State &state);

template <bool useSimplify = false>
void BM_FPLBinaryOperationCheck(benchmark::State &state) {
  std::vector<PresburgerRelation> setsA, setsB;
  readFunc(fileName, setsA, setsB);
  size_t num = setsA.size();
  for (auto _ : state) {
    for (size_t i = 0; i < num; ++i) {
      if (useSimplify) {
        // TODO:call simplify
      }
      binaryExecFunc(setsA[i], setsB[i]);
    }
  }
  unsigned long long relationSize = 0;
  for (auto &rel : setsA) {
    FPLCountPreburgerRelationSize(rel, relationSize);
  }
  for (auto &rel : setsB) {
    FPLCountPreburgerRelationSize(rel, relationSize);
  }
  state.counters["Constraint Size"] = relationSize;

  // log info
  std::vector<int> consSizes;
  std::vector<double> consTimes;
  for (size_t i = 0; i < num; ++i) {
    unsigned long long size = 0;
    FPLCountPreburgerRelationSize(setsA[i], size);
    FPLCountPreburgerRelationSize(setsB[i], size);
    consSizes.push_back(size);
    auto begin = std::chrono::steady_clock::now();
    binaryExecFunc(setsA[i], setsB[i]);
    auto end = std::chrono::steady_clock::now();
    consTimes.emplace_back(
        std::chrono::duration<double, std::nano>(end - begin).count());
  }
  auto logFileName =
      fileName + "_fpl" + (useSimplify ? "_simplify" : "") + "_info.csv";
  LogAllInfo(logFileName, consSizes, consTimes);
}

template void BM_FPLBinaryOperationCheck<false>(benchmark::State &state);
template void BM_FPLBinaryOperationCheck<true>(benchmark::State &state);