#include "FPLBench.h"
#include "utils.h"
#include <chrono>
#include <functional>

static std::function<void(std::string &, std::vector<PresburgerRelation> &,
                          std::vector<PresburgerRelation> &)>
    readFunc;
static std::function<void(PresburgerRelation &)> unaryExecFunc;
static std::function<void(PresburgerRelation &, PresburgerRelation &)>
    binaryExecFunc;
static std::function<void(PresburgerSimpifyRelation &)> unarySimplifyExecFunc;
static std::function<void(PresburgerSimpifyRelation &,
                          PresburgerSimpifyRelation &)>
    binarySimplifyExecFunc;
static std::function<PresburgerRelation(PresburgerRelation &)>
    unaryReturnExecFunc;
static std::function<PresburgerRelation(PresburgerRelation &,
                                        PresburgerRelation &)>
    binaryReturnExecFunc;
static std::function<void(PresburgerRelation &)> simplifyForCountFunc;
static std::string fileName;

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
    bool isValid = false;
    for (int i = 0; i <= (numDims + numSymbols); ++i) {
      int64_t param;
      in >> param;
      if (param) {
        isValid = true;
      }
      AffineExpr param_expr = getAffineConstantExpr(param, &ctx);
      if (i < (numDims + numSymbols)) {
        AffineExpr var = (i < numDims) ? getAffineDimExpr(i, &ctx)
                                       : getAffineSymbolExpr(i - numDims, &ctx);
        param_expr = param_expr * var;
      }
      equation = equation + param_expr;
    }
    AffineExpr rhs = getAffineConstantExpr(0, &ctx);
    if (isValid) {
      constraints.emplace_back(equation + rhs);
      isEqs.emplace_back(t < eqs ? true : false);
    }
  }
  return constraints.empty()
             ? IntegerSet::getEmptySet(numDims, numSymbols, &ctx)
             : IntegerSet::get(numDims, numSymbols, constraints, isEqs);
  // return IntegerSet::get(numDims, numSymbols, constraints, isEqs);
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
  IntegerRelation reli =
      eles[0].isEmptyIntegerSet()
          ? IntegerRelation::getUniverse(
                PresburgerSpace::getRelationSpace(0, numDims, numSymbols, 0))
          : affine::FlatAffineValueConstraints(eles[0]);
  PresburgerRelation rel(reli);
  for (unsigned i = 1, e = eles.size(); i < e; ++i) {
    IntegerRelation reli =
        eles[i].isEmptyIntegerSet()
            ? IntegerRelation::getUniverse(
                  PresburgerSpace::getRelationSpace(0, numDims, numSymbols, 0))
            : affine::FlatAffineValueConstraints(eles[i]);
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
                                   unsigned long long &size, bool useSimplify) {
  if (useSimplify) {
    simplifyForCountFunc(relation);
  }
  unsigned var = relation.getNumVars();
  for (auto ele : relation.getAllDisjuncts()) {
    size += ele.getNumConstraints() * (var + 1);
  }
}

void FPLOutputOneSet(const IntegerRelation &relation, std::ostream &out) {
  int eqs = relation.getNumEqualities(), ineqs = relation.getNumInequalities();
  out << relation.getNumEqualities() << " " << relation.getNumInequalities()
      << std::endl;
  for (int i = 0; i < eqs; ++i) {
    auto eq = relation.getEquality64(i);
    for (auto &num : eq) {
      out << num << " ";
    }
    out << std::endl;
  }
  for (int i = 0; i < ineqs; ++i) {
    auto ineq = relation.getInequality64(i);
    for (auto &num : ineq) {
      out << num << " ";
    }
    out << std::endl;
  }
  out << std::endl;
}

void FPLOutputMap(PresburgerRelation &relation, std::ostream &out) {
  out << relation.getSpace().getNumDimVars() << " "
      << relation.getNumSymbolVars() << " " << relation.getNumDisjuncts()
      << std::endl;
  auto disjuncts = relation.getAllDisjuncts();
  for (auto &disjunct : disjuncts) {
    FPLOutputOneSet(disjunct, out);
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

void FPLSetupUnion(const benchmark::State &state) {
  fileName = "./PresburgerSetUnion";
  readFunc = FPLParseThreeCaseUseTwoCase;
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    benchmark::DoNotOptimize(a.unionSet(b));
  };
  binaryReturnExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    return a.unionSet(b);
  };
}

void FPLSetupUnionSimplify(const benchmark::State &state) {
  fileName = "./PresburgerSetUnion";
  readFunc = FPLParseThreeCaseUseTwoCase;
  binarySimplifyExecFunc = [](PresburgerSimpifyRelation &a,
                              PresburgerSimpifyRelation &b) {
    benchmark::DoNotOptimize(a.unionSet(b));
  };
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {};
  simplifyForCountFunc = [](PresburgerRelation &a) {
    PresburgerSimpifyRelation rel(a);
    a = rel.simplify();
  };
  binaryReturnExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    return PresburgerSimpifyRelation(a).unionSet(b);
  };
}

void FPLSetupSubtract(const benchmark::State &state) {
  fileName = "./PresburgerSetSubtract";
  readFunc = FPLParseThreeCaseUseTwoCase;
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    benchmark::DoNotOptimize(a.subtract(b));
  };
  binaryReturnExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    return a.subtract(b);
  };
}

void FPLSetupSubtractSimplify(const benchmark::State &state) {
  fileName = "./PresburgerSetSubtract";
  readFunc = FPLParseThreeCaseUseTwoCase;
  binarySimplifyExecFunc = [](PresburgerSimpifyRelation &a,
                              PresburgerSimpifyRelation &b) {
    benchmark::DoNotOptimize(a.subtract(b));
  };
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {};
  simplifyForCountFunc = [](PresburgerRelation &a) {
    PresburgerSimpifyRelation rel(a);
    a = rel.simplify();
  };
  binaryReturnExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    return PresburgerSimpifyRelation(a).subtract(b);
  };
}

void FPLSetupComplement(const benchmark::State &state) {
  fileName = "./PresburgerSetComplement";
  readFunc = FPLParseTwoCaseUseOneCase;
  unaryExecFunc = [](PresburgerRelation &a) {
    benchmark::DoNotOptimize(a.complement());
  };
  unaryReturnExecFunc = [](PresburgerRelation &a) { return a.complement(); };
}

void FPLSetupComplementSimplify(const benchmark::State &state) {
  fileName = "./PresburgerSetComplement";
  readFunc = FPLParseTwoCaseUseOneCase;
  unarySimplifyExecFunc = [](PresburgerSimpifyRelation &a) {
    benchmark::DoNotOptimize(a.complement());
  };
  unaryExecFunc = [](PresburgerRelation &a) {};
  simplifyForCountFunc = [](PresburgerRelation &a) {
    PresburgerSimpifyRelation rel(a);
    a = rel.simplify();
  };
  unaryReturnExecFunc = [](PresburgerRelation &a) { return a.complement(); };
}

void FPLSetupIntersect(const benchmark::State &state) {
  fileName = "./PresburgerSetIntersect";
  readFunc = FPLParseThreeCaseUseTwoCase;
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    benchmark::DoNotOptimize(a.intersect(b));
  };
  binaryReturnExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    return a.intersect(b);
  };
}

void FPLSetupIntersectSimplify(const benchmark::State &state) {
  fileName = "./PresburgerSetIntersect";
  readFunc = FPLParseThreeCaseUseTwoCase;
  binarySimplifyExecFunc = [](PresburgerSimpifyRelation &a,
                              PresburgerSimpifyRelation &b) {
    benchmark::DoNotOptimize(a.intersect(b));
  };
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {};
  simplifyForCountFunc = [](PresburgerRelation &a) {
    PresburgerSimpifyRelation rel(a);
    a = rel.simplify();
  };
  binaryReturnExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    return PresburgerSimpifyRelation(a).intersect(b);
  };
}

void FPLSetupIsEqual(const benchmark::State &state) {
  fileName = "./PresburgerSetEqual";
  readFunc = FPLParseTwoCaseOntInt;
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    benchmark::DoNotOptimize(a.isEqual(b));
  };
  binaryReturnExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    return PresburgerRelation::getEmpty(a.getSpace());
  };
}

void FPLSetupIsEqualSimplify(const benchmark::State &state) {
  fileName = "./PresburgerSetEqual";
  readFunc = FPLParseTwoCaseOntInt;
  binarySimplifyExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    benchmark::DoNotOptimize(a.isEqual(b));
  };
  binaryExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {};
  simplifyForCountFunc = [](PresburgerRelation &a) {
    PresburgerSimpifyRelation rel(a);
    a = rel.simplify();
  };
  binaryReturnExecFunc = [](PresburgerRelation &a, PresburgerRelation &b) {
    return PresburgerRelation::getEmpty(a.getSpace());
  };
}

void FPLSetupIsEmpty(const benchmark::State &state) {
  fileName = "./PresburgerSetEmpty";
  readFunc = FPLParseOneCaseOneInt;
  unaryExecFunc = [](PresburgerRelation &a) {
    benchmark::DoNotOptimize(a.isIntegerEmpty());
  };
  unaryReturnExecFunc = [](PresburgerRelation &a) {
    return PresburgerRelation::getEmpty(a.getSpace());
  };
}

void FPLSetupIsEmptySimplify(const benchmark::State &state) {
  fileName = "./PresburgerSetEmpty";
  readFunc = FPLParseOneCaseOneInt;
  unarySimplifyExecFunc = [](PresburgerSimpifyRelation &a) {
    benchmark::DoNotOptimize(a.isIntegerEmpty());
  };
  unaryExecFunc = [](PresburgerRelation &a) {};
  simplifyForCountFunc = [](PresburgerRelation &a) {
    PresburgerSimpifyRelation rel(a);
    a = rel.simplify();
  };
  unaryReturnExecFunc = [](PresburgerRelation &a) {
    return PresburgerRelation::getEmpty(a.getSpace());
  };
}

template <bool useSimplify = false>
void BM_FPLUnaryOperationCheck(benchmark::State &state) {
  std::vector<PresburgerRelation> setsA, setsB;
  std::vector<PresburgerSimpifyRelation> setsAT;
  readFunc(fileName, setsA, setsB);
  if (useSimplify) {
    for (auto &rel : setsA) {
      simplifyForCountFunc(rel);
      setsAT.emplace_back(rel);
    }
  }
  size_t num = setsA.size();
  for (auto _ : state) {
    for (size_t i = 0; i < num; ++i) {
      if (useSimplify) {
        unarySimplifyExecFunc(setsAT[i]);
      }
      unaryExecFunc(setsA[i]);
    }
  }
  unsigned long long relationSize = 0;
  for (auto &rel : setsA) {
    FPLCountPreburgerRelationSize(rel, relationSize, useSimplify);
  }
  state.counters["Constraint Size"] = relationSize;
  unsigned long long resultSize = 0;
  for (auto &rel : setsA) {
    PresburgerRelation result = unaryReturnExecFunc(rel);
    FPLCountPreburgerRelationSize(result, resultSize, false);
  }
  state.counters["Result Size"] = resultSize;

  // output fpl relation for simplify
  if (useSimplify) {
    std::string outputFileName = fileName + "_fpl_simplify_relation";
    std::ofstream out(outputFileName);
    out << setsA.size() << std::endl;
    for (auto &rel : setsA) {
      simplifyForCountFunc(rel);
      FPLOutputMap(rel, out);
    }
    out.close();
  }

  // log info
  std::vector<int> consSizes;
  std::vector<double> consTimes;
  std::vector<int> resultSizes;
  for (size_t i = 0; i < num; ++i) {
    auto begin = std::chrono::steady_clock::now();
    // exec full func
    if (useSimplify) {
      unarySimplifyExecFunc(setsAT[i]);
    }
    unaryExecFunc(setsA[i]);
    auto end = std::chrono::steady_clock::now();
    consTimes.emplace_back(
        std::chrono::duration<double, std::nano>(end - begin).count());

    unsigned long long size = 0;
    FPLCountPreburgerRelationSize(setsA[i], size, useSimplify);
    consSizes.push_back(size);

    size = 0;
    PresburgerRelation result = unaryReturnExecFunc(setsA[i]);
    resultSizes.push_back(size);
  }
  auto logFileName =
      fileName + "_fpl" + (useSimplify ? "_simplify" : "") + "_info.csv";
  LogAllInfo(logFileName, consSizes, consTimes, resultSizes);
}

template void BM_FPLUnaryOperationCheck<false>(benchmark::State &state);
template void BM_FPLUnaryOperationCheck<true>(benchmark::State &state);

template <bool useSimplify = false>
void BM_FPLBinaryOperationCheck(benchmark::State &state) {
  std::vector<PresburgerRelation> setsA, setsB;
  std::vector<PresburgerSimpifyRelation> setsAT, setsBT;
  readFunc(fileName, setsA, setsB);
  size_t num = setsA.size();
  if (useSimplify) {
    for (size_t i = 0; i < num; ++i) {
      simplifyForCountFunc(setsA[i]);
      simplifyForCountFunc(setsB[i]);
      setsAT.emplace_back(setsA[i]);
      setsBT.emplace_back(setsB[i]);
    }
  }

  for (auto _ : state) {
    for (size_t i = 0; i < num; ++i) {
      if (useSimplify) {
        binarySimplifyExecFunc(setsAT[i], setsBT[i]);
      }
      binaryExecFunc(setsA[i], setsB[i]);
    }
  }
  unsigned long long relationSize = 0;
  for (auto &rel : setsA) {
    FPLCountPreburgerRelationSize(rel, relationSize, useSimplify);
  }
  for (auto &rel : setsB) {
    FPLCountPreburgerRelationSize(rel, relationSize, useSimplify);
  }
  // printf("Now exact size %llu\n", relationSize);
  state.counters["Constraint Size"] = relationSize;
  unsigned long long resultSize = 0;
  for (size_t i = 0; i < num; ++i) {
    PresburgerRelation result = binaryReturnExecFunc(setsA[i], setsB[i]);
    FPLCountPreburgerRelationSize(result, resultSize, false);
  }
  state.counters["Result Size"] = resultSize;

  // output fpl relation
  if (useSimplify) {
    std::string outputFileName = fileName + "_fpl_simplify_relation";
    std::ofstream out(outputFileName);
    out << num << std::endl;
    for (size_t i = 0; i < num; ++i) {
      simplifyForCountFunc(setsA[i]);
      simplifyForCountFunc(setsB[i]);
      FPLOutputMap(setsA[i], out);
      FPLOutputMap(setsB[i], out);
      PresburgerRelation result = binaryReturnExecFunc(setsA[i], setsB[i]);
      FPLOutputMap(result, out);
    }
    out.close();
  }

  // log info
  std::vector<int> consSizes;
  std::vector<int> resultSizes;
  std::vector<double> consTimes;
  for (size_t i = 0; i < num; ++i) {
    auto begin = std::chrono::steady_clock::now();
    // exec full func
    if (useSimplify) {
      binarySimplifyExecFunc(setsAT[i], setsBT[i]);
    }
    binaryExecFunc(setsA[i], setsB[i]);
    auto end = std::chrono::steady_clock::now();
    consTimes.emplace_back(
        std::chrono::duration<double, std::nano>(end - begin).count());

    unsigned long long size = 0;
    FPLCountPreburgerRelationSize(setsA[i], size, useSimplify);
    FPLCountPreburgerRelationSize(setsB[i], size, useSimplify);
    consSizes.push_back(size);

    size = 0;
    PresburgerRelation result = binaryReturnExecFunc(setsA[i], setsB[i]);
    FPLCountPreburgerRelationSize(result, size, false);
    resultSizes.push_back(size);
  }
  auto logFileName =
      fileName + "_fpl" + (useSimplify ? "_simplify" : "") + "_info.csv";
  LogAllInfo(logFileName, consSizes, consTimes, resultSizes);
}

template void BM_FPLBinaryOperationCheck<false>(benchmark::State &state);
template void BM_FPLBinaryOperationCheck<true>(benchmark::State &state);