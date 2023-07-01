#include "benchmark/benchmark.h"
#include "isl/constraint.h"
#include "isl/ctx.h"
#include "isl/id.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/space.h"
#include "isl/val.h"
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "ISLBench.h"
#include "utils.h"

void ISLInitId(isl_ctx *ctx, std::vector<isl_id *> &idNames) {
  for (int i = 0; i < 15; ++i) {
    char *name = new char[2];
    name[0] = 'a' + i;
    name[1] = '\0';
    idNames.emplace_back(isl_id_alloc(ctx, name, nullptr));
  }
}

isl_basic_map *ISLParseOneSet(std::istream &in, int &numDims, int &numSymbols,
                              isl_ctx *ctx, isl_space *space) {
  isl_basic_map *basic_map = isl_basic_map_universe(isl_space_copy(space));
  isl_local_space *ls = isl_local_space_from_space(space);
  int inEqs, eqs;
  in >> eqs >> inEqs;
  for (int t = 0; t < (eqs + inEqs); ++t) {
    isl_constraint *constraint;
    if (t < eqs) {
      constraint = isl_constraint_alloc_equality(isl_local_space_copy(ls));
    } else {
      constraint = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
    }
    for (int i = 0; i <= (numDims + numSymbols); ++i) {
      int64_t param;
      in >> param;
      isl_val *val = isl_val_int_from_si(ctx, param);
      if (i < numDims) {
        constraint =
            isl_constraint_set_coefficient_val(constraint, isl_dim_in, i, val);
      } else if (i < (numDims + numSymbols)) {
        constraint = isl_constraint_set_coefficient_val(
            constraint, isl_dim_param, i - numDims, val);
      } else {
        // val = isl_val_neg(val);
        constraint = isl_constraint_set_constant_val(constraint, val);
      }
    }
    basic_map = isl_basic_map_add_constraint(basic_map, constraint);
  }
  return basic_map;
}

isl_map *ISLParseOneCase(std::istream &in, isl_ctx *ctx,
                         std::vector<isl_id *> &idNames) {
  int numDims, numSymbols, numUnions;
  in >> numDims >> numSymbols >> numUnions;
  isl_space *space = isl_space_alloc(ctx, numSymbols, numDims, 0);
  for (int i = 0; i < numSymbols; ++i) {
    space = isl_space_set_dim_id(space, isl_dim_param, i, idNames[i]);
  }
  std::vector<isl_basic_map *> eles;
  for (int i = 0; i < numUnions; ++i) {
    eles.emplace_back(ISLParseOneSet(in, numDims, numSymbols, ctx, space));
    space = isl_basic_map_get_space(eles.back());
  }
  isl_map *map = isl_map_from_basic_map(eles[0]);
  for (unsigned i = 1, e = eles.size(); i < e; ++i) {
    isl_map *tmpMap = isl_map_from_basic_map(eles[i]);
    map = isl_map_union(map, tmpMap);
  }
  return map;
}

void ISLCopyMaps(std::vector<isl_map *> &sets,
                 std::vector<isl_map *> &setsCopy) {
  for (auto map : setsCopy) {
    isl_map_free(map);
  }
  setsCopy.clear();
  for (auto map : sets) {
    setsCopy.emplace_back(isl_map_copy(map));
  }
}

isl_stat ISLCountBasicMapSize(isl_basic_map *basic_map, void *user) {
  unsigned long long *size = static_cast<unsigned long long *>(user);
  unsigned dims = isl_basic_map_dim(basic_map, isl_dim_all);
  isl_size cons = isl_basic_map_n_constraint(basic_map);
  (*size) = (*size) + ((dims + 1) * cons);
  return isl_stat_ok;
}

void ISLCountMapSize(isl_map *map, unsigned long long &size) {
  isl_map_foreach_basic_map(map, ISLCountBasicMapSize, &size);
}

isl_stat ISLOutputBasicMap(isl_basic_map *basic_map, void *fs) {
  std::ostream *out = static_cast<std::ostream *>(fs);
  auto eqs_mat =
      isl_basic_map_equalities_matrix(basic_map, isl_dim_in, isl_dim_out,
                                      isl_dim_param, isl_dim_div, isl_dim_cst);
  auto ineqs_mat = isl_basic_map_inequalities_matrix(basic_map, isl_dim_in,
                                                     isl_dim_out, isl_dim_param,
                                                     isl_dim_div, isl_dim_cst);
  (*out) << isl_mat_rows(eqs_mat) << " " << isl_mat_rows(ineqs_mat)
         << std::endl;
  int eqs_row = isl_mat_rows(eqs_mat);
  int eqs_col = isl_mat_cols(eqs_mat);
  for (int i = 0; i < eqs_row; ++i) {
    for (int j = 0; j < eqs_col; ++j) {
      isl_val *val = isl_mat_get_element_val(eqs_mat, i, j);
      (*out) << isl_val_get_num_si(val) << " ";
    }
    (*out) << std::endl;
  }
  int ineqs_row = isl_mat_rows(ineqs_mat);
  int ineqs_col = isl_mat_cols(ineqs_mat);
  for (int i = 0; i < ineqs_row; ++i) {
    for (int j = 0; j < ineqs_col; ++j) {
      isl_val *val = isl_mat_get_element_val(ineqs_mat, i, j);
      (*out) << isl_val_get_num_si(val) << " ";
    }
    (*out) << std::endl;
  }
  (*out) << std::endl;
  return isl_stat_ok;
}

void ISLOutputMap(isl_map *map, std::ostream &out) {
  out << isl_map_dim(map, isl_dim_in) << " " << isl_map_dim(map, isl_dim_param)
      << " " << isl_map_n_basic_map(map) << std::endl;
  isl_map_foreach_basic_map(map, ISLOutputBasicMap, &out);
}

static std::vector<isl_map *> setsNoUse;

void ISLParseOneCaseOneInt(std::string &fileName, std::vector<isl_map *> &sets,
                           std::vector<isl_map *> &setsNull, isl_ctx *ctx,
                           std::vector<isl_id *> &idNames) {
  std::ifstream in(fileName);
  size_t num;
  in >> num;
  for (size_t i = 0; i < num; ++i) {
    int tmp;
    sets.emplace_back(ISLParseOneCase(in, ctx, idNames));
    in >> tmp;
  }
  in.close();
}

void ISLParseTwoCaseUseOneCase(std::string &fileName,
                               std::vector<isl_map *> &sets,
                               std::vector<isl_map *> &setsNull, isl_ctx *ctx,
                               std::vector<isl_id *> &idNames) {
  std::ifstream in(fileName);
  size_t num;
  in >> num;
  for (size_t i = 0; i < num; ++i) {
    sets.emplace_back(ISLParseOneCase(in, ctx, idNames));
    setsNoUse.emplace_back(ISLParseOneCase(in, ctx, idNames));
  }
  in.close();
}

void ISLParseTwoCaseOneInt(std::string &fileName, std::vector<isl_map *> &setsA,
                           std::vector<isl_map *> &setsB, isl_ctx *ctx,
                           std::vector<isl_id *> &idNames) {
  std::ifstream in(fileName);
  size_t num;
  in >> num;
  for (size_t i = 0; i < num; ++i) {
    int tmp;
    setsA.emplace_back(ISLParseOneCase(in, ctx, idNames));
    setsB.emplace_back(ISLParseOneCase(in, ctx, idNames));
    in >> tmp;
  }
  in.close();
}

void ISLParseThreeCaseUseTwoCase(std::string &fileName,
                                 std::vector<isl_map *> &setsA,
                                 std::vector<isl_map *> &setsB, isl_ctx *ctx,
                                 std::vector<isl_id *> &idNames) {
  std::ifstream in(fileName);
  size_t num;
  in >> num;
  for (size_t i = 0; i < num; ++i) {
    setsA.emplace_back(ISLParseOneCase(in, ctx, idNames));
    setsB.emplace_back(ISLParseOneCase(in, ctx, idNames));
    setsNoUse.emplace_back(ISLParseOneCase(in, ctx, idNames));
  }
  in.close();
}

static std::function<void(std::string &, std::vector<isl_map *> &,
                          std::vector<isl_map *> &, isl_ctx *,
                          std::vector<isl_id *> &)>
    readFunc;
static std::function<isl_map *(isl_map *, double &)> unaryExecFunc;
static std::function<isl_map *(isl_map *, isl_map *, double &)> binaryExecFunc;
static std::string fileName;

void ISLSetupUnion(const benchmark::State &state) {
  fileName = "./PresburgerSetUnion";
  readFunc = ISLParseThreeCaseUseTwoCase;
  binaryExecFunc = [](isl_map *a, isl_map *b, double &execTime) {
    auto start = std::chrono::high_resolution_clock::now();
    benchmark::DoNotOptimize(a = isl_map_union(a, b));
    auto end = std::chrono::high_resolution_clock::now();
    execTime +=
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    return a;
  };
}

void ISLSetupSubtract(const benchmark::State &state) {
  fileName = "./PresburgerSetSubtract";
  readFunc = ISLParseThreeCaseUseTwoCase;
  binaryExecFunc = [](isl_map *a, isl_map *b, double &execTime) {
    auto start = std::chrono::high_resolution_clock::now();
    benchmark::DoNotOptimize(a = isl_map_subtract(a, b));
    auto end = std::chrono::high_resolution_clock::now();
    execTime +=
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    return a;
  };
}

void ISLSetupComplement(const benchmark::State &state) {
  fileName = "./PresburgerSetComplement";
  readFunc = ISLParseTwoCaseUseOneCase;
  unaryExecFunc = [](isl_map *a, double &execTime) {
    auto start = std::chrono::high_resolution_clock::now();
    benchmark::DoNotOptimize(a = isl_map_complement(a));
    auto end = std::chrono::high_resolution_clock::now();
    execTime +=
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    return a;
  };
}

void ISLSetupIntersect(const benchmark::State &state) {
  fileName = "./PresburgerSetIntersect";
  readFunc = ISLParseThreeCaseUseTwoCase;
  binaryExecFunc = [](isl_map *a, isl_map *b, double &execTime) {
    auto start = std::chrono::high_resolution_clock::now();
    benchmark::DoNotOptimize(a = isl_map_intersect(a, b));
    auto end = std::chrono::high_resolution_clock::now();
    execTime +=
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    return a;
  };
}

void ISLSetupIsEqual(const benchmark::State &state) {
  fileName = "./PresburgerSetEqual";
  readFunc = ISLParseTwoCaseOneInt;
  binaryExecFunc = [](isl_map *a, isl_map *b, double &execTime) {
    auto start = std::chrono::high_resolution_clock::now();
    benchmark::DoNotOptimize(isl_map_is_equal(a, b));
    auto end = std::chrono::high_resolution_clock::now();
    execTime +=
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    setsNoUse.emplace_back(a);
    setsNoUse.emplace_back(b);
    return isl_map_empty(isl_map_get_space(a));
  };
}

void ISLSetupIsEmpty(const benchmark::State &state) {
  fileName = "./PresburgerSetEmpty";
  readFunc = ISLParseOneCaseOneInt;
  unaryExecFunc = [](isl_map *map, double &execTime) {
    auto start = std::chrono::high_resolution_clock::now();
    benchmark::DoNotOptimize(isl_map_is_empty(map));
    auto end = std::chrono::high_resolution_clock::now();
    execTime +=
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    setsNoUse.emplace_back(map);
    return isl_map_empty(isl_map_get_space(map));
  };
}

void BM_ISLUnaryOperationCheck(benchmark::State &state) {
  isl_ctx *ctx = isl_ctx_alloc();
  std::vector<isl_map *> setsA, setsB, setsCopyA;
  std::vector<isl_id *> ids;
  ISLInitId(ctx, ids);
  readFunc(fileName, setsA, setsB, ctx, ids);
  for (auto _ : state) {
    ISLCopyMaps(setsA, setsCopyA);

    double execTime = 0;

    for (size_t i = 0, n = setsCopyA.size(); i < n; ++i) {
      setsCopyA[i] = unaryExecFunc(setsCopyA[i], execTime);
    }
    state.SetIterationTime(execTime);
  }

  ISLCopyMaps(setsA, setsCopyA);
  unsigned long long resultSize = 0;
  for (size_t i = 0, n = setsCopyA.size(); i < n; ++i) {
    double tmp = 0;
    setsCopyA[i] = unaryExecFunc(setsCopyA[i], tmp);
    ISLCountMapSize(setsCopyA[i], resultSize);
  }
  state.counters["Result Size"] = resultSize;

  ISLCopyMaps(setsA, setsCopyA);
  unsigned long long relationSize = 0;
  for (size_t i = 0, n = setsCopyA.size(); i < n; ++i) {
    // setsCopyA[i] = isl_map_coalesce(setsCopyA[i]);
    ISLCountMapSize(setsCopyA[i], relationSize);
  }
  state.counters["Constraint Size"] = relationSize;

  // output isl relation
  std::string outputFileName = fileName + "_isl_relation";
  std::ofstream out(outputFileName);
  out << setsCopyA.size() << std::endl;
  for (size_t i = 0, n = setsCopyA.size(); i < n; ++i) {
    ISLOutputMap(setsCopyA[i], out);
  }
  out.close();

  // log info
  std::vector<int> consSizes;
  std::vector<double> consTimes;
  std::vector<int> resultSizes;
  ISLCopyMaps(setsA, setsCopyA);
  for (auto &map : setsCopyA) {
    unsigned long long size = 0;
    // map = isl_map_coalesce(map);
    ISLCountMapSize(map, size);
    consSizes.push_back(size);
  }
  ISLCopyMaps(setsA, setsCopyA);
  for (auto &map : setsCopyA) {
    double execTime = 0;
    map = unaryExecFunc(map, execTime);
    consTimes.emplace_back(execTime);
  }
  ISLCopyMaps(setsA, setsCopyA);
  for (auto &map : setsCopyA) {
    double tmp = 0;
    map = unaryExecFunc(map, tmp);
    unsigned long long size = 0;
    ISLCountMapSize(map, size);
    resultSizes.push_back(size);
  }
  auto logFileName = fileName + "_isl" + "_info.csv";
  LogAllInfo(logFileName, consSizes, consTimes, resultSizes);

  for (auto map : setsCopyA) {
    isl_map_free(map);
  }
  for (auto map : setsA) {
    isl_map_free(map);
  }
  for (auto map : setsNoUse) {
    isl_map_free(map);
  }
  setsNoUse.clear();
  for (auto id : ids) {
    isl_id_free(id);
  }
  isl_ctx_free(ctx);
}

void BM_ISLBinaryOperationCheck(benchmark::State &state) {
  isl_ctx *ctx = isl_ctx_alloc();
  std::vector<isl_map *> setsA, setsB, setsCopyA, setsCopyB;
  std::vector<isl_id *> ids;
  ISLInitId(ctx, ids);
  readFunc(fileName, setsA, setsB, ctx, ids);

  for (auto _ : state) {
    ISLCopyMaps(setsA, setsCopyA);
    setsCopyB.clear();
    ISLCopyMaps(setsB, setsCopyB);

    double execTime = 0;
    for (size_t i = 0, n = setsCopyA.size(); i < n; ++i) {
      setsCopyA[i] = binaryExecFunc(setsCopyA[i], setsCopyB[i], execTime);
    }
    state.SetIterationTime(execTime);
  }

  ISLCopyMaps(setsA, setsCopyA);
  setsCopyB.clear();
  ISLCopyMaps(setsB, setsCopyB);
  unsigned long long resultSize = 0;
  for (size_t i = 0, n = setsCopyA.size(); i < n; ++i) {
    double tmp = 0;
    setsCopyA[i] = binaryExecFunc(setsCopyA[i], setsCopyB[i], tmp);
    ISLCountMapSize(setsCopyA[i], resultSize);
  }
  state.counters["Result Size"] = resultSize;

  ISLCopyMaps(setsA, setsCopyA);
  setsCopyB.clear();
  ISLCopyMaps(setsB, setsCopyB);
  unsigned long long relationSize = 0;
  for (size_t i = 0, n = setsCopyA.size(); i < n; ++i) {
    // setsCopyA[i] = isl_map_coalesce(setsCopyA[i]);
    // setsCopyB[i] = isl_map_coalesce(setsCopyB[i]);
    ISLCountMapSize(setsCopyA[i], relationSize);
    ISLCountMapSize(setsCopyB[i], relationSize);
    isl_map_free(setsCopyB[i]);
  }
  state.counters["Constraint Size"] = relationSize;

  // output isl relation
  std::string outputFileName = fileName + "_isl_relation";
  std::ofstream out(outputFileName);
  out << setsCopyA.size() << std::endl;
  setsCopyB.clear();
  ISLCopyMaps(setsB, setsCopyB);
  for (size_t i = 0, n = setsCopyA.size(); i < n; ++i) {
    ISLOutputMap(setsCopyA[i], out);
    // setsCopyB[i] = isl_map_coalesce(setsCopyB[i]);
    ISLOutputMap(setsCopyB[i], out);
    double tmp = 0;
    setsCopyA[i] = binaryExecFunc(setsCopyA[i], setsCopyB[i], tmp);
    ISLOutputMap(setsCopyA[i], out);
  }
  out.close();

  // log info
  std::vector<int> consSizes;
  std::vector<double> consTimes;
  std::vector<int> resultSizes;
  ISLCopyMaps(setsA, setsCopyA);
  setsCopyB.clear();
  ISLCopyMaps(setsB, setsCopyB);
  for (int i = 0, n = setsCopyA.size(); i < n; ++i) {
    unsigned long long size = 0;
    // setsCopyA[i] = isl_map_coalesce(setsCopyA[i]);
    // setsCopyB[i] = isl_map_coalesce(setsCopyB[i]);
    ISLCountMapSize(setsCopyA[i], size);
    ISLCountMapSize(setsCopyB[i], size);
    consSizes.push_back(size);
    isl_map_free(setsCopyB[i]);
  }
  ISLCopyMaps(setsA, setsCopyA);
  setsCopyB.clear();
  ISLCopyMaps(setsB, setsCopyB);
  for (int i = 0, n = setsCopyA.size(); i < n; ++i) {
    double execTime = 0;
    setsCopyA[i] = binaryExecFunc(setsCopyA[i], setsCopyB[i], execTime);
    consTimes.emplace_back(execTime);
  }
  ISLCopyMaps(setsA, setsCopyA);
  setsCopyB.clear();
  ISLCopyMaps(setsB, setsCopyB);
  for (int i = 0, n = setsCopyA.size(); i < n; ++i) {
    double tmp = 0;
    setsCopyA[i] = binaryExecFunc(setsCopyA[i], setsCopyB[i], tmp);
    unsigned long long size = 0;
    ISLCountMapSize(setsCopyA[i], size);
    resultSizes.push_back(size);
  }

  auto logFileName = fileName + "_isl" + "_info.csv";
  LogAllInfo(logFileName, consSizes, consTimes, resultSizes);

  for (auto map : setsCopyA) {
    isl_map_free(map);
  }
  for (auto map : setsA) {
    isl_map_free(map);
  }
  for (auto map : setsB) {
    isl_map_free(map);
  }
  for (auto map : setsNoUse) {
    isl_map_free(map);
  }
  setsNoUse.clear();
  for (auto id : ids) {
    isl_id_free(id);
  }
  isl_ctx_free(ctx);
}