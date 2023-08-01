#ifndef MLIR_BENCHMARK_PRESBURGER_UTILS_H_
#define MLIR_BENCHMARK_PRESBURGER_UTILS_H_

#include <fstream>
#include <string>
#include <vector>

void LogAllInfo(std::string &fileName, std::vector<int> &consSizes,
                std::vector<double> &consTimes);

void LogAllInfo(std::string &fileName, std::vector<int> &consSizes,
                std::vector<double> &consTimes, std::vector<int> &resultSizes);

#endif