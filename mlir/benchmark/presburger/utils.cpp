#include "utils.h"

void LogAllInfo(std::string &fileName, std::vector<int> &consSizes,
                std::vector<double> &consTimes) {
  std::ofstream out(fileName);
  out << "id,size,time\n";
  for (int i = 0, n = consSizes.size(); i < n; ++i) {
    out << i << "," << consSizes[i] << "," << consTimes[i] << "\n";
  }
  out.close();
}

void LogAllInfo(std::string &fileName, std::vector<int> &consSizes,
                std::vector<double> &consTimes, std::vector<int> &resultSizes) {
  std::ofstream out(fileName);
  out << "id,size,time,result_size\n";
  for (int i = 0, n = consSizes.size(); i < n; ++i) {
    out << i << "," << consSizes[i] << "," << consTimes[i] << ","
        << resultSizes[i] << "\n";
  }
  out.close();
}