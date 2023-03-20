#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include <fstream>

using namespace mlir;
using namespace presburger;

void print(llvm::raw_ostream &os, SmallVector<IntegerSet, 4> &eles) {
  os << eles[0].getNumDims() << " " << eles[0].getNumSymbols() << " "
     << eles.size() << "\n";
  for (auto &ele : eles) {
    IntegerPolyhedron tmp = FlatAffineValueConstraints(ele);
    os << tmp.getNumEqualities() << " " << tmp.getNumInequalities() << "\n";
    for (unsigned i = 0, e = tmp.getNumEqualities(); i < e; ++i) {
      for (unsigned j = 0, f = tmp.getNumCols(); j < f; ++j) {
        os << tmp.atEq(i, j) << " ";
      }
      os << "\n";
    }
    for (unsigned i = 0, e = tmp.getNumInequalities(); i < e; ++i) {
      for (unsigned j = 0, f = tmp.getNumCols(); j < f; ++j) {
        os << tmp.atIneq(i, j) << " ";
      }
      os << "\n";
    }
  }
  os << "\n";
}

llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("Specify output filename"),
                   llvm::cl::value_desc("filename"));
llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                         llvm::cl::Required,
                                         llvm::cl::desc("<input file>"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  llvm::StringRef filename(InputFilename.c_str());
  std::ifstream in;
  in.open(InputFilename.c_str(), std::ios::in);
  std::error_code ec;
  llvm::raw_fd_ostream outs(OutputFilename.c_str(), ec);
  char *raw_data = new char[50020];
  MLIRContext context(MLIRContext::Threading::DISABLED);
  int n;
  in >> n;
  in.getline(raw_data, 2);
  outs << n << "\n\n";
  if (filename.contains("Equal")) {
    for (int i = 0; i < n; ++i) {
      in.getline(raw_data, 50000);
      llvm::StringRef a(raw_data);
      SmallVector<IntegerSet, 4> ia = parseMultipleIntegerSets(a, &context);
      print(outs, ia);
      in.getline(raw_data, 50000);
      llvm::StringRef b(raw_data);
      SmallVector<IntegerSet, 4> ib = parseMultipleIntegerSets(b, &context);
      print(outs, ib);
      int res;
      in >> res;
      in.getline(raw_data, 2);
      outs << res << "\n\n";
    }
  } else if (filename.contains("Complement")) {
    for (int i = 0; i < n; ++i) {
      in.getline(raw_data, 50000);
      llvm::StringRef a(raw_data);
      SmallVector<IntegerSet, 4> ia = parseMultipleIntegerSets(a, &context);
      print(outs, ia);
      in.getline(raw_data, 50000);
      llvm::StringRef b(raw_data);
      SmallVector<IntegerSet, 4> ib = parseMultipleIntegerSets(b, &context);
      print(outs, ib);
    }
  } else if (filename.contains("Empty")) {
    for (int i = 0; i < n; ++i) {
      in.getline(raw_data, 50000);
      llvm::StringRef a(raw_data);
      SmallVector<IntegerSet, 4> ia = parseMultipleIntegerSets(a, &context);
      print(outs, ia);
      int res;
      in >> res;
      in.getline(raw_data, 2);
      outs << res << "\n\n";
    }
  } else if (filename.contains("Intersect") || filename.contains("Subtract") ||
             filename.contains("Union")) {
    for (int i = 0; i < n; ++i) {
      in.getline(raw_data, 50000);
      llvm::StringRef a(raw_data);
      SmallVector<IntegerSet, 4> ia = parseMultipleIntegerSets(a, &context);
      print(outs, ia);
      in.getline(raw_data, 50000);
      llvm::StringRef b(raw_data);
      SmallVector<IntegerSet, 4> ib = parseMultipleIntegerSets(b, &context);
      print(outs, ib);
      in.getline(raw_data, 50000);
      llvm::StringRef c(raw_data);
      SmallVector<IntegerSet, 4> ic = parseMultipleIntegerSets(c, &context);
      print(outs, ic);
    }
  }
  in.close();
}