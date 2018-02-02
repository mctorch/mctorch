#pragma once
#include <memory>
#include <string>

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {
namespace script {

struct CompilationUnitImpl;
struct CompilationUnit {
  CompilationUnit();
  void define(const std::string& str);
  void defineExtern(const std::string& str, std::unique_ptr<NetDef> netdef);
  std::unique_ptr<NetBase> createNet(Workspace* ws, const std::string& name);
  std::string getProto(const std::string& functionName) const;
  ~CompilationUnit();

 private:
  std::unique_ptr<CompilationUnitImpl> pImpl;
};

 CompilationUnit jitScriptCompile(const std::string& script);
 std::vector<at::Tensor> jitScriptExecute(const CompilationUnit& cu,
                                          const std::vector<at::Tensor>& inputs);

} // namespace script
} // namespace jit
} // namespace torch
