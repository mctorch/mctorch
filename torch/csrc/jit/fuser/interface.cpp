#include <torch/csrc/jit/fuser/interface.h>

#include <torch/csrc/jit/fuser/compiler.h>
#include <torch/csrc/jit/fuser/executor.h>
#include <torch/csrc/jit/fuser/fallback.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>

#include <stdexcept>

namespace torch {
namespace jit {

namespace detail {

// Note: CPU fusion is currently disabled due to test flakiness
bool cpu_fuser_enabled = false;

} // namespace detail

int64_t registerFusion(const Node* fusion_group) {
  return fuser::registerFusion(fusion_group);
}

void runFusion(const int64_t key, Stack& stack) {
  const auto result = fuser::runFusion(key, stack);
  if (!result)
    fuser::runFallback(key, stack);
}

bool canFuseOnCPU() {
  return fuser::hasFusionBackend(at::DeviceType::CPU) &&
      detail::cpu_fuser_enabled;
}

bool canFuseOnGPU() {
  return fuser::hasFusionBackend(at::DeviceType::CUDA);
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

// Uses the above interface by stuffing the graph into a node and treating that
// node as a fusion group.
std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group =
      wrapper_graph->insertNode(wrapper_graph->createFusionGroup());
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);
  fuser::runFusion(key, stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // Creates a fusion group node
  auto wrapper_graph = std::make_shared<Graph>();
  Node* fusion_group =
      wrapper_graph->insertNode(wrapper_graph->createFusionGroup());
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // Creates the stack, registers and runs the fusion
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);

  std::string code;
  if (!fuser::runFusion(key, stack, &code)) {
    throw std::runtime_error("Could not run fusion for graph");
  }

  return code;
}

size_t nCompiledKernels() {
  return fuser::nCompiledKernels();
}

} // namespace jit
} // namespace torch
