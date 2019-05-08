#include <torch/csrc/jit/fuser/compiler.h>

#include <ATen/ATen.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/code_template.h>
#include <torch/csrc/jit/fuser/codegen.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>
#include <torch/csrc/jit/fuser/tensor_desc.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace torch {
namespace jit {
namespace fuser {

std::mutex fusion_backends_lock_;
static std::unordered_map<at::Device::Type, FusedKernelConstructor>&
getFusionBackends() {
  static std::unordered_map<at::Device::Type, FusedKernelConstructor>
      fusion_backends;
  return fusion_backends;
}

void registerFusionBackend(
    at::Device::Type backend_type,
    FusedKernelConstructor ctor) {
  std::lock_guard<std::mutex> guard(fusion_backends_lock_);
  getFusionBackends()[backend_type] = std::move(ctor);
}

bool hasFusionBackend(at::Device::Type backend_type) {
  std::lock_guard<std::mutex> guard(fusion_backends_lock_);
  return getFusionBackends().count(backend_type);
}

const FusedKernelConstructor& getConstructor(at::Device::Type backend_type) {
  std::lock_guard<std::mutex> guard(fusion_backends_lock_);
  return getFusionBackends().at(backend_type);
}

// Counter for number of kernels compiled, used for debugging and
// creating arbitrary kernel names.
static std::atomic<size_t> next_kernel_id{0};
static int debug_fusion{-1};

size_t nCompiledKernels() {
  return next_kernel_id.load();
}

int debugFuser() {
  if (debug_fusion < 0) {
    const char* debug_env = getenv("PYTORCH_FUSION_DEBUG");
    debug_fusion = debug_env ? atoi(debug_env) : 0;
  }
  return debug_fusion;
}

// If the given node is used once by a chunk node, returns that node.
// Returns nullptr otherwise.
static const Node* usedInFusedChunk(const Value* input) {
  const auto& uses = input->uses();
  if (uses.size() == 1) {
    const Node* user = uses[0].user;
    if (user->kind() == prim::ConstantChunk) {
      return user;
    }
  }
  return nullptr;
}

static void setInputChunkDescriptors(KernelSpec& spec) {
  // We only have as many chunk descriptors as tensor inputs,
  // furthermore we know that the tensor inputs are in the
  // beginning of the fusion group's inputs.
  spec.inputChunks().reserve(spec.nTensorInputs());
  for (int64_t i = 0; i < spec.nTensorInputs(); i++) {
    const Value* input = spec.graph()->inputs()[i];
    if (const Node* chunk = usedInFusedChunk(input)) {
      spec.inputChunks().emplace_back(
          chunk->i(attr::chunks), chunk->i(attr::dim));
    } else {
      spec.inputChunks().emplace_back(1, 0);
    }
  }
}

// Run a DFS traversal to find all inputs that affect a given output value
static std::vector<int64_t> getInputDependencies(const Value* output) {
  std::vector<const Value*> queue{output};
  std::unordered_set<const Value*> inputs;
  std::unordered_set<const Value*> seen;
  while (!queue.empty()) {
    const Value* val = queue.back();
    queue.pop_back();
    const Node* producer = val->node();
    // Here we assume that only tensor inputs are used in
    // the computation of the outputs.
    // This is currently true, as the only inputs will be
    // sizes (for _grad_sum_to_size as the derivative
    // of broadcasts), which will only be used after
    // the fusion kernel, and Tensors.
    // This needs to be revisited when you start allowing
    // other things e.g. nonconstant scalars.
    if (producer->kind() == prim::Param &&
        val->type()->isSubtypeOf(TensorType::get())) {
      inputs.insert(val);
      continue;
    }
    for (const Value* input : producer->inputs()) {
      if (/*bool inserted = */ seen.insert(input).second) {
        queue.push_back(input);
      }
    }
  }

  // Convert Value* into offsets into the graph's input list
  std::vector<int64_t> offsets;
  offsets.reserve(inputs.size());
  for (const Value* input : inputs) {
    offsets.push_back(input->offset());
  }

  std::sort(offsets.begin(), offsets.end());
  return offsets;
}

static void setInputBroadcastGroups(KernelSpec& spec) {
  std::unordered_set<std::vector<int64_t>, torch::hash<std::vector<int64_t>>>
      broadcast_groups;
  for (const Value* output : (spec.graph())->outputs()) {
    if (output->node()->kind() == prim::FusedConcat) {
      for (const Value* concat_input : output->node()->inputs()) {
        broadcast_groups.insert(getInputDependencies(concat_input));
      }
    } else {
      broadcast_groups.insert(getInputDependencies(output));
    }
  }
  std::copy(
      broadcast_groups.begin(),
      broadcast_groups.end(),
      std::back_inserter(spec.inputBroadcastGroups()));
}

// This function moves _grad_sum_to_size nodes along the computation graph
// of the fusion group to the outputs and then records the shape inputs
// in order for summation to be applied after the kernel.
// Note that the correctness relies on the invariant that
// _grad_sum_to_size is only applied to gradient nodes created by autodiff.
// This is important because it ensures that in the mul and div nodes only
// one argument (in the case of div the numerator) has a summed value.
// If two arguments to mul had one, we would be in trouble, but thanks
// to the chain rule, we're OK.
// Note that this means that one kernel output may lead to several fusion
// group outputs when several outputs had the same calculation except
// for the final _grad_sum_to_size. This is also the reason why
// we need to deduplicate kernel outputs at the end of this function.
void processGradSumToSize(KernelSpec& spec) {
  auto graph = spec.graph();

  std::vector<int64_t> outputGradSumToSizes(graph->outputs().size(), -1);

  // these are expressions that might occur during autotdiff operating
  // on the gradient (matmul would likely be, too but we don't fuse it)
  // note that for mul, we know (from the chain rule) that only one
  // factor will be stemming from a calculation involving gradients so
  // we know that we can move _grad_sum_to_size across it
  // Scan the graph. We will delete nodes. We want later (in the graph)
  // _grad_sum_to_size nodes to have priority over earlier ones. Thus
  // we scan backwards.
  for (auto it = graph->nodes().rbegin(); it != graph->nodes().rend(); it++) {
    auto* node = *it;
    if (node->kind() != aten::_grad_sum_to_size) {
      continue;
    }
    bool success = trackSingleGradSumToSizeToOutputs(
        node->output(), &outputGradSumToSizes);
    AT_ASSERT(success); // check that we didn't hit anything unknown

    // remove the GradSumToSize node, a new node outside the fusion graph
    // will be inserted below
    node->output()->replaceAllUsesWith(node->inputs()[0]);
    it.destroyCurrent();
  }

  // By removing the _grad_sum_to_size notes, we might end up with
  // duplicate outputs, e.g. when having the autodiff backwards of
  // x + y + z of something with x, y, z, those will have different
  // _grad_sum_to_sizes but of the same kernel output.

  // for each fusion group output, record the corresponding kernel
  // output and possibly a _grad_sum_to_size for that output
  auto& outputMapAndSizes = spec.outputMapAndSizes();
  AT_ASSERT(outputMapAndSizes.empty());
  std::unordered_map<const Value*, int64_t> reduced_output_indices;
  int64_t newo = 0;
  for (auto osize : outputGradSumToSizes) {
    auto it = reduced_output_indices.find(graph->outputs()[newo]);
    if (it == reduced_output_indices.end()) {
      reduced_output_indices.emplace(graph->outputs()[newo], newo);
      outputMapAndSizes.emplace_back(newo, osize);
      newo++;
    } else {
      graph->eraseOutput(newo);
      outputMapAndSizes.emplace_back(it->second, osize);
    }
  }
}

// Performs "upfront" compilation where storage is known but shapes are not.
// Currently identifies how to expand all tensors so that all intermediate
// tensors are the same shape, simplifying code generation.
// Broadcast groups and chunks are identified without shape information
// using logical properties of how each works. In particular, tensors
// are always expandable to the outputs of pointwise operations they
// or their descendants are involved in, which means that in a DAG of
// pointwise operations all tensors are expandable to the (single) output.
// Note: The logic is slightly complicated by concatenation and chunking.
static void upfrontCompilation(KernelSpec& spec) {
  setInputBroadcastGroups(spec);
  setInputChunkDescriptors(spec);
  processGradSumToSize(spec);
}

int64_t registerFusion(const Node* fusion_group) {
  auto graph = normalizeGraphForCache(fusion_group->g(attr::Subgraph));

  // Don't re-register the fusion if we can use a pre-existing one
  const auto maybe_spec = lookupGraph(graph);
  if (maybe_spec) {
    return (*maybe_spec)->key();
  }

  // Unconditionally create and register the fusion
  // This is necessary to support our global disable fusions flag: if someone
  // runs some code under no-fusions mode and then runs some code with fusions
  // enabled, the second time around the returned spec from the cache should
  // be a valid spec (must have had upfrontCompilation run on it).
  const auto key = store(graph);
  const auto maybe_retrieved_spec = retrieve(key);
  AT_ASSERT(maybe_retrieved_spec);
  upfrontCompilation(**maybe_retrieved_spec);

  return key;
}

std::shared_ptr<FusedKernel> compileKernel(
    const KernelSpec& spec,
    const ArgSpec& arg_spec,
    const std::vector<int64_t>& map_size,
    const at::Device device) {
  const std::vector<TensorDesc>& input_desc = arg_spec.descs();

  auto graph = spec.graph()->copy();

  for (size_t i = 0; i < input_desc.size(); i++) {
    const auto& desc = input_desc[i];
    graph->inputs()[i]->setType(DimensionedTensorType::create(
        desc.scalar_type,
        device,
        desc.nDim())); // TODO: nDim is bad, as it is collapsed
  }

  PropagateInputShapes(graph);

  // Creates chunk and flattened input descriptions
  std::vector<PartitionDesc> chunk_desc;
  std::vector<std::pair<const Value*, const c10::optional<TensorDesc>>> flat_inputs;
  {
    size_t input_index = 0;
    for (const auto& p : graph->inputs()) {
      if (p->type()->isSubtypeOf(FloatType::get())) {
        flat_inputs.emplace_back(p, c10::nullopt);
      }
      if (!p->type()->isSubtypeOf(TensorType::get())) {
        continue;
      }
      if (const Node* chunk = usedInFusedChunk(p)) {
        int64_t dim = chunk->i(attr::dim);
        int64_t chunks = chunk->i(attr::chunks);
        chunk_desc.emplace_back(input_desc[input_index++], chunks, dim);
        for (const auto* o : chunk->outputs()) {
          flat_inputs.emplace_back(o, *chunk_desc.back().subTensorDesc());
        }
      } else {
        chunk_desc.emplace_back();
        flat_inputs.emplace_back(p, input_desc[input_index++]);
      }
    }
  }

  // Creates output, concat, and flattened output descriptions
  std::vector<TensorDesc> output_desc;
  std::vector<PartitionDesc> concat_desc;
  std::vector<std::pair<const Value*, const TensorDesc>> flat_outputs;
  for (const Value* o : graph->outputs()) {
    // Creates output description
    std::vector<int64_t> sizes = map_size;
    if (o->node()->kind() == prim::FusedConcat) {
      sizes.at(o->node()->i(attr::dim)) *= o->node()->inputs().size();
    }
    auto scalar_type = o->type()->expect<c10::DimensionedTensorType const>()->scalarType();
    auto type = CompleteTensorType::create(scalar_type, device, sizes);
    output_desc.emplace_back(type);
    const auto& desc = output_desc.back();

    // Creates concat and flattened output descriptions (relies on output desc)
    if (o->node()->kind() != prim::FusedConcat) {
      concat_desc.emplace_back();
      flat_outputs.emplace_back(o, desc);
    } else {
      const auto cat = o->node();
      concat_desc.emplace_back(desc, cat->inputs().size(), cat->i(attr::dim));
      for (const auto& c : cat->inputs()) {
        flat_outputs.emplace_back(c, *concat_desc.back().subTensorDesc());
      }
    }
  }

  // Have checked the limit at graph_fuser. Assert nothing else changing that.
  AT_ASSERT((flat_inputs.size() + flat_outputs.size()) <=
            fusion_kernel_args_limit);

  const bool use_cuda = device.is_cuda();
  const std::string name = "kernel_" + std::to_string(next_kernel_id++);
  std::string code =
      generateKernel(name, *graph, flat_inputs, flat_outputs, use_cuda);
  const FusedKernelConstructor& kernel_ctor =
      getConstructor(use_cuda ? at::DeviceType::CUDA : at::DeviceType::CPU);
  return kernel_ctor(
      device.index(),
      name,
      code,
      input_desc,
      output_desc,
      chunk_desc,
      concat_desc,
      spec.hasRandom());
}

} // namespace fuser
} // namespace jit
} // namespace torch
