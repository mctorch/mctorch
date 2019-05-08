#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

namespace c10d {

class Reducer {
 public:
  // The constructor takes a list of variables for every model replica.
  // The bucket assignment for this reducer is specified as a list of
  // buckets, each of which is specified as a list of indices into the
  // variables list for **a single replica** (i.e. `variables[0]`).
  explicit Reducer(
      std::vector<std::vector<torch::autograd::Variable>> replicas,
      std::vector<std::vector<size_t>> bucket_indices,
      std::shared_ptr<c10d::ProcessGroup> process_group);

  // To (re-)initialize bucket assignment, pass a list of buckets, each
  // of which is specified by a list of indices in the variables list.
  // This function performs validation that the variables within a bucket
  // all live on the same device and have the same dimensionality.
  void initialize_buckets(std::vector<std::vector<size_t>> bucket_indices);

  // This function is called when the forward function has produced an output,
  // and the user wishes to reduce gradients in the backwards pass.
  // If they don't, and wish to accumulate gradients before reducing them,
  // a call to this function can simply be omitted.
  void prepare_for_backward(
      const std::vector<torch::autograd::Variable>& outputs);

  // Returns the relative time in nanoseconds when gradients were ready,
  // with respect to the time `prepare_for_backward` was called. The outer
  // vector is for model replicas and the inner vector is for parameters.
  std::vector<std::vector<int64_t>> get_backward_stats() const {
    return backward_stats_;
  }

 protected:
  std::mutex mutex_;
  std::vector<std::vector<torch::autograd::Variable>> replicas_;
  std::shared_ptr<c10d::ProcessGroup> process_group_;

  std::vector<std::vector<std::shared_ptr<torch::autograd::Function>>>
      grad_accumulators_;
  std::unordered_map<torch::autograd::Function*, std::tuple<int, int>> func_;

  bool expect_autograd_hooks_;
  bool require_finalize_;
  bool has_marked_unused_parameters_;
  size_t next_bucket_;

  void mark_variable_ready(
      size_t replica_index,
      size_t variable_index,
      bool called_from_autograd = false);

  void mark_bucket_ready(size_t bucket_index);

  void finalize_backward();

  // A bucket replica represents [1..N] gradients to be reduced,
  // with the same dtype, on the same device.
  //
  // Batching gradients together before reducing them can result in lower
  // overhead and/or faster time to completion. Only gradients of the same type
  // and on the same device can be batched. The tensor that represents the
  // flattened gradient uses the same type and is placed on the same device.
  // Buckets are filled as the gradients they hold are computed (triggered by
  // autograd hooks). Buckets are reduced in a predetemined order that is
  // identical across processes.
  //
  struct BucketReplica {
    // Flattened (1 dimensional) contents of bucket.
    at::Tensor contents;

    // Variables that contribute to this bucket replica. Use refcounted value
    // here so that we can easily unflatten the bucket contents into the
    // participating variables after reduction has completed.
    std::vector<torch::autograd::Variable> variables;

    // Per-variable offset/length into the flat bucket contents tensor.
    std::vector<size_t> offsets;
    std::vector<size_t> lengths;

    // Number of tensors to be added before this bucket is complete.
    // This is reset to `variables.size()` every iteration.
    size_t pending;

    // TODO(@pietern)
    // Memory copies from gradient tensors into the bucket are potentially
    // done on different CUDA streams. We record an event for every copy
    // so that we can synchronize with them prior to kicking off the reduction.
    // std::vector<at::cuda::CUDAEvent> events;
  };

  // A bucket holds N bucket replicas (1 per model replica).
  //
  // If every bucket in this struct is ready, the reduction can be kicked off.
  // One bucket per replica. Reduction is kicked off when every bucket is ready.
  //
  struct Bucket {
    std::vector<BucketReplica> replicas;

    // Number of replicas to be marked done before this bucket is ready.
    size_t pending;

    // Keep work handle around when this set of buckets is being reduced.
    std::shared_ptr<c10d::ProcessGroup::Work> work;
  };

  std::vector<Bucket> buckets_;

  // A variable locator locates a particular variable in the bucket
  // structure. The `bucket_index` field points to the bucket in the `buckets_`
  // vector. The `intra_bucket_index` field points to the index of the variable
  // in any of the vector fields in the bucket replica.
  struct VariableLocator {
    // Index into the `buckets_` variable.
    size_t bucket_index;
    // Index of parameter in single bucket replica.
    size_t intra_bucket_index;
  };

  // Map the index of a variable to its location in the bucket structure.
  std::vector<VariableLocator> variable_locators_;

  // We collect the relative timestamp of every gradient being ready
  // when executing autograd. This can be used to derive a timeline of
  // the point in time buckets were ready, or ideal bucket assignment/ordering.
  int64_t backward_stats_base_;
  std::vector<std::vector<int64_t>> backward_stats_;
};

std::vector<std::vector<size_t>> compute_bucket_assignment_by_size(
    const std::vector<at::Tensor>& tensors,
    std::vector<size_t> bucket_size);

} // namespace c10d
