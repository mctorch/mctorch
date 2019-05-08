#include <torch/csrc/autograd/functions/tensor.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace torch { namespace autograd {

auto CopyBackwards::apply(variable_list&& grads) -> variable_list {
  check_input_variables("CopyBackwards", grads, 1);
  auto& grad = grads[0];
  variable_list grad_inputs(2);
  if (should_compute_output(0)) {
    grad_inputs[0] = at::zeros_like(grad);
  }
  if (should_compute_output(1)) {
    at::DeviceGuard device_guard(src_device);
    // TODO: What if !grad.is_cuda(), but src_device is CUDA?
    // This code is kind of weirdly asymmetric.
    if (grad.is_cuda() && grad.device() != src_device) {
      grad_inputs[1] = grad.to(
          src_type->device_type(),
          src_type->scalarType(),
          /*non_blocking=*/false,
          /*copy=*/true);
    } else {
      grad_inputs[1] = grad.toType(*src_type);
    }
  }
  return grad_inputs;
}

CopySlices::CopySlices(
    const Variable& base_var,
    at::TensorGeometry view_,
    std::shared_ptr<Function> fn_)
    : Function(),
      base(base_var),
      view(std::move(view_)),
      fn(std::move(fn_)) {
  // Take the next_edges of fn as our own, except for index 0 which goes
  // to base instead of the view.
  add_input_metadata(base_var);
  const auto num_outputs = fn->num_outputs();
  next_edges_.reserve(num_outputs);
  add_next_edge(base_var.gradient_edge());
  for (size_t i = 1; i < num_outputs; i++) {
    add_next_edge(fn->next_edge(i));
  }
}

auto CopySlices::apply(variable_list&& inputs) -> variable_list {
  check_input_variables("CopySlices", inputs, 1);
  auto& grad = inputs[0];

  if (!fn) {
    throw std::runtime_error(ERR_BACKWARD_TWICE);
  }

  auto result = at::empty_strided(base.sizes(), base.strides(), grad.options());
  result.copy_(grad);

  auto offset = view.storage_offset() - base.storage_offset();
  auto grad_slice = result.as_strided(view.sizes(), view.strides(), offset);

  // TODO: We clone grad_slice because we modify it below and "fn" might save
  // it for the backward of res. We might be able to avoid the clone() if
  // double-backprop is disabled.
  auto res = (*fn)({ grad_slice.clone() });

  variable_list grad_inputs(num_outputs());
  for (size_t i = 0; i < res.size(); i++) {
    if (should_compute_output(i)) {
      AT_ASSERT(res[i].defined());
      if (i == 0) {
        grad_slice.copy_(res[i]);
        grad_inputs[i] = std::move(result); // NOLINT(bugprone-use-after-move)
      } else {
        grad_inputs[i] = std::move(res[i]);
      }
    }
  }

  return grad_inputs;
}

void CopySlices::release_variables() {
  fn = nullptr;
}

}} // namespace torch::autograd
