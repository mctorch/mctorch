#include <ATen/native/cuda/Normalization.cuh>

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda(const Tensor& self, const Tensor& weight, const Tensor& bias,
                                                   const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double epsilon) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_cuda", [&] {
      if (cuda::detail::canUse32BitIndexMath(self)) {
        return batch_norm_cuda_template<scalar_t, int32_t>(self, weight, bias, running_mean, running_var, train, momentum, epsilon);
      } else {
        return batch_norm_cuda_template<scalar_t, int64_t>(self, weight, bias, running_mean, running_var, train, momentum, epsilon);
      }
    });
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(const Tensor& grad_out, const Tensor& self, const Tensor& weight, const Tensor& running_mean, const Tensor& running_var,
                                                            const Tensor& save_mean, const Tensor& save_invstd, bool train, double epsilon, std::array<bool,3> grad_input_mask) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_backward_cuda", [&] {
      if (cuda::detail::canUse32BitIndexMath(self)) {
        return batch_norm_backward_cuda_template<scalar_t, int32_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      } else {
        return batch_norm_backward_cuda_template<scalar_t, int64_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      }
    });
}

std::tuple<Tensor, Tensor> batch_norm_stats_cuda(const Tensor& self, double epsilon) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_stats_cuda", [&] {
      if (cuda::detail::canUse32BitIndexMath(self)) {
        return batch_norm_stats_cuda_template<scalar_t, int32_t>(self, epsilon);
      } else {
        return batch_norm_stats_cuda_template<scalar_t, int64_t>(self, epsilon);
      }
    });
}

Tensor batch_norm_elemt_cuda(const Tensor& self, const Tensor& weight, const Tensor& bias,
                             const Tensor& mean, const Tensor& invstd, double epsilon) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_elemt", [&] {
      if (cuda::detail::canUse32BitIndexMath(self)) {
        return batch_norm_elemt_cuda_template<scalar_t, int32_t>(self, weight, bias, mean, invstd, epsilon);
      } else {
        return batch_norm_elemt_cuda_template<scalar_t, int64_t>(self, weight, bias, mean, invstd, epsilon);
      }
    });
}

// accepting input(self) here to determine template data types, since running_mean/running_var are optional
std::tuple<Tensor, Tensor> batch_norm_gather_stats_cuda(const Tensor& self, const Tensor& mean, const Tensor& invstd, const Tensor& running_mean,
                                                        const Tensor& running_var, double momentum, double epsilon, int64_t count) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_update_stats_cuda", [&] {
      int world_size = mean.size(1);
      using accscalar_t = at::acc_type<scalar_t, true>;
      if (cuda::detail::canUse32BitIndexMath(self)) {
        return batch_norm_gather_stats_cuda_template<scalar_t, accscalar_t, int32_t>(mean, invstd, running_mean, running_var, momentum, epsilon, static_cast<int32_t>(count));
      } else {
        return batch_norm_gather_stats_cuda_template<scalar_t, accscalar_t, int64_t>(mean, invstd, running_mean, running_var, momentum, epsilon, count);
      }
    });
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_cuda(const Tensor& self, const Tensor& input, const Tensor& mean,
                                                                           const Tensor& invstd, bool input_g, bool weight_g, bool bias_g) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_backward_reduce", [&] {
      if (cuda::detail::canUse32BitIndexMath(self)) {
        return batch_norm_backward_reduce_cuda_template<scalar_t, int32_t>(self, input, mean, invstd, input_g, weight_g, bias_g);
      } else {
        return batch_norm_backward_reduce_cuda_template<scalar_t, int64_t>(self, input, mean, invstd, input_g, weight_g, bias_g);
      }
    });
}

Tensor batch_norm_backward_elemt_cuda(const Tensor& self, const Tensor& input, const Tensor& mean, const Tensor& invstd,
                                      const Tensor& weight, const Tensor& mean_dy, const Tensor& mean_dy_xmu) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_backward_elemt", [&] {
      if (cuda::detail::canUse32BitIndexMath(self)) {
        return batch_norm_backward_elemt_cuda_template<scalar_t, int32_t>(self, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
      } else {
        return batch_norm_backward_elemt_cuda_template<scalar_t, int64_t>(self, input, mean, invstd, weight, mean_dy, mean_dy_xmu);
      }
    });
}

std::tuple<Tensor, Tensor> batch_norm_update_stats_cuda(
        const Tensor& self, const Tensor& running_mean, const Tensor& running_var, double momentum) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_backward", [&] {
      auto mean_st = running_mean.dtype();
      auto var_st = running_var.dtype();
      AT_CHECK(mean_st == var_st, "running_mean and running_var need to have the same data types");
      // <sigh> Some workloads depend on passing in half input and float stats, which is
      // usually handled by cuDNN. However, the JIT sometimes replaces cuDNN calls with this
      // one so it needs to support the same case, or people start to complain.
      bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
      if (cuda::detail::canUse32BitIndexMath(self)) {
        if (is_half_float) {
          return batch_norm_update_stats_cuda_template<at::Half, float, int32_t>(self, running_mean, running_var, momentum);
        } else {
          return batch_norm_update_stats_cuda_template<scalar_t, scalar_t, int32_t>(self, running_mean, running_var, momentum);
        }
      } else {
        if (is_half_float) {
          return batch_norm_update_stats_cuda_template<at::Half, float, int64_t>(self, running_mean, running_var, momentum);
        } else {
          return batch_norm_update_stats_cuda_template<scalar_t, scalar_t, int64_t>(self, running_mean, running_var, momentum);
        }
      }
    });
}

} } // namespace at::native
