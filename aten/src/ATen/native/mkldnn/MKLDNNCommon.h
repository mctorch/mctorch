#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>

namespace at { namespace native {

// Custom allocator using c10 CPU allocator for `ideep::tensor`
struct AllocForMKLDNN {
  template<class computation_t = void>
  static char* malloc(size_t size) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    return (char*)allocator->raw_allocate(size);
  }

  template<class computation_t = void>
  static void free(void* p) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    allocator->raw_deallocate(p);
  }
};

// Construct aten MKL-DNN tensor given an ideep tensor
Tensor new_with_itensor_mkldnn(ideep::tensor&& it, const TensorOptions& options);

// Construct aten MKL-DNN tensor given `sizes` for allocation
Tensor new_with_sizes_mkldnn(IntArrayRef sizes, const TensorOptions& options);

// Retrieve `ideep::tensor` from MKL-DNN tensor
ideep::tensor& itensor_from_mkldnn(const Tensor& mkldnn_tensor);

}}

#endif // AT_MKLDNN_ENABLED
