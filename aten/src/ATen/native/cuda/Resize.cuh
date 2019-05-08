#pragma once

#include <ATen/ATen.h>
#include <THC/THCTensor.hpp>

#include <c10/cuda/CUDAGuard.h>

namespace at { namespace native {

// These functions are called by native::resize_ as well as (legacy) THC resize.
// They are not in THC/THCTensor.cpp because the at namespace is easier
// to benchmark than THC; I can't get gbenchmark to call fns from THTensor.cpp

static inline void maybe_resize_storage_cuda(TensorImpl* self, int64_t new_size) {
  if (new_size + self->storage_offset() > 0) {
    if (!THTensor_getStoragePtr(self)) {
      AT_ERROR("Tensor: invalid null storage");
    }
    if (new_size + self->storage_offset() > self->storage().numel()) {
      THCStorage_resize(
          globalContext().getTHCState(),
          THTensor_getStoragePtr(self),
          new_size + self->storage_offset());
    }
  }
}

inline TensorImpl* resize_impl_cuda_(
    TensorImpl* self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // NB: We don't need to hold the device guard when calling from TH
  cuda::OptionalCUDAGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    for (size_t dim = 0; dim < size.size(); ++dim) {
      // FIXME: Don't rely on storage_size being negative because this
      // may not be true for some edge cases.
      if (size[dim] == 0) {
        storage_size = 0;
        break;
      }
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }
  maybe_resize_storage_cuda(self, storage_size);

  return self;
}

}}
