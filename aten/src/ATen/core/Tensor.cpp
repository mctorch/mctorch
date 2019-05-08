#include <ATen/core/Tensor.h>
#include <ATen/core/Formatting.h>
#include <ATen/core/Type.h>

#include <iostream>

namespace at {

void Tensor::enforce_invariants() {
  if (impl_.get() == nullptr) {
    throw std::runtime_error("TensorImpl with nullptr is not supported");
  }
  // Following line throws if the method is not a POD data type or is not
  // supported by ATen
  scalar_type();
  if (defined()) {
    // If it's a variable - we definitely not in C2 land
    if (!is_variable()) {
      AT_ASSERTM(
          impl_->dtype_initialized(),
          "Partially-initialized tensor not supported by at::Tensor");
      AT_ASSERTM(
          !impl_->is_sparse(),
          "Sparse Tensors are supported by at::Tensor, but invariant checking isn't implemented.  Please file a bug.");
      AT_ASSERTM(
          impl_->storage_initialized(),
          "Partially-initialized tensor not supported by at::Tensor");
    }
    // Ensure LegacyTypeDispatch is initialized. In ATen it's done in tensor
    // factory functions, but when we get a tensor from Caffe2 we might bypass
    // those factory functions.
    initializeLegacyTypeDispatchFor(*impl_);
  }
}

void Tensor::print() const {
  if (defined()) {
    std::cerr << "[" << dispatch_type().toString() << " " << sizes() << "]" << std::endl;
  } else {
    std::cerr << "[UndefinedTensor]" << std::endl;
  }
}

const char * Tensor::toString() const {
  return dispatch_type().toString();
}

} // namespace at
