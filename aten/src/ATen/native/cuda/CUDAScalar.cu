#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>

namespace at {
namespace native {

Scalar _local_scalar_dense_cuda(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(), "_local_scalar_dense_cuda", [&] {
        scalar_t value;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(cudaMemcpyAsync(&value, self.data<scalar_t>(), sizeof(scalar_t), cudaMemcpyDeviceToHost, stream));
        AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        r = Scalar(value);
      });
  return r;
}

}} // at::native
