#include <torch/csrc/autograd/profiler.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvToolsExt.h>

#include <sstream>

namespace torch { namespace autograd { namespace profiler {

namespace {

static inline void cudaCheck(cudaError_t result, const char * file, int line) {
  if(result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << cudaGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_CUDA_CHECK(result) cudaCheck(result,__FILE__,__LINE__);

struct CUDAMethods : public CUDAStubs {
  void record(int* device, CUDAEventStub* event, int64_t* cpu_ns) override {
    TORCH_CUDA_CHECK(cudaGetDevice(device));
    TORCH_CUDA_CHECK(cudaEventCreate(event));
    auto stream = at::cuda::getCurrentCUDAStream();
    *cpu_ns = getTime();
    TORCH_CUDA_CHECK(cudaEventRecord(*event, stream));
  }
  float elapsed(CUDAEventStub event, CUDAEventStub event2) override {
    TORCH_CUDA_CHECK(cudaEventSynchronize(event));
    TORCH_CUDA_CHECK(cudaEventSynchronize(event2));
    float ms;
    TORCH_CUDA_CHECK(cudaEventElapsedTime(&ms, event, event2));
    return ms*1000.0;
  }
  void nvtxMarkA(const char* name) override {
    ::nvtxMark(name);
  }
  void nvtxRangePushA(const char* name) override {
    ::nvtxRangePushA(name);
  }
  void nvtxRangePop() override {
    ::nvtxRangePop();
  }
  void onEachDevice(std::function<void(int)> op) override {
    at::cuda::OptionalCUDAGuard device_guard;
    int count = at::cuda::device_count();
    for(int i = 0; i < count; i++) {
      device_guard.set_index(i);
      op(i);
    }
  }
  void synchronize() override {
    cudaDeviceSynchronize();
  }
  bool enabled() override {
    return true;
  }

};

struct RegisterCUDAMethods {
  RegisterCUDAMethods() {
    static CUDAMethods methods;
    registerCUDAMethods(&methods);
  }
};
RegisterCUDAMethods reg;

} // namespaces
} // namespace profiler
} // namespace autograd
} // namespace torch
