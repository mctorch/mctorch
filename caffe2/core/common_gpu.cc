#include "caffe2/core/common_gpu.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "caffe2/core/asan.h"
#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

int NumCudaDevices() {
  if (getenv("CAFFE2_DEBUG_CUDA_INIT_ORDER")) {
    static bool first = true;
    if (first) {
      first = false;
      std::cerr << "DEBUG: caffe2::NumCudaDevices() invoked for the first time"
                << std::endl;
    }
  }
  static int count = -1;
  if (count < 0) {
    auto err = cudaGetDeviceCount(&count);
    switch (err) {
      case cudaSuccess:
        // Everything is good.
        break;
      case cudaErrorNoDevice:
        count = 0;
        break;
      case cudaErrorInsufficientDriver:
        LOG(WARNING) << "Insufficient cuda driver. Cannot use cuda.";
        count = 0;
        break;
      case cudaErrorInitializationError:
        LOG(WARNING) << "Cuda driver initialization failed, you might not "
                        "have a cuda gpu.";
        count = 0;
        break;
      case cudaErrorUnknown:
        LOG(ERROR) << "Found an unknown error - this may be due to an "
                      "incorrectly set up environment, e.g. changing env "
                      "variable CUDA_VISIBLE_DEVICES after program start. "
                      "I will set the available devices to be zero.";
        count = 0;
        break;
      case cudaErrorMemoryAllocation:
#if CAFFE2_ASAN_ENABLED
        // In ASAN mode, we know that a cudaErrorMemoryAllocation error will
        // pop up.
        LOG(ERROR) << "It is known that CUDA does not work well with ASAN. As "
                      "a result we will simply shut down CUDA support. If you "
                      "would like to use GPUs, turn off ASAN.";
        count = 0;
        break;
#else // CAFFE2_ASAN_ENABLED
        // If we are not in ASAN mode and we get cudaErrorMemoryAllocation,
        // this means that something is wrong before NumCudaDevices() call.
        LOG(FATAL) << "Unexpected error from cudaGetDeviceCount(). Did you run "
                      "some cuda functions before calling NumCudaDevices() "
                      "that might have already set an error? Error: "
                   << err;
        break;
#endif // CAFFE2_ASAN_ENABLED
      default:
        LOG(FATAL) << "Unexpected error from cudaGetDeviceCount(). Did you run "
                      "some cuda functions before calling NumCudaDevices() "
                      "that might have already set an error? Error: "
                   << err;
    }
  }
  return count;
}

namespace {
int gDefaultGPUID = 0;
}  // namespace

void SetDefaultGPUID(const int deviceid) {
  CAFFE_ENFORCE_LT(
      deviceid,
      NumCudaDevices(),
      "The default gpu id should be smaller than the number of gpus "
      "on this machine: ",
      deviceid,
      " vs ",
      NumCudaDevices());
  gDefaultGPUID = deviceid;
}

int GetDefaultGPUID() { return gDefaultGPUID; }

int CaffeCudaGetDevice() {
  int gpu_id = 0;
  CUDA_ENFORCE(cudaGetDevice(&gpu_id));
  return gpu_id;
}

void CaffeCudaSetDevice(const int id) {
  CUDA_ENFORCE(cudaSetDevice(id));
}

int GetGPUIDForPointer(const void* ptr) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

  if (err == cudaErrorInvalidValue) {
    // Occurs when the pointer is in the CPU address space that is
    // unmanaged by CUDA; make sure the last error state is cleared,
    // since it is persistent
    err = cudaGetLastError();
    CHECK(err == cudaErrorInvalidValue);
    return -1;
  }

  // Otherwise, there must be no error
  CUDA_ENFORCE(err);

  if (attr.CAFFE2_CUDA_PTRATTR_MEMTYPE == cudaMemoryTypeHost) {
    return -1;
  }

  return attr.device;
}

struct CudaDevicePropWrapper {
  CudaDevicePropWrapper() : props(NumCudaDevices()) {
    for (int i = 0; i < NumCudaDevices(); ++i) {
      CUDA_ENFORCE(cudaGetDeviceProperties(&props[i], i));
    }
  }

  vector<cudaDeviceProp> props;
};

const cudaDeviceProp& GetDeviceProperty(const int deviceid) {
  // According to C++11 standard section 6.7, static local variable init is
  // thread safe. See
  //   https://stackoverflow.com/questions/8102125/is-local-static-variable-initialization-thread-safe-in-c11
  // for details.
  static CudaDevicePropWrapper props;
  CAFFE_ENFORCE_LT(
      deviceid,
      NumCudaDevices(),
      "The gpu id should be smaller than the number of gpus ",
      "on this machine: ",
      deviceid,
      " vs ",
      NumCudaDevices());
  return props.props[deviceid];
}

void DeviceQuery(const int device) {
  const cudaDeviceProp& prop = GetDeviceProperty(device);
  std::stringstream ss;
  ss << std::endl;
  ss << "Device id:                     " << device << std::endl;
  ss << "Major revision number:         " << prop.major << std::endl;
  ss << "Minor revision number:         " << prop.minor << std::endl;
  ss << "Name:                          " << prop.name << std::endl;
  ss << "Total global memory:           " << prop.totalGlobalMem << std::endl;
  ss << "Total shared memory per block: " << prop.sharedMemPerBlock
     << std::endl;
  ss << "Total registers per block:     " << prop.regsPerBlock << std::endl;
  ss << "Warp size:                     " << prop.warpSize << std::endl;
#ifndef __HIP_PLATFORM_HCC__
  ss << "Maximum memory pitch:          " << prop.memPitch << std::endl;
#endif
  ss << "Maximum threads per block:     " << prop.maxThreadsPerBlock
     << std::endl;
  ss << "Maximum dimension of block:    "
     << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
     << prop.maxThreadsDim[2] << std::endl;
  ss << "Maximum dimension of grid:     "
     << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
     << prop.maxGridSize[2] << std::endl;
  ss << "Clock rate:                    " << prop.clockRate << std::endl;
  ss << "Total constant memory:         " << prop.totalConstMem << std::endl;
#ifndef __HIP_PLATFORM_HCC__
  ss << "Texture alignment:             " << prop.textureAlignment << std::endl;
  ss << "Concurrent copy and execution: "
     << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
#endif
  ss << "Number of multiprocessors:     " << prop.multiProcessorCount
     << std::endl;
#ifndef __HIP_PLATFORM_HCC__
  ss << "Kernel execution timeout:      "
     << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
#endif
  LOG(INFO) << ss.str();
  return;
}

bool GetCudaPeerAccessPattern(vector<vector<bool> >* pattern) {
  int gpu_count;
  if (cudaGetDeviceCount(&gpu_count) != cudaSuccess) return false;
  pattern->clear();
  pattern->resize(gpu_count, vector<bool>(gpu_count, false));
  for (int i = 0; i < gpu_count; ++i) {
    for (int j = 0; j < gpu_count; ++j) {
      int can_access = true;
      if (i != j) {
        if (cudaDeviceCanAccessPeer(&can_access, i, j)
                 != cudaSuccess) {
          return false;
        }
      }
      (*pattern)[i][j] = static_cast<bool>(can_access);
    }
  }
  return true;
}

bool TensorCoreAvailable() {
  // requires CUDA 9.0 and above
#if CUDA_VERSION < 9000
  return false;
#else
  int device = CaffeCudaGetDevice();
  auto& prop = GetDeviceProperty(device);

  return prop.major >= 7;
#endif
}

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
#ifndef __HIP_PLATFORM_HCC__
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
#endif
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif  // CUDA_VERSION >= 6050
#endif  // CUDA_VERSION >= 6000
#ifdef __HIP_PLATFORM_HCC__
  case rocblas_status_invalid_size:
    return "rocblas_status_invalid_size";
#endif
  }
  // To suppress compiler warning.
  return "Unrecognized cublas error string";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
#ifdef __HIP_PLATFORM_HCC__
  case HIPRAND_STATUS_NOT_IMPLEMENTED:
    return "HIPRAND_STATUS_NOT_IMPLEMENTED";
#endif
  }
  // To suppress compiler warning.
  return "Unrecognized curand error string";
}

// Turn on the flag g_caffe2_has_cuda_linked to true for HasCudaRuntime()
// function.
namespace {
class CudaRuntimeFlagFlipper {
 public:
  CudaRuntimeFlagFlipper() {
    internal::SetCudaRuntimeFlag();
  }
};
static CudaRuntimeFlagFlipper g_flipper;
} // namespace

}  // namespace caffe2
