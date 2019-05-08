#pragma once
// Please note that this file is
// used across both CPU and GPU.

#include <c10/macros/Macros.h>
#if defined(__CUDACC__)
#include <THC/THCDeviceUtils.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#elif defined(__HIPCC__)
#include <THH/THHDeviceUtils.cuh>
#include <ATen/native/hip/DeviceSqrt.cuh>
#else
#include <cmath>
#define device_sqrt std::sqrt
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
#define MAX(X, Y) ::max(X,Y)
#define MIN(X, Y) ::min(X,Y)
#else
#define MAX(X, Y) std::max(X,Y)
#define MIN(X, Y) std::min(X,Y)
#endif

// ROCM hcc doesn't work well with using std:: in kernel functions
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
#include <c10/cuda/CUDAMathCompat.h>
#define compat_pow c10::cuda::compat::pow
#else
#define compat_pow std::pow
#endif

namespace at { namespace native {

template <typename scalar_t, typename index_t, typename combine_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  combine_t nf;
  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0)  {}
  C10_DEVICE WelfordData(scalar_t mean, scalar_t m2, index_t n, combine_t nf) : mean(mean), m2(m2), n(n), nf(nf) {}
};


template <typename scalar_t, typename acc_scalar_t, typename index_t, typename combine_t>
struct WelfordOps {
  bool unbiased;
  bool take_sqrt;
 public:
  using acc_t = WelfordData<acc_scalar_t, index_t, combine_t>;
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data) const {
    acc_scalar_t delta = data - acc.mean;
    // using acc.nf(combine_t) here, as acc.n(index_t) would still be converted
    // accumulation in reduce is done through index_T
    acc_scalar_t new_mean = acc.mean + delta / (acc.nf + 1);
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      acc.n + 1,
      combine_t(acc.n + 1), // accumulate for combine_t uses index_t
    };
  }
  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    combine_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
      // setting acc.n as -1 since acc.n might not be able to represent the count
      // correctly within its range, setting it to -1 to avoid confusion
      -1,
      new_count
    };
  }
  inline C10_DEVICE scalar_t project(acc_t acc) const {
    combine_t divisor = unbiased ? (acc.nf - 1) : acc.nf;
    auto ret = (divisor > 0) ?
      (take_sqrt ? device_sqrt(acc.m2 / divisor) : (acc.m2 / divisor))
      : NAN;
    return (scalar_t) ret;
  }
#if defined(__CUDACC__) || defined(__HIPCC__)
  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.mean, offset)
      , WARP_SHFL_DOWN(acc.m2, offset)
      , WARP_SHFL_DOWN(acc.n, offset)
      , WARP_SHFL_DOWN(acc.nf, offset)
    };
  }
#endif
  WelfordOps(bool unbiased, bool take_sqrt)
    : unbiased(unbiased), take_sqrt(take_sqrt) {
  }
};

template <typename acc_t, typename factor_t>
struct MeanOps {
  factor_t factor;

  inline C10_DEVICE acc_t reduce(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return reduce(a, b);
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a * factor;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif

  MeanOps(factor_t factor): factor(factor) {
  }
};

template <typename acc_t>
struct AbsMinOps {

  inline C10_DEVICE acc_t reduce(acc_t acc, acc_t data) const {
    return MIN(acc, std::abs(data));
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return MIN(a, b);
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif
};

template <typename acc_t>
struct AbsMaxOps {

  inline C10_DEVICE acc_t reduce(acc_t acc, acc_t data) const {
    return MAX(acc, std::abs(data));
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return MAX(a, b);
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif
};

template <typename acc_t>
struct NormOps {
  acc_t norm;

  inline C10_DEVICE acc_t reduce(acc_t acc, acc_t data) const {
    return acc + compat_pow(std::abs(data), norm);
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return compat_pow(a, acc_t(1.0)/norm);
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif

  NormOps(acc_t norm): norm(norm) {
  }
};

template <typename acc_t>
struct NormZeroOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, acc_t data) const {
    return acc + (data==acc_t(0) ? acc_t(0) : acc_t(1));
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif
};

template <typename acc_t>
struct NormOneOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, acc_t data) const {
    return acc + std::abs(data);
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline C10_DEVICE acc_t project(acc_t a) const {
    return a;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_DEVICE acc_t warp_shfl_down(acc_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
#endif
};

}} // namespace at::native

#undef MAX
#undef MIN
