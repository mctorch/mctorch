#ifndef THC_TENSORMATH_COMPARE_CUH
#define THC_TENSORMATH_COMPARE_CUH

#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>

template <typename T, typename TOut>
struct TensorLTValueOp {
  TensorLTValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::lt(*in, value));
  }

  const T value;
};

template <typename T, typename TOut>
struct TensorGTValueOp {
  TensorGTValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::gt(*in, value));
  }

  const T value;
};


template <typename T, typename TOut>
struct TensorLEValueOp {
  TensorLEValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::le(*in, value));
  }

  const T value;
};

template <typename T, typename TOut>
struct TensorGEValueOp {
  TensorGEValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ge(*in, value));
  }

  const T value;
};

template <typename T, typename TOut>
struct TensorEQValueOp {
  TensorEQValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::eq(*in, value));
  }

  const T value;
};

template <typename T, typename TOut>
struct TensorNEValueOp {
  TensorNEValueOp(T v) : value(v) {}
  __device__ __forceinline__ void operator()(TOut* out, T* in) {
    *out = ScalarConvert<bool, TOut>::to(THCNumerics<T>::ne(*in, value));
  }

  const T value;
};

template<typename ScalarTypeOut, typename ScalarType, typename TensorTypeOut, typename TensorType, class Op>
void THC_logicalValue(THCState *state,
                      TensorTypeOut *self_,
                      TensorType *src,
                      Op op) {
  THCTensor_resize(state, self_, src->sizes(), {});

  if (!THC_pointwiseApply2<ScalarTypeOut, ScalarType>(state, self_, src, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

#endif // THC_TENSORMATH_COMPARE_CUH
