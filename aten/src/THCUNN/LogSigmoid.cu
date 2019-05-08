#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCApply.cuh>

#if defined(_MSC_VER) || defined(__HIP_PLATFORM_HCC__)
#define ZERO_MACRO zero<T>()
template <typename T>
inline __device__ typename std::enable_if<std::is_same<T, double>::value, T>::type zero() {
        return 0.;
}

template <typename T>
inline __device__ typename std::enable_if<!std::is_same<T, double>::value, T>::type zero() {
        return 0.f;
}
#else
#define ZERO_MACRO 0.f
#endif

template <typename T>
struct logSigmoid_updateOutput_functor
{
  __device__ void operator()(T *output, const T *input) const {
    const T max = fmaxType(ZERO_MACRO, -*input);
    const T z = THCNumerics<T>::exp(-max) + THCNumerics<T>::exp(-*input -max);
    *output = -(max + THCNumerics<T>::log(z));
  }
};


template <typename T>
struct logSigmoid_updateGradInput_functor
{
  __device__ void operator()(T *gradInput, const T *input, const T *gradOutput) const {
    const T max = fmaxType(ZERO_MACRO, -*input);
    const T z = THCNumerics<T>::exp(-max) + THCNumerics<T>::exp(-*input -max);
    T max_deriv = 0.f;
    T sign = -1.f;
    if (*input < 0.f){
        max_deriv = -1.f;
        sign = 1.f;
    }
    *gradInput = *gradOutput * (-max_deriv - sign*((z - 1.f)/z));
  }
};

template <>
struct logSigmoid_updateOutput_functor<half> {
  __device__ __forceinline__ void operator()(half* output, const half *input) const {
    float in = __half2float(*input);
    float max = fmaxType(0.f, -in);
    float z = THCNumerics<float>::exp(-max) + THCNumerics<float>::exp(-in - max);
    *output = __float2half(-(max + THCNumerics<float>::log(z)));
  }
};

template <>
struct logSigmoid_updateGradInput_functor<half> {
  __device__ __forceinline__ void operator()(half* gradInput, const half *input, const half *gradOutput) const {
    const float in = __half2float(*input);
    const float max = fmaxType(0.f, -in);
    const float z = THCNumerics<float>::exp(-max) + THCNumerics<float>::exp(-in - max);
    const float go = __half2float(*gradOutput);
    float max_deriv = 0.f;
    float sign = -1.f;
    if(in < 0.f){
        max_deriv = -1.f;
        sign = 1.f;
    }
    *gradInput = __float2half(go * (-max_deriv - sign*((z - 1.f)/z)));
  }
};

#include <THCUNN/generic/LogSigmoid.cu>
#include <THC/THCGenerateFloatTypes.h>
