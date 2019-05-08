#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCApply.cuh>

template <typename T>
struct sqrtupdateOutput_functor
{
  const T bias;

  sqrtupdateOutput_functor(T bias_)
    : bias(bias_)
  {}

  __device__ void operator()(T *output, const T *input) const
  {
    *output = sqrt(*input + bias);
  }
};

template <typename T>
struct sqrtupdateGradInput_functor
{
  sqrtupdateGradInput_functor() {}

  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = (THCNumerics<T>::eq(*output,ScalarConvert<float, T>::to(0.0f))) ? ScalarConvert<float, T>::to(0.0f) : ((ScalarConvert<float, T>::to(0.5f) * *gradOutput) / *output);
  }
};

#include <THCUNN/generic/Sqrt.cu>
#include <THC/THCGenerateFloatTypes.h>
