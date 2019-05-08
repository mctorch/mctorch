#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCApply.cuh>

template <typename T>
struct squareupdateOutput_functor
{
  __device__ void operator()(T* output, const T* input) const
  {
    *output = (*input) * (*input);
  }
};

template <typename T>
struct squareupdateGradInput_functor
{
  __device__ void operator()(T* gradInput, const T* input, const T* gradOutput) const
  {
    *gradInput = ScalarConvert<double, T>::to(2.0) * (*gradOutput) * (*input);
  }
};

#include <THCUNN/generic/Square.cu>
#include <THC/THCGenerateFloatTypes.h>
