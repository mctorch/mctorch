#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/Sqrt.cu"
#else

#include <THCUNN/common.h>

void THNN_(Sqrt_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal eps_)
{
  scalar_t eps = ScalarConvert<accreal, scalar_t>::to(eps_);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resizeAs)(state, output, input);
  THC_pointwiseApply2<scalar_t, scalar_t>(state, output, input, sqrtupdateOutput_functor<scalar_t>(eps));
}

void THNN_(Sqrt_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output)
{
  THCUNN_check_shape(state, output, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);
  THCTensor_(resizeAs)(state, gradInput, output);
  THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, gradInput, output, gradOutput, sqrtupdateGradInput_functor<scalar_t>());
}

#endif
