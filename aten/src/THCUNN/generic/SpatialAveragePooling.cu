#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/SpatialAveragePooling.cu"
#else

#include <THCUNN/common.h>
#include <THCUNN/generic/pooling_shape.h>

static inline void THNN_(SpatialAveragePooling_shapeCheck)(
  THCState *state,
  THCTensor *input, THCTensor *gradOutput,
  int kH, int kW, int dH, int dW, int padH, int padW, bool ceil_mode) {

  THArgCheck(kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = input->dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THCUNN_argCheck(state, !input->is_empty() && (ndim == 3 || ndim == 4), 2, input,
                  "non-empty 3D or 4D input tensor expected but got: %s");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2,
             "pad should be smaller than half of kernel size, but got "
             "padW = %d, padH = %d, kW = %d, kH = %d",
             padW, padH, kW, kH);

  int64_t nInputPlane = input->size(dimh-1);
  int64_t nInputRows = input->size(dimh);
  int64_t nInputCols = input->size(dimw);
  int64_t nOutputPlane = nInputPlane;

  int64_t nOutputCols = pooling_output_shape<int64_t>(nInputCols, kW, padW, dW, 1, ceil_mode);
  int64_t nOutputRows = pooling_output_shape<int64_t>(nInputRows, kH, padH, dH, 1, ceil_mode);

  if (nOutputCols < 1 || nOutputRows < 1)
    THError("Given input size: (%dx%dx%d). "
            "Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,nInputRows,nInputCols,nInputPlane,nOutputRows,nOutputCols);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim, dimf, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimh, nOutputRows);
    THCUNN_check_dim_size(state, gradOutput, ndim, dimw, nOutputCols);
  }
}

void THNN_(SpatialAveragePooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           bool ceil_mode,
           bool count_include_pad)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THNN_(SpatialAveragePooling_shapeCheck)
       (state, input, NULL, kH, kW, dH, dW,
        padH, padW, ceil_mode);

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int64_t nOutputCols, nOutputRows;

  if (input->dim() == 3) {
    nInputCols = input->size(2);
    nInputRows = input->size(1);
    nInputPlane = input->size(0);
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size(3);
    nInputRows = input->size(2);
    nInputPlane = input->size(1);
    batchSize = input->size(0);
  }

  nOutputCols = pooling_output_shape<int64_t>(nInputCols, kW, padW, dW, 1, ceil_mode);
  nOutputRows = pooling_output_shape<int64_t>(nInputRows, kH, padH, dH, 1, ceil_mode);

  input = THCTensor_(newContiguous)(state, input);
  scalar_t* input_data = THCTensor_(data)(state, input);

  THCTensor_(resize4d)(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);

  scalar_t* output_data = THCTensor_(data)(state, output);

  int count = THCTensor_(nElement)(state, output);

  if(count_include_pad)
    AvePoolForward<scalar_t, accreal, true>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
        count, input_data,
        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW, output_data);
  else
    AvePoolForward<scalar_t, accreal, false>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
        count, input_data,
        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW, output_data);
  THCudaCheck(cudaGetLastError());

  if(input->dim() == 3)
    THCTensor_(resize3d)(state, output, nInputPlane, nOutputRows, nOutputCols);

  THCTensor_(free)(state, input);

}

void THNN_(SpatialAveragePooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           bool ceil_mode,
           bool count_include_pad)
{
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);
  THNN_(SpatialAveragePooling_shapeCheck)
       (state, input, gradOutput, kH, kW, dH, dW,
        padH, padW, ceil_mode);

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int64_t nOutputCols, nOutputRows;
  int dimCol = 2;
  int dimRow = 1;

  if (input->dim() == 3) {
    nInputPlane = input->size(0);
    batchSize = 1;
  }
  else
  {
    dimCol = 3;
    dimRow = 2;
    nInputPlane = input->size(1);
    batchSize = input->size(0);
  }
  nInputCols = input->size(dimCol);
  nInputRows = input->size(dimRow);

  nOutputCols = pooling_output_shape<int64_t>(nInputCols, kW, padW, dW, 1, ceil_mode);
  nOutputRows = pooling_output_shape<int64_t>(nInputRows, kH, padH, dH, 1, ceil_mode);

  THCUNN_check_dim_size(state, gradOutput, input->dim(), dimRow, nOutputRows);
  THCUNN_check_dim_size(state, gradOutput, input->dim(), dimCol, nOutputCols);
  THCTensor_(resizeAs)(state, gradInput, input);

  int count = THCTensor_(nElement)(state, input);

  if(count_include_pad)
    AvePoolBackward<scalar_t, accreal, true>
      <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count,
        THCTensor_(data)(state, gradOutput),
        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW,
        THCTensor_(data)(state, gradInput));
  else
    AvePoolBackward<scalar_t, accreal, false>
      <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count,
        THCTensor_(data)(state, gradOutput),
        batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
        kH, kW, dH, dW, padH, padW,
        THCTensor_(data)(state, gradInput));
  THCudaCheck(cudaGetLastError());

  // clean
  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
