#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/VolumetricAveragePooling.cu"
#else

#include <THCUNN/generic/pooling_shape.h>

static inline void THNN_(VolumetricAveragePooling_shapeCheck)(
                         THCState *state,
                         THCTensor *input,
                         THCTensor *gradOutput,
                         int kT, int kW, int kH,
                         int dT, int dW, int dH,
                         int padT, int padW, int padH,
                         bool ceil_mode)
{
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;

  int ndim = input->dim();
  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->dim() == 5)
  {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  if (!input->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, input) == 4)
  {
    THArgCheck(input->size(dimw) >= kW && input->size(dimh) >= kH
               && input->size(dimt) >= kT, 2,
               "input image (T: %d H: %d W: %d) smaller than "
               "kernel size (kT: %d kH: %d kW: %d)",
               input->size(dimt), input->size(dimh), input->size(dimw),
               kT, kH, kW);

    /* sizes */
    inputSlices = THCTensor_(size)(state, input, 0);
    inputTime   = THCTensor_(size)(state, input, 1);
    inputHeight = THCTensor_(size)(state, input, 2);
    inputWidth  = THCTensor_(size)(state, input, 3);
  }
  else if (!input->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, input) == 5)
  {
    THArgCheck(input->size(dimw) >= kW && input->size(dimh) >= kH
               && input->size(dimt) >= kT, 2,
               "input image (T: %d H: %d W: %d) smaller than "
               "kernel size (kT: %d kH: %d kW: %d)",
               input->size(dimt), input->size(dimh), input->size(dimw),
               kT, kH, kW);

    /* sizes */
    inputSlices = THCTensor_(size)(state, input, 1);
    inputTime   = THCTensor_(size)(state, input, 2);
    inputHeight = THCTensor_(size)(state, input, 3);
    inputWidth  = THCTensor_(size)(state, input, 4);
  }
  else
  {
    AT_ERROR("non-empty 4D or 5D tensor expected, but got size: ", input->sizes());
  }

  // The second argument is the index of padH.
  THArgCheck(kT/2 >= padT && kW/2 >= padW && kH/2 >= padH, 11,
             "pad should not be greater than half of kernel size, but got "
             "padT = %d, padW = %d, padH = %d, kT = %d, kW = %d, kH = %d",
             padT, padW, padH, kT, kW, kH);

  int outputTime = pooling_output_shape<int>(inputTime, kT, padT, dT, 1, ceil_mode);
  int outputHeight = pooling_output_shape<int>(inputHeight, kH, padH, dH, 1, ceil_mode);
  int outputWidth = pooling_output_shape<int>(inputWidth, kW, padW, dW, 1, ceil_mode);

  if (gradOutput != NULL)
  {
     THCUNN_check_dim_size(state, gradOutput, ndim, dimN, inputSlices);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimt, outputTime);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimh, outputHeight);
     THCUNN_check_dim_size(state, gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(VolumetricAveragePooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int kT, int kW, int kH,
           int dT, int dW, int dH,
           int padT, int padW, int padH,
           bool ceil_mode,
           bool count_include_pad)
{
  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;

  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  int fiveDimensionalInput = THCTensor_(nDimensionLegacyNoScalars)(state, input) == 5;
  if (fiveDimensionalInput)
  {
    dimt++;
    dimh++;
    dimw++;
  }

  THNN_(VolumetricAveragePooling_shapeCheck)
       (state, input, NULL, kT, kW, kH, dT, dW, dH,
        padT, padW, padH, ceil_mode);

  if (!fiveDimensionalInput) /* 4D */
  {
    /* sizes */
    batchSize   = 1;
    inputSlices = THCTensor_(size)(state, input, 0);
    inputTime   = THCTensor_(size)(state, input, 1);
    inputHeight = THCTensor_(size)(state, input, 2);
    inputWidth  = THCTensor_(size)(state, input, 3);
  }
  else /* 5D */
  {
    /* sizes */
    batchSize   = THCTensor_(size)(state, input, 0);
    inputSlices = THCTensor_(size)(state, input, 1);
    inputTime   = THCTensor_(size)(state, input, 2);
    inputHeight = THCTensor_(size)(state, input, 3);
    inputWidth  = THCTensor_(size)(state, input, 4);
  }

  int outputTime = pooling_output_shape<int>(inputTime, kT, padT, dT, 1, ceil_mode);
  int outputHeight = pooling_output_shape<int>(inputHeight, kH, padH, dH, 1, ceil_mode);
  int outputWidth = pooling_output_shape<int>(inputWidth, kW, padW, dW, 1, ceil_mode);

  if (!fiveDimensionalInput) /* 4D */
  {
    /* resize output */
    THCTensor_(resize4d)(state, output, inputSlices,
                         outputTime, outputHeight, outputWidth);
  }
  else /* 5D */
  {
    THCTensor_(resize5d)(state, output, batchSize, inputSlices,
                         outputTime, outputHeight, outputWidth);
  }

  input = THCTensor_(newContiguous)(state, input);
  if (fiveDimensionalInput) {
    // Collapse batch and feature dimensions
    output = THCTensor_(newFoldBatchDim)(state, output);

    THCTensor *old_input = input;
    input = THCTensor_(newFoldBatchDim)(state, input);
    THCTensor_(free)(state, old_input);
  } else {
    THCTensor_(retain)(state, output);
  }

  THCDeviceTensor<scalar_t, 4> cudaInput;
  THCDeviceTensor<scalar_t, 4> cudaOutput;
  cudaInput  = toDeviceTensor<scalar_t, 4>(state, input);
  cudaOutput = toDeviceTensor<scalar_t, 4>(state, output);

  int totalZ = outputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);
  while (totalZ > 0) {
    dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
              THCCeilDiv(outputHeight, static_cast<int>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);

    switch (kW)
      {
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(1);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(2);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(3);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(4);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(5);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(6);
        LAUNCH_UPDATE_OUTPUT_KERNEL_WIDTH(7);
      default:
        cuda_VolumetricAveragePooling_updateOutput<scalar_t, accreal>
          <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
            cudaInput,
            cudaOutput,
            kT, kH, kW,
            dT, dH, dW,
            padT, padH, padW,
            count_include_pad,
            offsetZ);
        break;
      }
    totalZ -= 65535;
    offsetZ += 65535;
    THCudaCheck(cudaGetLastError());
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, output);
}

void THNN_(VolumetricAveragePooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int kT, int kW, int kH,
           int dT, int dW, int dH,
           int padT, int padW, int padH,
           bool ceil_mode,
           bool count_include_pad)
{
  THNN_(VolumetricAveragePooling_shapeCheck)
       (state, input, gradOutput, kT, kW, kH, dT, dW, dH,
        padT, padW, padH, ceil_mode);
  bool kernelsOverlap = (dT < kT) || (dH < kH) || (dW < kW);

  // Resize and initialize result tensor.
  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);

  int batchSize;
  int inputSlices;
  int inputTime;
  int inputHeight;
  int inputWidth;

  int outputTime;
  int outputHeight;
  int outputWidth;

  int fiveDimensionalInput = THCTensor_(nDimensionLegacyNoScalars)(state, input) == 5;
  if (!fiveDimensionalInput) /* 4D */
  {
    batchSize = 1;
    inputSlices  = THCTensor_(size)(state, input, 0);
    inputTime    = THCTensor_(size)(state, input, 1);
    inputHeight  = THCTensor_(size)(state, input, 2);
    inputWidth   = THCTensor_(size)(state, input, 3);

    outputTime   = THCTensor_(size)(state, gradOutput, 1);
    outputHeight = THCTensor_(size)(state, gradOutput, 2);
    outputWidth  = THCTensor_(size)(state, gradOutput, 3);
  }
  else
  {
    batchSize    = THCTensor_(size)(state, input, 0);
    inputSlices  = THCTensor_(size)(state, input, 1);
    inputTime    = THCTensor_(size)(state, input, 2);
    inputHeight  = THCTensor_(size)(state, input, 3);
    inputWidth   = THCTensor_(size)(state, input, 4);

    outputTime   = THCTensor_(size)(state, gradOutput, 2);
    outputHeight = THCTensor_(size)(state, gradOutput, 3);
    outputWidth  = THCTensor_(size)(state, gradOutput, 4);
  }

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  if (fiveDimensionalInput) {
    // Collapse batch and feature dimensions
    gradInput = THCTensor_(newFoldBatchDim)(state, gradInput);

    THCTensor *old_gradOutput = gradOutput;
    gradOutput = THCTensor_(newFoldBatchDim)(state, gradOutput);
    THCTensor_(free)(state, old_gradOutput);
  } else {
    THCTensor_(retain)(state, gradInput);
  }

  THCDeviceTensor<scalar_t, 4> cudaGradInput;
  THCDeviceTensor<scalar_t, 4> cudaGradOutput;
  cudaGradInput  = toDeviceTensor<scalar_t, 4>(state, gradInput);
  cudaGradOutput = toDeviceTensor<scalar_t, 4>(state, gradOutput);

  dim3 block(32, 8);

  // Optimizing for stride 1 is probably only of limited value, but this
  // specialization yields 3x speedup over the atomicAdd implementation.
  // Padding must be 0, otherwise, pool size may change.
  if (dT == 1 && dH == 1 && dW == 1 && padT == 0 && padH == 0 && padW == 0)
  {
    int totalZ = inputTime * inputSlices * batchSize;
    int offsetZ = 0;
    while (totalZ > 0) {
      dim3 grid(THCCeilDiv(inputWidth, static_cast<int>(block.x)),
                THCCeilDiv(inputHeight, static_cast<int>(block.y)),
                totalZ > 65535 ? 65535 : totalZ);
      cuda_VolumetricAveragePooling_updateGradInput_Stride1<scalar_t, accreal>
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          cudaGradOutput, cudaGradInput, kT, kH, kW, 1.0f/(kT * kH * kW), offsetZ);
      THCudaCheck(cudaGetLastError());
      totalZ -= 65535;
      offsetZ += 65535;
    }
  }
  else
  {
    int totalZ = outputTime * inputSlices * batchSize;
    int offsetZ = 0;
    while (totalZ > 0) {
      dim3 grid(THCCeilDiv(outputWidth, static_cast<int>(block.x)),
                THCCeilDiv(outputHeight, static_cast<int>(block.y)),
                totalZ > 65535 ? 65535 : totalZ);
      if (kernelsOverlap)
      {
        cuda_VolumetricAveragePooling_updateGradInput_atomicAdd<scalar_t, accreal>
          <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
            cudaGradOutput, cudaGradInput, kT, kH, kW, dT, dH, dW,
            padT, padH, padW, count_include_pad, offsetZ);
      }
      else
      {
        cuda_VolumetricAveragePooling_updateGradInput<scalar_t, accreal>
          <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
            cudaGradOutput, cudaGradInput, kT, kH, kW, dT, dH, dW,
            padT, padH, padW, count_include_pad, offsetZ);
      }
      THCudaCheck(cudaGetLastError());
      totalZ -= 65535;
      offsetZ += 65535;
    }
  }

  THCTensor_(free)(state, gradInput);
  THCTensor_(free)(state, gradOutput);
}

#endif
