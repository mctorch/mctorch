#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/SpatialClassNLLCriterion.cu"
#else

void THNN_(SpatialClassNLLCriterion_shapeCheck)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *weights)
{
  AT_CHECK(!target->is_empty() && target->dim() == 3, 1,
           "only batches of spatial targets supported (non-empty 3D tensors)" \
           " but got targets of size: : ", target->sizes());
  AT_CHECK(!input->is_empty() && input->dim() == 4, 2,
           "only batches of spatial inputs supported (non-empty 4D tensors), "      \
           "but got input of size: ", input->sizes());
  if (THCTensor_(size)(state, input, 0) != THCIndexTensor_(size)(state, target, 0) ||
      THCTensor_(size)(state, input, 2) != THCIndexTensor_(size)(state, target, 1) ||
      THCTensor_(size)(state, input, 3) != THCIndexTensor_(size)(state, target, 2)) {
    THCDescBuff input_size = THCTensor_(sizeDesc)(state, input);
    THCDescBuff target_size = THCIndexTensor_(sizeDesc)(state, target);
    THError("input and target batch or spatial sizes don't match: target %s, input %s",
            target_size.str, input_size.str);
  }

  if (weights && THCTensor_(nElement)(state, weights) != THCTensor_(size)(state, input, 1)) {
    THError("weight tensor should be defined either for all or no classes");
  }
}

static void THNN_(SpatialClassNLLCriterion_gradOutput_no_reduce_shapeCheck)(
           THCState *state,
           THCTensor *gradOutput,
           THCIndexTensor *target)
{
  AT_CHECK(!gradOutput->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, gradOutput) == 3, 2,
           "Expected non-empty dimension 3 but got gradOutput of size: ", gradOutput->sizes());
  if (THCTensor_(size)(state, gradOutput, 0) != THCIndexTensor_(size)(state, target, 0) ||
      THCTensor_(size)(state, gradOutput, 1) != THCIndexTensor_(size)(state, target, 1) ||
      THCTensor_(size)(state, gradOutput, 2) != THCIndexTensor_(size)(state, target, 2)) {
    THCDescBuff gradOutput_size = THCTensor_(sizeDesc)(state, gradOutput);
    THCDescBuff target_size = THCIndexTensor_(sizeDesc)(state, target);
    THError("gradOutput sizes don't match target sizes: target %s, gradOutput %s",
            target_size.str, gradOutput_size.str);
  }
}

void THNN_(SpatialClassNLLCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *output,
           int64_t reduction,
           THCTensor *weights,
           THCTensor *total_weight,
           int64_t ignore_index)
{
  THNN_(SpatialClassNLLCriterion_shapeCheck)(state, input, target, weights);
  THCTensor_(resize1d)(state, output, 1);
  THCTensor_(resize1d)(state, total_weight, 1);

  if (weights)
    THCUNN_assertSameGPU(state, 5, input, target, weights, output, total_weight);
  else
    THCUNN_assertSameGPU(state, 4, input, target, output, total_weight);

  if (reduction == Reduction::None) {
    int64_t batch_size = THCTensor_(size)(state, input, 0);
    int64_t H = THCTensor_(size)(state, input, 2);
    int64_t W = THCTensor_(size)(state, input, 3);

    THCTensor_(resize3d)(state, output, batch_size, H, W);

    if (weights) {
      weights = THCTensor_(newContiguous)(state, weights);
    }

    int64_t count = batch_size * H * W;
    SpatialClassNLLCriterion_updateOutput_no_reduce_kernel<scalar_t>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        count,
        toDeviceTensor<scalar_t, 4>(state, input),
        toDeviceTensor<THCIndex_t, 3>(state, target),
        toDeviceTensor<scalar_t, 3>(state, output),
        weights ? THCTensor_(data)(state, weights) : NULL,
        ignore_index);

    if (weights) {
      THCTensor_(free)(state, weights);
    }
    return;
  }

  input = THCTensor_(newContiguous)(state, input);
  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  scalar_t *input_data = THCTensor_(data)(state, input);
  scalar_t *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  THCIndex_t  *target_data = THCIndexTensor_(data)(state, target);
  scalar_t *output_data = THCTensor_(data)(state, output);
  scalar_t *total_weight_data = THCTensor_(data)(state, total_weight);

  THCIndex_t batch_size = THCIndexTensor_(size)(state, target, 0);
  THCIndex_t map_nelem = THCIndexTensor_(nElement)(state, target) / batch_size;
  int blocks_per_sample = GET_BLOCKS(map_nelem) / 128;
  blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
  int total_blocks = blocks_per_sample * batch_size;

  THCTensor_(fill)(state, output, ScalarConvert<int, scalar_t>::to(0));
  THCTensor_(fill)(state, total_weight, ScalarConvert<int, scalar_t>::to(0));

  cunn_SpatialClassNLLCriterion_updateOutput_kernel<scalar_t, accreal>
    <<<total_blocks, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      output_data,
      total_weight_data,
      input_data,
      target_data,
      weights_data,
      reduction == Reduction::Mean,
      THCTensor_(size)(state, input, 0),
      THCTensor_(size)(state, input, 1),
      THCTensor_(size)(state, input, 2) * THCTensor_(size)(state, input, 3),
      blocks_per_sample,
      ignore_index
  );
  THCudaCheck(cudaGetLastError());
  if (reduction == Reduction::Mean) {
    cunn_SpatialClassNLLCriterion_sizeAverage_kernel<<<1, 1, 0, THCState_getCurrentStream(state)>>>(
      output_data, total_weight_data
    );
    THCudaCheck(cudaGetLastError());
  }

  if (weights)
    THCTensor_(free)(state, weights);
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, input);
}

void THNN_(SpatialClassNLLCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCIndexTensor *target,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int64_t reduction,
           THCTensor *weights,
           THCTensor *total_weight,
           int64_t ignore_index)
{
  THNN_(SpatialClassNLLCriterion_shapeCheck)(state, input, target, weights);
  THCTensor_(resizeAs)(state, gradInput, input);
  THCTensor_(zero)(state, gradInput);
  THArgCheck(THCTensor_(isContiguous)(state, gradInput), 4,
             "gradInput must be contiguous");

  if (weights)
    THCUNN_assertSameGPU(state, 5, weights, input, target, gradInput, total_weight);
  else
    THCUNN_assertSameGPU(state, 4, input, target, gradInput, total_weight);

  if (reduction == Reduction::None) {
    THNN_(SpatialClassNLLCriterion_gradOutput_no_reduce_shapeCheck)(
        state,
        gradOutput,
        target);

    int64_t batch_size = THCTensor_(size)(state, input, 0);
    int64_t H = THCTensor_(size)(state, input, 2);
    int64_t W = THCTensor_(size)(state, input, 3);

    if (weights) {
      weights = THCTensor_(newContiguous)(state, weights);
    }

    int64_t count = batch_size * H * W;
    SpatialClassNLLCriterion_updateGradInput_no_reduce_kernel<scalar_t>
      <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        count,
        toDeviceTensor<THCIndex_t, 3>(state, target),
        toDeviceTensor<scalar_t, 3>(state, gradOutput),
        toDeviceTensor<scalar_t, 4>(state, gradInput),
        weights ? THCTensor_(data)(state, weights) : NULL,
        ignore_index);

    if (weights) {
      THCTensor_(free)(state, weights);
    }
    return;
  }

  input = THCTensor_(newContiguous)(state, input);
  weights = weights ? THCTensor_(newContiguous)(state, weights) : NULL;
  target = THCIndexTensor_(newContiguous)(state, target);

  scalar_t *gradOutput_data = THCTensor_(data)(state, gradOutput);
  scalar_t *weights_data = weights ? THCTensor_(data)(state, weights) : NULL;
  scalar_t *gradInput_data = THCTensor_(data)(state, gradInput);
  THCIndex_t *target_data = THCIndexTensor_(data)(state, target);
  scalar_t *total_weight_data = THCTensor_(data)(state, total_weight);

  THCIndex_t batch_size = THCIndexTensor_(size)(state, target, 0);
  THCIndex_t map_nelem = THCIndexTensor_(nElement)(state, target) / batch_size;
  int blocks_per_sample = GET_BLOCKS(map_nelem) / 128;
  blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
  int total_blocks = blocks_per_sample * batch_size;

  cunn_SpatialClassNLLCriterion_updateGradInput_kernel
    <<<total_blocks, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      gradInput_data,
      gradOutput_data,
      target_data,
      weights_data,
      total_weight_data,
      reduction == Reduction::Mean,
      THCTensor_(size)(state, input, 0),
      THCTensor_(size)(state, input, 1),
      THCTensor_(size)(state, input, 2) *THCTensor_(size)(state, input, 3),
      blocks_per_sample,
      ignore_index
  );
  THCudaCheck(cudaGetLastError());

  if (weights)
    THCTensor_(free)(state, weights);
  THCIndexTensor_(free)(state, target);
  THCTensor_(free)(state, input);
}

#endif
