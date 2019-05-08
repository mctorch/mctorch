#include <THC/THCTensorRandom.h>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensorMath.h>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorRandom.cuh>
#include <THC/THCGenerator.hpp>
#include <ATen/Config.h>

#include <ATen/cuda/_curand_mtgp32_host.h>

#include <thrust/functional.h>

#define MAX_NUM_BLOCKS 200
#define BLOCK_SIZE 256


THCGenerator* THCRandom_getGenerator(THCState* state);

/* Sets up generator. Allocates but does not create the generator states. Not thread-safe. */
__host__ void initializeGenerator(THCState *state, THCGenerator* gen)
{
  gen->state.gen_states = static_cast<curandStateMtgp32*>(THCudaMalloc(state, MAX_NUM_BLOCKS * sizeof(curandStateMtgp32)));
  gen->state.kernel_params = static_cast<mtgp32_kernel_params*>(THCudaMalloc(state, sizeof(mtgp32_kernel_params)));
}

/* Creates a new generator state given the seed. Not thread-safe. */
__host__ void createGeneratorState(THCGenerator* gen, uint64_t seed)
{
  if (curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, gen->state.kernel_params) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP constants failed.");
  }
  if (curandMakeMTGP32KernelState(gen->state.gen_states, mtgp32dc_params_fast_11213,
                                  gen->state.kernel_params, MAX_NUM_BLOCKS, seed) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP kernel state failed.");
  }
  // seed and offset for philox
  gen->state.initial_seed = seed;
  gen->state.philox_seed_offset = 0;
}

THC_API __host__ void THCRandom_getRNGState(THCState* state, THByteTensor *rng_state)
{
  THCGenerator* gen = THCRandom_getGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);

  // The RNG state comprises the MTPG32 states, the seed, and an offset used for Philox
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(gen->state.initial_seed);
  static const size_t offset_size = sizeof(gen->state.philox_seed_offset);
  static const size_t total_size = states_size + seed_size + offset_size;
  THByteTensor_resize1d(rng_state, total_size);
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  THCudaCheck(cudaMemcpy(THByteTensor_data(rng_state), gen->state.gen_states,
                         states_size, cudaMemcpyDeviceToHost));
  memcpy(THByteTensor_data(rng_state) + states_size, &gen->state.initial_seed, seed_size);
  memcpy(THByteTensor_data(rng_state) + states_size + seed_size, &gen->state.philox_seed_offset, offset_size);
}

__global__ void set_rngstate_kernel(curandStateMtgp32 *state, mtgp32_kernel_params *kernel)
{
#ifndef __HIP_PLATFORM_HCC__
  state[threadIdx.x].k = kernel;
#else
  state[threadIdx.x].set_params(kernel);
#endif
}

THC_API __host__ void THCRandom_setRNGState(THCState* state, THByteTensor *rng_state)
{
  THCGenerator* gen = THCRandom_getGenerator(state);
  std::lock_guard<std::mutex> lock(gen->mutex);

  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(gen->state.initial_seed);
  static const size_t offset_size = sizeof(gen->state.philox_seed_offset);
  static const size_t total_size = states_size + seed_size + offset_size;
  bool no_philox_seed = false;
  if (THByteTensor_nElement(rng_state) == total_size - offset_size) {
    no_philox_seed = true;
  }
  else {
    THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  }
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

  THCudaCheck(cudaMemcpy(gen->state.gen_states, THByteTensor_data(rng_state),
                         states_size, cudaMemcpyHostToDevice));
  set_rngstate_kernel<<<1, MAX_NUM_BLOCKS, 0, THCState_getCurrentStream(state)>>>(
      gen->state.gen_states, gen->state.kernel_params);
  memcpy(&gen->state.initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);
  if (!no_philox_seed) {
    memcpy(&gen->state.philox_seed_offset, THByteTensor_data(rng_state) + states_size + seed_size, offset_size);
  }
  else {
    gen->state.philox_seed_offset = 0;
  }
}

// Goes from (0, 1] to [0, 1). Note 1-x is not sufficient since for some floats
// eps near 0, 1-eps will round to 1.
template <typename T>
__device__ inline T reverse_bounds(T value) {
  if (THCNumerics<T>::eq(value, ScalarConvert<int, T>::to(1))) {
    return ScalarConvert<int, T>::to(0);
  }
  return value;
}


__device__ inline at::Half half_uniform_scale_and_shift(float x, double a, double b) {
  at::Half width = ScalarConvert<double, at::Half>::to(b - a);
  at::Half start = ScalarConvert<double, at::Half>::to(a);
  at::Half scaled = THCNumerics<at::Half>::mul(reverse_bounds(ScalarConvert<float, at::Half>::to(x)), width);
  return THCNumerics<at::Half>::add(scaled, start);
}

#define GENERATE_KERNEL1(NAME, T, ARG1, CURAND_T, CURAND_FUNC, TRANSFORM)      \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1)    \
{                                                                              \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                             \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {      \
    CURAND_T x = CURAND_FUNC(&state[blockIdx.x]);                              \
    if (i < size) {                                                            \
      T y = TRANSFORM;                                                         \
      result[i] = y;                                                           \
    }                                                                          \
  }                                                                            \
}

#define GENERATE_KERNEL2(NAME, T, ARG1, ARG2, CURAND_T, CURAND_FUNC, TRANSFORM)      \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1, ARG2)    \
{                                                                                    \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                   \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                      \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {            \
    CURAND_T x = CURAND_FUNC(&state[blockIdx.x]);                                    \
    if (i < size) {                                                                  \
      T y = TRANSFORM;                                                               \
      result[i] = y;                                                                 \
    }                                                                                \
  }                                                                                  \
}

// NOTE: curand_uniform is (0, 1] and we want [a, b)
GENERATE_KERNEL2(generate_uniform, float, float a, float b, float, curand_uniform, reverse_bounds(x) * (b-a) + a)
GENERATE_KERNEL2(generate_uniform, float, double a, double b, float, curand_uniform, reverse_bounds(x) * (b-a) + a)
GENERATE_KERNEL2(generate_uniform, double, double a, double b, double, curand_uniform_double, reverse_bounds(x) * (b-a) + a)

GENERATE_KERNEL2(generate_normal, float, double mean, double stdv, float, curand_normal, (x * stdv) + mean)
GENERATE_KERNEL2(generate_normal, double, double mean, double stdv, double, curand_normal_double, (x * stdv) + mean)

GENERATE_KERNEL1(generate_exponential, float, double lambda, float, curand_uniform, (float)(-1. / lambda * log(x)))
GENERATE_KERNEL1(generate_exponential, double, double lambda, double, curand_uniform_double, (double)(-1. / lambda * log(x)))

GENERATE_KERNEL2(generate_cauchy, float, double median, double sigma, float, curand_uniform, (float)(median + sigma * tan(M_PI*(x-0.5))))
GENERATE_KERNEL2(generate_cauchy, double, double median, double sigma, double, curand_uniform_double, (double)(median + sigma * tan(M_PI*(x-0.5))))

GENERATE_KERNEL2(generate_uniform, at::Half, double a, double b, float, curand_uniform, (half_uniform_scale_and_shift(x, a, b)))
GENERATE_KERNEL2(generate_normal, at::Half, double mean, double stdv, float, curand_normal, (ScalarConvert<float, at::Half>::to((x * stdv) + mean)))
GENERATE_KERNEL1(generate_exponential, at::Half, double lambda, float, curand_uniform, (ScalarConvert<float, at::Half>::to((float)(-1. / lambda * log(x)))))
GENERATE_KERNEL2(generate_cauchy, at::Half, double median, double sigma, float, curand_uniform, (ScalarConvert<float, at::Half>::to((float)(median + sigma * tan(M_PI*(x-0.5))))))

#include <THC/generic/THCTensorRandom.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorRandom.cu>
#include <THC/THCGenerateBoolType.h>

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2
