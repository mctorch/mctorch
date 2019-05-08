#pragma once

#include <torch/csrc/jit/code_template.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

/*with type_as not checking type of its input, a fusion group can have non-fp32
tensor as input. Correct code for this case is generated, however, nvrtc does
not know how to handle int*_t integer types, so typedefs help it handle those
cases*/

static auto type_declarations_template = CodeTemplate(R"(

#define POS_INFINITY INFINITY
#define NEG_INFINITY -INFINITY

typedef ${IndexType} IndexType;
template<typename T, size_t N>
struct TensorInfo {
  T* data;
  IndexType sizes[N];
  IndexType strides[N];
};
template<typename T>
struct TensorInfo<T, 0> {
  T * data;
};
)");

static auto cpu_compilation_unit_template = CodeTemplate(R"(
#include <math.h>
#include <cstddef>
#include <cstdint>

double rsqrt(double x) {
  return 1.0/sqrt(x);
}

float rsqrtf(float x) {
  return 1.0f/sqrtf(x);
}

double frac(double x) {
  return x - trunc(x);
}

float fracf(float x) {
  return x - truncf(x);
}

${type_declarations}

#define OMP_THRESHOLD 100000
static void ${kernelName}_kernel(IndexType totalElements, ${formals}) {
  #pragma omp parallel for if(totalElements > OMP_THRESHOLD)
  for (IndexType linearIndex = 0;
        linearIndex < totalElements;
        linearIndex += 1) {
      // Convert `linearIndex` into an offset of tensor:
      ${tensorOffsets}
      // calculate the results
      ${kernelBody}
    }
}

extern "C"
void ${kernelName}(IndexType totalElements, void ** args) {
  ${kernelName}_kernel(totalElements ${,argument_loads});
}
)");

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
