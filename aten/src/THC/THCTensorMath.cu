#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorMath.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCTensor.hpp>


#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
#include <thrust/system/cuda/execution_policy.h>
#endif
#include <cfloat>

template <typename T>
struct TensorFillOp {
  TensorFillOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* v) { *v = val; }

  const T val;
};

// copypasta from https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
template <typename Iterator>
class strided_range
{
 public:

  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct stride_functor : public thrust::unary_function<difference_type,
                                                        difference_type>
  {
    difference_type stride;

    stride_functor(difference_type stride)
        : stride(stride) {}

    __host__ __device__
    difference_type operator()(const difference_type& i) const
      {
        return stride * i;
      }
  };

  typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride)
      : first(first), last(last), stride(stride) {}

  iterator begin(void) const
    {
      return PermutationIterator(first,
                                 TransformIterator(CountingIterator(0),
                                                   stride_functor(stride)));
    }

  iterator end(void) const
    {
      return begin() + ((last - first) + (stride - 1)) / stride;
    }

 protected:
  Iterator first;
  Iterator last;
  difference_type stride;
};

struct idx_functor
{
  int64_t div;
  int64_t size;

  __host__ __device__
  idx_functor(int64_t div, int64_t size) : div(div), size(size) {}

  __host__ __device__
  int64_t operator()(int64_t val) {
    return (val / div) % size;
  }
};

template <typename T>
struct NonZeroOp
{
  NonZeroOp() {}
  __host__ __device__ bool operator()(T lhs) const {
    if (THCNumerics<T>::ne(lhs, ScalarConvert<float, T>::to(0.0))) {
      return true;
    } else {
      return false;
    }
  }
};

template <>
struct NonZeroOp<bool>
{
  NonZeroOp() {}
  __host__ __device__ bool operator()(bool lhs) const {
    return lhs != false;
  }
};

#include <THC/generic/THCTensorMath.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMath.cu>
#include <THC/THCGenerateBoolType.h>
