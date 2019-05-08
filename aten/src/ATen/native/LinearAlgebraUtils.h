#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <limits>
#include <sstream>

namespace at { namespace native {

/*
 * Clones a Tensor so that the following conditions hold:
 * If we think of a Tensor of having size (B, M, N), where B is any number
 * of batch dimensions, then:
 * - Each (M, N) matrix is in column major form
 * - Let Tensor P have size (B, M, N) and Q have size (B, M', N').
 *   Then when laid out in memory, the M by N matrix starting at
 *   P.data_ptr()[b * M * N] is of the same corresponding batch as the M' by N'
 *   matrix starting at Q.data_ptr()[b * M' * N'].
 */
static inline Tensor cloneBatchedColumnMajor(const Tensor& src) {
  // If src is already in batched column major format, then
  // this will be efficient (no reordering of the data will occur)
  // because the first transpose will make the tensor contiguous,
  // and cloning a contiguous tensor is fast.
  auto result = src.transpose(-2, -1).clone();
  result.transpose_(-2, -1);
  return result;
}

/*
 * Given batches of matrices with arbitrary batch dim,
 * computes the number of batches.
 */
static inline int64_t batchCount(const Tensor& batched_matrices) {
  int64_t result = 1;
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    result *= batched_matrices.size(i);
  }
  return result;
}

// Computes the number of elements of a matrix in a batched matrix tensor
static inline int64_t matrixStride(const Tensor& batched_matrices) {
  return batched_matrices.size(-1) * batched_matrices.size(-2);
}

/* Checks a necessary property for the triu and tril implementations, hence the name.
 * Here batch contiguity is checked for tensors with greater than 4 dimensions.
 * Contiguous tensors and tensors with less than 3 dimensions pass this check
 */ 
static inline bool checkTrilTriuBatchContiguous(const Tensor& tensor) {
  // Complete contiguity is the most desired property, which is why
  // we return true if the tensor is contiguous
  if (tensor.is_contiguous()) return true;

  int64_t dims = tensor.dim();

  // Tensors with dimension less than 4 are handled by default
  if (dims <= 3) return true;

  int64_t expected_stride = tensor.size(-1) * tensor.size(-2);
  for (int64_t i = dims - 3; i >= 0; i--) {
    if (expected_stride != tensor.stride(i)) return false;
    expected_stride *= tensor.size(i);
  }
  return true;
}

// Returns the epsilon value for floating types except half
static inline double _get_epsilon(const ScalarType& sc_type) {
  switch (sc_type) {
    case at::ScalarType::Float:
      return static_cast<double>(std::numeric_limits<float>::epsilon());
    case at::ScalarType::Double:
      return std::numeric_limits<double>::epsilon();
    default:
      AT_ERROR("This function doesn't handle types other than float and double");
  }
}

// Validates input shapes and devices for linear solve methods (gesv, cholesky_solve)
static inline void linearSolveCheckInputs(const Tensor& self, const Tensor& A) {
  int64_t self_is_cuda = self.is_cuda();
  int64_t A_is_cuda = A.is_cuda();

  std::stringstream ss;
  if (self_is_cuda != A_is_cuda) {
    ss << "Expected b and A to be on the same device, but found b on ";
    if (self_is_cuda) {
      ss << "GPU";
    } else {
      ss << "CPU";
    }
    ss << " and A on ";
    if (A_is_cuda) {
      ss << "GPU";
    } else {
      ss << "CPU";
    }
    ss << " instead.";
    AT_ERROR(ss.str());
  }

  AT_CHECK(A.size(-1) == A.size(-2),
           "A must be batches of square matrices, "
           "but they are ", A.size(-1), " by ", A.size(-2), " matrices");

  AT_CHECK(A.size(-1) == self.size(-2),
           "Incompatible matrix sizes for matmul: each A "
           "matrix is ", A.size(-1), " by ", A.size(-1),
           " but each b matrix is ", self.size(-2), " by ", self.size(-1));
}

// Validates input shapes for operations on batches of square matrices (inverse, cholesky, lu)
static inline void squareCheckInputs(const Tensor& self) {
  AT_CHECK(self.size(-1) == self.size(-2),
           "A must be batches of square matrices, "
           "but they are ", self.size(-1), " by ", self.size(-2), " matrices");
}

/*
 * Given a vector of int64_t infos, obtained after a batch operations,
 * this function checks if the computation over all these batches has been
 * successful (info = 0) or not, and report in case of the latter.
 */ 
static inline void batchCheckErrors(std::vector<int64_t>& infos, const char* name) {
  for (size_t i = 0; i < infos.size(); i++) {
    auto info = infos[i];
    if (info < 0) {
      AT_ERROR(name, ": For batch ", i, ": Argument ", -info, " has illegal value");
    } else if (info > 0) {
      AT_ERROR(name, ": For batch ", i, ": U(", info, ",", info, ") is zero, singular U.");
    }
  }
}

/*
 * This is an overloaded case of the previous function for a tensor of infos.
 */
static inline void batchCheckErrors(const Tensor& infos, const char* name) {
  auto batch_size = infos.numel();
  auto infos_cpu = infos.to(at::kCPU);
  auto infos_data = infos_cpu.data<int>();
  for (size_t i = 0; i < batch_size; i++) {
    auto info = infos_data[i];
    if (info < 0) {
      AT_ERROR(name, ": For batch ", i, ": Argument ", -info, " has illegal value");
    } else if (info > 0) {
      AT_ERROR(name, ": For batch ", i, ": U(", info, ",", info, ") is zero, singular U.");
    }
  }
}

/*
 * Given a info int, obtained after a single operation, this function check if the computation
 * has been successful (info = 0) or not, and report in case of the latter.
 */
static inline void singleCheckErrors(int64_t info, const char* name) {
  if (info < 0) {
    AT_ERROR(name, ": Argument ", -info, " has illegal value");
  } else if (info > 0) {
    AT_ERROR(name, ": U(", info, ",", info, ") is zero, singular U.");
  }
}

// Checks if all the Tensors in a TensorList are of the same dimensions
static inline void checkAllSameDim(TensorList tensors, int64_t dim) {
  for (auto &t : tensors) {
    AT_CHECK(t.dim() == dim, "Tensor dimension is ", t.dim(), ", expected ", dim, " instead.");
  }
}

static inline std::tuple<Tensor,Tensor> _linear_solve_broadcast_args(const Tensor& arg1, const Tensor& arg2) {
  linearSolveCheckInputs(arg1, arg2);

  // broadcast the batch dimensions of arg1 and arg2.
  IntArrayRef arg1_batch_sizes(arg1.sizes().data(), arg1.ndimension() - 2);
  IntArrayRef arg2_batch_sizes(arg2.sizes().data(), arg2.ndimension() - 2);
  std::vector<int64_t> expand_batch_portion = infer_size(arg1_batch_sizes, arg2_batch_sizes);

  std::vector<int64_t> arg1_expand_size({expand_batch_portion});
  arg1_expand_size.insert(arg1_expand_size.end(), { arg1.size(-2), arg1.size(-1) });

  std::vector<int64_t> arg2_expand_size({expand_batch_portion});
  arg2_expand_size.insert(arg2_expand_size.end(), { arg2.size(-2), arg2.size(-1) });

  Tensor arg1_broadcasted  = arg1.expand(arg1_expand_size);
  Tensor arg2_broadcasted = arg2.expand(arg2_expand_size);
  return std::make_tuple(arg1_broadcasted, arg2_broadcasted);
}

}}  // namespace at::native
