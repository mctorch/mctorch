#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>

#include <TH/THGeneral.h>

#include <cusparse.h>

namespace at { namespace native { namespace sparse { namespace cuda {


std::string getCusparseErrorString(cusparseStatus_t status) {
  switch(status)
  {
    case CUSPARSE_STATUS_SUCCESS:
      return "success";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "library not initialized";

    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "resource allocation failed";

    case CUSPARSE_STATUS_INVALID_VALUE:
      return "an invalid numeric value was used as an argument";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "an absent device architectural feature is required";

    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "an access to GPU memory space failed";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "the GPU program failed to execute";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "an internal operation failed";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "the matrix type is not supported by this function";

    case CUSPARSE_STATUS_ZERO_PIVOT:
      return "an entry of the matrix is either structural zero or numerical zero (singular block)";

    default:
      {
        std::ostringstream oss;
        oss << "unknown error " << static_cast<int64_t>(status);
        return oss.str();
      }
  }
}

inline void CUSPARSE_CHECK(cusparseStatus_t status)
{
  if (status != CUSPARSE_STATUS_SUCCESS) {
    AT_ERROR("cusparse runtime error: ", getCusparseErrorString(status));
  }
}

inline cusparseHandle_t setCUDASparseStream() {
  cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseSetStream(handle, at::cuda::getCurrentCUDAStream());
  return handle;
}

void Xcoo2csr(const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  AT_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
    "cusparseXcoo2csr only supports m, nnz with the bound [val] <= ",
    INT_MAX);

#ifdef __HIP_PLATFORM_HCC__
  // current shortcoming in hip/rocSPARSE
  if(nnz == 0 || m == 0) return;
#endif

  int i_nnz = (int)nnz;
  int i_m = (int)m;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(cusparseXcoo2csr(handle, coorowind, i_nnz, i_m, csrrowptr, CUSPARSE_INDEX_BASE_ZERO));
}

cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    AT_ERROR("trans must be one of: t, n, c");
  }
}

void adjustLd(char transb, int64_t m, int64_t n, int64_t k, int64_t *ldb, int64_t *ldc)
{
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    *ldc = m;

  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

/* Level 3 */
void Scsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseScsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

#ifdef __HIP_PLATFORM_HCC__
  // current shortcoming in hip/rocSPARSE
  if(n == 0) return;
#endif

  auto handle = setCUDASparseStream();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  CUSPARSE_CHECK(cusparseScsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
  CUSPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

void Dcsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseDcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;


#ifdef __HIP_PLATFORM_HCC__
  // current shortcoming in hip/rocSPARSE
  if(n == 0) return;
#endif

  auto handle = setCUDASparseStream();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  CUSPARSE_CHECK(cusparseDcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
  CUSPARSE_CHECK(cusparseDestroyMatDescr(desc));
  // TODO: Proper fix is to create real descriptor classes
}

/* format conversion */
void CreateIdentityPermutation(int64_t nnz, int *P) {
  AT_CHECK((nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  cusparseCreateIdentityPermutation(handle, i_nnz, P);
}

void Xcsrsort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <=",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(cusparseXcsrsort_bufferSizeExt(handle, i_m, i_n, i_nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

void Xcsrsort(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  CUSPARSE_CHECK(cusparseXcsrsort(handle, i_m, i_n, i_nnz, desc, csrRowPtr, csrColInd, P, pBuffer));
  CUSPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

void Xcoosort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcoosort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
}

void XcoosortByRow(int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "XcoosortByRow only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(cusparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows, cooCols, P, pBuffer));
}


}}}} // namespace at::native::sparse::cuda
