#pragma once

#include <memory>

#include <fbgemm/Fbgemm.h>
#include <fbgemm/src/FbgemmI8DepthwiseAvx2.h>

#include "caffe2/quantization/server/dnnlowp.h"

namespace caffe2 {

/**
 * Packed weight matrix for DNNLOWP Int8FC operator
 */
struct Int8FCDNNLowPPackedWeightBlob {
  std::vector<dnnlowp::TensorQuantizationParams> qparams;
  std::shared_ptr<std::vector<std::int32_t>> column_offsets;

  // The original tensor before packing
  Tensor original_tensor{CPU};

  std::shared_ptr<std::vector<std::int32_t>> bias;

  // Only for 32-bit accumulation
  std::shared_ptr<fbgemm::PackBMatrix<std::int8_t>> W;

  // Only for 16-bit accumulation
  // Dense matrix holding common values
  std::shared_ptr<fbgemm::PackBMatrix<std::int8_t, std::int16_t>> W_acc16;
  // Sparse matrix holding outliers
  std::shared_ptr<fbgemm::CompressedSparseColumn> W_outlier;
  int nbits_in_non_outlier;
};

/**
 * Packed weight matrix for DNNLOWP Int8Conv operator
 */
struct Int8ConvDNNLowPPackedWeightBlob : public Int8FCDNNLowPPackedWeightBlob {
  // Only for 32-bit accumulation
  std::shared_ptr<fbgemm::Packed3x3ConvMatrix> W_depthwise_3x3;
  std::shared_ptr<fbgemm::Packed3x3x3ConvMatrix> W_depthwise_3x3x3;
  std::shared_ptr<fbgemm::PackWeightMatrixForGConv<std::int8_t>> W_gconv;
};

} // namespace caffe2
