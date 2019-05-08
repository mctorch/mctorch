#pragma once

#include "caffe2/core/logging.h"
#include "caffe2/opt/shape_info.h"
#include "caffe2/proto/caffe2_pb.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace caffe2 {
// This struct stores the max bound size for batch in the general sense. We have
// the conventioal batch size and the look-up sequence, which is also batch in a
// sense.
struct CAFFE2_API BoundShapeSpec {
  explicit BoundShapeSpec(int64_t b, int64_t q)
      : max_batch_size(b), max_seq_size(q) {}
  int64_t max_batch_size;
  int64_t max_seq_size;
};

/// \class A class that does bound shape inference given a C2 net. Depending on
/// its type, each op have a maximum shape that it accepts. We define some
/// initial bound for certain dimension, for example max batch size or max
/// sequnce lookup size. And the inference will first infer the input size and
/// then propagates the bound shape down the network. For now the variable part
/// (bound part) is the first dimension of the shape, which usually corresponds
/// to the batch size or sequence lookup size.
class CAFFE2_API BoundShapeInferencer {
 public:
  explicit BoundShapeInferencer(const BoundShapeSpec& spec) : spec_(spec) {
    CAFFE_ENFORCE_GE(spec_.max_batch_size, 0);
    CAFFE_ENFORCE_GE(spec_.max_seq_size, 0);
  }

  void InferBoundShapeAndType(
      const NetDef& net,
      const std::unordered_map<std::string, ShapeInfo>& info);

  const ShapeInfoMap& shape_info() const {
    return shape_info_;
  }

  /// Print out all the shape info
  std::string PrintShapeInfo() const {
    std::stringstream ss;
    for (const auto& kv : shape_info_) {
      const auto& s = kv.second;
      ss << s.shape.name() << ": dim_type: " << s.dim_type << ", dims: [";
      for (const auto d : s.shape.dims()) {
        ss << d << ", ";
      }
      ss << "], dtype: " << s.shape.data_type() << "\n";
    }
    return ss.str();
  }

 private:
  TensorShape& CheckAndSetTensorShapeAndType(
      const std::string& name,
      ShapeInfo::DimType t,
      std::vector<int64_t> bound_dims,
      TensorProto::DataType type,
      bool is_quantized);

  void InferConcatInputs(const OperatorDef& op);

  void InferGivenTensorFill(const OperatorDef& op);
  void InferSparseLengthsSum(const OperatorDef& op);
  void InferFC(const OperatorDef& op);
  void InferConcat(const OperatorDef& op);
  void InferShape(const OperatorDef& op);
  void InferReshape(const OperatorDef& op);
  void InferLengthsRangeFill(const OperatorDef& op);

  // Standard shape/type inference using op schema registered shape inference
  // function
  void InferCommonOp(const OperatorDef& op);

  const BoundShapeSpec spec_;
  ShapeInfo::DimType current_dim_type_{ShapeInfo::DimType::BATCH};
  int64_t current_max_batch_size_{0};
  std::unordered_map<std::string, ShapeInfo> shape_info_;
};

} // namespace caffe2
