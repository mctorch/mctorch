#include "caffe2/operators/conv_transpose_op.h"
#include "caffe2/ideep/operators/conv_transpose_unpool_base_op.h"
#include "caffe2/ideep/operators/operator_fallback_ideep.h"
#include <vector>

using namespace caffe2;

namespace {

// TODO: The code below works around correctness issues with particular input shapes
// in MKL-DNN v0.17, will be removed with the fixes in MKL-DNN 0.18.
bool need_type_zero_pad(const mkldnn_memory_desc_t *pd) {
  if (pd->ndims == 0) {
    return false;
  }
  int p1 = 1, p2 = 1;
  for (int i = 0; i < pd->ndims; i++) {
    p1 *= pd->dims[i];
    p2 *= pd->layout_desc.blocking.padding_dims[i];
  }
  return (p1 != p2);
}

class IDEEPConvTransposeOp final : public IDEEPConvTransposeUnpoolBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS();
  // TODO: The code below works around correctness issues with particular input shapes
  // in MKL-DNN v0.17, will be removed with the fixes in MKL-DNN 0.18.
  using FALLBACK_OP = IDEEPFallbackOp<ConvTransposeOp<float, CPUContext>>;

  IDEEPConvTransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvTransposeUnpoolBase(operator_def, ws),
        training_mode_(
            OperatorBase::GetSingleArgument<int>("training_mode", 0)),
        fallback_(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
  }
  ~IDEEPConvTransposeOp() override {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);
    CAFFE_ENFORCE_EQ(X.ndims(), 4);
    CAFFE_ENFORCE_EQ(filter.ndims(), 4);
    CAFFE_ENFORCE_EQ(filter.get_dim(2), kernel_h());
    CAFFE_ENFORCE_EQ(filter.get_dim(3), kernel_w());

    ideep::tensor::dims Y_dims;
    const bool pre_converted = filter.get_public_format() == ideep::format::iohw;
    if (!pre_converted) {
      CAFFE_ENFORCE_EQ(
          filter.get_dim(0), X.get_dim(1),
          "filter number must be equal to input channel number");

      Y_dims = CalcOutputDims(X, filter.get_dim(1));

      ideep::tensor::dims filter_dims_mkldnn {filter.get_dim(1), filter.get_dim(0),
          filter.get_dim(2), filter.get_dim(3)};
      auto expected_descriptor =
          ideep::convolution_transpose_forward::expected_weights_descriptor(
              filter_dims_mkldnn,
              filter.get_data_type(),
              stride_,
              pad_tl(),
              pad_br());
      const bool weights_changed =
          (cached_weights_descriptor_ != filter.get_descriptor());
      if (weights_changed) {
        cached_weights_descriptor_ = filter.dup_descriptor();
      }

      if (training_mode_ || weights_changed) {
        auto filter_in = filter;
        // Framework has filters in IOHW while MKL-DNN requires OIHW,
        // we have to do explicit conversion here.
        filter_in.set_public_format(ideep::format::iohw);
        filter_.init(expected_descriptor);
        filter_.feed_from(filter_in);
      }

      // TODO: The code below works around correctness issues with particular input shapes
      // in MKL-DNN v0.17, will be removed with the fixes in MKL-DNN 0.18.
      if (need_type_zero_pad(filter_.get_mkldnn_memory_desc_t())) {
        return fallback_.Run(0);
      }
    } else {
      CAFFE_ENFORCE_EQ(
        filter.get_dim(1), X.get_dim(1),
        "filter number must be equal to input channel number");

      // TODO: The code below works around correctness issues with particular input shapes
      // in MKL-DNN v0.17, will be removed with the fixes in MKL-DNN 0.18.
      if (need_type_zero_pad(filter.get_mkldnn_memory_desc_t())) {
        return fallback_.Run(0);
      }

      Y_dims = CalcOutputDims(X, filter.get_dim(0));
    }

    if (InputSize() > BIAS) {
      const auto& bias = Input(BIAS);
      CAFFE_ENFORCE_EQ(bias.ndims(), 1, "bias must be 1D tensor");
      CAFFE_ENFORCE_EQ(
          bias.get_dim(0), pre_converted ? filter.get_dim(0) : filter.get_dim(1),
          "bias dimension must be equal to output channel number");

      ideep::convolution_transpose_forward::compute(
          X, pre_converted ? filter : filter_, bias, Y_dims, *Y, stride_, pad_tl(), pad_br());
    } else {
      ideep::convolution_transpose_forward::compute(
          X, pre_converted ? filter : filter_, Y_dims, *Y, stride_, pad_tl(), pad_br());
    }
    return true;
  }

 private:
  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);

  const bool training_mode_;
  ideep::tensor filter_;
  ideep::tensor::descriptor cached_weights_descriptor_;
  // TODO: The code below works around correctness issues with particular input shapes
  // in MKL-DNN v0.17, will be removed with the fixes in MKL-DNN 0.18.
  FALLBACK_OP fallback_;

};

class IDEEPConvTransposeGradientOp final : public IDEEPConvTransposeUnpoolBase {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS();
  // TODO: The code below works around correctness issues with particular input shapes
  // in MKL-DNN v0.17, will be removed with the fixes in MKL-DNN 0.18.
  using FALLBACK_OP = IDEEPFallbackOp<ConvTransposeGradientOp<float, CPUContext>>;

  IDEEPConvTransposeGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPConvTransposeUnpoolBase(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", false)),
        fallback_(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        pad_l() == pad_r() && pad_t() == pad_b(),
        "Uneven padding not supported.");
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
    CAFFE_ENFORCE(
        OperatorBase::GetSingleArgument<int>("training_mode", 0),
        "In order to backward propagate weights correctly, "
        "please set training_mode=1");
  }
  ~IDEEPConvTransposeGradientOp() override {}

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dfilter = Output(FILTER_GRAD);

    itensor dfilter_;
    itensor filter_;
    auto filter_in = filter;

    itensor::dims oihw_dims {filter.get_dim(1), filter.get_dim(0),
        filter.get_dim(2), filter.get_dim(3)};
    const bool pre_converted = (filter.get_public_format() == ideep::format::iohw);
    if (!pre_converted) {
      auto expected_descriptor =
            ideep::convolution_transpose_forward::expected_weights_descriptor(
              oihw_dims,
              filter.get_data_type(),
              stride_,
              pad_tl(),
              pad_br());
      // Framework has filters in IOHW while MKL-DNN requires OIHW,
      // we have to do explicit conversion here.
      filter_in.set_public_format(ideep::format::iohw);
      filter_.init(expected_descriptor);
      filter_.feed_from(filter_in);

      // TODO: The code below works around correctness issues with particular input shapes
      // in MKL-DNN v0.17, will be removed with the fixes in MKL-DNN 0.18.
      if (((filter_in.get_dim(0) % 8 != 0) && (stride_[0] * stride_[1] != 1)) ||
          need_type_zero_pad(filter_.get_mkldnn_memory_desc_t())) {
        return fallback_.Run(0);
      }
    } else {
      // TODO: The code below works around correctness issues with particular input shapes
      // in MKL-DNN v0.17, will be removed with the fixes in MKL-DNN 0.18.
      if (((filter_in.get_dim(0) % 8 != 0) && (stride_[0] * stride_[1] != 1)) ||
          need_type_zero_pad(filter.get_mkldnn_memory_desc_t())) {
        return fallback_.Run(0);
      }
    }

    if (no_bias_) {
      ideep::convolution_transpose_backward_weights::compute(
          X, dY, pre_converted ? filter.get_dims() : oihw_dims,
          pre_converted ? *dfilter : dfilter_, stride_, pad_tl(), pad_br());
    } else {
      auto* dbias = Output(BIAS_OR_INPUT_GRAD);
      ideep::convolution_transpose_backward_weights::compute(
          X,
          dY,
          pre_converted ? filter.get_dims() : oihw_dims,
          pre_converted ? *dfilter : dfilter_,
          *dbias,
          stride_,
          pad_tl(),
          pad_br());
    }

    if (!pre_converted) {
      // Framework has filters in IOHW while MKL-DNN requires OIHW,
      // we have to do explicit conversion here.
      dfilter_.set_public_format(ideep::format::iohw);
      dfilter->reinit(filter.get_descriptor());
      dfilter_.to_public(dfilter->get_data_handle());
    } else {
      dfilter->set_public_format(ideep::format::iohw);
    }

    if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
      auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
      ideep::convolution_transpose_backward_data::compute(
          dY, pre_converted ? filter : filter_, X.get_dims(), *dX, stride_, pad_tl(), pad_br());
    }

    return true;
  }

 private:
  bool no_bias_;
  // TODO: The code below works around correctness issues with particular input shapes
  // in MKL-DNN v0.17, will be removed with the fixes in MKL-DNN 0.18.
  FALLBACK_OP fallback_;

  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

REGISTER_IDEEP_OPERATOR(ConvTranspose, IDEEPConvTransposeOp);
REGISTER_IDEEP_OPERATOR(ConvTransposeGradient, IDEEPConvTransposeGradientOp);

} // namespace
