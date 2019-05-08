/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SMOOTH_L1_LOSS_OP_H_
#define SMOOTH_L1_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SmoothL1LossOp final : public Operator<Context> {
 public:
  SmoothL1LossOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        beta_(this->template GetSingleArgument<float>("beta", 1.)),
        scale_(this->template GetSingleArgument<float>("scale", 1.)) {
    CAFFE_ENFORCE(beta_ > 0);
    CAFFE_ENFORCE(scale_ >= 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float beta_; // Transition point from L1 to L2 loss
  float scale_; // Scale the loss by scale_
  Tensor buff_{Context::GetDeviceType()}; // Buffer for element-wise differences
};

template <typename T, class Context>
class SmoothL1LossGradientOp final : public Operator<Context> {
 public:
  SmoothL1LossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        beta_(this->template GetSingleArgument<float>("beta", 1.)),
        scale_(this->template GetSingleArgument<float>("scale", 1.)) {
    CAFFE_ENFORCE(beta_ > 0);
    CAFFE_ENFORCE(scale_ >= 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // No CPU implementation for now
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float beta_; // Transition point from L1 to L2 loss
  float scale_; // Scale the loss by scale_
  Tensor buff_{Context::GetDeviceType()}; // Buffer for element-wise differences
};

} // namespace caffe2

#endif // SMOOTH_L1_LOSS_OP_H_
