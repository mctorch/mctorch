#include "caffe2/operators/instance_norm_op.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

// Here lives two separate implementations of the forward and backward passes of
// instance normalization, one for NHWC order and the other for NCHW order.
// Two implementations allow us to make use of Eigen vectorized operations
// without an expensive tensor transpose operation.

template <typename T, typename Context>
bool InstanceNormOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(INPUT);

  CAFFE_ENFORCE(
      !IsInputOutputAlias(INPUT, OUTPUT),
      "Can't run InstanceNorm NHWC in-place");
  auto* mean = OutputSize() > 1 ? Output(MEAN) : &mean_;
  auto* inv_stdev = OutputSize() > 1 ? Output(INV_STDEV) : &inv_stdev_;
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const size_t offset = H * W * C;

  CAFFE_ENFORCE_EQ(Input(SCALE).numel(), C);
  CAFFE_ENFORCE_EQ(Input(BIAS).numel(), C);

  auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
  mean->Resize(N, C);
  inv_stdev->Resize(N, C);
  ConstEigenVectorArrayMap<T> scale(Input(SCALE).template data<T>(), C);
  ConstEigenVectorArrayMap<T> bias(Input(BIAS).template data<T>(), C);
  for (int n = 0; n < N; ++n) {
    ConstEigenArrayMap<T> Xmat(X.template data<T>() + offset * n, C, H * W);
    EigenArrayMap<T> Ymat(Y->template mutable_data<T>() + offset * n, C, H * W);
    EigenVectorArrayMap<T> mean_arr(
        mean->template mutable_data<T>() + n * C, C);
    EigenVectorArrayMap<T> inv_stdev_arr(
        inv_stdev->template mutable_data<T>() + n * C, C);

    // The following effectively does the row wise mean computation:
    //   mean_arr = Xmat.rowwise().mean();
    // but manually vectorizes over columns.
    mean_arr = Xmat.col(0);
    for (int i = 1; i < H * W; ++i) {
      mean_arr += Xmat.col(i);
    }
    mean_arr *= 1. / (H * W);
    Ymat = Xmat.colwise() - mean_arr;
    // The following effectively does row wise squared norm computation,
    // but manually vectorizes over columns similar to the mean case.
    inv_stdev_arr = Ymat.col(0) * Ymat.col(0);
    for (int i = 1; i < H * W; ++i) {
      inv_stdev_arr += Ymat.col(i) * Ymat.col(i);
    }
    inv_stdev_arr = (inv_stdev_arr / (H * W) + epsilon_).sqrt().inverse();
    Ymat = (Ymat.colwise() * (inv_stdev_arr * scale)).colwise() + bias;
  }
  return true;
}

template <typename T, typename Context>
bool InstanceNormOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);

  auto* mean = OutputSize() > 1 ? Output(MEAN) : &mean_;
  auto* inv_stdev = OutputSize() > 1 ? Output(INV_STDEV) : &inv_stdev_;
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);

  CAFFE_ENFORCE_EQ(scale.numel(), C);
  CAFFE_ENFORCE_EQ(bias.numel(), C);

  auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
  mean->Resize(N, C);
  inv_stdev->Resize(N, C);

  const auto* Xdata = X.template data<T>();
  auto* Ydata = Y->template mutable_data<T>();
  const auto* scale_data = scale.template data<T>();
  const auto* bias_data = bias.template data<T>();
  auto* mean_data = mean->template mutable_data<T>();
  auto* inv_stdev_data = inv_stdev->template mutable_data<T>();

  // TODO: benchmark parallelization strategies.
  for (auto i = 0; i < N * C; ++i) {
    ConstEigenVectorArrayMap<T> Xi(Xdata + H * W * i, H * W);
    const T Xi_mean = Xi.mean();
    const T squared_norm = (Xi - Xi_mean).matrix().squaredNorm();
    const T inv_stdev = 1.0 / std::sqrt(squared_norm / (H * W) + epsilon_);
    mean_data[i] = Xi_mean;
    inv_stdev_data[i] = inv_stdev;
    EigenVectorArrayMap<T> Yi(Ydata + H * W * i, H * W);
    const T channel_scale = inv_stdev * scale_data[i % C];
    const T channel_shift = bias_data[i % C] - Xi_mean * channel_scale;
    Yi = Xi * channel_scale + channel_shift;
  }

  return true;
}

REGISTER_CPU_OPERATOR(InstanceNorm, InstanceNormOp<float, CPUContext>);

OPERATOR_SCHEMA(InstanceNorm)
    .NumInputs(3)
    .NumOutputs(1, 3)
    .AllowInplace({{0,0}})
    .SetDoc(R"DOC(
The *InstanceNorm* op applies Instance Normalization over a 4D input as described in [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).

$$output = \frac{input-\mu_{input}}{\sqrt{\sigma_{input}^2} + \epsilon}*scale + bias$$

Notice, two of the outputs are optional so there are three output cases for this op. Case 1: output; Case 2: output, saved_mean; Case 3: output, saved_mean, saved_inv_stdev.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "InstanceNorm",
    ["input", "scale", "bias"],
    ["output"],
    epsilon=1e-5,
)

workspace.FeedBlob("input", np.random.randn(2, 1, 3, 3).astype(np.float32))
print("input:\n", workspace.FetchBlob("input"), "\n")

workspace.FeedBlob("scale", np.array([1.5]).astype(np.float32))
print("scale: ", workspace.FetchBlob("scale"))

workspace.FeedBlob("bias", np.array([1.]).astype(np.float32))
print("bias: ", workspace.FetchBlob("bias"))

workspace.RunOperatorOnce(op)
print("output:\n", workspace.FetchBlob("output"))

```

**Result**

```

input:
 [[[[ 0.97856593 -1.1832817  -0.2540021 ]
   [-1.3315694  -0.7485018   0.3787225 ]
   [-0.6826597  -1.4637762   0.57116514]]]


 [[[-0.44948956  0.85544354 -0.9315333 ]
   [-0.37202677 -0.22266895 -0.27194235]
   [ 0.4948163  -0.7296504   1.3393803 ]]]]

scale:  [1.5]
bias:  [1.]
output:
 [[[[ 3.5017493  -0.3791256   1.2890853 ]
   [-0.6453266   0.40137637  2.4249308 ]
   [ 0.5195738  -0.8826599   2.7703972 ]]]


 [[[ 0.12639964  2.856744   -0.8821926 ]
   [ 0.28847694  0.60098207  0.49788612]
   [ 2.1021945  -0.45978796  3.869297  ]]]]

```

</details>

)DOC")
    .Arg("epsilon", "*(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero.")
    .Arg("order", "*(type: string; default: \"NCHW\")* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is \"NHWC\".")
    .Input(0, "input", "The input 4-dimensional NCHW tensor to be operated on.")
    .Input(1, "scale", "The input 1-dimensional scale tensor of size *C*.")
    .Input(2, "bias", "The input 1-dimensional bias tensor of size *C*.")
    .Output(
        0,
        "output",
        "The output 4-dimensional tensor of the same shape as input.")
    .Output(
        1,
        "saved_mean",
        "(Optional) Saved mean used during training to speed up gradient computation. Should not be used for testing.")
    .Output(
        2,
        "saved_inv_stdev",
        "(Optional) Saved inverse stdev used during training to speed up gradient computation. Should not be used for testing.");

} // namespace caffe2
