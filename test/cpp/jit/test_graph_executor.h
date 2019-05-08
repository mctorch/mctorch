#pragma once

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"
#include "torch/csrc/jit/graph_executor.h"

namespace torch {
namespace jit {
namespace test {

void testGraphExecutor() {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2 * input_size;

  auto v = [](at::Tensor t) { return autograd::make_variable(t, false); };

  auto input = at::randn({batch_size, input_size}, at::kCUDA);
  auto hx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto cx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));

  auto g = build_lstm();
  GraphExecutor executor(g);
  auto stack = createStack({v(input), v(hx), v(cx), v(w_ih), v(w_hh)});
  executor.run(stack);
  ASSERT_EQ(stack.size(), 2);
  at::Tensor r0, r1;
  std::tie(r0, r1) = lstm(input, hx, cx, w_ih, w_hh);
  ASSERT_TRUE(almostEqual(Variable(stack[0].toTensor()).data(), r0));
  ASSERT_TRUE(almostEqual(Variable(stack[1].toTensor()).data(), r1));
}

} // namespace test
} // namespace jit
} // namespace torch
