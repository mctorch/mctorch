#pragma once

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/lower_grad_of.h"
#include "torch/csrc/jit/passes/requires_grad_analysis.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"
#include "torch/csrc/jit/tracer.h"

#include <ATen/ATen.h>
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/variable.h"

namespace torch {
namespace jit {
namespace test {

using namespace torch::autograd;

using var_meta_type = std::vector<int64_t>;
using var_meta_list = std::vector<var_meta_type>;
using test_fn_type = std::function<variable_list(const variable_list&)>;

struct ADTestSpec {
  ADTestSpec(const char* name, var_meta_list input_meta, test_fn_type test_fn)
      : name(name), input_meta(input_meta), test_fn(test_fn) {}

  variable_list operator()(const variable_list& inputs) const {
    return test_fn(inputs);
  };

  std::vector<Variable> make_vars() const {
    std::vector<Variable> out;
    for (const auto& m : input_meta) {
      out.push_back(torch::randn(m, at::requires_grad(true)));
    }
    return out;
  }

  const char* name;
  var_meta_list input_meta;
  test_fn_type test_fn;
};

variable_list get_grad_outputs(const variable_list& vars) {
  return fmap(vars, [](const Variable& v) -> Variable {
    return at::randn(v.sizes(), v.options());
  });
}

std::shared_ptr<Graph> trace(
    const ADTestSpec& test,
    const variable_list& vars_in) {
  Stack input_vars = fmap<IValue>(vars_in);
  std::vector<TypePtr> input_types;
  input_types.reserve(input_vars.size());
  for (auto i = 0; i < input_vars.size(); i++) {
    input_types.push_back(TensorType::get());
  }
  auto input_typeptr = TupleType::create(std::move(input_types));
  std::shared_ptr<tracer::TracingState> state;
  Stack trace_stack_in;
  std::tie(state, trace_stack_in) =
      tracer::enter(tracer::TypedStack(input_vars, input_typeptr));
  variable_list trace_vars_in = fmap(
      trace_stack_in, [](const IValue& v) { return Variable(v.toTensor()); });
  auto trace_vars_out = test(trace_vars_in);
  tracer::exit(fmap<IValue>(trace_vars_out));
  return state->graph;
}

variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs) {
  const auto get_edge = [](const Variable& v) { return v.gradient_edge(); };
  auto& engine = torch::autograd::Engine::get_default_engine();
  return engine.execute(
      fmap(outputs, get_edge),
      grad_outputs,
      true,
      false,
      fmap(inputs, get_edge));
}

void testADFormulas() {
  const auto unwrap = [](const Variable& v) { return v.data(); };

  using VL = variable_list;
  const var_meta_list binary_pointwise = {{2, 3, 4, 5}, {2, 3, 4, 5}};
  const var_meta_list unary_pointwise = {{2, 3, 4, 5}};
  const var_meta_list unary_pointwise_2d = {{2, 3}};
  const std::vector<ADTestSpec> ad_tests = {
      {"add",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] + v[1]}; }},
      {"sub",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] - v[1]}; }},
      {"mul",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] * v[1]}; }},
      {"sigmoid",
       unary_pointwise,
       [](const VL& v) -> VL { return {v[0].sigmoid()}; }},
      {"tanh",
       unary_pointwise,
       [](const VL& v) -> VL { return {v[0].tanh()}; }},
      {"t", unary_pointwise_2d, [](const VL& v) -> VL { return {v[0].t()}; }},
      {"view",
       unary_pointwise_2d,
       [](const VL& v) -> VL {
         return {v[0].view({3, 2})};
       }},
      {"expand",
       {{2, 1}},
       [](const VL& v) -> VL {
         return {v[0].expand({2, 3})};
       }},
      {"mm",
       {{10, 12}, {12, 15}},
       [](const VL& v) -> VL { return {v[0].mm(v[1])}; }},
      // TODO: enable once we'll be able to capture lists across
      // forward-backward
      //{"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].chunk(4, 1)); }},
      //{"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].chunk(3, 2)); }},
      //{"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].split(4, 1)); }},
      //{"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].split(3, 2)); }},
  };

  for (const auto& test : ad_tests) {
    // Get reference values form autograd
    auto vars_in = test.make_vars();
    auto vars_out = test(vars_in);
    auto var_grads_in = get_grad_outputs(vars_out);
    auto var_grads_out = grad(vars_out, vars_in, var_grads_in);

    // Trace and differentiate the op
    auto graph = trace(test, vars_in);
    EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
    ConstantPropagation(graph);
    auto grad_spec = differentiate(graph);
    LowerGradOf(*grad_spec.df);
    // Get outputs from the interpreter
    auto tensors_in = fmap(vars_in, unwrap);
    auto tensor_grads_in = fmap(var_grads_in, unwrap);
    tensor_list tensors_out, tensor_grads_out;
    std::tie(tensors_out, tensor_grads_out) =
        runGradient(grad_spec, tensors_in, tensor_grads_in);

    // Compare results
    auto expected_tensors_out = fmap(vars_out, unwrap);
    auto expected_tensor_grads_out = fmap(var_grads_out, unwrap);
    assertAllClose(tensors_out, expected_tensors_out);
    assertAllClose(tensor_grads_out, expected_tensor_grads_out);
  }
}

void testDifferentiate() {
  auto graph = std::make_shared<Graph>();
  at::ScalarType s = at::ScalarType::Float;
  auto type = CompleteTensorType::create(s, at::kCPU, {2, 3, 4}, {12, 4, 1});

  // Build up a fake graph
  auto a = SymbolicVariable::asNewInput(*graph, type);
  auto b = SymbolicVariable::asNewInput(*graph, type);
  auto c = a * b * a + b;
  graph->registerOutput(c.value());

  auto grad_spec = differentiate(graph);
  std::vector<size_t> expected_captured_inputs = {0, 1};
  std::vector<size_t> expected_captured_outputs = {1, 2};
  std::vector<size_t> expected_input_vjps = {0, 1};
  std::vector<size_t> expected_output_vjps = {0, 1};
  ASSERT_EQ(grad_spec.f_real_outputs, 1);
  ASSERT_EQ(grad_spec.df_input_captured_inputs, expected_captured_inputs);
  ASSERT_EQ(grad_spec.df_input_captured_outputs, expected_captured_outputs);
  ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
  ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
  testing::FileCheck()
      .check_count("aten::mul", 2)
      ->check("aten::size")
      ->check("aten::add")
      ->run(*grad_spec.f);
  testing::FileCheck()
      .check("prim::GradOf[name=\"aten::add\"]")
      ->check_count("prim::GradOf[name=\"aten::mul\"]", 2)
      ->check_count("AutogradAdd", 2)
      ->run(*grad_spec.df);
}

void testDifferentiateWithRequiresGrad() {
  // Build up a fake graph
  auto graph = std::make_shared<Graph>();
  auto a = SymbolicVariable::asNewInput(*graph);
  auto b = SymbolicVariable::asNewInput(*graph);
  auto d = b * b + b;
  auto e = (d + a) * a + b;
  graph->registerOutput(d.value());
  graph->registerOutput(e.value());

  auto a_var = autograd::make_variable(
      at::empty_strided(2, 2, at::CPU(at::kFloat).options()), true);
  auto b_var = autograd::make_variable(
      at::empty_strided(2, 2, at::CPU(at::kFloat).options()), false);

  ArgumentSpecCreator asc(*graph);
  asc.setInputTypes(*graph, asc.create(true, {a_var, b_var}));

  PropagateInputShapes(graph);
  PropagateRequiresGrad(graph);

  auto grad_spec = differentiate(graph);
  std::vector<size_t> expected_input_vjps = {1, 2}; // for e and %4 = (d + a)
  std::vector<size_t> expected_output_vjps = {0}; // only a requires grad
  ASSERT_EQ(grad_spec.f_real_outputs, 2);
  ASSERT_EQ(grad_spec.df_input_captured_inputs, std::vector<size_t>({0}));
  ASSERT_EQ(grad_spec.df_input_captured_outputs, std::vector<size_t>({2, 3}));
  ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
  ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
  testing::FileCheck()
      .check("aten::mul")
      ->check_count("aten::add", 2)
      ->check("aten::mul")
      ->check("aten::size")
      ->check("aten::add")
      ->run(*grad_spec.f);

  testing::FileCheck()
      .check_count("prim::GradOf[name=\"aten::mul\"]", 1, /*exactly*/ true)
      ->run(*grad_spec.df);
}

} // namespace test
} // namespace jit
} // namespace torch
