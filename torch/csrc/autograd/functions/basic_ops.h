#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/symbolic.h>

#include <memory>
#include <string>
#include <vector>

namespace torch { namespace autograd {

struct TORCH_API Error : public Function {
  Error(std::string msg, edge_list&& next_edges)
    : Function(std::move(next_edges))
    , msg(std::move(msg)) {}

  Error(std::string msg)
    : msg(std::move(msg)) {}

  variable_list apply(variable_list&& inputs) override;

  std::string msg;
};

// We print grad_fn names in tensor printing. For functions with backward
// NYI, grad_fn=<Error> will be printed if we use Error, which is confusing. So
// special case with a new NotImplemented function here.
struct TORCH_API NotImplemented : public Error {
  NotImplemented(const std::string& forward_fn, edge_list&& next_edges)
    : Error("derivative for " + forward_fn + " is not implemented",
            std::move(next_edges)) {}

  NotImplemented(const std::string& forward_fn)
    : Error("derivative for " + forward_fn + " is not implemented") {}
};

// Identity in forward, Error in backward. Used to implement @once_differentiable
struct TORCH_API DelayedError : public Function {
  DelayedError(std::string msg, int num_inputs)
    : msg(std::move(msg)) {
      for (int i = 0; i < num_inputs; i++)
        add_input_metadata(Function::undefined_input());
    }

  variable_list apply(variable_list&& inputs) override;

  std::string msg;
};

struct TORCH_API GraphRoot : public Function {
  GraphRoot(edge_list functions, variable_list inputs)
      : Function(std::move(functions)),
        outputs(std::move(inputs)) {}

  variable_list apply(variable_list&& inputs) override {
    return outputs;
  }

  variable_list outputs;
};

}}
