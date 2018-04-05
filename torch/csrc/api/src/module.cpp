#include <torch/nn/module.h>
#include <torch/nn/cursor.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch { namespace nn {

Module::Module(std::string name)
    : name_(std::move(name)), is_training_(false) {}

Module::~Module() = default;

std::unique_ptr<Module> Module::clone() {
  AT_ERROR("clone() has no default implementation");
}

// Train/Eval mode
void Module::train() {
  is_training_ = true;
}

void Module::eval() {
  is_training_ = false;
}

bool Module::is_training() const noexcept {
  return is_training_;
}

// Recursive Transformations
void Module::cpu() {}

void Module::cuda() {}

void Module::type(at::ScalarType new_type) {}
void Module::zero_grad() {}

// Recursive Accessors
ModuleCursor Module::modules() {
  return {};
}

// ModuleCursor with a different policy
ModuleCursor Module::children() {
  return {};
}

ParameterCursor Module::parameters() {
  return {};
}
BufferCursor Module::buffers() {
  return {};
}

// Serialization/Deserialization
void Module::serialize(Archive& archive) {}
void Module::deserialize(Archive&& archive) {}

const std::string& Module::name() const noexcept {
  return name_;
}

void Module::register_parameters(
    const std::unordered_map<std::string, Tensor>& parameters) {}

void Module::register_buffers(
    const std::unordered_map<std::string, Tensor>& buffers) {}

void Module::register_modules(
    const std::unordered_map<std::string, Module*>& modules) {}

void Module::register_modules(const std::vector<Module*>& modules) {}

}} // namespace torch::nn
