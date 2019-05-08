#include <torch/csrc/jit/python_arg_flatten.h>
#include <torch/csrc/utils/six.h>

#include <torch/csrc/autograd/grad_mode.h>

namespace torch {
namespace jit {
namespace python {

using namespace torch::autograd;
using namespace at;

// Alphabet used to describe structure of inputs/outputs (D for desc)
namespace D {
static constexpr char ListOpen = '[';
static constexpr char ListClose = ']';
static constexpr char TupleOpen = '(';
static constexpr char TupleClose = ')';
static constexpr char Variable = 'v';
} // namespace D

namespace {

template <typename T>
py::object cast_handle_sequence(std::vector<py::handle> objs) {
  auto num_objs = objs.size();
  T sequence{num_objs};
  for (size_t i = 0; i < num_objs; ++i)
    sequence[i] = py::reinterpret_borrow<py::object>(objs[i]);
  return sequence;
}

void flatten_rec(PyObject* obj, ParsedArgs& args) {
  auto& structure = args.desc.structure;
  if (six::isTuple(obj)) {
    structure.push_back(D::TupleOpen);
    for (auto item : py::reinterpret_borrow<py::tuple>(obj))
      flatten_rec(item.ptr(), args);
    structure.push_back(D::TupleClose);
  } else if (PyList_Check(obj)) {
    structure.push_back(D::ListOpen);
    for (auto item : py::reinterpret_borrow<py::list>(obj))
      flatten_rec(item.ptr(), args);
    structure.push_back(D::ListClose);
  } else if (THPVariable_Check(obj)) {
    auto& var = reinterpret_cast<THPVariable*>(obj)->cdata;
    args.vars.push_back(var);
    args.desc.metadata.emplace_back(var);
    args.desc.structure.push_back(D::Variable);
  } else {
    std::string msg =
        "Only tuples, lists and Variables supported as JIT inputs, but got ";
    msg += THPUtils_typename(obj);
    throw std::runtime_error(msg);
  }
}

} // anonymous namespace

ParsedArgs flatten(py::handle obj) {
  ParsedArgs args;
  args.desc.grad_enabled = autograd::GradMode::is_enabled();
  flatten_rec(obj.ptr(), args);
  return args;
}

namespace {

template <typename T>
py::object cast_sequence(std::vector<py::object> objs) {
  auto num_objs = objs.size();
  T sequence{num_objs};
  for (size_t i = 0; i < num_objs; ++i)
    sequence[i] = std::move(objs[i]);
  return std::move(sequence);
}

py::object unflatten_rec(
    ArrayRef<Variable>::iterator& var_it,
    ArrayRef<Variable>::iterator& var_it_end,
    std::string::const_iterator& desc_it) {
  char type = *desc_it++;
  if (type == D::TupleOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::TupleClose)
      objs.push_back(unflatten_rec(var_it, var_it_end, desc_it));
    ++desc_it;
    return cast_sequence<py::tuple>(objs);
  } else if (type == D::ListOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::ListClose)
      objs.push_back(unflatten_rec(var_it, var_it_end, desc_it));
    ++desc_it;
    return cast_sequence<py::list>(objs);
  } else {
    if (var_it == var_it_end)
      throw std::runtime_error("Not enough Variables given to unflatten");
    auto var = *var_it++;
    return py::reinterpret_steal<py::object>(THPVariable_Wrap(var));
  }
}

} // anonymous namespace

PyObject* unflatten(ArrayRef<Variable> vars, const IODescriptor& desc) {
  // NB: We don't do correctness checking on descriptor.
  // It has to be a correct bytes object produced by unflatten.
  auto vars_it = vars.begin();
  auto vars_it_end = vars.end();
  auto desc_it = desc.structure.begin();
  auto output = unflatten_rec(vars_it, vars_it_end, desc_it);
  if (vars_it != vars_it_end)
    throw std::runtime_error("Too many Variables given to unflatten");
  return output.release().ptr();
}

} // namespace python
} // namespace jit
} // namespace torch
