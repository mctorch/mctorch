#pragma once

#include <torch/csrc/python_headers.h>
#include <c10/core/ScalarType.h>

namespace c10 {
struct Device;
}

namespace at {
struct Type;
class Tensor;
} // namespace at

namespace torch { namespace tensors {

// Initializes the Python tensor type objects: torch.FloatTensor,
// torch.DoubleTensor, etc. and binds them in their containing modules.
void initialize_python_bindings();

// Sets the concrete type constructed by calls to torch.Tensor() and most
// factory methods on the torch module.
void set_default_tensor_type(const at::Type& type, const at::ScalarType scalar_type);

// Same as set_default_tensor_type() but takes a PyObject*
void py_set_default_tensor_type(PyObject* type_obj);

// Same as py_set_default_tensor_type, but only changes the dtype (ScalarType).
void py_set_default_dtype(PyObject* dtype_obj);

// Gets the ATen type object for the default tensor type. Note that the
// returned value will be a VariableType instance.
at::Type& get_default_tensor_type();

// Gets the ScalarType for the default tensor type.
at::ScalarType get_default_scalar_type();

}} // namespace torch::tensors
