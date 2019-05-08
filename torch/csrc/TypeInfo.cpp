#include <torch/csrc/TypeInfo.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

#include <c10/util/Exception.h>

#include <structmember.h>
#include <cstring>
#include <limits>
#include <sstream>

PyObject* THPFInfo_New(const at::ScalarType& type) {
  auto finfo = (PyTypeObject*)&THPFInfoType;
  auto self = THPObjectPtr{finfo->tp_alloc(finfo, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPDTypeInfo*>(self.get());
  self_->type = type;
  return self.release();
}

PyObject* THPIInfo_New(const at::ScalarType& type) {
  auto iinfo = (PyTypeObject*)&THPIInfoType;
  auto self = THPObjectPtr{iinfo->tp_alloc(iinfo, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPDTypeInfo*>(self.get());
  self_->type = type;
  return self.release();
}

PyObject* THPFInfo_str(THPFInfo* self) {
  std::ostringstream oss;
  oss << "finfo(type=" << self->type << ")";
  return THPUtils_packString(oss.str().c_str());
}

PyObject* THPIInfo_str(THPIInfo* self) {
  std::ostringstream oss;
  oss << "iinfo(type=" << self->type << ")";
  return THPUtils_packString(oss.str().c_str());
}

PyObject* THPFInfo_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
      "finfo(ScalarType type)",
      "finfo()",
  });

  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  AT_CHECK(r.idx < 2, "Not a type");
  at::ScalarType scalar_type;
  if (r.idx == 1) {
    scalar_type = torch::tensors::get_default_scalar_type();
    // The default tensor type can only be set to a floating point type/
    AT_ASSERT(at::isFloatingType(scalar_type));
  } else {
    scalar_type = r.scalartype(0);
    if (!at::isFloatingType(scalar_type)) {
      return PyErr_Format(
          PyExc_TypeError,
          "torch.finfo() requires a floating point input type. Use torch.iinfo to handle '%s'",
          type->tp_name);
    }
  }
  return THPFInfo_New(scalar_type);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIInfo_pynew(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
      "iinfo(ScalarType type)",
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  AT_CHECK(r.idx == 0, "Not a type");

  at::ScalarType scalar_type = r.scalartype(0);
  if (!at::isIntegralType(scalar_type)) {
    return PyErr_Format(
        PyExc_TypeError,
        "torch.iinfo() requires an integer input type. Use torch.finfo to handle '%s'",
        type->tp_name);
  }
  return THPIInfo_New(scalar_type);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDTypeInfo_compare(THPDTypeInfo* a, THPDTypeInfo* b, int op) {
  switch (op) {
    case Py_EQ:
      if (a->type == b->type) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    case Py_NE:
      if (a->type != b->type) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
  }
  return Py_INCREF(Py_NotImplemented), Py_NotImplemented;
}

static PyObject* THPDTypeInfo_bits(THPDTypeInfo* self, void*) {
  int bits = elementSize(self->type) * 8;
  return THPUtils_packInt64(bits);
}

static PyObject* THPFInfo_eps(THPFInfo* self, void*) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      self->type, "epsilon", [] {
        return PyFloat_FromDouble(
            std::numeric_limits<
                at::scalar_value_type<scalar_t>::type>::epsilon());
      });
}

static PyObject* THPFInfo_max(THPFInfo* self, void*) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self->type, "max", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::max());
  });
}

static PyObject* THPFInfo_min(THPFInfo* self, void*) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self->type, "min", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::lowest());
  });
}

static PyObject* THPIInfo_max(THPFInfo* self, void*) {
  return AT_DISPATCH_INTEGRAL_TYPES(self->type, "max", [] {
    return THPUtils_packInt64(std::numeric_limits<scalar_t>::max());
  });
}

static PyObject* THPIInfo_min(THPFInfo* self, void*) {
  return AT_DISPATCH_INTEGRAL_TYPES(self->type, "min", [] {
    return THPUtils_packInt64(std::numeric_limits<scalar_t>::lowest());
  });
}

static PyObject* THPFInfo_tiny(THPFInfo* self, void*) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self->type, "min", [] {
    return PyFloat_FromDouble(
        std::numeric_limits<at::scalar_value_type<scalar_t>::type>::min());
  });
}

static struct PyGetSetDef THPFInfo_properties[] = {
    {"bits", (getter)THPDTypeInfo_bits, nullptr, nullptr, nullptr},
    {"eps", (getter)THPFInfo_eps, nullptr, nullptr, nullptr},
    {"max", (getter)THPFInfo_max, nullptr, nullptr, nullptr},
    {"min", (getter)THPFInfo_min, nullptr, nullptr, nullptr},
    {"tiny", (getter)THPFInfo_tiny, nullptr, nullptr, nullptr},
    {nullptr}};

static PyMethodDef THPFInfo_methods[] = {
    {nullptr} /* Sentinel */
};

PyTypeObject THPFInfoType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.finfo", /* tp_name */
    sizeof(THPFInfo), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    nullptr, /* tp_print */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPFInfo_str, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    (reprfunc)THPFInfo_str, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    (richcmpfunc)THPDTypeInfo_compare, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPFInfo_methods, /* tp_methods */
    nullptr, /* tp_members */
    THPFInfo_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPFInfo_pynew, /* tp_new */
};

static struct PyGetSetDef THPIInfo_properties[] = {
    {"bits", (getter)THPDTypeInfo_bits, nullptr, nullptr, nullptr},
    {"max", (getter)THPIInfo_max, nullptr, nullptr, nullptr},
    {"min", (getter)THPIInfo_min, nullptr, nullptr, nullptr},
    {nullptr}};

static PyMethodDef THPIInfo_methods[] = {
    {nullptr} /* Sentinel */
};

PyTypeObject THPIInfoType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.iinfo", /* tp_name */
    sizeof(THPIInfo), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    nullptr, /* tp_print */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPIInfo_str, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    (reprfunc)THPIInfo_str, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    (richcmpfunc)THPDTypeInfo_compare, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPIInfo_methods, /* tp_methods */
    nullptr, /* tp_members */
    THPIInfo_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPIInfo_pynew, /* tp_new */
};

void THPDTypeInfo_init(PyObject* module) {
  if (PyType_Ready(&THPFInfoType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPFInfoType);
  if (PyModule_AddObject(module, "finfo", (PyObject*)&THPFInfoType) != 0) {
    throw python_error();
  }
  if (PyType_Ready(&THPIInfoType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPIInfoType);
  if (PyModule_AddObject(module, "iinfo", (PyObject*)&THPIInfoType) != 0) {
    throw python_error();
  }
}
