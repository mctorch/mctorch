#include <Python.h>

#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/THP.h"

namespace torch { namespace nn {

using namespace torch::autograd;

using PythonHandle = THPObjectPtr;

struct TensorHandle : public PythonHandle {
  TensorHandle(PyObject* obj)
    : PythonHandle(obj) {
    if (!THPModule_isTensor(obj))
      throw std::runtime_error(std::string("expected a Tensor, but got ") + Py_TYPE(obj)->tp_name);
    tensor = createTensor(obj);
  }

  at::Tensor tensor;
};

struct VariableHandle : public PythonHandle {
  VariableHandle(PyObject* obj)
    : PythonHandle(obj) {
    if (!THPVariable_Check(obj))
      throw std::runtime_error(std::string("expected a Variable, but got ") + Py_TYPE(obj)->tp_name);
    var = ((THPVariable*)obj)->cdata;
  }

  Variable var;
};

struct Module;

struct ModuleHandle : public PythonHandle {
  // Can't define yet, because Module is only forward declared at this point.
  ModuleHandle(PyObject* obj);

  Module* module;
};

template<typename HandleType>
struct CallbackDict {
  using value_type = HandleType;
  using key_type = std::string;
  using elems_type = std::unordered_map<key_type, value_type>;

  CallbackDict() {}
  CallbackDict(const CallbackDict&) = delete;
  CallbackDict(CallbackDict&&) = delete;

  value_type& at(const key_type& key) {
    return elems_.at(key);
  }

  void set(key_type&& key, value_type&& value) {
    // TODO: callback
    auto it = elems_.find(key);
    if (it == elems_.end()) {
      elems_.emplace(key, std::move(value));
      order_.emplace_back(std::move(key));
    } else {
      it->second = std::move(value);
    }
  }

  // TODO: don't store keys twice - they could only be stored in the list,
  // but then the map needs a custom comparator/hasher (that dereferences pointers)
  std::unordered_map<key_type, value_type> elems_;
  std::list<key_type> order_;
};

struct Module {
  Module() {}
  Module(const Module&) = delete;
  Module(Module&&) = delete;
  CallbackDict<ModuleHandle> modules;
  CallbackDict<VariableHandle> parameters;
  CallbackDict<TensorHandle> buffers;
};

ModuleHandle::ModuleHandle(PyObject *obj)
  : PythonHandle(obj) {
  module = py::cast<Module*>(py::handle(obj));
}

////////////////////////////////////////////////////////////////////////////////
// Python bindings
////////////////////////////////////////////////////////////////////////////////

template<typename T, typename Impl>
struct iterator {
  using dict_type = CallbackDict<T>;

  iterator(dict_type& dict)
    : dict(dict)
    , key_it(dict.order_.begin()) {}

  static void initPythonBindings(py::module& m, const char *name, const char *iter_kind) {
    auto iterator_name = std::string(name) + iter_kind + "Iterator";
    char *c_iterator_name = new char[iterator_name.length() + 1];
    std::memcpy(c_iterator_name, iterator_name.c_str(), iterator_name.length() + 1);
    py::class_<iterator>(m, c_iterator_name)
      .def("__iter__", [](py::handle self) { return self; })
      .def("__next__", [](iterator& self) { return Impl::next(self.dict, self.key_it); })
      .def("__len__", [](iterator& self) { return self.dict.order_.size(); });
  }

  dict_type& dict;
  std::list<std::string>::iterator key_it;
};


template<typename T>
void initPythonCallbackDict(py::module& m, const char *name) {
  using dict_type = CallbackDict<T>;
  using key_iterator = std::list<std::string>::iterator;

  struct ItemsIterator {
    static py::object next(dict_type& dict, key_iterator& key_it) {
      if (key_it == dict.order_.end())
        throw py::stop_iteration();
      auto& key = *key_it++;
      auto& value = dict.elems_.at(key);
      auto py_value = py::reinterpret_borrow<py::object>(value.get());
      return py::cast(std::make_pair(key, py_value));
    }
  };
  struct KeysIterator {
    static py::object next(dict_type& dict, key_iterator& key_it) {
      if (key_it == dict.order_.end())
        throw py::stop_iteration();
      return py::cast(*key_it++);
    }
  };
  struct ValuesIterator {
    static py::object next(dict_type& dict, key_iterator& key_it) {
      if (key_it == dict.order_.end())
        throw py::stop_iteration();
      return py::reinterpret_borrow<py::object>(dict.elems_.at(*key_it++).get());
    }
  };
  iterator<T, ItemsIterator>::initPythonBindings(m, name, "Items");
  iterator<T, KeysIterator>::initPythonBindings(m, name, "Keys");
  iterator<T, ValuesIterator>::initPythonBindings(m, name, "Values");

  py::class_<dict_type>(m, name)
    .def(py::init<>())
    .def("__getitem__", [](dict_type& d, const char* name) -> py::object {
      return py::reinterpret_borrow<py::object>(d.at(name).get());
    }, py::return_value_policy::move)
    .def("__setitem__", [](dict_type& d, const char* name, py::object obj) {
      d.set(name, obj.release().ptr());
    })
    .def("__getstate__", [](dict_type& d) {
      std::unordered_map<std::string, py::object> simple_elems;
      for (auto& item : d.elems_)
        simple_elems[item.first] = py::reinterpret_borrow<py::object>(item.second.get());
      auto order = d.order_;
      return std::make_pair(std::move(order), std::move(simple_elems));
    }, py::return_value_policy::move)
    .def("__setstate__", [](dict_type& d, py::tuple state) {
      for (auto& item : state[0].cast<py::list>())
        d.order_.emplace_back(py::cast<std::string>(item));
      for (auto& item : state[1].cast<py::dict>())
        d.elems_.emplace(py::cast<std::string>(item.first), item.second.inc_ref().ptr());
    })
    .def("items", [](dict_type& d) {
      return iterator<T, ItemsIterator>(d);
    }, py::return_value_policy::move, py::keep_alive<0, 1>())
    .def("values", [](dict_type& d) {
      return iterator<T, ValuesIterator>(d);
    }, py::return_value_policy::move, py::keep_alive<0, 1>())
    .def("keys", [](dict_type& d) {
      return iterator<T, KeysIterator>(d);
    }, py::return_value_policy::move, py::keep_alive<0, 1>());
}

void initPythonModule(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();
  initPythonCallbackDict<ModuleHandle>(m, "ModuleCallbackDict");
  initPythonCallbackDict<VariableHandle>(m, "VariableCallbackDict");
  initPythonCallbackDict<TensorHandle>(m, "TensorCallbackDict");
  py::class_<Module>(m, "ModuleBase")
    .def(py::init<>())
    .def_readonly("_modules", &Module::modules)
    .def_readonly("_parameters", &Module::parameters)
    .def_readonly("_buffers", &Module::buffers)
    .def("__getstate__", [](Module& self) {
      auto reference = py::return_value_policy::reference;
      auto getstate = [](const py::object& obj) -> py::object {
        return py::getattr(obj, "__getstate__")();
      };
      return std::make_tuple(getstate(py::cast(self.modules, reference)),
                             getstate(py::cast(self.parameters, reference)),
                             getstate(py::cast(self.buffers, reference)));
    }, py::return_value_policy::move)
    .def("__setstate__", [](Module& self, py::tuple t) {
      auto reference = py::return_value_policy::reference;
      py::cast(self.modules, reference).attr("__setstate__")(t[0]);
      py::cast(self.parameters, reference).attr("__setstate__")(t[1]);
      py::cast(self.buffers, reference).attr("__setstate__")(t[2]);
    });
}

}} // namespace torch::nn
