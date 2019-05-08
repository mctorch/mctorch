#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/Exceptions.h>

#include <iostream>

namespace torch { namespace autograd {

void PyAnomalyMetadata::store_stack() {
  AutoGIL gil;
  THPObjectPtr mod(PyImport_ImportModule("traceback"));
  if (!mod) {
    throw python_error();
  }

  THPObjectPtr list(PyObject_CallMethod(mod.get(), "format_stack", ""));
  if (!list) {
    throw python_error();
  }

  if (PyDict_SetItemString(dict(), ANOMALY_TRACE_KEY, list.get())) {
    throw python_error();
  }
}

void PyAnomalyMetadata::print_stack() {
  AutoGIL gil;
  if (!PyDict_Check(dict())) {
    throw std::runtime_error("Anomaly metadata is not a python dictionary.");
  }

  // PyDict_GetItemString returns a borrowed reference
  PyObject* stack(PyDict_GetItemString(dict(), ANOMALY_TRACE_KEY));
  if (!stack) {
    AT_WARN("No forward pass information available. Enable detect anomaly "
            "during forward pass for more information.");
    return;
  }

  THPObjectPtr empty_string(PyUnicode_FromString(""));
  if (!empty_string) {
    throw python_error();
  }

  // stack is a list of Python strings ending with newlines. Use join to convert
  // to a single string.
  THPObjectPtr msg(PyUnicode_Join(empty_string, stack));
  if (!msg) {
    throw python_error();
  }

  AT_WARN("Traceback of forward call that caused the error:\n",
          THPUtils_unpackString(msg.get()));
}

}}
