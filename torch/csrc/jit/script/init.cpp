#include "torch/csrc/jit/script/init.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch {
namespace jit {
namespace script {

    void initJitScriptBindings(PyObject* module) {
        auto m = py::handle(module).cast<py::module>();

        m.def("_jit_script_compile", jitScriptCompile);
        m.def("_jit_script_execute", jitScriptExecute)
    }

} // namespace script
} // namespace jit
} // namespace torch
