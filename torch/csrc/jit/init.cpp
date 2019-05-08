#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/canonicalize_ops.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/erase_number_types.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/inline_fork_wait.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <torch/csrc/jit/passes/onnx/fixup_onnx_loop.h>
#include <torch/csrc/jit/passes/onnx/peephole.h>
#include <torch/csrc/jit/passes/onnx/prepare_division_for_onnx.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/quantization.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/utils/check_alias_annotation.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/jit/python_arg_flatten.h>
#include <torch/csrc/jit/python_ir.h>
#include <torch/csrc/jit/python_tracer.h>
#include <torch/csrc/jit/script/compiler.h>
#include <torch/csrc/jit/script/init.h>
#include <torch/csrc/jit/script/jit_exception.h>
#include <torch/csrc/jit/script/python_tree_views.h>
#include <torch/csrc/jit/tracer.h>

#include <c10/macros/Export.h>
#include <caffe2/serialize/inline_container.h>

#include <ATen/core/function_schema.h>

#include <pybind11/functional.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace torch {
namespace jit {

using ::c10::Argument;
using ::c10::FunctionSchema;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;

namespace {

using autograd::variable_list;

bool loadPythonClasses() {
  // Leaving this code here, because it will likely be useful at some point
  // PyObject *jit_module = PyImport_ImportModule("torch.jit");
  // THPUtils_assert(jit_module, "class loader couldn't access "
  //"torch.jit module");
  // PyObject *jit_dict = PyModule_GetDict(jit_module);

  return true;
}
} // anonymous namespace

#if defined(_WIN32)
void runJITCPPTests(bool runCuda) {
  AT_ERROR("JIT tests not yet supported on Windows");
}
#else
CAFFE2_API void runJITCPPTests(bool runCuda);
#endif

void initJITBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::register_exception<JITException>(m, "JITException");

  py::class_<python::IODescriptor> iodescriptor(
      m, "IODescriptor"); // NOLINT(bugprone-unused-raii)

  m.def("_jit_init", loadPythonClasses)
      .def(
          "_jit_debug_fuser_num_cached_kernel_specs",
          torch::jit::fuser::debugNumCachedKernelSpecs)
      .def("_jit_pass_onnx", ToONNX)
      .def("_jit_pass_lower_all_tuples", LowerAllTuples)
      .def("_jit_pass_onnx_peephole", PeepholeOptimizeONNX)
      .def(
          "_jit_pass_onnx_constant_fold",
          [](std::shared_ptr<Graph>& graph,
             std::map<std::string, at::Tensor>& paramsDict) {
            ConstantFoldONNX(graph->block(), paramsDict); // overload resolution
            return paramsDict;
          },
          pybind11::return_value_policy::move)
      .def("_jit_pass_fuse", FuseGraph)
      .def(
          "_jit_pass_dce",
          [](std::shared_ptr<Graph>& g) {
            return EliminateDeadCode(g->block()); // overload resolution
          })
      .def(
          "_jit_pass_cse",
          [](std::shared_ptr<Graph>& g) {
            return EliminateCommonSubexpression(g); // overload resolution
          })
      .def(
          "_jit_pass_propagate_qinfo",
          [](std::shared_ptr<Graph>& g) { return PropagateQuantInfo(g); })
      .def(
          "_jit_pass_insert_observers",
          [](std::shared_ptr<Graph>& g, py::function pyObserverFunction) {
            // Create a new node that would be used in the insert observer pass:
            // all observer nodes will be cloned from this one.
            Node* new_node = g->createPythonOp(
                THPObjectPtr(pyObserverFunction.release().ptr()), "dd", {});
            InsertObserverNodes(g, new_node);
            // We don't need this node anymore, don't forget to remove it.
            new_node->destroy();
          })
      .def(
          "_jit_pass_insert_quantdequant",
          [](std::shared_ptr<Graph>& g) { return InsertQuantDequantNodes(g); })
      .def(
          "_jit_pass_quantlint",
          [](std::shared_ptr<Graph>& g) { return QuantLinting(g); })
      .def(
          "_jit_pass_fold_quant_inputs",
          [](std::shared_ptr<Graph>& g) {
            return FoldQuantNodesIntoInputsOutputs(g);
          })
      .def(
          "_jit_pass_remove_inplace_ops",
          [](std::shared_ptr<Graph> g) { return RemoveInplaceOps(g); })
      .def("_jit_pass_constant_pooling", ConstantPooling)
      .def(
          "_jit_pass_peephole",
          [](const std::shared_ptr<Graph>& g, bool addmm_fusion_enabled) {
            return PeepholeOptimize(g, addmm_fusion_enabled);
          },
          py::arg("graph"),
          py::arg("addmm_fusion_enabled") = false)
      .def(
          "_jit_pass_canonicalize",
          [](const std::shared_ptr<Graph>& g) { return Canonicalize(g); })
      .def("_jit_pass_lint", LintGraph)
      .def(
          "_jit_pass_complete_shape_analysis",
          [](std::shared_ptr<Graph> graph, py::tuple inputs, bool with_grad) {
            CompleteArgumentSpec spec(
                with_grad,
                evilDeprecatedBadCreateStackDoNotUse(inputs, graph->inputs()));
            auto graph_inputs = graph->inputs();
            AT_ASSERT(spec.size() == graph_inputs.size());
            for (size_t i = 0; i < graph_inputs.size(); ++i) {
              graph_inputs[i]->setType(spec.at(i));
            }
            PropagateInputShapes(graph);
          })
      .def("_jit_pass_remove_expands", RemoveExpands)
      .def("_jit_pass_erase_number_types", EraseNumberTypes)
      .def("_jit_pass_inline_fork_wait", InlineForkWait)
      .def("_jit_pass_prepare_division_for_onnx", PrepareDivisionForONNX)
      .def("_jit_pass_loop_unrolling", UnrollLoops)
      .def(
          "_jit_pass_constant_propagation",
          [](std::shared_ptr<Graph>& g) { return ConstantPropagation(g); })
      .def("_jit_pass_erase_shape_information", EraseShapeInformation)
      .def(
          "_jit_pass_create_autodiff_subgraphs",
          [](std::shared_ptr<Graph> graph) { CreateAutodiffSubgraphs(graph); })
      .def(
          "_jit_run_cpp_tests",
          [](bool runCuda) {
            // We have to release the GIL inside this method, because if we
            // happen to initialize the autograd engine in these tests, the
            // newly spawned worker threads will try to initialize their
            // PyThreadState*, and they need the GIL for this.
            AutoNoGIL _no_gil;
            return runJITCPPTests(runCuda);
          },
          py::arg("run_cuda"))
      .def(
          "_jit_flatten",
          [](py::handle& obj) {
            auto res = python::flatten(obj);
            return std::make_pair(res.vars, res.desc);
          })
      .def(
          "_jit_unflatten",
          [](autograd::variable_list vars, python::IODescriptor& desc) {
            return py::reinterpret_steal<py::object>(
                python::unflatten(vars, desc));
          })
      .def("_jit_pass_onnx_block", BlockToONNX)
      .def("_jit_pass_fixup_onnx_loops", FixupONNXLoops)
      .def("_jit_pass_canonicalize_ops", CanonicalizeOps)
      .def("_jit_pass_specialize_autogradzero", specializeAutogradZero)
      .def("_jit_override_can_fuse_on_cpu", &overrideCanFuseOnCPU)
      .def(
          "_jit_differentiate",
          [](Graph& g) {
            // the python binding slightly differs in semantics
            // it makes a copy of the input Graph, and works on that
            // jit::differentiate mutates the input Graph
            auto g_clone = g.copy();
            return differentiate(g_clone);
          })
      .def(
          "_jit_check_alias_annotation",
          [](std::shared_ptr<Graph> g,
             py::tuple args,
             const std::string& unqualified_op_name) {
            auto stack = toStack(args);
            checkAliasAnnotation(g, std::move(stack), unqualified_op_name);
          })
      .def(
          "_jit_fuser_get_fused_kernel_code",
          [](Graph& g, std::vector<at::Tensor> inps) {
            return debugGetFusedKernelCode(g, inps);
          });

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<CompleteArgumentSpec>(m, "CompleteArgumentSpec")
      .def("__repr__", [](CompleteArgumentSpec& self) {
        std::ostringstream s;
        s << self;
        return s.str();
      });
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ArgumentSpec>(m, "ArgumentSpec");
  py::class_<Code>(m, "Code").def("grad_executor_states", [](Code& c) {
    std::vector<GraphExecutorState> states;
    for (auto& e : c.grad_executors()) {
      states.emplace_back(e->getDebugState());
    }
    return states;
  });

  py::class_<ExecutionPlanState>(m, "ExecutionPlanState")
      .def_property_readonly(
          "graph", [](ExecutionPlanState& s) { return s.graph; })
      .def_property_readonly(
          "code", [](ExecutionPlanState& s) { return s.code; });

  py::class_<Gradient>(m, "Gradient")
      .def_property_readonly("f", [](Gradient& m) { return m.f; })
      .def_property_readonly("df", [](Gradient& m) { return m.df; })
      .def_property_readonly(
          "f_real_outputs", [](Gradient& m) { return m.f_real_outputs; })
      .def_property_readonly(
          "df_input_vjps", [](Gradient& m) { return m.df_input_vjps; })
      .def_property_readonly(
          "df_input_captured_inputs",
          [](Gradient& m) { return m.df_input_captured_inputs; })
      .def_property_readonly(
          "df_input_captured_outputs",
          [](Gradient& m) { return m.df_input_captured_outputs; })
      .def_property_readonly(
          "df_output_vjps", [](Gradient& m) { return m.df_output_vjps; });

  py::class_<GraphExecutorState>(m, "GraphExecutorState")
      .def_property_readonly(
          "graph", [](GraphExecutorState& s) { return s.graph; })
      .def_property_readonly(
          "execution_plans",
          [](GraphExecutorState& s) { return s.execution_plans; })
      .def_property_readonly(
          "fallback", [](GraphExecutorState& s) { return s.fallback; });

  py::class_<PyTorchStreamWriter>(m, "PyTorchFileWriter")
      .def(py::init<std::string>())
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             const char* data,
             size_t size) { return self.writeRecord(name, data, size); })
      .def("write_end_of_file", &PyTorchStreamWriter::writeEndOfFile);

  py::class_<PyTorchStreamReader>(m, "PyTorchFileReader")
      .def(py::init<std::string>())
      .def("get_record", [](PyTorchStreamReader& self, const std::string& key) {
        at::DataPtr data;
        size_t size;
        std::tie(data, size) = self.getRecord(key);
        return py::bytes(reinterpret_cast<const char*>(data.get()), size);
      });

  m.def(
      "_jit_get_operation",
      [](const std::string& qualified_name) {
        try {
          auto symbol = Symbol::fromQualString(qualified_name);
          auto operations = getAllOperatorsFor(symbol);
          AT_CHECK(!operations.empty(), "No such operator ", qualified_name);
          AT_CHECK(
              operations.size() == 1,
              "Found ",
              operations.size(),
              " overloads for operator ",
              qualified_name,
              "! Overloads are not supported from Python.");
          std::shared_ptr<Operator> op = operations[0];
          AT_ASSERT(op != nullptr);
          std::ostringstream docstring;
          docstring << "Automatically bound operator '" << qualified_name
                    << "' with schema: " << op->schema();
          return py::cpp_function(
              [op](py::args args, py::kwargs kwargs) {
                return invokeOperatorFromPython(
                    *op, std::move(args), std::move(kwargs));
              },
              py::name(qualified_name.c_str()),
              py::doc(docstring.str().c_str()));
        } catch (const c10::Error& error) {
          throw std::runtime_error(error.what_without_backtrace());
        }
      },
      py::arg("qualified_name"));

  m.def("parse_ir", [](const std::string& input) {
    auto graph = std::make_shared<Graph>();
    script::parseIR(input, &*graph);
    return graph;
  });

  py::class_<FunctionSchema>(m, "FunctionSchema")
      .def_property_readonly(
          "name", [](FunctionSchema& self) { return self.name(); })
      .def_property_readonly(
          "overload_name",
          [](FunctionSchema& self) { return self.overload_name(); })
      .def_property_readonly(
          "arguments", [](FunctionSchema& self) { return self.arguments(); })
      .def_property_readonly(
          "returns", [](FunctionSchema& self) { return self.returns(); })
      .def("__str__", [](FunctionSchema& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      });
  py::class_<Argument>(m, "Argument")
      .def_property_readonly("name", [](Argument& self) { return self.name(); })
      .def_property_readonly("type", [](Argument& self) { return self.type(); })
      .def_property_readonly(
          "N",
          [](Argument& self) -> py::object {
            return (self.N()) ? py::cast(*self.N()) : py::none();
          })
      .def_property_readonly("default_value", [](Argument& self) -> py::object {
        if (!self.default_value())
          return py::none();
        IValue v = *self.default_value();
        return toPyObject(std::move(v));
      });
  m.def("_jit_get_schemas_for_operator", [](const std::string& qualified_name) {
    auto symbol = Symbol::fromQualString(qualified_name);
    auto operations = getAllOperatorsFor(symbol);
    return fmap(operations, [](const std::shared_ptr<Operator>& op) {
      return op->schema();
    });
  });

  struct PythonFutureWrapper {
    explicit PythonFutureWrapper(c10::intrusive_ptr<c10::ivalue::Future> fut)
        : fut(std::move(fut)) {}

    c10::intrusive_ptr<c10::ivalue::Future> fut;
  };

  py::class_<PythonFutureWrapper>(m, "Future");

  m.def("fork", [](py::args args) {
    AT_ASSERT(args.size() >= 1);

    py::function f = py::cast<py::function>(args[0]);
    py::tuple args_tup(args.size() - 1);

    for (size_t i = 1; i < args.size(); ++i) {
      args_tup[i - 1] = args[i];
    }

    if (jit::tracer::isTracing()) {
      auto graph = jit::tracer::getTracingState()->graph;
      auto fork_node = graph->insertNode(graph->create(prim::fork, 1));
      auto body_block = fork_node->addBlock();

      Value* node_output;
      py::object py_func_output;
      auto retval = c10::make_intrusive<c10::ivalue::Future>();
      // Insert new trace ops into the fork op's sub-block
      WithInsertPoint guard(body_block);
      IValue output_ivalue;
      {
        tracer::WithNestedTracingFrame env_guard;

        // Run the user-supplied function
        py_func_output = f(*args_tup);

        // Convert the output of the user-supplied funciton to IValue. The type
        // information of this IValue is used both to record the correct type in
        // the trace.
        output_ivalue = toIValue(py_func_output);
        Value* out_val = jit::tracer::getNestedValueTrace(output_ivalue);
        body_block->registerOutput(out_val);
        node_output =
            fork_node->output()->setType(FutureType::create(out_val->type()));

        // Lambda lift into a Subgraph attribute
        torch::jit::script::lambdaLiftFork(fork_node);
      }

      // Record the ivalue in the tracer
      jit::tracer::setValueTrace(retval, node_output);

      // stuff the ivalue output in the Future
      retval->markCompleted(output_ivalue);

      return PythonFutureWrapper(retval);
    } else {
      auto retval = c10::make_intrusive<c10::ivalue::Future>();
      retval->markCompleted(toIValue(f(*args_tup)));
      return PythonFutureWrapper(retval);
    }
  });

  m.def("wait", [](PythonFutureWrapper& fut) {
    if (jit::tracer::isTracing()) {
      auto graph = jit::tracer::getTracingState()->graph;

      Value* fut_val = jit::tracer::getValueTrace(fut.fut);
      auto output = graph->insert(aten::wait, {fut_val});
      jit::tracer::setValueTrace(fut.fut->value(), output);
    }
    return fut.fut->value();
  });

  m.def("_jit_assert_is_instance", [](py::object obj, TypePtr type) {
    toIValue(obj, type);
  });

  initPythonIRBindings(module);
  tracer::initPythonTracerBindings(module);
  script::initTreeViewBindings(module);
  script::initJitScriptBindings(module);
}
} // namespace jit
} // namespace torch
