#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <torch/csrc/autograd/symbolic.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/onnx/onnx.h>

#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/import_export_helpers.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/python_print.h>
#include <torch/csrc/jit/pickler.h>

#include <caffe2/core/types.h>
#include <caffe2/proto/caffe2_pb.h>
#include <caffe2/proto/torch_pb.h>
#include <caffe2/serialize/inline_container.h>
#include <onnx/onnx_pb.h>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

namespace torch {
namespace jit {

namespace {
namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;

class ScriptModuleSerializer;

std::string getNodeStackTraceString(const Node* n) {
  std::stringstream ss;
  if (n->getSourceLocation()) {
    n->getSourceLocation()->highlight(ss);
  } else {
    ss << "<unknown location>";
  }
  return ss.str();
}

void validateBlock(
    Block* b,
    onnx_torch::OperatorExportTypes operator_export_type) {
  for (auto node : b->nodes()) {
    for (Block* sub_block : node->blocks()) {
      validateBlock(sub_block, operator_export_type);
    }
    // Macro'ed so we get a marginally better line number on failed export
#define FAIL_EXPORT(name)                          \
  throw std::runtime_error(                        \
      std::string("ONNX export failed: ") + name + \
      "\n\nGraph we tried to export:\n" + b->owningGraph()->toString());
    if (node->kind() == prim::PythonOp) {
      auto py_node = static_cast<PythonOp*>(node);
      FAIL_EXPORT(
          "Couldn't export Python operator " + py_node->name() +
          "\n\nDefined at:\n" + getNodeStackTraceString(node))
    } else {
      // Special error messages for certain types of operators
      if (node->kind() == aten::expand) {
        if (operator_export_type ==
            onnx_torch::OperatorExportTypes::ONNX_ATEN_FALLBACK) {
          WithInsertPoint guard(node);
          auto* new_node =
              b->owningGraph()->insertNode(b->owningGraph()->create(
                  Symbol(::c10::onnx::ATen),
                  node->inputs(),
                  node->outputs().size()));
          for (size_t i = 0; i < node->outputs().size(); ++i) {
            node->output(i)->replaceAllUsesWith(new_node->output(i));
          }
          new_node->s_(Symbol::fromQualString("attr::operator"), "expand");
        }
      }
      if (node->kind() == prim::PackPadded || node->kind() == prim::PadPacked) {
        FAIL_EXPORT(
            "Cannot export individual pack_padded_sequence or pad_packed_sequence; these operations must occur in pairs.\n\nUsage of this operation occurred at:\n" +
            getNodeStackTraceString(node));
      }
      bool is_aten_enabled = operator_export_type ==
              onnx_torch::OperatorExportTypes::ONNX_ATEN_FALLBACK ||
          operator_export_type == onnx_torch::OperatorExportTypes::ONNX_ATEN;
      if (!node->kind().is_onnx() && !node->kind().is_caffe2() &&
          !is_aten_enabled && !node->mustBeNone()) {
        FAIL_EXPORT(
            "Couldn't export operator " + node->kind().toDisplayString() +
            "\n\nDefined at:\n" + getNodeStackTraceString(node));
      }
    }
#undef FAIL_EXPORT
  }
}

void validateGraph(
    const std::shared_ptr<Graph>& graph,
    onnx_torch::OperatorExportTypes operator_export_type) {
  validateBlock(graph->block(), operator_export_type);
  EliminateDeadCode(graph->block());
}

class EncoderBase {
 public:
  EncoderBase(
      onnx_torch::OperatorExportTypes operator_export_type,
      bool strip_doc);

  onnx::ModelProto get_model_proto() {
    return model_proto_;
  }

 protected:
  // Using std::map instead of std::unordered_map for initializers
  // in EncodeGraph cosntructor so that the order in which initializers
  // get written to the ONNX graph is always the deterministic and
  // predictable. While this is not a ONNX requirement, it is needed
  // for testing purposes in tests that use _export_to_pretty_string()
  // for validating ONNX graphs.
  void EncodeGraph(
      onnx::GraphProto* graph_proto,
      const std::shared_ptr<Graph>& graph,
      const std::map<std::string, at::Tensor>& initializers =
          std::map<std::string, at::Tensor>());

  void EncodeBlock(
      onnx::GraphProto* graph_proto,
      const Block* block,
      const std::map<std::string, at::Tensor>& initializers =
          std::map<std::string, at::Tensor>());

  virtual void EncodeTensor(
      onnx::TensorProto* tensor_proto,
      const at::Tensor& tensor,
      const c10::optional<std::string> external_ref = {}) = 0;

  virtual void EncodeIntermediateValueInfo(
      onnx::GraphProto* graph_proto,
      const Value* n){};

  virtual void EncodeValueInfo(
      onnx::GraphProto* graph_proto,
      onnx::ValueInfoProto* v,
      const Value* n);

  void AddAttribute(
      onnx::NodeProto* node_proto,
      const jit::Node* node,
      const jit::Symbol name);

  onnx::ModelProto model_proto_;
  size_t num_blocks_;
  onnx_torch::OperatorExportTypes operator_export_type_;
  bool strip_doc_;
  std::set<std::string> domains_;
};

onnx::TensorProto_DataType ATenTypeToOnnxType(at::ScalarType at_type) {
  switch (at_type) {
    case at::kDouble:
      return onnx::TensorProto_DataType_DOUBLE;
    case at::kFloat:
      return onnx::TensorProto_DataType_FLOAT;
    case at::kHalf:
      return onnx::TensorProto_DataType_FLOAT16;
    case at::kByte:
      return onnx::TensorProto_DataType_UINT8;
    case at::kChar:
      return onnx::TensorProto_DataType_INT8;
    case at::kShort:
      return onnx::TensorProto_DataType_INT16;
    case at::kInt:
      return onnx::TensorProto_DataType_INT32;
    case at::kLong:
      return onnx::TensorProto_DataType_INT64;
    default:
      AT_ERROR("unexpected tensor scalar type");
  }
}

EncoderBase::EncoderBase(
    onnx_torch::OperatorExportTypes operator_export_type,
    bool strip_doc)
    : num_blocks_(0),
      operator_export_type_(operator_export_type),
      strip_doc_(strip_doc) {
  model_proto_.set_producer_name("pytorch");
  // we pin IR version to version 4 (01/22/2019) instead of using
  // onnx::IR_VERSION. with this change, the test_operators.py will be more
  // stable. only bump it when it's necessary
  model_proto_.set_ir_version(4);
  // TODO: set the producer version using appropriate function call
  model_proto_.set_producer_version("1.1");
}

void EncoderBase::EncodeValueInfo(
    onnx::GraphProto* graph_proto,
    onnx::ValueInfoProto* v,
    const Value* n) {
  v->set_name(n->uniqueName());
  onnx::TypeProto* t = v->mutable_type();
  onnx::TypeProto_Tensor* tensor_type = t->mutable_tensor_type();

  onnx::TensorShapeProto* shape = tensor_type->mutable_shape();
  if (CompleteTensorTypePtr node_type = n->type()->cast<CompleteTensorType>()) {
    const std::vector<std::int64_t>& sizes = node_type->sizes();
    for (size_t i = 0; i < sizes.size(); i++) {
      shape->add_dim();
      shape->mutable_dim(i)->set_dim_value(sizes[i]);
    }
    tensor_type->set_elem_type(ATenTypeToOnnxType(node_type->scalarType()));
  } else {
    tensor_type->set_elem_type(onnx::TensorProto_DataType_UNDEFINED);
  }
}

void EncoderBase::EncodeGraph(
    onnx::GraphProto* graph_proto,
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers) {
  EncodeBlock(graph_proto, graph->block(), initializers);
}

void EncoderBase::EncodeBlock(
    onnx::GraphProto* graph_proto,
    const Block* block,
    const std::map<std::string, at::Tensor>& initializers) {
  AT_ASSERT(graph_proto != nullptr);
  std::string block_name = "torch-jit-export";
  if (num_blocks_) {
    block_name += std::to_string(num_blocks_);
  }
  num_blocks_++;
  graph_proto->set_name(block_name);

  for (auto input : block->inputs()) {
    onnx::ValueInfoProto* v = graph_proto->add_input();
    EncodeValueInfo(graph_proto, v, input);
  }
  for (auto output : block->outputs()) {
    onnx::ValueInfoProto* v = graph_proto->add_output();
    EncodeValueInfo(graph_proto, v, output);
  }
  for (auto node : block->nodes()) {
    bool is_raw_export =
        operator_export_type_ == onnx_torch::OperatorExportTypes::RAW;
    if (node->mustBeNone() && !is_raw_export) {
      // None nodes are used to implement optional inputs. One
      // way to "not provide" an optional input is to create an
      // Undefined node, and pass its output as that input.
      continue;
    }
    auto p_n = graph_proto->add_node();
    if (node->getSourceLocation() && !strip_doc_) {
      std::stringstream ss;
      node->getSourceLocation()->highlight(ss);
      p_n->set_doc_string(ss.str());
    }
    for (auto input : node->inputs()) {
      if (input->node()->mustBeNone() && !is_raw_export) {
        p_n->add_input("");
      } else {
        p_n->add_input(input->uniqueName());
      }
    }
    for (auto output : node->outputs()) {
      p_n->add_output(output->uniqueName());
      EncodeIntermediateValueInfo(graph_proto, output);
    }
    if (!node->kind().is_onnx()) {
      p_n->set_domain(node->kind().domainString());
      domains_.insert(node->kind().domainString());
    }
    if (is_raw_export) {
      AT_ASSERT(!node->kind().is_onnx());
    } else if (operator_export_type_ == onnx_torch::OperatorExportTypes::ONNX) {
      AT_ASSERT(
          !node->kind().is_aten() && !node->kind().is_prim() &&
          !node->kind().is_attr());
    }
    p_n->set_op_type(node->kind().toUnqualString());
    for (auto attr_name : node->attributeNames()) {
      AddAttribute(p_n, node, attr_name);
    }
    if (is_raw_export && node->blocks().size() > 0) {
      auto blocks = p_n->add_attribute();
      blocks->set_name("_blocks");
      blocks->set_type(onnx::AttributeProto_AttributeType_GRAPHS);
      for (auto block : node->blocks()) {
        auto graph = blocks->add_graphs();
        EncodeBlock(graph, block, initializers);
      }
    }
    if (node->kind() == ::c10::onnx::Loop) {
      AT_ASSERT(node->blocks().size() == 1);

      auto body = p_n->add_attribute();
      body->set_name("body");
      body->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto g = body->mutable_g();
      EncodeBlock(g, node->blocks()[0]);
    }
    if (node->kind() == ::c10::onnx::If) {
      AT_ASSERT(node->blocks().size() == 2);

      auto true_branch = p_n->add_attribute();
      true_branch->set_name("then_branch");
      true_branch->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto true_g = true_branch->mutable_g();
      EncodeBlock(true_g, node->blocks()[0]);

      auto false_branch = p_n->add_attribute();
      false_branch->set_name("else_branch");
      false_branch->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto false_g = false_branch->mutable_g();
      EncodeBlock(false_g, node->blocks()[1]);
    }
  }
  AT_ASSERT(block->inputs().size() >= initializers.size());
  for (auto& name_tensor_pair : initializers) {
    auto p = graph_proto->add_initializer();
    p->set_name(name_tensor_pair.first);
    EncodeTensor(p, name_tensor_pair.second, name_tensor_pair.first);
  }
}

void EncoderBase::AddAttribute(
    onnx::NodeProto* node_proto,
    const jit::Node* node,
    const jit::Symbol name) {
  auto attr = node_proto->add_attribute();
  AT_ASSERT(name.is_attr());
  attr->set_name(name.toUnqualString());
  switch (node->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(node->f(name));
      attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
      break;
    case AttributeKind::fs:
      attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
      for (auto& v : node->fs(name))
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      attr->set_type(onnx::AttributeProto_AttributeType_INT);
      attr->set_i(node->i(name));
      break;
    case AttributeKind::is:
      attr->set_type(onnx::AttributeProto_AttributeType_INTS);
      for (auto& v : node->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      attr->set_s(node->s(name));
      break;
    case AttributeKind::ss:
      attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      for (auto& v : node->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
      auto t = attr->mutable_t();
      EncodeTensor(t, node->t(name));
    } break;
    case AttributeKind::ts:
      attr->set_type(onnx::AttributeProto_AttributeType_TENSORS);
      for (auto& v : node->ts(name)) {
        auto t = attr->add_tensors();
        EncodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto g = attr->mutable_g();
      EncodeGraph(g, node->g(name));
    } break;
    case AttributeKind::gs:
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPHS);
      for (auto& v : node->gs(name)) {
        auto g = attr->add_graphs();
        EncodeGraph(g, v);
      }
      break;
    default:
      throw std::runtime_error("unexpected attribute kind");
  }
}

class GraphEncoder : public EncoderBase {
 public:
  GraphEncoder(
      const std::shared_ptr<Graph>& graph,
      int64_t onnx_opset_version,
      onnx_torch::OperatorExportTypes operator_export_type,
      const std::map<std::string, at::Tensor>& initializers,
      bool defer_weight_export,
      bool strip_doc);

  RawDataExportMap get_raw_data_export_map() {
    return raw_data_export_map_;
  }

 private:
  void EncodeTensor(
      onnx::TensorProto* tensor_proto,
      const at::Tensor& tensor,
      const c10::optional<std::string> external_ref = {}) override;

  RawDataExportMap raw_data_export_map_;
  bool defer_weight_export_;
};

GraphEncoder::GraphEncoder(
    const std::shared_ptr<Graph>& graph,
    int64_t onnx_opset_version,
    onnx_torch::OperatorExportTypes operator_export_type,
    const std::map<std::string, at::Tensor>& initializers,
    bool defer_weight_export,
    bool strip_doc)
    : EncoderBase(operator_export_type, strip_doc),
      defer_weight_export_(defer_weight_export) {
  if (operator_export_type != onnx_torch::OperatorExportTypes::RAW) {
    validateGraph(graph, operator_export_type);
  }

  auto* imp = model_proto_.add_opset_import();
  // This is the version of ONNX operator set we are targeting
  imp->set_version(onnx_opset_version);

  EncodeGraph(model_proto_.mutable_graph(), graph, initializers);

  for (const std::string& domain : domains_) {
    auto* opset = model_proto_.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(0);
  }
}

void GraphEncoder::EncodeTensor(
    onnx::TensorProto* tensor_proto,
    const at::Tensor& tensor,
    const c10::optional<std::string> external_ref) {
  for (auto d : tensor.sizes()) {
    tensor_proto->add_dims(d);
  }
  tensor_proto->set_data_type(ATenTypeToOnnxType(tensor.scalar_type()));
  // CPU's HalfTensor doesn't have contiguous(), so first calling contiguous()
  auto t = tensor.contiguous().cpu();
  // Add a buffer to the raw_data_export_map for the caller to dump into an
  // external data store. If external_ref is not specified, we instead dump
  // the contiguous data into the protobuf itself
  if (defer_weight_export_ && external_ref) {
    // For now, we use the name of the tensor as the external lookup name to
    // avoid ONNX protobuf changes.
    AT_ASSERT(external_ref.value() == tensor_proto->name());
    AT_ASSERT(raw_data_export_map_.count(external_ref.value()) == 0);
    raw_data_export_map_[external_ref.value()] = t;
    tensor_proto->set_raw_data("__EXTERNAL");
  } else {
    AT_ASSERT(t.is_contiguous());
    tensor_proto->set_raw_data(std::string(
        static_cast<char*>(t.data_ptr()), t.element_size() * t.numel()));
  }
}

// this is a serializer class which saves script modules to pt files. the
// content of the file is written using PyTorchStreamWriter, for details please
// check caffe2/serialize/inline_container.h. all the records except the last
// one are tensor data, and the last record is a serialized ModelProto, defined
// in caffe2/proto/torch.proto. ModelProto contains all the metadata of the
// model, and it is serialized as json.
class ScriptModuleSerializer final {
 public:
  ScriptModuleSerializer(const std::string& filename);

  ScriptModuleSerializer(std::ostream* ofs);

  void serialize(
      const script::Module& module,
      const script::ExtraFilesMap& extra_files = script::ExtraFilesMap());

 private:
  void convertModel(
      const script::Module& module,
      torch::ModelDef* model_def,
      const script::ExtraFilesMap& extra_files);

  // add a tensor to the tensorTable
  // returns the offset into the tensor table
  size_t addTensor(const at::Tensor& tensor);

  // write the content of the tensor to the file/stream, and save the
  // offset in the storageMap_
  void convertAndWriteTensor(
      size_t tensor_id,
      const at::Tensor& tensor,
      torch::TensorDef* tensor_proto,
      std::unordered_map<const void*, std::string>& storageMap);

  // dump all the tensors in the tensorTable_ to a ModelDef (metadata) and
  // the file/stream (the content), assuming all the information of the
  // tensors has been collected. the method calls convertAndWriteTensor
  // to dump the content of a tensor
  void writeTensorTable(torch::ModelDef* model_def);

  void writeAttributeTable();
  void writeLibs(torch::ModelDef* model_def);

  void convertModule(
      const script::Module& module,
      const std::string& prefix,
      const std::string& name,
      torch::ModuleDef* module_def);

  void convertParameter(
      const script::Slot& param,
      torch::ParameterDef* param_def,
      bool is_parameter);

  void convertClass(const ClassTypePtr& type, torch::ModelDef* model_def);

  std::ofstream ofs_;
  caffe2::serialize::PyTorchStreamWriter writer_;

  // all tensors that will be stored
  std::vector<at::Tensor> tensor_table_;

  std::vector<IValue> attribute_table_;

  // all classes used by this module hierarchy
  std::vector<ClassTypePtr> class_table_;
  OrderedDict<ClassTypePtr, std::string> converted_classes_;
  std::unordered_map<ClassTypePtr, std::vector<ClassTypePtr>> class_to_deps_;

  static const size_t op_version_set = 0;
};

// ScriptModuleSerializer's methods
ScriptModuleSerializer::ScriptModuleSerializer(const std::string& filename)
    : writer_(filename.c_str()) {
  // TODO appropriate support for mmap, right now we still use stream writer
}

ScriptModuleSerializer::ScriptModuleSerializer(std::ostream* ofs)
    : ofs_(), writer_(ofs) {}

void ScriptModuleSerializer::serialize(
    const script::Module& module,
    const script::ExtraFilesMap& extra_files) {
  torch::ModelDef model_def;
  convertModel(module, &model_def, extra_files);
  std::string output;
  // NB: cannot use MessageToJsonString, since fbcode's protobuf is too old
  // be consistent with MessageToJsonString
  std::string url_prefix = "type.googleapis.com";
  std::unique_ptr<::google::protobuf::util::TypeResolver> resolver(
      ::google::protobuf::util::NewTypeResolverForDescriptorPool(
          url_prefix, model_def.GetDescriptor()->file()->pool()));
  ::google::protobuf::util::Status convert_result =
      ::google::protobuf::util::BinaryToJsonString(
          resolver.get(),
          url_prefix + "/" + model_def.GetDescriptor()->full_name(),
          model_def.SerializeAsString(),
          &output);
  if (!convert_result.ok()) {
    std::stringstream ss;
    ss << convert_result;
    AT_ERROR(ss.str());
  }
  writer_.writeRecord("model.json", output.data(), output.size());
  writer_.writeEndOfFile();
}

void ScriptModuleSerializer::writeLibs(torch::ModelDef* model_def) {
  // Convert all the classes that this model depends on
  for (const auto& class_type : class_table_) {
    convertClass(class_type, model_def);
  }

  // Mapping of filename => src. We need this because multiple clases may go in
  // the same file (e.g. foo.bar.Baz and foo.bar.Qux)

  // Aggregate classes into files by their qualified names
  std::unordered_map<std::string, std::ostringstream> fileToSrc;
  for (const auto& item : converted_classes_) {
    const auto& class_type = item.key();
    const auto& class_src = item.value();

    // For the type, foo.bar.Baz
    const std::string filename =
        ImportExportHelpers::qualifierToPath(class_type->qualifier());
    // End state: filename is "foo/bar.py", in which we will define a class
    // named Baz
    fileToSrc[filename] << class_src;
  }

  // Write out the files. We still have to do this in converted_classes_ order,
  // to maintain dependency order.
  for (const auto& item : converted_classes_) {
    const ClassTypePtr& class_type = item.key();
    const std::string filename =
        ImportExportHelpers::qualifierToPath(class_type->qualifier());
    const std::string& src = fileToSrc.at(filename).str();

    std::ostringstream lib_stream;
    lib_stream << "op_version_set = " << op_version_set << "\n";
    lib_stream << src;
    std::string lib_str = lib_stream.str();
    writer_.writeRecord(filename, lib_str.c_str(), lib_str.size());
  }
}

// python print the class and add to the converted_classes_. Recursively
// python print all classes that this class depends on.
void ScriptModuleSerializer::convertClass(
    const ClassTypePtr& class_type,
    torch::ModelDef* model_def) {
  if (converted_classes_.contains(class_type)) {
    return;
  }

  std::vector<ClassTypePtr> class_deps;
  std::ostringstream class_stream;
  PythonPrint(
      class_stream,
      class_type,
      tensor_table_,
      class_deps,
      /*enforce_importable=*/true);

  class_to_deps_[class_type] = class_deps;

  for (const auto& c : class_deps) {
    if (c == class_type) {
      // Don't re-process this class and enter an infinite loop. We need this
      // because we insert to converted_classes_ post-traversal, so the current
      // class isn't in there yet.
      continue;
    }
    convertClass(c, model_def);
  }
  // Insert *after* we've traversed the dependencies. This ensures that any
  // given class will appear after its dependencies in the order.
  converted_classes_.insert(class_type, class_stream.str());
}

void ScriptModuleSerializer::convertModel(
    const script::Module& module,
    torch::ModelDef* model_def,
    const script::ExtraFilesMap& extra_files) {
  model_def->set_producer_name("pytorch");
  model_def->set_producer_version("1.0"); // TODO: set the producer version
                                          // using appropriate function call
  model_def->set_proto_version(torch::ProtoVersion::PROTO_VERSION_NEWEST);

  convertModule(
      module, "", writer_.archiveName(), model_def->mutable_main_module());

  // This may write some attributes to the tensor_table_
  writeAttributeTable();

  writeTensorTable(model_def);
  writeLibs(model_def);

  // Write out extra files.
  for (const auto& kv : extra_files) {
    const std::string key = "extra/" + kv.first;
    writer_.writeRecord(key, kv.second.data(), kv.second.size());
  }
}

size_t ScriptModuleSerializer::addTensor(const at::Tensor& tensor) {
  tensor_table_.push_back(tensor);
  return tensor_table_.size() - 1;
}

void ScriptModuleSerializer::convertAndWriteTensor(
    size_t tensor_id,
    const at::Tensor& tensor,
    torch::TensorDef* tensor_proto,
    std::unordered_map<const void*, std::string>& storageMap) {
  for (auto d : tensor.sizes()) {
    tensor_proto->add_dims(d);
  }
  for (auto s : tensor.strides()) {
    tensor_proto->add_strides(s);
  }
  tensor_proto->set_data_type(caffe2::TypeMetaToDataType(
      at::scalarTypeToTypeMeta(tensor.scalar_type())));
  tensor_proto->set_offset(tensor.storage_offset());

  tensor_proto->set_requires_grad(tensor.requires_grad());

  uint64_t record_size = tensor.element_size() * tensor.storage().size();
  auto* key = tensor.storage().unsafeGetStorageImpl();

  auto storage_it = storageMap.find(key);
  if (storage_it == storageMap.end()) {
    at::Tensor storage_tensor = tensor;
    // TODO HIP support
    if (tensor.storage().device_type() == at::DeviceType::CUDA) {
      // NB: This new tensor is created to support cuda tensors.
      // Storages can be mutated when converting tensors from cuda to cpu,
      // and we need a cpu tensor to copy data from.
      storage_tensor = at::empty({0}, tensor.options())
                           .set_(
                               tensor.storage(),
                               /* storageOffset = */ 0,
                               /* size = */
                               {static_cast<int64_t>(tensor.storage().size())},
                               /* stride = */ {1})
                           .cpu();
      AT_ASSERT(
          storage_tensor.element_size() * storage_tensor.storage().size() ==
          record_size);
    }
    std::string name = "tensors/" + std::to_string(tensor_id);
    writer_.writeRecord(name, storage_tensor.storage().data(), record_size);
    storage_it = storageMap.insert({key, name}).first;
  }

  auto* data = tensor_proto->mutable_data();
  data->set_key(storage_it->second);

  // handle device case, set the device_detail and load to CUDA device
  std::stringstream ss;
  ss << tensor.device();
  tensor_proto->set_device(ss.str());
}

void ScriptModuleSerializer::writeTensorTable(torch::ModelDef* model_def) {
  std::unordered_map<const void*, std::string> storageMap;
  size_t tensor_id = 0;
  for (const at::Tensor& t : tensor_table_) {
    auto* tensor_proto = model_def->add_tensors();
    convertAndWriteTensor(tensor_id++, t, tensor_proto, storageMap);
  }
}

void ScriptModuleSerializer::writeAttributeTable() {
  Pickler pickler(&tensor_table_);
  pickler.start();
  for (const IValue& ivalue : attribute_table_) {
    pickler.addIValue(ivalue);
  }
  pickler.finish();
  writer_.writeRecord(
      "attributes.pkl", pickler.stack().data(), pickler.stack().size());
}

void ScriptModuleSerializer::convertModule(
    const script::Module& module,
    const std::string& prefix,
    const std::string& name,
    torch::ModuleDef* module_def) {
  module_def->set_name(name);
  module_def->set_optimize(module.is_optimized());
  for (const auto& elem : module.get_parameters()) {
    torch::ParameterDef* param_def = module_def->add_parameters();
    convertParameter(elem, param_def, /*is_buffer=*/false);
  }

  for (const auto& attribute : module.get_attributes()) {
    // Add attribute to ModuleDef
    torch::AttributeDef* attribute_def = module_def->add_attributes();
    attribute_def->set_name(attribute.name());
    attribute_def->set_type(attribute.type()->python_str());

    attribute_table_.push_back(attribute.value());
    attribute_def->set_id(attribute_table_.size() - 1);
  }

  std::stringstream module_name;
  if (prefix != "")
    module_name << prefix << "_";
  module_name << name;

  if (module.get_methods().size() > 0) {
    std::ostringstream methods;
    methods << "op_version_set = " << op_version_set << "\n";
    PythonPrint(
        methods,
        module.class_compilation_unit(),
        /*is_method=*/true,
        tensor_table_,
        class_table_,
        /*enforce_importable=*/true);
    torch::RecordRef* record = module_def->mutable_torchscript_arena();

    std::stringstream filename;
    filename << "code/" << module_name.str() << ".py";
    std::string methods_str = methods.str();
    writer_.writeRecord(
        filename.str(), methods_str.c_str(), methods_str.size());
    record->set_key(filename.str());
  }

  for (const auto& elem : module.get_modules()) {
    torch::ModuleDef* sub_def = module_def->add_submodules();
    convertModule(*elem, module_name.str(), elem->name(), sub_def);
  }
}

void ScriptModuleSerializer::convertParameter(
    const script::Slot& param,
    torch::ParameterDef* param_def,
    bool is_parameter) {
  param_def->set_name(param.name());
  param_def->set_is_buffer(is_parameter);
  param_def->set_tensor_id(addTensor(param.value().toTensor()));
}

// Pretty printing for ONNX
constexpr char indent_char = ' ';
constexpr size_t indent_multiplier = 2;

std::string idt(size_t indent) {
  return std::string(indent * indent_multiplier, indent_char);
}

std::string nlidt(size_t indent) {
  return std::string("\n") + idt(indent);
}

void dump(const onnx::TensorProto& tensor, std::ostream& stream) {
  stream << "TensorProto shape: [";
  for (int i = 0; i < tensor.dims_size(); ++i) {
    stream << tensor.dims(i) << (i == tensor.dims_size() - 1 ? "" : " ");
  }
  stream << "]";
}

void dump(const onnx::TensorShapeProto& shape, std::ostream& stream) {
  for (int i = 0; i < shape.dim_size(); ++i) {
    auto& dim = shape.dim(i);
    if (dim.has_dim_value()) {
      stream << dim.dim_value();
    } else {
      stream << "?";
    }
    stream << (i == shape.dim_size() - 1 ? "" : " ");
  }
}

void dump(const onnx::TypeProto_Tensor& tensor_type, std::ostream& stream) {
  stream << "Tensor dims: ";
  dump(tensor_type.shape(), stream);
}

void dump(const onnx::TypeProto& type, std::ostream& stream) {
  dump(type.tensor_type(), stream);
}

void dump(const onnx::ValueInfoProto& value_info, std::ostream& stream) {
  stream << "{name: \"" << value_info.name() << "\", type:";
  dump(value_info.type(), stream);
  stream << "}";
}

void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent);

void dump(
    const onnx::AttributeProto& attr,
    std::ostream& stream,
    size_t indent) {
  stream << "{ name: '" << attr.name() << "', type: ";
  if (attr.has_f()) {
    stream << "float, value: " << attr.f();
  } else if (attr.has_i()) {
    stream << "int, value: " << attr.i();
  } else if (attr.has_s()) {
    stream << "string, value: '" << attr.s() << "'";
  } else if (attr.has_g()) {
    stream << "graph, value:\n";
    dump(attr.g(), stream, indent + 1);
    stream << nlidt(indent);
  } else if (attr.has_t()) {
    stream << "tensor, value:";
    dump(attr.t(), stream);
  } else if (attr.floats_size()) {
    stream << "floats, values: [";
    for (int i = 0; i < attr.floats_size(); ++i)
      stream << attr.floats(i) << (i == attr.floats_size() - 1 ? "" : " ");
    stream << "]";
  } else if (attr.ints_size()) {
    stream << "ints, values: [";
    for (int i = 0; i < attr.ints_size(); ++i)
      stream << attr.ints(i) << (i == attr.ints_size() - 1 ? "" : " ");
    stream << "]";
  } else if (attr.strings_size()) {
    stream << "strings, values: [";
    for (int i = 0; i < attr.strings_size(); ++i)
      stream << "'" << attr.strings(i) << "'"
             << (i == attr.strings_size() - 1 ? "" : " ");
    stream << "]";
  } else if (attr.tensors_size()) {
    stream << "tensors, values: [";
    for (auto& t : attr.tensors()) {
      dump(t, stream);
    }
    stream << "]";
  } else if (attr.graphs_size()) {
    stream << "graphs, values: [";
    for (auto& g : attr.graphs()) {
      dump(g, stream, indent + 1);
    }
    stream << "]";
  } else {
    stream << "UNKNOWN";
  }
  stream << "}";
}

void dump(const onnx::NodeProto& node, std::ostream& stream, size_t indent) {
  stream << "Node {type: \"" << node.op_type() << "\", inputs: [";
  for (int i = 0; i < node.input_size(); ++i) {
    stream << node.input(i) << (i == node.input_size() - 1 ? "" : ",");
  }
  stream << "], outputs: [";
  for (int i = 0; i < node.output_size(); ++i) {
    stream << node.output(i) << (i == node.output_size() - 1 ? "" : ",");
  }
  stream << "], attributes: [";
  for (int i = 0; i < node.attribute_size(); ++i) {
    dump(node.attribute(i), stream, indent + 1);
    stream << (i == node.attribute_size() - 1 ? "" : ",");
  }
  stream << "]}";
}

void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent) {
  stream << idt(indent) << "GraphProto {" << nlidt(indent + 1) << "name: \""
         << graph.name() << "\"" << nlidt(indent + 1) << "inputs: [";
  for (int i = 0; i < graph.input_size(); ++i) {
    dump(graph.input(i), stream);
    stream << (i == graph.input_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "outputs: [";
  for (int i = 0; i < graph.output_size(); ++i) {
    dump(graph.output(i), stream);
    stream << (i == graph.output_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "initializers: [";
  for (int i = 0; i < graph.initializer_size(); ++i) {
    dump(graph.initializer(i), stream);
    stream << (i == graph.initializer_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "nodes: [" << nlidt(indent + 2);
  for (int i = 0; i < graph.node_size(); ++i) {
    dump(graph.node(i), stream, indent + 2);
    if (i != graph.node_size() - 1)
      stream << "," << nlidt(indent + 2);
  }
  stream << nlidt(indent + 1) << "]\n" << idt(indent) << "}\n";
}

void dump(
    const onnx::OperatorSetIdProto& operator_set_id,
    std::ostream& stream) {
  stream << "OperatorSetIdProto { domain: " << operator_set_id.domain() << "}";
}

void dump(const onnx::ModelProto& model, std::ostream& stream, size_t indent) {
  stream << idt(indent) << "ModelProto {" << nlidt(indent + 1)
         << "producer_name: \"" << model.producer_name() << "\""
         << nlidt(indent + 1) << "domain: \"" << model.domain() << "\""
         << nlidt(indent + 1) << "doc_string: \"" << model.doc_string() << "\"";
  if (model.has_graph()) {
    stream << nlidt(indent + 1) << "graph:\n";
    dump(model.graph(), stream, indent + 2);
  }
  if (model.opset_import_size()) {
    stream << idt(indent + 1) << "opset_import: [";
    for (auto& opset_imp : model.opset_import()) {
      dump(opset_imp, stream);
    }
    stream << "],\n";
  }
  stream << idt(indent) << "}\n";
}

std::string prettyPrint(const onnx::ModelProto& model) {
  std::stringstream ss;
  dump(model, ss, 0);
  return ss.str();
}

} // namespace

std::string pretty_print_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    bool google_printer) {
  auto graph_encoder = GraphEncoder(
      graph,
      onnx_opset_version,
      operator_export_type,
      initializers,
      defer_weight_export,
      true);
  if (google_printer) {
    return graph_encoder.get_model_proto().DebugString();
  }
  return prettyPrint(graph_encoder.get_model_proto());
}

// export_raw_ir will export IR ops without turning them into ONNX ops.
// The output will use the ONNX protobuf format, but the ops will not
// conform to the ONNX op specification. Thus, the output will not
// be interpretable by a ONNX-compatible framework. However, PyTorch or
// libtorch will be able to import the IR and play it back.
std::tuple<std::string, RawDataExportMap> export_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    bool strip_doc_string) {
  auto graph_encoder = GraphEncoder(
      graph,
      onnx_opset_version,
      operator_export_type,
      initializers,
      defer_weight_export,
      strip_doc_string);
  return std::make_tuple(
      graph_encoder.get_model_proto().SerializeAsString(),
      graph_encoder.get_raw_data_export_map());
}

void ExportModule(
    const script::Module& module,
    std::ostream& out,
    const script::ExtraFilesMap& extra_files) {
  ScriptModuleSerializer serializer(&out);
  serializer.serialize(module, extra_files);
}

void ExportModule(
    const script::Module& module,
    const std::string& filename,
    const script::ExtraFilesMap& extra_files) {
  ScriptModuleSerializer serializer(filename);
  serializer.serialize(module, extra_files);
}

} // namespace jit
} // namespace torch
