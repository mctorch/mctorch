#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/parser.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {
namespace script {

namespace {

// record of defined function
// Graph + metadata
struct FunctionDefinition {
  explicit FunctionDefinition(Def tree_)
      : tree(new Def(tree_)), _graph(new Graph()) {}

  explicit FunctionDefinition(std::unique_ptr<Graph> graph_)
      : tree(nullptr), graph(std::move(graph_)) {}

  bool isExtern() const {
    return tree == nullptr;
  }
  std::unique_ptr<Def> tree;
  std::unique_ptr<Graph> graph;
};

} // namespace

using SymbolTable = std::unordered_map<std::string, FunctionDefinition>;

struct DefCompiler {
  DefCompiler(FunctionDefinition& def, SymbolTable& symbol_table)
      : def(def),
        graph_stack({def.graph.get()}),
        symbol_table(symbol_table) {}
  void run() {
    auto& tree = *def.tree;
    cur().set_name(tree.name().name());
    for (auto input : tree.params()) {
      auto& name = input.ident().name();
      map(name, name);
      def.inputs.push_back(name);
    }
    for (auto output : tree.returns()) {
      auto& name = output.ident().name();
      map(name, name);
      def.outputs.push_back(name);
    }
    emitStatements(tree.statements());
  }
  void emitExpressionStatement(TreeRef stmt) {
    // expression with no used outputs
    emit(stmt, {});
  }
  void emitStatements(const ListView<TreeRef>& statements) {
    for (auto stmt : statements) {
      switch (stmt->kind()) {
        case TK_IF:
          emitIf(If(stmt));
          break;
        case TK_WHILE:
          emitWhile(While(stmt));
          break;
        case TK_ASSIGN:
          emitAssignment(Assign(stmt));
          break;
        case TK_GLOBAL:
          for (auto ident : stmt->trees()) {
            auto name = Ident(ident).name();
            map(name, name);
          }
          break;
        default:
          emitExpressionStatement(stmt);
          break;
      }
    }
  }
  void map(const std::string& name, const std::string& value) {
    env[name] = value;
  }
  const std::string& lookup(const Ident& ident) {
    if (env.count(ident.name()) == 0)
      throw ErrorReport(ident) << "undefined value " << ident.name();
    return env[ident.name()];
  }
  void emitAssignment(const Assign& stmt) {
    std::vector<std::string> outputs;
    for (auto lhs : stmt.lhs()) {
      std::string name = getLHS(lhs);
      // use of "_" gets renamed in Caffe2 graphs so that two uses
      // don't unintentionally interfere with each other
      if (name == "_") {
        name = fresh();
      }
      outputs.push_back(name);
    }
    if (stmt.reduction() != '=') {
      if (stmt.lhs().size() != 1) {
        throw ErrorReport(stmt)
            << "reductions are only allow when there is a single variable "
            << "on the left-hand side.";
      }
      auto lhs = stmt.lhs()[0];
      auto expr =
          Compound::create(stmt.reduction(), stmt.range(), {lhs, stmt.rhs()});
      emit(expr, outputs);
    } else {
      emit(stmt.rhs(), outputs);
    }
    int i = 0;
    for (auto ident : stmt.lhs()) {
      if (ident->kind() == TK_IDENT)
        map(Ident(ident).name(), outputs.at(i));
      i++;
    }
  }

  void emitIf(const If& stmt) {
      // TODO: add support for control flow ops
  }

  void emitWhile(const While& stmt) {
      // TODO: add support for control flow ops
  }

  std::string getLHS(const TreeRef& tree) {
    switch (tree->kind()) {
      case TK_IDENT: {
        return Ident(tree).name();
      } break;
      case '.': {
        auto sel = Select(tree);
        std::string lhs = getValue(sel.value());
        // TODO: check whether this subname exists in object lhs
        return lhs + "/" + sel.selector().name();
      } break;
      default: {
        throw ErrorReport(tree)
            << "This expression cannot appear on the left-hand size of an assignment";
      } break;
    }
  }
  Symbol getValue(const TreeRef& tree) {
    switch (tree->kind()) {
      case TK_IDENT: {
        return lookup(Ident(tree));
      } break;
      case '.': {
        auto sel = Select(tree);
        std::string lhs = getValue(sel.value());
        // TODO: check whether this subname exists in object lhs
        return lhs + "/" + sel.selector().name();
      } break;
      default: {
        std::string name = fresh();
        emit(tree, {name});
        return name;
      } break;
    }
  }
  std::string fresh(std::string prefix = "$t") {
    return std::string(prefix) + std::to_string(next_fresh++);
  }
  Symbol kindSymbol(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return kAdd;
      case '-':
        if (ninputs == 1)
          return kneg;
        else
          return kSub;
      case '*':
        return kMul;
      case '/':
        return kDiv;
      case TK_NE:
        return kne;
      case TK_EQ:
        return keq;
      case '<':
        return klt;
      case '>':
        return kgt;
      case TK_LE:
        return kle;
      case TK_GE:
        return kge;
      case TK_IF_EXPR:
        return k__if__;
      case TK_AND:
        return k__and__;
      case TK_OR:
        return k__or__;
      case TK_NOT:
        return k__not__;
      default:
        throw std::runtime_error("unknown kind " + std::to_string(kind));
    }
  }
  void fillArg(Argument* arg, const Attribute& attr) {
    std::string name = attr.name().name();
    arg->set_name(name);
    auto value = attr.value();
    // TODO: handle non-float attributes
    switch (value->kind()) {
      case TK_CONST: {
        auto v = value->tree(0)->doubleValue();
        auto f = value->tree(1)->stringValue();
        if (f == "f")
          arg->set_f(v);
        else
          arg->set_i(v);
      } break;
      case TK_LIST:
        for (auto t : value->trees()) {
          auto v = t->tree(0)->doubleValue();
          auto f = t->tree(1)->stringValue();
          if (f == "f")
            arg->add_floats(v);
          else
            arg->add_ints(v);
        }
        break;
    }
  }
  template <typename Trees>
  std::vector<std::string> getValues(const Trees& trees) {
    std::vector<std::string> result;
    for (const auto& tree : trees) {
      result.push_back(getValue(tree));
    }
    return result;
  }

  bool renameLookup(
      std::unordered_map<std::string, std::string>& rename_map,
      const std::string& name,
      std::string& rename) {
    // first look for name in the map directly
    auto it = rename_map.find(name);
    if (it != rename_map.end()) {
      rename = it->second;
      return true;
    }
    // otherwise if we have a rename entry like a => b and a name "a/foo/bar"
    // then replace it with "b/foo/bar"
    auto p = name.find("/");
    if (p == std::string::npos)
      return false;
    it = rename_map.find(name.substr(0, p));
    if (it != rename_map.end()) {
      rename = it->second + name.substr(p);
      return true;
    }
    return false;
  }
  void renameOp(
      std::unordered_map<std::string, std::string>& rename_map,
      const Apply& apply,
      const std::string& prefix,
      bool isExtern,
      OperatorDef* new_op) {
    for (size_t i = 0; i < new_op->input().size(); i++) {
      auto& name = new_op->input(i);
      std::string renamed;
      bool defined = renameLookup(rename_map, name, renamed);
      if (!isExtern && !defined) {
        throw ErrorReport(apply)
            << " unexpected undefined name '" << name
            << "' while attempting to inline '" << apply.name().name() << "'";
      } else if (!defined) {
        // extern function using a global name, assign it an identity mapping
        rename_map[name] = name;
      }
      new_op->set_input(i, renamed);
    }
    for (size_t i = 0; i < new_op->output().size(); i++) {
      auto& name = new_op->output(i);
      std::string renamed;
      if (!renameLookup(rename_map, name, renamed)) {
        renamed = prefix + name;
        rename_map[name] = renamed;
      }
      new_op->set_output(i, renamed);
    }
    // TODO: add support for control flow ops
  }

  bool hasBypassRename(const Apply& apply) {
    for (auto attr : apply.attributes()) {
      if (attr.name().name() == "rename") {
        if (attr.value()->kind() != TK_CONST) {
          throw ErrorReport(attr.value()) << "expected a single constant";
        }
        return attr.value()->tree(0)->doubleValue() == 0;
      }
    }
    return false;
  }

  // emit a function call by inlining the function's Graph into our
  // Graph, renaming temporaries func_name<unique_id>/orig_name
  // renaming only happens for values defined by the function
  // that are not marked outputs

  // inputs/outputs are passed by reference
  void emitFunctionCall(Apply& apply, const std::vector<std::string>& outputs) {
    std::string fname = apply.name().name();
    std::string prefix = fresh(fname) + "/";
    auto& fn = symbol_table.at(apply.name().name());
    bool isExtern = fn.isExtern();
    auto inputs = getValues(apply.inputs());
    std::unordered_map<std::string, std::string> rename_map;
    if (inputs.size() != fn.inputs.size()) {
      throw ErrorReport(apply) << fname << " expected " << fn.inputs.size()
                               << " values but received " << inputs.size();
    }
    for (size_t i = 0; i < inputs.size(); i++) {
      rename_map[fn.inputs[i]] = inputs[i];
    }
    if (outputs.size() != fn.outputs.size()) {
      throw ErrorReport(apply) << fname << " expected " << fn.outputs.size()
                               << " values but received " << outputs.size();
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      rename_map[fn.outputs[i]] = outputs[i];
    }
    for (auto& op : fn.graph->op()) {
      auto new_op = cur().add_op();
      new_op->CopyFrom(op);
      if (hasBypassRename(apply)) {
        prefix = "";
      }
      renameOp(rename_map, apply, prefix, isExtern, new_op);
    }
  }
  void expectOutputs(
      const TreeRef& tree,
      const std::vector<std::string>& outputs,
      size_t size) {
    if (outputs.size() != size) {
      throw ErrorReport(tree)
          << "expected operator to produce " << outputs.size()
          << " outputs but it produced " << size;
    }
  }
  void appendOutputs(
      const TreeRef& tree,
      OperatorDef* op,
      const std::vector<std::string>& outputs,
      size_t size) {
    expectOutputs(tree, outputs, size);
    for (size_t i = 0; i < size; i++) {
      op->add_output(outputs[i]);
    }
  }
  void emitNode(
      const Apply& apply,
      const std::vector<std::string>& outputs) {
    // must be before add_op
    auto values = getValues(apply.inputs());
    if (values.size() < schema->min_input() ||
        values.size() > schema->max_input()) {
      if (schema->min_input() == schema->max_input()) {
        throw ErrorReport(apply) << "operator expects " << schema->min_input()
                                 << " inputs but found " << values.size();
      } else {
        throw ErrorReport(apply)
            << "operator takes between " << schema->min_input() << " and "
            << schema->max_input() << " inputs but found " << values.size()
            << ".";
      }
    }
    auto numActualOutputs = schema->CalculateOutput(values.size());
    if (numActualOutputs != kCannotComputeNumOutputs &&
        outputs.size() != numActualOutputs) {
      throw ErrorReport(apply)
          << "operator produces " << numActualOutputs
          << " outputs but matched to " << outputs.size() << " outputs";
    }
    auto op = cur().add_op();
    op->set_type(apply.name().name());
    for (auto& v : values) {
      op->add_input(v);
    }
    // assume 1 output unless matched to more
    appendOutputs(apply, op, outputs, outputs.size());
    for (auto attribute : apply.attributes()) {
      fillArg(op->add_arg(), attribute);
    }
    // Ok, we checked the stuff where we can easily give a friendly error
    // message, now verify against the schema and report the error at the line
    if (!schema->Verify(*op)) {
      throw ErrorReport(apply) << "failed schema checking";
    }
  }

  // Emit an operation, writing results into 'outputs'.
  // This will _always_ compute something, unlike 'getValue' which simply
  // returns an already computed reference if possible.
  // So if 'tree' is an identifier or nested identifier (foo.bar)
  // this will cause it to be _copied_ into outputs.
  void emit(const TreeRef& tree, const std::vector<std::string>& outputs) {
    switch (tree->kind()) {
      case TK_IDENT:
      case '.': {
        auto op = cur().add_op();
        op->set_type("Copy");
        op->add_input(getValue(tree));
        appendOutputs(tree, op, outputs, 1);
      } break;
      case TK_NE:
      case TK_EQ:
      case '<':
      case '>':
      case TK_LE:
      case TK_GE:
      case '-':
      case '*':
      case '/':
      case '+':
      case TK_AND:
      case TK_OR:
      case TK_NOT: {
        // must be before add_op
        auto values = getValues(tree->trees());
        auto op = cur().add_op();
        op->set_type(operatorName(tree->kind(), tree->trees().size()));
        for (auto& v : values) {
          op->add_input(v);
        }
        appendOutputs(tree, op, outputs, 1);
        auto broadcast = op->add_arg();
        broadcast->set_name("broadcast");
        broadcast->set_i(1);
      } break;
      case TK_IF_EXPR: {
        // TODO: add support for control flow ops
      }
      case TK_APPLY: {
        auto apply = Apply(tree);
        // Handle built-ins like zeros, ones, etc
        if (builtins.count(apply.name().name()) > 0) {
          builtins[apply.name().name()](this, apply, outputs);
          break;
        }
        if (symbol_table.count(apply.name().name()) > 0) {
          emitFunctionCall(apply, outputs);
          break;
        }
        // TODO: do schema check here
        emitNode(apply, outputs);
      } break;
      case TK_CAST: {
        auto cast = Cast(tree);
        auto c2type = getType(cast.type());
        auto input = getValue(cast.input());
        auto op = cur().add_op();
        op->set_type("Cast");
        op->add_input(input);
        appendOutputs(tree, op, outputs, 1);
        auto arg = op->add_arg();
        arg->set_name("to");
        arg->set_i(c2type);
      } break;
      case TK_CONST: {
        expectOutputs(tree, outputs, 1);
        emitConst(
            tree->tree(0)->doubleValue(),
            outputs[0],
            tree->tree(1)->stringValue());
      } break;
      case TK_GATHER: {
        const auto gather = Gather(tree);
        desugarAndEmitOperator(
            "Gather",
            gather.range(),
            {gather.value(), gather.indices()},
            outputs);
        break;
      }
      case TK_SLICE: {
        const auto slice = Slice(tree);
        desugarAndEmitOperator(
            "Slice",
            slice.range(),
            {slice.value(), slice.startOr(0), slice.endOr(-1)},
            outputs);
        break;
      }
      default:
        throw ErrorReport(tree) << "NYI: " << tree;
        break;
    }
  }

  // Desugars constructs that are syntactic sugar and emits the corresponding
  // operator invocation, e.g. tensor[indices] -> tensor.Gather(indices).
  void desugarAndEmitOperator(
      const std::string& operatorName,
      const SourceRange& range,
      TreeList&& inputs,
      const std::vector<std::string>& outputs) {
    const auto applyName = Ident::create(range, operatorName);
    const auto applyInputs =
        Compound::create(TK_LIST, range, std::move(inputs));
    const auto applyAttributes = Compound::create(TK_LIST, range, {});
    const auto apply =
        Apply::create(range, applyName, applyInputs, applyAttributes);
    emitOperator(Apply(apply), outputs);
  }

  TensorProto_DataType getType(int type) {
    switch (type) {
      case TK_INT:
        return TensorProto_DataType_INT32;
      case TK_FLOAT:
        return TensorProto_DataType_FLOAT;
      case TK_LONG:
        return TensorProto_DataType_INT64;
      case TK_BOOL:
        return TensorProto_DataType_BOOL;
      default:
        throw std::runtime_error(
            "expected type token: " + std::to_string(type));
    }
  }

  OperatorDef* emitConst(
      double v,
      const std::string& output,
      const std::string& type_ident) {
    auto op = cur().add_op();
    op->set_type("ConstantFill");
    auto dtype = op->add_arg();
    dtype->set_name("dtype");
    auto value = op->add_arg();
    value->set_name("value");
    if (type_ident == "f") {
      dtype->set_i(TensorProto_DataType_FLOAT);
      value->set_f(v);
    } else if (type_ident == "LL") {
      dtype->set_i(TensorProto_DataType_INT64);
      value->set_i(v);
    } else if (type_ident == "b") {
      dtype->set_i(TensorProto_DataType_BOOL);
      value->set_i(v != 0);
    } else if (type_ident == "i") {
      dtype->set_i(TensorProto_DataType_INT32);
      value->set_i(v);
    } else {
      throw std::runtime_error("unknown type_ident " + type_ident);
    }
    auto shape = op->add_arg();
    shape->set_name("shape");
    shape->add_ints(1);
    op->add_output(output);
    return op;
  }
  Graph& cur() {
    return *graph_stack.back();
  }
  FunctionDefinition& def; // the def being constructed
  std::unordered_map<std::string, std::string>
      env; // map from name in Def to name in Graph
  std::vector<Graph*> graph_stack;
  SymbolTable& symbol_table;
  int next_fresh = 0;

 private:
  void emitFillOp(const Apply& apply, const std::vector<std::string>& outputs) {
    auto builtin_type = apply.name().name();
    auto values = getValues(apply.inputs());
    if (values.size() > 1) {
      throw ErrorReport(apply)
          << "Built-in " << builtin_type << " accepts 0 or 1 inputs.";
    }
    bool has_shape = false;
    for (const auto& attribute : apply.attributes()) {
      if (attribute.name().name() == "shape") {
        has_shape = true;
      } else {
        throw ErrorReport(apply)
            << "Unrecognized attribute " << attribute.name().name()
            << " for built-in " << builtin_type;
      }
    }
    if (builtin_type == "zeros" || builtin_type == "ones") {
      if ((values.size() != 1) && !has_shape) {
        throw ErrorReport(apply)
            << "Built-in " << builtin_type
            << " requires either 1 input or 1 shape attribute";
      }
    } else {
      // zeros_like or ones_like
      if (values.size() != 1) {
        throw ErrorReport(apply)
            << "Built-in " << builtin_type << " requires 1 input";
      }
    }

    auto op = cur().add_op();
    op->set_type("ConstantFill");
    if (values.size()) {
      op->add_input(values[0]);
      auto* input_as_shape = op->add_arg();
      input_as_shape->set_name("input_as_shape");
      if (builtin_type.find("_like") != std::string::npos) {
        // zeros_like, ones_like take the shape of the input as constant
        // tensor shape
        input_as_shape->set_i(0);
      } else {
        // zeros, ones take the values in the tensor as constant tensor
        // shape
        input_as_shape->set_i(1);
      }
    } else {
      fillArg(op->add_arg(), apply.attributes()[0]);
    }

    auto value = op->add_arg();
    value->set_name("value");
    if (builtin_type.find("ones") != std::string::npos) {
      value->set_f(1.0f);
    } else {
      value->set_f(0.0f);
    }
    appendOutputs(apply, op, outputs, 1);
  }
  // emitModule doesn't actually do anything except for allow
  // statements like a = Module() to register 'a' as a valid identifier
  // so that a.b = ... will work
  void emitModule(const Apply& apply, const std::vector<std::string>& outputs) {
    expectOutputs(apply, outputs, 1);
  }
  std::unordered_map<
      std::string,
      std::function<void(
          DefCompiler*,
          const Apply&,
          const std::vector<std::string>& outputs)>>
      builtins{{"zeros", &DefCompiler::emitFillOp},
               {"zeros_like", &DefCompiler::emitFillOp},
               {"ones", &DefCompiler::emitFillOp},
               {"ones_like", &DefCompiler::emitFillOp},
               {"Module", &DefCompiler::emitModule}};
};

struct CompilationUnitImpl {
  void defineFunction(const Def& def) {
    if (functions.count(def.name().name()) > 0) {
      throw ErrorReport(def) << def.name().name() << " already defined.";
    }
    DefCompiler c(
        functions.emplace(def.name().name(), FunctionDefinition(def))
            .first->second,
        functions);
    c.run();
  }

  void define(const std::string& str) {
    Parser p(str);
    while (p.lexer().cur().kind != TK_EOF) {
      defineFunction(Def(p.parseFunction()));
    }
  }

  const Graph& getGraph(const std::string& str) {
    if (functions.count(str) == 0)
      throw ErrorReport() << "undefined function: " << str << "\n";
    auto& def = functions.at(str);
    return *def.graph;
  }

 private:
  friend struct DefCompiler;
  SymbolTable functions;
};

CompilationUnit::CompilationUnit() : pImpl(new CompilationUnitImpl()) {}

void CompilationUnit::define(const std::string& str) {
  return pImpl->define(str);
}

const Graph& CompilationUnit::getGraph(
    const std::string& str) {
  return pImpl->getGraph(str);
}

CompilationUnit::~CompilationUnit() {}

} // namespace script
} // namespace jit
} // namespace torch
