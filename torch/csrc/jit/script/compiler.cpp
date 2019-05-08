#include <torch/csrc/jit/script/compiler.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/hooks_for_testing.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/script/final_returns.h>
#include <torch/csrc/jit/script/parser.h>
#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/script/script_type_parser.h>

#include <torch/csrc/jit/constants.h>

#include <c10/util/Optional.h>

#include <atomic>
#include <climits>
#include <set>

namespace torch {
namespace jit {
namespace script {

using SugaredValuePtr = std::shared_ptr<SugaredValue>;
using FunctionTable = std::unordered_map<std::string, Function&>;
using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
using AttributeMap = std::unordered_map<std::string, Const>;
using ListAttributeMap = std::unordered_map<std::string, std::vector<Const>>;

using TypeAndRange = std::pair<TypePtr, const SourceRange*>;

// Holds mappings from a variable name to a refined type for that variable
// E.g if x is not None is true than we can refine x from type t? to t.
struct Refinements {
  // using ordered map for deterministic graph output
  std::map<std::string, TypeAndRange> mappings_;

  void setRefinement(const std::string& name, TypeAndRange mapping) {
    mappings_[name] = std::move(mapping);
  }

  c10::optional<TypeAndRange> getRefinement(const std::string& name) const {
    const auto& maybe_mapping = mappings_.find(name);
    if (maybe_mapping == mappings_.end()) {
      return c10::nullopt;
    }
    return maybe_mapping->second;
  }

  // return the intersection of the values to type mappings between this
  // types can be unified
  void intersectRefinements(const Refinements& other) {
    Refinements ret;
    for (const auto& name_mapping : mappings_) {
      const auto& name = name_mapping.first;
      const auto& mapping = name_mapping.second;
      if (auto other_mapping = other.getRefinement(name_mapping.first)) {
        const auto maybe_unified_type =
            unifyTypes(mapping.first, other_mapping->first);
        if (maybe_unified_type) {
          ret.setRefinement(
              name, TypeAndRange(*maybe_unified_type, mapping.second));
        }
      }
    }
    mappings_ = std::move(ret.mappings_);
  }

  // return the union of the values to type mappings in a and b whose
  // types can be unified
  void unionRefinements(const Refinements& other) {
    Refinements ret;
    for (const auto& name_mapping : mappings_) {
      const auto& name = name_mapping.first;
      const auto& mapping = name_mapping.second;
      TypePtr t_1 = mapping.first;
      if (auto other_mapping = other.getRefinement(name_mapping.first)) {
        TypePtr t_2 = other_mapping->first;
        c10::optional<TypePtr> maybe_unified_type = c10::nullopt;
        if (t_1->isSubtypeOf(t_2)) {
          maybe_unified_type = t_1;
        } else if (t_2->isSubtypeOf(t_1)) {
          maybe_unified_type = t_2;
        }
        if (maybe_unified_type) {
          ret.setRefinement(
              name, TypeAndRange(*maybe_unified_type, mapping.second));
        }
      } else {
        ret.setRefinement(name, mapping);
      }
    }

    for (auto& name_mapping : other.mappings_) {
      if (!getRefinement(name_mapping.first)) {
        ret.setRefinement(name_mapping.first, name_mapping.second);
      }
    }

    mappings_ = std::move(ret.mappings_);
  }
};

// When a comparison like x is None is made, we associate type refinements
// with its true value and its false value. If a boolean that has refinements
// associated with it is used in a conditional of an if statememt, the true and
// false refinements are inserted into the corresponding blocks

struct BoolInfo {
  BoolInfo(Refinements true_refinements, Refinements false_refinements)
      : true_refinements_(std::move(true_refinements)),
        false_refinements_(std::move(false_refinements)){};
  BoolInfo() = default;

  Refinements true_refinements_;
  Refinements false_refinements_;

  BoolInfo* mergeOr(const BoolInfo& other) {
    // if the result of an OR is true, either a & b could have been true,
    // so we take the intersection of a.true_refinements & b.true_refinements.
    // if the result is false, both a and b had to be false,
    // so we take their union.
    true_refinements_.intersectRefinements(other.true_refinements_);
    false_refinements_.unionRefinements(other.false_refinements_);
    return this;
  }

  BoolInfo* mergeAnd(const BoolInfo& other) {
    // if the result of an AND is true, both a & b had to be true,
    // so we take the union of a.true_refinements and b.true_refinements.
    // if the result is false, either a or b could have been false,
    // so we take their intersection.
    true_refinements_.unionRefinements(other.true_refinements_);
    false_refinements_.intersectRefinements(other.false_refinements_);
    return this;
  }
};

static Value* asSimple(const SugaredValuePtr& value) {
  if (SimpleValue* sv = dynamic_cast<SimpleValue*>(value.get())) {
    return sv->getValue();
  }
  return nullptr;
}
// we consider _N where N is a number, to be a non-meaningful name
// and do not record it as a unique name. This allows python printing to
// be able to export and import more consistently named graphs
static bool meaningfulName(const std::string& name) {
  if (name.size() == 0)
    return false;
  if (name[0] == '$')
    return false;
  if (name[0] != '_')
    return true;
  for (size_t i = 1; i < name.size(); ++i) {
    if (!isdigit(name[i]))
      return true;
  }
  return false;
}

// Auxiliary data structure for desugaring variable binding into our always
// explicitly scoped language as we descend down nested control structures in
// the frontend (which themselves don't introduce scopes)
//
// The algorithm is roughly as follows:
// 1) While emitting a block within a control operator, add inputs and outputs
//      from the block for each value referenced (both "reads" and "writes").
//      This sets the value up as a candidate loop carried dependency.
// 2) When we reach the end of the block, examine all the values in the current
//      scope's value map. If the name also resides in an outer scope with a
//      different Value*, this is a true loop-carried dependency. If not, this
//      value was not assigned to. Replace all references to the block input
//      with the Value* pointed to in the tightest enclosing scope. Then delete
//      that block input and output.
// 3) When we emit the actual control operator, take all of the loop-carried
//      dependency values as inputs and return them as outputs from the control
//      op
//
//  Note that an alternative implementation could only add the loop-carried dep
//      inputs and outputs when we see a value that is mutated. This, however
//      requires replacing all references to that value *within the current
//      block* with a new input. That is to say: we need to traverse the pre-
//      decessor nodes and replace inputs that reference that value with the
//      newly-created input. This could be made less expensive with a change to
//      the IR API, but for now we choose to pessimisitically create inputs and
//      delete unnecessary ones later with replaceAllusesWith().
struct Environment {
  Environment(
      Function& method,
      ResolverPtr resolver,
      Block* b,
      std::shared_ptr<Environment> next = nullptr)
      : method(method),
        resolver(std::move(resolver)),
        b(b),
        next(std::move(next)) {}

  Function& method;
  ResolverPtr resolver;
  std::vector<std::string> captured_inputs;
  std::unordered_map<std::string, std::string> error_messages;
  Block* b;

  std::shared_ptr<Environment> next;

  // set type error in the lowest environment. if the variable is used after an
  // error has been set, then we will use the more informative error message
  void setVariableTypeError(const std::string& name, const std::string& msg) {
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();
    }
    runner->error_messages[name] = msg;
  }

  // see if type error has been set for a variable
  c10::optional<std::string> findVariableTypeError(const std::string& name) {
    auto runner = this;
    while (runner->next) {
      runner = runner->next.get();
    }
    auto msg = runner->error_messages.find(name);
    if (msg != runner->error_messages.end()) {
      return msg->second;
    } else {
      return c10::nullopt;
    }
  }

  SugaredValuePtr findInThisFrame(const std::string& name) {
    auto it = value_table.find(name);
    if (it != value_table.end()) {
      return it->second;
    }
    return nullptr;
  }

  SugaredValuePtr findInParentFrame(const std::string& name) {
    return next ? next->findInAnyFrame(name) : nullptr;
  }

  SugaredValuePtr findInAnyFrame(const std::string& name) {
    for (auto runner = this; runner; runner = runner->next.get()) {
      if (auto r = runner->findInThisFrame(name)) {
        return r;
      }
    }
    return nullptr;
  }

  Value* getValueInThisFrame(const SourceRange& loc, const std::string& name) {
    return value_table.at(name)->asValue(loc, method);
  }

  SugaredValuePtr createCapturedInput(Value* orig, const std::string& name) {
    // insert the captured input alphabetically in the capture list.
    // this ensures consistency of the order of loop-carried dependencies
    // even when the use in the loop is in a different order
    size_t insert_pos = 0;
    while (insert_pos < captured_inputs.size() &&
           name > captured_inputs[insert_pos]) {
      insert_pos++;
    }
    captured_inputs.insert(captured_inputs.begin() + insert_pos, name);

    // Create the input
    const size_t loop_carried_block_inputs_offset = 1;
    Value* new_input =
        b->insertInput(loop_carried_block_inputs_offset + insert_pos)
            ->setType(orig->type());

    // Associate this name with this value
    auto sv = std::make_shared<SimpleValue>(new_input);
    value_table[name] = sv;

    return sv;
  }

  SugaredValuePtr createCapturedInputIfNeeded(
      const SourceRange& loc,
      const std::string& ident) {
    auto in_frame = findInThisFrame(ident);
    if (in_frame) {
      return in_frame;
    }

    // recursively handles the case where parent blocks are also loops
    auto from_parent =
        next ? next->createCapturedInputIfNeeded(loc, ident) : nullptr;

    // recursively create the captured input if it is the loop block
    if (from_parent && getBlockOwningKind() == prim::Loop) {
      if (Value* simple_val = asSimple(from_parent))
        from_parent = createCapturedInput(simple_val, ident);
    }
    return from_parent;
  }

  Block* block() {
    return b;
  }
  Symbol getBlockOwningKind() {
    Symbol owning_kind = Symbol();
    if (b->owningNode()) {
      owning_kind = b->owningNode()->kind();
    }
    return owning_kind;
  }

  void setVar(const SourceRange& loc, const std::string& name, Value* value) {
    setSugaredVar(loc, name, std::make_shared<SimpleValue>(value));
  }

  void setSugaredVar(
      const SourceRange& loc,
      const std::string& name,
      SugaredValuePtr value) {
    Value* as_simple_value = asSimple(value);
    if (as_simple_value && !as_simple_value->hasUniqueName() &&
        meaningfulName(name) &&
        // note: if the value wasn't defined in this block, we might be giving a
        // name only used inside this block to a value outside of this. this is
        // not normally helpful for debugging and causes import/export jitter.
        as_simple_value->node()->owningBlock() == block()) {
      as_simple_value->setUniqueName(name);
    }
    // prevent re-assignment involving any sugared values
    // any reassignment like:
    // a = ...
    // while ...
    //   a = ..
    // requires 'a' to be first-class in the graph since its value depends on
    // control flow
    if (auto parent = findInParentFrame(name)) {
      if (!as_simple_value) {
        throw ErrorReport(loc)
            << "Cannot re-assign '" << name << "' to a value of type "
            << value->kind() << " because " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }
      Value* simple_parent = asSimple(parent);
      if (!simple_parent) {
        throw ErrorReport(loc)
            << "Cannot re-assign '" << name << "' because it has type "
            << value->kind() << " and " << name
            << " is not a first-class value.  Only reassignments to first-class values are allowed";
      }
      if (!as_simple_value->type()->isSubtypeOf(
              unshapedType(simple_parent->type()))) {
        std::stringstream errMsg;
        errMsg << "variable '" << name << "' previously has type "
               << simple_parent->type()->str()
               << " but is now being assigned to a value of type "
               << as_simple_value->type()->str();
        // Special-cased error msg if we're trying to assign to a tensor list.
        if (simple_parent->type()->kind() == TypeKind::ListType &&
            as_simple_value->type()->kind() == TypeKind::ListType) {
          errMsg << "\n. (Note: empty lists are constructed as Tensor[]; "
                 << "if you want an empty list of a different type, "
                 << "use `torch.jit.annotate(List[T], [])`, "
                 << "where `T` is the type of elements in the list)";
        }
        throw ErrorReport(loc) << errMsg.str();
      }
    }
    if (as_simple_value)
      createCapturedInputIfNeeded(loc, name);
    value_table[name] = std::move(value);
  }

  SugaredValuePtr getSugaredVar(const Ident& ident, bool required = true) {
    return getSugaredVar(ident.name(), ident.range());
  }
  Value* getVar(const Ident& ident) {
    return getSugaredVar(ident)->asValue(ident.range(), method);
  }

  SugaredValuePtr getSugaredVar(
      const std::string& ident,
      const SourceRange& range,
      bool required = true) {
    auto retval = createCapturedInputIfNeeded(range, ident);

    if (!retval) {
      static std::unordered_map<std::string, SugaredValuePtr> globals = {
          {"print", std::make_shared<PrintValue>()},
          {"float", std::make_shared<CastValue>(FloatType::get(), prim::Float)},
          {"int", std::make_shared<CastValue>(IntType::get(), prim::Int)},
          {"bool", std::make_shared<CastValue>(BoolType::get(), prim::Bool)},
          {"getattr", std::make_shared<GetAttrValue>()},
          {"isinstance", std::make_shared<IsInstanceValue>()},
          // todo(zach): remove when we can correctly export torch.full via ONNX
          // or we have implicit conversion that can convert numbers to tensors
          {"_to_tensor",
           std::make_shared<CastValue>(TensorType::get(), prim::NumToTensor)},
          {"len", std::make_shared<BuiltinFunction>(aten::len, at::nullopt)},
          {"hash", std::make_shared<BuiltinFunction>(aten::hash, at::nullopt)},
          {"min", std::make_shared<BuiltinFunction>(prim::min, at::nullopt)},
          {"max", std::make_shared<BuiltinFunction>(prim::max, at::nullopt)},
          {"abs", std::make_shared<BuiltinFunction>(prim::abs, at::nullopt)},
          {"list", std::make_shared<BuiltinFunction>(aten::list, at::nullopt)},
          {"ord", std::make_shared<BuiltinFunction>(aten::ord, at::nullopt)},
          {"rangelist",
           std::make_shared<BuiltinFunction>(prim::rangelist, at::nullopt)},
      };
      auto it = globals.find(ident);
      if (it != globals.end()) {
        retval = it->second;
      }
    }

    if (!retval) {
      if (auto type = resolver->resolveType(ident)) {
        const auto class_type = type->expect<ClassType>();
        retval = std::make_shared<script::ClassValue>(class_type);
      }
    }

    if (!retval) {
      retval = resolver->resolveValue(ident, method, range);
    }

    if (!retval && required) {
      // check if this value was not emitted in an if statement because of a
      // type mismatch. if it was, then we print a more informative error msg
      if (auto msg = findVariableTypeError(ident)) {
        throw ErrorReport(range) << *msg << "and was used here";
      }
      throw ErrorReport(range) << "undefined value " << ident;
    }
    return retval;
  }

  Value* getVar(const std::string& ident, const SourceRange& range) {
    return getSugaredVar(ident, range)->asValue(range, method);
  }

  // Given that after emitting statements in a block, we've added block inputs
  // for all value references and assignments, delete inputs for which there was
  // no assignment, only references.
  void deleteExtraInputs(const SourceRange& loc) {
    // note: skip i == 0, it is the loop trip count for inputs
    // and the loop condition for outputs.
    // captured_inputs is indexed by i - 1 since it only contains loop
    // carried dependencies
    //          inputs: loop_counter, lcd0, lcd1, ...
    //         outputs: loop_condition, lcd0, lcd1, ...
    // captured_inputs: lcd0, lcd1, ...
    AT_ASSERT(b->inputs().size() == b->outputs().size());
    AT_ASSERT(b->inputs().size() == captured_inputs.size() + 1);
    for (size_t i = b->inputs().size() - 1; i > 0; i--) {
      // nothing changed along this loop
      if (b->inputs()[i] == b->outputs()[i]) {
        auto name = captured_inputs[i - 1];
        Value* orig = findInParentFrame(name)->asValue(loc, method);
        b->inputs()[i]->replaceAllUsesWith(orig);
        b->eraseInput(i);
        b->eraseOutput(i);
        captured_inputs.erase(captured_inputs.begin() + i - 1);
      }
    }
  }
  std::vector<std::string> definedVariables() {
    std::vector<std::string> result;
    for (auto& kv : value_table) {
      result.push_back(kv.first);
    }
    return result;
  }

 private:
  ValueTable value_table;
};

template <class T>
static Value* materializeConstant(
    T val,
    Graph& graph,
    const SourceRange& r,
    std::unordered_map<T, Value*>& map) {
  auto existing_constant = map.find(val);
  if (existing_constant != map.end()) {
    return existing_constant->second;
  }

  WithInsertPoint guard(graph.block()->nodes().front());
  auto new_constant = graph.insertConstant(val, nullptr, r);
  map[val] = new_constant;

  return new_constant;
}

static Value* ensureInt(const SourceRange& range, Value* v) {
  if (!v->type()->isSubtypeOf(IntType::get())) {
    throw ErrorReport(range)
        << "expected a int but found a " << v->type()->str();
  }
  return v;
}

inline bool isSupportedListElementType(const TypePtr& type) {
  return type->isSubtypeOf(TensorType::get()) ||
      type->isSubtypeOf(NumberType::get());
}

// Information for each def being emitted.
// Defs can be nested to support closures so we need a stack of this information
// Currently records information about the functions return type.
struct DefContext {
  TypePtr declared_return_type_; // nullptr if not annotated
  TypePtr merged_return_type_; // nullptr if a Return has not been seen yet
};

struct to_ir {
  to_ir(
      const Def& def,
      ResolverPtr resolver_,
      const Self& self,
      Function& method) // method being constructed
      : method(method),
        graph(method.graph()),
        resolver(std::move(resolver_)),
        typeParser_(resolver),
        environment_stack(nullptr) {
    AT_ASSERT(resolver);
    pushFrame(graph->block(), /*starts_def=*/true);

    // Type annotations exclude explicitly typing the "self" parameter, so in
    // the case that this is a method with self we expect one fewer parameter
    // annotation than the number of parameters this Def takes.
    if (self && def.decl().params().size() == 0) {
      throw ErrorReport(def.decl().params().range())
          << "methods must have a self argument";
    }

    method.setSchema(emitDef(def, self, graph->block()));

    runCleanupPasses(graph);
  }

 private:
  Function& method;
  std::shared_ptr<Graph> graph;
  ResolverPtr resolver;
  std::unordered_map<int64_t, Value*> integral_constants;
  std::unordered_map<double, Value*> fp_constants;
  ScriptTypeParser typeParser_;

  // Singly-linked list of environments. This top element contains a member
  // `next` that points to the most immediate enclosing scope's value.
  std::shared_ptr<Environment> environment_stack;
  std::vector<DefContext> def_stack_;

  void pushFrame(Block* b, bool starts_def = false) {
    if (starts_def) {
      def_stack_.emplace_back();
    }
    environment_stack =
        std::make_shared<Environment>(method, resolver, b, environment_stack);
  }
  std::shared_ptr<Environment> popFrame(bool ends_def = false) {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    if (ends_def) {
      def_stack_.pop_back();
    }
    return old_frame;
  }

  void runCleanupPasses(std::shared_ptr<Graph>& to_clean) {
    // remove any uses of tuples that we inserted that are not needed
    LowerSimpleTuples(to_clean);
    ConstantPooling(to_clean);
  }

  FunctionSchema emitDef(const Def& def, const Self& self, Block* block) {
    auto schema = extractSchemaFromDef(def, self);
    // TODO need guards on init returning none
    if (schema.returns().size() == 1) {
      def_stack_.back().declared_return_type_ = schema.returns().at(0).type();
    }
    std::vector<Argument> arguments =
        emitFormalArguments(def, self, schema, block);

    // body
    auto stmts_list = moveAllReturnsToEnd(def.statements());
    emitStatements(stmts_list.begin(), stmts_list.end());
    std::vector<Argument> returns = {emitOutput(def.range(), schema, block)};
    return {def.name().name(), "", std::move(arguments), std::move(returns)};
  }

  std::vector<IValue> evaluateDefaults(
      const SourceRange& r,
      const std::vector<Expr>& default_types,
      const std::vector<Expr>& default_exprs) {
    std::vector<IValue> default_values;
    if (default_exprs.empty())
      return default_values;
    // To evaluate the default expressions, we create a graph with no inputs,
    // and whose returns are the default values we need.
    // We then run constant prop on this graph and check the results are
    // constant. This approach avoids having to have separate handling of
    // default arguments from standard expressions by piecing together existing
    // machinery for graph generation, constant propgation, and constant
    // extraction.
    auto tuple_type = Subscript::create(
        r,
        Var::create(r, Ident::create(r, "Tuple")),
        List<Expr>::create(r, default_types));
    auto blank_decl = Decl::create(
        r, List<Param>::create(r, {}), Maybe<Expr>::create(r, tuple_type));

    auto tuple_expr =
        TupleLiteral::create(r, List<Expr>::create(r, default_exprs));
    auto ret = Return::create(r, tuple_expr);
    auto def = Def::create(
        r,
        Ident::create(r, "defaults"),
        blank_decl,
        List<Stmt>::create(r, {ret}));
    auto m = std::make_shared<Module>();
    CompilationUnit cu;
    cu.define({def}, {resolver}, nullptr);
    Stack stack;
    cu.get_function("defaults").run(stack);
    return stack.at(0).toTuple()->elements();
  }

  std::vector<Argument> parseArgsFromDecl(const Decl& decl, const Self& self) {
    auto params_begin = decl.params().begin();
    auto params_end = decl.params().end();
    if (self) {
      ++params_begin;
    }
    std::vector<Argument> retval;

    std::vector<Expr> default_types;
    std::vector<Expr> default_exprs;
    // gather any non-empty default arguments
    for (auto it = params_begin; it != params_end; ++it) {
      auto param = *it;
      auto def = param.defaultValue();
      if (def.present()) {
        default_types.emplace_back(param.type());
        default_exprs.emplace_back(def.get());
      }
    }
    auto default_values =
        evaluateDefaults(decl.range(), default_types, default_exprs);

    auto defaults_it = default_values.begin();
    for (auto it = params_begin; it != params_end; ++it) {
      auto decl_arg = *it;

      TypePtr type;
      c10::optional<int32_t> N;

      // BroadcastList list can only appear at the argument level
      if (auto maybe_broad_list =
              typeParser_.parseBroadcastList(decl_arg.type())) {
        type = maybe_broad_list->first;
        N = maybe_broad_list->second;
      } else {
        type = typeParser_.parseTypeFromExpr(decl_arg.type());
        N = c10::nullopt;
      }
      c10::optional<IValue> default_value = c10::nullopt;
      if (decl_arg.defaultValue().present()) {
        default_value = *defaults_it++;
      }
      auto arg = Argument(
          decl_arg.ident().name(),
          type,
          N,
          default_value,
          decl_arg.kwarg_only());
      retval.push_back(arg);
    }
    return retval;
  }

  std::vector<Argument> parseReturnFromDecl(const Decl& decl) {
    // we represent no annoation on a return type as having no values in the
    // schema's return() list
    // in emitReturn we take the actual return value to be the value of the
    // return statement if no one was provided here
    if (!decl.return_type().present())
      return {};

    if (typeParser_.parseBroadcastList(decl.return_type().get()))
      throw ErrorReport(decl.return_type().range())
          << "Broadcastable lists cannot appear as a return type";
    auto parsed_type = typeParser_.parseTypeFromExpr(decl.return_type().get());
    return {Argument(
        "",
        parsed_type,
        /*N =*/c10::nullopt,
        /*default_value =*/c10::nullopt,
        /*kwarg_only =*/false)};
  }
  FunctionSchema extractSchemaFromDef(const Def& def, const Self& self) {
    const auto name = def.name().name();
    std::vector<Argument> args = parseArgsFromDecl(def.decl(), self);
    std::vector<Argument> returns = parseReturnFromDecl(def.decl());
    return FunctionSchema(
        name, "", std::move(args), std::move(returns), false, false);
  }

  std::vector<Argument> emitFormalArguments(
      const Def& def,
      const Self& self,
      const FunctionSchema& schema,
      Block* block) {
    std::vector<Argument> arguments; // for schema
    // inputs
    auto it = def.decl().params().begin();
    auto end = def.decl().params().end();
    auto expected_annotation_size = def.decl().params().size();
    if (self) {
      expected_annotation_size--;
    }
    if (schema.arguments().size() != expected_annotation_size) {
      throw ErrorReport(def.decl().params().range())
          << "Number of type annotations for"
          << " function parameters (" << schema.arguments().size() << ")"
          << " does not match the number of parameters on the function ("
          << expected_annotation_size << ")!";
    }

    if (self) {
      AT_ASSERT(it != end);
      const auto& name = (*it).ident().name();
      Value* new_input = block->addInput()->setUniqueName(name);
      environment_stack->setSugaredVar(
          (*it).ident().range(), name, self(new_input));
      arguments.emplace_back(name, new_input->type());
      ++it;
    }
    size_t arg_annotation_idx = 0;
    for (; it != end; ++it) {
      auto& name = (*it).ident().name();
      // Add the input to the graph
      Value* new_input = block->addInput();
      if (meaningfulName(name)) {
        new_input->setUniqueName(name);
      }
      environment_stack->setVar((*it).ident().range(), name, new_input);

      // Record the type for the schema and set the Type on the Value*
      arguments.push_back(schema.arguments().at(arg_annotation_idx++));
      new_input->setType(arguments.back().type());
    }
    return arguments;
  }

  Argument emitOutput(
      const SourceRange& range,
      const FunctionSchema& schema,
      Block* block) {
    // rewrites ensure there is always a return statement in program
    AT_ASSERT(def_stack_.back().merged_return_type_);
    // outputs
    Value* result = environment_stack->getVar("$return", range);
    block->registerOutput(result);
    return Argument("", def_stack_.back().merged_return_type_);
  }

  void emitStatements(const List<Stmt>& statements) {
    return emitStatements(statements.begin(), statements.end());
  }
  std::pair<std::shared_ptr<Graph>, Value*> lambdaLift(Block* block) {
    auto subgraph = std::make_shared<Graph>();
    // note: type is set later on pack_context and context when we know it
    Node* pack_context =
        graph->insertNode(graph->create(prim::TupleConstruct, {}, 1));
    Value* context = subgraph->addInput("context");
    // cannot use createTupleUnpack because the type is not known yet
    Node* unpack_context =
        subgraph->insertNode(subgraph->create(prim::TupleUnpack, {context}, 0));

    std::unordered_map<Value*, Value*> captures;
    auto env = [&](Value* v) -> Value* {
      auto it = captures.find(v);
      if (it != captures.end()) {
        return it->second;
      }
      pack_context->addInput(v);
      Value* r = unpack_context->addOutput()->copyMetadata(v);
      captures[v] = r;
      return r;
    };
    subgraph->block()->cloneFrom(block, env);
    auto context_type = TupleType::create(
        fmap(pack_context->inputs(), [](Value* v) { return v->type(); }));
    pack_context->output()->setType(context_type);
    context->setType(context_type);
    return std::make_pair(std::move(subgraph), pack_context->output());
  }
  // XXX - right now closures are used _only_ for defining gradients internally
  // There are several unfinished aspects that make them unusable generally
  // 1. We do not have a type, ivalue, operator to represent prim::Function, so
  // closure_node has type None
  //    and any graphs that contain it cannot be run
  // 2. There is no export logic for it yet, so it cannot be
  // exported/python_printed
  // 3. There is nothing preventing the assignment of already existing variables
  // inside the closures
  //    the changes to those variables will just get forgotten.
  // 4. There is no parsing support in frontend.py, this is intentional since it
  //    prevents people from accidentally using this feature.
  void emitClosure(const Def& def) {
    Node* closure_node = graph->insertNode(graph->create(prim::Function, 1));
    closure_node->output()->setType(
        NoneType::get()); // it is not a real thing yet, so just say the type is
                          // none.
    Block* block = closure_node->addBlock();
    {
      WithInsertPoint guard(block);
      pushFrame(block, /*starts_def=*/true);
      emitDef(
          def,
          nullptr,
          block); // ignore schema return, we just wont use it for now since we
                  // never create a Method for the closure
      popFrame(/*ends_def=*/true);
    }
    std::shared_ptr<Graph> subgraph;
    Value* context;
    std::tie(subgraph, context) = lambdaLift(block);
    runCleanupPasses(subgraph);
    closure_node->eraseBlock(0);
    closure_node->g_(attr::Subgraph, std::move(subgraph));
    auto tup =
        graph->insertNode(graph->createTuple({closure_node->output(), context}))
            ->output();
    environment_stack->setVar(def.name().range(), def.name().name(), tup);
  }

  void emitReturn(const Return& stmt) {
    Value* result = emitExpr(stmt.expr());
    TypePtr result_type = def_stack_.back().declared_return_type_;
    // result type is annotated, every return must convert to that type
    if (result_type) {
      // this guard skips implicit conversion from None -> Tensor for the return
      // type. otherwise forgetting a return a function returning a tensor will
      // cause a None to be converted to a tensor.
      if (!(result_type->isSubtypeOf(TensorType::get()) &&
            result->type()->isSubtypeOf(NoneType::get()))) {
        result = tryConvertToType(
            stmt.range(),
            *graph,
            result_type,
            result,
            /*allow_conversions=*/true);
      }

      if (!result->type()->isSubtypeOf(result_type)) {
        throw ErrorReport(stmt.range())
            << "Return value was annotated as having type "
            << result_type->python_str() << " but is actually of type "
            << result->type()->python_str();
      }
    } else {
      result_type = def_stack_.back().merged_return_type_;
      if (!result_type) {
        result_type = result->type();
      }
      if (!unifyTypes(result_type, result->type())) {
        throw ErrorReport(stmt.range())
            << "Previous return statement returned a value of type "
            << result_type->python_str()
            << " but this return statement returns a value of type "
            << result->type()->python_str();
      }
    }
    AT_ASSERT(result_type);
    def_stack_.back().merged_return_type_ = result_type;
    environment_stack->setVar(stmt.range(), "$return", result);
  }

  void emitStatements(
      List<Stmt>::const_iterator begin,
      List<Stmt>::const_iterator end) {
    for (; begin != end; ++begin) {
      auto stmt = *begin;
      switch (stmt.kind()) {
        case TK_IF:
          emitIf(If(stmt));
          break;
        case TK_WHILE:
          emitWhile(While(stmt));
          break;
        case TK_FOR:
          emitFor(For(stmt));
          break;
        case TK_ASSIGN:
          emitAssignment(Assign(stmt));
          break;
        case TK_AUG_ASSIGN:
          emitAugAssignment(AugAssign(stmt));
          break;
        case TK_GLOBAL:
          for (auto ident : Global(stmt).names()) {
            const auto& name = Ident(ident).name();
            environment_stack->setVar(
                ident.range(), name, graph->addInput(name));
          }
          break;
        case TK_EXPR_STMT: {
          auto expr = ExprStmt(stmt).expr();
          emitSugaredExpr(expr, 0);
        } break;
        case TK_RAISE:
          emitRaise(Raise(stmt).range());
          break;
        case TK_ASSERT:
          emitAssert(Assert(stmt));
          break;
        case TK_RETURN: {
          emitReturn(Return(stmt));
        } break;
        case TK_PASS:
          // Emit nothing for pass
          break;
        case TK_DEF:
          emitClosure(Def(stmt));
          break;
        default:
          throw ErrorReport(stmt)
              << "Unrecognized statement kind " << kindToString(stmt.kind());
      }
    }
  }

  std::shared_ptr<Environment> emitSingleIfBranch(
      Block* b,
      const List<Stmt>& branch,
      const Refinements& refinements) {
    pushFrame(b);
    WithInsertPoint guard(b);
    insertRefinements(refinements);
    emitStatements(branch);
    return popFrame();
  }

  Node* create(Symbol kind, const SourceRange& loc, size_t n_outputs) {
    return graph->create(kind, n_outputs)
        ->setSourceLocation(std::make_shared<SourceRange>(loc));
  }

  Value* emitTernaryIf(const TernaryIf& expr) {
    const auto& bool_info = findRefinements(expr.cond());
    Value* cond_value = emitCond(expr.cond());
    auto true_expr = [&] {
      insertRefinements(bool_info.true_refinements_);
      return emitExpr(expr.true_expr());
    };
    auto false_expr = [&] {
      insertRefinements(bool_info.false_refinements_);
      return emitExpr(expr.false_expr());
    };
    return emitIfExpr(expr.range(), cond_value, true_expr, false_expr);
  }

  Value* emitListComprehension(const ListComp& lc) {
    // this avoids a race condition where we would re-use the same temp name
    static std::atomic<size_t> tmp_count{0};
    const auto tmp_name =
        std::string("___list_acc") + std::to_string(tmp_count++);
    const auto list_value = emitExpr(lc.iter());
    if (list_value->type()->kind() != TypeKind::ListType) {
      // TODO: constraining iterators to be simple lists for now
      // as it makes easy to get list's element type.
      throw ErrorReport(lc.range())
          << "iterator expression is expected to be a list";
    }
    auto elem_types = list_value->type()->containedTypes();
    // TODO: users can easily change the type to (x,1) or float(x)
    // as in `float(x) for x in my_list_of_ints`
    // eventually, we would probably want to temporarily inject x
    // so we can evaluate the generator expression (e.g. `float(x)`) depending
    // on x

    // given `[x*2 for x in my_list]` this generates the following AST:
    // __list_acc = []
    // for x in my_list:
    //  __list_acc.append(x*2)
    const auto n = graph->insertNode(
        graph->createList(elem_types.at(0), at::ArrayRef<Value*>{}));
    environment_stack->setVar(lc.range(), tmp_name, n->output());
    const auto tmp_list_ident = Ident::create(lc.range(), tmp_name);
    const auto tmp_list_var = Var::create(lc.range(), tmp_list_ident);
    const auto append_ident = Ident::create(lc.range(), "append");
    const auto dot_op = Select::create(lc.range(), tmp_list_var, append_ident);
    const auto append_args_list = List<Expr>::create(lc.range(), {lc.elt()});
    const auto append_attrs = List<Attribute>::create(lc.range(), {});
    const auto apply_append =
        Apply::create(lc.range(), dot_op, append_args_list, append_attrs);
    const auto expr_stmt = ExprStmt::create(lc.range(), apply_append);
    const auto stmt_list = List<Stmt>::create(lc.range(), {expr_stmt});
    const auto iters_list = List<Expr>::create(lc.range(), {lc.iter()});
    const auto targets_list = List<Expr>::create(lc.range(), {lc.target()});
    const auto for_loop =
        For::create(lc.range(), targets_list, iters_list, stmt_list);
    emitFor(for_loop);
    return n->output();
  }

  // Insert subtyping refinements
  void insertRefinements(const Refinements& ref) {
    for (const auto& name_mappings : ref.mappings_) {
      const std::string& name = name_mappings.first;
      auto type = name_mappings.second.first;
      const auto& range = *name_mappings.second.second;
      Value* v = environment_stack->getVar(name, range);
      if (type != NoneType::get()) {
        Value* output = graph->insert(prim::unchecked_unwrap_optional, {v});
        environment_stack->setVar(range, name, output);
      }
      // todo @eellison - revisit inserting Nones when None subtypes Optional
    }
  }

  Value* emitShortCircuitIf(
      const SourceRange& loc,
      const TreeRef& first_expr,
      const TreeRef& second_expr,
      bool is_or) {
    const auto first_bool_info = findRefinements(first_expr);
    Value* first_value = emitCond(Expr(first_expr));

    // if the second expr in the short circuit is not evaluated,
    // than the first expression is False if the short circuit
    // is an `and` and True if the short circuit is an `or`.
    // `False and expr` -> False, `True or expr` -> True
    //
    // inserting it as a constant makes optimization easier

    Value* first_value_returned;

    const Refinements* first_expr_refinements;
    const Refinements* second_expr_refinements;
    // if it's an OR the first expr is emitted in the true branch
    // and the second expr in the false branch, if it's an AND the opposite
    if (is_or) {
      first_value_returned = graph->insertConstant(true, nullptr, loc);
      first_expr_refinements = &first_bool_info.true_refinements_;
      second_expr_refinements = &first_bool_info.false_refinements_;
    } else {
      first_value_returned = graph->insertConstant(false, nullptr, loc);
      first_expr_refinements = &first_bool_info.false_refinements_;
      second_expr_refinements = &first_bool_info.true_refinements_;
    }

    auto get_first_expr = [&] {
      insertRefinements(*first_expr_refinements);
      return first_value_returned;
    };

    auto get_second_expr = [&] {
      insertRefinements(*second_expr_refinements);
      return emitCond(Expr(second_expr));
    };

    // if this is an OR, eval second expression if first expr is False
    // If this is an AND, eval second expression if first expr is True
    if (is_or) {
      return emitIfExpr(loc, first_value, get_first_expr, get_second_expr);
    } else {
      return emitIfExpr(loc, first_value, get_second_expr, get_first_expr);
    }
  }

  Value* emitIfExpr(
      const SourceRange& range,
      Value* cond_value,
      std::function<Value*()> true_expr,
      std::function<Value*()> false_expr) {
    Node* n = graph->insertNode(create(prim::If, range, 0));

    n->addInput(cond_value);
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    auto emit_if_expr = [this](Block* b, std::function<Value*()> expr_value) {
      pushFrame(b);
      WithInsertPoint guard(b);
      Value* out_val = expr_value();
      b->registerOutput(out_val);
      popFrame();
    };

    emit_if_expr(true_block, std::move(true_expr));
    emit_if_expr(false_block, std::move(false_expr));

    auto true_type = true_block->outputs().at(0)->type();
    auto false_type = false_block->outputs().at(0)->type();
    auto unified = unifyTypes(true_type, false_type);
    if (!unified) {
      throw ErrorReport(range)
          << "if-expression's true branch has type " << true_type->str()
          << " but false branch has type " << false_type->str();
    }

    // Add op outputs
    auto expr_value = n->addOutput()->setType(*unified); // Resulting value

    return expr_value;
  }

  Value* emitCond(const Expr& cond) {
    Value* v = emitExpr(cond);
    if (!v->type()->isSubtypeOf(BoolType::get())) {
      Value* cast_v = emitBuiltinCall(
          cond.get()->range(),
          *v->owningGraph(),
          prim::Bool,
          c10::nullopt,
          {v},
          {},
          /*required*/ false);
      if (cast_v == nullptr) {
        ErrorReport error(cond);
        error
            << "expected a bool, int, float, or Tensor expression for condition but found "
            << v->type()->str();
        throw error;
      } else {
        v = cast_v;
      }
    }
    return v;
  }

  void emitIfElseBlocks(Value* cond_value, const If& stmt) {
    Node* n = graph->insertNode(create(prim::If, stmt.range(), 0));
    n->addInput(cond_value);
    const auto bool_info = findRefinements(stmt.cond());
    auto* true_block = n->addBlock();
    auto* false_block = n->addBlock();

    // Emit both blocks once to get the union of all mutated values
    auto save_true = emitSingleIfBranch(
        true_block, stmt.trueBranch(), bool_info.true_refinements_);
    auto save_false = emitSingleIfBranch(
        false_block, stmt.falseBranch(), bool_info.false_refinements_);

    // In python, every variable assigned in an if statement escapes
    // the scope of the if statement (all variables are scoped to the function).
    // Script is a subset of python: we consider variables to be in scope
    // as long as there is a definition of the variable along all paths
    // through the if statemnent
    // ----
    // if ...:
    //   a =
    // else:
    //   ...
    // ... = a  # error, a is not defined along all paths
    // ----
    // if ...:
    //   a =
    // else:
    //   a =
    // ... = a # OK, a is defined along all paths
    // ----
    // a = ...
    // if ...:
    //   a =
    // ... = a # OK, a is defined along all paths

    // ordered set, because we want deterministic graph output
    std::set<std::string> mutated_variables;

    for (auto& v : save_true->definedVariables()) {
      if (save_false->findInAnyFrame(v)) {
        mutated_variables.insert(v);
      }
    }
    for (auto& v : save_false->definedVariables()) {
      if (save_true->findInAnyFrame(v)) {
        mutated_variables.insert(v);
      }
    }

    // Register outputs in each block
    for (const auto& x : mutated_variables) {
      auto tv = save_true->getVar(x, stmt.range());
      auto fv = save_false->getVar(x, stmt.range());
      auto unified = unifyTypes(tv->type(), fv->type());

      // attempt to unify the types. we allow variables to be set to different
      // types in each branch as long as that variable is not already in scope,
      // or if that variable does not get used later. here, we save the error
      // so that the error message will be more informative in the case that is
      // used later. When a is accessed in (a + 1), the error will get printed
      // if cond:
      //    a = 1
      // else:
      //    a = tensor
      // b = a + 1
      //
      if (!unified) {
        ErrorReport error(stmt);
        error << "Type mismatch: " << x << " is set to type "
              << tv->type()->str() << " in the true branch"
              << " and type " << fv->type()->str() << " in the false branch";
        if (save_true->findInParentFrame(x) ||
            save_false->findInParentFrame(x)) {
          throw error;
        } else {
          // error gets saved in the lowest environment because all
          // variables are scoped to the function. doesn't matter if this
          // accessed through save_true or save_false
          save_true->setVariableTypeError(x, error.what());
          continue;
        }
      }
      true_block->registerOutput(tv);
      false_block->registerOutput(fv);
      environment_stack->setVar(
          stmt.range(), x, n->addOutput()->setType(*unified));
    }
  }

  void emitIf(const If& stmt) {
    // NOTE: emitIf checks on If stmt condition to see if the cond AST kind ==
    // is/is not, for such cases we do meta programming and disable emitting the
    // corresponding branches
    Expr cond = stmt.cond();

    if (cond.kind() != TK_IS && cond.kind() != TK_ISNOT) {
      // emit normal IF stmt for cases except TK_IS and TK_ISNOT
      Value* cond_value = emitCond(cond);
      emitIfElseBlocks(cond_value, stmt);
      return;
    }
    // meta programming on AST for is/is not cases and emit branches base on the
    // possible output of cond
    auto cond_op = BinOp(cond);
    SugaredValuePtr lhs_val = emitSugaredExpr(cond_op.lhs(), 1);
    SugaredValuePtr rhs_val = emitSugaredExpr(cond_op.rhs(), 1);

    List<Stmt> always_none_branch =
        cond.kind() == TK_IS ? stmt.trueBranch() : stmt.falseBranch();
    List<Stmt> never_none_branch =
        cond.kind() == TK_IS ? stmt.falseBranch() : stmt.trueBranch();

    auto lhs_none = lhs_val->isNone();
    auto rhs_none = rhs_val->isNone();

    // Dispatch logic (A: ALWAYS, N: NEVER, M: MAYBE):
    //
    // AA, -> emit always_none_branch
    // AN , NA-> emit never_none_branch
    // MA, MM, MN, NM, NN, AM -> emit both conditional branches

    if (lhs_none == ALWAYS && rhs_none == ALWAYS) {
      // None is/is not None: only emit the always_none_branch
      emitStatements(always_none_branch);
    } else if (
        (lhs_none == ALWAYS && rhs_none == NEVER) ||
        (lhs_none == NEVER && rhs_none == ALWAYS)) {
      // lhs_val/rhs_val with A/M: only emit never_none_branch
      emitStatements(never_none_branch);
    } else {
      // all other cases for lhs_val and rhs_val
      // emit the whole If stmt as usual, finish emitCond first
      auto lhs_range = cond_op.lhs().get()->range();
      auto rhs_range = cond_op.rhs().get()->range();

      auto kind = getNodeKind(cond.kind(), cond.get()->trees().size());
      Value* cond_value = emitBuiltinCall(
          cond.get()->range(),
          *method.graph(),
          kind,
          c10::nullopt,
          {lhs_val->asValue(lhs_range, method),
           rhs_val->asValue(rhs_range, method)},
          {},
          /*required=*/true);
      emitIfElseBlocks(cond_value, stmt);
    }
  }

  // *********************** Loop Operators ************************************
  // Emits a loop operators conforming to the semantics specified at
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#experimental-loop
  // TODO: implement scan_outputs

  // the format of the Loop instruction is:
  // loop_carried_outputs* = Loop(max_trip_count, start_condition,
  //                              loop_carried_inputs*)
  //                    block0(loop_counter, loop_carried_block*) {
  //                       <body>
  //                       -> (continue_condition, loop_carried_block_outputs*)
  //                    }
  // all loop_carried_... lists are the same length and represent the value of
  // loop-carried variables whose definitions are updated as the loop executes
  // in a way that ensure single static assignment.

  void emitLoopCommon(
      SourceRange range,
      const List<Stmt>& body,
      const std::function<void(Value*, std::shared_ptr<Environment>)>&
          current_element_assigner,
      c10::optional<Expr> cond,
      Value* max_trip_count_val = nullptr) {
    Value* cond_val = nullptr;
    Node* n = graph->insertNode(create(prim::Loop, range, 0));
    WithInsertPoint guard(n);

    if (!max_trip_count_val)
    {
      max_trip_count_val = materializeConstant(
          std::numeric_limits<int64_t>::max(),
          *graph,
          range,
          integral_constants);
    }

    cond_val = (cond) ? emitCond(cond.value())
                      : graph->insertConstant(true, nullptr, range);
    n->addInput(max_trip_count_val);
    n->addInput(cond_val);
    auto* body_block = n->addBlock();
    Value* trip_count =
        body_block->addInput()->setType(IntType::get()); // Iteration num

    {
      pushFrame(body_block);
      WithInsertPoint guard(body_block);

      // current_element_assigner uses an induction variable
      // to set a current element
      if (current_element_assigner)
      {
        current_element_assigner(trip_count, environment_stack);
      }

      emitStatements(body);

      // Also emit the conditional
      cond_val = (cond) ? emitCond(cond.value())
                        : graph->insertConstant(true, nullptr, range);
      body_block->registerOutput(cond_val);
      auto body_frame = popFrame();
      auto outer_frame = environment_stack;

      // Add block outputs to correspond to each captured input
      // some of these will be removed.
      for (const auto& x : body_frame->captured_inputs) {
        auto fv = body_frame->getValueInThisFrame(range, x);
        body_block->registerOutput(fv);
      }

      // Remove inputs for values that did not mutate within the
      // block
      body_frame->deleteExtraInputs(range);

      // register node inputs/outputs for the true loop carried deps,
      for (size_t i = 0; i < body_frame->captured_inputs.size(); ++i) {
        auto x = body_frame->captured_inputs[i];
        n->addInput(outer_frame->getVar(x, range));
        // body_block->inputs(): loop_counter, lcd0, lcd1, ...
        // captured_inputs: lcd0, lcd1, ...
        auto typ = body_block->inputs()[i + 1]->type();
        outer_frame->setVar(range, x, n->addOutput()->setType(typ));
      }
    }
  }

  void emitForRange(
      const SourceRange& range,
      const Ident& target,
      const List<Expr>& args,
      const List<Stmt>& body) {
    // TODO: start, stop, step loop
    if (args.size() != 1) {
      throw ErrorReport(range)
          << "range() expects 1 argument but got " << args.size();
    }
    auto max_trip_count_val = ensureInt(range, emitExpr(args[0]));
    const auto& ident_name = target.name();
    auto assigner = [ident_name, range](Value* index, std::shared_ptr<Environment> env) {
      env->setVar(range, ident_name, index);
    };
    emitLoopCommon(range, body, assigner, {}, max_trip_count_val);
  }

  void emitForInListLoop(
      const For& stmt,
      const std::shared_ptr<torch::jit::script::SimpleValue>& siv) {
    auto targets = stmt.targets();
    auto itrs = stmt.itrs();
    auto body = stmt.body();
    auto& range = stmt.range();
    auto target = Var(targets[0]).name();

    auto listArg = siv->asValue(range, method);
    auto max_trip_count_val = emitBuiltinCall(
        range,
        *graph,
        aten::len,
        c10::nullopt,
        {listArg},
        {},
        /*required=*/true);
    const auto& ident_name = target.name();
    auto assigner = [ident_name, range, listArg, this](
                        Value* index, std::shared_ptr<Environment> env) {
      auto cur_elm = emitBuiltinCall(
          range,
          *this->graph,
          aten::select,
          c10::nullopt,
          {listArg, index},
          {},
          /*required=*/true);
      env->setVar(range, ident_name, cur_elm);
    };
    emitLoopCommon(range, body, assigner, {}, max_trip_count_val);
  }

  void emitFor(const For& stmt) {
    // For now, we only support range loops. e.g. for i in range(3): ...
    auto targets = stmt.targets();
    auto itrs = stmt.itrs();
    auto body = stmt.body();

    if (stmt.itrs().size() != 1) {
      throw ErrorReport(stmt)
          << "List of iterables is not supported currently.";
    }
    if (targets.size() != 1) {
      throw ErrorReport(stmt)
          << "Iteration variable unpacking is not supported";
    }

    if (targets[0].kind() != TK_VAR) {
      throw ErrorReport(targets[0])
          << "unexpected expression in variable initialization of for loop";
    }
    auto target = Var(targets[0]).name();

    // match range(<expr>) style loops
    // itrs must consist of a single Apply node
    if (itrs[0].kind() == TK_APPLY) {
      Apply range_iterator = Apply(itrs[0]);
      if (range_iterator.callee().kind() == TK_VAR) {
        Var var = Var(range_iterator.callee());
        if (var.name().name() == "range") {
          return emitForRange(
              stmt.range(), target, range_iterator.inputs(), body);
        }
      }
    }

    // it isn't a range(<expr>) loop, treat it as a sugared value that maybe can
    // be unrolled
    auto sv = emitSugaredExpr(itrs[0], 1);
    // check if a value is simple and list-like
    if (auto siv = std::dynamic_pointer_cast<SimpleValue>(sv)) {
      if (siv->getValue()->type()->kind() == TypeKind::ListType) {
        emitForInListLoop(stmt, siv);
        return;
      }
    }
    auto instances = sv->asTuple(stmt.range(), method);
    const std::string& target_name = target.name();
    pushFrame(environment_stack->block());
    for (const auto& inst : instances) {
      environment_stack->setSugaredVar(itrs[0].range(), target_name, inst);
      emitStatements(body);
    }

    for (const auto& n : environment_stack->definedVariables()) {
      if (environment_stack->findInParentFrame(n)) {
        environment_stack->next->setVar(
            stmt.range(), n, environment_stack->getVar(n, stmt.range()));
      }
    }
    popFrame();
  }

  void emitWhile(const While& stmt) {
    auto cond = stmt.cond();
    emitLoopCommon(stmt.range(), stmt.body(), nullptr, cond, nullptr);
  }

  // Currently we do not support assigning exceptions to variables,
  // a = Exception("hi")
  // raise a
  //
  // We ignore the expression following raise
  //
  // NYI: add exception logic to control-flow nodes
  // if True:
  //   a = 1
  // else
  //   raise Exception("Hi")
  // print(a)
  void emitRaise(const SourceRange& loc) {
    const std::string exception = "Exception";
    auto string_input = insertConstant(*graph, exception, nullptr, loc);
    graph->insert(prim::RaiseException, {string_input}, {}, loc);
  }

  void emitAssert(const Assert& stmt) {
    Value* cond_value = emitCond(stmt.test());
    Node* n = graph->insertNode(create(prim::If, stmt.range(), 0));

    n->addInput(cond_value);
    /* true_block =*/n->addBlock();
    auto* false_block = n->addBlock();

    // if assert test is false throw exception
    pushFrame(false_block);
    WithInsertPoint guard(false_block);
    emitRaise(stmt.range());
    popFrame();
  }

  // Validate that the `lhs` Expr's in an assignment statement are valid. That
  // is:
  //
  // 1) All lhs Expr's are either Var or Starred nodes
  // 2) There is at most one Starred node in the lhs Expr
  // 3) A Starred node can only appear when there is another non-Starred lhs
  //    Expr. Concretely this means that `*abc = func()` is illegal. Unpacking
  //    all outputs into a tuple is covered by `abc = func()`.
  bool calcNumStarredUnpack(const List<Expr>& lhs, const SourceRange& r) {
    size_t num_normal_assign = 0;
    size_t num_starred = 0;
    for (const auto& assignee : lhs) {
      if (assignee.kind() == TK_VAR || assignee.kind() == TK_SUBSCRIPT) {
        num_normal_assign++;
      } else if (assignee.kind() == TK_STARRED) {
        num_starred++;
      } else {
        throw ErrorReport(assignee) << "lhs of assignment must be a variable, "
                                    << "subscript, or starred expression.";
      }
    }

    if (num_starred > 1) {
      throw ErrorReport(r)
          << "Only one starred expression is allowed on the lhs.";
    }

    if (num_starred > 0 && num_normal_assign == 0) {
      throw ErrorReport(r) << "A Starred expression may only appear on the "
                           << "lhs within the presence of another non-starred"
                           << " expression.";
    }

    return num_starred;
  }

  // Get the appropriate builtin op for this augmented assignment
  // If the RHS is a tensor, return the corresponding ATen in-place op
  // If it's a list of scalars, then return the corresponding list augment op
  Symbol getAugOp(const AugAssign& stmt, bool isTensor) {
    switch (stmt.aug_op()) {
      case '+':
        return isTensor ? aten::add_ : aten::add;
      case '-':
        return isTensor ? aten::sub_ : aten::sub;
      case '/':
        return isTensor ? aten::div_ : aten::div;
      case '*':
        return isTensor ? aten::mul_ : aten::mul;
      default:
        throw ErrorReport(stmt)
            << "Unknown augmented assignment: " << kindToString(stmt.aug_op());
    }
  }

  // Emit nodes for augmented assignments like `+=`
  void emitAugAssignment(const AugAssign& stmt) {
    switch (stmt.lhs().kind()) {
      case TK_VAR: {
        emitAugAssignmentToVar(stmt);
      } break;
      case '.': {
        emitAugAssignmentToSelectVar(stmt);
      } break;
      case TK_SUBSCRIPT: {
        emitAugAssignmentToSubscript(stmt);
      } break;
      default:
        throw ErrorReport(stmt.lhs())
            << "unexpected expression on "
            << "left-hand side of augmented assignment.";
    }
  }

  // This will be called when there is a class param or module buffer
  // mutation which make the LHS of the expr be a select expression
  //
  // Example like:
  // class A(Module):
  //  def __init__():
  //    self.register_buffer("running_var", torch.zeros(1))
  //
  //  def forward():
  //    self.num_batches += 1
  //
  // In this case we will only consider the scenario that the module
  // buffer type is a tensor, and we emit the corresponding tensor
  // in place op, and throw error for other unsupported types
  void emitAugAssignmentToSelectVar(const AugAssign& stmt) {
    const auto lhs = Select(stmt.lhs());
    const auto lhsSugaredVar =
        environment_stack->getSugaredVar(Var(lhs.value()).name());
    const auto lhsValue =
        lhsSugaredVar->attr(lhs.range(), method, lhs.selector().name())
            ->asValue(lhs.range(), method);
    if (lhsValue->type()->isSubtypeOf(TensorType::get())) {
      // for module parameter/buffer assignment, only consider tensor types,
      // emit the corresponding in-place op
      const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()));
      const auto self = NamedValue(stmt.lhs().range(), "self", lhsValue);
      emitBuiltinCall(
          stmt.range(),
          *method.graph(),
          getAugOp(stmt, /*isTensor=*/true),
          self,
          {rhs},
          {},
          /*required=*/true);

    } else {
      throw ErrorReport(stmt.lhs())
          << "left-hand side of augmented assignment to module "
          << "parameters/buffers can only be tensor types";
    }
  }

  void emitAugAssignmentToVar(const AugAssign& stmt) {
    const auto lhs = Var(stmt.lhs());
    const auto lhsValue = environment_stack->getSugaredVar(lhs.name())
                              ->asValue(lhs.range(), method);
    if (lhsValue->type()->isSubtypeOf(TensorType::get())) {
      // for tensors, emit the corresponding in-place op
      const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()));
      const auto self = NamedValue(stmt.lhs().range(), "self", lhsValue);
      const auto output = emitBuiltinCall(
          stmt.range(),
          *method.graph(),
          getAugOp(stmt, /*isTensor=*/true),
          self,
          {rhs},
          {},
          /*required=*/true);

      environment_stack->setVar(lhs.range(), lhs.name().name(), output);
    } else {
      // for primitive types, desugar into a simple assignment
      //   e.g. foo += 1 becomes foo.2 = foo + 1
      Ident lhs = Var(stmt.lhs()).name();
      Expr expr = BinOp::create(
          stmt.range(),
          stmt.aug_op(),
          Var::create(lhs.range(), lhs),
          stmt.rhs());
      environment_stack->setVar(lhs.range(), lhs.name(), emitExpr(expr));
    }
  }

  void emitAugAssignmentToSubscript(const AugAssign& stmt) {
    // Process the base list value
    const auto lhs = Subscript(stmt.lhs());
    const auto sliceable = emitExpr(lhs.value());

    if (sliceable->type()->isSubtypeOf(TensorType::get())) {
      // If it's a tensor, just fully evaluate the subscript operation and emit
      // an in-place assignment
      std::vector<Value*> tensorIndices;
      Value* sliced;
      std::tie(sliced, tensorIndices) = emitIntAndSliceIndexing(
          lhs.range(), sliceable, lhs.subscript_exprs());

      const auto slicedArg = NamedValue(stmt.lhs().range(), "self", sliced);
      const auto rhs = NamedValue(stmt.rhs().range(), emitExpr(stmt.rhs()));
      if (tensorIndices.size() == 0) {
        // Common case: we only tried to index with int and slices. Emit the
        // correct augmented assignment op to the sliced value
        emitBuiltinCall(
            stmt.range(),
            *method.graph(),
            getAugOp(stmt, /*isTensor=*/true),
            slicedArg,
            {rhs},
            {},
            /*required=*/true);
      } else {
        // Special case: we tried to do "advanced indexing". Lower this expr
        // into `index` and `index_put_` ops with tensordices of Tensor?[]
        const auto indices = graph
                                 ->insertNode(graph->createList(
                                     OptionalType::ofTensor(), tensorIndices))
                                 ->output();
        const auto indexed =
            graph->insert(aten::index, {slicedArg, indices}, {}, stmt.range());
        const auto augmented = emitBuiltinCall(
            stmt.range(),
            *method.graph(),
            getAugOp(stmt, /*isTensor=*/true),
            indexed,
            {rhs},
            {},
            /*required=*/true);
        graph->insert(
            aten::index_put_,
            {slicedArg, indices, augmented},
            {},
            stmt.range());
      }
    } else {
      // Otherwise, it should be a list.  Lower this expression into:
      //     list.set_item(get_item(idx).add_(value))
      // similar to how Python handles things.
      const auto listType = sliceable->type()->cast<ListType>();
      AT_ASSERT(listType != nullptr);

      bool isTensorList =
          listType->getElementType()->isSubtypeOf(TensorType::get());

      // Get the idx to augment
      const auto subscriptExprs = lhs.subscript_exprs();
      if (subscriptExprs.size() != 1) {
        throw ErrorReport(subscriptExprs)
            << "Sliced expression not yet supported for"
            << " subscripted list augmented assignment. "
            << "File a bug if you want this.";
      }
      const auto idxValue = emitExpr(subscriptExprs[0]);

      const auto listArg = NamedValue(lhs.value().range(), "list", sliceable);
      const auto idxArg = NamedValue(subscriptExprs.range(), "idx", idxValue);
      const auto valueArg =
          NamedValue(stmt.rhs().range(), "value", emitExpr(stmt.rhs()));

      const auto getItem =
          graph->insert(aten::select, {listArg, idxArg}, {}, stmt.range());
      const auto augmentedItem = graph->insert(
          getAugOp(stmt, isTensorList), {getItem, valueArg}, {}, stmt.range());
      graph->insert(
          aten::_set_item, {listArg, idxArg, augmentedItem}, {}, stmt.range());
    }
  }

  // Emit mutating assignments like `foo[0] = bar`
  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const Expr& rhs) {
    emitSubscriptAssign(stmtRange, lhs, NamedValue(rhs.range(), emitExpr(rhs)));
  }

  void emitSubscriptAssign(
      const SourceRange& stmtRange,
      const Subscript& lhs,
      const NamedValue& rhs) {
    // First check the base value.
    auto sliceable = emitExpr(lhs.value());

    // If it's a tensor, copy the RHS data into it
    if (sliceable->type()->isSubtypeOf(TensorType::get())) {
      std::vector<Value*> tensorIndices;
      Value* sliced;
      // Handle multi-dimensional slicing: first emit int/slice indexing
      // TODO: the Python equivalent code has special-cased copy_to
      // broadcasting to match NumPy semantics (see PR#4853). We can't
      // replicate that without knowing the size of the Tensor; so really that
      // code should be moved into the aten function
      std::tie(sliced, tensorIndices) = emitIntAndSliceIndexing(
          lhs.range(), sliceable, lhs.subscript_exprs());

      const auto slicedArg = NamedValue(lhs.range(), sliced);
      if (tensorIndices.size() == 0) {
        // Common case: we only tried to index with int and slices. Copy the
        // RHS into the resulting tensor.
        graph->insert(aten::copy_, {slicedArg, rhs}, {}, stmtRange);
      } else {
        // Special case: we tried to do "advanced indexing" with a tensor.
        // Dispatch to `aten::index_put_` with tensorindices of Tensor?[]
        const auto indices = graph
                                 ->insertNode(graph->createList(
                                     OptionalType::ofTensor(), tensorIndices))
                                 ->output();

        graph->insert(
            aten::index_put_, {slicedArg, indices, rhs}, {}, stmtRange);
      }

      // Otherwise, this is a list. Dispatch to aten::_set_item to both select
      // and assign
    } else {
      const auto subscript = lhs.subscript_exprs();
      if (subscript.size() != 1 || subscript[0].kind() == TK_SLICE_EXPR) {
        throw ErrorReport(subscript)
            << "Sliced expression not yet supported for"
            << " subscripted list assignment. "
            << "File a bug if you want this.";
      }

      std::vector<NamedValue> args;
      args.emplace_back(lhs.value().range(), "list", sliceable);
      args.emplace_back(
          lhs.subscript_exprs().range(), "idx", emitExpr(subscript[0]));
      args.push_back(rhs);

      graph->insert(aten::_set_item, args, {}, stmtRange);
    }
  }

  void emitTupleAssign(const TupleLiteral& tl, const Expr& rhs) {
    size_t n_binders = tl.inputs().size();
    bool starred_unpack = calcNumStarredUnpack(tl.inputs(), tl.range());
    if (starred_unpack)
      n_binders--;
    auto output = emitSugaredExpr(rhs, n_binders);
    auto outputs = output->asTuple(
        rhs.range(),
        method,
        starred_unpack ? c10::nullopt : c10::optional<size_t>{n_binders});
    if (outputs.size() < n_binders) {
      throw ErrorReport(tl)
          << "need " << (starred_unpack ? "at least " : "") << n_binders
          << " values to unpack but found only " << outputs.size();
    }
    if (outputs.size() > n_binders && !starred_unpack) {
      throw ErrorReport(tl) << "too many values to unpack: need " << n_binders
                            << " but found " << outputs.size();
    }
    int i = 0;
    for (auto assignee : tl.inputs()) {
      switch (assignee.kind()) {
        case TK_SUBSCRIPT:
          emitSubscriptAssign(
              rhs.range(),
              Subscript(assignee),
              NamedValue(
                  rhs.range(), outputs.at(i)->asValue(rhs.range(), method)));
          i++;
          break;
        case TK_VAR:
          environment_stack->setSugaredVar(
              assignee.range(), Var(assignee).name().name(), outputs.at(i));
          i++;
          break;
        case TK_STARRED: {
          auto var = Starred(assignee).expr();
          if (var.kind() != TK_VAR) {
            throw ErrorReport(var)
                << "Cannot pack a tuple into a non-variable.";
          }
          size_t n_matched = outputs.size() - n_binders;
          ArrayRef<std::shared_ptr<SugaredValue>> outputs_ref = outputs;
          auto values = fmap(
              outputs_ref.slice(i, n_matched),
              [&](const std::shared_ptr<SugaredValue>& v) {
                return v->asValue(assignee.range(), method);
              });
          auto tup = graph->insertNode(graph->createTuple(values))->output();
          environment_stack->setVar(var.range(), Var(var).name().name(), tup);
          i += n_matched;
        } break;
        default:
          throw ErrorReport(assignee)
              << "unexpected expression on the left-hand side";
      }
    }
  }

  void emitAssignment(const Assign& stmt) {
    switch (stmt.lhs().kind()) {
      case TK_VAR: {
        auto v = Var(stmt.lhs());
        environment_stack->setSugaredVar(
            v.range(), v.name().name(), emitSugaredExpr(stmt.rhs(), 1));
      } break;
      case TK_TUPLE_LITERAL:
        emitTupleAssign(TupleLiteral(stmt.lhs()), stmt.rhs());
        break;
      case '.':
        emitSelectAssign(stmt);
        break;
      case TK_SUBSCRIPT:
        emitSubscriptAssign(stmt.range(), Subscript(stmt.lhs()), stmt.rhs());
        break;
      default:
        throw ErrorReport(stmt.lhs())
            << "unexpected expression on left-hand side of assignment.";
    }
  }

  void emitSelectAssign(const Assign& stmt) {
    const auto lhs = Select(stmt.lhs());
    const auto basename = Var(lhs.value()).name();
    const auto rhsValue =
        emitSugaredExpr(stmt.rhs(), 1)->asValue(stmt.rhs().range(), method);
    auto userObject = environment_stack->getSugaredVar(basename);
    userObject->setAttr(stmt.range(), method, lhs.selector().name(), rhsValue);
  }

  NodeKind getNodeKind(int kind, int ninputs) {
    switch (kind) {
      case '+':
        return aten::add;
      case '-':
        return aten::sub;
      case TK_UNARY_MINUS:
        return aten::neg;
      case '*':
        return aten::mul;
      case TK_POW:
        return aten::pow;
      case '@':
        return aten::matmul;
      case TK_STARRED:
        return prim::Starred;
      case '/':
        return aten::div;
      case '%':
        return aten::remainder;
      case TK_NE:
        return aten::ne;
      case TK_EQ:
        return aten::eq;
      case '<':
        return aten::lt;
      case '>':
        return aten::gt;
      case TK_LE:
        return aten::le;
      case TK_GE:
        return aten::ge;
      case TK_AND:
        return aten::__and__;
      case TK_OR:
        return aten::__or__;
      case TK_IS:
        return aten::__is__;
      case TK_ISNOT:
        return aten::__isnot__;
      case TK_NOT:
        return aten::__not__;
      case TK_FLOOR_DIV:
        return aten::floordiv;
      case '&':
        return aten::__and__;
      case '|':
        return aten::__or__;
      case '^':
        return aten::__xor__;
      default:
        throw std::runtime_error("unknown kind " + std::to_string(kind));
    }
  }

  std::vector<NamedValue> getNamedValues(
      const TreeList& trees,
      bool maybe_unpack) {
    std::vector<NamedValue> values;
    for (const auto& tree : trees) {
      if (maybe_unpack && tree->kind() == TK_STARRED) {
        auto starred = Starred(tree);
        auto entries = emitSugaredExpr(starred.expr(), 1)
                           ->asTuple(starred.range(), method);
        for (const auto& entry : entries) {
          values.emplace_back(
              tree->range(), entry->asValue(starred.range(), method));
        }
      } else {
        values.emplace_back(tree->range(), emitExpr(Expr(tree)));
      }
    }
    return values;
  }
  std::vector<NamedValue> getNamedValues(
      const List<Expr>& trees,
      bool maybe_unpack) {
    return getNamedValues(trees.tree()->trees(), maybe_unpack);
  }

  std::vector<Value*> getValues(const TreeList& trees, bool maybe_unpack) {
    return toValues(*graph, getNamedValues(trees, maybe_unpack));
  }
  std::vector<Value*> getValues(const List<Expr>& trees, bool maybe_unpack) {
    return getValues(trees.tree()->trees(), maybe_unpack);
  }

  std::vector<NamedValue> emitAttributes(const List<Attribute>& attributes) {
    return fmap(attributes, [&](const Attribute& attr) {
      return NamedValue(
          attr.range(), attr.name().name(), emitExpr(attr.value()));
    });
  }

  void checkApplyExpr(Apply& apply, SourceRange& loc) {
    if (apply.inputs().size() != 2) {
      throw ErrorReport(loc) << Var(apply.callee()).name().name()
                             << " expected exactly two arguments but found "
                             << apply.inputs().size();
    }
    if (apply.attributes().size() > 0) {
      throw ErrorReport(loc)
          << Var(apply.callee()).name().name() << " takes no keyword arguments";
    }
  }

  std::shared_ptr<SugaredValue> emitApplyExpr(Apply& apply, size_t n_binders) {
    auto sv = emitSugaredExpr(apply.callee(), 1);
    auto loc = apply.callee().range();
    if (auto fork_value = dynamic_cast<ForkValue*>(sv.get())) {
      auto& trees = apply.inputs().tree()->trees();
      if (trees.size() < 1) {
        throw ErrorReport(loc) << "Expected at least one argument to fork()";
      }

      auto forked = emitSugaredExpr(Expr(trees[0]), 1);
      TreeList sliced_trees(trees.begin() + 1, trees.end());
      auto inputs = getNamedValues(sliced_trees, true);
      auto attributes = emitAttributes(apply.attributes());
      return emitForkExpr(loc, forked, inputs, attributes);
    } else if (auto annotate_value = dynamic_cast<AnnotateValue*>(sv.get())) {
      checkApplyExpr(apply, loc);
      TypePtr type = typeParser_.parseTypeFromExpr(apply.inputs()[0]);
      Value* expr = tryConvertToType(
          apply.range(),
          *graph,
          type,
          emitExpr(apply.inputs()[1], type),
          /*allow_conversions=*/true);

      // This is to ensure even if user forgets to call annotate None with the
      // Optional wrapper type, we still generate the correct value with the
      // Optional type. e.g. it makes annoate(Tensor, None) to behave the same
      // with annotate(Optional[Tensor], None). It also maintains the backward
      // compatibility of exported model on Optional undefined tensor/None
      auto opt_type = expr->type()->cast<OptionalType>();
      bool forget_opt_annotate =
          opt_type && *opt_type->getElementType() == *type;

      if (!forget_opt_annotate && !expr->type()->isSubtypeOf(type)) {
        throw ErrorReport(apply.inputs())
            << "expected an expression of type " << type->python_str()
            << " but found " << expr->type()->python_str();
      }
      return std::make_shared<SimpleValue>(expr);
    } else if (auto getattr = dynamic_cast<GetAttrValue*>(sv.get())) {
      checkApplyExpr(apply, loc);
      auto obj = emitSugaredExpr(apply.inputs()[0], 1);
      auto selector = apply.inputs()[1];
      if (selector.kind() != TK_STRINGLITERAL) {
        throw ErrorReport(loc)
            << "getattr's second argument must be a string literal";
      }
      const std::string& name = StringLiteral(selector).text();
      return obj->attr(apply.range(), method, name);
    } else if (auto isinstance = dynamic_cast<IsInstanceValue*>(sv.get())) {
      // NOTE: for `isinstance` builtin call in JIT, we only check the static
      // types on the inputs to evaluate, and insert the corresponding constant
      // node
      std::function<bool(Expr, Expr)> isInstanceCheck = [&](Expr obj,
                                                            Expr classinfo) {
        if (classinfo.kind() == TK_TUPLE_LITERAL) {
          // handle the case for recursive tuple classinfo
          // return true if obj is an instance of any of the types
          for (Expr e : TupleLiteral(classinfo).inputs()) {
            if (isInstanceCheck(obj, e)) {
              return true;
            }
          }
          return false;
        }
        auto type_name = typeParser_.parseBaseTypeName(classinfo);
        if (!type_name) {
          throw ErrorReport(classinfo.range())
              << "type must be a type identifier";
        }
        auto val = emitExpr(obj);
        // Special casing for list and tuple since isintance(x, list) and
        // isinstance(x, tuple) does not accept List[int] / Tuple[int] like
        // subscript type annotation in python
        if (*type_name == "list" && val->type()->cast<ListType>()) {
          return true;
        } else if (*type_name == "tuple" && val->type()->cast<TupleType>()) {
          return true;
        } else if (val->type()->cast<OptionalType>()) {
          throw ErrorReport(loc)
              << "Optional isinstance check is not supported, "
              << "consider use is/isnot None instead";
        } else {
          TypePtr type = typeParser_.parseTypeFromExpr(classinfo);
          if (val->type()->isSubtypeOf(type)) {
            return true;
          }
        }
        return false;
      };
      checkApplyExpr(apply, loc);
      bool is_instance_val =
          isInstanceCheck(apply.inputs()[0], apply.inputs()[1]);
      return std::make_shared<SimpleValue>(
          graph->insertConstant(is_instance_val, nullptr, loc));
    } else if (auto classNew = dynamic_cast<ClassNewMethod*>(sv.get())) {
      if (apply.inputs().size() != 1) {
        throw ErrorReport(loc) << "Only one argument to __new__ allowed";
      }
      auto arg = emitSugaredExpr(apply.inputs()[0], 1);
      auto class_arg = dynamic_cast<ClassValue*>(arg.get());
      if (!class_arg) {
        throw ErrorReport(loc)
            << "Expected class value as argument to __new__, got "
            << arg->kind() << " instead";
      }
      if (class_arg->type_ != classNew->type_) {
        throw ErrorReport(loc) << "Argument to __new__() must match the class "
                               << "you are calling __new__() on. "
                               << "Got: " << class_arg->type_->str()
                               << ", expected: " << classNew->type_->str();
      }

      return classNew->createObject(apply.range(), method);
    } else {
      auto inputs = getNamedValues(apply.inputs(), true);
      auto attributes = emitAttributes(apply.attributes());
      return sv->call(loc, method, inputs, attributes, n_binders);
    }
  }

  BoolInfo findRefinements(const TreeRef& tree) {
    switch (tree->kind()) {
      case TK_IS:
      case TK_ISNOT: {
        const auto& inputs = tree->trees();
        if (inputs.at(0)->kind() == TK_VAR && inputs.at(1)->kind() == TK_NONE) {
          const std::string& var_name = Var(inputs[0]).name().name();
          Refinements true_info, false_info;
          auto type =
              environment_stack->getVar(var_name, inputs[0]->range())->type();
          if (auto opt_type = type->cast<OptionalType>()) {
            false_info.setRefinement(
                var_name,
                TypeAndRange(opt_type->getElementType(), &tree->range()));
            true_info.setRefinement(
                var_name, TypeAndRange(NoneType::get(), &tree->range()));
          }
          if (tree->kind() == TK_IS) {
            return BoolInfo(true_info, false_info);
          } else {
            return BoolInfo(false_info, true_info);
          }
        }
      } break;
      case TK_NOT: {
        const auto& inputs = tree->trees();
        auto bool_info = findRefinements(inputs[0]);
        return BoolInfo(
            bool_info.false_refinements_, bool_info.true_refinements_);
      }
      case TK_OR:
      case TK_AND: {
        const auto& inputs = tree->trees();
        auto first = findRefinements(inputs[0]);
        auto second = findRefinements(inputs[1]);
        if (tree->kind() == TK_OR) {
          return *first.mergeOr(second);
        } else {
          return *first.mergeAnd(second);
        }
      }
    }
    return BoolInfo();
  }

  Value* emitExpr(const Expr& tree, const TypePtr& type_hint = nullptr) {
    return emitSugaredExpr(tree, 1, type_hint)->asValue(tree.range(), method);
  }

  NodeKind reverseComparision(NodeKind kind) {
    if (kind == aten::lt) {
      return aten::gt;
    } else if (kind == aten::le) {
      return aten::ge;
    } else if (kind == aten::gt) {
      return aten::lt;
    } else if (kind == aten::ge) {
      return aten::le;
    }
    throw std::runtime_error(
        "reverseComparision: unsupported NodeKind. File a bug");
  }

  // any expression that can produce a SugaredValue is handled here
  // expressions that only return a single Value* are handled in emitSimpleExpr
  // type_hint is set if there is a type that this value is expected to be
  // e.g. a : List[int] = []
  // or a = torch.jit.annotate(List[int], [])
  // the caller is responsible for checking that the result matches type_hint
  // emitSugaredExpr is free to ignore it.
  std::shared_ptr<SugaredValue> emitSugaredExpr(
      const Expr& tree,
      size_t n_binders,
      const TypePtr& type_hint = nullptr) {
    switch (tree.kind()) {
      case TK_VAR:
        return environment_stack->getSugaredVar(Var(tree).name());
      case '.': {
        auto select = Select(tree);
        auto sv = emitSugaredExpr(select.value(), 1);
        return sv->attr(select.range(), method, select.selector().name());
      }
      case TK_APPLY: {
        auto apply = Apply(tree);
        return emitApplyExpr(apply, n_binders);
      } break;
      default:
        return std::make_shared<SimpleValue>(emitSimpleExpr(tree, type_hint));
    }
  }

  Value* emitNegate(const TreeRef& tree) {
    const auto& inputs = tree->trees();
    auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);

    auto neg_val = emitBuiltinCall(
        tree->range(),
        *method.graph(),
        aten::neg,
        c10::nullopt,
        named_values,
        {},
        /*required=*/true);

    // constant fold the input if possible
    auto maybe_constant_input = toIValue(neg_val->node()->input());
    if (!maybe_constant_input) {
      return neg_val;
    }
    auto op = getOperation(neg_val->node());
    Stack stack;
    stack.push_back(*maybe_constant_input);
    op(stack);
    AT_ASSERT(stack.size() == 1);
    return graph->insertConstant(stack[0], nullptr, tree->range());
  }

  // This function extract a new graph from its original subgraph
  std::shared_ptr<SugaredValue> emitForkExpr(
      SourceRange loc,
      const std::shared_ptr<SugaredValue>& forked,
      at::ArrayRef<NamedValue> inputs,
      at::ArrayRef<NamedValue> attributes) {
    // Build the fork node without inputs
    auto fork_node =
        method.graph()
            ->insertNode(method.graph()->create(prim::fork, 1))
            ->setSourceLocation(std::make_shared<SourceRange>(loc));
    auto body_block = fork_node->addBlock();

    // Build a template of the graph to be executed
    Value* node_output;
    {
      WithInsertPoint guard(body_block);
      auto fn_sugared_output = forked->call(loc, method, inputs, attributes, 1);
      auto fn_simple_output = fn_sugared_output->asValue(loc, method);
      body_block->registerOutput(fn_simple_output);
      node_output = fork_node->output()->setType(
          FutureType::create(fn_simple_output->type()));
    }
    // Lambda lift block(0) into attr::Subgraph
    lambdaLiftFork(fork_node);
    runCleanupPasses(fork_node->g(attr::Subgraph));
    return std::make_shared<SimpleValue>(node_output);
  }

  Value* emitSimpleExpr(
      const TreeRef& tree,
      const TypePtr& type_hint = nullptr) {
    switch (tree->kind()) {
      case '@':
      case TK_POW:
      case TK_IS:
      case TK_ISNOT:
      case TK_NE:
      case TK_EQ:
      case '<':
      case '>':
      case TK_LE:
      case TK_GE:
      case '*':
      case '/':
      case '+':
      case '-':
      case '%':
      case '&':
      case '|':
      case '^':
      case TK_FLOOR_DIV: {
        const auto& inputs = tree->trees();
        auto kind = getNodeKind(tree->kind(), inputs.size());
        auto named_values = getNamedValues(inputs, /*maybe_unpack=*/false);
        return emitBuiltinCall(
            tree->range(),
            *method.graph(),
            kind,
            c10::nullopt,
            named_values,
            {},
            /*required=*/true);
      }
      case TK_NOT: {
        Value* input = emitCond(Expr(tree->trees()[0]));
        return emitBuiltinCall(
            tree->range(),
            *method.graph(),
            aten::__not__,
            c10::nullopt,
            {input},
            {},
            /*required=*/true);
      }

      case TK_UNARY_MINUS: {
        return emitNegate(tree);
      }
      case TK_AND:
      case TK_OR: {
        const auto& inputs = tree->trees();
        return emitShortCircuitIf(
            tree->range(), inputs[0], inputs[1], tree->kind() == TK_OR);
      }
      case TK_STARRED: {
        throw ErrorReport(tree)
            << "Unexpected starred expansion. File a bug report.";
      }
      case TK_CONST: {
        return emitConst(Const(tree));
      } break;
      case TK_TRUE: {
        return graph->insertConstant(true, nullptr, tree->range());
      } break;
      case TK_FALSE: {
        return graph->insertConstant(false, nullptr, tree->range());
      } break;
      case TK_NONE: {
        return graph->insertConstant(IValue(), nullptr, tree->range());
      } break;
      case TK_SUBSCRIPT: {
        return emitSubscript(Subscript(tree));
      } break;
      case TK_IF_EXPR: {
        return emitTernaryIf(TernaryIf(tree));
      } break;
      case TK_STRINGLITERAL: {
        return emitStringLiteral(StringLiteral(tree));
      } break;
      case TK_LIST_LITERAL: {
        auto ll = ListLiteral(tree);
        auto values = getValues(ll.inputs(), /*maybe_unpack=*/true);

        // determine the element type of the list
        // if we have a type hint of List[T], use T
        // if the list is non-empty use type_of(list[0])
        // otherwise assume it is List[Tensor]
        TypePtr elem_type = TensorType::get();
        if (type_hint && type_hint->kind() == TypeKind::ListType) {
          elem_type = type_hint->expect<ListType>()->getElementType();
        } else if (!values.empty()) {
          elem_type = values.at(0)->type();
        }

        // Tensors are special because they have dymnamic properties. So any
        // list containing tensors should be typed with the unified typeof all
        // the elements.
        if (elem_type->isSubtypeOf(TensorType::get())) {
          for (const auto& value : values) {
            elem_type = unifyTypes(elem_type, value->type()).value();
          }
        }
        for (auto v : values) {
          if (!v->type()->isSubtypeOf(elem_type)) {
            throw ErrorReport(tree)
                << "Lists must contain only a single type, expected: "
                << *elem_type << " but found " << *v->type() << " instead";
          }
        }
        Value* result =
            graph->insertNode(graph->createList(elem_type, values))->output();
        return result;
      } break;
      case TK_TUPLE_LITERAL: {
        auto ll = TupleLiteral(tree);
        auto values = getValues(ll.inputs(), /*maybe_unpack=*/true);
        return graph->insertNode(graph->createTuple(values))->output();
      } break;
      case TK_DICT_LITERAL: {
        auto dl = DictLiteral(tree);
        auto key_trees = dl.key_inputs().tree()->trees();
        auto value_trees = dl.value_inputs().tree()->trees();
        AT_ASSERT(key_trees.size() == value_trees.size());
        std::vector<Value*> keys, values;
        for (size_t i = 0; i < key_trees.size(); ++i) {
          keys.push_back(emitExpr(Expr(key_trees[i])));
          values.push_back(emitExpr(Expr(value_trees[i])));
        }

        TypePtr key_type = nullptr;
        TypePtr value_type = nullptr;

        if (type_hint && type_hint->kind() == TypeKind::DictType) {
          auto dict_type = type_hint->expect<DictType>();
          key_type = dict_type->getKeyType();
          value_type = dict_type->getValueType();
        } else if (!keys.empty()) {
          key_type = keys.at(0)->type();
          value_type = values.at(0)->type();
        } else {
          key_type = StringType::get();
          value_type = TensorType::get();
        }
        AT_ASSERT(key_type != nullptr && value_type != nullptr);

        return graph
            ->insertNode(graph->createDict(key_type, value_type, keys, values))
            ->output();
      } break;
      case TK_LIST_COMP: {
        auto lc = ListComp(tree);
        return emitListComprehension(lc);
      } break;
      default:
        throw ErrorReport(tree) << "Cannot emit expr for: " << tree;
        break;
    }
  }

  Value* emitConst(const Const& c) {
    if (c.isFloatingPoint())
      return materializeConstant(
          c.asFloatingPoint(), *graph, c.range(), fp_constants);
    else
      return materializeConstant(
          c.asIntegral(), *graph, c.range(), integral_constants);
  }

  Value* emitStringLiteral(const StringLiteral& c) {
    return insertConstant(*graph, c.text(), nullptr, c.range());
  }

  // Desugars select indexing: tensor[i] -> tensor.select(dim, i)
  Value* emitSelect(
      const SourceRange& loc,
      Value* input,
      Value* dim,
      Value* index) {
    return emitBuiltinCall(
        loc, *graph, aten::select, c10::nullopt, {input, dim, index}, {}, true);
  }

  // Desugars slice indexing: tensor[begin:end] -> tensor.slice(dim, begin, end,
  // 1)
  Value* emitSlice(
      const SourceRange& loc,
      Value* input,
      Value* dim, // Only used for tensor slicing
      const SliceExpr& slice) {
    std::vector<NamedValue> args;
    args.reserve(4);
    args.emplace_back(loc, "self", input);

    // XXX: If list slicing becomes more complicated or stops using
    // aten::slice, we should separate it from this function.
    if (dim) {
      AT_ASSERT(input->type()->isSubtypeOf(TensorType::get()));

      args.emplace_back(dim);
    } else {
      AT_ASSERT(!input->type()->isSubtypeOf(TensorType::get()));
    }

    args.emplace_back(loc, "begin", emitExpr(Expr(slice.startOr(0))));
    const auto has_end = slice.end().present();
    if (has_end) {
      args.emplace_back(loc, "end", emitExpr(Expr(slice.end().get())));
    }
    if (input->type()->cast<TupleType>()) {
      if (has_end) {
        return emitTupleSlice(loc, args[0], args[1], /*end*/ args[2]);
      } else {
        return emitTupleSlice(loc, args[0], args[1], c10::nullopt);
      }
    }
    NamedValue step =
        NamedValue(loc, "step", graph->insertConstant(1, nullptr, loc));
    return emitBuiltinCall(
        loc, *graph, aten::slice, c10::nullopt, args, {step}, true);
  }

  Value* emitUnsqueeze(const SourceRange& loc, Value* input, int64_t dim) {
    return emitBuiltinCall(
        loc,
        *graph,
        aten::unsqueeze,
        c10::nullopt,
        {input, graph->insertConstant(dim, nullptr, loc)},
        {},
        true);
  }

  Value* emitIndex(
      const SourceRange& loc,
      Value* input,
      at::ArrayRef<Value*> indices) {
    // NB: the index of aten::index should be a type of List[Optional[Tensor]],
    // this is to support the case like t[:, :, 1] where : here indicates a
    // None/undefined tensor(optional tensor)
    auto* index =
        graph->insertNode(graph->createList(OptionalType::ofTensor(), indices))
            ->output();
    return emitBuiltinCall(
        loc, *graph, aten::index, c10::nullopt, {input, index}, {}, true);
  }

  // Emits multidimensional slicing with int and slice indices.
  // Returns:
  // - Value*: the input after it has been indexed by int and slice indices.
  // - vector<Value*>: A list of tensor Value* indices that have not been
  // applied yet.
  //   Should be NULL at indices where sliceable (post-slicing) isn't indexed by
  //   a tensor.
  std::pair<Value*, std::vector<Value*>> emitIntAndSliceIndexing(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    std::vector<Value*> tensor_indices;
    size_t dim = 0;

    auto handle_tensor = [&](Value* tensor) {
      // NB: tensor_indices can have None holes because of how at::index works.
      tensor_indices.resize(dim + 1);
      tensor_indices[dim] = tensor;
      dim++;
    };

    // before ellipsis, dimension index should be `dim`
    // after ellipsis, dimension index should be `-offset`
    int offset = 0;
    size_t ellipsis_dim = 0;
    auto insert_value_for_dim = [&](int64_t dim) {
      return (offset == 0)
          ? graph->insertConstant(dim, nullptr, loc)
          :
          // NB: offset is incremented to move to the next dimension index
          graph->insertConstant(offset++, nullptr, loc);
    };

    for (const auto& subscript_expr : subscript_exprs) {
      // NB: ellipsis_dim is **always** incremented
      // (comparing to dim) in order to compute
      // the correct offsets for the remaining
      // dimension indices following an ellipsis "..."
      // token
      ellipsis_dim++;
      if (subscript_expr.kind() == TK_DOTS) {
        offset = -(subscript_exprs.size() - ellipsis_dim);
        ++dim;
        continue;
      }
      if (subscript_expr.kind() == TK_SLICE_EXPR) {
        auto dim_val = insert_value_for_dim(dim);
        sliceable =
            emitSlice(loc, sliceable, dim_val, SliceExpr(subscript_expr));
        ++dim;
        continue;
      }
      auto index = emitExpr(subscript_expr, OptionalType::ofTensor());
      if (index->type() == IntType::get()) {
        // NB: note, select squeezes out a dimension,
        // so dim is **not** incremented
        auto dim_val = insert_value_for_dim(dim);
        sliceable = emitSelect(loc, sliceable, dim_val, index);
        continue;
      } else if (index->type()->isSubtypeOf(NoneType::get())) {
        sliceable = emitUnsqueeze(loc, sliceable, dim);
        dim++;
        continue;
      } else if (index->type()->isSubtypeOf(OptionalType::ofTensor())) {
        // NB:index type can either be a Tensor or : (None of Optional Tensor)
        handle_tensor(index);
        continue;
      }
      throw ErrorReport(loc)
          << "Unsupported operation: indexing tensor with unsupported index type '"
          << index->type()->str()
          << "'. Only ints, slices, and tensors are supported";
    }
    // at::index takes in a List[Optional[Tensor]] where some dims can be None.
    // create None node with optional tensor output type and pass to at::index.
    for (auto& index : tensor_indices) {
      if (index == nullptr) {
        index =
            graph->insertNode(graph->createNone(TensorType::get()))->output();
      }
    }
    return std::make_pair(sliceable, tensor_indices);
  }

  // Desugars multidim slicing into slice/select/index/unsqueeze calls.
  //
  // XXX: Errors in user code are not elegantly reported.
  // Let's say someone were to do the following:
  //   @torch.jit.script
  //   def fn(x):
  //       return x[0, 1]
  //   fn(torch.randn(5))
  // Because we desugar this into two aten::select ops, the error message
  // complains about aten::select failing rather than there "not being
  // enough dimensions to index".
  //
  // The strategy is to slice and select the tensor for int and slices first
  // in one pass and then apply at::index on the result of the
  // slicing/selecting. Call the tensor after we've applied slice / select the
  // `sliced`. tensor_indices should have the same size as sliced.dim():
  // - tensor_indices[i] = NULL if we should not index `sliced` at dim i
  // - tensor_indices[i] = t if we should index `sliced` at dim i with tensor t.
  Value* emitMultidimSlicing(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    if (!sliceable->type()->isSubtypeOf(TensorType::get())) {
      throw ErrorReport(loc)
          << "Unsupported operation: attempted to use multidimensional "
          << "indexing on a non-tensor type.";
    }

    std::vector<Value*> tensor_indices;
    std::tie(sliceable, tensor_indices) =
        emitIntAndSliceIndexing(loc, sliceable, subscript_exprs);

    if (tensor_indices.empty()) {
      // XXX: Might need to at::alias this when we support mutability
      return sliceable;
    }

    return emitIndex(loc, sliceable, tensor_indices);
  }

  // Desugars slice syntactic sugar tensor[begin:end] -> tensor.slice(begin,
  // end).
  Value* emitBasicSlice(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    AT_ASSERT(subscript_exprs.size() == 1);
    AT_ASSERT(subscript_exprs[0].kind() == TK_SLICE_EXPR);
    auto slice_exp = SliceExpr(subscript_exprs[0]);
    Value* maybe_dim = nullptr;
    if (sliceable->type()->isSubtypeOf(TensorType::get())) {
      // If the sliceable object is a tensor, specify a default dimension
      maybe_dim = graph->insertConstant(0, nullptr, loc);
    }
    return emitSlice(loc, sliceable, maybe_dim, slice_exp);
  }

  int64_t getTupleIndexVal(
      const SourceRange& loc,
      const TupleTypePtr& tuple_type,
      Value* idx_val,
      bool allow_out_of_bounds) {
    int64_t index;
    at::optional<IValue> ivalue = toIValue(idx_val);
    if (ivalue && ivalue->isInt()) {
      index = ivalue->to<int64_t>();
    } else {
      throw ErrorReport(loc) << "tuple indices must be integer constants";
    }
    // set index to be positive to simplify logic in runtime
    int64_t adj_index = index;
    int64_t tuple_len = tuple_type->elements().size();
    if (index < 0) {
      adj_index = tuple_len + index;
    }
    if (!allow_out_of_bounds && (adj_index >= tuple_len || adj_index < 0)) {
      throw ErrorReport(loc) << "Tuple index out of range. Tuple is length "
                             << tuple_len << " and index is " << index;
    }
    return adj_index;
  }

  Value* emitTupleIndex(
      const SourceRange& loc,
      Value* tuple_val,
      Value* idx_val) {
    auto tuple_typ = tuple_val->type()->cast<TupleType>();
    auto adj_index = getTupleIndexVal(
        loc, tuple_typ, idx_val, /*allow_out_of_bounds*/ false);
    return graph->insertNode(graph->createTupleIndex(tuple_val, adj_index))
        ->output();
  }

  Value* emitDictIndex(
      const SourceRange& loc,
      Value* dict_val,
      Value* key_val) {
    auto dict_type = dict_val->type()->cast<DictType>();
    AT_ASSERT(key_val->type()->isSubtypeOf(dict_type->getKeyType()));
    return graph->insertNode(graph->createDictIndex(dict_val, key_val))
        ->output();
  }

  Value* emitTupleSlice(
      const SourceRange& loc,
      const NamedValue& tuple_val,
      const NamedValue& beg_val,
      const at::optional<NamedValue>& end_val) {
    auto tuple_type = tuple_val.value(*graph)->type()->expect<TupleType>();
    int64_t beg = getTupleIndexVal(
        loc, tuple_type, beg_val.value(*graph), /*allow_out_of_bounds*/ true);
    int64_t end;
    int64_t tuple_len = tuple_type->elements().size();
    if (end_val) {
      end = getTupleIndexVal(loc, tuple_type, end_val->value(*graph), true);
    } else {
      end = tuple_len;
    }
    // slicing does not throw out of bounds errors
    end = std::min(std::max((int64_t)0, end), tuple_len);
    beg = std::min(std::max((int64_t)0, beg), tuple_len);

    return graph
        ->insertNode(graph->createTupleSlice(tuple_val.value(*graph), beg, end))
        ->output();
  }

  Value* emitSubscript(const Subscript& subscript) {
    return emitSubscript(
        subscript.range(),
        emitExpr(subscript.value()),
        subscript.subscript_exprs());
  }

  Value* emitSubscript(
      const SourceRange& loc,
      Value* sliceable,
      const List<Expr>& subscript_exprs) {
    if (subscript_exprs.size() != 1) {
      return emitMultidimSlicing(loc, sliceable, subscript_exprs);
    }
    if (subscript_exprs[0].kind() == TK_SLICE_EXPR) {
      return emitBasicSlice(loc, sliceable, subscript_exprs);
    } else {
      return emitBasicGather(loc, sliceable, subscript_exprs);
    }
  }

  // Desugars gather syntactic sugar foo[i]
  Value* emitBasicGather(
      const SourceRange& loc,
      Value* gatherable,
      const List<Expr>& subscript_exprs) {
    AT_ASSERT(subscript_exprs.size() == 1);

    if (gatherable->type()->kind() == TypeKind::ListType) {
      // if it's a list, emit a regular index selection op
      auto* idx = emitExpr(subscript_exprs[0]);
      return emitBuiltinCall(
          loc, *graph, aten::select, c10::nullopt, {gatherable, idx}, {}, true);
    } else if (gatherable->type()->isSubtypeOf(TensorType::get())) {
      return emitMultidimSlicing(loc, gatherable, subscript_exprs);
    } else if (auto tuple_type = gatherable->type()->cast<TupleType>()) {
      auto* idx = emitExpr(subscript_exprs[0]);
      return emitTupleIndex(loc, gatherable, idx);
    } else if (auto dict_type = gatherable->type()->cast<DictType>()) {
      auto* idx = emitExpr(subscript_exprs[0]);
      return emitDictIndex(loc, gatherable, idx);
    } else if (auto string_type = gatherable->type()->cast<StringType>()) {
      auto* idx = emitExpr(subscript_exprs[0]);
      return emitBuiltinCall(
          loc,
          *graph,
          prim::StringIndex,
          c10::nullopt,
          {gatherable, idx},
          {},
          true);
    } else {
      throw ErrorReport(loc) << "Indexing only supported on List, Dict, "
                                "Tensor, Tuple, and str but got type '"
                             << gatherable->type()->python_str() << "'";
    }
  }
};

struct MethodResolver : public Resolver {
  explicit MethodResolver(
      const Resolver* otherResolver,
      const std::unordered_map<std::string, std::shared_ptr<Function>>& functionTable)
      : otherResolver_(otherResolver), functionTable_(functionTable) {}

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const SourceRange& loc) const override {
    auto it = functionTable_.find(name);
    if (it != functionTable_.end()) {
      return std::make_shared<MethodValue>(c10::nullopt, it->second);
    }
    return otherResolver_->resolveValue(name, m, loc);
  }

  TypePtr resolveType(const std::string& name) const override {
    return otherResolver_->resolveType(name);
  }

 private:
  const Resolver* otherResolver_;
  const std::unordered_map<std::string, std::shared_ptr<Function>>& functionTable_;
};

void CompilationUnit::define(
    const std::vector<Def>& definitions,
    const std::vector<ResolverPtr>& resolvers,
    const Self& self) {
  AT_ASSERT(definitions.size() == resolvers.size());
  auto resolver_it = resolvers.begin();
  std::vector<Function*> methods;
  std::unordered_map<std::string, std::shared_ptr<Function>> function_table;
  for (const Def& def : definitions) {
    const std::string& name = def.name().name();
    ResolverPtr resolver = *resolver_it++;
    AT_ASSERT(resolver);
    if (!self) {
      // if self is defined, then these are methods and do not go into the
      // global namespace otherwise, they get defined together so we add them to
      // the function table so the methods can see each other
      resolver =
          std::make_shared<MethodResolver>(resolver.get(), function_table);
    }
    auto creator = [def, resolver, self](Function& method) {
      AT_ASSERT(resolver);
      to_ir(def, resolver, self, method);
    };
    std::shared_ptr<Function> fn(
        new Function(name, is_optimized(), std::make_shared<Graph>(), creator));
    function_table[name] = fn;
    methods.push_back(fn.get());
    register_function(std::move(fn));
  }
  for (Function* method : methods) {
    method->ensure_defined();
  }
}

void CompilationUnit::define(
    const std::string& source,
    const ResolverPtr& resolver,
    const Self& self) {
  Parser p(source);
  std::vector<Def> definitions;
  std::vector<ResolverPtr> resolvers;
  while (p.lexer().cur().kind != TK_EOF) {
    auto def = Def(p.parseFunction(/*is_method=*/bool(self)));
    definitions.push_back(def);
    resolvers.push_back(resolver);
  }
  define(definitions, resolvers, self);
}

void lambdaLiftFork(Node* fork_node) {
  // Fork a new graph from its orignal owning graph
  auto forked_graph = std::make_shared<Graph>();
  auto body_block = fork_node->blocks()[0];

  // Make sure we capture everything in the new graph.
  // The uncaptured values will be added to the fork signature.
  std::unordered_map<Value*, Value*> uncaptures_map;
  auto env = [&](Value* v) -> Value* {
    if (!uncaptures_map.count(v)) {
      // Capture values for both graphs
      uncaptures_map[v] =
          forked_graph->addInput()->copyMetadata(v);
      fork_node->addInput(v);
    }
    return uncaptures_map[v];
  };

  forked_graph->block()->cloneFrom(body_block, env);

  // Separate the subgraph and clean up the orignal one
  fork_node->g_(attr::Subgraph, forked_graph);
  fork_node->eraseBlock(0);
}

} // namespace script
} // namespace jit
} // namespace torch
