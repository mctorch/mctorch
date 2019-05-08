#pragma once

#include <torch/csrc/jit/attributes.h>
#include <torch/csrc/jit/graph_node_list.h>
#include <torch/csrc/jit/named_value.h>
#include <torch/csrc/jit/scope.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/utils/disallow_copy.h>
#include <torch/csrc/utils/python_stub.h>

#include <ATen/ATen.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <functional>
#include <iostream>
#include <unordered_set>
#include <vector>

// Forward declare, the real meat is in python_ir.cpp
template<class T>
class THPPointer;
using THPObjectPtr = THPPointer<PyObject>;
using pyobj_list = std::vector<THPObjectPtr>;

namespace torch {
namespace jit {

using ::c10::Argument;
using ::c10::FunctionSchema;
using ::c10::Symbol;

using ::c10::ivalue::List;
using ::c10::ivalue::Shared;

using ::c10::IValue;
using ::c10::ivalue::Future;
using ::c10::ivalue::Tuple;

using ::c10::ivalue::BoolList;
using ::c10::ivalue::DoubleList;
using ::c10::ivalue::GenericList;
using ::c10::ivalue::IntList;
using ::c10::ivalue::TensorList;

using ::c10::ivalue::ConstantString;

#define C10_USING(T) using ::c10::T;
C10_FORALL_TYPES(C10_USING)
#undef C10_USING

#define C10_USING(T) using ::c10::T##Ptr;
C10_FORALL_TYPES(C10_USING)
#undef C10_USING

using ::c10::Type;
using ::c10::TypeEnv;
using ::c10::TypePtr;

using ::c10::getTypePtr;
using ::c10::MatchTypeReturn;
using ::c10::TypeKind;

using ::c10::fmap;

namespace prim {
using namespace ::c10::prim;
}
namespace attr {
using namespace ::c10::attr;
}
namespace aten {
using namespace ::c10::aten;
}

// Graph represents one "function" of computation.
// It uses a simple ownership model where the graph owns all the nodes inside
// it. All references inside the graph are raw pointers. Destroying the Graph
// will invalidate any pointers to nodes in the graph.
struct Graph;

// Node is the base class of the IR graph. It represents one computation
// and dependencies on a list of Values. The "prim-ops", so to speak.
struct Node;

// A Value represents an input or output to node that is either a
// Tensor or an opaque Handle object, as determined by type().
struct Value;

TORCH_API std::ostream& operator<<(std::ostream& out, const Graph& g);
TORCH_API std::ostream& operator<<(std::ostream& out, const Node& n);

// A list of nodes, with inputs and outputs
struct Block;

// Each use is represented by this type, see Node::uses()
// 'user' is the consumer of the value, offset is the index into
// 'user's input this where the produces will be found.
struct Use {
  Use(Node* user, size_t offset) : user(user), offset(offset) {}
  Node* user;
  size_t offset;

  bool operator==(const Use& b) {
    return user == b.user && offset == b.offset;
  }
};

// Note [User node does not uniquely identify use]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A while back, we wrote some code manipulating uses that looked like this:
//
//    for (auto& use : used_val->uses_) {
//      if (use.user == this_node) {
//        use.offset += 1;
//        break;
//      }
//    }
//
// This code is trying to find a particular use (our node's use) to update it.
// However, it's wrong: there may be *multiple* uses of a value %x in a node,
// as might be the case in this IR:
//
//    %y = Add %x %x
//
// In this case, there are two uses of %x whose user is the node 'Add %x %x'.
// So, "use induced by this node" is not a well-formed concept.
//
// If you are looking for "use induced by an input", it's best to use
// findUseForInput() to get it.

// the list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier
using node_list = std::vector<Node*>;
using value_list = std::vector<Value*>;
using use_list = std::vector<Use>;
template <typename T>
using ArrayRef = at::ArrayRef<T>;
using NodeKind = Symbol;
using topo_position_t = int64_t;
using ValueSet = std::unordered_set<const Value*>;

struct Value {
  TH_DISALLOW_COPY_AND_ASSIGN(Value);
  Value(Node* node_, size_t offset_);

 private:
  friend struct Node;
  friend struct Graph;
  Node* node_;
  size_t offset_;
  size_t unique_ = 0; // unique id
  use_list uses_;
  std::string unique_name_;
  TypePtr type_;

 public:
  Value* setType(TypePtr type);
  void inferTypeFrom(const at::Tensor& output) {
    setType(CompleteTensorType::create(output));
  }
  const TypePtr& type() const {
    AT_ASSERT(type_ != nullptr);
    return type_;
  }
  bool requires_grad() const {
    return type()->requires_grad();
  }
  bool isCompleteTensor() const {
    return type()->kind() == TypeKind::CompleteTensorType;
  }
  TORCH_API bool mustBeNone() const;
  TORCH_API bool mustNotBeNone() const;
  size_t unique() const {
    return unique_;
  }
  bool hasUniqueName() const {
    return !unique_name_.empty();
  }
  static bool isValidName(const std::string& name);
  TORCH_API Value* setUniqueName(const std::string& name);
  std::string uniqueName() const {
    if (hasUniqueName()) {
      return unique_name_;
    }
    return std::to_string(unique());
  }
  TORCH_API std::string uniqueNameBase() const;
  Node* node() {
    return node_;
  }
  size_t offset() const {
    return offset_;
  }
  void setOffset(size_t offset) {
    offset_ = offset;
  }
  const Node* node() const {
    return node_;
  }
  Graph* owningGraph();
  const Graph* owningGraph() const;
  // TODO: make this more const correct
  const use_list& uses() const {
    return uses_;
  }

  bool hasUses() const {
    return !uses().empty();
  }

  TORCH_API void replaceFirstUseWith(Value* newValue);

  // Replaces all uses of this value with 'newValue'.
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%3, %3)
  // Execute: %3.replaceAllUsesWith(%6)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%6)
  //          %5 = h(%6, %6)
  TORCH_API void replaceAllUsesWith(Value* newValue);

  TORCH_API Value* copyMetadata(Value* from);
};

struct TORCH_API Node {
  TH_DISALLOW_COPY_AND_ASSIGN(Node);
  friend struct Graph;
  friend struct Block;
  friend struct Value;
  friend graph_node_list;
  friend const_graph_node_list;
  friend graph_node_list_iterator;
  friend const_graph_node_list_iterator;

 private:
  const NodeKind kind_;
  std::vector<Value*> inputs_;
  std::vector<Value*> outputs_;
  // subblocks
  std::vector<Block*> blocks_;
  Graph* graph_;
  Block* owning_block_;
  std::shared_ptr<SourceLocation> source_location_;
  ScopePtr scope_;
  // Assumes FunctionSchemas are persistent, so we don't manage their lifetime.
  // This field is effective a cache that's populated on attribute lookups and
  // invalidated every time we perform an operation that could potentially
  // change the schema. note: mutable because schema_ is effectively a cache
  mutable const FunctionSchema* schema_;
  topo_position_t topo_position_ = 0;

 protected:
  Node(Graph* graph_, NodeKind kind_); // defined after graph
 public:
  // each node but Return/Param
  // is associated with exactly one place in the node list...
  // of the graph_
  // this circular is a doubly-linked list, the Return node is used as the
  // sentinel for the beginning and end of the list such that the list never has
  // null pointers next_in_graph[0] is next pointer next_in_graph[1] is prev
  // pointer using an array to allow the same iterator class for forward and
  // reverse node lists This list represents a topological sort
  Node* next_in_graph[2] = {nullptr, nullptr};

  Node*& next() {
    return next_in_graph[kNextDirection];
  }
  Node*& prev() {
    return next_in_graph[kPrevDirection];
  }
  Node* const& next() const {
    return next_in_graph[kNextDirection];
  }
  Node* const& prev() const {
    return next_in_graph[kPrevDirection];
  }

  NodeKind kind() const {
    return kind_;
  }
  Node* setSourceLocation(std::shared_ptr<SourceLocation> sl) {
    source_location_ = std::move(sl);
    return this;
  }
  std::shared_ptr<SourceLocation> getSourceLocation() const {
    return source_location_;
  }
  Graph* owningGraph() {
    return graph_;
  }
  const Graph* owningGraph() const {
    return graph_;
  }
  Block* owningBlock() {
    return owning_block_;
  }
  const Block* owningBlock() const {
    return owning_block_;
  }
  ScopePtr scope() {
    return scope_;
  }
  void setScope(ScopePtr scope) {
    scope_ = std::move(scope);
  }
  std::string scopeName() const {
    if (!scope_) {
      return "";
    }
    return scope_->namesFromRoot();
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  at::ArrayRef<Value*> inputs() {
    return inputs_;
  }
  at::ArrayRef<const Value*> inputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {inputs_.data(), inputs_.size()};
  }
  // NB: This returns an ArrayRef; that means that it will
  // get invalidated if you resize inputs (e.g., using addInput)
  // We can't return a std::vector<Node*>& because there's no
  // way to soundly cast to std::vector<const Node*> (an insane
  // implementation of std::vector could make this representationally
  // different.)
  at::ArrayRef<Value*> outputs() {
    return outputs_;
  }
  at::ArrayRef<const Value*> outputs() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {outputs_.data(), outputs_.size()};
  }
  Value* output(size_t i) const {
    return outputs_.at(i);
  }
  bool hasUses() const {
    for (auto o : outputs()) {
      if (!o->uses().empty()) {
        return true;
      }
    }
    return false;
  }

  void replaceAllUsesWith(Node* n);

  // lots of things like chunk have a single input or single output, so we have
  // a helper to make accessing it easier
  Value* input() {
    AT_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  Value* output() {
    AT_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const Value* output() const {
    AT_ASSERT(outputs_.size() == 1);
    return outputs_.at(0);
  }
  const Value* input() const {
    AT_ASSERT(inputs_.size() == 1);
    return inputs_.at(0);
  }
  // Access a particular input.  This is a checked index.
  Value* input(size_t i) const {
    return inputs_.at(i);
  }

  Value* namedInput(Symbol name) const;

  c10::optional<IValue> get(Symbol name) const;

  template <typename T>
  c10::optional<T> get(Symbol name) const {
    if (auto v = get(name)) {
      return v->template to<T>();
    }
    return c10::nullopt;
  }

  // Returns true if the value of input name is statically known
  bool is_constant(Symbol name) const {
    return static_cast<bool>(get(name));
  }
  bool mustBeNone() const;

  bool isNondeterministic() const;
  bool hasSideEffects() const;

  // Graphs

  // Note [Topological invariant]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // We always maintain an up-to-date topological ordering of all nodes via
  // the next()/prev() links.  All transformations to graphs must preserve
  // this topological ordering: for example, it is only valid to 'addInput'
  // with an input which is topologically before the current node.
  //
  // Usually, it is obvious whether or not topological order is maintained;
  // for example, if you are adding nodes to the end of the topsort, it's
  // impossible for them to refer to inputs that are not in the topsort.
  // If it is not obvious, please comment accordingly.

  // Add 'node' as an input to 'this' at the end of existing
  // arguments.  Returns the added node for ease of chaining.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.addInput(%4)
  // Result:  %3 = f(%1, %2, %4)
  Value* addInput(Value* value);

  // Add 'value' as an input to 'this' at the specified position in the
  // arguments. Returns the added value for ease of chaining.
  Value* insertInput(size_t i, Value* value);

  // Replace the input of 'this' at position 'i' with
  // 'newValue', returning the old node.
  //
  // Given:   %3 = f(%1, %2)
  // Execute: %3.replaceInput(1, %4)
  // Result:  %3 = f(%1, %4)
  Value* replaceInput(size_t i, Value* newValue);

  // Replace all occurrences of 'from' in the inputs of this
  // node with 'to'. Corresponds to llvm's replaceUsesOfWith.
  //
  // Given:   %3 = f(%1, %2, %1)
  // Execute: %3.replaceInputWith(%1, %4)
  // Result:  %3 = f(%4, %2, %4)
  void replaceInputWith(Value* from, Value* to);

  Value* addOutput();

  Value* insertOutput(size_t i);

  void eraseOutput(size_t i);

  Block* addBlock();
  void eraseBlock(size_t i);

  // Each Node can have a list of subblocks. These are used to define structured
  // nested control flow operators such as If and Loop.
  // The meaning of a block is specific to the kind of node it is in, but
  // all blocks share these semantics:
  // * Nested lexical scoping: If a node 'Parent' has a subblock which contains
  //   a node 'Child', Child can use any value that was in scope for the Parent
  //   node in addition to any values defined before 'Child' in the subblock.
  // * The list of inputs to the block are in scope for the duration of the
  //   block
  // * the outputs of the Parent node are not in scope for the subblocks
  // Typically the inputs to a block that represents control flow act as
  // as the equivalents phi-nodes in standard SSA form,
  // defining a new Value to represent any term that has multiple
  // definitions depending on how control flowed. Outputs of the node containing
  // control flow serve a similiar purpose defining new values for variables
  // that would have different defintions depending on which way control flowed.

  at::ArrayRef<Block*> blocks() {
    return blocks_;
  }
  at::ArrayRef<const Block*> blocks() const {
    // Vectors are not convertible in const-ness of elements, but
    // raw pointers are.
    return {blocks_.data(), blocks_.size()};
  }

  // Is 'this' before 'n' in the topological order?
  bool isBefore(const Node* n) const;

  // Is 'this' after 'n' in the topological order?
  bool isAfter(const Node* n) const;

  // Insert unattached 'this' node before 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given:   %3 = f(%1, %2)
  //          %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertBefore(%4)
  // Result:  %3 = f(%1, %2)
  //          %5 = h(%1)
  //          %4 = g(%3)
  Node* insertBefore(Node* n);

  // Insert unattached 'this' node after 'n' in the topological order.
  // Returns this (for chaining).
  //
  // Given: %3 = f(%1, %2)
  //        %4 = g(%3)
  // and unattached: %5 = h(%1)
  // Execute: %5.insertAfter(%4)
  // Result:  %3 = f(%1, %2)
  //          %4 = g(%3)
  //          %5 = h(%1)
  Node* insertAfter(Node* n);

  // Move 'this' (already in the graph) after 'n' in the topological order.
  //
  // NOTE: Does not check that value dependencies are preserved, see
  //   AliasDb::moveAfterTopologicallyValid
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.moveAfter(%3)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  //
  void moveAfter(Node* n);

  // Move a node 'n' (already in the graph) before 'this' in the topological
  // order.
  //
  // NOTE: Does not check that value dependencies are preserved, see
  //   AliasDb::moveBeforeTopologicallyValid
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %3.moveBefore(%2)
  // Result: %3 = g(%1)
  //         %2 = f(%1)
  void moveBefore(Node* n);

  // Remove the input at 'i' from this node.
  //
  // WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
  // removeInput.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeInput(1)
  // Result: %3 = f(%1)
  void removeInput(size_t i);

  // Remove all inputs from a node.
  //
  // Given: %3 = f(%1, %2)
  // Execute: %3.removeAllInputs()
  // Result: %3 = f()
  void removeAllInputs();

  // iterators of the node list starting at this node
  // useful for resuming a search starting at this node
  inline graph_node_list_iterator iterator() {
    return {this, 0};
  }
  inline graph_node_list_iterator reverseIterator() {
    return iterator().reverse();
  }
  inline const_graph_node_list_iterator iterator() const {
    return {this, 0};
  }
  inline const_graph_node_list_iterator reverseIterator() const {
    return iterator().reverse();
  }

  // Remove 'this' from the instruction list and deallocate it.
  //
  // Invariant: no outputs of 'this' may have any uses.
  //
  // Given: %2 = f(%1)
  //        %3 = g(%1)
  // Execute: %2.destroy()
  // Result: %3 = g(%1)
  void destroy();

  // Dynamically cast this node to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid..
  //
  // Example usage: if(auto s = n.cast<Select>()) { ... }
  //
  // TODO: Make this const correct
  template <typename T>
  T* cast() {
    if (T::Kind == kind()) {
      return static_cast<T*>(this);
    }
    return nullptr;
  }
  template <typename T>
  T* expect() {
    AT_CHECK(
        T::Kind == kind(),
        "expected a ",
        T::Kind.toDisplayString(),
        " but found a ",
        kind().toDisplayString());
    return static_cast<T*>(this);
  }

  // XXX: this function is meant to be used with string literals only!
  bool matches(
      const char* signature_literal,
      at::ArrayRef<Symbol> const_inputs = {}) const;

  const FunctionSchema& schema() const {
    if (!schema_) {
      findSchema();
    }
    return *schema_;
  }
  const FunctionSchema* maybeSchema() const;

  void dump() const;

  std::ostream& print(
      std::ostream& out,
      size_t level,
      std::vector<const Node*>* groups) const;

  virtual ~Node() = default;

  // Methods for accessing attributes
  void copyAttributes(const Node& rhs) {
    values_.clear();
    for (const AVPtr& i : rhs.values_) {
      values_.push_back(i->clone());
    }
  }
  bool hasAttribute(Symbol name) const {
    AT_ASSERT(name.is_attr());
    return findAttr(name, false) != values_.end();
  }
  bool hasAttributeS(const std::string& name) const {
    return hasAttribute(Symbol::attr(name));
  }
  AttributeKind kindOf(Symbol name) const {
    AT_ASSERT(name.is_attr());
    return (*findAttr(name, true))->kind();
  }
  AttributeKind kindOfS(const std::string& name) const {
    return kindOf(Symbol::attr(name));
  }
  Node* removeAttribute(Symbol name) {
    AT_ASSERT(name.is_attr());
    values_.erase(findAttr(name, true));
    return this;
  }
  Node* removeAttributeS(const std::string& name) {
    return removeAttribute(Symbol::attr(name));
  }
  bool hasAttributes() const {
    return values_.size() > 0;
  }
  size_t numAttributes() const {
    return values_.size();
  }
  // The names are returned in order, since name actually is the index.
  std::vector<Symbol> attributeNames() const {
    std::vector<Symbol> names;
    for (const AVPtr& a : values_) {
      names.push_back(a->name);
    }
    return names;
  }
  std::vector<const char*> attributeNamesS() const {
    std::vector<const char*> names;
    for (const AVPtr& a : values_) {
      names.push_back(a->name.toUnqualString());
    }
    return names;
  }

#define CREATE_ACCESSOR(Kind, method)                           \
  Node* method##_(Symbol name, Kind##Attr::ConstructorType v) { \
    return setAttr<Kind##Attr>(                                 \
        name, std::forward<Kind##Attr::ConstructorType>(v));    \
  }                                                             \
  const Kind##Attr::ValueType& method(Symbol name) const {      \
    return getAttr<Kind##Attr>(name);                           \
  }

  CREATE_ACCESSOR(Float, f)
  CREATE_ACCESSOR(Floats, fs)
  CREATE_ACCESSOR(String, s)
  CREATE_ACCESSOR(Strings, ss)
  CREATE_ACCESSOR(Int, i)
  CREATE_ACCESSOR(Ints, is)
  CREATE_ACCESSOR(Graph, g)
  CREATE_ACCESSOR(Graphs, gs)

#undef CREATE_ACCESSOR

  // Our Graphs are not very const-correct, so we need to allow returning
  // non-const references too
  GraphAttr::ValueType& g(Symbol name) {
    return getAttr<GraphAttr>(name);
  }

  // does not use CREATE_ACCESSOR because we need additional asserts
  Node* t_(Symbol name, TensorAttr::ConstructorType v) {
    AT_ASSERT(!v.defined() || v.is_variable());
    return setAttr<TensorAttr>(
        name, std::forward<TensorAttr::ConstructorType>(v));
  }
  const TensorAttr::ValueType& t(Symbol name) const {
    return getAttr<TensorAttr>(name);
  }

  Node* ts_(Symbol name, TensorsAttr::ConstructorType v) {
    for (const at::Tensor& t : v) {
      AT_ASSERT(!t.defined() || t.is_variable());
    }
    return setAttr<TensorsAttr>(
        name, std::forward<TensorsAttr::ConstructorType>(v));
  }
  const TensorsAttr::ValueType& ts(Symbol name) const {
    return getAttr<TensorsAttr>(name);
  }

 private:
  void printAttrValue(std::ostream& out, const Symbol& name) const;
  void printAttributes(std::ostream& out, bool ignore_subgraph) const;

  template <typename T>
  Node* setAttr(Symbol name, typename T::ConstructorType v) {
    AT_ASSERT(name.is_attr());
    auto it = findAttr(name, false);
    auto nv = AVPtr(new T(name, std::forward<typename T::ConstructorType>(v)));
    if (it == values_.end()) {
      values_.push_back(std::move(nv));
    } else {
      *it = std::move(nv);
    }
    return this;
  }
  template <typename T>
  typename T::ValueType& getAttr(Symbol name) const {
    AT_ASSERT(name.is_attr());
    auto it = findAttr(name, true);
    auto* child = dynamic_cast<T*>(it->get());
    if (child == nullptr) {
      throw AttributeError(name, true);
    }
    return child->value();
  }
  using AVPtr = AttributeValue::Ptr;
  // NB: For determinism, we use a vector rather than a hash map.  This does
  // mean that lookups are O(n), so you shouldn't use Attributes to store
  // a big pile of messages.
  std::vector<AVPtr> values_;
  std::vector<AVPtr>::iterator findAttr(Symbol name, bool required) {
    AT_ASSERT(name.is_attr());
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
      return v->name == name;
    });
    if (required && it == values_.end()) {
      throw AttributeError(name, false);
    }
    AT_ASSERT(!required || it != values_.end());
    return it;
  }
  std::vector<AVPtr>::const_iterator findAttr(Symbol name, bool required)
      const {
    AT_ASSERT(name.is_attr());
    auto it = std::find_if(values_.begin(), values_.end(), [&](const AVPtr& v) {
      return v->name == name;
    });
    if (required && it == values_.end()) {
      throw AttributeError(name, false);
    }
    AT_ASSERT(!required || it != values_.end());
    return it;
  }

  enum class MoveSide { BEFORE, AFTER };
  bool isBeforeOrAfter(const Node* n, MoveSide moveSide) const;

  std::pair<Value*, const Argument&> findInput(Symbol name);
  void findSchema() const;
  // Lookup iterator in use list of _input i_ that corresponds to its use of
  // _this_
  use_list::iterator findUseForInput(size_t i);

  // remove the use of input i, this sets input i to nullptr, but
  // is only used internally to Node before setting it to a new value
  // or erasing the entry from the list.
  Value* dropInput(size_t i);

  bool inBlockList() const {
    if (next() == nullptr) {
      AT_ASSERT(prev() == nullptr);
    }
    return next() != nullptr;
  }

  void removeFromList();
  void lint() const;

  void assignTopoPosition();

 protected:
  // subclasses must override
  // this function is used by createClone to initialize a new version
  // of a node in another graph. It should allocate a new instance of the same
  // concrete type as 'this', but in graph 'g' which might be different
  // than graph_
  virtual Node* allocNewInstance(Graph* g) {
    return new Node(g, kind());
  }
  // create a copy of all properties of Node s into this.
  // subclasses should extend if they have additional information to copy.
  // 'this' will be allocated with s->allocNewInstance(g) so it should have
  // the same concrete type as 's'
  //
  virtual void cloneFrom(Node* s);
};

struct Block {
  friend struct Node;
  friend struct Graph;

  TH_DISALLOW_COPY_AND_ASSIGN(Block);
  TORCH_API Block(Graph* graph_, Node* node_);

  at::ArrayRef<Value*> inputs() {
    return input_->outputs();
  }
  at::ArrayRef<const Value*> inputs() const {
    const auto& inputs = input_->outputs();
    return {inputs.data(), inputs.size()};
  }
  at::ArrayRef<Value*> outputs() {
    return output_->inputs();
  }
  at::ArrayRef<const Value*> outputs() const {
    return static_cast<const Node*>(output_)->inputs();
  }
  graph_node_list nodes() {
    return {output_, kNextDirection};
  }
  const_graph_node_list nodes() const {
    return {output_, kNextDirection};
  }
  Node* return_node() {
    return output_;
  }
  const Node* return_node() const {
    return output_;
  }
  Node* param_node() {
    return input_;
  }
  const Node* param_node() const {
    return input_;
  }
  Graph* owningGraph() {
    return graph_;
  }
  const Graph* owningGraph() const {
    return graph_;
  }
  Node* owningNode() {
    return owning_node_;
  }
  const Node* owningNode() const {
    return owning_node_;
  }

  Value* addInput(std::string name = "") {
    Value* v = input_->addOutput();
    v->setUniqueName(std::move(name));
    return v;
  }
  Value* insertInput(size_t i, std::string name = "") {
    Value* v = input_->insertOutput(i);
    v->setUniqueName(std::move(name));
    return v;
  }
  void eraseInput(size_t i) {
    input_->eraseOutput(i);
  }
  size_t registerOutput(Value* v) {
    output_->addInput(v);
    return outputs().size() - 1;
  }
  size_t insertOutput(size_t i, Value* n) {
    output_->insertInput(i, n);
    return i;
  }
  void eraseOutput(size_t i) {
    output_->removeInput(i);
  }

  Node* appendNode(Node* n) {
    AT_ASSERT(n->graph_ == graph_ && !n->inBlockList());
    n->insertBefore(output_);
    return n;
  }
  Node* prependNode(Node* n) {
    AT_ASSERT(n->graph_ == graph_ && !n->inBlockList());
    n->insertAfter(output_);
    return n;
  }
  // clone all inputs, nodes, and outputs from src and append them
  // to the inputs, nodes, and outputs of this block
  // value_map is used whenever a node in src references a free variable
  // in src to look up its corresponding value
  TORCH_API void cloneFrom(Block* src, std::function<Value*(Value*)> value_map);
  TORCH_API void remapTypes(const std::function<TypePtr(TypePtr)>& type_map);
 private:
  void reIndexTopology();

  // should only be called in the constructor
  Node* initOutput(Node* p) {
    p->next() = p;
    p->prev() = p;
    return p;
  }

  // get rid of all nodes
  // destroys in reverse order so that uses internal to this block
  // do not have to be removed before you can destroy the block
  void destroy();

  Graph* const graph_;
  // holds outputs in a way that can be reflected
  // as a Use object
  // also used as the beginning/end of the circular node list to avoid
  // having corner cases where the list is empty.
  Node* const output_;
  Node* const input_;
  Node* const
      owning_node_; // either the node that has this block or nullptr for root
};

struct Graph {
  TH_DISALLOW_COPY_AND_ASSIGN(Graph);
  friend struct Node;
  friend struct Value;
  friend struct Block;

 private:
  // only used to keep track of allocated nodes
  // actual representation of Graph is done with
  // inputs, outputs, nodes

  std::unordered_set<const Node*> all_nodes;
  std::unordered_set<const Value*> all_values;
  std::unordered_set<const Block*> all_blocks;
  size_t next_unique_;

  std::unordered_map<std::string, Value*> unique_names_;

  ScopePtr current_scope_;

  Block* const block_;
  // when insertNode() is called, the node is inserted before this node
  // by default this is set to append to the top level block
  Node* insert_before_;

 public:
  Graph(ScopePtr scope_root)
      : next_unique_(0),
        current_scope_(std::move(scope_root)),
        block_(new Block(this, nullptr)),
        insert_before_(return_node()) {}

  Graph() : Graph(c10::make_intrusive<Scope>()) {}

  at::ArrayRef<Value*> inputs() {
    return block_->inputs();
  }
  at::ArrayRef<const Value*> inputs() const {
    const Block& block = *block_;
    return block.inputs();
  }
  at::ArrayRef<Value*> outputs() {
    return block_->outputs();
  }
  at::ArrayRef<const Value*> outputs() const {
    const Block& block = *block_;
    return block.outputs();
  }
  graph_node_list nodes() {
    return block_->nodes();
  }
  const_graph_node_list nodes() const {
    const Block& block = *block_;
    return block.nodes();
  }
  Node* param_node() {
    return block_->param_node();
  }
  const Node* param_node() const {
    return block_->param_node();
  }
  Node* return_node() {
    return block_->return_node();
  }
  const Node* return_node() const {
    return block_->return_node();
  }
  const std::unordered_map<std::string, Value*>& uniqueNames() const {
    return unique_names_;
  }

  void push_scope(const std::string& scope_name) {
    current_scope_ = current_scope_->push(Symbol::scope(scope_name));
  }
  void pop_scope() {
    current_scope_ = current_scope_->parent();
  }
  ScopePtr current_scope() {
    return current_scope_;
  }
  void set_current_scope(ScopePtr scope) {
    current_scope_ = std::move(scope);
  }

  Value* addInput(std::string name = "") {
    return block_->addInput(std::move(name));
  }
  Value* insertInput(size_t i, std::string name = "") {
    return block_->insertInput(i, std::move(name));
  }
  void eraseInput(size_t i) {
    block_->eraseInput(i);
  }
  size_t registerOutput(Value* n) {
    return block_->registerOutput(n);
  }
  void eraseOutput(size_t i) {
    block_->eraseOutput(i);
  }

  TORCH_API Node* create(NodeKind kind, size_t num_outputs = 1);
  TORCH_API Node* create(
      NodeKind kind,
      ArrayRef<Value*> inputs,
      size_t num_outputs = 1);

  TORCH_API Node* createNone(
      TypePtr typ); // value of None with type Optional[typ]
  TORCH_API Node* createAutogradZero();
  TORCH_API Node* createFusionGroup();
  TORCH_API Node* createDifferentiableSubgraph();
  TORCH_API Node* createTuple(
      at::ArrayRef<Value*> values,
      c10::OptNameList field_names = c10::nullopt);
  TORCH_API Node* createTupleUnpack(Value* v);
  TORCH_API Node* createTupleIndex(Value* tup, int64_t index);
  TORCH_API Node* createTupleSlice(Value* tup, int64_t beg, int64_t end);
  TORCH_API Node* createList(
      const TypePtr& elem_type,
      at::ArrayRef<Value*> values);
  TORCH_API Node* createListUnpack(Value* v, size_t size);
  TORCH_API Node* createDict(
      const TypePtr& key_type,
      const TypePtr& value_type,
      at::ArrayRef<Value*> keys,
      at::ArrayRef<Value*> values);
  TORCH_API Node* createDictIndex(Value* dict, Value* index);
  TORCH_API Node* createNumToTensor(Value* value);
  TORCH_API Node* createImplicitTensorToNum(const TypePtr& type, Value* value);
  TORCH_API Node* createObject(const ClassTypePtr& type);
  TORCH_API Node* createSetAttr(
      Value* obj,
      const std::string& field,
      Value* newValue);
  TORCH_API Node* createGetAttr(Value* obj, const std::string& field);
  TORCH_API Value* insertGetAttr(Value* obj, const std::string& field) {
    return insertNode(createGetAttr(obj, field))->output();
  }

  // Note: defined in python_ir.cpp and can be used only in python extension
  Node* createPythonOp(
      THPObjectPtr&& pyobj,
      const std::string& cconv,
      pyobj_list&& scalar_args);
  // clone n, making a new node in _this_ graph.
  // use node_map to translate inputs of n to inputs of the cloned node
  // if copy_blocks is false, it will not recursively clone the nested blocks
  // this node contains.
  TORCH_API Node* createClone(
      Node* n,
      const std::function<Value*(Value*)>& value_map,
      bool copy_blocks = true);

  // Insert constant IValue into the graph. If the type cannot be fully deduced
  // from the ivalue, as with a None that is set to t?, use result_type
  TORCH_API Value* insertConstant(
      IValue val,
      const TypePtr& result_type = nullptr,
      c10::optional<SourceRange> loc = c10::nullopt,
      c10::optional<ScopePtr> scope = c10::nullopt);

  // Schema-driven insert:
  // This inserts a node into the graph with inputs determined from args and
  // kwargs using Python argument matching rules, and checks that the op matches
  // a known schema.
  //
  // If this node successfully completes, it guarentees the node
  // is a correctly-formed invocation of opname
  TORCH_API Value* insert(
      Symbol opname,
      at::ArrayRef<NamedValue> args,
      at::ArrayRef<NamedValue> kwargs = {},
      const c10::optional<SourceRange>& range = {});

  Node* appendNode(Node* n) {
    return block_->appendNode(n);
  }

  Node* prependNode(Node* n) {
    return block_->prependNode(n);
  }

  // insert before insert_before_ node
  // initialized to insert at the end of the top level block
  // can be changed with setInsertPoint()
  Node* insertNode(Node* n) {
    AT_ASSERT(
        insert_before_->inBlockList() &&
        "insert point node is no longer in a block list");
    return n->insertBefore(insert_before_);
  }
  // set where nodes are inserted to append to the end of this block
  void setInsertPoint(Block* b) {
    AT_ASSERT(b->owningGraph() == this);
    insert_before_ = b->return_node();
  }
  // set where nodes are inserted to insert _before_ this node
  // for implementation simplicity we only support inserting before a node for
  // now
  void setInsertPoint(Node* n) {
    AT_ASSERT(n->owningGraph() == this && n->inBlockList());
    insert_before_ = n;
  }
  Node* insertPoint() {
    return insert_before_;
  }

  // the top level block
  Block* block() {
    return block_;
  }
  const Block* block() const {
    return block_;
  }

  // Checks well-formedness and invariants of graph
  TORCH_API void lint() const;
  // for use in debugger
  TORCH_API void dump() const;

  TORCH_API ~Graph();

  TORCH_API std::string toString() const;

  friend TORCH_API std::ostream& operator<<(std::ostream& out, const Graph& g);

  TORCH_API std::shared_ptr<Graph> copy();
  TORCH_API void remapTypes(const std::function<TypePtr(TypePtr)>& type_map);

 private:
  TORCH_API void freeNode(Node* n);
  TORCH_API void freeValue(Value* v);
  TORCH_API void freeBlock(Block* b);
};

/** \brief An utility class for setting temporary insertion points.
 *
 * When an object of this class is created, it stores the current insertion
 * point, sets the new one, and restores the original insertion point  when the
 * object is destroyed.
 */
struct WithInsertPoint {
  WithInsertPoint(Node* n) : prev_(n->owningGraph()->insertPoint()) {
    n->owningGraph()->setInsertPoint(n);
  }
  WithInsertPoint(Block* b) : WithInsertPoint(b->return_node()) {}

  ~WithInsertPoint() {
    prev_->owningGraph()->setInsertPoint(prev_);
  }

 private:
  Node* prev_;
};

/** \brief An utility class for setting temporary scopes.
 *
 * When an object of this class is created, it stores the current scope, sets
 * the new one, and restores the original scope when the object is destroyed.
 */
struct WithCurrentScope {
  WithCurrentScope(Graph& g, ScopePtr scope)
      : graph_(&g), prev_scope_(g.current_scope()) {
    g.set_current_scope(std::move(scope));
  }
  ~WithCurrentScope() {
    graph_->set_current_scope(prev_scope_);
  }

 private:
  Graph* graph_;
  ScopePtr prev_scope_;
};

inline Value::Value(Node* node_, size_t offset_)
    : node_(node_),
      offset_(offset_),
      unique_(node_->graph_->next_unique_++),
      type_(TensorType::get()) {
  node_->graph_->all_values.emplace(this);
}

inline Value* Value::setType(TypePtr type) {
  AT_ASSERT(type);
  type_ = std::move(type);
  for (Use& use : uses_) {
    use.user->schema_ = nullptr;
  }
  return this;
}

inline Graph* Value::owningGraph() {
  return node()->owningGraph();
}

inline const Graph* Value::owningGraph() const {
  return node()->owningGraph();
}

/************* All nodes not required to be defined before Graph **************/
struct ProfileOp : public Node {
  static constexpr Symbol Kind = ::c10::prim::profile;
  ProfileOp(Graph* graph, std::function<void(std::vector<IValue>&)> callback)
      : Node(graph, ::c10::prim::profile), callback_(callback) {}

  void cloneFrom(Node* other_) override;
  Node* allocNewInstance(Graph* g) override;

  const std::function<void(std::vector<IValue>&)>& getCallback() const {
    return callback_;
  }

 private:
  std::function<void(std::vector<IValue>&)> callback_;
};

// execute a Python function, used for Ops we can't optimize but that we want to
// optimize around
//
// Note: actual implementation (ConcretePythonOp) is defined in python_ir.cpp
// which is not included in libtorch.so. We still include some bits and pieces
// of PythonOp here to enable writing simple passes generically. In general,
// python-aware bits need to be moved to the descendant classes.
struct TORCH_API PythonOp : public Node {
  using Node::Node;

  // should this Python function be skipped over when exported (i.e. for
  // debugging functions that only run in Python)
  bool ignore_on_export = false;

  virtual std::string name() const = 0;
  virtual void writeScalars(std::ostream& out) const = 0;
  void cloneFrom(Node* other_) override = 0;
  Node* allocNewInstance(Graph* g) override = 0;
  // recover the autograd.Function instance, if this PythonOp's function
  // was originally SomeFunction.apply
  // used in ONNX for discovering symbolics
  virtual c10::optional<THPObjectPtr> autogradFunction() const = 0;

  virtual void lint_python() const = 0;
};

TORCH_API void LintGraph(std::shared_ptr<Graph>& graph);

TORCH_API at::ArrayRef<Value*> createTupleUnpack(Value* v);
// unpack_outputs - if true, and the callee returns a single tuple value, then
// insert a tuple unpack node
//                  and return the resulting values
TORCH_API std::vector<Value*> inlineCallTo(
    Graph& g,
    Graph& callee,
    ArrayRef<Value*> inputs,
    bool unpack_outputs = false);

} // namespace jit
} // namespace torch
