#include <torch/csrc/jit/passes/alias_analysis.h>

#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

bool AliasDb::shouldAnnotate(const TypePtr& type) {
  return type->isSubtypeOf(TensorType::get()) ||
      type->kind() == TypeKind::ListType ||
      type->kind() == TypeKind::TupleType ||
      type->kind() == TypeKind::DictType || type->kind() == TypeKind::VarType ||
      type->kind() == TypeKind::FutureType ||
      type->kind() == TypeKind::ClassType ||
      (type->kind() == TypeKind::OptionalType &&
       shouldAnnotate(type->cast<OptionalType>()->getElementType()));
}

// We only need to annotate values that either are mutable or could contain
// mutable types.
bool AliasDb::shouldAnnotate(const Value* v) {
  return shouldAnnotate(v->type());
}

bool AliasDb::isContainerType(const TypePtr& type) {
  if (type->kind() == TypeKind::FutureType) {
    return isContainerType(type->cast<FutureType>()->getElementType());
  } else if (type->kind() == TypeKind::OptionalType) {
    return isContainerType(type->cast<OptionalType>()->getElementType());
  } else {
    return type->containedTypes().size() > 0;
  }
}

AliasDb::~AliasDb() = default;

AliasDb::AliasDb(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
  memoryDAG_ = torch::make_unique<MemoryDAG>();
  analyze(graph_);
}

// Does `n` use or write to any wildcard aliases?
bool AliasDb::hasWildcard(const Node* n) const {
  for (const auto input : n->inputs()) {
    if (isWildcard(input)) {
      return true;
    }
  }
  for (const auto output : n->outputs()) {
    if (isWildcard(output)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::isWildcard(const Value* v) const {
  return wildcards_.count(v);
}

bool AliasDb::writesTo(Node* n, const Value* v) const {
  if (!shouldAnnotate(v)) {
    // This is a primitive type
    return false;
  }
  if (isWildcard(v)) {
    return wildcardWriters_.count(n);
  }

  if (!elementMap_.count(v) || !writeIndex_.count(n)) {
    return false;
  }

  // Can short-circuit if we know this node writes directly to `v`
  if (writeIndex_.at(n).count(v)) {
    return true;
  }

  // Otherwise, check if `v` may alias any of written-to values in `n`
  const auto vSet = ValueSet{v};
  return mayAlias(vSet, writeIndex_.at(n));
}

bool AliasDb::hasWriters(const Node* n) const {
  for (const auto input : n->inputs()) {
    if (hasWriters(input)) {
      return true;
    }
  }
  for (const auto output : n->outputs()) {
    if (hasWriters(output)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::hasWriters(const Value* v) const {
  if (isWildcard(v)) {
    // If `n` has a wildcard, any write in the graph may write to it.
    // So the only way we know there are no writers is if there are no writes
    // at all.
    return numWrites_ != 0;
  }

  if (!elementMap_.count(v)) {
    return false;
  }

  if (wildcardWriters_.size() > 0) {
    // A write to the wildcard may be a write to any value.
    return true;
  }

  if (isWriteCacheStale_) {
    rebuildWriteCache();
  }

  for (const auto loc : elementMap_.at(v)->getMemoryLocations()) {
    if (writeCache_.count(loc)) {
      return true;
    }
  }

  return false;
}

bool AliasDb::hasWrites(Node* n) const {
  for (const auto input : n->inputs()) {
    if (writesTo(n, input)) {
      return true;
    }
  }
  for (const auto output : n->outputs()) {
    if (writesTo(n, output)) {
      return true;
    }
  }
  return false;
}

bool AliasDb::writesToInputAlias(Node* n) const {
  std::vector<const Value*> writes;
  for (const auto input : n->inputs()) {
    if (writesTo(n, input)) {
      writes.push_back(input);
    }
  }
  for (const auto output : n->outputs()) {
    if (writesTo(n, output)) {
      writes.push_back(output);
    }
  }

  // For all writes, check if the written value may alias a graph input
  return std::any_of(writes.cbegin(), writes.cend(), [&](const Value* v) {
    return std::any_of(
        graph_->inputs().cbegin(),
        graph_->inputs().cend(),
        [&](const Value* graphInput) {
          return shouldAnnotate(graphInput) && mayAlias(graphInput, v);
        });
  });
}

void AliasDb::getWritesImpl(Block* b, ValueSet& ret, bool recurseBlocks) const {
  for (auto node : b->nodes()) {
    getWritesImpl(node, ret, recurseBlocks);
  }
}

void AliasDb::getWritesImpl(Node* n, ValueSet& ret, bool recurseBlocks) const {
  for (const auto input : n->inputs()) {
    if (writesTo(n, input)) {
      ret.insert(input);
    }
  }
  for (const auto output : n->outputs()) {
    if (writesTo(n, output)) {
      ret.insert(output);
    }
  }

  if (recurseBlocks) {
    for (auto block : n->blocks()) {
      getWritesImpl(block, ret, recurseBlocks);
    }
  }
}

// Get all writes by all nodes in a block, recursively exploring sub-blocks
ValueSet AliasDb::getWrites(Block* b) const {
  ValueSet writes;
  getWritesImpl(b, writes, /*recurseBlocks=*/true);
  return writes;
}

// Does `n` write to an alias of one of the values in `vs`?
bool AliasDb::writesToAlias(Node* n, const ValueSet& vs, bool recurseBlocks)
    const {
  const auto writtenTo = getWrites(n, recurseBlocks);
  return mayAlias(vs, writtenTo);
}

std::unordered_set<const Value*> AliasDb::getWrites(Node* n, bool recurseBlocks)
    const {
  ValueSet writes;
  getWritesImpl(n, writes, recurseBlocks);
  return writes;
}

void AliasDb::getReadsImpl(Node* n, ValueSet& ret, bool recurseBlocks) const {
  for (const auto input : n->inputs()) {
    ret.insert(input);
  }
  for (const auto output : n->outputs()) {
    ret.insert(output);
  }

  if (recurseBlocks) {
    for (auto block : n->blocks()) {
      for (auto node : block->nodes()) {
        getReadsImpl(node, ret, recurseBlocks);
      }
    }
  }
}

ValueSet AliasDb::getReads(Node* n, bool recurseBlocks) const {
  ValueSet reads;
  getReadsImpl(n, reads, recurseBlocks);
  return reads;
}

void AliasDb::dump() const {
  std::cout << "\n===1. GRAPH===\n";
  graph_->dump();

  std::cout << "\n===2. ALIAS DB===\n";
  for (const auto& ptrPair : elementMap_) {
    const auto element = ptrPair.second;
    if (element->pointsTo.size() > 0) {
      std::cout << element->value->uniqueName() << " points to: ";
      for (const auto pointedTo : element->pointsTo) {
        std::cout << pointedTo->value->uniqueName() << ", ";
      }
      std::cout << "\n";
    }
    if (element->contained_elements.size() > 0) {
      std::cout << element->value->uniqueName() << " contains: ";
      for (const auto contained : element->contained_elements) {
        std::cout << contained->value->uniqueName() << ", ";
      }
      std::cout << "\n";
    }
  }

  std::cout << "\n===3. WILDCARDS===\n";
  for (const auto wildcard : wildcards_) {
    std::cout << wildcard->uniqueName() << ", ";
  }
  std::cout << "\n";

  std::cout << "\n===4. Writes===\n";
  for (const auto& pr : writeIndex_) {
    const auto node = pr.first;
    const auto& values = pr.second;
    std::cout << *node;
    std::cout << "  ";
    for (const auto value : values) {
      std::cout << value->uniqueName() << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

// TODO: need to create a dummy "graph input alias" value in MemoryDAG for all
// inputs of the same type to point to. Currently they all point to the first
// element, which is technically wrong.
void AliasDb::makeAllAlias(const std::vector<Value*>& values) {
  if (values.size() > 0) {
    giveFreshAlias(values[0]);
  }
  for (const auto value : values) {
    makePointerTo(value, values[0]);
  }
}

void AliasDb::analyze(const std::shared_ptr<Graph>& graph) {
  // Assign aliases to the graph's inputs, assuming that all inputs of a given
  // type may alias to each other.

  // 1. Partition inputs by their type
  std::map<TypeKind, std::vector<Value*>> listTypes;
  std::unordered_map<TupleTypePtr, std::vector<Value*>> tupleTypes;
  std::unordered_map<DictTypePtr, std::vector<Value*>> dictTypes;
  std::unordered_map<ClassTypePtr, std::vector<Value*>> classTypes;
  std::vector<Value*> tensors;

  for (auto input : graph->inputs()) {
    auto inputType = input->type();
    // unwrap optional types
    if (inputType->kind() == TypeKind::OptionalType) {
      inputType = inputType->cast<OptionalType>()->getElementType();
    }

    if (inputType->isSubtypeOf(TensorType::get())) {
      tensors.push_back(input);
    } else if (inputType->kind() == TypeKind::ListType) {
      auto containedType = inputType->containedTypes().at(0);
      // All tensor subtypes may alias to each other, so we should consider all
      // lists of them to alias to each other.
      if (containedType->isSubtypeOf(TensorType::get())) {
        containedType = TensorType::get();
      }
      listTypes[containedType->kind()].push_back(input);
    } else if (inputType->kind() == TypeKind::TupleType) {
      auto tupleType = inputType->cast<TupleType>();
      tupleTypes[tupleType].push_back(input);
    } else if (inputType->kind() == TypeKind::DictType) {
      auto dictType = inputType->cast<DictType>();
      dictTypes[dictType].push_back(input);
    } else if (inputType->kind() == TypeKind::ClassType) {
      auto classType = inputType->cast<ClassType>();
      classTypes[classType].push_back(input);
    } else {
      AT_ASSERT(!shouldAnnotate(input));
    }
  }

  // 2. Make all partitions alias each other
  for (const auto& pr : listTypes) {
    makeAllAlias(pr.second);
  }
  for (const auto& pr : tupleTypes) {
    makeAllAlias(pr.second);
  }
  for (const auto& pr : dictTypes) {
    makeAllAlias(pr.second);
  }
  for (const auto& pr : classTypes) {
    makeAllAlias(pr.second);
  }
  makeAllAlias(tensors);

  analyze(graph->block());
}

void AliasDb::analyze(Block* block) {
  for (auto node : block->nodes()) {
    analyze(node);
  }
}

void AliasDb::analyze(Node* node) {
  analyzeImpl(node);

  // After analyzing, update the wildcard index
  if (hasWildcard(node)) {
    wildcardNodes_.insert(node);
  }
}

// The basic strategy is:
//   1. Retrieve alias information for every input.
//   2. Use the node's schema's alias annotations to propgagate alias/write
//      information to the outputs. For unschematized nodes, a special analyzer
//      will have to be handwritten.
void AliasDb::analyzeImpl(Node* node) {
  // These nodes are not schematized, so we need to handle them specially
  switch (node->kind()) {
    case prim::If:
      return analyzeIf(node);
    case prim::Loop:
      return analyzeLoop(node);
    case prim::FusionGroup:
    case prim::DifferentiableGraph:
      return analyzeSubgraph(node);
    case prim::fork:
      return analyzeFork(node);
    case aten::wait:
      return analyzeWait(node);
    case prim::TupleConstruct:
      return analyzeTupleConstruct(node);
    case prim::GradOf:
      return analyzeGradOf(node);
    case prim::Constant:
    case prim::DictConstruct:
    case prim::ListConstruct:
    case prim::AutogradZero:
    case prim::AutogradAdd:
    case prim::FusedConcat:
    case prim::MMTreeReduce:
    case prim::MMBatchSide:
    case prim::BroadcastSizes:
    case prim::ChunkSizes:
    case prim::Function:
    case prim::CreateObject:
      return analyzeCreator(node);
    case prim::TupleUnpack:
    case prim::TupleIndex:
    case prim::DictIndex:
    case prim::TupleSlice:
    case prim::ListUnpack:
    case prim::PythonOp:
    case prim::GetAttr:
      return analyzeExtractor(node);
    case prim::ConstantChunk:
      return analyzeChunk(node);
    case prim::BroadcastingChunk:
      return analyzeBroadcastingChunk(node);
    case prim::SetAttr:
      return analyzeSetAttr(node);
    case aten::add:
    case aten::sub:
    case aten::mul:
    case aten::div: {
      // This is necessary because we sometimes get unschematized combinations
      // of Tensor/primitive.
      auto maybeSchema = node->maybeSchema();
      if (!maybeSchema) {
        return analyzeCreator(node);
      }
      // If the node has a schema, fall through and analyze it normally
      break;
    }
    case prim::Print:
      // These ops do nothing
      return;
    default:
      AT_ASSERT(!aliasAnalysisHasSpecialCaseFor(node->kind()));
  }

  const auto& schema = node->schema();
  if (schema.is_vararg() || schema.is_varret()) {
    const auto hasMutableOutputs = std::any_of(
        node->outputs().cbegin(),
        node->outputs().cend(),
        [](const Value* output) { return shouldAnnotate(output); });

    // We don't have alias info for this node. Either schematize it, or
    // add it an analyze* method for it.
    if (hasMutableOutputs) {
      throw script::ErrorReport(node->getSourceLocation())
          << "Alias information not found for node. File a bug report.\n"
          << "Node: " << *node << "\n";
    }
  }

  // see [custom operator aliasing]
  if (!node->kind().is_aten() && !node->kind().is_prim()) {
    return analyzeCustomOp(node);
  }

  // Bind formal alias annotation to actual alias sets
  std::unordered_map<Symbol, Value*> formalToActual;
  for (size_t i = 0; i < schema.arguments().size(); i++) {
    const auto& formal = schema.arguments()[i].alias_info();
    const auto& actualValue = node->inputs().at(i);
    // Skip if there's no alias annotation
    if (!formal) {
      continue;
    }

    // If this type cannot alias, continue. Can occur with a VarType schema
    if (!shouldAnnotate(actualValue)) {
      continue;
    }

    // We don't support composite types for alias analysis yet.
    AT_ASSERT(formal->containedTypes().size() == 0);
    // TODO neither unions nor wildcards make sense on an input. We should
    // disallow them in function schema
    AT_ASSERT(!formal->isWildcard())
    const auto& formalAlias = formal->beforeSet();

    // skip if we've already bound this alias
    if (formalToActual.count(formalAlias) != 0) {
      continue;
    }

    // Bind the formal to the actual
    formalToActual[formalAlias] = actualValue;

    // Record writes
    if (formal->isWrite()) {
      registerWrite(actualValue, node);
    }
  }

  // Use the formal-actual mapping to give aliases to the outputs
  for (size_t i = 0; i < schema.returns().size(); i++) {
    const auto actual = node->outputs().at(i);
    const auto& formal = schema.returns()[i].alias_info();
    if (!formal) {
      // This is a fresh tensor
      giveFreshAlias(actual);
      continue;
    }

    // If this type cannot alias, continue. Can occur with a VarType schema
    if (!shouldAnnotate(actual)) {
      continue;
    }

    // We don't support composite types for alias analysis yet.
    AT_ASSERT(formal->containedTypes().size() == 0);

    if (formal->isWildcard()) {
      setWildcard(actual);
      continue;
    }

    for (const auto& formalAlias : formal->beforeSets()) {
      // If we encounter an alias annotation that wasn't in the inputs:
      if (!formalToActual.count(formalAlias)) {
        // If this alias is not seen elsewhere and is the only annotation on
        // the output, it's equivalent to being fresh:
        //   e.g. foo(Tensor(a) self) -> Tensor(b)
        if (formal->beforeSets().size() == 1) {
          giveFreshAlias(actual);
        }
        // Or it is the form of a|fresh, which we can ignore, taking the
        // conservative assumption that the output must alias `a`, e.g
        //   aten::cuda(Tensor(a) self) -> Tensor(a|fresh)

        // Don't assign an alias set in that case.
        continue;
      }

      auto toAlias = formalToActual.at(formalAlias);
      makePointerTo(actual, toAlias);
    }

    // Record writes
    if (formal->isWrite()) {
      registerWrite(actual, node);
    }
  }
}
// Register the fact that `n` writes to `v`.
void AliasDb::registerWrite(const Value* v, Node* n) {
  if (!shouldAnnotate(v)) {
    // don't need to register a write if the value isn't mutable
    return;
  }

  numWrites_++;

  if (isWildcard(v)) {
    wildcardWriters_.insert(n);
    return;
  }

  AT_ASSERT(elementMap_.count(v));
  writeIndex_[n].insert(v);
}

void AliasDb::analyzeIf(Node* node) {
  // For if statements, the alias set of an output is the union of the
  // alias sets generated by the if and else block
  const auto trueBlock = node->blocks().at(0);
  const auto falseBlock = node->blocks().at(1);
  analyze(trueBlock);
  analyze(falseBlock);

  for (size_t i = 0; i < node->outputs().size(); i++) {
    const auto nodeOutput = node->outputs()[i];

    const auto trueOutput = trueBlock->outputs().at(i);
    const auto falseOutput = falseBlock->outputs().at(i);

    makePointerTo(nodeOutput, trueOutput);
    makePointerTo(nodeOutput, falseOutput);
  }
}

void AliasDb::analyzeLoop(Node* node) {
  const auto bodyBlock = node->blocks().at(0);
  const auto loopCarriedInputs = node->inputs().slice(2); // skip max, cond
  const auto blockInputs = bodyBlock->inputs().slice(1); // skip trip
  const auto blockOutputs = bodyBlock->outputs().slice(1); // skip trip
  AT_ASSERT(loopCarriedInputs.size() == blockInputs.size());
  AT_ASSERT(blockOutputs.size() == node->outputs().size());

  // Run alias analysis on the loop body, iterating until the block output
  // alias info converges.
  // Copy node input aliases to block input
  mapAliases(blockInputs, loopCarriedInputs);

  // Populate block output alias info by analyzing the body
  analyze(bodyBlock);

  // Copy the alias info from the block output to the node output
  mapAliases(node->outputs(), blockOutputs);
}

void AliasDb::analyzeGradOf(Node* node) {
  const auto grad_of_block = node->blocks().at(0);
  analyze(grad_of_block);
  mapAliases(node->outputs(), grad_of_block->outputs());
}

void AliasDb::analyzeSubgraph(Node* node) {
  const auto subgraph = node->g(attr::Subgraph).get();

  subgraphToOwner_.insert({subgraph, node});

  const auto subgraphBlock = subgraph->block();
  mapAliases(subgraphBlock->inputs(), node->inputs());

  analyze(subgraphBlock);

  // TODO(suo): the subgraph outputs and node outputs are NOT NECESSARILY the
  // same length. Autodifferentiation maybe capture additional outputs in the
  // subgraph block.
  AT_ASSERT(subgraphBlock->outputs().size() >= node->outputs().size());
  for (size_t i = 0; i < node->outputs().size(); i++) {
    makePointerTo(node->outputs()[i], subgraphBlock->outputs()[i]);
  }
}

// For nodes that generate a fresh value from nothing
void AliasDb::analyzeCreator(Node* node) {
  for (Value* output : node->outputs()) {
    giveFreshAlias(output);
  }
}

// For nodes that extract values from a composite type. Right now, this just
// gives up and creates wildcards for everything.
void AliasDb::analyzeExtractor(Node* node) {
  for (const auto output : node->outputs()) {
    if (shouldAnnotate(output)) {
      setWildcard(output);
    }
  }
}

// For torch.chunk(), all returned tensors may alias the input tensor
void AliasDb::analyzeChunk(Node* node) {
  for (auto output : node->outputs()) {
    makePointerTo(output, node->input());
  }
}

// Propagate aliasing and write information from the subgraph outputs to the
// outputs of the corresponding aten::wait() calls, since that's where the
// values will eventually emerge.
void AliasDb::analyzeFork(Node* node) {
  const auto subgraph = node->g(attr::Subgraph).get();
  subgraphToOwner_.insert({subgraph, node});

  const auto subgraphBlock = subgraph->block();
  mapAliases(subgraphBlock->inputs(), node->inputs());
  analyze(subgraphBlock);

  // Give the future that the fork emits a fresh value
  for (const auto output : node->outputs()) {
    giveFreshAlias(output);
  }
}

void AliasDb::analyzeWait(Node* node) {
  const auto fut = node->input();
  AT_ASSERT(fut->type()->kind() == TypeKind::FutureType);

  if (isWildcard(fut)) {
    for (const auto output : node->outputs()) {
      setWildcard(output);
    }
    return;
  }

  const auto originFuts = getMemoryLocations(fut);
  for (const auto originFut : originFuts) {
    const auto subgraphNode = originFut->node();

    const auto subgraph = subgraphNode->g(attr::Subgraph).get();
    const auto subgraphWrites = getWrites(subgraph->block());

    // Retrieve aliasing info from the subgraph
    mapAliases(node->outputs(), subgraph->outputs());

    // Propagate write information to the `wait` node.
    //
    // We need to do this for all writes in the entire subgraph, so that we
    // disallow reorders past a call to "aten::wait".
    //
    // Consider the following Fork where the subgraph writes to %a:
    //
    //   %c : Future[Tensor] = prim::Fork(%a, %b) <-- writes to %a
    //   ...
    //   aten::wait(%c)
    //   aten::use(%a)   <-- we can't move this node before the `wait` safely!
    //
    // Say we define the "live interval" of a fork the interval between the
    // `fork` and its first corresponding `wait` (inclusive).
    //
    // Any writes in the subgraph can happen at any point in the live interval,
    // so it's not safe to re-order any reads to those memory locations from
    // outside the live interval to inside.
    //
    // In reality, any reads *inside* the live interval are undefined behavior,
    // since the writes may or may not have been executed yet. But we'll let
    // users do that and shoot themselves in the foot for now.
    for (const auto write : subgraphWrites) {
      registerWrite(write, node);
    }
  }
}

void AliasDb::analyzeTupleConstruct(Node* node) {
  // Because we currently mark all Tuples as needing annotation
  // (even those containing just prmitive types), an element needs to be created
  // for TupleConstruct. When that changes we can create an element
  // only if it contains elements which need annotation
  getOrCreateElement(node->output());
  for (const auto& input : node->inputs()) {
    if (shouldAnnotate(input)) {
      addToContainedElements(input, node->output());
    }
  }
}

// SetAttr: writes to the `self` field
void AliasDb::analyzeSetAttr(Node* node) {
  const auto self = node->inputs().at(0);
  AT_ASSERT(self->type()->kind() == TypeKind::ClassType);
  registerWrite(self, node);
}

// Custom ops may write to any input and produce wildcards
void AliasDb::analyzeCustomOp(Node* node) {
  for (const auto input : node->inputs()) {
    registerWrite(input, node);
  }

  // TODO(suo): we can make the more refined assumption that outputs may only
  // alias any input.
  for (const auto output : node->outputs()) {
    setWildcard(output);
  }
}

// BroadcastingChunk: all inputs are broadcasted, and then individually chunked.
// This is an intermediate node used only in the graph fuser.
void AliasDb::analyzeBroadcastingChunk(Node* node) {
  auto inputs = node->inputs();
  auto outputs = node->outputs();
  auto nchunks = node->i(attr::chunks);
  for (size_t index = 0; index < inputs.size(); ++index) {
    // Each inputs[i] is aliased by exactly `nchunks` distinct output tensors:
    // inputs[i] produces chunks outputs[i * nchunks + k] for k in [0..nchunks)
    auto output_begin = outputs.begin() + index * nchunks;
    for (auto it = output_begin; it != output_begin + nchunks; ++it) {
      makePointerTo(*it, inputs.at(index));
    }
  }
}

// Register the fact that `value` is a pointer to `to`
void AliasDb::makePointerTo(const Value* from, const Value* to) {
  if (!shouldAnnotate(from)) {
    AT_ASSERT(!shouldAnnotate(to));
    return;
  }

  if (from == to) {
    return;
  }

  // Special case: if `from` is an optional, `to` could be a None. Don't
  // create a pointer in that case
  if (from->type()->kind() == TypeKind::OptionalType &&
      to->type()->kind() == TypeKind::NoneType) {
    return;
  }

  // At this point, we should be dealing with two mutable types.
  AT_ASSERT(shouldAnnotate(from) && shouldAnnotate(to));

  // If either value is a wildcard, don't insert anything into the graph;
  // wildcards are tracked separately since they have different aliasing rules.
  if (isWildcard(to) || isWildcard(from)) {
    setWildcard(from);
    return;
  }

  auto fromEl = getOrCreateElement(from);
  auto toEl = getOrCreateElement(to);

  memoryDAG_->makePointerTo(fromEl, toEl);
}

void AliasDb::addToContainedElements(
    const Value* elem,
    const Value* container) {
  if (!shouldAnnotate(elem)) {
    return;
  }

  // wildcards tracked separately
  if (isWildcard(elem)) {
    return;
  }

  AT_ASSERT(isContainerType(container->type()));

  auto elemEl = getOrCreateElement(elem);
  auto contEl = getOrCreateElement(container);

  memoryDAG_->addToContainedElements(elemEl, contEl);
}

bool AliasDb::mayAlias(const Value* a, const Value* b) const {
  if (!shouldAnnotate(a) || !shouldAnnotate(b)) {
    return false;
  }

  if (isWildcard(a) || isWildcard(b)) {
    return true;
  }

  return memoryDAG_->mayAlias(elementMap_.at(a), elementMap_.at(b));
}

bool AliasDb::cannotCheckAliasContainment(const Value* elem) const {
  if (isWildcard(elem)) {
    return true;
  }

  if (isContainerType(elem->type())) {
    if (elem->node()->kind() != prim::TupleConstruct) {
      return true;
    }
    auto inps = elem->node()->inputs();
    return std::any_of(inps.begin(), inps.end(), [&](const Value* v) {
      return cannotCheckAliasContainment(v);
    });
  }

  return false;
}

bool AliasDb::mayContainAlias(Value* a, Value* b) const {
  const std::vector<Value*> a_vec = {a};
  const std::vector<Value*> b_vec = {b};

  return mayContainAlias(a_vec, b_vec);
}

bool AliasDb::mayContainAlias(
    const at::ArrayRef<Value*>& a,
    const at::ArrayRef<Value*>& b) const {
  std::vector<Element*> a_elements;
  for (const auto& val : a) {
    if (cannotCheckAliasContainment(val)) {
      return true;
    }
    if (shouldAnnotate(val)) {
      a_elements.push_back(elementMap_.at(val));
    }
  }

  if (a_elements.size() == 0) {
    return false;
  }

  std::vector<Element*> b_elements;
  for (const auto& val : b) {
    if (cannotCheckAliasContainment(val)) {
      return true;
    }
    if (shouldAnnotate(val)) {
      b_elements.push_back(elementMap_.at(val));
    }
  }
  return memoryDAG_->mayContainAlias(a_elements, b_elements);
}

// Make each value in the `from` list point to its partner in the `to` list
void AliasDb::mapAliases(at::ArrayRef<Value*> from, at::ArrayRef<Value*> to) {
  AT_ASSERT(to.size() == from.size());
  for (size_t i = 0; i < to.size(); i++) {
    makePointerTo(from[i], to[i]);
  }
}

void AliasDb::giveFreshAlias(const Value* value) {
  if (!shouldAnnotate(value)) {
    return;
  }

  if (isTracked(value)) {
    // Inside a loop, we may have given a fresh alias to this value already, so
    // skip
    return;
  }

  elementMap_[value] = memoryDAG_->makeFreshValue(value);
}

Element* AliasDb::getOrCreateElement(const Value* value) {
  if (!isTracked(value)) {
    giveFreshAlias(value);
  }
  return elementMap_.at(value);
}

bool AliasDb::isTracked(const Value* v) const {
  return isWildcard(v) || elementMap_.count(v);
}

bool AliasDb::moveAfterTopologicallyValid(Node* n, Node* movePoint) {
  return tryMove(n, movePoint, MoveSide::AFTER, /*dryRun=*/false);
}

bool AliasDb::couldMoveAfterTopologically(Node* n, Node* movePoint) {
  return tryMove(n, movePoint, MoveSide::AFTER, /*dryRun=*/true);
}

bool AliasDb::moveBeforeTopologicallyValid(Node* n, Node* movePoint) {
  // We have to distinguish the move side (instead of just moving after
  // n->prev()). Consider the following example:
  // If the dependency graph looks like
  //   n -> movePoint -> o
  // then moveBefore(o) will end up with
  //   n, o, movePoint
  // but moveAfter(n) will return false.
  return tryMove(n, movePoint, MoveSide::BEFORE, /*dryRun=*/false);
}

bool AliasDb::couldMoveBeforeTopologically(Node* n, Node* movePoint) {
  return tryMove(n, movePoint, MoveSide::BEFORE, /*dryRun=*/true);
}

// Helper for topologically-safe node moves. See `tryMove()` for details.
class AliasDb::WorkingSet {
 public:
  explicit WorkingSet(Node* mover, const AliasDb& aliasDb) : aliasDb_(aliasDb) {
    add(mover);
  }

  // Add `n` to the working set
  void add(Node* n) {
    nodes_.push_back(n);
    for (const auto user : getUsersSameBlock(n)) {
      users_.insert(user);
    }

    for (const auto& write : aliasDb_.getWrites(n, /*recurseBlocks=*/true)) {
      writes_.insert(write);
    }
    for (const auto& read : aliasDb_.getReads(n, /*recurseBlocks=*/true)) {
      reads_.insert(read);
    }
    if (aliasDb_.hasWildcard(n)) {
      numWildcards_++;
    }
  }

  void eraseMover() {
    auto mover = nodes_.front();
    for (const auto user : getUsersSameBlock(mover)) {
      const auto it = users_.find(user);
      if (it != users_.end()) {
        users_.erase(it);
      }
    }

    for (const auto& write :
         aliasDb_.getWrites(mover, /*recurseBlocks=*/true)) {
      const auto it = writes_.find(write);
      if (it != writes_.end()) {
        writes_.erase(it);
      }
    }
    for (const auto& read : aliasDb_.getReads(mover, /*recurseBlocks=*/true)) {
      const auto it = reads_.find(read);
      if (it != reads_.end()) {
        reads_.erase(it);
      }
    }
    if (aliasDb_.hasWildcard(mover)) {
      numWildcards_--;
    }
    nodes_.pop_front();
  }

  const std::list<Node*>& nodes() {
    return nodes_;
  }

  // Does the working set depend on `n`?
  bool dependsOn(Node* n) const {
    if (nodes_.empty()) {
      return false;
    }

    return hasDataDependency(n) || hasMutabilityDependency(n);
  }

 private:
  bool hasDataDependency(Node* n) const {
    if (n->isAfter(nodes_.front())) {
      return producesFor(n);
    } else {
      return consumesFrom(n);
    }
  }

  bool hasMutabilityDependency(Node* n) const {
    // 1. Handle wildcard dependencies:
    // If the working set has a wildcard, `n` can't write to anything.
    if (numWildcards_ > 0 && aliasDb_.hasWrites(n)) {
      return true;
    }

    // If `n` has a wildcard, the working set can't write to anything.
    if (aliasDb_.hasWildcard(n) && writes_.size() > 0) {
      return true;
    }

    // 2. Handle regular mutable dependencies
    // Check that `n` does not write to anything used by the working set
    const auto nWrites = aliasDb_.getWrites(n, /*recurseBlocks=*/true);
    if (aliasDb_.mayAlias(nWrites, reads_)) {
      return true;
    }

    // Check that the working set doesn't write to anything that `n` uses.
    const auto nReads = aliasDb_.getReads(n, /*recurseBlocks=*/true);
    if (aliasDb_.mayAlias(writes_, nReads)) {
      return true;
    }
    return false;
  }

  // Does the working set produce any values consumed by `n`?
  bool producesFor(Node* n) const {
    // This equivalent to asking: does the total use-set of all the nodes in the
    // working set include `n`?
    return users_.count(n) != 0;
  }

  // Does the working set consume any values produced by `n`?
  bool consumesFrom(Node* n) const {
    const auto users = getUsersSameBlock(n);
    return std::any_of(nodes_.begin(), nodes_.end(), [&](Node* node) {
      return users.count(node) != 0;
    });
  }

  // Get all users of outputs of `n`, in the same block as `n`.
  // This means if there is an `if` node that uses an output of `n` in some
  // inner sub-block, we will consider the whole `if` node a user of `n`.
  std::unordered_set<Node*> getUsersSameBlock(Node* n) const {
    std::unordered_set<Node*> users;
    for (const auto output : n->outputs()) {
      for (const auto& use : output->uses()) {
        if (auto sameBlock = findSameBlock(use.user, n)) {
          users.insert(sameBlock);
        }
      }
    }
    return users;
  }

  // Traverse `target`'s blockchain upward until we find a node that shares a
  // block with `n`.
  //
  // If one can't be found (say, because `n` is an inner block and target is
  // outside), then return nullptr. Since we can only reorder nodes within a
  // block, `target` would be irrelevant.
  static Node* findSameBlock(Node* target, Node* n) {
    AT_ASSERT(target->owningGraph() == n->owningGraph());
    if (target->owningBlock() == n->owningBlock()) {
      return target;
    } else {
      // This user is in a sub-block. Traverse the blockchain upward until
      // we arrive at a node that shares a block with `this`
      auto curNode = target;
      while (curNode->owningBlock() != n->owningBlock()) {
        curNode = curNode->owningBlock()->owningNode();
        if (curNode == nullptr) {
          return curNode;
        }
      }
      return curNode;
    }
  }

  const AliasDb& aliasDb_;
  std::list<Node*> nodes_;
  // users => # of working set nodes it uses
  std::unordered_multiset<Node*> users_;
  // Values written to by the working set => number of nodes writing to value
  std::unordered_multiset<const Value*> writes_;
  std::unordered_multiset<const Value*> reads_;
  size_t numWildcards_ = 0;
};

// Try to move `toMove` before/after `movePoint` while preserving value
// dependencies. Returns false iff such a move could not be made.
//
// If `dryRun` is set, don't actually execute the move, just check if the move
// is possible
//
// The basic approach is: have a "working set" that we are moving forward, one
// node at a time. When we can't move past a node (because it depends on the
// working set), then add it to the working set and keep moving until we hit
// `moveAfter`.
bool AliasDb::tryMove(
    Node* toMove,
    Node* movePoint,
    MoveSide moveSide,
    bool dryRun) {
  AT_ASSERT(toMove->owningBlock() == movePoint->owningBlock());
  if (toMove == movePoint) {
    return true;
  }

  // 1. Move from `this` toward movePoint, building up the working set of
  // dependencies
  WorkingSet workingSet(toMove, *this);

  int direction;
  if (toMove->isAfter(movePoint)) {
    direction = kPrevDirection;
  } else {
    direction = kNextDirection;
  }

  auto curNode = toMove->next_in_graph[direction];
  // Move forward one node at a time
  while (curNode != movePoint) {
    if (workingSet.dependsOn(curNode)) {
      // If we can't move past this node, add it to the working set
      workingSet.add(curNode);
    }
    curNode = curNode->next_in_graph[direction];
  }

  // 2. Decide whether we can move it all to `movePoint`.

  // Say we are moving directly before movePoint and `toMove` starts before
  // movePoint in the graph. The move looks like
  //
  //  `toMove`            `toMove`         |
  //  <dependencies>  ->  `movePoint`      | `toMove` and deps are split
  //  `movePoint`         <dependencies>   |
  //
  // Contrast with the case where `toMove` starts AFTER movePoint:
  //
  //  `movePoint`           <dependencies>   |
  //  <dependencies>  ->    `toMove`         | `toMove` and deps are together
  //  `toMove`              `movePoint`      |
  //
  // In the first case, we need to split `this` off from its dependencies, so we
  // can move the dependencies below `movePoint` and keep `toMove` above.
  const bool splitToMoveAndDeps =
      (moveSide == MoveSide::BEFORE && toMove->isBefore(movePoint)) ||
      (moveSide == MoveSide::AFTER && toMove->isAfter(movePoint));

  if (splitToMoveAndDeps) {
    // remove `this` from dependencies to be moved past `movePoint`
    workingSet.eraseMover();
  }

  // Check if we can move the working set past the move point
  if (workingSet.dependsOn(movePoint)) {
    // if we can't, then there are intermediate dependencies between the
    // `this` and `movePoint`, so we can't do the move
    return false;
  }

  if (dryRun) {
    return true;
  }

  // 3. Execute the move
  AT_ASSERT(curNode == movePoint);
  if (splitToMoveAndDeps) {
    // Move `toMove`
    move(toMove, movePoint, moveSide);

    // Then move all of its dependencies on the other side of `movePoint`
    const auto reversed =
        moveSide == MoveSide::BEFORE ? MoveSide::AFTER : MoveSide::BEFORE;
    for (auto n : workingSet.nodes()) {
      move(n, curNode, reversed);
      curNode = n;
    }
  } else {
    // Just append/prepend everything to `movePoint`
    for (auto n : workingSet.nodes()) {
      move(n, curNode, moveSide);
      curNode = n;
    }
  }
  return true;
}

// Helper function so we can generalize `tryMove`
void AliasDb::move(Node* toMove, Node* movePoint, MoveSide moveSide) {
  switch (moveSide) {
    case MoveSide::BEFORE:
      toMove->moveBefore(movePoint);
      break;
    case MoveSide::AFTER:
      toMove->moveAfter(movePoint);
      break;
  }
}

bool AliasDb::hasUntrackedEffects(Node* node) const {
  bool touchesWildcard = false;
  if (const auto lastWildcard = getLastWildcard()) {
    touchesWildcard = hasWrites(node) &&
        (isBeforeSameGraph(node, *lastWildcard) || node == *lastWildcard);
  }

  return writesToInputAlias(node) || touchesWildcard;
}

// Nodes must be in the same graph in order to do `isBefore` or `isAfter`. This
// traverses the subgraph "chain" upward until we find two nodes that share an
// owning graph.
//
// NOTE: this is n^2 in subgraph depth. Right now the maximum depth is like 2,
// but if we ever do huge nested subgraphs we'll need to reconsider this.
bool AliasDb::isBeforeSameGraph(const Node* a, const Node* b) const {
  auto lhs = a;
  while (true) {
    auto rhs = b;
    while (true) {
      if (lhs->owningGraph() == rhs->owningGraph()) {
        return lhs->isBefore(rhs);
      }
      if (!subgraphToOwner_.count(rhs->owningGraph())) {
        break;
      }
      rhs = subgraphToOwner_.at(rhs->owningGraph());
    }
    if (!subgraphToOwner_.count(lhs->owningGraph())) {
      break;
    }
    lhs = subgraphToOwner_.at(lhs->owningGraph());
  }
  AT_ASSERT(false);
}

c10::optional<const Node*> AliasDb::getLastWildcard() const {
  auto it = std::max_element(
      wildcardNodes_.cbegin(),
      wildcardNodes_.cend(),
      [this](const Node* a, const Node* b) { return isBeforeSameGraph(a, b); });
  if (it != wildcardNodes_.end()) {
    return *it;
  } else {
    return c10::nullopt;
  }
}

bool aliasAnalysisHasSpecialCaseFor(Symbol symbol) {
  // WARNING: by adding a case to this list, you are asserting that you have
  // added a case for the unschematized node in AliasDb::analyze
  const static std::unordered_set<Symbol> handled = {
      prim::If,
      prim::Loop,
      prim::FusionGroup,
      prim::DifferentiableGraph,
      prim::Constant,
      prim::DictConstruct,
      prim::ListConstruct,
      prim::TupleConstruct,
      prim::AutogradZero,
      prim::FusedConcat,
      prim::GradOf,
      prim::MMTreeReduce,
      prim::MMBatchSide,
      prim::BroadcastSizes,
      prim::ChunkSizes,
      prim::Function,
      prim::TupleUnpack,
      prim::TupleIndex,
      prim::DictIndex,
      prim::TupleSlice,
      prim::ListUnpack,
      prim::PythonOp,
      prim::ConstantChunk,
      prim::BroadcastingChunk,
      prim::fork,
      prim::CreateObject,
      prim::AutogradAdd,
      prim::GetAttr,
      prim::SetAttr,
      aten::wait,
      aten::add,
      aten::sub,
      aten::mul,
      aten::div,
  };

  // Operators that should not be used by alias analysis
  const static std::unordered_set<Symbol> purposefully_not_handled = {
      prim::Print,
      prim::Load,
      prim::Store,
      prim::Drop,
      at::onnx::Reshape,
      at::onnx::Shape,
      prim::AutogradAnyNonZero,
      prim::AutogradAdd,
  };

  return handled.count(symbol) || purposefully_not_handled.count(symbol);
}

// Register `v` as a wildcard value.
void AliasDb::setWildcard(const Value* v) {
  if (!shouldAnnotate(v)) {
    return;
  }
  wildcards_.insert(v);
}

void AliasDb::rebuildWriteCache() const {
  for (const auto& pr : writeIndex_) {
    const auto& writtenValues = pr.second;

    for (const auto value : writtenValues) {
      for (const auto loc : elementMap_.at(value)->getMemoryLocations()) {
        writeCache_.insert(loc);
      }
    }
  }
  isWriteCacheStale_ = false;
}

ValueSet AliasDb::getMemoryLocations(const Value* v) const {
  ValueSet ret;
  if (!elementMap_.count(v)) {
    return ret;
  }

  for (const auto el : elementMap_.at(v)->getMemoryLocations()) {
    ret.insert(el->value);
  }
  return ret;
}
} // namespace jit
} // namespace torch
