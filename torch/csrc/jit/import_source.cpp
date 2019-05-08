#include "import_source.h"

#include <torch/csrc/jit/script/parser.h>
#include <torch/csrc/jit/script/resolver.h>

namespace torch {
namespace jit {
namespace script {

struct OpsValue : public SugaredValue {
  OpsValue(size_t version) : version_(version) {}
  std::string kind() const override {
    return "ops";
  }
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    return std::make_shared<BuiltinModule>(field, version_);
  }
  size_t version_;
};

struct ConstantValue : public SugaredValue {
  ConstantValue(IValue value) : value_(std::move(value)) {}
  IValue value_;
  std::string kind() const override {
    return "constant";
  }
  Value* asValue(const SourceRange& loc, Function& m) override {
    return m.graph()->insertConstant(value_);
  }
};

// Represents nested class namespaces, like `foo.bar.Baz`.
// Right now these namespaces can only contain other namespaces or a class type.
struct TORCH_API ClassNamespaceValue : public SugaredValue {
  explicit ClassNamespaceValue(std::string name) : basename_(std::move(name)) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& name) override {
    const auto fullName = basename_ + "." + name;
    if (auto classType = ClassType::get(fullName)) {
      return std::make_shared<ClassValue>(classType);
    }

    return std::make_shared<ClassNamespaceValue>(fullName);
  }
  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  std::string basename_;
};

// This value maps attributes CONSTANTS.c0 CONSTANTS.c1 to entries
// in the 'constants' vector. This table is will be stored in a container format
// and given to the import_method when restoring the code.
struct ConstantTableValue : public SugaredValue {
  ConstantTableValue(ArrayRef<at::Tensor> constants) : constants_(constants) {}
  std::string kind() const override {
    return "CONSTANTS";
  }
  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    const char* field_s = field.c_str();
    char* end;
    int64_t offset = std::strtoll(field_s + 1, &end, 10);
    if (field.size() < 2 || *end != 0)
      throw ErrorReport(loc) << "invalid constant specifier: " << field;
    if (offset < 0 || size_t(offset) >= constants_.size()) {
      throw ErrorReport(loc) << "constant index " << offset
                             << " is out of bounds (constant table has "
                             << constants_.size() << " entries).";
    }
    Value* value = m.graph()->insertConstant(constants_[offset], nullptr, loc);
    return std::make_shared<SimpleValue>(value);
  }

 private:
  ArrayRef<at::Tensor> constants_;
};

// A resolver that doesn't rely on Python, and understands references to model
// constants.
struct SourceResolver : public Resolver {
  explicit SourceResolver(
      size_t version,
      const std::vector<at::Tensor>& constant_table) {
    env_ = {
        {"torch", std::make_shared<BuiltinModule>("aten", version)},
        {"ops", std::make_shared<OpsValue>(version)},
        // Constants present in the model. Used to resolve "CONSTANTS.n" to the
        // actual value
        {"CONSTANTS", std::make_shared<ConstantTableValue>(constant_table)},
        {"fork", std::make_shared<ForkValue>()},
        {"annotate", std::make_shared<AnnotateValue>()},
        {"inf",
         std::make_shared<ConstantValue>(
             std::numeric_limits<double>::infinity())},
        {"nan",
         std::make_shared<ConstantValue>(
             std::numeric_limits<double>::quiet_NaN())},
    };
  }

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const SourceRange& loc) const override {
    auto it = env_.find(name);
    if (it != env_.end()) {
      return it->second;
    }

    if (name == "__torch__") {
      return std::make_shared<ClassNamespaceValue>(name);
    }
    return nullptr;
  }

  TypePtr resolveType(const std::string& name) const override {
    return ClassType::get(name);
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<SugaredValue>> env_;
};

struct SourceImporter {
  SourceImporter(
      const std::string& src,
      const std::vector<at::Tensor>& constant_table,
      const std::function<void(const std::string&)>& import_callback)
      : p_(src),
        import_callback_(import_callback),
        constant_table_(constant_table) {
    version_ = parseVersionNumber();
    resolver_ = std::make_shared<SourceResolver>(version_, constant_table_);
  }

  void importLibs(const std::string& class_qualifier) {
    auto& L = p_.lexer();

    while (L.cur().kind != TK_EOF) {
      parseImportsAndDoCallback();

      std::vector<Def> definitions;
      std::vector<ResolverPtr> resolvers;
      auto class_def = ClassDef(p_.parseClass());
      for (const auto& method_def : class_def.defs()) {
        definitions.emplace_back(method_def);
        resolvers.emplace_back(resolver_);
      }

      auto cu = std::make_shared<CompilationUnit>();
      const auto qualified_classname =
          class_qualifier + "." + class_def.name().name();
      auto class_type = ClassType::create(qualified_classname, cu);
      auto self = [&](Value* v) {
        v->setType(class_type);
        return std::make_shared<SimpleValue>(v);
      };
      cu->define(definitions, resolvers, self);
    }
  }

  void importFunctions(CompilationUnit& cu, const Self& self) {
    parseImportsAndDoCallback();

    std::vector<Def> definitions;
    std::vector<ResolverPtr> resolvers;
    while (p_.lexer().cur().kind != TK_EOF) {
      auto def = Def(p_.parseFunction(/*is_method=*/bool(self)));
      definitions.emplace_back(def);
      resolvers.emplace_back(resolver_);
    }
    cu.define(definitions, resolvers, self);
  }

  size_t parseVersionNumber() {
    auto& L = p_.lexer();
    auto range = L.cur().range;
    auto name = L.expect(TK_IDENT).text();
    L.expect('=');
    std::string version_text = L.expect(TK_NUMBER).text();
    L.expect(TK_NEWLINE);
    auto version = Const::create(L.cur().range, version_text);
    if (name != "op_version_set")
      throw ErrorReport(range) << "expected an assignment to op_version_set";
    if (!version.isIntegral())
      throw ErrorReport(range)
          << "expected an integral version but found " << version.text();
    return size_t(version.asIntegral());
  }

  void parseImportsAndDoCallback() {
    // Gather all imports
    auto& L = p_.lexer();
    std::vector<std::string> imports;
    while (L.nextIf(TK_IMPORT)) {
      std::ostringstream s;
      while (L.cur().kind != TK_NEWLINE) {
        s << L.cur().text();
        L.next();
      }
      L.expect(TK_NEWLINE);
      const auto str = s.str();
      AT_ASSERT(!str.empty());
      imports.push_back(str);
    }

    // Call the callback to actually compile them
    for (const auto& import : imports) {
      if (import_callback_) {
        import_callback_(import);
      }
    }
  }

 private:
  Parser p_;
  size_t version_;
  const std::function<void(const std::string&)>& import_callback_;
  const std::vector<at::Tensor>& constant_table_;
  std::shared_ptr<SourceResolver> resolver_;
};

void import_functions(
    CompilationUnit& cu,
    const std::string& src,
    const std::vector<at::Tensor>& constant_table,
    const Self& self,
    const std::function<void(const std::string&)>& import_callback) {
  SourceImporter importer(src, constant_table, import_callback);
  importer.importFunctions(cu, self);
}

void import_methods(
    const std::shared_ptr<Module>& mod,
    const std::string& src,
    const std::vector<at::Tensor>& constant_table,
    const std::function<void(const std::string&)>& import_callback) {
  auto self = [&](Value* v) {
    v->setType(mod->module_object()->type());
    return std::make_shared<SimpleValue>(v);
  };
  import_functions(
      mod->module_object()->type()->compilation_unit(),
      src,
      constant_table,
      self,
      import_callback);
}

void import_libs(
    const std::string& class_qualifier,
    const std::string& src,
    const std::vector<at::Tensor>& constant_table,
    const std::function<void(const std::string&)>& import_callback) {
  SourceImporter importer(src, constant_table, import_callback);
  importer.importLibs(class_qualifier);
}

} // namespace script
} // namespace jit
} // namespace torch
