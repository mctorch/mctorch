#pragma once

#include <cstdint>
#include <string>

namespace torch { namespace nn {

struct Module;

enum class CursorPolicy { BFS, DFS };

// "Cursor"s allow recursive traversal of the module tree with a depth-first or
// breadth-first policy. They have `begin()` and `end()` and can be used like
// containers for iteration
//
class ModuleCursor {
 public:
  // This should just be detail::OrderedDict::Item
  struct Item {
    const std::string& key() const noexcept {
      return "";
    }
    Module& value() const noexcept {
      return *m;
    }
    Module& operator*() {
      return *m;
    }
    Module* m;
  };

  // Returns *this;
  ModuleCursor& begin() noexcept;
  ModuleCursor& end() noexcept;

  ModuleCursor& operator++();
  ModuleCursor operator++(int);

  bool operator==(const ModuleCursor& other) const noexcept;
  bool operator!=(const ModuleCursor& other) const noexcept;

  // Can this maybe return the concrete type?
  Item operator*();

  Item next();
  Item next_bfs();
  Item next_dfs();
  Item next(CursorPolicy policy);

  template <typename Function>
  void apply(const Function& function) {
    for (auto module : *this) {
      function(module.key(), module.value());
    }
  }

  // Make this Ref = reference_wrapper<T>?
  Module* find(const std::string& key) const noexcept;
  bool contains(const std::string& key) const noexcept;

  size_t count() const noexcept;

 private:
  friend class Module;
  explicit ModuleCursor(
      Module* module,
      CursorPolicy default_policy = CursorPolicy::DFS);
};

/// Implemented in terms of ModuleCursor
struct ParameterCursor {};

/// Maybe just a typedef to ^
struct BufferCursor {};

}} // namespace torch::nn
