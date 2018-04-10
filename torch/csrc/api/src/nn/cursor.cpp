#include <torch/nn/cursor.h>
#include <torch/nn/module.h>

#include <cstdint>
#include <string>

namespace torch { namespace nn {

ModuleCursor::ModuleCursor(Module* module, CursorPolicy default_policy) {}

// Returns *this;
ModuleCursor& ModuleCursor::begin() noexcept {
  return *this;
}
ModuleCursor& ModuleCursor::end() noexcept {
  return *this;
}

ModuleCursor& ModuleCursor::operator++() {
  next();
  return *this;
}

ModuleCursor ModuleCursor::operator++(int) {
  auto old = *this;
  ++*this;
  return old;
}

bool ModuleCursor::operator==(const ModuleCursor& other) const noexcept {
  return false;
}
bool ModuleCursor::operator!=(const ModuleCursor& other) const noexcept {
  return false;
}

// Can this maybe return the concrete type?
ModuleCursor::Item ModuleCursor::operator*() {
  return {};
}

ModuleCursor::Item ModuleCursor::next() {
  return {};
}
ModuleCursor::Item ModuleCursor::next_bfs() {
  return {};
}
ModuleCursor::Item ModuleCursor::next_dfs() {
  return {};
}
ModuleCursor::Item ModuleCursor::next(CursorPolicy policy) {
  return {};
}

Module* ModuleCursor::find(const std::string& key) const noexcept {
  return nullptr;
}
bool ModuleCursor::contains(const std::string& key) const noexcept {
  return false;
}

size_t ModuleCursor::count() const noexcept {
  size_t value = 0;
  // apply([&value] (const std::string&, Module&) { ++value; });
  return value;
}

}} // namespace torch::nn
