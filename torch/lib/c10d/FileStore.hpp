#pragma once

#include <sys/types.h>

#include <unordered_map>

#include <c10d/Store.hpp>

namespace c10d {

class FileStore : public Store {
 public:
  explicit FileStore(const std::string& path, int numWorkers);

  virtual ~FileStore();

  void set(const std::string& key, const std::vector<uint8_t>& value) override;

  std::vector<uint8_t> get(const std::string& key) override;

  int64_t add(const std::string& key, int64_t value) override;

  bool check(const std::vector<std::string>& keys) override;

  void wait(const std::vector<std::string>& keys) override;

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

 protected:
  int64_t addHelper(const std::string& key, int64_t i);

  std::string path_;
  off_t pos_;

  int numWorkers_;
  const std::string cleanupKey_;
  const std::string regularPrefix_;

  std::unordered_map<std::string, std::vector<uint8_t>> cache_;
};

} // namespace c10d
