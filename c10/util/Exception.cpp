#include "c10/util/Exception.h"
#include "c10/util/Backtrace.h"
#include "c10/util/Type.h"
#include "c10/util/Logging.h"

#include <iostream>
#include <numeric>
#include <string>

namespace c10 {

Error::Error(
    const std::string& new_msg,
    const std::string& backtrace,
    const void* caller)
    : msg_stack_{new_msg}, backtrace_(backtrace), caller_(caller) {
  msg_ = msg();
  msg_without_backtrace_ = msg_without_backtrace();
}

// PyTorch-style error message
// Error::Error(SourceLocation source_location, const std::string& msg)
// NB: This is defined in Logging.cpp for access to GetFetchStackTrace

// Caffe2-style error message
Error::Error(
    const char* file,
    const uint32_t line,
    const char* condition,
    const std::string& msg,
    const std::string& backtrace,
    const void* caller)
    : Error(
          str("[enforce fail at ",
              detail::StripBasename(file),
              ":",
              line,
              "] ",
              condition,
              ". ",
              msg,
              "\n"),
          backtrace,
          caller) {}

std::string Error::msg() const {
  return std::accumulate(
             msg_stack_.begin(), msg_stack_.end(), std::string("")) +
      backtrace_;
}

std::string Error::msg_without_backtrace() const {
  return std::accumulate(msg_stack_.begin(), msg_stack_.end(), std::string(""));
}

void Error::AppendMessage(const std::string& new_msg) {
  msg_stack_.push_back(new_msg);
  // Refresh the cache
  // TODO: Calling AppendMessage O(n) times has O(n^2) cost.  We can fix
  // this perf problem by populating the fields lazily... if this ever
  // actually is a problem.
  msg_ = msg();
  msg_without_backtrace_ = msg_without_backtrace();
}

void Warning::warn(SourceLocation source_location, std::string msg) {
  warning_handler_(source_location, msg.c_str());
}

void Warning::set_warning_handler(handler_t handler) {
  warning_handler_ = handler;
}

void Warning::print_warning(
    const SourceLocation& source_location,
    const char* msg) {
  std::cerr << "Warning: " << msg << " (" << source_location << ")\n";
}

Warning::handler_t Warning::warning_handler_ = &Warning::print_warning;

std::string GetExceptionString(const std::exception& e) {
#ifdef __GXX_RTTI
  return demangle(typeid(e).name()) + ": " + e.what();
#else
  return std::string("Exception (no RTTI available): ") + e.what();
#endif // __GXX_RTTI
}

} // namespace c10
