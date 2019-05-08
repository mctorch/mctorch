#include "c10/util/Backtrace.h"
#include "c10/util/Optional.h"
#include "c10/util/Type.h"

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if (defined(__ANDROID__)) ||                                                 \
    (defined(__APPLE__) &&                                                    \
     (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE)) || \
    defined(_WIN32) || defined(__EMSCRIPTEN__)
// No backtrace on mobile, windows and emscripten platforms.
#define SUPPORTS_BACKTRACE 0
#else
#define SUPPORTS_BACKTRACE 1
#include <cxxabi.h>
#include <execinfo.h>
#endif

namespace c10 {

// TODO: This backtrace retrieval can be implemented on Windows via the Windows
// API using `CaptureStackBackTrace` and `SymFromAddr`.
// https://stackoverflow.com/questions/5693192/win32-backtrace-from-c-code
// https://stackoverflow.com/questions/26398064/counterpart-to-glibcs-backtrace-and-backtrace-symbols-on-windows
// https://msdn.microsoft.com/en-us/library/windows/desktop/bb204633%28v=vs.85%29.aspx.
#if SUPPORTS_BACKTRACE
namespace {

struct FrameInformation {
  /// If available, the demangled name of the function at this frame, else
  /// whatever (possibly mangled) name we got from `backtrace()`.
  std::string function_name;
  /// This is a number in hexadecimal form (e.g. "0xdead") representing the
  /// offset into the function's machine code at which the function's body
  /// starts, i.e. skipping the "prologue" that handles stack manipulation and
  /// other calling convention things.
  std::string offset_into_function;
  /// NOTE: In debugger parlance, the "object file" refers to the ELF file that
  /// the symbol originates from, i.e. either an executable or a library.
  std::string object_file;
};

bool is_python_frame(const FrameInformation& frame) {
  return frame.object_file == "python" || frame.object_file == "python3" ||
      (frame.object_file.find("libpython") != std::string::npos);
}

c10::optional<FrameInformation> parse_frame_information(
    const std::string& frame_string) {
  FrameInformation frame;

  // This is the function name in the CXX ABI mangled format, e.g. something
  // like _Z1gv. Reference:
  // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling
  std::string mangled_function_name;

#if defined(__GLIBCXX__)
  // In GLIBCXX, `frame_string` follows the pattern
  // `<object-file>(<mangled-function-name>+<offset-into-function>)
  // [<return-address>]`

  auto function_name_start = frame_string.find("(");
  if (function_name_start == std::string::npos) {
    return c10::nullopt;
  }
  function_name_start += 1;

  auto offset_start = frame_string.find('+', function_name_start);
  if (offset_start == std::string::npos) {
    return c10::nullopt;
  }
  offset_start += 1;

  const auto offset_end = frame_string.find(')', offset_start);
  if (offset_end == std::string::npos) {
    return c10::nullopt;
  }

  frame.object_file = frame_string.substr(0, function_name_start - 1);
  frame.offset_into_function =
      frame_string.substr(offset_start, offset_end - offset_start);

  // NOTE: We don't need to parse the return address because
  // we already have it from the call to `backtrace()`.

  mangled_function_name = frame_string.substr(
      function_name_start, (offset_start - 1) - function_name_start);
#elif defined(_LIBCPP_VERSION)
  // In LIBCXX, The pattern is
  // `<frame number> <object-file> <return-address> <mangled-function-name> +
  // <offset-into-function>`
  std::string skip;
  std::istringstream input_stream(frame_string);
  // operator>>() does not fail -- if the input stream is corrupted, the
  // strings will simply be empty.
  input_stream >> skip >> frame.object_file >> skip >> mangled_function_name >>
      skip >> frame.offset_into_function;
#else
#warning Unknown standard library, backtraces may have incomplete debug information
  return c10::nullopt;
#endif // defined(__GLIBCXX__)

  // Some system-level functions don't have sufficient debug information, so
  // we'll display them as "<unknown function>". They'll still have a return
  // address and other pieces of information.
  if (mangled_function_name.empty()) {
    frame.function_name = "<unknown function>";
    return frame;
  }

  frame.function_name = demangle(mangled_function_name.c_str());
  return frame;
}

} // anonymous namespace
#endif // SUPPORTS_BACKTRACE

std::string get_backtrace(
    size_t frames_to_skip,
    size_t maximum_number_of_frames,
    bool skip_python_frames) {
#if SUPPORTS_BACKTRACE

  // We always skip this frame (backtrace).
  frames_to_skip += 1;

  std::vector<void*> callstack(
      frames_to_skip + maximum_number_of_frames, nullptr);
  // backtrace() gives us a list of return addresses in the current call stack.
  // NOTE: As per man (3) backtrace it can never fail
  // (http://man7.org/linux/man-pages/man3/backtrace.3.html).
  auto number_of_frames =
      ::backtrace(callstack.data(), static_cast<int>(callstack.size()));

  // Skip as many frames as requested. This is not efficient, but the sizes here
  // are small and it makes the code nicer and safer.
  for (; frames_to_skip > 0 && number_of_frames > 0;
       --frames_to_skip, --number_of_frames) {
    callstack.erase(callstack.begin());
  }

  // `number_of_frames` is strictly less than the current capacity of
  // `callstack`, so this is just a pointer subtraction and makes the subsequent
  // code safer.
  callstack.resize(static_cast<size_t>(number_of_frames));

  // `backtrace_symbols` takes the return addresses obtained from `backtrace()`
  // and fetches string representations of each stack. Unfortunately it doesn't
  // return a struct of individual pieces of information but a concatenated
  // string, so we'll have to parse the string after. NOTE: The array returned
  // by `backtrace_symbols` is malloc'd and must be manually freed, but not the
  // strings inside the array.
  std::unique_ptr<char*, std::function<void(char**)>> raw_symbols(
      ::backtrace_symbols(callstack.data(), static_cast<int>(callstack.size())),
      /*deleter=*/free);
  const std::vector<std::string> symbols(
      raw_symbols.get(), raw_symbols.get() + callstack.size());

  // The backtrace string goes into here.
  std::ostringstream stream;

  // Toggles to true after the first skipped python frame.
  bool has_skipped_python_frames = false;

  for (size_t frame_number = 0; frame_number < callstack.size();
       ++frame_number) {
    const auto frame = parse_frame_information(symbols[frame_number]);

    if (skip_python_frames && frame && is_python_frame(*frame)) {
      if (!has_skipped_python_frames) {
        stream << "<omitting python frames>\n";
        has_skipped_python_frames = true;
      }
      continue;
    }

    // frame #<number>:
    stream << "frame #" << frame_number << ": ";

    if (frame) {
      // <function_name> + <offset> (<return-address> in <object-file>)
      stream << frame->function_name << " + " << frame->offset_into_function
             << " (" << callstack[frame_number] << " in " << frame->object_file
             << ")\n";
    } else {
      // In the edge-case where we couldn't parse the frame string, we can
      // just use it directly (it may have a different format).
      stream << symbols[frame_number] << "\n";
    }
  }

  return stream.str();
#else // !SUPPORTS_BACKTRACE
  return "(no backtrace available)";
#endif // SUPPORTS_BACKTRACE
}

} // namespace c10
