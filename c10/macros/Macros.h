#ifndef C10_MACROS_MACROS_H_
#define C10_MACROS_MACROS_H_

/* Main entry for c10/macros.
 *
 * In your code, include c10/macros/Macros.h directly, instead of individual
 * files in this folder.
 */

// For build systems that do not directly depend on CMake and directly build
// from the source directory (such as Buck), one may not have a cmake_macros.h
// file at all. In this case, the build system is responsible for providing
// correct macro definitions corresponding to the cmake_macros.h.in file.
//
// In such scenarios, one should define the macro
//     C10_USING_CUSTOM_GENERATED_MACROS
// to inform this header that it does not need to include the cmake_macros.h
// file.

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#include "c10/macros/cmake_macros.h"
#endif // C10_USING_CUSTOM_GENERATED_MACROS

#include "c10/macros/Export.h"

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define C10_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;        \
  classname& operator=(const classname&) = delete

#define CONCAT_IMPL(x, y) x##y
#define MACRO_CONCAT(x, y) CONCAT_IMPL(x, y)

#define MACRO_EXPAND(args) args

/// C10_NODISCARD - Warn if a type or return value is discarded.
#define C10_NODISCARD
#if __cplusplus > 201402L && defined(__has_cpp_attribute)
#if __has_cpp_attribute(nodiscard)
#undef C10_NODISCARD
#define C10_NODISCARD [[nodiscard]]
#endif
// Workaround for llvm.org/PR23435, since clang 3.6 and below emit a spurious
// error when __has_cpp_attribute is given a scoped attribute in C mode.
#elif __cplusplus && defined(__has_cpp_attribute)
#if __has_cpp_attribute(clang::warn_unused_result)
#undef C10_NODISCARD
#define C10_NODISCARD [[clang::warn_unused_result]]
#endif
#endif

// suppress an unused variable.
#ifdef _MSC_VER
#define C10_UNUSED
#else
#define C10_UNUSED __attribute__((__unused__))
#endif //_MSC_VER

// Simply define the namespace, in case a dependent library want to refer to
// the c10 namespace but not any nontrivial files.
namespace c10 {} // namespace c10
namespace c10 { namespace cuda {} }
namespace c10 { namespace hip {} }

// Since C10 is the core library for caffe2 (and aten), we will simply reroute
// all abstractions defined in c10 to be available in caffe2 as well.
// This is only for backwards compatibility. Please use the symbols from the
// c10 namespace where possible.
namespace caffe2 { using namespace c10; }
namespace at { using namespace c10; }
namespace at { namespace cuda { using namespace c10::cuda; }}

// WARNING!!! THIS IS A GIANT HACK!!!
// This line means you cannot simultaneously include c10/hip
// and c10/cuda and then use them from the at::cuda namespace.
// This is true in practice, because HIPIFY works inplace on
// files in ATen/cuda, so it assumes that c10::hip is available
// from at::cuda.  This namespace makes that happen.  When
// HIPIFY is no longer out-of-place, we can switch the cuda
// here to hip and everyone is happy.
namespace at { namespace cuda { using namespace c10::hip; }}

// C10_NORETURN
#if defined(_MSC_VER)
#define C10_NORETURN __declspec(noreturn)
#else
#define C10_NORETURN __attribute__((noreturn))
#endif

// C10_LIKELY/C10_UNLIKELY
//
// These macros provide parentheses, so you can use these macros as:
//
//    if C10_LIKELY(some_expr) {
//      ...
//    }
//
// NB: static_cast to boolean is mandatory in C++, because __builtin_expect
// takes a long argument, which means you may trigger the wrong conversion
// without it.
//
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr)    (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr)  (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr)    (expr)
#define C10_UNLIKELY(expr)  (expr)
#endif

#include <sstream>
#include <string>

#if defined(__CUDACC__) || defined(__HIPCC__)
// Designates functions callable from the host (CPU) and the device (GPU)
#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define C10_HOST __host__
// constants from (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
// The maximum number of threads per multiprocessor is 1024 for Turing architecture (7.5)
// but 2048 for previous architectures. You'll get warnings if you exceed these constants.
// Hence, the following macros adjust the input values from the user to resolve potential warnings.
#if __CUDA_ARCH__ >= 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block size.
// 256 is a good number for this fallback and should give good occupancy and
// versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
// NOTE: if you are thinking of constexpr-ify the inputs to launch bounds, it
//       turns out that although __launch_bounds__ can take constexpr, it
//       can't take a constexpr that has anything to do with templates.
//       Currently we use launch_bounds that depend on template arguments in
//       Loops.cuh, Reduce.cuh and LossCTC.cuh. Hence, C10_MAX_THREADS_PER_BLOCK and
//       C10_MIN_BLOCKS_PER_SM are kept as macros.
// Suppose you were planning to write __launch_bounds__(a, b), based on your performance tuning on a modern GPU.
// Instead, you should write __launch_bounds__(C10_MAX_THREADS_PER_BLOCK(a), C10_MIN_BLOCKS_PER_SM(a, b)),
// which will also properly respect limits on old architectures.
#define C10_MAX_THREADS_PER_BLOCK(val) (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define C10_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm) ((((threads_per_block)*(blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) ? (blocks_per_sm) : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) / (threads_per_block))))
// C10_LAUNCH_BOUNDS is analogous to __launch_bounds__
#define C10_LAUNCH_BOUNDS_0 __launch_bounds__(256, 4) // default launch bounds that should give good occupancy and versatility across all architectures.
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))
#define C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))), (C10_MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))
#else
#define C10_HOST_DEVICE
#define C10_HOST
#define C10_DEVICE
#endif

#ifdef __HIP_PLATFORM_HCC__
#define C10_HIP_HOST_DEVICE __host__ __device__
#else
#define C10_HIP_HOST_DEVICE
#endif

#if defined(__ANDROID__)
#define C10_ANDROID 1
#define C10_MOBILE 1
#elif (                   \
    defined(__APPLE__) && \
    (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define C10_IOS 1
#define C10_MOBILE 1
#elif (defined(__APPLE__) && TARGET_OS_MAC)
#define C10_IOS 1
#endif // ANDROID / IOS / MACOS

// Portably determine if a type T is trivially copyable or not.
#if __GNUG__ && __GNUC__ < 5
#define C10_IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#define C10_IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif

#endif // C10_MACROS_MACROS_H_
