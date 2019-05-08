// !!!! PLEASE READ !!!!
// Minimize (transitively) included headers from _avx*.cc because some of the
// functions defined in the headers compiled with platform dependent compiler
// options can be reused by other translation units generating illegal
// instruction run-time error.

// Common utilities for writing performance kernels and easy dispatching of
// different backends.
/*
The general workflow shall be as follows, say we want to
implement a functionality called void foo(int a, float b).

In foo.h, do:
   void foo(int a, float b);

In foo_avx512.cc, do:
   void foo__avx512(int a, float b) {
     [actual avx512 implementation]
   }

In foo_avx2.cc, do:
   void foo__avx2(int a, float b) {
     [actual avx2 implementation]
   }

In foo_avx.cc, do:
   void foo__avx(int a, float b) {
     [actual avx implementation]
   }

In foo.cc, do:
   // The base implementation should *always* be provided.
   void foo__base(int a, float b) {
     [base, possibly slow implementation]
   }
   decltype(foo__base) foo__avx512;
   decltype(foo__base) foo__avx2;
   decltype(foo__base) foo__avx;
   void foo(int a, float b) {
     // You should always order things by their preference, faster
     // implementations earlier in the function.
     AVX512_DO(foo, a, b);
     AVX2_DO(foo, a, b);
     AVX_DO(foo, a, b);
     BASE_DO(foo, a, b);
   }

*/
// Details: this functionality basically covers the cases for both build time
// and run time architecture support.
//
// During build time:
//    The build system should provide flags CAFFE2_PERF_WITH_AVX512,
//    CAFFE2_PERF_WITH_AVX2, and CAFFE2_PERF_WITH_AVX that corresponds to the
//    __AVX512F__, __AVX512DQ__, __AVX512VL__, __AVX2__, and __AVX__ flags the
//    compiler provides. Note that we do not use the compiler flags but rely on
//    the build system flags, because the common files (like foo.cc above) will
//    always be built without __AVX512F__, __AVX512DQ__, __AVX512VL__, __AVX2__
//    and __AVX__.
// During run time:
//    we use cpuid to identify cpu support and run the proper functions.

#pragma once

#include "caffe2/utils/cpuid.h"

// DO macros: these should be used in your entry function, similar to foo()
// above, that routes implementations based on CPU capability.

#define BASE_DO(funcname, ...) return funcname##__base(__VA_ARGS__);

#ifdef CAFFE2_PERF_WITH_AVX512
#define AVX512_DO(funcname, ...)                       \
  if (GetCpuId().avx512f() && GetCpuId().avx512dq() && \
      GetCpuId().avx512vl()) {                         \
    return funcname##__avx512(__VA_ARGS__);            \
  }
#else // CAFFE2_PERF_WITH_AVX512
#define AVX512_DO(funcname, ...)
#endif // CAFFE2_PERF_WITH_AVX512

#ifdef CAFFE2_PERF_WITH_AVX2
#define AVX2_DO(funcname, ...)            \
  if (GetCpuId().avx2()) {                \
    return funcname##__avx2(__VA_ARGS__); \
  }
#define AVX2_FMA_DO(funcname, ...)             \
  if (GetCpuId().avx2() && GetCpuId().fma()) { \
    return funcname##__avx2_fma(__VA_ARGS__);  \
  }
#else // CAFFE2_PERF_WITH_AVX2
#define AVX2_DO(funcname, ...)
#define AVX2_FMA_DO(funcname, ...)
#endif // CAFFE2_PERF_WITH_AVX2

#ifdef CAFFE2_PERF_WITH_AVX
#define AVX_DO(funcname, ...)            \
  if (GetCpuId().avx()) {                \
    return funcname##__avx(__VA_ARGS__); \
  }
#define AVX_F16C_DO(funcname, ...)             \
  if (GetCpuId().avx() && GetCpuId().f16c()) { \
    return funcname##__avx_f16c(__VA_ARGS__);  \
  }
#else // CAFFE2_PERF_WITH_AVX
#define AVX_DO(funcname, ...)
#define AVX_F16C_DO(funcname, ...)
#endif // CAFFE2_PERF_WITH_AVX
