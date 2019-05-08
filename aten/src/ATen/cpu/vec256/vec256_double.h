#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#if defined(__AVX__) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

#if defined(__AVX__) && !defined(_MSC_VER)

template <> class Vec256<double> {
private:
  __m256d values;
public:
  static constexpr int size() {
    return 4;
  }
  Vec256() {}
  Vec256(__m256d v) : values(v) {}
  Vec256(double val) {
    values = _mm256_set1_pd(val);
  }
  Vec256(double val1, double val2, double val3, double val4) {
    values = _mm256_setr_pd(val1, val2, val3, val4);
  }
  operator __m256d() const {
    return values;
  }
  template <int64_t mask>
  static Vec256<double> blend(const Vec256<double>& a, const Vec256<double>& b) {
    return _mm256_blend_pd(a.values, b.values, mask);
  }
  static Vec256<double> blendv(const Vec256<double>& a, const Vec256<double>& b,
                               const Vec256<double>& mask) {
    return _mm256_blendv_pd(a.values, b.values, mask.values);
  }
  static Vec256<double> arange(double base = 0., double step = 1.) {
    return Vec256<double>(base, base + step, base + 2 * step, base + 3 * step);
  }
  static Vec256<double> set(const Vec256<double>& a, const Vec256<double>& b,
                            int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
    }
    return b;
  }
  static Vec256<double> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));

    __at_align32__ double tmp_values[size()];
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(double));
    return _mm256_load_pd(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      double tmp_values[size()];
      _mm256_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(double));
    }
  }
  const double& operator[](int idx) const  = delete;
  double& operator[](int idx) = delete;
  Vec256<double> map(double (*f)(double)) const {
    __at_align32__ double tmp[4];
    store(tmp);
    for (int64_t i = 0; i < 4; i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<double> abs() const {
    auto mask = _mm256_set1_pd(-0.f);
    return _mm256_andnot_pd(mask, values);
  }
  Vec256<double> acos() const {
    return Vec256<double>(Sleef_acosd4_u10(values));
  }
  Vec256<double> asin() const {
    return Vec256<double>(Sleef_asind4_u10(values));
  }
  Vec256<double> atan() const {
    return Vec256<double>(Sleef_atand4_u10(values));
  }
  Vec256<double> erf() const {
    return Vec256<double>(Sleef_erfd4_u10(values));
  }
  Vec256<double> erfc() const {
    return Vec256<double>(Sleef_erfcd4_u15(values));
  }
  Vec256<double> exp() const {
    return Vec256<double>(Sleef_expd4_u10(values));
  }
  Vec256<double> expm1() const {
    return Vec256<double>(Sleef_expm1d4_u10(values));
  }
  Vec256<double> log() const {
    return Vec256<double>(Sleef_logd4_u10(values));
  }
  Vec256<double> log2() const {
    return Vec256<double>(Sleef_log2d4_u10(values));
  }
  Vec256<double> log10() const {
    return Vec256<double>(Sleef_log10d4_u10(values));
  }
  Vec256<double> log1p() const {
    return Vec256<double>(Sleef_log1pd4_u10(values));
  }
  Vec256<double> sin() const {
    return map(std::sin);
  }
  Vec256<double> sinh() const {
    return map(std::sinh);
  }
  Vec256<double> cos() const {
    return map(std::cos);
  }
  Vec256<double> cosh() const {
    return map(std::cos);
  }
  Vec256<double> ceil() const {
    return _mm256_ceil_pd(values);
  }
  Vec256<double> floor() const {
    return _mm256_floor_pd(values);
  }
  Vec256<double> frac() const;
  Vec256<double> neg() const {
    return _mm256_xor_pd(_mm256_set1_pd(-0.), values);
  }
  Vec256<double> round() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vec256<double> tan() const {
    return map(std::tan);
  }
  Vec256<double> tanh() const {
    return Vec256<double>(Sleef_tanhd4_u10(values));
  }
  Vec256<double> trunc() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vec256<double> sqrt() const {
    return _mm256_sqrt_pd(values);
  }
  Vec256<double> reciprocal() const {
    return _mm256_div_pd(_mm256_set1_pd(1), values);
  }
  Vec256<double> rsqrt() const {
    return _mm256_div_pd(_mm256_set1_pd(1), _mm256_sqrt_pd(values));
  }
  Vec256<double> pow(const Vec256<double> &b) const {
    return Vec256<double>(Sleef_powd4_u10(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vec256<double> operator==(const Vec256<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ);
  }

  Vec256<double> operator!=(const Vec256<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_NEQ_OQ);
  }

  Vec256<double> operator<(const Vec256<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LT_OQ);
  }

  Vec256<double> operator<=(const Vec256<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LE_OQ);
  }

  Vec256<double> operator>(const Vec256<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GT_OQ);
  }

  Vec256<double> operator>=(const Vec256<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GE_OQ);
  }
};

template <>
Vec256<double> inline operator+(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_add_pd(a, b);
}

template <>
Vec256<double> inline operator-(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_sub_pd(a, b);
}

template <>
Vec256<double> inline operator*(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_mul_pd(a, b);
}

template <>
Vec256<double> inline operator/(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_div_pd(a, b);
}

// frac. Implement this here so we can use subtraction.
Vec256<double> Vec256<double>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<double> inline maximum(const Vec256<double>& a, const Vec256<double>& b) {
  Vec256<double> max = _mm256_max_pd(a, b);
  Vec256<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<double> inline minimum(const Vec256<double>& a, const Vec256<double>& b) {
  Vec256<double> min = _mm256_min_pd(a, b);
  Vec256<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(min, isnan);
}

template <>
Vec256<double> inline operator&(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_and_pd(a, b);
}

template <>
Vec256<double> inline operator|(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_or_pd(a, b);
}

template <>
Vec256<double> inline operator^(const Vec256<double>& a, const Vec256<double>& b) {
  return _mm256_xor_pd(a, b);
}

template <>
void convert(const double* src, double* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vec256<double>::size()); i += Vec256<double>::size()) {
    _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

#ifdef __AVX2__
template <>
Vec256<double> inline fmadd(const Vec256<double>& a, const Vec256<double>& b, const Vec256<double>& c) {
  return _mm256_fmadd_pd(a, b, c);
}
#endif

#endif

}}}
