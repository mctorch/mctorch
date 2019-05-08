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

template <> class Vec256<float> {
private:
  __m256 values;
public:
  static constexpr int size() {
    return 8;
  }
  Vec256() {}
  Vec256(__m256 v) : values(v) {}
  Vec256(float val) {
    values = _mm256_set1_ps(val);
  }
  Vec256(float val1, float val2, float val3, float val4,
         float val5, float val6, float val7, float val8) {
    values = _mm256_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8);
  }
  operator __m256() const {
    return values;
  }
  template <int64_t mask>
  static Vec256<float> blend(const Vec256<float>& a, const Vec256<float>& b) {
    return _mm256_blend_ps(a.values, b.values, mask);
  }
  static Vec256<float> blendv(const Vec256<float>& a, const Vec256<float>& b,
                              const Vec256<float>& mask) {
    return _mm256_blendv_ps(a.values, b.values, mask.values);
  }
  static Vec256<float> arange(float base = 0.f, float step = 1.f) {
    return Vec256<float>(
      base,            base +     step, base + 2 * step, base + 3 * step,
      base + 4 * step, base + 5 * step, base + 6 * step, base + 7 * step);
  }
  static Vec256<float> set(const Vec256<float>& a, const Vec256<float>& b,
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
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
    }
    return b;
  }
  static Vec256<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
    __at_align32__ float tmp_values[size()];
    std::memcpy(
        tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
    return _mm256_loadu_ps(tmp_values);
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      float tmp_values[size()];
      _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
  const float& operator[](int idx) const  = delete;
  float& operator[](int idx) = delete;
  Vec256<float> map(float (*f)(float)) const {
    __at_align32__ float tmp[8];
    store(tmp);
    for (int64_t i = 0; i < 8; i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> abs() const {
    auto mask = _mm256_set1_ps(-0.f);
    return _mm256_andnot_ps(mask, values);
  }
  Vec256<float> acos() const {
    return Vec256<float>(Sleef_acosf8_u10(values));
  }
  Vec256<float> asin() const {
    return Vec256<float>(Sleef_asinf8_u10(values));
  }
  Vec256<float> atan() const {
    return Vec256<float>(Sleef_atanf8_u10(values));
  }
  Vec256<float> erf() const {
    return Vec256<float>(Sleef_erff8_u10(values));
  }
  Vec256<float> erfc() const {
    return Vec256<float>(Sleef_erfcf8_u15(values));
  }
  Vec256<float> exp() const {
    return Vec256<float>(Sleef_expf8_u10(values));
  }
  Vec256<float> expm1() const {
    return Vec256<float>(Sleef_expm1f8_u10(values));
  }
  Vec256<float> log() const {
    return Vec256<float>(Sleef_logf8_u10(values));
  }
  Vec256<float> log2() const {
    return Vec256<float>(Sleef_log2f8_u10(values));
  }
  Vec256<float> log10() const {
    return Vec256<float>(Sleef_log10f8_u10(values));
  }
  Vec256<float> log1p() const {
    return Vec256<float>(Sleef_log1pf8_u10(values));
  }
  Vec256<float> frac() const;
  Vec256<float> sin() const {
    return map(std::sin);
  }
  Vec256<float> sinh() const {
    return map(std::sinh);
  }
  Vec256<float> cos() const {
    return map(std::cos);
  }
  Vec256<float> cosh() const {
    return map(std::cosh);
  }
  Vec256<float> ceil() const {
    return _mm256_ceil_ps(values);
  }
  Vec256<float> floor() const {
    return _mm256_floor_ps(values);
  }
  Vec256<float> neg() const {
    return _mm256_xor_ps(_mm256_set1_ps(-0.f), values);
  }
  Vec256<float> round() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vec256<float> tan() const {
    return map(std::tan);
  }
  Vec256<float> tanh() const {
    return Vec256<float>(Sleef_tanhf8_u10(values));
  }
  Vec256<float> trunc() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vec256<float> sqrt() const {
    return _mm256_sqrt_ps(values);
  }
  Vec256<float> reciprocal() const {
    return _mm256_div_ps(_mm256_set1_ps(1), values);
  }
  Vec256<float> rsqrt() const {
    return _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(values));
  }
  Vec256<float> pow(const Vec256<float> &b) const {
    return Vec256<float>(Sleef_powf8_u10(values, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vec256<float> operator==(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ);
  }

  Vec256<float> operator!=(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_NEQ_OQ);
  }

  Vec256<float> operator<(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LT_OQ);
  }

  Vec256<float> operator<=(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_LE_OQ);
  }

  Vec256<float> operator>(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GT_OQ);
  }

  Vec256<float> operator>=(const Vec256<float>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_GE_OQ);
  }
};

template <>
Vec256<float> inline operator+(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_add_ps(a, b);
}

template <>
Vec256<float> inline operator-(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_sub_ps(a, b);
}

template <>
Vec256<float> inline operator*(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_mul_ps(a, b);
}

template <>
Vec256<float> inline operator/(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_div_ps(a, b);
}

// frac. Implement this here so we can use subtraction
Vec256<float> Vec256<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<float> inline maximum(const Vec256<float>& a, const Vec256<float>& b) {
  Vec256<float> max = _mm256_max_ps(a, b);
  Vec256<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<float> inline minimum(const Vec256<float>& a, const Vec256<float>& b) {
  Vec256<float> min = _mm256_min_ps(a, b);
  Vec256<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_ps(min, isnan);
}

template <>
Vec256<float> inline operator&(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_and_ps(a, b);
}

template <>
Vec256<float> inline operator|(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_or_ps(a, b);
}

template <>
Vec256<float> inline operator^(const Vec256<float>& a, const Vec256<float>& b) {
  return _mm256_xor_ps(a, b);
}

template <>
void convert(const float* src, float* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vec256<float>::size()); i += Vec256<float>::size()) {
    _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

#ifdef __AVX2__
template <>
Vec256<float> inline fmadd(const Vec256<float>& a, const Vec256<float>& b, const Vec256<float>& c) {
  return _mm256_fmadd_ps(a, b, c);
}
#endif

#endif

}}}
