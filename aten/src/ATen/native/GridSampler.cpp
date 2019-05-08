#include <ATen/native/GridSampler.h>
#include <ATen/ATen.h>
#include <ATen/Device.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/Layout.h>
#include <ATen/cpu/vml.h>
#include <ATen/native/cpu/GridSamplerKernel.h>
#include <c10/util/Exception.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at { namespace native {

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

namespace {

  template<typename scalar_t>
  static inline scalar_t clip_coordinates(scalar_t in, int64_t clip_limit) {
    return std::min(static_cast<scalar_t>(clip_limit - 1), std::max(in, static_cast<scalar_t>(0)));
  }

  // clip_coordinates_set_grad works similarly to clip_coordinates except that
  // it also returns the `d output / d input` via pointer argument `grad_in`.
  // This is useful in the backward pass of grid_sampler.
  template<typename scalar_t>
  static inline scalar_t clip_coordinates_set_grad(scalar_t in, int64_t clip_limit,
                                                   scalar_t *grad_in) {
    if (in < static_cast<scalar_t>(0)) {
      *grad_in = static_cast<scalar_t>(0);
      return static_cast<scalar_t>(0);
    } else {
      scalar_t max = static_cast<scalar_t>(clip_limit - 1);
      if (in > max) {
        *grad_in = static_cast<scalar_t>(0);
        return max;
      } else {
        *grad_in = static_cast<scalar_t>(1);
        return in;
      }
    }
  }

  template<typename scalar_t>
  static inline scalar_t reflect_coordinates(scalar_t in, int64_t clip_limit) {
    if (clip_limit == static_cast<int64_t>(1)) {
      return static_cast<scalar_t>(0);
    }
    in = std::fabs(in);
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    scalar_t extra = std::fmod(in, max);
    int flips = static_cast<int>(std::floor(in / max));
    if (flips % 2 == 0) {
      return extra;
    } else {
      return max - extra;
    }
  }

  // reflect_coordinates_set_grad works similarly to reflect_coordinates except
  // that it also returns the `d output / d input` via pointer argument
  // `grad_in`.
  // This is useful in the backward pass of grid_sampler.
  template<typename scalar_t>
  static inline scalar_t reflect_coordinates_set_grad(scalar_t in, int64_t clip_limit,
                                                      scalar_t *grad_in) {
    if (clip_limit == static_cast<int64_t>(1)) {
      *grad_in = static_cast<scalar_t>(0);
      return static_cast<scalar_t>(0);
    }
    int grad_in_mult_;
    if (in < static_cast<scalar_t>(0)) {
      grad_in_mult_ = -1;
      in = -in;
    } else {
      grad_in_mult_ = 1;
    }
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    // `fmod` returns same sign as `in`, which is positive after the `if` above.
    scalar_t extra = std::fmod(in, max);
    int flips = static_cast<int>(std::floor(in / max));
    if (flips % 2 == 0) {
      *grad_in = static_cast<scalar_t>(grad_in_mult_);
      return extra;
    } else {
      *grad_in = static_cast<scalar_t>(-grad_in_mult_);
      return max - extra;
    }
  }

  static inline bool within_bounds_2d(int64_t h, int64_t w, int64_t H, int64_t W) {
    return h >= 0 && h < H && w >= 0 && w < W;
  }

  static inline bool within_bounds_3d(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) {
    return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
  }

  template<typename scalar_t>
  static inline void safe_add_2d(scalar_t *data, int64_t h, int64_t w,
                                 int64_t sH, int64_t sW, int64_t H, int64_t W,
                                 scalar_t delta) {
    if (within_bounds_2d(h, w, H, W)) {
      data[h * sH + w * sW] += delta;
    }
  }

  template<typename scalar_t>
  static inline void safe_add_3d(scalar_t *data, int64_t d, int64_t h, int64_t w,
                                 int64_t sD, int64_t sH, int64_t sW,
                                 int64_t D, int64_t H, int64_t W,
                                 scalar_t delta) {
    if (within_bounds_3d(d, h, w, D, H, W)) {
      data[d * sD + h * sH + w * sW] += delta;
    }
  }

  template<typename scalar_t>
  Tensor grid_sampler_3d_cpu_impl(const Tensor& input, const Tensor& grid,
                                  GridSamplerInterpolation interpolation_mode,
                                  GridSamplerPadding padding_mode) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_D = input.size(2);
    int64_t inp_H = input.size(3);
    int64_t inp_W = input.size(4);
    int64_t out_D = grid.size(1);
    int64_t out_H = grid.size(2);
    int64_t out_W = grid.size(3);
    auto output = at::empty({N, C, out_D, out_H, out_W}, input.options());
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sD = input.stride(2);
    int64_t inp_sH = input.stride(3);
    int64_t inp_sW = input.stride(4);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sD = grid.stride(1);
    int64_t grid_sH = grid.stride(2);
    int64_t grid_sW = grid.stride(3);
    int64_t grid_sCoor = grid.stride(4);
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sD = output.stride(2);
    int64_t out_sH = output.stride(3);
    int64_t out_sW = output.stride(4);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *out_ptr = output.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
      scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
      scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
      for (int64_t d = 0; d < out_D; ++d) {
        for (int64_t h = 0; h < out_H; ++h) {
          for (int64_t w = 0; w < out_W; ++w) {
            // get the corresponding input x, y, z co-ordinates from grid
            scalar_t *grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
            scalar_t ix = *grid_ptr_NDHW;
            scalar_t iy = grid_ptr_NDHW[grid_sCoor];
            scalar_t iz = grid_ptr_NDHW[2 * grid_sCoor];

            // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
            ix = ((ix + 1) / 2) * (inp_W - 1);
            iy = ((iy + 1) / 2) * (inp_H - 1);
            iz = ((iz + 1) / 2) * (inp_D - 1);

            if (padding_mode == GridSamplerPadding::Border) {
              // clip coordinates to image borders
              ix = clip_coordinates(ix, inp_W);
              iy = clip_coordinates(iy, inp_H);
              iz = clip_coordinates(iz, inp_D);
            } else if (padding_mode == GridSamplerPadding::Reflection) {
              // reflect coordinates by image borders
              ix = reflect_coordinates(ix, inp_W);
              iy = reflect_coordinates(iy, inp_H);
              iz = reflect_coordinates(iz, inp_D);
            }

            if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
              // get corner pixel values from (x, y, z)
              // for 4d, we used north-east-south-west
              // for 5d, we add top-bottom
              int64_t ix_tnw = static_cast<int64_t>(std::floor(ix));
              int64_t iy_tnw = static_cast<int64_t>(std::floor(iy));
              int64_t iz_tnw = static_cast<int64_t>(std::floor(iz));

              int64_t ix_tne = ix_tnw + 1;
              int64_t iy_tne = iy_tnw;
              int64_t iz_tne = iz_tnw;

              int64_t ix_tsw = ix_tnw;
              int64_t iy_tsw = iy_tnw + 1;
              int64_t iz_tsw = iz_tnw;

              int64_t ix_tse = ix_tnw + 1;
              int64_t iy_tse = iy_tnw + 1;
              int64_t iz_tse = iz_tnw;

              int64_t ix_bnw = ix_tnw;
              int64_t iy_bnw = iy_tnw;
              int64_t iz_bnw = iz_tnw + 1;

              int64_t ix_bne = ix_tnw + 1;
              int64_t iy_bne = iy_tnw;
              int64_t iz_bne = iz_tnw + 1;

              int64_t ix_bsw = ix_tnw;
              int64_t iy_bsw = iy_tnw + 1;
              int64_t iz_bsw = iz_tnw + 1;

              int64_t ix_bse = ix_tnw + 1;
              int64_t iy_bse = iy_tnw + 1;
              int64_t iz_bse = iz_tnw + 1;

              // get surfaces to each neighbor:
              scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
              scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
              scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
              scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
              scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
              scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
              scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
              scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

              // calculate bilinear weighted pixel value and set output pixel
              scalar_t *out_ptr_NCDHW = out_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
              scalar_t *inp_ptr_NC = inp_ptr_N;
              for (int c = 0; c < C; ++c, out_ptr_NCDHW += out_sC, inp_ptr_NC += inp_sC) {
                //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
                // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
                // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
                // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
                *out_ptr_NCDHW = static_cast<scalar_t>(0);
                if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                  *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
                }
                if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                  *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
                }
                if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                  *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
                }
                if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                  *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
                }
                if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                  *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
                }
                if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                  *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
                }
                if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                  *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
                }
                if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                  *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
                }
              }
            } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
              int64_t ix_nearest = static_cast<int64_t>(std::round(ix));
              int64_t iy_nearest = static_cast<int64_t>(std::round(iy));
              int64_t iz_nearest = static_cast<int64_t>(std::round(iz));

              // assign nearest neighor pixel value to output pixel
              scalar_t *out_ptr_NCDHW = out_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
              scalar_t *inp_ptr_NC = inp_ptr_N;
              for (int c = 0; c < C; ++c, out_ptr_NCDHW += out_sC, inp_ptr_NC += inp_sC) {
                if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
                  *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
                } else {
                  *out_ptr_NCDHW = static_cast<scalar_t>(0);
                }
              }
            }
          }
        }
      }
    }
    return output;
  }

  template<typename scalar_t>
  std::tuple<Tensor, Tensor>
  grid_sampler_3d_backward_cpu_impl(const Tensor& grad_output,
                                    const Tensor& input, const Tensor& grid,
                                    GridSamplerInterpolation interpolation_mode,
                                    GridSamplerPadding padding_mode) {
    auto grad_input = at::zeros_like(input);
    auto grad_grid = at::empty_like(grid);
    // If interpolation mode is Nearest, then grad_grid is not filled in the
    // loop below.
    if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      grad_grid.zero_();
    }
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_D = input.size(2);
    int64_t inp_H = input.size(3);
    int64_t inp_W = input.size(4);
    int64_t out_D = grid.size(1);
    int64_t out_H = grid.size(2);
    int64_t out_W = grid.size(3);
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sD = input.stride(2);
    int64_t inp_sH = input.stride(3);
    int64_t inp_sW = input.stride(4);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sD = grid.stride(1);
    int64_t grid_sH = grid.stride(2);
    int64_t grid_sW = grid.stride(3);
    int64_t grid_sCoor = grid.stride(4);
    int64_t gOut_sN = grad_output.stride(0);
    int64_t gOut_sC = grad_output.stride(1);
    int64_t gOut_sD = grad_output.stride(2);
    int64_t gOut_sH = grad_output.stride(3);
    int64_t gOut_sW = grad_output.stride(4);
    int64_t gInp_sN = grad_input.stride(0);
    int64_t gInp_sC = grad_input.stride(1);
    int64_t gInp_sD = grad_input.stride(2);
    int64_t gInp_sH = grad_input.stride(3);
    int64_t gInp_sW = grad_input.stride(4);
    int64_t gGrid_sN = grad_grid.stride(0);
    int64_t gGrid_sW = grad_grid.stride(3);
    scalar_t *inp_ptr = input.data<scalar_t>();
    scalar_t *grid_ptr = grid.data<scalar_t>();
    scalar_t *gOut_ptr = grad_output.data<scalar_t>();
    scalar_t *gInp_ptr = grad_input.data<scalar_t>();
    scalar_t *gGrid_ptr = grad_grid.data<scalar_t>();
    // loop over each output pixel
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t n = 0; n < N; ++n) {
      scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
      scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
      scalar_t *gGrid_ptr_NDHW = gGrid_ptr + n * gGrid_sN;
      for (int64_t d = 0; d < out_D; ++d) {
        for (int64_t h = 0; h < out_H; ++h) {
          for (int64_t w = 0; w < out_W; ++w, gGrid_ptr_NDHW += gGrid_sW /* grad_grid is contiguous */ ) {
            // get the corresponding input x, y, z co-ordinates from grid
            scalar_t *grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
            scalar_t ix = *grid_ptr_NDHW;
            scalar_t iy = grid_ptr_NDHW[grid_sCoor];
            scalar_t iz = grid_ptr_NDHW[2 * grid_sCoor];

            // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
            ix = ((ix + 1) / 2) * (inp_W - 1);
            iy = ((iy + 1) / 2) * (inp_H - 1);
            iz = ((iz + 1) / 2) * (inp_D - 1);

            // multipliers for gradients on ix, iy, and iz
            // E.g.,  0 for out-of-bound indices when GridSamplerPadding::Border
            scalar_t gix_mult, giy_mult, giz_mult;
            if (padding_mode == GridSamplerPadding::Border) {
              // clip coordinates to image borders
              ix = clip_coordinates_set_grad(ix, inp_W, &gix_mult);
              iy = clip_coordinates_set_grad(iy, inp_H, &giy_mult);
              iz = clip_coordinates_set_grad(iz, inp_D, &giz_mult);
            } else if (padding_mode == GridSamplerPadding::Reflection) {
              // reflect coordinates by image borders
              ix = reflect_coordinates_set_grad(ix, inp_W, &gix_mult);
              iy = reflect_coordinates_set_grad(iy, inp_H, &giy_mult);
              iz = reflect_coordinates_set_grad(iz, inp_D, &giz_mult);
            } else {  // padding_mode == GridSamplerPadding::Zeros
              gix_mult = static_cast<scalar_t>(1);
              giy_mult = static_cast<scalar_t>(1);
              giz_mult = static_cast<scalar_t>(1);
            }

            if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
              // get corner pixel values from (x, y, z)
              // for 4d, we used north-east-south-west
              // for 5d, we add top-bottom
              int64_t ix_tnw = static_cast<int64_t>(std::floor(ix));
              int64_t iy_tnw = static_cast<int64_t>(std::floor(iy));
              int64_t iz_tnw = static_cast<int64_t>(std::floor(iz));

              int64_t ix_tne = ix_tnw + 1;
              int64_t iy_tne = iy_tnw;
              int64_t iz_tne = iz_tnw;

              int64_t ix_tsw = ix_tnw;
              int64_t iy_tsw = iy_tnw + 1;
              int64_t iz_tsw = iz_tnw;

              int64_t ix_tse = ix_tnw + 1;
              int64_t iy_tse = iy_tnw + 1;
              int64_t iz_tse = iz_tnw;

              int64_t ix_bnw = ix_tnw;
              int64_t iy_bnw = iy_tnw;
              int64_t iz_bnw = iz_tnw + 1;

              int64_t ix_bne = ix_tnw + 1;
              int64_t iy_bne = iy_tnw;
              int64_t iz_bne = iz_tnw + 1;

              int64_t ix_bsw = ix_tnw;
              int64_t iy_bsw = iy_tnw + 1;
              int64_t iz_bsw = iz_tnw + 1;

              int64_t ix_bse = ix_tnw + 1;
              int64_t iy_bse = iy_tnw + 1;
              int64_t iz_bse = iz_tnw + 1;

              // get surfaces to each neighbor:
              scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
              scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
              scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
              scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
              scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
              scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
              scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
              scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

              scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
              scalar_t *gOut_ptr_NCDHW = gOut_ptr + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
              scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
              scalar_t *inp_ptr_NC = inp_ptr_N;
              // calculate bilinear weighted pixel value and set output pixel
              for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
                scalar_t gOut = *gOut_ptr_NCDHW;

                // calculate and set grad_input
                safe_add_3d(gInp_ptr_NC, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut);
                safe_add_3d(gInp_ptr_NC, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut);
                safe_add_3d(gInp_ptr_NC, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut);
                safe_add_3d(gInp_ptr_NC, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut);
                safe_add_3d(gInp_ptr_NC, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut);
                safe_add_3d(gInp_ptr_NC, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut);
                safe_add_3d(gInp_ptr_NC, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut);
                safe_add_3d(gInp_ptr_NC, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut);

                // calculate grad_grid
                if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                  scalar_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
                  gix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * gOut;
                  giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * gOut;
                  giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gOut;
                }
                if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                  scalar_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
                  gix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * gOut;
                  giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz)    * gOut;
                  giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gOut;
                }
                if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                  scalar_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
                  gix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * gOut;
                  giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * gOut;
                  giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gOut;
                }
                if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                  scalar_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
                  gix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * gOut;
                  giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz)    * gOut;
                  giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gOut;
                }
                if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                  scalar_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
                  gix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * gOut;
                  giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * gOut;
                  giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gOut;
                }
                if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                  scalar_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
                  gix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * gOut;
                  giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw)    * gOut;
                  giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gOut;
                }
                if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                  scalar_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
                  gix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * gOut;
                  giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * gOut;
                  giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gOut;
                }
                if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                  scalar_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
                  gix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * gOut;
                  giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw)    * gOut;
                  giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gOut;
                }
              }

              // un-normalize grad_grid values back to [-1, 1] constraints
              gix = gix * (inp_W - 1) / 2;
              giy = giy * (inp_H - 1) / 2;
              giz = giz * (inp_D - 1) / 2;

              // assuming grad_grid is contiguous
              gGrid_ptr_NDHW[0] = gix_mult * gix;
              gGrid_ptr_NDHW[1] = giy_mult * giy;
              gGrid_ptr_NDHW[2] = giz_mult * giz;
            } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
              int64_t ix_nearest = static_cast<int64_t>(std::round(ix));
              int64_t iy_nearest = static_cast<int64_t>(std::round(iy));
              int64_t iz_nearest = static_cast<int64_t>(std::round(iz));

              // assign nearest neighor pixel value to output pixel
              scalar_t *gOut_ptr_NCDHW = gOut_ptr + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
              scalar_t *gInp_ptr_NC = gInp_ptr + n * gInp_sN;
              for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
                // calculate and set grad_input
                safe_add_3d(gInp_ptr_NC, iz_nearest, iy_nearest, ix_nearest,
                            gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW);
              }
            }
          }
        }
      }
    }
    return std::make_tuple(grad_input, grad_grid);
  }

}  // namespace

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_2d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode) {
  return grid_sampler_2d_cpu_kernel(kCPU, input, grid, interpolation_mode, padding_mode);
}

DEFINE_DISPATCH(grid_sampler_2d_cpu_kernel);


// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_3d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode) {
  return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler3d_cpu", [&] {
    return grid_sampler_3d_cpu_impl<scalar_t>(
      input, grid, static_cast<GridSamplerInterpolation>(interpolation_mode),
      static_cast<GridSamplerPadding>(padding_mode));
  });
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode) {
  return grid_sampler_2d_backward_cpu_kernel(kCPU, grad_output, input, grid, interpolation_mode, padding_mode);
}

DEFINE_DISPATCH(grid_sampler_2d_backward_cpu_kernel);

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                             int64_t interpolation_mode, int64_t padding_mode) {
  return AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler_3d_backward_cpu", [&] {
    return grid_sampler_3d_backward_cpu_impl<scalar_t>(
      grad_output, input, grid,
      static_cast<GridSamplerInterpolation>(interpolation_mode),
      static_cast<GridSamplerPadding>(padding_mode));
  });
}

Tensor grid_sampler(const Tensor& input, const Tensor& grid,
                    int64_t interpolation_mode, int64_t padding_mode) {
  AT_CHECK(
    input.defined() && grid.defined(),
    "grid_sampler(): expected input and grid to not be undefined, but input "
    "is ", input, " and grid is ", grid);
  auto input_opt = input.options();
  auto grid_opt = grid.options();
  AT_CHECK(
    input_opt.device() == grid_opt.device(),
    "grid_sampler(): expected input and grid to be on same device, but input "
    "is on ", input_opt.device(), " and grid is on ", grid_opt.device());
  AT_CHECK(
    input_opt.dtype() == grid_opt.dtype(),
    "grid_sampler(): expected input and grid to have same dtype, but input "
    "has ", input_opt.dtype(), " and grid has ", grid_opt.dtype());
  AT_CHECK(
    input_opt.layout() == kStrided && grid_opt.layout() == kStrided,
    "grid_sampler(): expected input and grid to have torch.strided layout, but "
    "input has ", input_opt.layout(), " and grid has ", grid_opt.layout());
  AT_CHECK(
    (input.dim() == 4 || input.dim() == 5) && input.dim() == grid.dim(),
    "grid_sampler(): expected 4D or 5D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
  AT_CHECK(
    input.size(0) == grid.size(0),
    "grid_sampler(): expected grid and input to have same batch size, but got "
    "input with sizes ", input.sizes(), " and grid with sizes ", grid.sizes());
  AT_CHECK(
    grid.size(-1) == input.dim() - 2,
    "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
    "dimension, but got grid with sizes ", grid.sizes());
  for (int64_t i = 2; i < input.dim(); i++) {
    AT_CHECK(input.size(i) > 0,
      "grid_sampler(): expected input to have non-empty spatial dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being "
      "empty");
  }
  // cudnn does not support inputs larger than 1024
  if (at::native::cudnn_is_acceptable(input) &&
      at::native::cudnn_is_acceptable(grid) &&
      static_cast<GridSamplerInterpolation>(interpolation_mode) == GridSamplerInterpolation::Bilinear &&
      static_cast<GridSamplerPadding>(padding_mode) == GridSamplerPadding::Zeros &&
      input.dim() == 4 &&
      input.size(1) <= 1024) {
    return cudnn_grid_sampler(input, grid);
  }
  if (input.dim() == 4) {
    return at::grid_sampler_2d(input, grid, interpolation_mode, padding_mode);
  } else {
    return at::grid_sampler_3d(input, grid, interpolation_mode, padding_mode);
  }
}

}}  // namespace at::native
