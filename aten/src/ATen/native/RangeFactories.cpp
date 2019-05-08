#include <ATen/NativeFunctions.h>
#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>
#include <ATen/Dispatch.h>
#include <cmath>
#include <limits>

namespace at { namespace native {


Tensor& linspace_cpu_out(Tensor& result, Scalar start, Scalar end, int64_t steps) {
  AT_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else {
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "linspace_cpu", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t *data_ptr = r.data<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
        scalar_t is = static_cast<scalar_t>(p_begin);
        for (int64_t i = p_begin; i < p_end; ++i, ++is) {
          data_ptr[i] = scalar_start + step*is;
        }
      });
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  return result;
}

Tensor& logspace_cpu_out(Tensor& result, Scalar start, Scalar end, int64_t steps, double base) {
  AT_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(std::pow(base, start.to<double>()));
  } else {
    AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "logspace_cpu", [&]() {
      scalar_t scalar_base = static_cast<scalar_t>(base);
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t *data_ptr = r.data<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
        scalar_t is = static_cast<scalar_t>(p_begin);
        for (int64_t i = p_begin; i < p_end; ++i, ++is) {
          data_ptr[i]= std::pow(scalar_base, scalar_start + step*is);
        }
      });
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  return result;
}

Tensor& range_cpu_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  AT_DISPATCH_ALL_TYPES(result.scalar_type(), "range_cpu", [&]() {
    using accscalar_t = at::acc_type<scalar_t, false>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    AT_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    AT_CHECK(std::isfinite(static_cast<double>(xstart)) &&
             std::isfinite(static_cast<double>(xend)),
             "unsupported range: ", xstart, " -> ", xend);
    AT_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
             "upper bound and larger bound inconsistent with step sign");
    int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);
    if (result.numel() != size) {
      result.resize_({size});
    }
    Tensor r = result.is_contiguous() ? result : result.contiguous();
    scalar_t *data_ptr = r.data<scalar_t>();

    at::parallel_for(0, size, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      scalar_t is = p_begin;
      for (int64_t i = p_begin; i < p_end; ++i, ++is) {
        data_ptr[i] = xstart + is * xstep;
      }
    });
    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  });

  return result;
}

Tensor& arange_cpu_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  AT_DISPATCH_ALL_TYPES(result.scalar_type(), "arange_cpu", [&]() {
    using accscalar_t = at::acc_type<scalar_t, false>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    // we use double precision for (start - end) / step
    // to compute size_d for consistency across devices.
    // The problem with using accscalar_t is that accscalar_t might be float32 on gpu for a float32 scalar_t,
    // but double on cpu for the same,
    // and the effective output size starts differing on CPU vs GPU because of precision issues, which
    // we dont want.
    // the corner-case we do want to take into account is int64_t, which has higher precision than double
    double size_d;
    if (std::is_same<scalar_t, int64_t>::value) {
      size_d = std::ceil(static_cast<double>(end.to<accscalar_t>() - start.to<accscalar_t>())
                         / step.to<accscalar_t>());
    } else {
      size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>())
                         / step.to<double>());
    }

    AT_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    AT_CHECK(std::isfinite(static_cast<double>(xstart)) &&
             std::isfinite(static_cast<double>(xend)),
             "unsupported range: ", xstart, " -> ", xend);
    AT_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
             "upper bound and larger bound inconsistent with step sign");

    AT_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
             "invalid size, possible overflow?");
    int64_t size = static_cast<int64_t>(size_d);

    if (result.numel() != size) {
      result.resize_({size});
    }
    Tensor r = result.is_contiguous() ? result : result.contiguous();
    scalar_t *data_ptr = r.data<scalar_t>();

    at::parallel_for(0, size, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      scalar_t is = p_begin;
      for (int64_t i = p_begin; i < p_end; ++i, ++is) {
        data_ptr[i] = xstart + is * xstep;
      }
    });
    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  });

  return result;
}

}} // namespace at::native
