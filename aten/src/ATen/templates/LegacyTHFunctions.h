#pragma once

// ${generated_comment}

namespace at {
namespace legacy {
namespace th {

namespace detail {

static inline LegacyTHDispatcher & infer_dispatcher(const Tensor & t) {
  AT_CHECK(t.defined(), "undefined Tensor");
  return getLegacyTHDispatcher(t);
}
static inline LegacyTHDispatcher & infer_dispatcher(const TensorList & tl) {
  AT_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
  return getLegacyTHDispatcher(tl[0]);
}

} // namespace detail

// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument

}
}
}

// FIXME: this is temporary until we start generating into at::legacy::th

#include <ATen/Functions.h>

namespace at {
namespace legacy {
namespace th {
  using namespace at;
}
}
}
