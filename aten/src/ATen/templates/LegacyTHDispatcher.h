#pragma once

// ${generated_comment}

#include <c10/core/TensorTypeIdRegistration.h>

namespace at {

struct CAFFE2_API LegacyTHDispatcher {
  explicit LegacyTHDispatcher(TensorTypeId type_id)
      : type_id_(type_id) {}

  virtual ~LegacyTHDispatcher() {}

protected:
  TensorTypeId type_id_;
};

} // namespace th

