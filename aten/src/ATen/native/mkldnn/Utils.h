#pragma once

#include <c10/util/ArrayRef.h>
#include <vector>

namespace at { namespace native {

std::vector<int64_t> conv_output_size(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation);

std::vector<int64_t> pool_output_sizes(
    IntArrayRef input_size,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);
}}
