from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from operator_benchmark import benchmark_core, benchmark_utils

import torch

"""PyTorch performance microbenchmarks.

This module contains PyTorch-specific functionalities for performance
microbenchmarks.
"""


def PyTorchOperatorTestCase(test_name, op_type, input_shapes, op_args, run_mode):
    """Benchmark Tester function for Pytorch framework.
    """
    inputs = [torch.from_numpy(benchmark_utils.numpy_random_fp32(*input)) for input in input_shapes]

    def benchmark_func(num_runs):
        op_type(*(inputs + [num_runs]))

    benchmark_core.add_benchmark_tester("PyTorch", test_name, input_shapes, op_args, run_mode, benchmark_func)
