#!/usr/bin/env python

from __future__ import print_function

import subprocess


TESTS = [
    (
        "Checking that caffe2.python is available",
        "from caffe2.python import core",
    ),
    (
        "Checking that MKL is available",
        "import torch; exit(0 if torch.backends.mkl.is_available() else 1)",
    ),
    (
        "Checking that CUDA archs are setup correctly",
        "import torch; torch.randn([3,5]).cuda()",
    ),
    (
        "Checking that magma is available",
        "import torch; torch.rand(1).cuda(); exit(0 if torch.cuda.has_magma else 1)",
    ),
    (
        "Checking that CuDNN is available",
        "import torch; exit(0 if torch.backends.cudnn.is_available() else 1)",
    ),
]


if __name__ == "__main__":

    for description, python_commands in TESTS:
        print(description)
        command_args = ["python", "-c", python_commands]
        command_string = " ".join(command_args)
        print("Command:", command_string)
        subprocess.check_call(command_args)
