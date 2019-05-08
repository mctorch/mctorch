import argparse
from os.path import dirname, abspath
import sys

# By appending pytorch_root to sys.path, this module can import other torch
# modules even when run as a standalone script. i.e., it's okay either you
# do `python build_libtorch.py` or `python -m tools.build_libtorch`.
pytorch_root = dirname(dirname(abspath(__file__)))
sys.path.append(pytorch_root)

# If you want to modify flags or environmental variables that is set when
# building torch, you should do it in tools/setup_helpers/configure.py.
# Please don't add it here unless it's only used in LibTorch.
from tools.build_pytorch_libs import build_caffe2

if __name__ == '__main__':
    # Placeholder for future interface. For now just gives a nice -h.
    parser = argparse.ArgumentParser(description='Build libtorch')
    options = parser.parse_args()

    build_caffe2(version=None, cmake_python_library=None,
                 build_python=False, rerun_cmake=True, build_dir='.')
