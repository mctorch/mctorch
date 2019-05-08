#!/bin/bash

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export PATH="/usr/local/bin:$PATH"

# Set up conda environment
export PYTORCH_ENV_DIR="${HOME}/pytorch-ci-env"
# If a local installation of conda doesn't exist, we download and install conda
if [ ! -d "${PYTORCH_ENV_DIR}/miniconda3" ]; then
  mkdir -p ${PYTORCH_ENV_DIR}
  curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ${PYTORCH_ENV_DIR}/miniconda3.sh
  bash ${PYTORCH_ENV_DIR}/miniconda3.sh -b -p ${PYTORCH_ENV_DIR}/miniconda3
fi
export PATH="${PYTORCH_ENV_DIR}/miniconda3/bin:$PATH"
source ${PYTORCH_ENV_DIR}/miniconda3/bin/activate
conda install -y mkl mkl-include numpy pyyaml setuptools cmake cffi ninja six
pip install -q hypothesis "librosa>=0.6.2" psutil

# faulthandler become built-in since 3.3
if [[ ! $(python -c "import sys; print(int(sys.version_info >= (3, 3)))") == "1" ]]; then
  pip install -q faulthandler
fi

if [ -z "${IN_CIRCLECI}" ]; then
  rm -rf ${PYTORCH_ENV_DIR}/miniconda3/lib/python3.6/site-packages/torch*
fi

git submodule sync --recursive
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${PYTORCH_ENV_DIR}/miniconda3/

# Test PyTorch
if [ -z "${IN_CIRCLECI}" ]; then
  if [[ "${BUILD_ENVIRONMENT}" == *cuda9.2* ]]; then
    # Eigen gives "explicit specialization of class must precede its first use" error
    # when compiling with Xcode 9.1 toolchain, so we have to use Xcode 8.2 toolchain instead.
    export DEVELOPER_DIR=/Library/Developer/CommandLineTools
  else
    export DEVELOPER_DIR=/Applications/Xcode9.app/Contents/Developer
  fi
fi
export MACOSX_DEPLOYMENT_TARGET=10.9
export CXX=clang++
export CC=clang
# If we run too many parallel jobs, we will OOM
export MAX_JOBS=2

export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}

# Download torch binaries in the test jobs
if [ -z "${IN_CIRCLECI}" ]; then
  rm -rf ${PYTORCH_ENV_DIR}/miniconda3/lib/python3.6/site-packages/torch*
  aws s3 cp s3://ossci-macos-build/pytorch/${IMAGE_COMMIT_TAG}.7z ${IMAGE_COMMIT_TAG}.7z
  7z x ${IMAGE_COMMIT_TAG}.7z -o"${PYTORCH_ENV_DIR}/miniconda3/lib/python3.6/site-packages"
fi

# Test that OpenMP is enabled
pushd test
if [[ ! $(python -c "import torch; print(int(torch.backends.openmp.is_available()))") == "1" ]]; then
  echo "Build should have OpenMP enabled, but torch.backends.openmp.is_available() is False"
  exit 1
fi
popd

test_python_all() {
  echo "Ninja version: $(ninja --version)"
  python test/run_test.py --verbose
  assert_git_not_dirty
}

test_libtorch() {
  # C++ API

  if [[ "$BUILD_TEST_LIBTORCH" == "1" ]]; then
    # NB: Install outside of source directory (at the same level as the root
    # pytorch folder) so that it doesn't get cleaned away prior to docker push.
    # But still clean it before we perform our own build.

    echo "Testing libtorch"

    CPP_BUILD="$PWD/../cpp-build"
    rm -rf $CPP_BUILD
    mkdir -p $CPP_BUILD/caffe2

    BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
    pushd $CPP_BUILD/caffe2
    VERBOSE=1 DEBUG=1 python $BUILD_LIBTORCH_PY
    popd

    python tools/download_mnist.py --quiet -d test/cpp/api/mnist

    # Unfortunately it seems like the test can't load from miniconda3
    # without these paths being set
    export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$PWD/miniconda3/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/miniconda3/lib"
    TORCH_CPP_TEST_MNIST_PATH="test/cpp/api/mnist" "$CPP_BUILD"/caffe2/bin/test_api

    assert_git_not_dirty
  fi
}

test_custom_script_ops() {
  echo "Testing custom script operators"
  pushd test/custom_operator
  # Build the custom operator library.
  rm -rf build && mkdir build
  pushd build
  SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" cmake ..
  make VERBOSE=1
  popd

  # Run tests Python-side and export a script module.
  python test_custom_ops.py -v
  python model.py --export-script-module=model.pt
  # Run tests C++-side and load the exported script module.
  build/test_custom_ops ./model.pt
  popd
  assert_git_not_dirty
}


if [ -z "${BUILD_ENVIRONMENT}" ] || [[ "${BUILD_ENVIRONMENT}" == *-test ]]; then
  test_python_all
  test_libtorch
  test_custom_script_ops
else
  if [[ "${BUILD_ENVIRONMENT}" == *-test1 ]]; then
    test_python_all
  elif [[ "${BUILD_ENVIRONMENT}" == *-test2 ]]; then
    test_libtorch
    test_custom_script_ops
  fi
fi
