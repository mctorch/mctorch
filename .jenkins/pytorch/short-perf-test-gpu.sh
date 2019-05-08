#!/bin/bash

# shellcheck disable=SC2034
COMPACT_JOB_NAME="short-perf-test-gpu"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

pushd .jenkins/pytorch/perf_test

echo "Running GPU perf test for PyTorch..."

# Trying to uninstall PyYAML can cause problem. Workaround according to:
# https://github.com/pypa/pip/issues/5247#issuecomment-415571153
pip install -q awscli --ignore-installed PyYAML

# Set multipart_threshold to be sufficiently high, so that `aws s3 cp` is not a multipart read
# More info at https://github.com/aws/aws-cli/issues/2321
aws configure set default.s3.multipart_threshold 5GB

if [[ "$COMMIT_SOURCE" == master ]]; then
    # Get current master commit hash
    export MASTER_COMMIT_ID=$(git log --format="%H" -n 1)
fi

# Find the master commit to test against
git remote add upstream https://github.com/pytorch/pytorch.git
git fetch upstream
IFS=$'\n'
master_commit_ids=($(git rev-list upstream/master))
for commit_id in "${master_commit_ids[@]}"; do
    if aws s3 ls s3://ossci-perf-test/pytorch/gpu_runtime/${commit_id}.json; then
        LATEST_TESTED_COMMIT=${commit_id}
        break
    fi
done
aws s3 cp s3://ossci-perf-test/pytorch/gpu_runtime/${LATEST_TESTED_COMMIT}.json gpu_runtime.json

if [[ "$COMMIT_SOURCE" == master ]]; then
    # Prepare new baseline file
    cp gpu_runtime.json new_gpu_runtime.json
    python update_commit_hash.py new_gpu_runtime.json ${MASTER_COMMIT_ID}
fi

# Include tests
. ./test_gpu_speed_mnist.sh
. ./test_gpu_speed_word_language_model.sh
. ./test_gpu_speed_cudnn_lstm.sh
. ./test_gpu_speed_lstm.sh
. ./test_gpu_speed_mlstm.sh

# Run tests
if [[ "$COMMIT_SOURCE" == master ]]; then
    run_test test_gpu_speed_mnist 20 compare_and_update
    run_test test_gpu_speed_word_language_model 20 compare_and_update
    run_test test_gpu_speed_cudnn_lstm 20 compare_and_update
    run_test test_gpu_speed_lstm 20 compare_and_update
    run_test test_gpu_speed_mlstm 20 compare_and_update
else
    run_test test_gpu_speed_mnist 20 compare_with_baseline
    run_test test_gpu_speed_word_language_model 20 compare_with_baseline
    run_test test_gpu_speed_cudnn_lstm 20 compare_with_baseline
    run_test test_gpu_speed_lstm 20 compare_with_baseline
    run_test test_gpu_speed_mlstm 20 compare_with_baseline
fi

if [[ "$COMMIT_SOURCE" == master ]]; then
    # This could cause race condition if we are testing the same master commit twice,
    # but the chance of them executing this line at the same time is low.
    aws s3 cp new_gpu_runtime.json s3://ossci-perf-test/pytorch/gpu_runtime/${MASTER_COMMIT_ID}.json --acl public-read
fi

popd
