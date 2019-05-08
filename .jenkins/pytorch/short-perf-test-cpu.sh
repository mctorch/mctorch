#!/bin/bash

# shellcheck disable=SC2034
COMPACT_JOB_NAME="short-perf-test-cpu"

SCRIPT_PARENT_DIR=$(dirname "${BASH_SOURCE[0]}")

# shellcheck source=.jenkins/pytorch/common.sh
source "$SCRIPT_PARENT_DIR/common.sh"

cd .jenkins/pytorch/perf_test

echo "Running CPU perf test for PyTorch..."

pip install -q awscli

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
    if aws s3 ls s3://ossci-perf-test/pytorch/cpu_runtime/${commit_id}.json; then
        LATEST_TESTED_COMMIT=${commit_id}
        break
    fi
done
aws s3 cp s3://ossci-perf-test/pytorch/cpu_runtime/${LATEST_TESTED_COMMIT}.json cpu_runtime.json

if [[ "$COMMIT_SOURCE" == master ]]; then
    # Prepare new baseline file
    cp cpu_runtime.json new_cpu_runtime.json
    python update_commit_hash.py new_cpu_runtime.json ${MASTER_COMMIT_ID}
fi

# Include tests
. ./test_cpu_speed_mini_sequence_labeler.sh
. ./test_cpu_speed_mnist.sh
. ./test_cpu_speed_torch.sh
. ./test_cpu_speed_torch_tensor.sh

# Run tests
export TEST_MODE="compare_with_baseline"
if [[ "$COMMIT_SOURCE" == master ]]; then
    export TEST_MODE="compare_and_update"
fi

# Operator tests
run_test test_cpu_speed_torch ${TEST_MODE}
run_test test_cpu_speed_torch_tensor ${TEST_MODE}

# Sample model tests
run_test test_cpu_speed_mini_sequence_labeler 20 ${TEST_MODE}
run_test test_cpu_speed_mnist 20 ${TEST_MODE}

if [[ "$COMMIT_SOURCE" == master ]]; then
    # This could cause race condition if we are testing the same master commit twice,
    # but the chance of them executing this line at the same time is low.
    aws s3 cp new_cpu_runtime.json s3://ossci-perf-test/pytorch/cpu_runtime/${MASTER_COMMIT_ID}.json --acl public-read
fi
