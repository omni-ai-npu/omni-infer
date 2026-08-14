#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Script to run tests for omni-npu

set -e

# NPU tests use HCCL for distributed communication. Prevent vLLM's CUDA
# communicator fallback from initializing PyNccl/NCCL in spawned workers.
export VLLM_DISABLE_PYNCCL="${VLLM_DISABLE_PYNCCL:-1}"

# Parse command line arguments
TEST_TYPE="all"
pytest_args=()
durations_out=""
seen_sep=false
TB_ARG="--tb=short"

if [[ $# -gt 0 ]]; then
    case "$1" in
        unit|integration|all)
            TEST_TYPE="$1"
            shift
            ;;
    esac
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            seen_sep=true
            shift
            pytest_args+=("$@")
            break
            ;;
        --durations-out)
            durations_out="$2"
            shift 2
            ;;
        unit|integration|all)
            if [[ "${seen_sep}" == false ]]; then
                TEST_TYPE="$1"
                shift
            else
                pytest_args+=("$1")
                shift
            fi
            ;;
        *)
            pytest_args+=("$1")
            shift
            ;;
    esac
done

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing test dependencies..."
    pip install -e ".[test]"
fi

# Ensure pytest-split is installed for --splits/--group support
if ! python3 -c "import pytest_split" 2>/dev/null; then
    echo "pytest-split not found. Installing..."
    pip install pytest-split
fi

# Ensure pytest-cov is installed for coverage collection
if ! python3 -c "import pytest_cov" 2>/dev/null; then
    echo "pytest-cov not found. Installing..."
    pip install pytest-cov
fi
HAS_COV=true

# Multi-docker CI env hook: force PYTHONPATH inside container
if [[ "${CI_MULTI_DOCKER:-0}" == "1" ]]; then
    export PYTHONPATH="/workspace/omniinfer/omni:/workspace/omniinfer/tests:${PYTHONPATH:-}"
fi

# durations plugin args (optional)
duration_args=()
if [[ -n "${durations_out}" ]]; then
    # Ensure tests package is importable for pytest plugin
    duration_args=( -p ut_CI_check.ut_CI_durations_plugin --durations-out "${durations_out}" )
fi
echo "[INFO] PYTHONPATH: $PYTHONPATH"
cd ..
echo "[INFO] git status"
GIT_PAGER=cat git status
echo "[INFO] git branch"
GIT_PAGER=cat git branch --show-current
GIT_PAGER=cat git log -5 --pretty=%s
cd -

case "$TEST_TYPE" in
    unit)
        echo "Running unit tests (no NPU required)..."
        if [ "$HAS_COV" = true ]; then
            pytest unit/ "${TB_ARG}" \
                "${duration_args[@]}" \
                --cov=omni_npu \
                --cov-report=term-missing \
                --cov-report=html \
                --cov-config=./.coveragerc \
                -v "${pytest_args[@]}"
        else
            pytest unit/ -v "${TB_ARG}" "${duration_args[@]}" "${pytest_args[@]}"
        fi
        ;;
    integration)
        echo "Running integration tests (requires NPU hardware)..."
        echo "  - Single-device tests with pytest"
        pytest integration/ "${TB_ARG}" -v -k "not TestNPUCommunicatorMultiDevice" "${duration_args[@]}" "${pytest_args[@]}"
        echo ""
        echo "  - Multi-device tests with torchrun (2 NPUs)"
        torchrun --nproc_per_node=2 -m pytest integration/distributed/test_communicator.py::TestNPUCommunicatorMultiDevice -v "${TB_ARG}" "${pytest_args[@]}"
        ;;
    all)
        echo "Running all tests..."
        if [ "$HAS_COV" = true ]; then
            echo "[INFO] About to run: pytest ${TB_ARG} --cov=omni_npu --cov-report=term-missing --cov-report=html --cov-config=./.coveragerc -v ${pytest_args[*]}"
            pytest "${TB_ARG}" \
                "${duration_args[@]}" \
                --cov=omni_npu \
                --cov-report=term-missing \
                --cov-report=html \
                --cov-config=./.coveragerc \
                -v "${pytest_args[@]}"
        else
            echo "[INFO] About to run: pytest -v ${TB_ARG} ${pytest_args[*]}"
            pytest -v "${TB_ARG}" "${duration_args[@]}" "${pytest_args[@]}"
        fi
        ;;
    *)
        echo "Usage: $0 [unit|integration|all]"
        echo ""
        echo "  unit        - Run unit tests only (no NPU required)"
        echo "  integration - Run integration tests only (requires NPU)"
        echo "  all         - Run all tests (default)"
        exit 1
        ;;
esac

echo ""
if [ "$HAS_COV" = true ] && ([ "$TEST_TYPE" = "unit" ] || [ "$TEST_TYPE" = "all" ]); then
    echo "Coverage report saved to htmlcov/index.html"
fi
echo "Tests completed!"
