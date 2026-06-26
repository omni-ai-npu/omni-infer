#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Script to run tests for omni-cache

set -e

# Default values
TB_ARG="--tb=short"
HAS_COV=true
TEST_TYPE="all"
SETUP_HUGETLBFS=true

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HUGETLBFS_SCRIPT="$PROJECT_ROOT/omni_cache/tools/setup_hugetlbfs_2MB.sh"

# Parse command line arguments
pytest_args=()

usage() {
    echo "Usage: $0 [OPTIONS] [-- PYTEST_ARGS]"
    echo ""
    echo "Options:"
    echo "  --no-cov          Run without coverage report"
    echo "  --no-hugetlbfs    Skip hugetlbfs setup"
    echo "  --unit            Run unit tests only"
    echo "  --module          Run module tests only"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                         # Run all tests (with hugetlbfs setup)"
    echo "  $0 --no-hugetlbfs          # Run without hugetlbfs setup"
    echo "  $0 --unit                  # Run unit tests with coverage"
    echo "  $0 --module                # Run module tests with coverage"
    echo "  $0 --no-cov                # Run without coverage"
    echo "  $0 -k \"test_kv\"            # Run tests matching pattern"
    echo "  $0 -- -x                   # Stop on first failure"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --)
            shift
            pytest_args+=("$@")
            break
            ;;
        --no-cov)
            HAS_COV=false
            shift
            ;;
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --module)
            TEST_TYPE="module"
            shift
            ;;
        --setup-hugetlbfs)
            SETUP_HUGETLBFS=true
            shift
            ;;
        --no-hugetlbfs)
            SETUP_HUGETLBFS=false
            shift
            ;;
        --help|-h)
            usage
            exit 0
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

# Ensure pytest-cov is installed for coverage collection
if [ "$HAS_COV" = true ]; then
    if ! python3 -c "import pytest_cov" 2>/dev/null; then
        echo "pytest-cov not found. Installing..."
        pip install pytest-cov
    fi
fi

# Setup hugetlbfs if requested
if [ "$SETUP_HUGETLBFS" = true ]; then
    if [ -f "$HUGETLBFS_SCRIPT" ]; then
        echo "[INFO] Setting up hugetlbfs..."
        if [ "$EUID" -ne 0 ]; then
            echo "[INFO] Running hugetlbfs setup with sudo..."
            sudo bash "$HUGETLBFS_SCRIPT"
        else
            bash "$HUGETLBFS_SCRIPT"
        fi
        echo "[INFO] Hugetlbfs setup completed"
    else
        echo "[WARN] Hugetlbfs setup script not found: $HUGETLBFS_SCRIPT"
        echo "[WARN] Skipping hugetlbfs setup"
    fi
fi

echo "[INFO] PYTHONPATH: $PYTHONPATH"
cd ..
echo "[INFO] git status"
GIT_PAGER=cat git status
echo "[INFO] git branch"
GIT_PAGER=cat git branch --show-current
GIT_PAGER=cat git log -5 --pretty=%s
cd -

# Determine test directories based on TEST_TYPE
case "$TEST_TYPE" in
    unit)
        test_dirs=("unit_tests/")
        echo "[INFO] Running unit tests only"
        ;;
    module)
        test_dirs=("module_tests/")
        echo "[INFO] Running module tests only"
        ;;
    all)
        test_dirs=("unit_tests/" "module_tests/")
        echo "[INFO] Running all tests"
        ;;
esac

# Run tests
if [ "$HAS_COV" = true ]; then
    echo "[INFO] Running with coverage report"
    pytest "${test_dirs[@]}" "${TB_ARG}" \
        --cov=omni_cache \
        --cov-report=term-missing \
        --cov-report=html \
        -v "${pytest_args[@]}"
    echo ""
    echo "Coverage report saved to htmlcov/index.html"
else
    echo "[INFO] Running without coverage"
    pytest "${test_dirs[@]}" -v "${TB_ARG}" "${pytest_args[@]}"
fi

echo ""
echo "Tests completed!"