#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Configure project name
PROJECT_NAME="sglang"

# Parse command line arguments (supports --build-path to specify project directory)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # default project directory

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-path)
            if [[ -n "$2" ]]; then
                PROJECT_DIR="$2"
                PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)" # Convert to absolute path
                shift 2
            else
                echo "Error: --build-path requires a directory path"
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknow option '$1'"
            echo "Usage: $0 [--build-path /path/to/project]"
            exit 1
            ;;
    esac
done

# define output directory
DIST_DIR="${PROJECT_DIR}/dist"

# Check if pyproject.toml exists the project directory
if [ ! -f "${PROJECT_DIR}/pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in '${PROJECT_DIR}'"
    exit 1
fi

# Function: Clean up old build files
clean_old_build() {
    echo "Cleaning old build files..."
    rm -rf "${PROJECT_DIR}/build"
    rm -rf "${PROJECT_DIR}/${PROJECT_NAME}.egg-info"
    rm -rf "${DIST_DIR}"
    mkdir -p "${DIST_DIR}"
}

# Function: Check and install build dependencies
check_dependencies() {
    echo "Checking build dependencies..."
    # Retrieve build dependencies specified in pyproject.toml's [build-system] section
    REQUIRES=$(grep -A 2 "\[build-system\]" "${PROJECT_DIR}/pyproject.toml" | grep "requires" | sed -E 's/requires = \[(.*)\]/\1/' | tr -d '"' | tr ',' ' ')
    # Check if each dependency is already installed
    for dep in $REQUIRES; do
        pkg=$(echo "$dep" | sed -E 's/([a-zA-Z0-9_-]+).*/\1/')
        if ! python3 -c "import $pkg" &> /dev/null; then
            echo "Installing missing dependency: $dep"
            if ! python3 -m pip install "$dep"; then
                echo "Error: Failed to install $dep"
                exit 1
            fi
        fi
    done
}

# Function: Build the whl package
build_whl() {
    echo "Starting to build whl package for NPU environment..."
    # Use configuration from pyproject.toml for building
    if python3 -m build --wheel --outdir "${DIST_DIR}" "${PROJECT_DIR}"; then
        echo "--------------------------------------------"
        echo "Build successful! Generated whl package:"
        ls -l "${DIST_DIR}"/*.whl
        return 0
    else
        echo "Error: Build failed"
        return 1
    fi
}

# Main workflow
clean_old_build
check_dependencies

# Execute the build precess
if build_whl; then
    echo "--------------------------------------------"
    echo "Package built successfully. Output directory: ${DIST_DIR}"
else
    echo "--------------------------------------------"
    echo "Build failed. Check error messages above"
    exit 1
fi

echo "Patching NPU adaptation..."
cd "$(dirname "${BASH_SOURCE[0]}")"/../infer_engines
bash bash_install_sglang.sh
cd -

echo "============================================="
exit 0