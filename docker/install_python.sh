#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
set -exo pipefail

# Check if the installation directory argument is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <install_directory> <python_version>"
    echo "Example: $0 /opt/python3.11.12 3.11.12"
    exit 1
fi

INSTALL_DIR="$1"
PYTHON_VERSION="$2"
PYTHON_TGZ="Python-${PYTHON_VERSION}.tgz"
PYTHON_URL="https://mirrors.huaweicloud.com/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"

# Create a temporary download directory
TMP_DIR="/tmp"
echo "Downloading Python ${PYTHON_VERSION} from ${PYTHON_URL}"
wget -P "${TMP_DIR}" --no-check-certificate "${PYTHON_URL}"

# Create the parent directory of the installation directory (if it doesn't exist)
mkdir -p "$(dirname "${INSTALL_DIR}")"

# Extract the source code to a temporary directory (to avoid polluting the installation directory)
echo "Extracting Python source..."
tar -xzf "${TMP_DIR}/${PYTHON_TGZ}" -C "${TMP_DIR}"

# Enter the source code directory
SRC_DIR="${TMP_DIR}/Python-${PYTHON_VERSION}"
if [ ! -d "${SRC_DIR}" ]; then
    echo "Failed to find the extracted Python source directory: ${SRC_DIR}"
    exit 1
fi

# Install build dependencies
echo "Installing build dependencies..."


# Configure the build prefix to the target installation directory
echo "Configuring and building Python..."
cd "${SRC_DIR}" && ./configure --with-system-ffi --enable-optimizations --enable-shared --prefix="${INSTALL_DIR}"

make -j$(nproc)
make install

# Clean up temporary files
echo "Cleaning up..."
rm -rf "${SRC_DIR}"
rm -rf "${TMP_DIR}/${PYTHON_TGZ}"

echo "Python ${PYTHON_VERSION} has been successfully installed to ${INSTALL_DIR}"