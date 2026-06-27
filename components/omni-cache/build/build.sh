#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Usage examples:
#   # default: build using current environment, project dir = script parent dir
#   ./build/build.sh
#
#   # override project dir explicitly
#   PROJECT_DIR=/path/to/repo ./build/build.sh
#
#   # build inside an isolated venv
#   USE_VENV=1 VENV_DIR=.venv ./build/build.sh
#
set -euo pipefail

# Configurable env vars
OUT_DIR="${OUT_DIR:-dist}"
USE_VENV="${USE_VENV:-0}"                # 0 = use current environment; 1 = create/use venv
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-1}"  # install numpy, concurrent_log_handler by default
INSTALL_TORCH="${INSTALL_TORCH:-0}"      # default do not auto-install torch
TORCH_WHEEL_URL="${TORCH_WHEEL_URL:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
EXTRA_PIP_ARGS="${EXTRA_PIP_ARGS:-}"
EXTRA_PIP_ARGS_ARRAY=()
if [ -n "${EXTRA_PIP_ARGS}" ]; then
  read -r -a EXTRA_PIP_ARGS_ARRAY <<< "${EXTRA_PIP_ARGS}"
fi
# PROJECT_DIR: directory to pass as source to python -m build; default = parent dir of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

die() { echo "ERROR: $*" >&2; exit 1; }

echo "== omni-cache wheel build helper =="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "USE_VENV=${USE_VENV}"
if [ "${USE_VENV}" = "1" ]; then
  echo "VENV_DIR=${VENV_DIR}"
fi
echo "INSTALL_RUNTIME_DEPS=${INSTALL_RUNTIME_DEPS}"
echo "INSTALL_TORCH=${INSTALL_TORCH}"
echo "TORCH_WHEEL_URL=${TORCH_WHEEL_URL:+(provided)}"
echo "TORCH_INDEX_URL=${TORCH_INDEX_URL:+(provided)}"
echo

# Helper: run pip install with the chosen python
install_with_python() {
  local py="$1"; shift
  "${py}" -m pip install "$@" "${EXTRA_PIP_ARGS_ARRAY[@]}"
}

# Choose python executable
if [ "${USE_VENV}" = "1" ]; then
  # create venv if needed
  if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtualenv at ${VENV_DIR}..."
    python -m venv "${VENV_DIR}" || die "Failed to create virtualenv"
  else
    echo "Using existing virtualenv at ${VENV_DIR}"
  fi

  if [ -x "${VENV_DIR}/bin/python" ]; then
    PY="${VENV_DIR}/bin/python"
  elif [ -x "${VENV_DIR}/Scripts/python.exe" ]; then
    PY="${VENV_DIR}/Scripts/python.exe"
  else
    die "Cannot find python in ${VENV_DIR}"
  fi
else
  # use current environment python
  if command -v python >/dev/null 2>&1; then
    PY="python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    die "No python executable found in PATH"
  fi
fi

echo "Using python: $(${PY} -c 'import sys; print(sys.executable)')"


# Optionally install runtime deps that are required during egg_info/import-time
if [ "${INSTALL_RUNTIME_DEPS}" != "0" ]; then
  echo "Installing runtime deps into chosen environment: numpy, concurrent_log_handler, msgpack (if missing)..."
  if ! install_with_python "${PY}" numpy concurrent_log_handler msgpack; then
    echo "Warning: installing runtime deps failed. If setup.py or package import requires them at build-time, install them manually and re-run."
  fi
else
  echo "Skipping automatic runtime dependencies installation (INSTALL_RUNTIME_DEPS=0)"
fi

# Optionally install torch into the chosen environment (only if requested)
if [ "${INSTALL_TORCH}" = "1" ]; then
  echo "Installing torch into the chosen environment..."
  if [ -n "${TORCH_WHEEL_URL}" ]; then
    install_with_python "${PY}" "${TORCH_WHEEL_URL}" || die "Failed to install torch from TORCH_WHEEL_URL"
  else
    if [ -n "${TORCH_INDEX_URL}" ]; then
      install_with_python "${PY}" --index-url "${TORCH_INDEX_URL}" torch || {
        echo "Failed to install torch using TORCH_INDEX_URL. Try providing TORCH_WHEEL_URL or install torch manually."
        exit 1
      }
    else
      install_with_python "${PY}" torch || {
        echo "Automatic pip install torch failed. Try setting TORCH_WHEEL_URL or install torch manually into the environment and re-run."
        exit 1
      }
    fi
  fi
else
  echo "Not auto-installing torch (INSTALL_TORCH=${INSTALL_TORCH}). If setup.py imports torch at import time, ensure it is available in the chosen environment."
fi

# Quick import check for build tools
echo "Sanity check: can import build package..."
"${PY}" - <<'PYCODE' || die "Python in chosen environment failed"
import sys
try:
    import build, setuptools, wheel
except Exception as e:
    print("Error importing build tools:", e, file=sys.stderr)
    raise
print("Build tools import OK")
PYCODE

# Install omni-cache in editable mode (triggers build_py for native C++ components)
echo "Running: ${PY} -m pip install -e ${PROJECT_DIR} --no-build-isolation ${EXTRA_PIP_ARGS}"
"${PY}" -m pip install -e "${PROJECT_DIR}" --no-build-isolation "${EXTRA_PIP_ARGS_ARRAY[@]}" || {
  echo
  echo "Build failed. Common causes:"
  echo " - missing runtime dependency at import-time (numpy/concurrent_log_handler/msgpack/torch)"
  echo " - missing native headers or toolkits (e.g., ASCEND_TOOLKIT_HOME for Ascend)"
  echo
  echo "Native C++ builds (tensor_register_lib and ox) are now triggered automatically from setup.py."
  echo "If the error mentions missing Python packages, install them into the chosen environment and retry."
  exit 1
}

echo
"${PY}" -m pip show omni-cache
