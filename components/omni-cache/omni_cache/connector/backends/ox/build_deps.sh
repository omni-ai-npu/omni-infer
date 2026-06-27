#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -euo pipefail

MSGPACK_VER="${OMNI_CACHE_MSGPACK_VER:-7.0.0}"
MSGPACK_TARBALL="msgpack-cxx-${MSGPACK_VER}.tar.gz"
MSGPACK_URL="${OMNI_CACHE_MSGPACK_URL:-https://github.com/msgpack/msgpack-c/releases/download/cpp-${MSGPACK_VER}/${MSGPACK_TARBALL}}"
MSGPACK_SHA256="${OMNI_CACHE_MSGPACK_SHA256:-7504b7af7e7b9002ce529d4f941e1b7fb1fb435768780ce7da4abaac79bb156f}"
MSGPACK_DIR="msgpack-cxx-${MSGPACK_VER}"

LIBZMQ_VER="${OMNI_CACHE_LIBZMQ_VER:-4.3.5}"
LIBZMQ_TARBALL="zeromq-${LIBZMQ_VER}.tar.gz"
LIBZMQ_URL="${OMNI_CACHE_LIBZMQ_URL:-https://github.com/zeromq/libzmq/releases/download/v${LIBZMQ_VER}/${LIBZMQ_TARBALL}}"
LIBZMQ_SHA256="${OMNI_CACHE_LIBZMQ_SHA256:-6653ef5910f17954861fe72332e68b03ca6e4d9c7160eb3a8de5a5a913bfab43}"
LIBZMQ_DIR="zeromq-${LIBZMQ_VER}"

CPPZMQ_VER="${OMNI_CACHE_CPPZMQ_VER:-4.11.0}"
CPPZMQ_TARBALL="v${CPPZMQ_VER}.tar.gz"
CPPZMQ_URL="${OMNI_CACHE_CPPZMQ_URL:-https://github.com/zeromq/cppzmq/archive/refs/tags/${CPPZMQ_TARBALL}}"
CPPZMQ_SHA256="${OMNI_CACHE_CPPZMQ_SHA256:-0fff4ff311a7c88fdb76fceefba0e180232d56984f577db371d505e4d4c91afd}"
CPPZMQ_DIR="cppzmq-${CPPZMQ_VER}"

INSTALL_PREFIX="${OMNI_CACHE_OX_DEPS_PREFIX:-/usr}"
WORK_DIR="${OMNI_CACHE_OX_DEPS_WORK_DIR:-}"
ALLOW_UNVERIFIED="${OMNI_CACHE_ALLOW_UNVERIFIED_DEPS:-0}"
INSECURE_DOWNLOAD="${OMNI_CACHE_INSECURE_DOWNLOAD:-1}"

cleanup() {
  if [[ -n "${_OMNI_CACHE_TMP_WORK_DIR:-}" ]]; then
    rm -rf "${_OMNI_CACHE_TMP_WORK_DIR}"
  fi
}
trap cleanup EXIT

init_work_dir() {
  if [[ -z "${WORK_DIR}" ]]; then
    _OMNI_CACHE_TMP_WORK_DIR="$(mktemp -d -t omni-cache-ox-deps.XXXXXX)"
    WORK_DIR="${_OMNI_CACHE_TMP_WORK_DIR}"
  fi
  mkdir -p "${WORK_DIR}"
}

dependencies_available() {
  local probe_dir
  probe_dir="$(mktemp -d -t omni-cache-ox-probe.XXXXXX)"
  cat > "${probe_dir}/probe.cpp" <<'EOF'
#include <msgpack.hpp>
#include <zmq.hpp>

int main() {
  zmq::context_t context;
  msgpack::sbuffer buffer;
  return 0;
}
EOF
  if "${CXX:-g++}" -std=c++20 "${probe_dir}/probe.cpp" -lzmq \
      -o "${probe_dir}/probe" >/dev/null 2>&1; then
    rm -rf "${probe_dir}"
    return 0
  fi
  rm -rf "${probe_dir}"
  return 1
}

download() {
  local url="$1"
  local file="$2"

  if command -v wget >/dev/null 2>&1; then
    if [[ "${INSECURE_DOWNLOAD}" == "1" ]]; then
      wget --no-check-certificate --timeout=15 --tries=2 \
        -O "${file}" "${url}"
    else
      wget --timeout=15 --tries=2 -O "${file}" "${url}"
    fi
  elif command -v curl >/dev/null 2>&1; then
    if [[ "${INSECURE_DOWNLOAD}" == "1" ]]; then
      curl -L -k --fail --connect-timeout 15 --max-time 120 --retry 1 \
        -o "${file}" "${url}"
    else
      curl -L --fail --connect-timeout 15 --max-time 120 --retry 1 \
        -o "${file}" "${url}"
    fi
  else
    echo "Neither wget nor curl is available for downloading ${url}" >&2
    return 1
  fi
}

verify_sha256() {
  local file="$1"
  local expected="$2"

  if [[ -z "${expected}" ]]; then
    if [[ "${ALLOW_UNVERIFIED}" == "1" ]]; then
      echo "[verify ] ${file}: skipped by OMNI_CACHE_ALLOW_UNVERIFIED_DEPS=1" >&2
      return
    fi
    echo "Missing SHA256 for ${file}. Set the matching OMNI_CACHE_*_SHA256 or OMNI_CACHE_ALLOW_UNVERIFIED_DEPS=1." >&2
    exit 1
  fi
  echo "${expected}  ${file}" | sha256sum -c -
}

download_and_extract() {
  local url="$1"
  local tarball="$2"
  local sha256="$3"

  echo "[download] ${url}"
  if ! download "${url}" "${tarball}"; then
    echo "Failed to download OX build dependencies." >&2
    echo "If this is an offline environment, install msgpack-cxx, libzmq, and cppzmq with the system package manager, then rerun the installation." >&2
    echo "No specific dependency version is required; the installed packages only need to compile and link the OX backend." >&2
    exit 1
  fi
  echo "[verify ] ${tarball}"
  verify_sha256 "${tarball}" "${sha256}"
  echo "[extract ] ${tarball}"
  tar -zxf "${tarball}"
}

build_install_msgpack() {
  echo "[build   ] msgpack-c ${MSGPACK_VER}"
  cd "${MSGPACK_DIR}"
  mkdir -p build && cd build
  cmake .. \
    -DMSGPACK_BUILD_EXAMPLES=OFF \
    -DMSGPACK_BUILD_TESTS=OFF \
    -DMSGPACK_USE_BOOST=OFF \
    -DMSGPACK_ENABLE_SHARED=ON \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
  make -j"$(nproc)"
  make install
  cd "${WORK_DIR}"
}

build_install_libzmq() {
  echo "[build   ] libzmq ${LIBZMQ_VER}"
  cd "${LIBZMQ_DIR}"
  mkdir -p build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
  make -j"$(nproc)"
  make install
  cd "${WORK_DIR}"
}

build_install_cppzmq() {
  echo "[build   ] cppzmq ${CPPZMQ_VER}"
  cd "${CPPZMQ_DIR}"
  mkdir -p build && cd build
  cmake .. \
    -DCPPZMQ_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
  make -j"$(nproc)"
  make install
  cd "${WORK_DIR}"
}

main() {
  if dependencies_available; then
    echo "[skip    ] OX build dependencies are already available"
    return
  fi

  echo "[missing ] OX requires msgpack-cxx, libzmq, and cppzmq"
  init_work_dir
  cd "${WORK_DIR}"

  download_and_extract "${MSGPACK_URL}" "${MSGPACK_TARBALL}" "${MSGPACK_SHA256}"
  build_install_msgpack

  download_and_extract "${LIBZMQ_URL}" "${LIBZMQ_TARBALL}" "${LIBZMQ_SHA256}"
  build_install_libzmq

  download_and_extract "${CPPZMQ_URL}" "${CPPZMQ_TARBALL}" "${CPPZMQ_SHA256}"
  build_install_cppzmq
}

main "$@"
