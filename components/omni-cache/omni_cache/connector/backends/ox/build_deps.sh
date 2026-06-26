#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -euo pipefail

MSGPACK_VER="${OMNI_CACHE_MSGPACK_VER:-7.0.0}"
MSGPACK_TARBALL="msgpack-cxx-${MSGPACK_VER}.tar.gz"
MSGPACK_URL="${OMNI_CACHE_MSGPACK_URL:-https://github.com/msgpack/msgpack-c/releases/download/cpp-${MSGPACK_VER}/${MSGPACK_TARBALL}}"
MSGPACK_DIR="msgpack-cxx-${MSGPACK_VER}"

LIBZMQ_VER="${OMNI_CACHE_LIBZMQ_VER:-4.3.5}"
LIBZMQ_TARBALL="zeromq-${LIBZMQ_VER}.tar.gz"
LIBZMQ_URL="${OMNI_CACHE_LIBZMQ_URL:-https://github.com/zeromq/libzmq/releases/download/v${LIBZMQ_VER}/${LIBZMQ_TARBALL}}"
LIBZMQ_DIR="zeromq-${LIBZMQ_VER}"

CPPZMQ_VER="${OMNI_CACHE_CPPZMQ_VER:-4.11.0}"
CPPZMQ_TARBALL="v${CPPZMQ_VER}.tar.gz"
CPPZMQ_URL="${OMNI_CACHE_CPPZMQ_URL:-https://github.com/zeromq/cppzmq/archive/refs/tags/${CPPZMQ_TARBALL}}"
CPPZMQ_DIR="cppzmq-${CPPZMQ_VER}"

INSTALL_PREFIX="${OMNI_CACHE_OX_DEPS_PREFIX:-/usr}"
WORK_DIR="${OMNI_CACHE_OX_DEPS_WORK_DIR:-}"

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

download() {
  local url="$1"
  local file="$2"

  if command -v wget >/dev/null 2>&1; then
    wget --no-check-certificate -O "${file}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -k -o "${file}" "${url}"
  else
    echo "Neither wget nor curl is available for downloading ${url}" >&2
    exit 1
  fi
}

download_and_extract() {
  local url="$1"
  local tarball="$2"

  echo "[download] ${url}"
  download "${url}" "${tarball}"
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
  init_work_dir
  cd "${WORK_DIR}"

  download_and_extract "${MSGPACK_URL}" "${MSGPACK_TARBALL}"
  build_install_msgpack

  download_and_extract "${LIBZMQ_URL}" "${LIBZMQ_TARBALL}"
  build_install_libzmq

  download_and_extract "${CPPZMQ_URL}" "${CPPZMQ_TARBALL}"
  build_install_cppzmq
}

main "$@"
