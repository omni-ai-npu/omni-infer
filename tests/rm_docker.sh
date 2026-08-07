#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ut_config.sh"

echo "[INFO] Removing containers: ${UT_CONTAINER_NAMES[*]}"

for CONTAINER_NAME in "${UT_CONTAINER_NAMES[@]}"; do
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[INFO] Removing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  else
    echo "[WARN] Container ${CONTAINER_NAME} does not exist. Skipping."
  fi
done

echo "[INFO] Done."
