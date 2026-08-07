#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <host_omni_root> [host_coverage_dir]"
  echo "Example: $0 /path/to/omni-npu"
  exit 1
fi

HOST_OMNI_DIR="$1"
HOST_COV_DIR="${2:-${HOST_OMNI_DIR}/tests/coverage_from_dockers}"

# Merge inside this container to avoid host path mapping issues
MERGE_CONTAINER="${MERGE_CONTAINER:-DT_1}"
OMNI_IN_CONTAINER="${OMNI_IN_CONTAINER:-/workspace/omniinfer/components/omni-npu}"
MERGE_DIR_IN_CONTAINER="${MERGE_DIR_IN_CONTAINER:-/workspace/coverage_merge}"

RCFILE_IN_CONTAINER="${OMNI_IN_CONTAINER}/tests/.coveragerc"
MERGED_COV="${HOST_COV_DIR}/.coverage"
HTML_DIR="${HOST_COV_DIR}/htmlcov"
REPORT_TXT="${HOST_COV_DIR}/coverage_report.txt"

if [ ! -d "${HOST_COV_DIR}" ]; then
  echo "[ERROR] coverage dir not found: ${HOST_COV_DIR}"
  exit 1
fi

shopt -s nullglob
cov_files=("${HOST_COV_DIR}"/.coverage.*)
shopt -u nullglob

if [ ${#cov_files[@]} -eq 0 ]; then
  echo "[WARN] no per-container coverage files found in ${HOST_COV_DIR}"
  exit 0
fi

rm -f "${MERGED_COV}"
rm -rf "${HTML_DIR}"
rm -f "${REPORT_TXT}"

echo "[INFO] combining coverage data files in container: ${MERGE_CONTAINER}"
docker exec "${MERGE_CONTAINER}" /bin/bash -c "
  set -e
  mkdir -p '${MERGE_DIR_IN_CONTAINER}'
  rm -f '${MERGE_DIR_IN_CONTAINER}/.coverage.'*
  rm -f '${MERGE_DIR_IN_CONTAINER}/.coverage'
  rm -rf '${MERGE_DIR_IN_CONTAINER}/htmlcov'
"

for f in "${cov_files[@]}"; do
  base="$(basename "${f}")"
  docker cp "${f}" "${MERGE_CONTAINER}:${MERGE_DIR_IN_CONTAINER}/${base}"
done

docker exec "${MERGE_CONTAINER}" /bin/bash -c "
  set -e
  cd '${MERGE_DIR_IN_CONTAINER}'
  coverage combine --keep --data-file .coverage .coverage.*
  coverage html --rcfile '${RCFILE_IN_CONTAINER}' --data-file .coverage -d htmlcov
  coverage report --rcfile '${RCFILE_IN_CONTAINER}' --data-file .coverage -m | tee coverage_report.txt
  coverage xml --rcfile '${RCFILE_IN_CONTAINER}' --data-file .coverage -o coverage.xml
"

echo "[INFO] copying merged reports from container..."
docker cp "${MERGE_CONTAINER}:${MERGE_DIR_IN_CONTAINER}/.coverage" "${MERGED_COV}"
docker cp "${MERGE_CONTAINER}:${MERGE_DIR_IN_CONTAINER}/htmlcov" "${HTML_DIR}"
docker cp "${MERGE_CONTAINER}:${MERGE_DIR_IN_CONTAINER}/coverage_report.txt" "${REPORT_TXT}"
docker cp "${MERGE_CONTAINER}:${MERGE_DIR_IN_CONTAINER}/coverage.xml" "${HOST_COV_DIR}/coverage.xml"

echo "[INFO] merged coverage file: ${MERGED_COV}"
