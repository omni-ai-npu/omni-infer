#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/install_logs"
mkdir -p "${LOG_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ut_config.sh"

OMNI_ROOT="${1:?Usage: $0 <omni-npu_root_dir>}"
CONTAINER_OMNI_ROOT="${UT_CONTAINER_OMNI_ROOT_DEFAULT}"

HOST_COV_DIR="${OMNI_ROOT}/tests/coverage_from_dockers"
CONTAINER_COV_FILE="${CONTAINER_OMNI_ROOT}/tests/.coverage"
mkdir -p "${HOST_COV_DIR}"
MERGE_COV="${MERGE_COV:-1}"

HOST_DURATIONS_DIR="${OMNI_ROOT}/tests/test_durations_from_dockers"
CONTAINER_DURATIONS_DIR="${CONTAINER_OMNI_ROOT}/tests/test_durations_from_dockers"
mkdir -p "${HOST_DURATIONS_DIR}"

pids=()

for CONTAINER_NAME in "${!UT_CONTAINER_TEST_ARGS[@]}"; do
  echo "[INFO] Syncing and running tests in ${CONTAINER_NAME}"

  TEST_ARGS_STR="${UT_CONTAINER_TEST_ARGS[$CONTAINER_NAME]}"

  docker exec "${CONTAINER_NAME}" /bin/bash -c "
    . ~/.bashrc
    set -e
    cd ${CONTAINER_OMNI_ROOT}
    # pip install -e '.[tests]' > /dev/null
    mkdir -p ${CONTAINER_OMNI_ROOT}
    rm -rf ${CONTAINER_OMNI_ROOT}/*
    rm -rf ${CONTAINER_OMNI_ROOT}/.??* || true
    cp -r ${OMNI_ROOT}/* ${CONTAINER_OMNI_ROOT}/
    cp -r ${OMNI_ROOT}/.gitmodules ${CONTAINER_OMNI_ROOT}/
    cp -r ${OMNI_ROOT}/.git ${CONTAINER_OMNI_ROOT}/ 2>/dev/null || true
    pip install --no-build-isolation -e /workspace/omniinfer

    cd ${CONTAINER_OMNI_ROOT}/tests
    mkdir -p ${CONTAINER_DURATIONS_DIR}
    CI_MULTI_DOCKER=1 bash ./run_tests.sh \
      --durations-out ${CONTAINER_DURATIONS_DIR}/test_durations_${CONTAINER_NAME}.json \
      ${TEST_ARGS_STR}
  " > "${LOG_DIR}/${CONTAINER_NAME}.log" 2>&1 &

  pids+=($!)
done

echo "[INFO] Waiting for all containers to finish..."

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

echo "[INFO] Collecting per-container durations..."
for c in "${!UT_CONTAINER_TEST_ARGS[@]}"; do
  src="${CONTAINER_DURATIONS_DIR}/test_durations_${c}.json"
  dst="${HOST_DURATIONS_DIR}/test_durations_${c}.json"
  if docker exec "${c}" /bin/bash -c "test -f ${src}"; then
    echo "[INFO] docker cp ${c}:${src} -> ${dst}"
    docker cp "${c}:${src}" "${dst}"
  else
    echo "[WARN] durations json not found in ${c}: ${src}"
  fi
done

MERGED_DURATIONS_JSON="${OMNI_ROOT}/tests/test_durations_merged.json"
MERGE_SCRIPT="${OMNI_ROOT}/tests/ut_CI_check/ut_CI_merge_test_durations_json.py"
if [[ -f "${MERGE_SCRIPT}" ]]; then
  args=(--out "${MERGED_DURATIONS_JSON}")
  for f in "${HOST_DURATIONS_DIR}"/test_durations_*.json; do
    if [[ -f "${f}" ]]; then
      args+=(--container-json "${f}")
    fi
  done
  if [[ ${#args[@]} -gt 2 ]]; then
    python3 "${MERGE_SCRIPT}" "${args[@]}" || true
  else
    echo "[WARN] no durations json files to merge in ${HOST_DURATIONS_DIR}"
  fi
else
  echo "[WARN] merge script not found: ${MERGE_SCRIPT}"
fi

echo "[INFO] Collecting coverage files..."
rm -f "${HOST_COV_DIR}/.coverage."*
for c in "${!UT_CONTAINER_TEST_ARGS[@]}"; do
  if docker exec "${c}" /bin/bash -c "test -f ${CONTAINER_COV_FILE}"; then
    echo "[INFO] docker cp ${c}:${CONTAINER_COV_FILE} -> ${HOST_COV_DIR}/.coverage.${c}"
    docker cp "${c}:${CONTAINER_COV_FILE}" "${HOST_COV_DIR}/.coverage.${c}"
  else
    echo "[WARN] coverage file not found in ${c}: ${CONTAINER_COV_FILE}"
  fi
done

if [[ "${MERGE_COV}" == "1" ]]; then
  echo "[INFO] Merging coverage on host..."
  bash "${OMNI_ROOT}/tests/multi_docker_collect_coverage.sh" "${OMNI_ROOT}" || true
else
  echo "[INFO] MERGE_COV=0, skip coverage merge."
fi

exit "${fail}"
