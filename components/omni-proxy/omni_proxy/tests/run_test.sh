#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -euo pipefail

export PROXY_VLLM_POOL=1 # A switch for preparing vLLM to speed up proxy UT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TEST_DIRS=(
  "${SCRIPT_DIR}"  # scan test_*.py
)

find "${SCRIPT_DIR}" -name "test_*.py"
GEN_PROXY_COV_SCRIPT="${SCRIPT_DIR}/gen_proxy_cov.sh" 
SETUP_VLLM_MOCK_SCRIPT="${SCRIPT_DIR}/setup_vllm_mock.sh" 

echo "[INFO] Setup VLLM."
if [[ -f "${SETUP_VLLM_MOCK_SCRIPT}" ]]; then
  bash "${SETUP_VLLM_MOCK_SCRIPT}"
else
  echo "[WARN] ${SETUP_VLLM_MOCK_SCRIPT} does not exist"
fi

stop_mock_processes() {
  echo "[INFO] Stopping VLLM and PROXY..."
  python "${SCRIPT_DIR}/run_vllm_mock.py" stop || true
  python "${SCRIPT_DIR}/run_proxy.py" stop || true
  python "${SCRIPT_DIR}/run_vllm_mock_epd.py" stop || true
  pkill -9 nginx || true
  echo "[INFO] Cleaning Done"
}


stop_mock_processes

target_path=("${TEST_DIRS[@]}")  
report_name="pytest-proxy-mock.xml"

cmd=(
  pytest 
  --tb=long -sv                
  $SCRIPT_DIR/test_*.py       
)

echo "[INFO] execute pytest command: ${cmd[@]}"
set +e  

unset http_proxy
unset https_proxy

( cd "${SCRIPT_DIR}" && stdbuf -oL -eL "${cmd[@]}" ) 2>&1 
TEST_EXIT_CODE=$?  
set -e

collect_coverage_reports() {
  if [[ -f "${GEN_PROXY_COV_SCRIPT}" ]]; then
    bash "${GEN_PROXY_COV_SCRIPT}"
  else
    echo "[WARN] proxy coverage script ${GEN_PROXY_COV_SCRIPT} does not exist"
  fi

}
collect_coverage_reports "${SCRIPT_DIR}"

echo "[INFO] Cleaning..."
stop_mock_processes

echo "[INFO] Done..."

exit ${TEST_EXIT_CODE}