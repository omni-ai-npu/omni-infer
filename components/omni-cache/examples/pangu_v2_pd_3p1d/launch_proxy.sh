#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# launch_proxy.sh — build (if needed) and launch the omni-proxy for a
# Pangu V2 hybrid 3P-1D deployment.
#
# Runs inside the P2 container (omnicache_pangu_p2) on Node 2.
# The proxy uses the omni-proxy code at
# /workspace/omniinfer/components/omni-proxy/.
#
# The caller (launch_pd.sh or manual) must supply PREFILL_ENDPOINTS and
# DECODE_ENDPOINTS as comma-separated ip:port lists.
#
# Usage:
#     PREFILL_ENDPOINTS=10.0.0.1:8000,10.0.0.2:8000,10.0.0.2:8001 \
#     DECODE_ENDPOINTS=10.0.0.1:8082,10.0.0.1:8083,...              \
#       bash launch_proxy.sh
#
#     CONFIG_PROFILE=high-throughput bash launch_proxy.sh
#     REBUILD=1 bash launch_proxy.sh               # force rebuild

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ─── PYTHONHASHSEED must match vLLM for APC block-hash matching ─────────
: "${PYTHONHASHSEED:=123}"
export PYTHONHASHSEED

# ─── Load configuration ───────────────────────────────────────────────────
source "$SCRIPT_DIR/configs/base.sh"
if [[ -n "${CONFIG_PROFILE:-}" ]]; then
    profile_path="$SCRIPT_DIR/configs/${CONFIG_PROFILE}.sh"
    [[ -f "$profile_path" ]] || {
        echo "ERROR: config profile not found: $profile_path" >&2
        exit 1
    }
    source "$profile_path"
    echo "[launch_proxy] loaded config profile: $CONFIG_PROFILE"
fi

# ─── Endpoints (must be provided by caller) ──────────────────────────────
if [[ -z "${PREFILL_ENDPOINTS:-}" ]]; then
    echo "ERROR: PREFILL_ENDPOINTS is required (comma-separated ip:port list)" >&2
    exit 1
fi
if [[ -z "${DECODE_ENDPOINTS:-}" ]]; then
    echo "ERROR: DECODE_ENDPOINTS is required (comma-separated ip:port list)" >&2
    exit 1
fi

# ─── Proxy-specific defaults ─────────────────────────────────────────────
: "${PROXY_LISTEN_PORT:=7150}"
: "${PROXY_CORE_NUM:=4}"
: "${PROXY_START_CORE_INDEX:=16}"
: "${PROXY_LOG_LEVEL:=info}"
: "${PROXY_PD_POLICY:=sequential}"
: "${PROXY_MAX_BATCH_NUM_TOKEN:=512000}"
: "${PROXY_PREFILL_MAX_NUM_SEQS:=2}"
: "${PROXY_DECODE_MAX_NUM_SEQS:=4}"
: "${PROXY_PREFILL_STARVATION_TIMEOUT:=400}"
: "${PROXY_SCHEDULE_ALGO:=default}"
: "${PROXY_STREAM_OPS:=off}"
: "${PROXY_PREFILL_POD_SIZE:=1}"
: "${PROXY_DECODE_POD_SIZE:=1}"

: "${REBUILD:=0}"

# ─── Paths inside the container ──────────────────────────────────────────
OMNI_PROXY_DIR="/workspace/omniinfer/components/omni-proxy"
OMNI_PROXY_SH="${OMNI_PROXY_DIR}/omni_proxy/omni_proxy.sh"
BUILD_SH="${OMNI_PROXY_DIR}/omni_proxy/build.sh"

: "${LOG_DIR:=$SCRIPT_DIR/logs/proxy}"
PROXY_LOG_FILE="${LOG_DIR}/nginx_error.log"
PROXY_ACCESS_LOG_FILE="${LOG_DIR}/nginx_access.log"

mkdir -p "$LOG_DIR"

# ─── Dump launch configuration ──────────────────────────────────────────
{
    echo "======================================================================"
    echo "Proxy Launch Configuration"
    echo "======================================================================"
    echo "timestamp:         $(date '+%Y-%m-%d %H:%M:%S')"
    echo "hostname:          $(hostname)"
    echo "config_profile:    ${CONFIG_PROFILE:-default}"
    echo ""
    echo "--- Proxy ---"
    echo "PROXY_LISTEN_PORT=${PROXY_LISTEN_PORT}"
    echo "PROXY_CORE_NUM=${PROXY_CORE_NUM}"
    echo "PROXY_START_CORE_INDEX=${PROXY_START_CORE_INDEX}"
    echo "PROXY_LOG_LEVEL=${PROXY_LOG_LEVEL}"
    echo "PROXY_PD_POLICY=${PROXY_PD_POLICY}"
    echo "PROXY_MAX_BATCH_NUM_TOKEN=${PROXY_MAX_BATCH_NUM_TOKEN}"
    echo "PROXY_PREFILL_MAX_NUM_SEQS=${PROXY_PREFILL_MAX_NUM_SEQS}"
    echo "PROXY_DECODE_MAX_NUM_SEQS=${PROXY_DECODE_MAX_NUM_SEQS}"
    echo "PROXY_SCHEDULE_ALGO=${PROXY_SCHEDULE_ALGO}"
    echo "PROXY_STREAM_OPS=${PROXY_STREAM_OPS}"
    echo ""
    echo "--- Model ---"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "PYTHONHASHSEED=${PYTHONHASHSEED}"
    echo ""
    echo "--- Endpoints ---"
    echo "PREFILL_ENDPOINTS=${PREFILL_ENDPOINTS}"
    echo "DECODE_ENDPOINTS=${DECODE_ENDPOINTS}"
    echo ""
    echo "--- Logs ---"
    echo "PROXY_LOG_FILE=${PROXY_LOG_FILE}"
    echo "PROXY_ACCESS_LOG_FILE=${PROXY_ACCESS_LOG_FILE}"
} > "$LOG_DIR/launch_config.log"
echo "[launch_proxy] config dumped to: $LOG_DIR/launch_config.log"

# ─── Build (if needed) ──────────────────────────────────────────────────
if [[ "$REBUILD" == "1" ]] || ! command -v nginx &>/dev/null; then
    echo "[launch_proxy] building omni-proxy..."
    cd "${OMNI_PROXY_DIR}"
    bash "${BUILD_SH}" --skip-extras
    cd "$SCRIPT_DIR"
    echo "[launch_proxy] build complete"
else
    echo "[launch_proxy] nginx binary found, skipping build (set REBUILD=1 to force)"
fi

# ─── Launch ──────────────────────────────────────────────────────────────
PROXY_CMD=(
    bash "$OMNI_PROXY_SH"
    --listen-port "$PROXY_LISTEN_PORT"
    --prefill-endpoints "$PREFILL_ENDPOINTS"
    --decode-endpoints "$DECODE_ENDPOINTS"
    --log-file "$PROXY_LOG_FILE"
    --log-level "$PROXY_LOG_LEVEL"
    --access-log-file "$PROXY_ACCESS_LOG_FILE"
    --core-num "$PROXY_CORE_NUM"
    --start-core-index "$PROXY_START_CORE_INDEX"
    --omni-proxy-pd-policy "$PROXY_PD_POLICY"
    --omni-proxy-model-path "$MODEL_PATH"
    --omni-proxy-max-batch-num-token "$PROXY_MAX_BATCH_NUM_TOKEN"
    --omni-proxy-prefill-max-num-seqs "$PROXY_PREFILL_MAX_NUM_SEQS"
    --omni-proxy-decode-max-num-seqs "$PROXY_DECODE_MAX_NUM_SEQS"
    --omni-proxy-prefill-starvation-timeout "$PROXY_PREFILL_STARVATION_TIMEOUT"
    --omni-proxy-schedule-algo "$PROXY_SCHEDULE_ALGO"
    --stream-ops "$PROXY_STREAM_OPS"
    --prefill-pod-size "$PROXY_PREFILL_POD_SIZE"
    --decode-pod-size "$PROXY_DECODE_POD_SIZE"
    --subrequest-output-buffer-size 10M
)

echo "[launch_proxy] starting omni-proxy on port $PROXY_LISTEN_PORT"
echo "[launch_proxy] prefill: $PREFILL_ENDPOINTS"
echo "[launch_proxy] decode:  $DECODE_ENDPOINTS"
echo "[launch_proxy] model:   $MODEL_PATH"
echo "[launch_proxy] error log:  $PROXY_LOG_FILE"
echo "[launch_proxy] access log: $PROXY_ACCESS_LOG_FILE"
echo "${PROXY_CMD[*]}" > "$LOG_DIR/proxy_cmd.log"
echo "[launch_proxy] proxy command saved to: $LOG_DIR/proxy_cmd.log"

exec "${PROXY_CMD[@]}"
