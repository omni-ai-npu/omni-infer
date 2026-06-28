#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# launch_pd.sh — launch prefill + decode + proxy from outside containers.
#
# Topology (1 node, 16 NPU cards):
#   P (prefill)  cards 0-7   TP=8  port=8000
#   D (decode)   cards 8-15  DP=8  TP=1  ports=8082-8089
#   Proxy        runs in P container, port=7150
#
# Usage:
#     bash examples/pangu_v2_pd/launch_pd.sh
#     ENABLE_OMNI_CACHE=0 bash .../launch_pd.sh                           # baseline
#     CONFIG_PROFILE=high-throughput bash .../launch_pd.sh                 # profile
#     LAUNCH_MODE=all bash .../launch_pd.sh                               # prefill + decode + proxy
#     OMNI_KV_DUMP_GEAR=step OMNI_MOCK_SCHEDULE=1 bash .../launch_pd.sh   # debug
#
# Containers (auto-discovered or overridable):
#   PREFILL_CONTAINER   default omnicache_pangu_p0
#   DECODE_CONTAINER    default omnicache_pangu_d0
#   PROXY_CONTAINER     default omnicache_pangu_p0  (reuses prefill container)
#
# Launch modes:
#   both    = prefill + decode (default)
#   all     = prefill + decode + proxy
#   prefill = prefill only
#   decode  = decode only
#   proxy   = proxy only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Load configuration ───────────────────────────────────────────────────
source "$SCRIPT_DIR/configs/base.sh"
if [[ -n "${CONFIG_PROFILE:-}" ]]; then
    profile_path="$SCRIPT_DIR/configs/${CONFIG_PROFILE}.sh"
    [[ -f "$profile_path" ]] || {
        echo "ERROR: config profile not found: $profile_path" >&2
        exit 1
    }
    source "$profile_path"
fi

# ─── Container names, model, repo paths ───────────────────────────────────
: "${PREFILL_CONTAINER:=omnicache_pangu_p0}"
: "${DECODE_CONTAINER:=omnicache_pangu_d0}"
: "${PROXY_CONTAINER:=omnicache_pangu_p0}"
: "${MODEL_PATH:=/data/models/iter_0011840}"
: "${SERVED_MODEL_NAME:=pangu_ultra_moe}"
# Auto-detect: this script lives at examples/pangu_v2_pd/ inside the repo.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${SCRIPT_DIR_CONTAINER:=$REPO_ROOT/examples/pangu_v2_pd}"

DOCKER="sudo -n docker"

# ─── Networking ───────────────────────────────────────────────────────────
_detect_ip() {
    local container="$1"
    local ip
    ip=$(sudo -n docker exec "$container" hostname -I 2>/dev/null | awk '{print $1}')
    if [[ -z "$ip" ]]; then
        ip="127.0.0.1"
    fi
    echo "$ip"
}
: "${PREFILL_IP:=$(_detect_ip "$PREFILL_CONTAINER")}"
: "${P_NODE_LIST_PREFILL:=${PREFILL_IP}}"
: "${P_NODE_LIST_DECODE:=${PREFILL_IP}}"

# ─── HCCL / deterministic ────────────────────────────────────────────────
: "${HCCL_DETERMINISTIC:=false}"
: "${ACL_OP_DETERMINISTIC:=0}"

# ─── Diagnostics (passthrough) ────────────────────────────────────────────
: "${OMNI_KV_DUMP_GEAR:=off}"
: "${OMNI_KV_DUMP_BRANCH:=}"
: "${OMNI_KV_DUMP_MAX:=0}"
: "${OMNI_MOCK_SCHEDULE:=0}"

# ─── vLLM ─────────────────────────────────────────────────────────────────
: "${VLLM_LOGGING_LEVEL:=INFO}"
: "${VLLM_WORKER_MULTIPROC_METHOD:=fork}"

# ─── Launch mode: both | prefill | decode | proxy | all ──────────────────
# "both" = prefill + decode (original behavior)
# "all"  = prefill + decode + proxy
: "${LAUNCH_MODE:=both}"

# Verify all required config vars are set (base.sh must define them).
for _v in ENABLE_OMNI_CACHE SERVED_MODEL_NAME OMNI_MOCK_SCHEDULE \
         MTP ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL \
         CHUNKED_PREFILL_TOKEN_THRESHOLD BLOCK_SIZE; do
    if [[ -z "${!_v:-}" ]]; then
        echo "ERROR: required config variable '$_v' is not set — check base.sh" >&2
        exit 1
    fi
done

echo "=== Launching PD pair ==="
echo "    profile:         ${CONFIG_PROFILE:-default}"
echo "    mode:            $( [[ $ENABLE_OMNI_CACHE == 1 ]] && echo omnicache || echo baseline )"
echo "    served model:    $SERVED_MODEL_NAME"
echo "    mock_schedule:   $OMNI_MOCK_SCHEDULE"
echo "    mtp:             $MTP"
echo "    prefix_caching:  $ENABLE_PREFIX_CACHING"
echo "    chunk_prefill:   $ENABLE_CHUNKED_PREFILL (threshold=$CHUNKED_PREFILL_TOKEN_THRESHOLD)"
echo "    block_size:      $BLOCK_SIZE"
echo "    prefill: $PREFILL_CONTAINER  (IP=$PREFILL_IP P_NODE_LIST=$P_NODE_LIST_PREFILL)"
echo "    decode:  $DECODE_CONTAINER   (P_NODE_LIST=$P_NODE_LIST_DECODE)"
echo "    launch_mode:     $LAUNCH_MODE"

# Env vars passed to both prefill and decode containers
_PASSTHROUGH_VARS=(
    SERVED_MODEL_NAME MODEL_PATH CONFIG_PROFILE
    ENABLE_OMNI_CACHE ENABLE_HOST_MAPPING
    HCCL_DETERMINISTIC ACL_OP_DETERMINISTIC
    OMNI_KV_DUMP_GEAR OMNI_KV_DUMP_BRANCH OMNI_KV_DUMP_MAX OMNI_MOCK_SCHEDULE
    VLLM_LOGGING_LEVEL VLLM_WORKER_MULTIPROC_METHOD
    MTP ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL
    CHUNKED_PREFILL_TOKEN_THRESHOLD BLOCK_SIZE ENFORCE_EAGER
    OMNI_CACHE_PACKED_HBM DISABLE_GATHER_SELECTION
    BSZ MAX_LEN OMNI_CACHE_LAYER_BYTES MAP_SIZE_BYTES
)
_DOCKER_ENV_ARGS=()
for _var in "${_PASSTHROUGH_VARS[@]}"; do
    [[ -n "${!_var:-}" ]] || continue
    _DOCKER_ENV_ARGS+=("-e" "${_var}=${!_var}")
done
_DOCKER_ENV_ARGS+=("-e" "SCRIPT_DIR_CONTAINER=${SCRIPT_DIR_CONTAINER}")

# ─── Launch prefill ───────────────────────────────────────────────────────
if [[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "prefill" ]]; then
    echo "[launch_pd] starting prefill..."
    sudo -n docker exec -d \
        "${_DOCKER_ENV_ARGS[@]}" \
        -e "P_NODE_LIST=${P_NODE_LIST_PREFILL}" \
        -e "TP_SIZE=${TP_SIZE}" \
        -e "DP_SIZE=${DP_SIZE}" \
        -e "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}" \
        "$PREFILL_CONTAINER" bash -lc '
        source ~/.bashrc
        mkdir -p "$SCRIPT_DIR_CONTAINER/logs/prefill"
        nohup bash "$SCRIPT_DIR_CONTAINER/launch_prefill.sh" &>"$SCRIPT_DIR_CONTAINER/logs/prefill/prefill_launch.log" &
    '
else
    echo "[launch_pd] skipping prefill (LAUNCH_MODE=$LAUNCH_MODE)"
fi

# ─── Launch decode ────────────────────────────────────────────────────────
if [[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "decode" ]]; then
    echo "[launch_pd] starting decode..."
    sudo -n docker exec -d \
        "${_DOCKER_ENV_ARGS[@]}" \
        -e "P_NODE_LIST=${P_NODE_LIST_DECODE}" \
        -e "DECODE_DP_SIZE=${DECODE_DP_SIZE}" \
        "$DECODE_CONTAINER" bash -lc '
        source ~/.bashrc
        mkdir -p "$SCRIPT_DIR_CONTAINER/logs/decode"
        nohup bash "$SCRIPT_DIR_CONTAINER/launch_decode.sh" &>"$SCRIPT_DIR_CONTAINER/logs/decode/decode_launch.log" &
    '
else
    echo "[launch_pd] skipping decode (LAUNCH_MODE=$LAUNCH_MODE)"
fi

# ─── Launch proxy ─────────────────────────────────────────────────────────
if [[ "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "proxy" ]]; then
    # Build prefill endpoint: PREFILL_IP:PORT
    _prefill_eps="${PREFILL_IP}:${PORT}"

    # Build decode endpoint list: PREFILL_IP:PORT_BASE .. PORT_BASE+DP-1
    _decode_eps=""
    for ((i=0; i<DECODE_DP_SIZE; i++)); do
        _port=$((PORT_BASE + i))
        [[ -n "$_decode_eps" ]] && _decode_eps+=","
        _decode_eps+="${PREFILL_IP}:${_port}"
    done

    echo "[launch_pd] starting proxy in $PROXY_CONTAINER..."
    sudo -n docker exec -d \
        "${_DOCKER_ENV_ARGS[@]}" \
        -e "PYTHONHASHSEED=123" \
        -e "PREFILL_ENDPOINTS=${_prefill_eps}" \
        -e "DECODE_ENDPOINTS=${_decode_eps}" \
        "$PROXY_CONTAINER" bash -lc '
        source ~/.bashrc
        mkdir -p "$SCRIPT_DIR_CONTAINER/logs/proxy"
        nohup bash "$SCRIPT_DIR_CONTAINER/launch_proxy.sh" &>"$SCRIPT_DIR_CONTAINER/logs/proxy/proxy_launch.log" &
    '
else
    echo "[launch_pd] skipping proxy (LAUNCH_MODE=$LAUNCH_MODE)"
fi

echo "[launch_pd] Done. Check ports:"
[[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "prefill" ]] && echo "    curl http://127.0.0.1:8000/health   (prefill)"
[[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "decode" ]]  && echo "    curl http://127.0.0.1:8082/health   (decode)"
[[ "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "proxy" ]]  && echo "    curl http://127.0.0.1:7150/omni_proxy/health   (proxy)"
