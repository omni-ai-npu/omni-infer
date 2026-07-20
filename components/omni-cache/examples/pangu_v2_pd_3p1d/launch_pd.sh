#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# launch_pd.sh — launch prefill + decode from outside containers (3P-1D).
#
# Topology (2 nodes, 16 NPU cards each):
#   Node 1 (prefill-only):  P0 (container 0, cards 0-7) + P1 (container 1, cards 8-15)
#   Node 2 (PD):            P2 (prefill container, cards 0-7) + D0 (decode container, cards 8-15)
#
# Usage:
#   On Node 1:
#     LAUNCH_MODE=prefill_2x bash .../launch_pd.sh
#
#   On Node 2 (default — same as original 1P-1D layout):
#     bash .../launch_pd.sh
#     LAUNCH_MODE=both bash .../launch_pd.sh
#
#   Additional modes:
#     LAUNCH_MODE=prefill bash .../launch_pd.sh        # single prefill only
#     LAUNCH_MODE=decode  bash .../launch_pd.sh        # decode only
#
# Containers (overridable):
#   PREFILL_CONTAINER_0  default omnicache_pangu_p0   (Node 1 P0)
#   PREFILL_CONTAINER_1  default omnicache_pangu_p1   (Node 1 P1)
#   PREFILL_CONTAINER    default omnicache_pangu_p0   (Node 2 P2, or single prefill)
#   DECODE_CONTAINER     default omnicache_pangu_d0   (Node 2 D0)

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

# ─── Container names ──────────────────────────────────────────────────────
: "${PREFILL_CONTAINER_0:=omnicache_pangu_p0}"
: "${PREFILL_CONTAINER_1:=omnicache_pangu_p1}"
: "${PREFILL_CONTAINER:=omnicache_pangu_p2}"
: "${DECODE_CONTAINER:=omnicache_pangu_d0}"
: "${PROXY_CONTAINER:=omnicache_pangu_p2}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${SCRIPT_DIR_CONTAINER:=$REPO_ROOT/examples/pangu_v2_pd_3p1d}"

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
if [[ -z "${LOCAL_NODE_IP:-}" ]]; then
    if [[ "${LAUNCH_MODE:-both}" == "prefill_2x" ]]; then
        LOCAL_NODE_IP="$(_detect_ip "$PREFILL_CONTAINER_0")"
    else
        LOCAL_NODE_IP="$(_detect_ip "$PREFILL_CONTAINER")"
    fi
fi
: "${PREFILL_NODE_IPS:=${LOCAL_NODE_IP}}"

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

: "${ENABLE_CHUNKED_PREFILL:=false}"
: "${CHUNKED_PREFILL_TOKEN_THRESHOLD:=2048}"

# ─── Launch mode: both | prefill | decode | proxy | prefill_2x | all ─────
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

echo "=== Launching 3P-1D ==="
echo "    profile:         ${CONFIG_PROFILE:-default}"
echo "    mode:            $( [[ $ENABLE_OMNI_CACHE == 1 ]] && echo omnicache || echo baseline )"
echo "    served model:    $SERVED_MODEL_NAME"
echo "    mock_schedule:   $OMNI_MOCK_SCHEDULE"
echo "    mtp:             $MTP"
echo "    prefix_caching:  $ENABLE_PREFIX_CACHING"
echo "    chunk_prefill:   $ENABLE_CHUNKED_PREFILL (threshold=$CHUNKED_PREFILL_TOKEN_THRESHOLD)"
echo "    block_size:      $BLOCK_SIZE"
echo "    num_prefill:     $NUM_PREFILL_INSTANCES"
echo "    launch_mode:     $LAUNCH_MODE"

# Build env-var passthrough for containers
PASSTHROUGH_ENV="SERVED_MODEL_NAME='$SERVED_MODEL_NAME'"
[[ -n "${CONFIG_PROFILE:-}" ]] && PASSTHROUGH_ENV+=" CONFIG_PROFILE='$CONFIG_PROFILE'"

# ─── Helper: launch a prefill instance into a container ───────────────────
_launch_prefill() {
    local container="$1"
    local prefill_rank="$2"
    local devices="$3"
    local log_suffix="${4:-prefill}"
    local base_port="${5:-$BASE_PORT}"
    local vllm_port="${6:-$PORT}"
    local kv_port="${7:-14579}"

    echo "[launch_pd] starting prefill rank=$prefill_rank in $container (devices=$devices, PORT=$vllm_port, BASE_PORT=$base_port, KV_PORT=$kv_port)..."
    sudo -n docker exec -d "$container" bash -c "
        source ~/.bashrc
        export $PASSTHROUGH_ENV
        export MODEL_PATH='$MODEL_PATH'
        export TP_SIZE='$TP_SIZE'
        export DP_SIZE='$DP_SIZE'
        export PORT='$vllm_port'
        export ASCEND_RT_VISIBLE_DEVICES='$devices'
        export ENABLE_OMNI_CACHE='$ENABLE_OMNI_CACHE'
        export ENABLE_HOST_MAPPING='$ENABLE_HOST_MAPPING'
        export P_NODE_PORT_LIST='$P_NODE_PORT_LIST'
        export BASE_PORT='$base_port'
        export KV_PORT='$kv_port'
        export HCCL_DETERMINISTIC='$HCCL_DETERMINISTIC'
        export ACL_OP_DETERMINISTIC='$ACL_OP_DETERMINISTIC'
        export OMNI_KV_DUMP_GEAR='$OMNI_KV_DUMP_GEAR'
        export OMNI_KV_DUMP_BRANCH='$OMNI_KV_DUMP_BRANCH'
        export OMNI_KV_DUMP_MAX='$OMNI_KV_DUMP_MAX'
        export OMNI_MOCK_SCHEDULE='$OMNI_MOCK_SCHEDULE'
        export VLLM_LOGGING_LEVEL='$VLLM_LOGGING_LEVEL'
        export VLLM_WORKER_MULTIPROC_METHOD='$VLLM_WORKER_MULTIPROC_METHOD'
        export MTP='$MTP'
        export ENABLE_PREFIX_CACHING='$ENABLE_PREFIX_CACHING'
        export ENABLE_CHUNKED_PREFILL='$ENABLE_CHUNKED_PREFILL'
        export CHUNKED_PREFILL_TOKEN_THRESHOLD='$CHUNKED_PREFILL_TOKEN_THRESHOLD'
        export BLOCK_SIZE='$BLOCK_SIZE'
        export ENFORCE_EAGER='$ENFORCE_EAGER'
        export OMNI_CACHE_PACKED_HBM='$OMNI_CACHE_PACKED_HBM'
        export DISABLE_GATHER_SELECTION='$DISABLE_GATHER_SELECTION'
        export BSZ='$BSZ'
        export MAX_LEN='$MAX_LEN'
        export OMNI_CACHE_LAYER_BYTES='$OMNI_CACHE_LAYER_BYTES'
        export MAP_SIZE_BYTES='$MAP_SIZE_BYTES'
        export PREFILL_RANK='$prefill_rank'
        export NUM_PREFILL_INSTANCES='$NUM_PREFILL_INSTANCES'
        mkdir -p $SCRIPT_DIR_CONTAINER/logs/$log_suffix
        nohup bash $SCRIPT_DIR_CONTAINER/launch_prefill.sh &>$SCRIPT_DIR_CONTAINER/logs/$log_suffix/prefill_launch.log &
    "
}

# ─── Node 1: prefill_2x — two prefill instances ──────────────────────────
if [[ "$LAUNCH_MODE" == "prefill_2x" ]]; then
    : "${P0_BASE_PORT:=16077}"
    : "${P1_BASE_PORT:=16078}"
    : "${P0_PORT:=8000}"
    : "${P1_PORT:=8001}"
    : "${P0_KV_PORT:=14579}"
    : "${P1_KV_PORT:=14580}"
    _launch_prefill "$PREFILL_CONTAINER_0" 0 "0,1,2,3,4,5,6,7"   "prefill_0" "$P0_BASE_PORT" "$P0_PORT" "$P0_KV_PORT"
    _launch_prefill "$PREFILL_CONTAINER_1" 1 "8,9,10,11,12,13,14,15"   "prefill_1" "$P1_BASE_PORT" "$P1_PORT" "$P1_KV_PORT"

    echo "[launch_pd] Done (prefill_2x). Check ports:"
    echo "    curl http://<node1>:$P0_PORT/health   (prefill P0)"
    echo "    curl http://<node1>:$P1_PORT/health   (prefill P1)"
    exit 0
fi

# ─── Node 2: prefill + decode (or individual) ────────────────────────────

# Launch prefill (P2, rank=2)
if [[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "prefill" ]]; then
    _launch_prefill "$PREFILL_CONTAINER" 2 "$ASCEND_RT_VISIBLE_DEVICES" "prefill_2"
else
    echo "[launch_pd] skipping prefill (LAUNCH_MODE=$LAUNCH_MODE)"
fi

# Launch decode (D0)
if [[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "decode" ]]; then
    echo "[launch_pd] starting decode..."
    sudo -n docker exec -d "$DECODE_CONTAINER" bash -c "
        source ~/.bashrc
        export $PASSTHROUGH_ENV
        export MODEL_PATH='$MODEL_PATH'
        export DECODE_DP_SIZE='$DECODE_DP_SIZE'
        export DECODE_TP_SIZE='$DECODE_TP_SIZE'
        export DEVICE_START='$DEVICE_START'
        export PORT_BASE='$PORT_BASE'
        export ENABLE_OMNI_CACHE='$ENABLE_OMNI_CACHE'
        export ENABLE_HOST_MAPPING='$ENABLE_HOST_MAPPING'
        export OMNI_CACHE_PACKED_HBM='$OMNI_CACHE_PACKED_HBM'
        export P_NODE_PORT_LIST='$P_NODE_PORT_LIST'
        export HCCL_DETERMINISTIC='$HCCL_DETERMINISTIC'
        export ACL_OP_DETERMINISTIC='$ACL_OP_DETERMINISTIC'
        export OMNI_KV_DUMP_GEAR='$OMNI_KV_DUMP_GEAR'
        export OMNI_KV_DUMP_BRANCH='$OMNI_KV_DUMP_BRANCH'
        export OMNI_KV_DUMP_MAX='$OMNI_KV_DUMP_MAX'
        export OMNI_MOCK_SCHEDULE='$OMNI_MOCK_SCHEDULE'
        export VLLM_LOGGING_LEVEL='$VLLM_LOGGING_LEVEL'
        export VLLM_WORKER_MULTIPROC_METHOD='$VLLM_WORKER_MULTIPROC_METHOD'
        export MTP='$MTP'
        export ENABLE_PREFIX_CACHING='$ENABLE_PREFIX_CACHING'
        export ENABLE_CHUNKED_PREFILL='$ENABLE_CHUNKED_PREFILL'
        export CHUNKED_PREFILL_TOKEN_THRESHOLD='$CHUNKED_PREFILL_TOKEN_THRESHOLD'
        export BLOCK_SIZE='$BLOCK_SIZE'
        export ENFORCE_EAGER='$ENFORCE_EAGER'
        export DISABLE_GATHER_SELECTION='$DISABLE_GATHER_SELECTION'
        export BSZ='$BSZ'
        export MAX_LEN='$MAX_LEN'
        export OMNI_CACHE_LAYER_BYTES='$OMNI_CACHE_LAYER_BYTES'
        export MAP_SIZE_BYTES='$MAP_SIZE_BYTES'
        export NUM_PREFILL_INSTANCES='$NUM_PREFILL_INSTANCES'
        mkdir -p $SCRIPT_DIR_CONTAINER/logs/decode
        nohup bash $SCRIPT_DIR_CONTAINER/launch_decode.sh &>$SCRIPT_DIR_CONTAINER/logs/decode/decode_launch.log &
    "
else
    echo "[launch_pd] skipping decode (LAUNCH_MODE=$LAUNCH_MODE)"
fi

# Launch proxy
if [[ "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "proxy" ]]; then
    # PREFILL_NODE_IPS is semicolon-separated: "ip_p0;ip_p1;ip_p2".
    IFS=';' read -ra _p_ips <<< "$PREFILL_NODE_IPS"
    : "${P0_PORT:=8000}"
    : "${P1_PORT:=8001}"
    _p2_ip="${_p_ips[2]:-$LOCAL_NODE_IP}"
    _p0_ip="${_p_ips[0]:-$LOCAL_NODE_IP}"
    _p1_ip="${_p_ips[1]:-$LOCAL_NODE_IP}"
    _prefill_eps="${_p2_ip}:${PORT},${_p0_ip}:${P0_PORT},${_p1_ip}:${P1_PORT}"

    # Build decode endpoint list: LOCAL_NODE_IP:PORT_BASE .. PORT_BASE+DP-1
    _decode_eps=""
    for ((i=0; i<DECODE_DP_SIZE; i++)); do
        _port=$((PORT_BASE + i))
        [[ -n "$_decode_eps" ]] && _decode_eps+=","
        _decode_eps+="${LOCAL_NODE_IP}:${_port}"
    done

    echo "[launch_pd] starting proxy in $PROXY_CONTAINER..."
    sudo -n docker exec -d "$PROXY_CONTAINER" bash -c "
        source ~/.bashrc
        export PYTHONHASHSEED=123
        export $PASSTHROUGH_ENV
        export MODEL_PATH='$MODEL_PATH'
        export PREFILL_ENDPOINTS='$_prefill_eps'
        export DECODE_ENDPOINTS='$_decode_eps'
        mkdir -p $SCRIPT_DIR_CONTAINER/logs/proxy
        nohup bash $SCRIPT_DIR_CONTAINER/launch_proxy.sh &>$SCRIPT_DIR_CONTAINER/logs/proxy/proxy_launch.log &
    "
else
    echo "[launch_pd] skipping proxy (LAUNCH_MODE=$LAUNCH_MODE)"
fi

echo "[launch_pd] Done. Check ports:"
[[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "prefill" ]] && echo "    curl http://127.0.0.1:8000/health   (prefill P2)"
[[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "decode" ]]  && echo "    curl http://127.0.0.1:8082/health   (decode D0)"
[[ "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "proxy" ]]  && echo "    curl http://127.0.0.1:7150/omni_proxy/health   (proxy)"
