#!/usr/bin/env bash
#
# launch_pd.sh — launch prefill + decode from outside containers.
#
# Usage:
#     bash examples/pangu_v2_pd/launch_pd.sh
#     ENABLE_OMNI_CACHE=0 bash .../launch_pd.sh                           # baseline
#     CONFIG_PROFILE=high-throughput bash .../launch_pd.sh                 # profile
#     OMNI_KV_DUMP_GEAR=step OMNI_MOCK_SCHEDULE=1 bash .../launch_pd.sh   # debug
#
# Containers (auto-discovered or overridable):
#   PREFILL_CONTAINER   default yyx-container-p
#   DECODE_CONTAINER    default yyx-container-d

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
: "${PREFILL_CONTAINER:=ldy-92B-prefill}"
: "${DECODE_CONTAINER:=ldy-92B-decode}"
: "${MODEL_PATH:=/data/models/iter_0011840}"
: "${SERVED_MODEL_NAME:=pangu_ultra_moe}"
# Auto-detect: this script lives at examples/pangu_v2_pd/ inside the repo.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${SCRIPT_DIR_CONTAINER:=$REPO_ROOT/examples/pangu_v2_pd}"

# : "${SCRIPT_DIR_CONTAINER:=/data/ldy/92B_service/pangu_v2_pd}"

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

# ─── Launch mode: both | prefill | decode ────────────────────────────────
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
_PASSTHROUGH_ENV=""
for _var in "${_PASSTHROUGH_VARS[@]}"; do
    [[ -n "${!_var:-}" ]] || continue
    _PASSTHROUGH_ENV+=" ${_var}='${!_var}'"
done

# ─── Launch prefill ───────────────────────────────────────────────────────
if [[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "prefill" ]]; then
    echo "[launch_pd] starting prefill..."
    sudo -n docker exec -d "$PREFILL_CONTAINER" bash -c "
        source ~/.bashrc
        export $_PASSTHROUGH_ENV
        export P_NODE_LIST='$P_NODE_LIST_PREFILL'
        export TP_SIZE='$TP_SIZE'
        export DP_SIZE='$DP_SIZE'
        export ASCEND_RT_VISIBLE_DEVICES='$ASCEND_RT_VISIBLE_DEVICES'
        mkdir -p $SCRIPT_DIR_CONTAINER/logs/prefill
        nohup bash $SCRIPT_DIR_CONTAINER/launch_prefill.sh &>$SCRIPT_DIR_CONTAINER/logs/prefill/prefill_launch.log &
    "
else
    echo "[launch_pd] skipping prefill (LAUNCH_MODE=$LAUNCH_MODE)"
fi

# ─── Launch decode ────────────────────────────────────────────────────────
if [[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "decode" ]]; then
    echo "[launch_pd] starting decode..."
    sudo -n docker exec -d "$DECODE_CONTAINER" bash -c "
        source ~/.bashrc
        export $_PASSTHROUGH_ENV
        export P_NODE_LIST='$P_NODE_LIST_DECODE'
        export DECODE_DP_SIZE='$DECODE_DP_SIZE'
        mkdir -p $SCRIPT_DIR_CONTAINER/logs/decode
        nohup bash $SCRIPT_DIR_CONTAINER/launch_decode.sh &>$SCRIPT_DIR_CONTAINER/logs/decode/decode_launch.log &
    "
else
    echo "[launch_pd] skipping decode (LAUNCH_MODE=$LAUNCH_MODE)"
fi

echo "[launch_pd] Done. Check ports:"
[[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "prefill" ]] && echo "    curl http://127.0.0.1:8000/health   (prefill)"
[[ "$LAUNCH_MODE" == "both" || "$LAUNCH_MODE" == "decode" ]]  && echo "    curl http://127.0.0.1:8082/health   (decode)"
