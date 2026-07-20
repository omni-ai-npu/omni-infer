#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# launch_decode.sh — bring up the decode side of a Pangu V2 hybrid PD
# deployment. Decode is data-parallel: this script starts DECODE_DP_SIZE
# independent vLLM instances, each bound to a disjoint set of NPU dies.
#
# When ENABLE_OMNI_CACHE=0 (baseline): uses vLLM's LLMDataDistConnector.
# When ENABLE_OMNI_CACHE=1 (default): uses OmniCacheConnector with
# host-backed KV pool via hugetlbfs.
#
# Usage:
#     bash launch_decode.sh                               # OmniCache, DP=8
#     ENABLE_OMNI_CACHE=0 bash launch_decode.sh           # baseline mode
#     ENABLE_HOST_MAPPING=0 bash launch_decode.sh         # HM=0 path
#     CONFIG_PROFILE=high-throughput bash launch_decode.sh
#     DECODE_DP_SIZE=4 bash launch_decode.sh              # ad-hoc overrides
#
# Configuration: defaults are in configs/base.sh; set CONFIG_PROFILE to
# load an optional override file (e.g. configs/high-throughput.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ─── Load configuration ───────────────────────────────────────────────────
export ROLE=decode
source "$SCRIPT_DIR/configs/base.sh"
if [[ -n "${CONFIG_PROFILE:-}" ]]; then
    profile_path="$SCRIPT_DIR/configs/${CONFIG_PROFILE}.sh"
    [[ -f "$profile_path" ]] || {
        echo "ERROR: config profile not found: $profile_path" >&2
        exit 1
    }
    source "$profile_path"
    echo "[launch_decode] loaded config profile: $CONFIG_PROFILE"
fi

# Verify all required config vars are set (base.sh must define them).
for _v in MODEL_PATH SERVED_MODEL_NAME DECODE_DP_SIZE \
         BSZ MAX_LEN BLOCK_SIZE ENFORCE_EAGER \
         ENABLE_OMNI_CACHE MTP ENABLE_PREFIX_CACHING; do
    [[ -n "${!_v:-}" ]] || {
        echo "ERROR: required config variable '$_v' is not set" >&2
        exit 1
    }
done

# ─── Derived defaults (D-specific, not in config) ────────────────────────
: "${OMNI_CACHE_MMAP_FILE:=omni_cache_d}"
: "${OMNI_CACHE_MMAP_PATH:=/dev/hugepages/${OMNI_CACHE_MMAP_FILE}}"
: "${OMNI_CACHE_DSA_MMAP_FILE:=omni_cache_decode_dsa}"
: "${OMNI_CACHE_DSA_MMAP_PATH:=/dev/hugepages/${OMNI_CACHE_DSA_MMAP_FILE}}"
: "${OMNI_CACHE_DSA_MAP_SIZE_BYTES:=$(( MAP_SIZE_BYTES * 80 / 100 ))}"
if [[ "$ENABLE_OMNI_CACHE" == 1 ]]; then
    : "${ENABLE_HOST_MAPPING:=1}"
    : "${NUM_GPU_BLOCKS_OVERRIDE:=11800}"
    : "${KV_CACHE_MEMORY_BYTES:=OMNI_CACHE_LAYER_BYTES * HYBRID_ATTN_GROUP_SIZE}"
else
    export ENABLE_OMNI_CACHE=0
    export ENABLE_HOST_MAPPING=0
fi

SETUP_HUGETLBFS_SH="$REPO_ROOT/tools/setup/setup_hugetlbfs_2MB.sh"
: "${LOG_DIR:=$SCRIPT_DIR/logs/decode}"

export BASE_PORT="${BASE_PORT:-16077}"
export ZMQ_BASE_PORT="${ZMQ_BASE_PORT:-16555}"

# ─── Export env to every vLLM child ───────────────────────────────────────
export \
    ENABLE_OMNI_CACHE ENABLE_HOST_MAPPING HYBRID_ATTN_GROUP_SIZE OMNI_CACHE_LOCAL_DP_SIZE \
    OMNI_CACHE_MLA_SWA_DEBUG OMNI_CACHE_ATTN_PLUGINS \
    OMNI_REUSE_PREFILLED_TOKENS OMNI_SKIP_DECODE_TOKENIZE \
    OMNI_KV_DUMP_GEAR OMNI_KV_DUMP_DIR OMNI_KV_DUMP_BRANCH OMNI_KV_DUMP_MAX \
    MTP ENABLE_PREFIX_CACHING BLOCK_SIZE VLLM_PLUGINS OMNI_NPU_PATCHES_DIR OMNI_NPU_VLLM_PATCHES \
    VLLM_LOGGING_LEVEL VLLM_WORKER_MULTIPROC_METHOD \
    TP_NNODES \
    ASCEND_GLOBAL_LOG_LEVEL HCCL_OP_EXPANSION_MODE \
    HCCL_INTRA_ROCE_ENABLE HCCL_INTRA_PCIE_ENABLE HCCL_BUFFSIZE \
    GLOO_SOCKET_IFNAME HCCL_SOCKET_IFNAME \
    DISABLE_GATHER_SELECTION \
    ROLE
if [[ -n "$P_NODE_PORT_LIST" ]]; then
    export P_NODE_PORT_LIST
fi
if [[ "$ENABLE_OMNI_CACHE" == 1 ]]; then
    export \
        OMNI_CACHE_MMAP_PATH OMNI_CACHE_LAYER_BYTES \
        ENABLE_OMNI_CACHE_DSA_SPLIT OMNI_CACHE_DSA_MMAP_PATH
fi
export CURRENT_TIME_STAMP="$(date '+%m%d_%H%M')"

mkdir -p "$LOG_DIR"

# ─── Sanity ───────────────────────────────────────────────────────────────
[[ -d "$MODEL_PATH" ]] \
    || { echo "ERROR: MODEL_PATH not a directory: $MODEL_PATH" | tee -a "$LOG_DIR/launch_error.log" >&2; exit 1; }
command -v vllm >/dev/null \
    || { echo "ERROR: vllm not on PATH" | tee -a "$LOG_DIR/launch_error.log" >&2; exit 1; }

# ─── Dump launch configuration ────────────────────────────────────────────
{
    echo "======================================================================"
    echo "Launch Configuration Dump"
    echo "======================================================================"
    echo "timestamp:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "hostname:        $(hostname)"
    echo "script:          launch_decode.sh"
    echo "role:            decode"
    echo "config_profile:  ${CONFIG_PROFILE:-default}"
    echo "omni_mode:       $( [[ $ENABLE_OMNI_CACHE == 1 ]] && echo omnicache || echo baseline )"
    echo ""
    echo "--- Serving ---"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
    echo "PORT_BASE=${PORT_BASE}"
    echo "DECODE_TP_SIZE=${DECODE_TP_SIZE}"
    echo "DECODE_DP_SIZE=${DECODE_DP_SIZE}"
    echo "DEVICE_START=${DEVICE_START}"
    echo "BSZ=${BSZ}"
    echo "MAX_LEN=${MAX_LEN}"
    echo "BLOCK_SIZE=${BLOCK_SIZE}"
    echo "DIST_BACKEND=${DIST_BACKEND}"
    echo "ENFORCE_EAGER=${ENFORCE_EAGER}"
    echo "EXPERT_PARALLEL=${EXPERT_PARALLEL}"
    echo ""
    echo "--- OmniCache ---"
    echo "ENABLE_OMNI_CACHE=${ENABLE_OMNI_CACHE}"
    echo "ENABLE_HOST_MAPPING=${ENABLE_HOST_MAPPING}"
    echo "HYBRID_ATTN_GROUP_SIZE=${HYBRID_ATTN_GROUP_SIZE}"
    echo "OMNI_CACHE_LOCAL_DP_SIZE=${OMNI_CACHE_LOCAL_DP_SIZE}"
    echo "OMNI_CACHE_MLA_SWA_DEBUG=${OMNI_CACHE_MLA_SWA_DEBUG}"
    echo "OMNI_CACHE_ATTN_PLUGINS=${OMNI_CACHE_ATTN_PLUGINS}"
    echo "OMNI_REUSE_PREFILLED_TOKENS=${OMNI_REUSE_PREFILLED_TOKENS}"
    echo "OMNI_SKIP_DECODE_TOKENIZE=${OMNI_SKIP_DECODE_TOKENIZE}"
    echo "DISABLE_GATHER_SELECTION=${DISABLE_GATHER_SELECTION}"
    echo "OMNI_CACHE_MMAP_FILE=${OMNI_CACHE_MMAP_FILE}"
    echo "OMNI_CACHE_MMAP_PATH=${OMNI_CACHE_MMAP_PATH}"
    echo "OMNI_CACHE_LAYER_BYTES=${OMNI_CACHE_LAYER_BYTES}"
    echo "MAP_SIZE_BYTES=${MAP_SIZE_BYTES}"
    echo "NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-N/A}"
    echo "ENABLE_OMNI_CACHE_DSA_SPLIT=${ENABLE_OMNI_CACHE_DSA_SPLIT}"
    echo "OMNI_CACHE_DSA_MMAP_PATH=${OMNI_CACHE_DSA_MMAP_PATH}"
    echo "OMNI_CACHE_DSA_MAP_SIZE_BYTES=${OMNI_CACHE_DSA_MAP_SIZE_BYTES}"
    echo "OMNI_KV_DUMP_GEAR=${OMNI_KV_DUMP_GEAR}"
    echo "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
    echo ""
    echo "--- vLLM / plugins ---"
    echo "VLLM_PLUGINS=${VLLM_PLUGINS}"
    echo "OMNI_NPU_PATCHES_DIR=${OMNI_NPU_PATCHES_DIR}"
    echo "OMNI_NPU_VLLM_PATCHES=${OMNI_NPU_VLLM_PATCHES}"
    echo "VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}"
    echo "MTP=${MTP}"
    echo "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
    echo ""
    echo "--- Networking ---"
    echo "BASE_PORT=${BASE_PORT}"
    echo "ZMQ_BASE_PORT=${ZMQ_BASE_PORT}"
    echo "TP_NNODES=${TP_NNODES}"
    echo "ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL}"
    echo "HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE}"
    echo "HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE}"
    echo "HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE}"
    echo "HCCL_BUFFSIZE=${HCCL_BUFFSIZE}"
    echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
    echo "HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}"
    echo ""
    echo "--- Environment ---"
    env | sort
} > "$LOG_DIR/launch_config.log"
echo "[launch_decode] config dumped to: $LOG_DIR/launch_config.log"

# ─── Reserve hugepages (only needed by OmniCache) ─────────────────────────
if [[ "$ENABLE_OMNI_CACHE" == 1 ]]; then
    [[ -f "$SETUP_HUGETLBFS_SH" ]] \
        || { echo "ERROR: missing $SETUP_HUGETLBFS_SH" >&2; exit 1; }
    echo "[launch_decode] reserving hugetlbfs (file=$OMNI_CACHE_MMAP_FILE size=${MAP_SIZE_BYTES}B)"
    MAP_SIZE_BYTES="$MAP_SIZE_BYTES" OMNI_FILE="$OMNI_CACHE_MMAP_FILE" \
        bash "$SETUP_HUGETLBFS_SH"
else
    echo "[launch_decode] ENABLE_OMNI_CACHE=0 — using LLMDataDistConnector (no hugetlbfs reservation)"
fi

if [[ "$ENABLE_OMNI_CACHE" == 1 && "$ENABLE_OMNI_CACHE_DSA_SPLIT" == 1 ]]; then
    PRIMARY_PAGES=$(( (MAP_SIZE_BYTES + (2 * 1024 * 1024) - 1) / (2 * 1024 * 1024) ))
    DSA_PAGES=$(( (OMNI_CACHE_DSA_MAP_SIZE_BYTES + (2 * 1024 * 1024) - 1) / (2 * 1024 * 1024) ))
    DSA_TOTAL_PAGES=$(( PRIMARY_PAGES + DSA_PAGES ))
    echo "[launch_decode] reserving DSA hugetlbfs (file=$OMNI_CACHE_DSA_MMAP_FILE size=${OMNI_CACHE_DSA_MAP_SIZE_BYTES}B)"
    MAP_SIZE_BYTES="$OMNI_CACHE_DSA_MAP_SIZE_BYTES" OMNI_FILE="$OMNI_CACHE_DSA_MMAP_FILE" \
        bash "$SETUP_HUGETLBFS_SH" "$DSA_TOTAL_PAGES"
fi

# KV connector
if [[ "${ENABLE_OMNI_CACHE}" == "1" ]]; then
    KV_CONNECTOR="OmniCacheConnector"
else
    KV_CONNECTOR="LLMDataDistConnector"
fi

# ─── Fan out DP instances ─────────────────────────────────────────────────
DECODE_PIDS=()

cleanup() {
    local status=$?
    echo "[launch_decode] shutting down ${#DECODE_PIDS[@]} DP instances"
    for pid in "${DECODE_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait "${DECODE_PIDS[@]}" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

FORCE_EAGER=$ENFORCE_EAGER
echo "[launch_decode] mode=$( [[ $ENABLE_OMNI_CACHE == 1 ]] && echo omnicache || echo baseline )"
echo "[launch_decode] profile=${CONFIG_PROFILE:-default}"
echo "[launch_decode] DP=$DECODE_DP_SIZE TP=$DECODE_TP_SIZE devices start at die $DEVICE_START"
echo "[launch_decode] ENABLE_HOST_MAPPING=$ENABLE_HOST_MAPPING model=$MODEL_PATH"
echo "[launch_decode] logs under $LOG_DIR/decode_{0..$((DECODE_DP_SIZE-1))}.log"

for ((rank=0; rank<DECODE_DP_SIZE; rank++)); do
    port=$((PORT_BASE + rank))
    first_die=$((DEVICE_START + rank * DECODE_TP_SIZE))
    devices=""
    for ((i=0; i<DECODE_TP_SIZE; i++)); do
        [[ -n "$devices" ]] && devices+=","
        devices+="$((first_die + i))"
    done

    if [[ "${ENABLE_OMNI_CACHE}" == "1" ]]; then
        kv_psize=$((NUM_PREFILL_INSTANCES + DECODE_DP_SIZE))
    else
        kv_psize=1
    fi
    kv_conf_base='{"kv_connector":"%s","kv_role":"kv_consumer","kv_rank":%d,"kv_parallel_size":%d,"kv_connector_extra_config":{"kv_producer_dp_size":%d}}'
    kv_conf=$(printf "$kv_conf_base" \
        "$KV_CONNECTOR" "$((NUM_PREFILL_INSTANCES + rank))" "$kv_psize" 1)

    VLLM_CMD=(
        vllm serve "$MODEL_PATH"
        --served-model-name "$SERVED_MODEL_NAME"
        --host 0.0.0.0
        --port "$port"
        --dtype bfloat16
        --max-model-len "$MAX_LEN"
        --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-$((BSZ * (MTP + 1)))}"
        --max-num-seqs "$BSZ"
        --distributed-executor-backend "$DIST_BACKEND"
        --gpu-memory-utilization 0.88
        --trust-remote-code
        --tensor-parallel-size "$DECODE_TP_SIZE"
        --data-parallel-size "$DECODE_DP_SIZE"
        --data-parallel-rank "$rank"
        --no-disable-hybrid-kv-cache-manager
        --kv-transfer-config "$kv_conf"
        --reasoning-parser pangu 
        --enable-auto-tool-choice
        --tool-call-parser pangu
        --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
        --additional-config '{"enable_low_latency": true, "npugraph_ex_config" :{"enable": true, "super_kernel_optimze": false, "static_kernel_compile": true}}'
    )
    if [[ "${MTP}" -ge "1" ]]; then
        spec_conf=$(printf '{"num_speculative_tokens": %d, "method": "deepseek_mtp"}' "$MTP")
        VLLM_CMD+=(--speculative-config "$spec_conf")
    fi
    # Prefix caching (APC)
    if [[ "${ENABLE_PREFIX_CACHING}" == "1" && "${ENABLE_OMNI_CACHE}" != "1" ]]; then
        VLLM_CMD+=(--enable-prefix-caching)
        VLLM_CMD+=(--block-size "$BLOCK_SIZE")
    else
        VLLM_CMD+=(--no-enable-prefix-caching)
    fi
    # Chunked prefill
    if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
        VLLM_CMD+=(--enable-chunked-prefill)
        if [[ "${CHUNKED_PREFILL_TOKEN_THRESHOLD}" -gt "0" ]]; then
            VLLM_CMD+=(--long-prefill-token-threshold "$CHUNKED_PREFILL_TOKEN_THRESHOLD")
        fi
    else
        VLLM_CMD+=(--no-enable-chunked-prefill)
    fi

    # if [[ "$ENABLE_OMNI_CACHE" == 1 ]]; then
    #     VLLM_CMD+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
    #     VLLM_CMD+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
    # fi
    if [[ "$FORCE_EAGER" == 1 ]]; then
        VLLM_CMD+=(--enforce-eager)
    else
        _interval=1
        [[ "${MTP}" -ge 1 ]] && _interval=4
        _cap_size=$((BSZ * _interval))
        _comp_cfg='{"level":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":['"$_cap_size"'],"backend":"eager","compile_sizes":[4,8,32]}'
        VLLM_CMD+=(--compilation-config "$_comp_cfg")
    fi
    [[ "$EXPERT_PARALLEL" == 1 ]] && VLLM_CMD+=(--enable-expert-parallel)

    echo "[launch_decode] rank=$rank port=$port devices=$devices"
    echo "ASCEND_RT_VISIBLE_DEVICES=$devices ${VLLM_CMD[@]}" > "$LOG_DIR/vllm_cmd_rank${rank}.log"
    ASCEND_RT_VISIBLE_DEVICES="$devices" \
        "${VLLM_CMD[@]}" &> "$LOG_DIR/decode_${rank}.log" &
    DECODE_PIDS+=("$!")
    sleep 2
done

echo "[launch_decode] started ${#DECODE_PIDS[@]} DP instances; PIDs=${DECODE_PIDS[*]}"
wait "${DECODE_PIDS[@]}"
