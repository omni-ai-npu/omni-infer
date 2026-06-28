#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# launch_prefill.sh — bring up the prefill side of a Pangu V2 hybrid PD
# deployment.
#
# When ENABLE_OMNI_CACHE=0 (baseline): uses vLLM's LLMDataDistConnector.
# When ENABLE_OMNI_CACHE=1 (default): uses OmniCacheConnector with
# host-backed KV transfer via hugetlbfs.
#
# Usage:
#     bash launch_prefill.sh
#     ENABLE_OMNI_CACHE=0 bash launch_prefill.sh          # baseline mode
#     CONFIG_PROFILE=high-throughput bash launch_prefill.sh
#     PORT=8100 TP_SIZE=4 bash launch_prefill.sh          # ad-hoc overrides
#
# Configuration: defaults are in configs/base.sh; set CONFIG_PROFILE to
# load an optional override file (e.g. configs/high-throughput.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ─── Load configuration ───────────────────────────────────────────────────
export ROLE=prefill
source "$SCRIPT_DIR/configs/base.sh"
if [[ -n "${CONFIG_PROFILE:-}" ]]; then
    profile_path="$SCRIPT_DIR/configs/${CONFIG_PROFILE}.sh"
    [[ -f "$profile_path" ]] || {
        echo "ERROR: config profile not found: $profile_path" >&2
        exit 1
    }
    source "$profile_path"
    echo "[launch_prefill] loaded config profile: $CONFIG_PROFILE"
fi

# Verify all required config vars are set (base.sh must define them).
for _v in MODEL_PATH SERVED_MODEL_NAME TP_SIZE DP_SIZE \
         BSZ MAX_LEN BLOCK_SIZE ENFORCE_EAGER \
         ENABLE_OMNI_CACHE MTP ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL \
         CHUNKED_PREFILL_TOKEN_THRESHOLD; do
    [[ -n "${!_v:-}" ]] || {
        echo "ERROR: required config variable '$_v' is not set" >&2
        exit 1
    }
done

# ─── Derived defaults (P-specific, not in config) ────────────────────────
: "${KV_PARALLEL_SIZE:=1}"
: "${OMNI_CACHE_MMAP_FILE:=omni_cache_p}"
: "${OMNI_CACHE_MMAP_PATH:=/dev/hugepages/${OMNI_CACHE_MMAP_FILE}}"
export ENABLE_HOST_MAPPING=0
ENFORCE_EAGER=1
if [[ "$ENABLE_OMNI_CACHE" == 1 ]]; then
    : "${NUM_GPU_BLOCKS_OVERRIDE:=50000}"
    : "${KV_CACHE_MEMORY_BYTES:=OMNI_CACHE_LAYER_BYTES * HYBRID_ATTN_GROUP_SIZE}"
    : "${OMNI_CACHE_PACKED_HBM:=1}"
    : "${P_NODE_LIST:=127.0.0.1}"
else
    export ENABLE_OMNI_CACHE=0
    : "${P_NODE_LIST:=192.168.0.12}"
fi

SETUP_HUGETLBFS_SH="$REPO_ROOT/tools/setup/setup_hugetlbfs_2MB.sh"
: "${LOG_DIR:=$SCRIPT_DIR/logs/prefill}"

export BASE_PORT=16077
export ZMQ_BASE_PORT=16555

# ─── Export environment for vLLM child ────────────────────────────────────
export \
    ENABLE_OMNI_CACHE HYBRID_ATTN_GROUP_SIZE OMNI_CACHE_LOCAL_DP_SIZE \
    OMNI_CACHE_MLA_SWA_DEBUG OMNI_CACHE_ATTN_PLUGINS \
    OMNI_REUSE_PREFILLED_TOKENS OMNI_SKIP_DECODE_TOKENIZE \
    VLLM_PLUGINS OMNI_NPU_PATCHES_DIR OMNI_NPU_VLLM_PATCHES VLLM_LOGGING_LEVEL \
    MTP ENABLE_PREFIX_CACHING ENABLE_CHUNKED_PREFILL CHUNKED_PREFILL_TOKEN_THRESHOLD \
    ASCEND_RT_VISIBLE_DEVICES P_NODE_LIST TP_NNODES \
    ASCEND_GLOBAL_LOG_LEVEL HCCL_OP_EXPANSION_MODE \
    HCCL_INTRA_ROCE_ENABLE HCCL_INTRA_PCIE_ENABLE HCCL_BUFFSIZE \
    GLOO_SOCKET_IFNAME HCCL_SOCKET_IFNAME \
    ROLE
if [[ "$ENABLE_OMNI_CACHE" == 1 ]]; then
    export OMNI_CACHE_MMAP_PATH OMNI_CACHE_PACKED_HBM OMNI_CACHE_LAYER_BYTES
fi
export CURRENT_TIME_STAMP="$(date '+%m%d_%H%M')"

mkdir -p "$LOG_DIR"

# ─── Sanity ───────────────────────────────────────────────────────────────
[[ -d "$MODEL_PATH" ]] \
    || { echo "ERROR: MODEL_PATH not a directory: $MODEL_PATH" | tee -a "$LOG_DIR/launch_error.log" >&2; exit 1; }
command -v vllm >/dev/null \
    || { echo "ERROR: 'vllm' not on PATH" | tee -a "$LOG_DIR/launch_error.log" >&2; exit 1; }

# ─── Dump launch configuration ────────────────────────────────────────────
_dump_launch_config() {
    local dump_file="$LOG_DIR/launch_config.log"
    {
        echo "======================================================================"
        echo "Launch Configuration Dump"
        echo "======================================================================"
        echo "timestamp:       $(date '+%Y-%m-%d %H:%M:%S')"
        echo "hostname:        $(hostname)"
        echo "script:          launch_prefill.sh"
        echo "role:            prefill"
        echo "config_profile:  ${CONFIG_PROFILE:-default}"
        echo "omni_mode:       $( [[ $ENABLE_OMNI_CACHE == 1 ]] && echo omnicache || echo baseline )"
        echo ""
        echo "--- Serving ---"
        echo "MODEL_PATH=${MODEL_PATH}"
        echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
        echo "PORT=${PORT}"
        echo "TP_SIZE=${TP_SIZE}"
        echo "DP_SIZE=${DP_SIZE}"
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
        echo "OMNI_CACHE_PACKED_HBM=${OMNI_CACHE_PACKED_HBM}"
        echo "DISABLE_GATHER_SELECTION=${DISABLE_GATHER_SELECTION}"
        echo "OMNI_CACHE_MMAP_FILE=${OMNI_CACHE_MMAP_FILE}"
        echo "OMNI_CACHE_MMAP_PATH=${OMNI_CACHE_MMAP_PATH}"
        echo "OMNI_CACHE_LAYER_BYTES=${OMNI_CACHE_LAYER_BYTES}"
        echo "MAP_SIZE_BYTES=${MAP_SIZE_BYTES}"
        echo "NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-N/A}"
        echo "OMNI_KV_DUMP_GEAR=${OMNI_KV_DUMP_GEAR}"
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
        echo "P_NODE_LIST=${P_NODE_LIST}"
        echo "BASE_PORT=${BASE_PORT}"
        echo "ZMQ_BASE_PORT=${ZMQ_BASE_PORT}"
        echo "TP_NNODES=${TP_NNODES}"
        echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
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
    } > "$dump_file"
    echo "[launch_prefill] config dumped to: $dump_file"
}
_dump_launch_config

# ─── Reserve hugepages (only needed by OmniCache host pool) ───────────────
if [[ "$ENABLE_OMNI_CACHE" == 1 ]]; then
    [[ -f "$SETUP_HUGETLBFS_SH" ]] \
        || { echo "ERROR: missing $SETUP_HUGETLBFS_SH" >&2; exit 1; }
    echo "[launch_prefill] reserving hugetlbfs (file=$OMNI_CACHE_MMAP_FILE size=${MAP_SIZE_BYTES}B)"
    MAP_SIZE_BYTES="$MAP_SIZE_BYTES" OMNI_FILE="$OMNI_CACHE_MMAP_FILE" \
        bash "$SETUP_HUGETLBFS_SH"
else
    echo "[launch_prefill] ENABLE_OMNI_CACHE=0 — skipping hugetlbfs reservation"
fi

# ─── Choose the KV connector ──────────────────────────────────────────────
if [[ "${ENABLE_OMNI_CACHE}" == "1" ]]; then
    KV_CONNECTOR="OmniCacheConnector"
else
    KV_CONNECTOR="LLMDataDistConnector"
fi
KV_TRANSFER_CONF=$(printf '{"kv_connector":"%s","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":%d,"kv_connector_extra_config":{"p_node_list":["%s"],"kv_producer_dp_size":1}}' \
    "$KV_CONNECTOR" "$KV_PARALLEL_SIZE" "$P_NODE_LIST")

# ─── Build and run the vLLM command ───────────────────────────────────────
VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$PORT"
    --dtype bfloat16
    --max-model-len "$MAX_LEN"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-$MAX_LEN}"
    --max-num-seqs "$BSZ"
    --allowed-local-media-path /
    --distributed-executor-backend "$DIST_BACKEND"
    --gpu-memory-utilization 0.88
    --trust-remote-code
    --tensor-parallel-size "$TP_SIZE"
    --data-parallel-size "$DP_SIZE"
    --no-disable-hybrid-kv-cache-manager
    --reasoning-parser pangu
    --enable-auto-tool-choice
    --tool-call-parser pangu
    --kv-transfer-config "$KV_TRANSFER_CONF"
    --additional-config '{"enable_low_latency": true, "npugraph_ex_config" :{"enable": true, "super_kernel_optimze": false, "static_kernel_compile": true}}'
    --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
)
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
    VLLM_CMD+=(--enable-prefix-caching)
    VLLM_CMD+=(--block-size "$BLOCK_SIZE")
else
    VLLM_CMD+=(--no-enable-prefix-caching)
fi
# Chunked prefill
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
    VLLM_CMD+=(--enable-chunked-prefill)
    if [[ "${CHUNKED_PREFILL_TOKEN_THRESHOLD:-0}" -gt "0" ]]; then
        VLLM_CMD+=(--long-prefill-token-threshold "$CHUNKED_PREFILL_TOKEN_THRESHOLD")
    fi
else
    VLLM_CMD+=(--no-enable-chunked-prefill)
fi
if [[ "${MTP}" -ge "1" ]]; then
    spec_conf=$(printf '{"num_speculative_tokens": %d, "method": "deepseek_mtp"}' "$MTP")
    VLLM_CMD+=(--speculative-config "$spec_conf")
fi
if [[ "$ENABLE_OMNI_CACHE" == 1 ]]; then
    VLLM_CMD+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
    VLLM_CMD+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi

[[ "$ENFORCE_EAGER" == 1 ]] && VLLM_CMD+=(--enforce-eager)
[[ "$EXPERT_PARALLEL" == 1 ]] && VLLM_CMD+=(--enable-expert-parallel)

echo "[launch_prefill] mode=$( [[ $ENABLE_OMNI_CACHE == 1 ]] && echo omnicache || echo baseline )"
echo "[launch_prefill] profile=${CONFIG_PROFILE:-default}"
echo "[launch_prefill] prefix_caching=$ENABLE_PREFIX_CACHING chunk_prefill=$ENABLE_CHUNKED_PREFILL (threshold=$CHUNKED_PREFILL_TOKEN_THRESHOLD)"
echo "[launch_prefill] starting vllm: port=$PORT tp=$TP_SIZE dp=$DP_SIZE model=$MODEL_PATH"
echo "[launch_prefill] serving log:   $LOG_DIR/serving.log"
exec "${VLLM_CMD[@]}" &> "$LOG_DIR/serving.log"
