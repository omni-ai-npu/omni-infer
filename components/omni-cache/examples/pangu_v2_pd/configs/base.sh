# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# base.sh — default configuration for Pangu V2 hybrid PD deployment.
# Sourced by launch_prefill.sh / launch_decode.sh / launch_proxy.sh.
# Every value can be overridden via environment variable or a profile config.

# ─── Serving ──────────────────────────────────────────────────────────────
: "${MODEL_PATH:=/data/models/iter_0011840-W8A8-0509-skip_shared_experts/}"
: "${SERVED_MODEL_NAME:=pangu_ultra_moe}"
: "${BSZ:=8}"
: "${MAX_LEN:=8192}"
: "${DIST_BACKEND:=mp}"
: "${ENFORCE_EAGER:=1}"
: "${EXPERT_PARALLEL:=1}"
: "${BLOCK_SIZE:=128}"

# ─── OmniCache / Pangu V2 hybrid ──────────────────────────────────────────
: "${ENABLE_OMNI_CACHE:=1}"
: "${ENABLE_HOST_MAPPING:=1}"
: "${HYBRID_ATTN_GROUP_SIZE:=16}"
: "${OMNI_CACHE_LOCAL_DP_SIZE:=8}"
: "${OMNI_CACHE_MLA_SWA_DEBUG:=1}"
: "${OMNI_CACHE_ATTN_PLUGINS:=}"
: "${OMNI_REUSE_PREFILLED_TOKENS:=0}"
: "${OMNI_SKIP_DECODE_TOKENIZE:=0}"
: "${OMNI_KV_DUMP_GEAR:=off}"
: "${DISABLE_GATHER_SELECTION:=1}"
: "${OMNI_CACHE_PACKED_HBM:=1}"
: "${ENABLE_OMNI_CACHE_DSA_SPLIT:=0}"

# ─── vLLM + plugins ───────────────────────────────────────────────────────
: "${VLLM_PLUGINS:=omni-npu,omni_npu_patches,omni_pangu_models,omni_custom_models}"
: "${OMNI_NPU_PATCHES_DIR:=pangu_v2_hybrid,pangu_v2_moe}"
: "${OMNI_NPU_VLLM_PATCHES:=ALL}"
: "${VLLM_LOGGING_LEVEL:=INFO}"
: "${MTP:=0}"
: "${ENABLE_PREFIX_CACHING:=0}"
: "${ENABLE_CHUNKED_PREFILL:=0}"
: "${MAX_BATCHED_TOKENS:=}"   # empty = use default per-side logic
: "${CHUNKED_PREFILL_TOKEN_THRESHOLD:=0}"

# ─── Cluster / networking ─────────────────────────────────────────────────
: "${TP_NNODES:=1}"
: "${ASCEND_GLOBAL_LOG_LEVEL:=3}"
: "${HCCL_OP_EXPANSION_MODE:=AIV}"
: "${HCCL_INTRA_ROCE_ENABLE:=1}"
: "${HCCL_INTRA_PCIE_ENABLE:=0}"
: "${HCCL_BUFFSIZE:=700}"
: "${GLOO_SOCKET_IFNAME:=enp23s0f3}"
: "${HCCL_SOCKET_IFNAME:=enp23s0f3}"

# ─── Prefill-specific defaults ────────────────────────────────────────────
: "${PORT:=8000}"
: "${TP_SIZE:=8}"
: "${DP_SIZE:=1}"
: "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

# ─── Decode-specific defaults ─────────────────────────────────────────────
: "${PORT_BASE:=8082}"
: "${DECODE_TP_SIZE:=1}"
: "${DECODE_DP_SIZE:=8}"
: "${DEVICE_START:=8}"
: "${VLLM_WORKER_MULTIPROC_METHOD:=fork}"

# ─── OmniCache memory (conditionally used by scripts) ─────────────────────
: "${OMNI_CACHE_LAYER_BYTES:=17179869184}"
: "${MAP_SIZE_BYTES:=536870912000}"
: "${KV_CACHE_MEMORY_BYTES:=OMNI_CACHE_LAYER_BYTES * HYBRID_ATTN_GROUP_SIZE}"