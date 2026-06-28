# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# high-throughput.sh — high-throughput profile overrides for Pangu V2 hybrid PD.
# Source after base.sh.  Set CONFIG_PROFILE=high-throughput to activate.
# Uses direct assignment (not :=) because base.sh has already set defaults.

# ─── Shared overrides (both prefill & decode) ─────────────────────────────
MTP=3
HYBRID_ATTN_GROUP_SIZE=17

# ─── Role-specific overrides ──────────────────────────────────────────────
if [[ "${ROLE:-}" == "prefill" ]]; then
    BSZ=8
    MAX_LEN=512000
    OMNI_CACHE_LAYER_BYTES=30541989660
    MAP_SIZE_BYTES=549755813888           # 512 GiB
    NUM_GPU_BLOCKS_OVERRIDE=160000
    KV_CACHE_MEMORY_BYTES=$(( OMNI_CACHE_LAYER_BYTES * HYBRID_ATTN_GROUP_SIZE ))
    ENFORCE_EAGER=1
    DISABLE_GATHER_SELECTION=1
    ENABLE_PREFIX_CACHING=1
    ENABLE_CHUNKED_PREFILL=1
    MAX_BATCHED_TOKENS=102400
    CHUNKED_PREFILL_TOKEN_THRESHOLD=$MAX_BATCHED_TOKENS
elif [[ "${ROLE:-}" == "decode" ]]; then
    BSZ=3
    MAX_LEN=512000
    OMNI_CACHE_LAYER_BYTES=20541989660
    MAP_SIZE_BYTES=549755813888           # 512 GiB
    NUM_GPU_BLOCKS_OVERRIDE=21000
    KV_CACHE_MEMORY_BYTES=$(( OMNI_CACHE_LAYER_BYTES * HYBRID_ATTN_GROUP_SIZE / OMNI_CACHE_LOCAL_DP_SIZE ))
    ENFORCE_EAGER=0
    DISABLE_GATHER_SELECTION=1
    ENABLE_PREFIX_CACHING=0
    ENABLE_HOST_MAPPING=0
    ENABLE_CHUNKED_PREFILL=1
fi
