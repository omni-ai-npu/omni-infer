#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# Run MiniMax M2 in single-instance mixed deployment mode on 8 NPUs.
# Override the defaults below with environment variables when needed.

set -eo pipefail

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

MODEL_PATH=${MODEL_PATH:-<YOUR_MODEL_PATH>}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-minimax}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-196608}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-14}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
CUDAGRAPH_CAPTURE_SIZE=${CUDAGRAPH_CAPTURE_SIZE:-${MAX_NUM_SEQS}}
COMPILE_SIZE=${COMPILE_SIZE:-${MAX_NUM_SEQS}}

export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export ASCEND_RT_VISIBLE_DEVICES
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export OMNI_NPU_VLLM_PATCHES_ALL=${OMNI_NPU_VLLM_PATCHES_ALL:-1}
export OMNI_NPU_VLLM_PATCHES=${OMNI_NPU_VLLM_PATCHES:-ALL}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
export ASCEND_PLATFORM=${ASCEND_PLATFORM:-A2}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}
export ENABLE_PREFILL_PROFILER=${ENABLE_PREFILL_PROFILER:-false}

if [ "${CLEAR_VLLM_COMPILE_CACHE:-1}" = "1" ]; then
    rm -rf /root/.cache/vllm/torch_compile_cache
fi

VLLM_PLUGINS=${VLLM_PLUGINS:-omni-npu,omni_npu_patches,omni_custom_models} \
vllm serve "${MODEL_PATH}" \
    --trust-remote-code \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-chunked-prefill \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --distributed-executor-backend mp \
    --compilation-config "{\"level\": 3, \"cudagraph_mode\":\"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZE}], \"backend\":\"eager\", \"compile_sizes\":[${COMPILE_SIZE}]}" \
    ${EXTRA_ARGS:-}
