#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_GLOBAL_LOG_LEVEL=3
model="/your/path/to/model"
log_file="/your/path/to/log_file"

export GLOO_SOCKET_IFNAME=enp23s0f3
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0

export VLLM_LOGGING_LEVEL=DEBUG
VLLM_PLUGINS="omni-npu,omni-cache" vllm serve \
    "$model" \
    --served-model-name deepseek \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 8 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --tensor-parallel-size 16 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --kv-transfer-config '{
        "kv_connector": "LLMDataDistConnector",
        "kv_role": "kv_producer",
        "kv_rank": 0,
        "kv_parallel_size": 1
    }' \
    --enforce-eager &> "${log_file}"
