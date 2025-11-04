#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# 代理用于hugging face模型下载
export http_proxy=${HTTP_PROXY}
export https_proxy=${HTTP_PROXY}

source ~/.bashrc

# 清理内存
python -c "import torch; torch.npu.empty_cache()"

ASCEND_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}  # 默认值为 0
exec env ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEVICES} \
    python -m vllm.entrypoints.openai.api_server \
    --host ${HOST} \
    --port ${PORT} \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --block-size 128 \
    --distributed-executor-backend mp \
    --max-num-batched-tokens 20000 \
    --max-num-seqs 128 \
    "$@"