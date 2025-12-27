#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# 代理用于hugging face模型下载
export http_proxy=${HTTP_PROXY}
export https_proxy=${HTTP_PROXY}

# Defaults previously set in Dockerfile (moved here so runtime overrides work):
# GLOO_SOCKET_IFNAME, VLLM_* options, HOST, PORT, TZ
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME
VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_USE_V1
VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-fork}
export VLLM_WORKER_MULTIPROC_METHOD
VLLM_ENABLE_MC2=${VLLM_ENABLE_MC2:-0}
export VLLM_ENABLE_MC2
USING_LCCL_COM=${USING_LCCL_COM:-0}
export USING_LCCL_COM
VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-DEBUG}
export VLLM_LOGGING_LEVEL
HOST=${HOST:-0.0.0.0}
export HOST
PORT=${PORT:-8301}
export PORT
TZ=${TZ:-Asia/Shanghai}
export TZ

source ~/.bashrc

# Apply timezone at container runtime in case it wasn't set during image build
if [ -n "$TZ" ] && [ -f /usr/share/zoneinfo/$TZ ]; then
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
fi

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