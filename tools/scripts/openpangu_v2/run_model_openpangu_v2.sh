#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.


export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=500
export GLOO_SOCKET_IFNAME="enp23s0f3"
export TP_SOCKET_IFNAME="enp23s0f3"
export HCCL_SOCKET_IFNAME="enp23s0f3"
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_LOGGING_LEVEL="INFO"
export HCCL_INTRA_ROCE_ENABLE="1"
export HCCL_INTRA_PCIE_ENABLE="0"
export VLLM_WORKER_MULTIPROC_METHOD="fork"

model=/mnt/sfs_turbo/bucket-910c-6055/rk/ckpt/openPangu-92B/4K_part1_seed42_20260202/iter_0025000_hf/

export VLLM_PLUGINS="omni-npu,omni_pangu_models,omni_npu_patches"
export OMNI_NPU_PATCHES_DIR="pangu_sink_swa_mla"
export OMNI_NPU_VLLM_PATCHES="ALL"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PORT=8000

rm -rf /root/.cache/vllm/torch_compile_cache/

vllm serve "$model" \
    --served-model-name openpangu_v2 \
    --host 0.0.0.0 \
    --port ${PORT} \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.9 \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --compilation-config '{"level": 3, "cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[32], "backend":"eager", "compile_sizes":[32]}' \
    2>&1 | tee "./server_${PORT}.log"