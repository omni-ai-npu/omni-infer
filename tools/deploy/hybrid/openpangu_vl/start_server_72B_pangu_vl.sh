#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

VPC_PREFIX=$(echo "$IP" | cut -d'.' -f1-2)
POD_INET_IP=$(hostname -I | tr ' ' '\n' | grep -o "^$VPC_PREFIX\.[0-9]\+\.[0-9]\+" | head -n 1)
export SOCKET_IFNAME=$(ifconfig | grep -B 1 "$POD_INET_IP" | head -n 1 | awk '{print $1}' | sed 's/://')
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=500
export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"
export TP_SOCKET_IFNAME="$SOCKET_IFNAME"
export HCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
export VLLM_LOGGING_LEVEL="INFO"
export HCCL_INTRA_ROCE_ENABLE="1"
export HCCL_INTRA_PCIE_ENABLE="0"
export VLLM_WORKER_MULTIPROC_METHOD="fork"
#export VLLM_RPC_TIMEOUT=1800
OUTPUT_TEXT_DIR=${OUTPUT_TEXT_DIR:=./}
mkdir -p ${OUTPUT_TEXT_DIR}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_GLOBAL_LOG_LEVEL=3
export OMNI_NPU_VLLM_PATCHES=ALL
export ASCEND_PLATFORM="A2"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:=1200}

export allowed_local_media_path=${allowed_local_media_path:=/}
export LOAD_CKPT_DIR=${LOAD_CKPT_DIR:=/cache/pangu_72b/}

VLLM_PLUGINS="omni-npu,omni_npu_add_models,omni_npu_patches" vllm serve "$LOAD_CKPT_DIR" \
--served-model-name pangu_mm_auto \
--host 0.0.0.0 \
--port 8000 \
--dtype bfloat16 \
--max-model-len 32786 \
--max-num-batched-tokens 32786 \
--max-num-seqs 16 \
--no-enable-chunked-prefill \
--no-enable-prefix-caching \
--distributed-executor-backend mp \
--gpu-memory-utilization 0.75 \
--trust-remote-code \
--tensor-parallel-size 4 \
--data-parallel-size 1 \
--enable-expert-parallel \
--compilation-config '{"level": 3, "cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], "backend":"eager", "compile_sizes":[1,2,8]}' 2>&1 | tee "$OUTPUT_TEXT_DIR/inference_$VPC_PREFIX.log"
#--speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 1}' \
#--enforce-eager
