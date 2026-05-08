#!/bin/bash

## PATH
MOUNT_PATH="${MOUNT_PATH:-/data/w84156052/qwen}"
BUCKET_PATH="${BUCKET_PATH:-master-0210}"
BASE_LOG_PATH="${BASE_LOG_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/logs}"
MODEL_PATH="${MODEL_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/models/qwen-32b/Qwen3-VL-32B-w8a8c16-visualblock-bf16}"   # 量化权重  建议并发 80
# MODEL_PATH="${MODEL_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/models/qwen-32b/Qwen3-VL-32B-Instruct}"                  # 非量化权重

PORT="${PORT:-8000}"

## ENV
source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export GLOO_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_BUFFSIZE=256
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_DISABLE_COMPILE_CACHE=1
export OMNI_NPU_VLLM_PATCHES="ALL"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_LOGGING_LEVEL=INFO
export PATH="$HOME/.local/bin:$PATH"

export XGRAMMAR_DISABLE_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_TRITON_FLASH_ATTN=0

mkdir -p "${BASE_LOG_PATH}/qwen-32b/logs"
LOG_NAME_PREFIX="${POD_IP:-$(hostname)}"
LOG_FILE="${BASE_LOG_PATH}/qwen-32b/logs/${LOG_NAME_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

{
  echo "==== Launch Config ===="
  echo "MOUNT_PATH=$MOUNT_PATH"
  echo "BUCKET_PATH=$BUCKET_PATH"
  echo "BASE_LOG_PATH=$BASE_LOG_PATH"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "PORT=$PORT"
  echo "POD_IP=${POD_IP:-}"
  echo "LOG_FILE=$LOG_FILE"
  echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "======================="
} | tee "$LOG_FILE"

VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models" vllm serve "$MODEL_PATH" \
  --served-model-name qwen3-vl \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 163840 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --swap-space 64 \
  --allowed-local-media-path "$MOUNT_PATH/$BUCKET_PATH/" \
  --media-io-kwargs '{"video":{"fps":2,"num_frames":-1}}' \
  --limit-mm-per-prompt '{"image":2048}' \
  --enable-prompt-tokens-details \
  --compilation-config '{"level": 3, "cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4,8,16,32,64,128,256], "backend":"eager", "compile_sizes":[4,8,16,32,64,128,256]}' 2>&1 | tee -a "$LOG_FILE"