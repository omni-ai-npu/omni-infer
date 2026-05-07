#!/bin/bash

## PATH
MOUNT_PATH="${MOUNT_PATH:-/home/ma-user/work}"
BUCKET_PATH="${BUCKET_PATH:-bucket-pangu-green-guiyang}"
BASE_LOG_PATH="${BASE_LOG_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/logs}"
MODEL_PATH="${MODEL_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/models/GPT/gpt-oss-120b-cint8/}"
TIKTOKEN_ENCODINGS_PATH="${TIKTOKEN_ENCODINGS_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/deploy/GPT/gpt-oss/scripts/}"

PORT="${PORT:-7000}"

if [ -n "$SERVICE_NAME" ]; then 
  BASE_LOG_PATH="${BASE_LOG_PATH}/${MODEL_NAME}/${SERVICE_NAME}"
fi

## ENV
source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export GLOO_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_DISABLE_COMPILE_CACHE=1
export HCCL_BUFFSIZE=500
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_GLOBAL_LOG_LEVEL=3
export OMNI_NPU_VLLM_PATCHES="ALL"
export ASCEND_PLATFORM="A2"
export TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_PATH}
export VLLM_LOGGING_LEVEL=INFO
export OMP_NUM_THREADS=1
export OMNI_NPU_PATCHES_DIR="gpt_oss"

# 使用单一时间戳，确保整个脚本中使用同一个时间
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_LOG_PATH="${BASE_LOG_PATH}/${TIMESTAMP}"

mkdir -p "$BASE_LOG_PATH"
LOG_NAME_PREFIX="${POD_IP:-$(hostname)}"
LOG_FILE="${BASE_LOG_PATH}/${LOG_NAME_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

{
  echo "==== Launch Config ===="
  echo "MOUNT_PATH=$MOUNT_PATH"
  echo "BUCKET_PATH=$BUCKET_PATH"
  echo "BASE_LOG_PATH=$BASE_LOG_PATH"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "TIKTOKEN_ENCODINGS_PATH=$TIKTOKEN_ENCODINGS_PATH"
  echo "PORT=$PORT"
  echo "POD_IP=${POD_IP:-}"
  echo "LOG_FILE=$LOG_FILE"
  echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "======================="
} | tee "$LOG_FILE"

VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_custom_models" vllm serve "$MODEL_PATH" \
  --served-model-name gpt-oss \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 128000 \
  --max-num-seqs 800 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  --compilation-config '{"level": 3, "cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[800], "backend":"eager", "compile_sizes":[800]}' \
  --trust-remote-code 2>&1 | tee -a "$LOG_FILE"
