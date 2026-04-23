#!/bin/bash

## PATH
MOUNT_PATH="${MOUNT_PATH:-/home/work}"
BUCKET_PATH="${BUCKET_PATH:-bucket-pangu-green-guiyang}"
BASE_LOG_PATH="${BASE_LOG_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/logs}"
MODEL_PATH="${MODEL_PATH:-${MOUNT_PATH}/${BUCKET_PATH}/omni-elb/models/qwen3-omni/Qwen3-Omni-30B-A3B-Captioner}"

PORT="${PORT:-8000}"

## ENV
source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export GLOO_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_BUFFSIZE=500
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_DISABLE_COMPILE_CACHE=1
export OMNI_NPU_VLLM_PATCHES="ALL"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_LOGGING_LEVEL=INFO
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "${BASE_LOG_PATH}/qwen3-omni-30b/logs"
LOG_NAME_PREFIX="${POD_IP:-$(hostname)}"
LOG_FILE="${BASE_LOG_PATH}/qwen3-omni-30b/logs/${LOG_NAME_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

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
  --served-model-name qwen3-omni \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 512 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --allowed-local-media-path "$MOUNT_PATH/$BUCKET_PATH/" \
  --media-io-kwargs '{"video":{"fps":2,"num_frames":-1}}' \
  --compilation-config '{"level": 3, "cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[36,72,128,256,512], "backend":"eager", "compile_sizes":[36,72,128,256,512]}' 2>&1 | tee -a "$LOG_FILE"
