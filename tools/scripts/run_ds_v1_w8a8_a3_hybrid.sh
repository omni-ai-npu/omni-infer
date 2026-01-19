#!/bin/bash


IP="7.150.12.18"
PORT=8000
LOG_FILE="./server.log"
MODEL_PATH="/data/models/DeepSeek-V31-Terminus-INT8"
# Parse long options
parse_long_option() {
    case "$1" in
        --ip)
            IP="$2"
            ;;
        --port)
            PORT="$2"
            ;;
        --log-file)
            LOG_FILE="$2"
            ;;
        --model)
            MODEL_PATH="$2"
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 0
            ;;
    esac
    return 0
}

# Parse options
# Modified main loop
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            parse_long_option "$1" "$2"  # Parse without shift
            shift 2  # Shift in main loop
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 0
            ;;
    esac
done

export HCCL_OP_EXPANSION_MODE="AIV"

rm -rf /root/.cache/vllm/torch_compile_cache
export VLLM_DISABLE_COMPILE_CACHE=1

export HCCL_BUFFSIZE=500
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_GLOBAL_LOG_LEVEL=3

VLLM_PLUGINS="omni-npu" vllm serve "$MODEL_PATH" \
  --served-model-name deepseek \
  --host $IP \
  --port $PORT \
  --dtype bfloat16 \
  --max-model-len 44000 \
  --max-num-batched-tokens 44000 \
  --max-num-seqs 16 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --compilation-config '{"level": 3, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], "backend":"eager", "compile_sizes":[1,2,8]}' 2>&1 | tee "${LOG_FILE}" &
