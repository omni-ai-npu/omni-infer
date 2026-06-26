#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# Default parameter values
BSZ=16
MAX_LEN=32768
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
EXPERT_PARALLEL=true
MTP=false
ENFORCE_EAGER=false
MODEL=""
USER_COMPILATION_CONFIG=""
SERVED_MODEL_NAME="deepseek"
LOG_DIR="$(dirname "$0")/logs"
LOG_LEVEL=DEBUG
PORT=8080
DIST_BACKEND=mp

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bsz)
            BSZ="$2"
            shift 2
            ;;
        --max-len)
            MAX_LEN="$2"
            shift 2
            ;;
        --tp-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --dp-size)
            DATA_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --enforce-eager)
            ENFORCE_EAGER=true
            shift
            ;;
        --no-ep)
            EXPERT_PARALLEL=false
            shift
            ;;
        --mtp)
            MTP=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --dist-backend)
            DIST_BACKEND="$2"
            shift 2
            ;;
        --compilation-config)
            USER_COMPILATION_CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --bsz BATCH_SIZE                Maximum batch size (default: 16)"
            echo "  --max-len LENGTH                Maximum sequence length (default: 32768)"
            echo "  --tp-size SIZE                  Tensor parallel size (default: 1)"
            echo "  --dp-size SIZE                  Data parallel size (default: 1)"
            echo "  --enforce-eager                 Use enforce eager mode"
            echo "  --no-ep                         Do not use expert parallelism (must set for dense models)"
            echo "  --mtp                           Use MTP (default: false)"
            echo "  --model PATH                    Model path (required)"
            echo "  --served-model-name NAME        Served model name (default: deepseek)"
            echo "  --log-dir DIR                   Log directory (default: ./logs)"
            echo "  --log-level LEVEL               Logging level (default: DEBUG)"
            echo "  --port PORT                     Server port (default: 8080)"
            echo "  --dist-backend BACKEND          Distributed backend (default: mp)"
            echo "  --compilation-config JSON       vLLM compilation config override"
            echo "  --help                          Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set environment variables
export VLLM_LOGGING_LEVEL="$LOG_LEVEL"
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_OP_EXPANSION_MODE=AIV
export GLOO_SOCKET_IFNAME=enp23s0f3
export HCCL_SOCKET_IFNAME=enp23s0f3

# Validate model path
if [[ -z "$MODEL" || ! -d "$MODEL" ]]; then
    echo "Error: Please specify a valid model directory path using --model option."
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Build vllm command
VLLM_CMD=(
    vllm
    serve
    "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$PORT"
    --dtype bfloat16
    --max-model-len "$MAX_LEN"
    --max-num-batched-tokens "$MAX_LEN"
    --max-num-seqs "$BSZ"
    --no-enable-chunked-prefill
    --no-enable-prefix-caching
    --allowed-local-media-path /
    --distributed-executor-backend "$DIST_BACKEND"
    --gpu-memory-utilization 0.88
    --trust-remote-code
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
)

gen_arith_seq() {
    local k=$1
    local n=$2
    local s=$3
    local result=""
    for ((i=s; i<n+s; i++)); do
        if [[ -n "$result" ]]; then
            result+=","
        fi
        result+="$((i*k))"
    done
    echo "$result"
}

# Add enforce-eager or compilation-config parameter
if [[ "$ENFORCE_EAGER" = true ]]; then
    VLLM_CMD+=(--enforce-eager)
else
    if [[ -n "$USER_COMPILATION_CONFIG" ]]; then
        VLLM_CMD+=(--compilation-config "$USER_COMPILATION_CONFIG")
    else
        [[ "$MTP" = true ]] && interval=2 || interval=1
        gear=$(gen_arith_seq $interval $BSZ 1)
        COMPILATION_CONFIG=$(printf '{"level":3, "cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[%s], "backend":"eager", "compile_sizes":[1,2,8]}' "$gear")
        VLLM_CMD+=(--compilation-config "$COMPILATION_CONFIG")
    fi
fi
if [[ "$EXPERT_PARALLEL" = true ]]; then
    VLLM_CMD+=(--enable-expert-parallel)
fi

if [[ "$MTP" = true ]]; then
    VLLM_CMD+=(--speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}')
fi

# Display configuration
echo "Starting vLLM server with parameters:"
echo "  Model: $MODEL"
echo "  Batch Size: $BSZ"
echo "  Max Length: $MAX_LEN"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Data Parallel Size: $DATA_PARALLEL_SIZE"
echo "  Enforce Eager: $ENFORCE_EAGER"
echo "  Expert Parallel: $EXPERT_PARALLEL"
echo "  Use MTP: $MTP"
echo "  Port: $PORT"
echo "  Log Directory: $LOG_DIR"
if [[ "$ENFORCE_EAGER" != true ]]; then
    echo "  Compilation Config: $COMPILATION_CONFIG"
fi

# Execute command and redirect output to log file
echo ""
echo "Command to execute:"
echo "${VLLM_CMD[@]}"
echo ""

ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" "${VLLM_CMD[@]}" &> "${LOG_DIR}/serving.log"
