#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# Default parameter values
BSZ=16
MAX_LEN=32768
PREFILL_TP_SIZE=1
DECODE_TP_SIZE=1
DECODE_DP_SIZE=1
EXPERT_PARALLEL=true
MTP=false
ENFORCE_EAGER=false
MODEL=""
LOG_DIR="$(dirname "$0")/logs"
LOG_LEVEL=DEBUG
PORT=8081
DIST_BACKEND=mp

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bsz) BSZ="$2"; shift 2 ;;
        --max-len) MAX_LEN="$2"; shift 2 ;;
        --prefill-tp-size) PREFILL_TP_SIZE="$2"; shift 2 ;;
        --decode-tp-size) DECODE_TP_SIZE="$2"; shift 2 ;;
        --decode-dp-size) DECODE_DP_SIZE="$2"; shift 2 ;;
        --enforce-eager) ENFORCE_EAGER=true; shift ;;
        --no-ep) EXPERT_PARALLEL=false; shift ;;
        --mtp) MTP=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --dist-backend) DIST_BACKEND="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --bsz BATCH_SIZE                Maximum batch size (default: 16)"
            echo "  --max-len LENGTH                Maximum sequence length (default: 32768)"
            echo "  --prefill-tp-size SIZE          Prefill tensor parallel size (default: 1)"
            echo "  --decode-tp-size SIZE           Decode tensor parallel size (default: 1)"
            echo "  --decode-dp-size SIZE           Decode data parallel size (default: 1)"
            echo "  --enforce-eager                 Use enforce eager mode (decode only)"
            echo "  --no-ep                         Do not use expert parallelism (must set for dense models)"
            echo "  --mtp                           Use MTP (default: false)"
            echo "  --model PATH                    Model path (required)"
            echo "  --log-dir DIR                   Log directory (default: ./logs)"
            echo "  --log-level LEVEL               Logging level (default: DEBUG)"
            echo "  --port PORT                     Prefill server port (default: 8081)"
            echo "  --dist-backend BACKEND          Distributed backend (default: mp)"
            echo "  --help                          Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
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
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_BUFFSIZE=700

# Validate model path
if [[ -z "$MODEL" || ! -d "$MODEL" ]]; then
    echo "Error: Please specify a valid model directory path using --model option." >&2
    exit 1
fi

# Calculate total NPU devices needed
TOTAL_NPUS=$((PREFILL_TP_SIZE + DECODE_TP_SIZE * DECODE_DP_SIZE))

# Validate that we have enough NPU devices
if [[ $TOTAL_NPUS -gt 16 ]]; then
    echo "Error: Not enough NPU devices available. Requested $TOTAL_NPUS devices, but maximum is 16." >&2
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

gen_arith_seq() {
    local k=$1 n=$2 s=$3 result=
    for ((i=s; i<n+s; i++)); do
        [[ -n $result ]] && result+=","
        result+="$((i*k))"
    done
    echo "$result"
}

# Build vllm command arguments into an array variable named by caller
# usage: build_vllm_args <out_array_name> <tp> <dp> <port> [extra args...]
build_vllm_args() {
    local __outvar=$1; shift
    local tensor_parallel_size=$1 data_parallel_size=$2 port=$3; shift 3
    local extra_args=("$@")

    local args=(
        vllm
        serve
        "$MODEL"
        --served-model-name deepseek
        --host 0.0.0.0
        --port "$port"
        --dtype bfloat16
        --max-model-len "$MAX_LEN"
        --max-num-batched-tokens "$MAX_LEN"
        --max-num-seqs "$BSZ"
        --no-enable-chunked-prefill
        --no-enable-prefix-caching
        --distributed-executor-backend "$DIST_BACKEND"
        --gpu-memory-utilization 0.88
        --trust-remote-code
        --tensor-parallel-size "$tensor_parallel_size"
        --data-parallel-size "$data_parallel_size"
    )

    if [[ "$EXPERT_PARALLEL" == true ]]; then
        args+=(--enable-expert-parallel)
    fi

    if [[ "$MTP" == true ]]; then
        args+=(--speculative-config '{"num_speculative_tokens": 1, "method": "deepseek_mtp"}')
    fi

    # Append any extra args (already properly separated)
    args+=("${extra_args[@]}")

    # Export as a nameref array
    eval "$__outvar"='("${args[@]}")'
}

# Pretty-printer for commands: prints env + args with shell escaping
print_cmd() {
    local env_kv=$1; shift
    local -a cmd=( "$@" )
    if [[ -n $env_kv ]]; then
        printf '%s ' "$env_kv"
    fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
}

# Display configuration
echo "Starting vLLM servers with PD separation:"
echo "  Model: $MODEL"
echo "  Batch Size: $BSZ"
echo "  Max Length: $MAX_LEN"
echo "  Prefill Tensor Parallel Size: $PREFILL_TP_SIZE (NPUs: 0-$((PREFILL_TP_SIZE-1)))"
echo "  Decode Tensor Parallel Size: $DECODE_TP_SIZE"
echo "  Decode Data Parallel Size: $DECODE_DP_SIZE"
echo "  Enforce Eager (Decode only): $ENFORCE_EAGER"
echo "  Expert Parallel: $EXPERT_PARALLEL"
echo "  Use MTP: $MTP"
echo "  Prefill Port: $PORT"
echo "  Log Directory: $LOG_DIR"

echo ""
echo "Starting prefill server..."

PREFILL_DEVICES=$(gen_arith_seq 1 "$PREFILL_TP_SIZE" 0)
PREFILL_PORT=$PORT
printf -v PREFILL_XSF_CONF '{
        "kv_connector": "LLMDataDistConnector",
        "kv_role": "kv_producer",
        "kv_rank": 0,
        "kv_parallel_size": %d
    }' $((DECODE_DP_SIZE+1))

build_vllm_args PREFILL_CMD "$PREFILL_TP_SIZE" 1 "$PREFILL_PORT" \
    --enforce-eager \
    --kv-transfer-config "$PREFILL_XSF_CONF"

echo "Prefill command:"
print_cmd "ASCEND_RT_VISIBLE_DEVICES=\"$PREFILL_DEVICES\"" "${PREFILL_CMD[@]}"

# Run prefill server in background
ASCEND_RT_VISIBLE_DEVICES="$PREFILL_DEVICES" \
    "${PREFILL_CMD[@]}" &> "${LOG_DIR}/prefill.log" &
PREFILL_PID=$!
echo "Prefill server started with PID: $PREFILL_PID"

echo ""
echo "Starting decode server(s)..."

DECODE_PIDS=()
for ((dp_rank=0; dp_rank<DECODE_DP_SIZE; dp_rank++)); do
    DECODE_DEVICE_START=$((PREFILL_TP_SIZE + dp_rank * DECODE_TP_SIZE))
    DECODE_DEVICES=$(gen_arith_seq 1 "$DECODE_TP_SIZE" "$DECODE_DEVICE_START")
    DECODE_PORT=$((PORT + 1 + dp_rank))

    # Prepare extra arguments
    printf -v DECODE_XSF_CONF '{
        "kv_connector": "LLMDataDistConnector",
        "kv_role": "kv_consumer",
        "kv_rank": %d,
        "kv_parallel_size": %d
    }' $((dp_rank+1)) $((DECODE_DP_SIZE+1))
    decode_extra_args=(--kv-transfer-config "$DECODE_XSF_CONF")
    if [[ "$ENFORCE_EAGER" == true ]]; then
        decode_extra_args+=(--enforce-eager)
    else
        [[ "$MTP" == true ]] && interval=2 || interval=1
        gear=$(gen_arith_seq "$interval" "$BSZ" 1)

        # Build JSON string into COMPILATION_CONFIG — it's just raw characters.
        # No extra quoting; keep it a single element later.
        printf -v COMPILATION_CONFIG \
            '{"level":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[%s],"backend":"eager","compile_sizes":[1,2,8]}' \
            "$gear"

        # Keep JSON as its own array element
        decode_extra_args+=(--compilation-config "$COMPILATION_CONFIG")
    fi

    if [[ $DECODE_DP_SIZE -gt 1 ]]; then
        decode_extra_args+=(--data-parallel-rank "$dp_rank")
    fi

    build_vllm_args DECODE_CMD "$DECODE_TP_SIZE" "$DECODE_DP_SIZE" "$DECODE_PORT" "${decode_extra_args[@]}"

    echo "Decode command for dp_rank=$dp_rank:"
    print_cmd "ASCEND_RT_VISIBLE_DEVICES=\"$DECODE_DEVICES\"" "${DECODE_CMD[@]}"

    ASCEND_RT_VISIBLE_DEVICES="$DECODE_DEVICES" \
        "${DECODE_CMD[@]}" &> "${LOG_DIR}/decode_${dp_rank}.log" &
    PID=$!
    DECODE_PIDS+=("$PID")
    echo "Decode server (dp_rank=$dp_rank) started with PID: $PID"
done

cleanup() {
    echo ""
    echo "Shutting down servers..."

    if kill -0 "$PREFILL_PID" 2>/dev/null; then
        echo "Stopping prefill server (PID: $PREFILL_PID)..."
        kill "$PREFILL_PID" 2>/dev/null
        wait "$PREFILL_PID" 2>/dev/null || true
    fi

    for pid in "${DECODE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping decode server (PID: $pid)..."
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null || true
        fi
    done

    echo "All servers stopped."
}

trap cleanup EXIT INT TERM

echo ""
echo "All servers started. Press Ctrl+C to stop."
echo "Prefill server logs: ${LOG_DIR}/prefill.log"
echo "Decode server logs: ${LOG_DIR}/decode*.log"

wait "$PREFILL_PID" "${DECODE_PIDS[@]}"
