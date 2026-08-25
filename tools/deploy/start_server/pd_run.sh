#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# Default parameters
# llmdatadist-specific parameters
GLOBAL_RANK_TABLE_FILE_PATH="1p1d_save_dir/global_ranktable_merge.json"
RANK_TABLE_FILE_PATH="save_dir_64/local_ranktable_127.0.0.1_0123.json"
LOCAL_DECODE_SERVER_IP_LIST="127.0.0.1"
GLOBAL_DECODE_SERVER_IP_LIST="127.0.0.1"
OMNI_PD_ROLE="${OMNI_PD_ROLE:-prefill}"
OMNI_PD_PREFILL_POD_NUM="${OMNI_PD_PREFILL_POD_NUM:-1}"
OMNI_PD_DECODE_POD_NUM="${OMNI_PD_DECODE_POD_NUM:-1}"
OMNI_LLMDATADIST_ZMQ_PORT="${VLLM_LLMDATADIST_ZMQ_PORT:-${OMNI_LLMDATADIST_ZMQ_PORT:-5568}}"
# Ascend-specific parameters
HCCL_INTRA_ROCE_ENABLE=1
HCCL_INTRA_PCIE_ENABLE=0
ascend_rt_set=0
USE_INVENTORY_DEVICES=0
# Multi-API Server specific parameters
NUM_SERVERS=1
NUM_DP=1
SERVER_OFFSET=0
MASTER_IP="127.0.0.1"
MASTER_PORT=8503
BASE_API_PORT=9001
# vLLM framework parameters
GLOO_SOCKET_IFNAME="enp23s0f3"
TP_SOCKET_IFNAME="enp23s0f3"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
MODEL_PATH=""
TP=4
PP=1
SERVED_MODEL_NAME="pangu_v2_moe"
MAX_MODEL_LEN=4096
LOG_DIR="apiserverlog"
# PD separation parameters
KV_CONNECTOR="AscendHcclConnectorV1"
KV_BUFFER_DEVICE="npu"
KV_ROLE="kv_producer"
KV_RANK=0
KV_ENGINE_ID=0
KV_PARALLEL_SIZE=2

GPU_UTIL=0.9
EXTRA_ARGS=""
ADDITIONAL_CONFIG=""
HCCL_BUFFSIZE=0
HCCL_OP_EXPANSION_MODE=""
NUM_SPECULATIVE_TOKENS=1
KV_TRANSFER_CONFIG_OVERRIDE=""

# Help information
print_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                        Display this help message"
    echo "  --global-rank-table-path         llmdatadist-specific: Global rank table file path. For merged P/D instances, usually global_ranktable_merge.json (default: $GLOBAL_RANK_TABLE_FILE_PATH)"
    echo "  --rank-table-path                llmdatadist-specific: Local rank table file path for P or D instances. Usually local_ranktable_{IP}_rank.json; for cross-machine D instances, use local_ranktable_merge*.json (default: $RANK_TABLE_FILE_PATH)"
    echo "  --local-decode-server-ip-list    llmdatadist-specific: IP list of current D instance. Separate multiple IPs with commas, maintaining same order as ranktable (default: $LOCAL_DECODE_SERVER_IP_LIST)"
    echo "  --global-decode-server-ip-list   llmdatadist-specific: IP list of all D instances. Combination of all d instances' LOCAL_DECODE_SERVER_IP_LIST, separated by ';'. For 1d scenarios, same as LOCAL_DECODE_SERVER_IP_LIST (default: $GLOBAL_DECODE_SERVER_IP_LIST)"
    echo "  --role                           llmdatadist-specific: Instance role type. Use 'prefill' for P, 'decode' for D (default: $OMNI_PD_ROLE)"
    echo "  --prefill-pod-num                llmdatadist-specific: Number of P instances (default: $OMNI_PD_PREFILL_POD_NUM)"
    echo "  --decode-pod-num                 llmdatadist-specific: Number of D instances (default: $OMNI_PD_DECODE_POD_NUM)"
    echo "  --omni-llmdatadist-zmq-port      llmdatadist-specific: ZMQ port for llmdatadist connector (must be string) (default: $OMNI_LLMDATADIST_ZMQ_PORT)"
    echo "  --hcc-intra-roce-enable          Ascend-specific: Set to 1 for A3, enable intra-HCCL ROCE (default: $HCCL_INTRA_ROCE_ENABLE)"
    echo "  --hcc-intra-pcie-enable          Ascend-specific: Set to 0 for A3, enable intra-HCCL PCIE (default: $HCCL_INTRA_PCIE_ENABLE)"
    echo "  --ascend-rt-visible-devices      Ascend-specific: Visible physical devices for the instance. (default: $ASCEND_RT_VISIBLE_DEVICES)"
    echo "  --use-inventory-devices          Ascend-specific: Set to 1 to derive each server's ASCEND_RT_VISIBLE_DEVICES by slicing the inherited (inventory) ascend_rt_visible_devices per rank, so vLLM can start from arbitrary physical devices instead of always 0,1 (default: $USE_INVENTORY_DEVICES)"
    echo "  --num-servers                    Multi-API Server: Number of API servers (default: $NUM_SERVERS)"
    echo "  --num-dp                         Multi-API Server: Data parallel size (≥ number of servers) (default: $NUM_DP)"
    echo "  --server-offset                  Multi-API Server: Server offset for multi-node setup. For dual-node A3, set to 16 on d_2 instance (default: $SERVER_OFFSET)"
    echo "  --master-ip                      Multi-API Server: Master node IP for multi-node setup. For dual-node A3, set to head node IP (corresponds to vllm data-parallel-address) (default: $MASTER_IP)"
    echo "  --master-port                    Multi-API Server: Master node Gloo socket communication port (corresponds to vllm data-parallel-rpc-port) (default: $MASTER_PORT)"
    echo "  --base-api-port                  Multi-API Server: Base API port for multi API servers (default: $BASE_API_PORT)"
    echo "  --gloo-socket-ifname             vLLM framework: DP communication parameter. Your network interface. Query with: ip -4 route list 0/0 | awk '{print $5}' | head -n 1 (default: $GLOO_SOCKET_IFNAME)"
    echo "  --tp-socket-ifname               vLLM framework: DP communication parameter. Your network interface. Query with: ip -4 route list 0/0 | awk '{print $5}' | head -n 1 (default: $TP_SOCKET_IFNAME)"
    echo "  --vllm-logging-level             vLLM framework: VLLM logging level. Default INFO, set to DEBUG for debugging (default: $VLLM_LOGGING_LEVEL)"
    echo "  --vllm-worker-multiproc-method   vLLM framework: VLLM worker process method (fork or spawn) (default: $VLLM_WORKER_MULTIPROC_METHOD)"
    echo "  --model-path                     vLLM framework: Model path (default: $MODEL_PATH)"
    echo "  --max-model-len                  vLLM framework: Maximum model length (default: $MAX_MODEL_LEN)"
    echo "  --tp                             vLLM framework: Tensor parallel (default: $TP)"
    echo "  --pp                             vLLM framework: Pipeline parallel (default: $PP)"
    echo "  --served-model-name              vLLM framework: Served model name (default: $SERVED_MODEL_NAME)"
    echo "  --log-dir                        vLLM framework: Log directory (default: $LOG_DIR)"
    echo "  --kv-connector                   vLLM framework: PD separation parameter, kv connector name (default: $KV_CONNECTOR)"
    echo "  --kv-buffer-device               vLLM framework: PD separation parameter, kv transfer buffer device (default: $KV_BUFFER_DEVICE)"
    echo "  --kv-role                        vLLM framework: PD separation parameter, kv role (p: kv_producer, d: kv_consumer) (default: $KV_ROLE)"
    echo "  --kv-rank                        vLLM framework: PD separation parameter, kv rank (p_num/d_num-1) (default: $KV_RANK)"
    echo "  --kv-engine-id                   vLLM framework: PD separation parameter, kv engine ID (default: $KV_ENGINE_ID)"
    echo "  --kv-parallel-size               vLLM framework: PD separation parameter, kv parallel size (equal to num_p + num_d) (default: $KV_PARALLEL_SIZE)"
    echo "  --extra-args                     vLLM framework: Additional VLLM arguments (space-separated, e.g., '--enable-expert-parallel') (default: $EXTRA_ARGS)"
    echo "  --additional-args                vLLM framework: Additional VLLM arguments"
    echo "  --hccl-op-expansion-mode         vLLM framework: HCCL_OP_EXPANSION_MODE"
    echo "  --hccl-buffsize                  vLLM framework: HCCL_BUFFSIZE"
    echo "  --num-speculative-tokens         vLLM framework: Speculative decoding parameter, number of speculative tokens per step (default: $NUM_SPECULATIVE_TOKENS)"
    exit 0
}

# Parse long options
parse_long_option() {
    case "$1" in
        --global-rank-table-path)
            GLOBAL_RANK_TABLE_FILE_PATH="$2"
            ;;
        --rank-table-path)
            RANK_TABLE_FILE_PATH="$2"
            ;;
        --local-decode-server-ip-list)
            LOCAL_DECODE_SERVER_IP_LIST="$2"
            ;;
        --global-decode-server-ip-list)
            GLOBAL_DECODE_SERVER_IP_LIST="$2"
            ;;
        --role)
            OMNI_PD_ROLE="$2"
            ;;
        --prefill-pod-num)
            OMNI_PD_PREFILL_POD_NUM="$2"
            ;;
        --decode-pod-num)
            OMNI_PD_DECODE_POD_NUM="$2"
            ;;
        --omni-llmdatadist-zmq-port)
            OMNI_LLMDATADIST_ZMQ_PORT="$2"
            ;;
        --hcc-intra-roce-enable)
            HCCL_INTRA_ROCE_ENABLE="$2"
            ;;
        --hcc-intra-pcie-enable)
            HCCL_INTRA_PCIE_ENABLE="$2"
            ;;
        --ascend-rt-visible-devices)
            ASCEND_RT_VISIBLE_DEVICES="$2"
            ascend_rt_set=1
            ;;
        --use-inventory-devices)
            USE_INVENTORY_DEVICES="$2"
            ;;
        --num-servers)
            NUM_SERVERS="$2"
            ;;
        --num-dp)
            NUM_DP="$2"
            ;;
        --server-offset)
            SERVER_OFFSET="$2"
            ;;
        --master-ip)
            MASTER_IP="$2"
            ;;
        --master-port)
            MASTER_PORT="$2"
            ;;
        --base-api-port)
            BASE_API_PORT="$2"
            ;;
        --gloo-socket-ifname)
            GLOO_SOCKET_IFNAME="$2"
            ;;
        --tp-socket-ifname)
            TP_SOCKET_IFNAME="$2"
            ;;
        --vllm-logging-level)
            VLLM_LOGGING_LEVEL="$2"
            ;;
        --vllm-worker-multiproc-method)
            VLLM_WORKER_MULTIPROC_METHOD="$2"
            ;;
        --model-path)
            MODEL_PATH="$2"
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            ;;
        --tp)
            TP="$2"
            ;;
        --pp)
            PP="$2"
            ;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            ;;
        --log-dir)
            LOG_DIR="$2"
            ;;
        --kv-connector)
            KV_CONNECTOR="$2"
            ;;
        --kv-buffer-device)
            KV_BUFFER_DEVICE="$2"
            ;;
        --kv-role)
            KV_ROLE="$2"
            ;;
        --kv-rank)
            KV_RANK="$2"
            ;;
        --kv-engine-id)
            KV_ENGINE_ID="$2"
            ;;
        --kv-parallel-size)
            KV_PARALLEL_SIZE="$2"
            ;;
        --kv-transfer-config)
            KV_TRANSFER_CONFIG_OVERRIDE="$2"
            ;;
        --extra-args)
            EXTRA_ARGS="$2"
            ;;
        --gpu-util)
            GPU_UTIL="$2"
            ;;
        --additional-config)
            ADDITIONAL_CONFIG="$2"
            ;;
        --hccl-buffsize)
            HCCL_BUFFSIZE="$2"
            ;;
        --hccl-op-expansion-mode)
            HCCL_OP_EXPANSION_MODE="$2"
            ;;
        --num-speculative-tokens)
            NUM_SPECULATIVE_TOKENS="$2"
            ;;
        --help)
            print_help
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_help
            ;;
    esac
    return 0
}

# Parse options
# Modified main loop
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            ;;
        --*)
            parse_long_option "$1" "$2"  # Parse without shift
            shift 2  # Shift in main loop
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_help
            ;;
    esac
done

# Build KV transfer config JSON
if [ -n "$KV_TRANSFER_CONFIG_OVERRIDE" ]; then
    KV_TRANSFER_CONFIG="$KV_TRANSFER_CONFIG_OVERRIDE"
else
    KV_TRANSFER_CONFIG=$(cat <<EOF
{
    "kv_connector": "$KV_CONNECTOR",
    "kv_role": "$KV_ROLE",
    "kv_rank": $KV_RANK,
    "kv_parallel_size": $KV_PARALLEL_SIZE,
    "kv_port": $OMNI_LLMDATADIST_ZMQ_PORT
}
EOF
)
fi

# Export environment variables
export GLOBAL_RANK_TABLE_FILE_PATH
export RANK_TABLE_FILE_PATH
export LOCAL_DECODE_SERVER_IP_LIST
export GLOBAL_DECODE_SERVER_IP_LIST
export OMNI_PD_ROLE
export OMNI_PD_PREFILL_POD_NUM
export OMNI_PD_DECODE_POD_NUM
export OMNI_LLMDATADIST_ZMQ_PORT

export HCCL_INTRA_ROCE_ENABLE
export HCCL_INTRA_PCIE_ENABLE
if [ $ascend_rt_set -eq 1 ]; then
    export ASCEND_RT_VISIBLE_DEVICES
    echo "ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
fi
export GLOO_SOCKET_IFNAME
export TP_SOCKET_IFNAME
export VLLM_LOGGING_LEVEL
export VLLM_WORKER_MULTIPROC_METHOD
export SERVER_OFFSET
export PYTHONPATH=/usr/local/Ascend/CANN-7.7/toolkit/python/site-packages:$PYTHONPATH
export USING_LCCL_COM=0
# Set this variable to enable proc_bind.
# export CPU_AFFINITY_CONF=2

if [ -n "$HCCL_OP_EXPANSION_MODE" ]; then
    export HCCL_OP_EXPANSION_MODE
    echo "HCCL_OP_EXPANSION_MODE: $HCCL_OP_EXPANSION_MODE"
fi
if [ $HCCL_BUFFSIZE -gt 0 ] ; then
    export HCCL_BUFFSIZE
    echo "HCCL_BUFFSIZE: $HCCL_BUFFSIZE"
fi

export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=120
# 随路拷贝
export TNG_HOST_COPY=1
# 使能双页表 pd 分离
export AUTO_USE_UC_MEMORY=1
# export TASK_QUEUE_ENABLE=2

# enable to overwrite request IDs
export ENABLE_OVERWRITE_REQ_IDS=1

# Print current configuration
echo "==== Current Configuration ===="
echo "GLOBAL_RANK_TABLE_FILE_PATH: $GLOBAL_RANK_TABLE_FILE_PATH"
echo "RANK_TABLE_FILE_PATH: $RANK_TABLE_FILE_PATH"
echo "LOCAL_DECODE_SERVER_IP_LIST: $LOCAL_DECODE_SERVER_IP_LIST"
echo "GLOBAL_DECODE_SERVER_IP_LIST: $GLOBAL_DECODE_SERVER_IP_LIST"
echo "OMNI_PD_ROLE: $OMNI_PD_ROLE"
echo "OMNI_PD_PREFILL_POD_NUM: $OMNI_PD_PREFILL_POD_NUM"
echo "OMNI_PD_DECODE_POD_NUM: $OMNI_PD_DECODE_POD_NUM"
echo "OMNI_LLMDATADIST_ZMQ_PORT: $OMNI_LLMDATADIST_ZMQ_PORT"
echo "HCCL_INTRA_ROCE_ENABLE: $HCCL_INTRA_ROCE_ENABLE"
echo "HCCL_INTRA_PCIE_ENABLE: $HCCL_INTRA_PCIE_ENABLE"
echo "NUM_SERVERS: $NUM_SERVERS"
echo "NUM_DP: $NUM_DP"
echo "SERVER_OFFSET: $SERVER_OFFSET"
echo "MASTER_IP: $MASTER_IP"
echo "MASTER_PORT: $MASTER_PORT"
echo "BASE_API_PORT: $BASE_API_PORT"
echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
echo "TP_SOCKET_IFNAME: $TP_SOCKET_IFNAME"
echo "VLLM_LOGGING_LEVEL: $VLLM_LOGGING_LEVEL"
echo "VLLM_WORKER_MULTIPROC_METHOD: $VLLM_WORKER_MULTIPROC_METHOD"
echo "MODEL_PATH: $MODEL_PATH"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "TP: $TP"
echo "PP: $PP"
echo "SERVED_MODEL_NAME: $SERVED_MODEL_NAME"
echo "LOG_DIR: $LOG_DIR"
echo "KV_TRANSFER_CONFIG: $KV_TRANSFER_CONFIG"
echo "EXTRA_ARGS: $EXTRA_ARGS"
echo "GPU_UTIL: $GPU_UTIL"
echo "ADDITIONAL_CONFIG: $ADDITIONAL_CONFIG"
echo "TNG_HOST_COPY: $TNG_HOST_COPY"
echo "CPU_AFFINITY_CONF: $CPU_AFFINITY_CONF"
echo "AUTO_USE_UC_MEMORY: $AUTO_USE_UC_MEMORY"
echo "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES: $RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"
echo "RAY_CGRAPH_get_timeout: $RAY_CGRAPH_get_timeout"
echo "TASK_QUEUE_ENABLE: $TASK_QUEUE_ENABLE"
echo "USE_INVENTORY_DEVICES: $USE_INVENTORY_DEVICES"
echo "HIXL_LOCAL_COMM_RES_ENABLE: ${HIXL_LOCAL_COMM_RES_ENABLE:-false}"
echo "HIXLP_ENDPOINT_PATH: ${HIXLP_ENDPOINT_PATH:-/etc/hixlep}"
echo "=================="

# Generate the UB endpoint configs consumed by the hixl backend. Platforms
# without UB skip this path unless HIXL_LOCAL_COMM_RES_ENABLE is explicitly set.
case "$(echo "${HIXL_LOCAL_COMM_RES_ENABLE:-false}" | tr '[:upper:]' '[:lower:]')" in
  0|false|no)
    echo "HIXL_LOCAL_COMM_RES_ENABLE off, skipping UB endpoint config generation"
    ;;
  *)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export HIXLP_ENDPOINT_PATH="${HIXLP_ENDPOINT_PATH:-/etc/hixlep}"
    if ! mkdir -p "$HIXLP_ENDPOINT_PATH" 2>/dev/null || [ ! -w "$HIXLP_ENDPOINT_PATH" ]; then
        export HIXLP_ENDPOINT_PATH="${HOME:-/tmp}/.hixlep"
        echo "default endpoint dir not writable, using $HIXLP_ENDPOINT_PATH"
    fi
    if ! mkdir -p "$HIXLP_ENDPOINT_PATH"; then
        echo "ERROR: cannot create $HIXLP_ENDPOINT_PATH; set HIXLP_ENDPOINT_PATH to a writable directory" >&2
        exit 1
    fi
    gen_ep_script="$SCRIPT_DIR/../model_arts/generate_ep_server_pod.py"
    gen_ep_cmd=(python "$gen_ep_script")
    # P and D may share one container and generate the same files concurrently.
    if command -v flock >/dev/null 2>&1; then
        gen_ep_cmd=(flock "$HIXLP_ENDPOINT_PATH/.generate.lock" "${gen_ep_cmd[@]}")
    fi
    if ! "${gen_ep_cmd[@]}"; then
        echo "ERROR: failed to generate endpoint configs in $HIXLP_ENDPOINT_PATH" >&2
        exit 1
    fi
    ;;
esac

EXTRA_ARGS="$EXTRA_ARGS"
# Execute Python script

common_operations() {
  local mtp_args=""
  if [ "$NUM_SPECULATIVE_TOKENS" -ne 0 ]; then
    mtp_args="--enable-mtp"
  fi
  local inventory_devices_args=""
  if [ "$USE_INVENTORY_DEVICES" = "1" ]; then
    inventory_devices_args="--use-inventory-devices"
  fi
  python start_api_servers.py \
    --num-servers "$NUM_SERVERS" \
    --num-dp "$NUM_DP" \
    --server-offset "$SERVER_OFFSET" \
    --model-path "$MODEL_PATH" \
    --master-ip "$MASTER_IP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --master-port "$MASTER_PORT" \
    --base-api-port "$BASE_API_PORT" \
    --tp "$TP" \
    --pp "$PP" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --log-dir "$LOG_DIR" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --gpu-util "$GPU_UTIL" \
    --additional-config "$ADDITIONAL_CONFIG" \
    $mtp_args \
    $inventory_devices_args \
    --num-speculative-tokens "$NUM_SPECULATIVE_TOKENS" \
    --extra-args "$EXTRA_ARGS"
}

if [ $(echo -n "$NODE_IP_LIST" | tr -cd ',' | wc -c) -ge 1 ]; then
  if [ "$IP" = "$HOST_IP" ]; then
    export RAY_USAGE_STATS_ENABLED=0
    ray start --head --num-gpus=$NUM_SERVERS
    sleep 10s
    common_operations
  else
    sleep 5s
    command="ray start --address='$HOST_IP:6379' --num-gpus=$NUM_SERVERS &> /dev/null"
    echo $command
    cost_time=0
    end_time=300
    while true; do
      if [ $cost_time -ge $end_time ]; then
        echo "error, conneciton timeout"
        exit 1
      fi

      eval $command
      if [ $? -eq 0 ]; then
        echo "succeed to connect to ray head node"
        break
      else
        echo "failed to connect to ray head node, wait 5s....."
        sleep 5
        cost_time=$((cost + 5))
      fi
    done
  fi
else
  common_operations
fi
