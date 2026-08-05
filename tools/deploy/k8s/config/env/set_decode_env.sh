#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${CONFIG_DIR:=$( dirname "${CUR_DIR}" )}"
: "${ROOT_DIR:=$( dirname "${CONFIG_DIR}" )}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/basic/endpoint_tools.sh
source "${TOOL_DIR}"/env/env_tools.sh

# 启动脚本使用的环境变量
set_env_from_arg_or_default "CANN_JSON_PATH" "--cann-json-path" "/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json" "$@"

set_env "CURRENT_STARTUP_ROLE" "decode"
set_env "RANK_TABLE_PATH" "${CONFIG_DIR}"/ranktable
set_env "LOCAL_RANK_TABLE_FILE_NAME" "rank_table.json"
set_env "RANK_TABLE_FILE_PATH" "${RANK_TABLE_PATH}"/"${LOCAL_RANK_TABLE_FILE_NAME}"

role_head_ip=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_head_ip.py)
set_env "HEAD_IP" "${role_head_ip}" "Head Pod IP of Decode instance"
local_device_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_local_device_size.py)
set_env "LOCAL_DEVICE_SIZE" "${local_device_size}" "Device count of single Pod"
role_node_rank=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_node_rank.py)
set_env "NODE_RANK" "${role_node_rank}" "Pod index of Decode instance"
role_device_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_device_size.py)
set_env "ROLE_DEVICE_SIZE" "${role_device_size}" "Device count of entire Decode instance"
role_node_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_node_size.py)
set_env "ROLE_POD_SIZE" "${role_node_size}" "Pod count of entire Decode instance"
api_port_list=$(seq "${PD_BASE_API_PORT}" $((PD_BASE_API_PORT + LOCAL_DEVICE_SIZE - 1)) | tr '\n' ',' | sed 's/,$//')
role_servers=$(cross_join_ips_and_ports "${DECODE_SERVER_IP_LIST}" "${api_port_list}")
set_env "ROLE_SERVERS" "${role_servers}" "API servers of entire Decode instance"

set_env_from_arg_or_default "ENABLE_TORCHAIR_CACHE" "--enable-torchair-cache" 0 "$@"
set_env_from_arg_or_default "TORCHAIR_CACHE_PATH" "--torchair-cache-path" "" "$@"
set_env_from_arg_or_default "TRANSFER_TORCHAIR_CACHE" "--transfer-torchair-cache" 0 "$@"

# 启动参数使用的环境变量
set_env_from_arg_or_default "ADDITIONAL_CONFIG" "--additional-config" '{"compilation-config": {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], "backend":"eager", "compile_sizes":[1,2,8]}}' "$@"
set_env_from_arg_or_default "dp" "--num-dp" "${role_device_size}" "$@"
set_env_from_arg_or_default "GPU_UTIL" "--gpu-util" 0.92 "$@"
set_env_from_arg_or_default "HCCL_BUFFSIZE" "--hccl-buffsize" 400 "$@"
set_env_from_arg_or_default "HCCL_OP_EXPANSION_MODE" "--hccl-op-expansion-mode" "AIV" "$@"
set_env_from_arg_or_default "KV_ROLE" "--kv-role" "kv_consumer" "$@"
set_env_from_arg_or_default "NUM_SERVERS" "--num-servers" "${LOCAL_DEVICE_SIZE}" "$@"
set_env_from_arg_or_default "MAX_MODEL_LEN" "--max-model-len" 65536 "$@"
set_env_from_arg_or_default "MAX_NUM_SEQS" "--max-num-seqs" 8 "$@"
set_env_from_arg_or_default "MODEL_PATH" "--model-path" "/home/mind/model" "$@"
set_env_from_arg_or_default "SERVER_OFFSET" "--server-offset" "$((role_node_rank * LOCAL_DEVICE_SIZE))" "$@"
set_env_from_arg_or_default "TP" "--tp" 1 "$@"
set_env_from_arg_or_default "VLLM_LOGGING_LEVEL" "--vllm-logging-level" "INFO" "$@"
set_env_from_arg_or_default "KV_CONNECTOR" "--kv-connector" "LLMDataDistConnector" "$@"

set_env_from_arg_or_default "ADD_ARGS" "--add-args" "" "$@"
set_env_from_arg_or_default "EXTRA_ARGS" "--extra-args" "--enable-expert-parallel --max-num-seqs ${MAX_NUM_SEQS} --no-enable-chunked-prefill --dtype bfloat16 --distributed-executor-backend mp " "$@"

# vLLM使用的其它环境变量
set_env_from_arg_or_default "CPU_AFFINITY_CONF" "--cpu-affinity-conf" 2 "$@"
set_env_from_arg_or_default "HCCL_RDMA_TIMEOUT" "--hccl-rdma-timeout" 20 "$@"
set_env_from_arg_or_default "PYTORCH_NPU_ALLOC_CONF" "--pytorch-npu-alloc-conf" "expandable_segments:True" "$@"

# ray使用的环境变量
set_env_from_arg_or_default "GLOO_SOCKET_IFNAME" "--gloo-socket-ifname" "${SOCKET_IFNAME}" "$@"
set_env_from_arg_or_default "HCCL_SOCKET_IFNAME" "--hccl-socket-ifname" "${SOCKET_IFNAME}" "$@"
set_env_from_arg_or_default "TP_SOCKET_IFNAME" "--tp-socket-ifname" "${SOCKET_IFNAME}" "$@"
