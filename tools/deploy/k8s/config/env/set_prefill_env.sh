#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${CONFIG_DIR:=$( dirname "${CUR_DIR}" )}"
: "${ROOT_DIR:=$( dirname "${CONFIG_DIR}" )}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/env/env_tools.sh

# 启动脚本使用的环境变量
set_env "CURRENT_STARTUP_ROLE" "prefill"
set_env "RANK_TABLE_PATH" "${CONFIG_DIR}"/ranktable
set_env "LOCAL_RANK_TABLE_FILE_NAME" "rank_table.json"
set_env "RANK_TABLE_FILE_PATH" "${RANK_TABLE_PATH}"/"${LOCAL_RANK_TABLE_FILE_NAME}"

role_head_ip=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_head_ip.py)
set_env "HEAD_IP" "${role_head_ip}" "Head Pod IP of Prefill instance"
local_device_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_local_device_size.py)
set_env "LOCAL_DEVICE_SIZE" "${local_device_size}" "Device count of single Pod"
role_device_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_device_size.py)
set_env "ROLE_DEVICE_SIZE" "${role_device_size}" "Device count of entire Prefill instance"
role_ip_list=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_ip_list.py)
set_env "ROLE_IP_LIST" "${role_ip_list}" "Pod IPs of entire Prefill instance"
role_node_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_node_size.py)
set_env "ROLE_POD_SIZE" "${role_node_size}" "Pod count of entire Prefill instance"
set_env "ROLE_SERVERS" "${HEAD_IP}:${PD_BASE_API_PORT}" "API servers of entire Prefill instance"

# 启动参数使用的环境变量
set_env_from_arg_or_default "ADDITIONAL_CONFIG" "--additional-config" '' "$@"
set_env_from_arg_or_default "GPU_UTIL" "--gpu-util" 0.90 "$@"
set_env_from_arg_or_default "HCCL_BUFFSIZE" "--hccl-buffsize" 50 "$@"
set_env_from_arg_or_default "HCCL_OP_EXPANSION_MODE" "--hccl-op-expansion-mode" "AIV" "$@"
set_env_from_arg_or_default "KV_RANK" "--kv-rank" "$((SERVER_GROUP_ID - 1))" "$@"
set_env_from_arg_or_default "KV_ROLE" "--kv-role" "kv_producer" "$@"
set_env_from_arg_or_default "LONG_PREFILL_TOKEN_THRESHOLD" "--long-prefill-token-threshold" 32768 "$@"
set_env_from_arg_or_default "MAX_MODEL_LEN" "--max-model-len" 65536 "$@"
set_env_from_arg_or_default "MAX_NUM_BATCHED_TOKENS" "--max-num-batched-tokens" 32768 "$@"
set_env_from_arg_or_default "MAX_NUM_SEQS" "--max-num-seqs" 8 "$@"
set_env_from_arg_or_default "MODEL_PATH" "--model-path" "/home/mind/model" "$@"
set_env_from_arg_or_default "TP" "--tp" "${role_device_size}" "$@"
set_env_from_arg_or_default "VLLM_LOGGING_LEVEL" "--vllm-logging-level" "INFO" "$@"
set_env_from_arg_or_default "KV_CONNECTOR" "--kv-connector" "LLMDataDistConnector" "$@"

set_env_from_arg_or_default "ADD_ARGS" "--add-args" "" "$@"
if [[ ${ROLE_POD_SIZE} != "1" ]]; then
    set_env_from_arg_or_default "EXTRA_ARGS" "--extra-args" "--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --enforce-eager --enable-expert-parallel --max-num-seqs ${MAX_NUM_SEQS} --no-enable-chunked-prefill --dtype bfloat16 --long-prefill-token-threshold ${LONG_PREFILL_TOKEN_THRESHOLD} --distributed-executor-backend ray ${ADD_ARGS}" "$@"
else
    set_env_from_arg_or_default "EXTRA_ARGS" "--extra-args" "--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --enforce-eager --enable-expert-parallel --max-num-seqs ${MAX_NUM_SEQS} --no-enable-chunked-prefill --dtype bfloat16 --long-prefill-token-threshold ${LONG_PREFILL_TOKEN_THRESHOLD} ${ADD_ARGS}" "$@"
fi

visible_devices=$(seq 0 $((LOCAL_DEVICE_SIZE - 1)) | tr '\n' ',' | sed 's/,$//')
set_env "ASCEND_RT_VISIBLE_DEVICES" "${visible_devices}" "A sequence with a step of 1 from 0 whose size equals device count of single Pod ${LOCAL_DEVICE_SIZE}"
set_env "ROLE" "${CURRENT_STARTUP_ROLE}"

# vLLM使用的其它环境变量
set_env_from_arg_or_default "CPU_AFFINITY_CONF" "--cpu-affinity-conf" 2 "$@"
set_env_from_arg_or_default "FORCE_ENABLE_CHUNK_PREFILL" "--force-enable-chunk-prefill" 1 "$@"
set_env_from_arg_or_default "HCCL_RDMA_TIMEOUT" "--hccl-rdma-timeout" 20 "$@"
set_env_from_arg_or_default "PYTORCH_NPU_ALLOC_CONF" "--pytorch-npu-alloc-conf" "expandable_segments:True" "$@"
set_env_from_arg_or_default "VALIDATORS_CONFIG_PATH" "--validators-config-path" "${CONFIG_DIR}/validation/validators_config.json" "$@"

if [[ ${ROLE_POD_SIZE} != "1" ]]; then
    set_env "MASTER_NODE_IP" "${role_head_ip}" "Head Pod IP of Prefill instance"
    set_env "NNODES" "${role_node_size}" "Pod count of Prefill instance"
    set_env "NODE_IP_LIST" "${role_ip_list}" "Pod IPs of entire Prefill instance"
    role_node_rank=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_node_rank.py)
    set_env "NODE_RANK" "${role_node_rank}" "Pod index of Prefill instance"
fi

# ray使用的环境变量
set_env_from_arg_or_default "GLOO_SOCKET_IFNAME" "--gloo-socket-ifname" "${SOCKET_IFNAME}" "$@"
set_env_from_arg_or_default "HCCL_SOCKET_IFNAME" "--hccl-socket-ifname" "${SOCKET_IFNAME}" "$@"
set_env_from_arg_or_default "RAY_CGRAPH_get_timeout" "--ray-cgraph-get-timeout" 7200 "$@"
set_env_from_arg_or_default "RAY_DEDUP_LOGS" "--ray-dedup-logs" 0 "$@"
set_env_from_arg_or_default "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES" "--ray-ascend-visible-devices" 1 "$@"
set_env_from_arg_or_default "RAY_USAGE_STATS_ENABLED" "--ray-usage-stats-enabled" 0 "$@"
set_env_from_arg_or_default "TP_SOCKET_IFNAME" "--tp-socket-ifname" "${SOCKET_IFNAME}" "$@"
