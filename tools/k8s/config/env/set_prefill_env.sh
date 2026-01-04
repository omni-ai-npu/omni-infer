#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${CONFIG_DIR:=$( dirname "${CUR_DIR}" )}"
: "${ROOT_DIR:=$( dirname "${CONFIG_DIR}" )}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/env/env_tools.sh

# 启动脚本使用的环境变量
set_env "CURRENT_STARTUP_ROLE" "prefill"
# 启动参数使用的环境变量

set_env_from_arg_or_default "GPU_UTIL" "--gpu-util" 0.90 "$@"
set_env_from_arg_or_default "HCCL_BUFFSIZE" "--hccl-buffsize" 50 "$@"
set_env_from_arg_or_default "HCCL_OP_EXPANSION_MODE" "--hccl-op-expansion-mode" "AIV" "$@"
set_env_from_arg_or_default "KV_RANK" "--kv-rank" "$((SERVER_GROUP_ID - 1))" "$@"
set_env_from_arg_or_default "KV_ROLE" "--kv-role" "kv_producer" "$@"
set_env_from_arg_or_default "LONG_PREFILL_TOKEN_THRESHOLD" "--long-prefill-token-threshold" 32768 "$@"
set_env_from_arg_or_default "MAX_MODEL_LEN" "--max-model-len" 65536 "$@"
set_env_from_arg_or_default "MAX_NUM_BATCHED_TOKENS" "--max-num-batched-tokens" 32768 "$@"
set_env_from_arg_or_default "MAX_NUM_SEQS" "--max-num-seqs" 8 "$@"
set_env_from_arg_or_default "NUM_SPECULATIVE_TOKENS" "--num-speculative-tokens" 0 "$@"
set_env_from_arg_or_default "MODEL_EXTRA_CFG_PATH" "--model-extra-cfg-path" "${CODE_PATH}/tests/test_config/test_config_prefill.json"

set_env_from_arg_or_default "VLLM_LOGGING_LEVEL" "--vllm-logging-level" "INFO" "$@"
echo "XXXXXXXXXXXXXXXXXXXXXXX set_prefill_env XXXXXXXXXXXXXXXXXXXXXXXXX"$@
set_env_from_arg_or_default "ADD_ARGS" "--add-args" "" "$@"

if [ ${NODE_IPS} ]; then
    D_RANKTABLE_LIST=()
    P_RANKTABLE_LIST=()
    IFS=',' read -r -a arr <<< "$NODE_IPS"
    for i in "${!arr[@]}";do
        if [ $i -eq $((${SERVER_GROUP_ID} * ${NUM_PER_PREFILL_POD})) ]; then
            role_head_ip="${arr[$i]}"
            break
        fi
    done
    local_device_size=${local_device_size}
    role_device_size=$((${local_device_size}*${NUM_PER_PREFILL_POD}))
    role_ip_list=()
    for i in "${!arr[@]}";do
        if [ $i -ge $((${SERVER_GROUP_ID} * ${NUM_PER_PREFILL_POD})) ] && [ $i -lt $(($((${SERVER_GROUP_ID} * ${NUM_PER_PREFILL_POD})) + ${NUM_PER_PREFILL_POD})) ]; then
            role_ip_list+=("${arr[$i]}")
            break
        fi
    done
    role_node_size=${NUM_PER_PREFILL_POD}
    role_node_rank=$((${PREFILL_RANK} % ${NUM_PER_PREFILL_POD}))
    export ADDITIONAL_CONFIG=${PREFILL_ADDITIONAL_CONFIG}
    echo "PREFILL_ADDITIONAL_CONFIG is "${PREFILL_ADDITIONAL_CONFIG}
    if [[ ${role_node_size} != "1" ]]; then
        set_env_from_arg_or_default "EXTRA_ARGS" "--extra-args" "--disable-log-requests --enable-expert-parallel --enforce-eager --max-num-batched-tokens 16384 --max-num-seqs 8 --no-enable-prefix-caching --distributed-executor-backend ray ${ADD_ARGS}" "$@"
    else
        set_env_from_arg_or_default "EXTRA_ARGS" "--extra-args" "--disable-log-requests --enable-expert-parallel --enforce-eager --max-num-batched-tokens 16384 --max-num-seqs 8 --no-enable-prefix-caching ${ADD_ARGS}" "$@"
    fi
    set_env_from_arg_or_default "MODEL_PATH" "--model-path" "/tmp/home/mind/model" "$@"
    set_env_from_arg_or_default "VLLM_ENABLE_MC2" "--vllm-enable-mc2" 0 "$@"
else
    set_env_from_arg_or_default "ADDITIONAL_CONFIG" "--additional-config" '' "$@"
    set_env "RANK_TABLE_PATH" "${CONFIG_DIR}"/ranktable
    set_env "LOCAL_RANK_TABLE_FILE_NAME" "rank_table.json"
    set_env "RANK_TABLE_FILE_PATH" "${RANK_TABLE_PATH}"/"${LOCAL_RANK_TABLE_FILE_NAME}"
    set_env_from_arg_or_default "MODEL_PATH" "--model-path" "/home/mind/model" "$@"
    set_env_from_arg_or_default "VLLM_ENABLE_MC2" "--vllm-enable-mc2" 1 "$@"
    role_head_ip=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_head_ip.py)
    local_device_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_local_device_size.py)
    role_device_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_device_size.py)
    role_ip_list=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_ip_list.py)
    role_node_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_node_size.py)
    role_node_rank=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_node_rank.py)
    if [[ ${ROLE_POD_SIZE} != "1" ]]; then
        set_env_from_arg_or_default "EXTRA_ARGS" "--extra-args" "--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs ${MAX_NUM_SEQS} --no-enable-prefix-caching --no-enable-chunked-prefill --enable-reasoning --reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser=ascend_adapt_bf16 --chat-template ${CONFIG_DIR}/chattemplate/tool_chat_template_bf16_v3_clear.jinja --tool-parser-plugin=${CONFIG_DIR}/chattemplate/ascend_deepseekv31_tool_parser.py --long-prefill-token-threshold ${LONG_PREFILL_TOKEN_THRESHOLD} --distributed-executor-backend ray ${ADD_ARGS}" "$@"
    else
        set_env_from_arg_or_default "EXTRA_ARGS" "--extra-args" "--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --enforce-eager --enable-expert-parallel --disable-log-requests --max-num-seqs ${MAX_NUM_SEQS} --no-enable-prefix-caching --no-enable-chunked-prefill --enable-reasoning --reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser=ascend_adapt_bf16 --chat-template ${CONFIG_DIR}/chattemplate/tool_chat_template_bf16_v3_clear.jinja --tool-parser-plugin=${CONFIG_DIR}/chattemplate/ascend_deepseekv31_tool_parser.py --long-prefill-token-threshold ${LONG_PREFILL_TOKEN_THRESHOLD} ${ADD_ARGS}" "$@"
    fi
fi


set_env "HEAD_IP" "${role_head_ip}" "Head Pod IP of Prefill instance"

set_env "LOCAL_DEVICE_SIZE" "${local_device_size}" "Device count of single Pod"

set_env "ROLE_DEVICE_SIZE" "${role_device_size}" "Device count of entire Prefill instance"

set_env "ROLE_IP_LIST" "${role_ip_list}" "Pod IPs of entire Prefill instance"

set_env "ROLE_POD_SIZE" "${role_node_size}" "Pod count of entire Prefill instance"
set_env "ROLE_SERVERS" "${HEAD_IP}:${PD_BASE_API_PORT}" "API servers of entire Prefill instance"

set_env_from_arg_or_default "TP" "--tp" "${role_device_size}" "$@"
set_env "KV_PARALLEL_SIZE" "${TP}" "the same as tp_size"

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