#!/bin/bash

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
set_env_from_arg_or_default "MAX_MODEL_LEN" "--max-model-len" 65536 "$@"
set_env_from_arg_or_default "MAX_NUM_SEQS" "--max-num-seqs" 8 "$@"
set_env_from_arg_or_default "NUM_SPECULATIVE_TOKENS" "--num-speculative-tokens" 0 "$@"
set_env_from_arg_or_default "ADD_ARGS" "--add-args" "" "$@"
if [ ${NODE_IPS} ]; then
    D_RANKTABLE_LIST=()
    IFS=',' read -r -a arr <<< "$NODE_IPS"
    for i in "${!arr[@]}";do
        if [ $i -ge $(($NUM_PER_PREFILL_POD * $PREFILL_POD_NUM)) ]; then
            D_RANKTABLE_LIST+=("${arr[$i]}")
        fi
    done
    default_decode_ip_list=$(IFS=,; echo "${D_RANKTABLE_LIST[*]}")
    set_env "DECODE_SERVER_IP_LIST" "${default_decode_ip_list}" "Pod IPs of all Decode instances"

    export MA_CURRENT_IP="${MA_CURRENT_IP}"

    # 从 NODE_IPS 中提取第一个 IP
    IFS=',' read -r role_head_ip _ <<< "$DECODE_SERVER_IP_LIST"

    local_device_size=${local_device_size}

    # 取出当前ip对应的rank
    role_node_rank=-1
    rank=-1
    for ip in $(echo "$DECODE_SERVER_IP_LIST" | tr ',' '\n'); do
        rank=$((${rank} + 1))
        if [ "$ip" = "$MA_CURRENT_IP" ]; then
            role_node_rank="${rank}"
            break
        fi
    done

    ip_count=$(echo "$DECODE_SERVER_IP_LIST" | tr ',' '\n' | grep -c '^')
    role_device_size=$((${ip_count} * ${local_device_size}))

    # 统计 NODE_IPS 非空行个数
    role_node_size=$(echo "$DECODE_SERVER_IP_LIST" | tr ',' '\n' | grep -c '^')
    export ADDITIONAL_CONFIG=${DECODE_ADDITIONAL_CONFIG}
    echo "DECODE_ADDITIONAL_CONFIG is "${DECODE_ADDITIONAL_CONFIG}
    set_env_from_arg_or_default "EXTRA_ARGS" "--extra-args" "--disable-log-requests --enable-expert-parallel --max-num-seqs 60 --no-enable-prefix-caching --preemption-mode swap ${ADD_ARGS}" "$@"
    set_env_from_arg_or_default "MODEL_PATH" "--model-path" "/tmp/home/mind/model" "$@"
    set_env_from_arg_or_default "VLLM_ENABLE_MC2" "--vllm-enable-mc2" 0 "$@"
else
    set_env "RANK_TABLE_PATH" "${CONFIG_DIR}"/ranktable
    set_env "LOCAL_RANK_TABLE_FILE_NAME" "rank_table.json"
    set_env "RANK_TABLE_FILE_PATH" "${RANK_TABLE_PATH}"/"${LOCAL_RANK_TABLE_FILE_NAME}"
    set_env_from_arg_or_default "MODEL_PATH" "--model-path" "/home/mind/model" "$@"

    role_head_ip=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_head_ip.py)  #  done

    local_device_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_local_device_size.py)  #  每个ip的device数量

    role_node_rank=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_node_rank.py)  #  done

    role_device_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_device_size.py)  #  所有ip的device数量

    role_node_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_role_node_size.py)  #  done
    set_env_from_arg_or_default "VLLM_ENABLE_MC2" "--vllm-enable-mc2" 1 "$@"
    set_env_from_arg_or_default "ADDITIONAL_CONFIG" "--additional-config" '{"graph_model_compile_config": {"level":1, "use_ge_graph_cached":false}, "enable_omni_attn":false}' "$@"
    set_env_from_arg_or_default "EXTRA_ARGS" "--extra-args" "--enable-expert-parallel --disable-log-requests --max-num-seqs ${MAX_NUM_SEQS} --no-enable-prefix-caching --no-enable-chunked-prefill --preemption-mode swap --enable-reasoning --reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser=ascend_adapt_bf16 --chat-template ${CONFIG_DIR}/chattemplate/tool_chat_template_bf16_v3_clear.jinja --tool-parser-plugin=${CONFIG_DIR}/chattemplate/ascend_deepseekv31_tool_parser.py ${ADD_ARGS}" "$@"
fi
set_env "HEAD_IP" "${role_head_ip}" "Head Pod IP of Decode instance"
set_env "LOCAL_DEVICE_SIZE" "${local_device_size}" "Device count of single Pod"
set_env "NODE_RANK" "${role_node_rank}" "Pod index of Decode instance"
set_env "ROLE_DEVICE_SIZE" "${role_device_size}" "Device count of entire Decode instance"
set_env "ROLE_POD_SIZE" "${role_node_size}" "Pod count of entire Decode instance"

api_port_list=$(seq "${PD_BASE_API_PORT}" $((PD_BASE_API_PORT + LOCAL_DEVICE_SIZE - 1)) | tr '\n' ',' | sed 's/,$//')
role_servers=$(cross_join_ips_and_ports "${DECODE_SERVER_IP_LIST}" "${api_port_list}")
set_env "ROLE_SERVERS" "${role_servers}" "API servers of entire Decode instance"

set_env_from_arg_or_default "ENABLE_TORCHAIR_CACHE" "--enable-torchair-cache" 0 "$@"
set_env_from_arg_or_default "TORCHAIR_CACHE_PATH" "--torchair-cache-path" "" "$@"
set_env_from_arg_or_default "TRANSFER_TORCHAIR_CACHE" "--transfer-torchair-cache" 0 "$@"

# 启动参数使用的环境变量

set_env_from_arg_or_default "dp" "--num-dp" "${role_device_size}" "$@"
set_env_from_arg_or_default "GPU_UTIL" "--gpu-util" 0.8 "$@"
set_env_from_arg_or_default "HCCL_BUFFSIZE" "--hccl-buffsize" 400 "$@"
set_env_from_arg_or_default "HCCL_OP_EXPANSION_MODE" "--hccl-op-expansion-mode" "AIV" "$@"
set_env_from_arg_or_default "KV_ROLE" "--kv-role" "kv_consumer" "$@"
set_env_from_arg_or_default "NUM_SERVERS" "--num-servers" "${LOCAL_DEVICE_SIZE}" "$@"
set_env_from_arg_or_default "MODEL_EXTRA_CFG_PATH" "--model-extra-cfg-path" "${CODE_PATH}/tests/test_config/test_config_decode.json"

set_env_from_arg_or_default "SERVER_OFFSET" "--server-offset" "$((role_node_rank * LOCAL_DEVICE_SIZE))" "$@"
set_env_from_arg_or_default "TP" "--tp" 1 "$@"
set_env "KV_PARALLEL_SIZE" "${TP}" "the same as tp_size"

set_env_from_arg_or_default "VLLM_LOGGING_LEVEL" "--vllm-logging-level" "INFO" "$@"

# set_env_from_arg_or_default "VALIDATORS_CONFIG_PATH" "--validators-config-path" "${CONFIG_DIR}/validation/validators_config.json" "$@"

# vLLM使用的其它环境变量
set_env_from_arg_or_default "CPU_AFFINITY_CONF" "--cpu-affinity-conf" 2 "$@"
set_env_from_arg_or_default "HCCL_RDMA_TIMEOUT" "--hccl-rdma-timeout" 20 "$@"
set_env_from_arg_or_default "PYTORCH_NPU_ALLOC_CONF" "--pytorch-npu-alloc-conf" "expandable_segments:True" "$@"

# ray使用的环境变量
set_env_from_arg_or_default "GLOO_SOCKET_IFNAME" "--gloo-socket-ifname" "${SOCKET_IFNAME}" "$@"
set_env_from_arg_or_default "HCCL_SOCKET_IFNAME" "--hccl-socket-ifname" "${SOCKET_IFNAME}" "$@"
set_env_from_arg_or_default "TP_SOCKET_IFNAME" "--tp-socket-ifname" "${SOCKET_IFNAME}" "$@"