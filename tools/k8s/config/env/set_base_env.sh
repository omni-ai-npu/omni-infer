#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${CONFIG_DIR:=$( dirname "${CUR_DIR}" )}"
: "${ROOT_DIR:=$( dirname "${CONFIG_DIR}" )}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${HOME}"/.bashrc
source "${TOOL_DIR}"/basic/endpoint_tools.sh
source "${TOOL_DIR}"/env/env_tools.sh

# 启动脚本使用的环境变量

set_env_from_arg_or_default "CODE_PATH" "--ascend-path" "/workspace/omniinfer" "$@"

set_env_from_arg_or_default "MAIN_THREAD_SLEEP_INTERVAL" "--main-thread-sleep-interval" 5 "$@"
set_env_from_arg_or_default "MAIN_THREAD_LOOP_TASK_CYCLE" "--main-thread-loop-task-cycle" 1440 "$@"
set_env_from_arg_or_default "PORT" "--port" 8080 "$@"


set_env_from_arg_or_default "PROXY_BACKEND" "--proxy-backend" "global-proxy" "$@"

set_env_from_arg_or_default "RAY_LOG_DIR" "--ray-log-dir" "/tmp/ray" "$@"
set_env_from_arg_or_default "RAY_LOG_TO_KEEP_IN_DAY" "--ray-log-to-keep-in-day" 7 "$@"
set_env_from_arg_or_default "RAY_PORT" "--ray-port" 6379 "$@"
set_env_from_arg_or_default "SCRIPT_DIR" "--script-dir" "${CODE_PATH}/tools/scripts" "$@"

# 启动参数使用的环境变量
set_env_from_arg_or_default "SERVED_MODEL_NAME" "--served-model-name" "DeepSeek" "$@"


set_env_from_arg_or_default "PD_BASE_API_PORT" "--pd-base-api-port" "$((PORT + 200))" "$@"
set_env_from_arg_or_default "PD_PORT" "--pd-port" $((PORT + 100)) "$@"
set_env_from_arg_or_default "PROXY_PORT" "--proxy-port" "${PORT}" "$@"

# vLLM使用的环境变量
set_env_from_arg_or_default "ASCEND_GLOBAL_LOG_LEVEL" "--ascend-global_log_level" 3 "$@"
set_env_from_arg_or_default "DECODE_POD_NUM" "--decode-pod-size" 1 "$@"
set_env_from_arg_or_default "ENABLE_LOGGING_CONFIG" "--enable-logging-config" 0 "$@"
set_env_from_arg_or_default "HCCL_IF_BASE_PORT" "--hccl-if-base-port" 32000 "$@"
if [ ${NODE_IPS} ]; then
    D_RANKTABLE_LIST=()
    P_RANKTABLE_LIST=()
    IFS=',' read -r -a arr <<< "$NODE_IPS"
    for i in "${!arr[@]}";do
        if [ $i -ge $(($NUM_PER_PREFILL_POD * $PREFILL_POD_NUM)) ]; then
            D_RANKTABLE_LIST+=("${arr[$i]}")
        fi
        if [ $i -lt $(($NUM_PER_PREFILL_POD * $PREFILL_POD_NUM)) ] && [ $(($i % ${NUM_PER_PREFILL_POD})) -eq 0 ]; then
            P_RANKTABLE_LIST+=("${arr[$i]}")
        fi
    done
    prefill_head_ip_list=$(IFS=,; echo "${P_RANKTABLE_LIST[*]}")

    default_decode_ip_list=$(IFS=,; echo "${D_RANKTABLE_LIST[*]}")
    set_env "DECODE_SERVER_IP_LIST" "${default_decode_ip_list}" "Pod IPs of all Decode instances"
    set_env_from_arg_or_default "BASE_LOG_DIR" "--base-log-dir" "/tmp/workspace/mis/logs" "$@"
    set_env_from_arg_or_default "GLOBAL_RANK_TABLE_FILE_PATH" "--global-rank-table-file-path" "/tmp/user/global/config/global_rank_table.json" "$@"
    set_env_from_arg_or_default "SOCKET_IFNAME" "--socket-ifname" "${SOCKET_IFNAME}" "$@"
else
    set_env_from_arg_or_default "BASE_LOG_DIR" "--base-log-dir" "/workspace/mis/logs" "$@"
    set_env_from_arg_or_default "GLOBAL_RANK_TABLE_FILE_PATH" "--global-rank-table-file-path" "/user/global/config/global_rank_table.json" "$@"
    set_env_from_arg_or_default "PRE_STOP_LOG_PATH" "--pre-stop-log-path" "${BASE_LOG_DIR}/pre_stop.log" "$@"
    set_env_from_arg_or_default "PROBE_LOG_PATH" "--probe-log-path" "${BASE_LOG_DIR}/probe.log" "$@"
    default_prefill_instance_size=$(python3 -u "${TOOL_DIR}"/ranktable/get_default_prefill_instance_size.py)
    default_decode_ip_list=$(python3 -u "${TOOL_DIR}"/ranktable/get_default_decode_ip_list.py)
    prefill_head_ip_list=$(python3 -u "${TOOL_DIR}"/ranktable/get_prefill_head_ip_list.py)
    set_env_from_arg_or_default "SOCKET_IFNAME" "--socket-ifname" "eth0" "$@"
fi
set_env_from_arg_or_default "RAY_LOG_CLEANER_LOG_PATH" "--ray-log-cleaner-log-path" "${BASE_LOG_DIR}/ray_log_cleaner.log" "$@"
set_env_from_arg_or_default "LOG_PATH" "--log-path" "${BASE_LOG_DIR}" "$@"
set_env_from_arg_or_default "LOG_PATH_IN_EXECUTOR" "--log-path-in-executor" "${BASE_LOG_DIR}" "$@"
set_env_from_arg_or_default "PREFILL_POD_NUM" "--prefill-pod-size" "${default_prefill_instance_size}" "$@"


if [ "${ENABLE_LOGGING_CONFIG}" = "1" ]; then
    set_env_from_arg_or_default "VLLM_LOGGING_CONFIG_PATH" "--vllm-logging-config-path" "${SCRIPT_DIR}/logging_config.json" "$@"
fi


set_env "DECODE_SERVER_IP_LIST" "${default_decode_ip_list}" "Pod IPs of all Decode instances"
default_kv_parallel_size=$((PREFILL_POD_NUM + 1))
set_env "KV_PARALLEL_SIZE" "${default_kv_parallel_size}" "1 more than ENV PREFILL_POD_NUM"
set_env "SERVER_IP_LIST" "${DECODE_SERVER_IP_LIST}" "Identical to ENV DECODE_SERVER_IP_LIST"

if [ ${NUM_SERVERS} ]; then
    NUM_SERVERS=$NUM_SERVERS
else
    NUM_SERVERS=${local_device_size}
fi
api_port_list=$(seq "${PD_BASE_API_PORT}" $((PD_BASE_API_PORT + ${NUM_SERVERS} - 1)) | tr '\n' ',' | sed 's/,$//')
decode_endpoints=$(cross_join_ips_and_ports "${DECODE_SERVER_IP_LIST}" "${api_port_list}")
set_env "DECODE_SERVERS" "${decode_endpoints}" "Endpoints of all decode api servers"


prefill_endpoints=$(cross_join_ips_and_ports "${prefill_head_ip_list}" "${PD_BASE_API_PORT}")
set_env "PREFILL_SERVERS" "${prefill_endpoints}" "Endpoints of all prefill api servers"