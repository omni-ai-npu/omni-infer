#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${ROOT_DIR:=${CUR_DIR}}"
: "${CONFIG_DIR:=${ROOT_DIR}/config}"
: "${ROLE_DIR:=${ROOT_DIR}/role}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh
source "${TOOL_DIR}"/env/env_tools.sh
source "${TOOL_DIR}"/grace/cleanup.sh
source "${TOOL_DIR}"/health/npu_checker.sh

# set SIGTERM trap
trap 'stop_all_processes_and_check_npus' SIGTERM

echo_with_time "------------> begin the process on A3 Omni 0.4.2"

# set entrypoint env
echo_with_time "------------> set entrypoint env"
unset RANK_TABLE_FILE
export GLOBAL_RANK_TABLE_FILE_PATH_KEY="GLOBAL_RANK_TABLE_FILE_PATH"
export TRANS_GLOBAL_RANK_TABLE_FILE_PATH="${CONFIG_DIR}"/ranktable/global_rank_table.json
echo_with_time "------------> set entrypoint env finished"

# check ranktable
echo_with_time "------------> start to check global rank table"
python3 "${TOOL_DIR}"/ranktable/rank_table_checker.py &
RANK_TABLE_CHECKER_PID=$!
wait "${RANK_TABLE_CHECKER_PID}"
echo "global rank table is as below:"
cat "${!GLOBAL_RANK_TABLE_FILE_PATH_KEY}"
echo_with_time "------------> check global rank table finished"

# gen global ranktable
echo_with_time "------------> start to gen global rank table"
python3 "${TOOL_DIR}"/ranktable/global_rank_table_generator.py
export GLOBAL_RANK_TABLE_FILE_PATH="${TRANS_GLOBAL_RANK_TABLE_FILE_PATH}"
export SERVER_GROUP_ID=$((SERVER_GROUP_ID - 1))
echo_with_time "------------> gen global rank table finished"

# set base env
source "$CONFIG_DIR"/env/set_base_env.sh "$@"

# prepare for pre-stop
{
    echo "export DECODE_SERVERS=${DECODE_SERVERS}"
    echo "export PREFILL_SERVERS=${PREFILL_SERVERS}"
    echo "export PRE_STOP_LOG_PATH=${PRE_STOP_LOG_PATH}"
} >> "$CONFIG_DIR"/env/set_pre_stop_env.sh

# prepare for log
echo_with_time "------------> prepare log path"
mkdir -p "${BASE_LOG_DIR}"
sed -i "s|{{PROBE_LOG_PATH}}|${PROBE_LOG_PATH}|g" "${ROOT_DIR}"/health.sh
echo_with_time "------------> prepare log path finished"

echo_with_time "------------> current group: ${SERVER_GROUP_ID}"
if [ "${SERVER_GROUP_ID}" -eq -1 ]; then
    echo_with_time "------------> current role: proxy"
    if [ "${PROXY_BACKEND}" = "global-proxy" ]; then
        bash "${ROLE_DIR}"/start_global_proxy.sh "$@"
    else
        bash "${ROLE_DIR}"/start_route_server.sh "$@"
    fi
elif [ "${SERVER_GROUP_ID}" -ge 0 ] && [ "${SERVER_GROUP_ID}" -lt "${PREFILL_POD_NUM}" ]; then
    echo_with_time "------------> current role: prefill"
    bash "${ROLE_DIR}"/start_prefill.sh "$@"
elif [ "${SERVER_GROUP_ID}" -ge "${PREFILL_POD_NUM}" ] && [ "${SERVER_GROUP_ID}" -lt "$((PREFILL_POD_NUM + DECODE_POD_NUM))" ]; then
    echo_with_time "------------> current role: decode"
    bash "${ROLE_DIR}"/start_decode.sh "$@"
fi

echo "hold process..."
loop_count=0
while true; do
    ((loop_count++))
    if [ ${loop_count} -ge "${MAIN_THREAD_LOOP_TASK_CYCLE}" ]; then
        python3 "${TOOL_DIR}"/ray/clean_ray_logs.py
        loop_count=0
    fi
    sleep "${MAIN_THREAD_SLEEP_INTERVAL}"
done
