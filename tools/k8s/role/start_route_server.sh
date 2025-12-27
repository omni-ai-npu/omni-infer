#!/bin/bash

# set default env
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
: "${ROOT_DIR:=$( dirname "${CUR_DIR}" )}"
: "${CONFIG_DIR:=${ROOT_DIR}/config}"
: "${TOOL_DIR:=${ROOT_DIR}/tool}"

# import dependent tool
source "${TOOL_DIR}"/basic/basic_shell_tools.sh
source "${TOOL_DIR}"/health/npu_checker.sh

# set specific env
echo_with_time "------------> set environment variable for ${CURRENT_STARTUP_ROLE}"
source "${CONFIG_DIR}"/env/set_prefill_env.sh "$@"
echo_with_time "------------> set environment variable for ${CURRENT_STARTUP_ROLE} finished"

echo_with_time "------------> check if NPUs are ready for ${CURRENT_STARTUP_ROLE} (target: ${LOCAL_DEVICE_SIZE})"
if all_npu_ready "${LOCAL_DEVICE_SIZE}"; then
    echo "NPUs are ready (NPU count matches)"
else
    exit 1
fi
echo_with_time "------------> check if NPUs are ready for ${CURRENT_STARTUP_ROLE} finished"

echo_with_time "------------> set variable of health.sh for ${CURRENT_STARTUP_ROLE}"
sed -i "s|{{CURRENT_PD_ROLE}}|${CURRENT_STARTUP_ROLE}|g" "${ROOT_DIR}"/health.sh
sed -i "s|{{ENDPOINTS}}|${ROLE_SERVERS}|g" "${ROOT_DIR}"/health.sh
if [[ ${POD_IP} = "${HEAD_IP}" ]]; then
    sed -i "s|{{IS_HEAD}}|1|g" "${ROOT_DIR}"/health.sh
else
    sed -i "s|{{IS_HEAD}}|0|g" "${ROOT_DIR}"/health.sh
fi
echo_with_time "------------> set variable of health.sh for ${CURRENT_STARTUP_ROLE} finished"

# generate ranktable
echo_with_time "------------> generate local ranktable for ${CURRENT_STARTUP_ROLE}"
python3 "${TOOL_DIR}"/ranktable/local_rank_table_generator.py
echo_with_time "------------> generate local ranktable for ${CURRENT_STARTUP_ROLE} finished"

echo_with_time "------------> check all environment variable for ${CURRENT_STARTUP_ROLE}"
env
echo_with_time "------------> check all environment variable for ${CURRENT_STARTUP_ROLE} finished"

echo_with_time "------------> start Omni service"
export TOOL_DIR="${TOOL_DIR}"
cd "${SCRIPT_DIR}" || exit
bash pd_run.sh \
    --global-rank-table-path "${!GLOBAL_RANK_TABLE_FILE_PATH_KEY}" \
    --rank-table-path "${RANK_TABLE_FILE_PATH}" \
    --local-decode-server-ip-list "${SERVER_IP_LIST}" \
    --global-decode-server-ip-list "${SERVER_IP_LIST}" \
    --prefill-pod-num "${PREFILL_POD_NUM}" \
    --gloo-socket-ifname "${SOCKET_IFNAME}" \
    --tp-socket-ifname "${SOCKET_IFNAME}" \
    --model-path "${MODEL_PATH}" \
    --master-ip "${HEAD_IP}" \
    --role "${CURRENT_STARTUP_ROLE}" \
    --kv-role "${KV_ROLE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --master-port "${PD_PORT}" \
    --base-api-port "${PD_BASE_API_PORT}" \
    --tp "${TP}" \
    --vllm-logging-level "${VLLM_LOGGING_LEVEL}" \
    --ascend-rt-visible-devices "${ASCEND_RT_VISIBLE_DEVICES}" \
    --kv-rank "${KV_RANK}" \
    --kv-engine-id "${KV_RANK}" \
    --kv-parallel-size "${KV_PARALLEL_SIZE}" \
    --gpu-util "${GPU_UTIL}" \
    --vllm-enable-mc2 "${VLLM_ENABLE_MC2}" \
    --extra-args "${EXTRA_ARGS}" \
    --hccl-buffsize "${HCCL_BUFFSIZE}" \
    --hccl-op-expansion-mode "${HCCL_OP_EXPANSION_MODE}" \
    --additional-config "${ADDITIONAL_CONFIG}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --log-dir "${LOG_PATH}" 2>&1 | tee "${LOG_PATH}"/run_prefill.log &