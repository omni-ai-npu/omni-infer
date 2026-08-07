#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

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
source "${CONFIG_DIR}"/env/set_decode_env.sh "$@"
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
sed -i "s|{{TRANSFER_TORCHAIR_CACHE}}|${TRANSFER_TORCHAIR_CACHE}|g" "${ROOT_DIR}"/health.sh
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

# 图编译缓存
if [ -n "${TORCHAIR_CACHE_PATH}" ]; then
    if [ "${ENABLE_TORCHAIR_CACHE:-}" = "1" ]; then
        echo_with_time "------------> prepare torchair cache"
        cp -Rf "${TORCHAIR_CACHE_PATH}" "${SCRIPT_DIR}"/
        echo_with_time "------------> prepare torchair cache finished"
    elif [ "${TRANSFER_TORCHAIR_CACHE:-}" = "1" ]; then
        echo_with_time "------------> prepare to transfer torchair cache"
        TORCHAIR_CACHE_PARENT_PATH=$( dirname "${TORCHAIR_CACHE_PATH}" )
        if [[ ${POD_IP} = "${HEAD_IP}" ]]; then
            rm -rf "${TORCHAIR_CACHE_PATH}"
            mkdir -p "${TORCHAIR_CACHE_PARENT_PATH}"
        fi
        sed -i "s|{{TORCHAIR_CACHE_SRC_PATH}}|${SCRIPT_DIR}/.torchair_cache|g" "${ROOT_DIR}"/health.sh
        sed -i "s|{{TORCHAIR_CACHE_DEST_PATH}}|${TORCHAIR_CACHE_PARENT_PATH}/|g" "${ROOT_DIR}"/health.sh
    else
        echo_with_time "------------> skip preparing torchair cache"
    fi
fi

if [ ${KV_CONNECTOR} == "LMCacheConnectorV1" ]; then
        export MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH
        export LMCACHE_CONFIG_FILE=$LMCACHE_MOONCAKE_CONFIG_PATH
        KV_ROLE="kv_both"
fi

echo_with_time "------------> start Omni service"
# python "${SCRIPT_DIR}"/process_nz_config.py "${CANN_JSON_PATH}"
cd "${SCRIPT_DIR}" || exit
export VLLM_DISABLE_COMPILE_CACHE=1
export TASK_QUEUE_ENABLE=1
export VLLM_CUSTOM_MODEL_ENABLE=False 
bash pd_run.sh \
    --global-rank-table-path "${!GLOBAL_RANK_TABLE_FILE_PATH_KEY}" \
    --rank-table-path "${RANK_TABLE_FILE_PATH}" \
    --local-decode-server-ip-list "${SERVER_IP_LIST}" \
    --global-decode-server-ip-list "${SERVER_IP_LIST}" \
    --prefill-pod-num "${PREFILL_POD_NUM}" \
    --gloo-socket-ifname "${SOCKET_IFNAME}" \
    --tp-socket-ifname "${SOCKET_IFNAME}" \
    --num-servers "${NUM_SERVERS}" \
    --num-dp "${dp}" \
    --server-offset "${SERVER_OFFSET}" \
    --model-path "${MODEL_PATH}" \
    --master-ip "${HEAD_IP}" \
    --role "${CURRENT_STARTUP_ROLE}" \
    --kv-role "${KV_ROLE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --master-port "${PD_PORT}" \
    --base-api-port "${PD_BASE_API_PORT}" \
    --tp "${TP}" \
    --vllm-logging-level "${VLLM_LOGGING_LEVEL}" \
    --kv-rank "${PREFILL_POD_NUM}" \
    --kv-engine-id "${PREFILL_POD_NUM}" \
    --kv-parallel-size "${KV_PARALLEL_SIZE}" \
    --kv-connector ${KV_CONNECTOR} \
    --gpu-util "${GPU_UTIL}" \
    --extra-args "${EXTRA_ARGS}" \
    --hccl-buffsize "${HCCL_BUFFSIZE}" \
    --num-speculative-tokens 0 \
    --hccl-op-expansion-mode "${HCCL_OP_EXPANSION_MODE}" \
    --additional-config "${ADDITIONAL_CONFIG}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --log-dir "${LOG_PATH}" 2>&1 | tee "${LOG_PATH}"/run_decode.log &
