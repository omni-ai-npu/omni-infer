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

# 等待文件存在（替换固定sleep）
wait_for_file() {
  local file_path="$1"
  local timeout="${2:-300}"
  local interval=1
  local elapsed=0

  echo "等待文件: $file_path（超时时间: $timeout 秒）"
  while true; do
    if [ -f "$file_path" ]; then
      echo "文件已存在: $file_path"
      return 0
    fi
    if [ $elapsed -ge $timeout ]; then
      echo "错误：等待文件超时（$timeout 秒）: $file_path"
      exit 1
    fi
    sleep $interval
    elapsed=$((elapsed + interval))
  done
}
# 适配云道场景生产ranktable
echo "-----------------------------DECODE XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------------------"
echo "XXXXXXXXXXXXXXXXXXXX DECODE NODE_IPS is "$NODE_IPS
echo "-----------------------------DECODE XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------------------"
if [ ${NODE_IPS} ]; then
    # 生成Ranktable
    echo "生成DECODE节点Ranktable..."
    python "${CODE_PATH}/tools/scripts/pd_ranktable_tools.py" \
      --mode gen \
      --decode-server-list "${device_list}" \
      --save-dir "$DECODE_RANKTABLE_SAVE_PATH"
    # 打印环境变量（调试用）
    echo "======== DECODE节点环境变量 ========"
    env | grep -E '^(ROLE|HOST_IP|MODEL_LEN_MAX_DECODE|DECODE_|KV_|GPU_UTIL|SERVER_OFFSET)' | sort
    echo "======================================"
    echo "NUM_PER_DECODE_POD is "${NUM_PER_DECODE_POD}
    PREFILL_SERVER_LIST=()
    DECODE_SERVER_LIST=()
    # 如果是多机组P的场景
    if [ "${NUM_PER_PREFILL_POD}" -gt 1 ]; then
        for i in "${!arr[@]}";do
            # 收集当前实例下所有从节点的ranktable
            if [ $i -lt $((${PREFILL_POD_NUM} * ${NUM_PER_PREFILL_POD})) ] && [ $(($i % ${NUM_PER_PREFILL_POD})) -eq 0 ]; then
                # 等待Ranktable文件生成
                wait_for_file "$PREFILL_RANKTABLE_SAVE_PATH/merge/${arr[$i]}/local_ranktable_merge.json"
                PREFILL_SERVER_LIST+=("$PREFILL_RANKTABLE_SAVE_PATH/merge/${arr[$i]}/local_ranktable_merge.json")
            fi
        done
    else
        for i in "${!arr[@]}";do
            # 收集当前实例下所有从节点的ranktable
            if [ $i -lt $((${PREFILL_POD_NUM} * ${NUM_PER_PREFILL_POD})) ]; then
                # 等待Ranktable文件生成
                wait_for_file $PREFILL_RANKTABLE_SAVE_PATH"/local_ranktable_"${arr[$i]}"_${device_collection}.json"
                PREFILL_SERVER_LIST+=($PREFILL_RANKTABLE_SAVE_PATH"/local_ranktable_"${arr[$i]}"_${device_collection}.json")
            fi
        done
    fi
    # 若是多机组D的场景(默认场景）
    if [ "${NUM_PER_DECODE_POD}" -gt 1 ]; then
        # 获取当前实例的master节点
        echo "多机组D场景"
        echo "DECODE_RANK is" ${DECODE_RANK}" master_rank is "$(($((${SERVER_GROUP_ID} - ${PREFILL_POD_NUM})) * ${NUM_PER_DECODE_POD}))
        IFS=',' read -r -a arr <<< "$NODE_IPS"
        # 如果是decode主节点，则需要merge
        if [ $((${DECODE_RANK} % $NUM_PER_DECODE_POD)) -eq 0 ]; then
            # 等待所有Decode节点Ranktable生成（给从节点20秒生成时间）
            echo "当前是主节点，创建merge目录"
            if [ ! -d "$DECODE_RANKTABLE_SAVE_PATH/merge/${MA_CURRENT_IP}" ]; then
                echo "创建merge目录"
                mkdir -p "$DECODE_RANKTABLE_SAVE_PATH/merge/${MA_CURRENT_IP}"
            fi
            sleep 20
            # 动态生成Decode节点Ranktable文件名列表
            D_RANKTABLE_LIST=()

            for i in "${!arr[@]}";do
                rank_temp=$(($i - $((${PREFILL_POD_NUM} * ${NUM_PER_PREFILL_POD}))))
                # 收集当前实例下所有从节点的ranktable
                if [ ${rank_temp} -ge ${DECODE_RANK} ] && [ ${rank_temp} -lt $(("${DECODE_RANK}" + "${NUM_PER_DECODE_POD}")) ]; then
                    D_RANKTABLE_LIST+=("local_ranktable_"${arr[$i]}"_${device_collection}.json")
                fi
            done

            # 合并本地Decode Ranktable
            echo "合并所有Decode节点Ranktable: ${D_RANKTABLE_LIST[@]}"
            cd "$DECODE_RANKTABLE_SAVE_PATH" || exit 1
            python "${CODE_PATH}/tools/scripts/pd_ranktable_tools.py" \
              --mode merge-local \
              --local-ranktable-list "${D_RANKTABLE_LIST[@]}" \
              --save-dir "$DECODE_RANKTABLE_SAVE_PATH/merge/${MA_CURRENT_IP}"
            for i in "${!arr[@]}";do
                # 收集当前实例下所有从节点的ranktable
                PREFILL_TOTAL_NUM_TEMP=$((${PREFILL_POD_NUM} * ${NUM_PER_PREFILL_POD}))
                if [ $i -ge ${PREFILL_TOTAL_NUM_TEMP} ] && [ $(($(($i - ${PREFILL_TOTAL_NUM_TEMP})) % ${NUM_PER_DECODE_POD})) -eq 0 ]; then
                    # 等待Ranktable文件生成
                    wait_for_file "$DECODE_RANKTABLE_SAVE_PATH/merge/${arr[$i]}/local_ranktable_merge.json"
                    DECODE_SERVER_LIST+=("$DECODE_RANKTABLE_SAVE_PATH/merge/${arr[$i]}/local_ranktable_merge.json")
                fi
            done

            # 合并Prefill和Decode的Ranktable
            echo "合并Prefill和Decode节点Ranktable"
            echo "prefill-server-list is "${PREFILL_SERVER_LIST[@]}
            echo "decode-server-list is "${DECODE_SERVER_LIST[@]}
            python "${CODE_PATH}/tools/scripts/pd_ranktable_tools.py" \
              --mode merge-all \
              --prefill-server-list "${PREFILL_SERVER_LIST[@]}" \
              --decode-server-list "${DECODE_SERVER_LIST[@]}" \
              --save-dir "$DECODE_RANKTABLE_SAVE_PATH"

            # 复制Ranktable到全局目录
            cp -r "$DECODE_RANKTABLE_SAVE_PATH"/* "$RANK_TABLE_PATH"/
            cp -r "$DECODE_RANKTABLE_SAVE_PATH"/* "$RANK_TABLE_PATH_D"/
            cp -r "$DECODE_RANKTABLE_SAVE_PATH"/global_ranktable_merge.json "$PREFILL_RANKTABLE_SAVE_PATH"/
            echo "已复制合并后的Ranktable到全局目录: $RANK_TABLE_PATH"
            # 等待全局Ranktable文件生成
            wait_for_file "$GLOBAL_RANK_TABLE_FILE_PATH"
            sleep 60  # 给从节点足够时间同步Ranktable
        else
            # 等待主节点合并Ranktable
            echo "当前是从节点, 等待主节点merge local_rank_table以及全局ranktable"
            wait_for_file "$DECODE_RANKTABLE_SAVE_PATH/global_ranktable_merge.json"
            wait_for_file "$DECODE_RANKTABLE_SAVE_PATH/merge/${role_head_ip}/local_ranktable_merge.json"
            # 复制Ranktable到全局目录
            cp -r "$DECODE_RANKTABLE_SAVE_PATH"/* "$RANK_TABLE_PATH"/
            cp -r "$DECODE_RANKTABLE_SAVE_PATH"/* "$RANK_TABLE_PATH_D"/
            echo "已同步主节点Ranktable到本地全局目录: $RANK_TABLE_PATH"
        fi
        RANK_TABLE_FILE_PATH="$RANK_TABLE_PATH/merge/${role_head_ip}/local_ranktable_merge.json"
    else
        echo "单机组D场景"
        DECODE_SERVER_LIST="${DECODE_RANKTABLE_SAVE_PATH}/local_ranktable_"${MA_CURRENT_IP}"_${device_collection}.json"
        # 合并Prefill和Decode的Ranktable
        echo "合并Prefill和Decode节点Ranktable"
        echo "prefill-server-list is "${PREFILL_SERVER_LIST[@]}
        echo "decode-server-list is "${DECODE_SERVER_LIST[@]}
        python "${CODE_PATH}/tools/scripts/pd_ranktable_tools.py" \
          --mode merge-all \
          --prefill-server-list "${PREFILL_SERVER_LIST[@]}" \
          --decode-server-list "${DECODE_SERVER_LIST[@]}" \
          --save-dir "$DECODE_RANKTABLE_SAVE_PATH"
        # 复制Ranktable到全局目录
        cp -r "$DECODE_RANKTABLE_SAVE_PATH"/* "$RANK_TABLE_PATH"/
        cp -r "$DECODE_RANKTABLE_SAVE_PATH"/* "$RANK_TABLE_PATH_D"/
        cp -r "$DECODE_RANKTABLE_SAVE_PATH"/global_ranktable_merge.json "$PREFILL_RANKTABLE_SAVE_PATH"/
        echo "已复制合并后的Ranktable到全局目录: $RANK_TABLE_PATH"
        # 等待全局Ranktable文件生成
        wait_for_file "$GLOBAL_RANK_TABLE_FILE_PATH"
        RANK_TABLE_FILE_PATH="$RANK_TABLE_PATH/local_ranktable_"${MA_CURRENT_IP}"_${device_collection}.json"
        sleep 60  # 给从节点足够时间同步Ranktable
    fi
    # 适配图缓存
    graph_cache=$(echo "$ADDITIONAL_CONFIG" | sed -n 's/.*"use_ge_graph_cached"[[:space:]]*:[[:space:]]*\([^,}]*\).*/\1/p')
    echo_with_time "graph_cache is "${graph_cache} >> ${LOG_PATH}/server.log 2>&1

    if [[ "${graph_cache}" == "true" ]]; then
        if [ ${SUB_COMPILE_CACHE_PATH} ]; then
            COMPILE_CACHE_PATH="${MODEL_PATH}/$SUB_COMPILE_CACHE_PATH"
        else
            new_sub_dir="/compile_cache"
            COMPILE_CACHE_PATH="${MODEL_PATH}/$new_sub_dir"
        fi
        COMPILE_CACHE_PATH="${COMPILE_CACHE_PATH}/pod_${DECODE_RANK}"
        if [ -d "$COMPILE_CACHE_PATH" ]; then
            echo "Directory $COMPILE_CACHE_PATH already exists." >> ${LOG_PATH}/server.log 2>&1
        else
            echo "Creating directory $COMPILE_CACHE_PATH ..." >> ${LOG_PATH}/server.log 2>&1
            mkdir -p "$COMPILE_CACHE_PATH"
            echo "Directory created." >> ${LOG_PATH}/server.log 2>&1
        fi

        echo "COMPILE_CACHE_PATH full path is $COMPILE_CACHE_PATH" >> ${LOG_PATH}/server.log 2>&1
        export TORCHAIR_CACHE_HOME=${COMPILE_CACHE_PATH}
    fi
else
    echo_with_time "------------> generate local ranktable for ${CURRENT_STARTUP_ROLE}"
    python3 "${TOOL_DIR}"/ranktable/local_rank_table_generator.py
    echo_with_time "------------> generate local ranktable for ${CURRENT_STARTUP_ROLE} finished"

    echo_with_time "------------> check all environment variable for ${CURRENT_STARTUP_ROLE}"
    env
    echo_with_time "------------> check all environment variable for ${CURRENT_STARTUP_ROLE} finished"
fi
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

echo_with_time "------------> start Omni service"
# python "${SCRIPT_DIR}"/process_nz_config.py "${CANN_JSON_PATH}"
cd "${SCRIPT_DIR}" || exit
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
    --gpu-util "${GPU_UTIL}" \
    --vllm-enable-mc2 "${VLLM_ENABLE_MC2}" \
    --extra-args "${EXTRA_ARGS}" \
    --hccl-buffsize "${HCCL_BUFFSIZE}" \
    --hccl-op-expansion-mode "${HCCL_OP_EXPANSION_MODE}" \
    --additional-config "${ADDITIONAL_CONFIG}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" \
    --log-dir "${LOG_PATH}" 2>&1 | tee "${LOG_PATH}"/run_decode.log &