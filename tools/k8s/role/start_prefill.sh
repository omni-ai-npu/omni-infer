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
echo "-----------------------------PREFILL XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------------------"
echo "XXXXXXXXXXXXXXXXXXXX PREFILL NODE_IPS is "$NODE_IPS
echo "-----------------------------PREFILL XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-----------------------------------"
# 等待Ray主节点就绪
wait_ray_head() {
  local ray_head_ip="$1"
  local ray_head_port="${2:-6379}"
  local timeout=300
  local interval=5
  local elapsed=0

  echo "等待Ray主节点就绪: $ray_head_ip:$ray_head_port（超时时间: $timeout 秒）"
  while true; do
    if ray status --address="$ray_head_ip:$ray_head_port" &> /dev/null; then
      echo "Ray主节点已就绪: $ray_head_ip:$ray_head_port"
      return 0
    fi
    if [ $elapsed -ge $timeout ]; then
      echo "错误：等待Ray主节点超时（$timeout 秒）: $ray_head_ip:$ray_head_port"
      exit 1
    fi
    sleep $interval
    elapsed=$((elapsed + interval))
  done
}
if [ ${NODE_IPS} ]; then
    # 生成Ranktable
    echo "云道场景生成PREFILL节点Ranktable..."
    python "${CODE_PATH}/tools/scripts/pd_ranktable_tools.py" \
      --mode gen \
      --prefill-server-list "${device_list}" \
      --save-dir "$PREFILL_RANKTABLE_SAVE_PATH"
    # 若是多机组P的场景
    if [ "${NUM_PER_PREFILL_POD}" -gt 1 ]; then
        # 若当前节点是prefill主节点，则需要merge
        if [ $((${PREFILL_RANK} % $NUM_PER_PREFILL_POD)) -eq 0 ];then
            if [ ! -d "$PREFILL_RANKTABLE_SAVE_PATH/merge/${MA_CURRENT_IP}" ]; then
                mkdir -p "$PREFILL_RANKTABLE_SAVE_PATH/merge/${MA_CURRENT_IP}"
            fi
            sleep 20
            IFS=',' read -r -a arr <<< "$NODE_IPS"
            # 动态生成Prefill节点Ranktable文件名列表
            P_RANKTABLE_LIST=()
            for i in "${!arr[@]}";do
                # 收集当前实例下所有从节点的ranktable
                if [ $i -ge ${PREFILL_RANK} ] && [ $i -lt $(("${PREFILL_RANK}" + "${NUM_PER_PREFILL_POD}")) ]; then
                    P_RANKTABLE_LIST+=("local_ranktable_"${arr[$i]}"_${device_collection}.json")
                fi
            done

            # 合并本地Prefill Ranktable
            echo "合并所有PREFILL节点Ranktable"
            cd "$PREFILL_RANKTABLE_SAVE_PATH" || exit 1
            python "${CODE_PATH}/tools/scripts/pd_ranktable_tools.py" \
              --mode merge-local \
              --local-ranktable-list "${P_RANKTABLE_LIST[@]}" \
              --save-dir "$PREFILL_RANKTABLE_SAVE_PATH/merge/${MA_CURRENT_IP}"
            RANK_TABLE_FILE_PATH="$RANK_TABLE_PATH/merge/${MA_CURRENT_IP}/local_ranktable_merge.json"
            export RANK_TABLE_FILE_PATH=${RANK_TABLE_FILE_PATH}
            echo "local rank_table_file_path is "${RANK_TABLE_FILE_PATH}
            # 启动ray
            cd "${CODE_PATH}/tools/scripts" || exit 1
            # 停止现有Ray进程
            ray stop --force &> /dev/null
            echo "已停止现有Ray进程"
            # 启动Ray主节点
            ray start --head \
            --num-gpus=1
            >> "${LOG_PATH}" 2>&1
            echo "Ray主节点启动中..."
            wait_ray_head "$MA_CURRENT_IP" 6379  # 等待Ray就绪
        else
            RANK_TABLE_FILE_PATH="$RANK_TABLE_PATH/merge/${role_head_ip}/local_ranktable_merge.json"
            # 等待Ray主节点就绪
            wait_ray_head "$HEAD_IP" 6379

            export RANK_TABLE_FILE_PATH=${RANK_TABLE_FILE_PATH}
            echo "local rank_table_file_path is "${RANK_TABLE_FILE_PATH}
            # 启动Ray从节点
            echo "启动Ray从节点..."
            ray_start_cmd="ray start \
            --address=\"$HEAD_IP:6379\" \
            --num-gpus=1 \
            &> /dev/null"
            cost_time=0
            end_time=300
            while true; do
            if [ $cost_time -ge $end_time ]; then
                echo "错误：连接Ray主节点超时" >> "$LOG_PATH/omni_cli.log"
                exit 1
            fi

            eval "$ray_start_cmd"
            if [ $? -eq 0 ]; then
                echo "Ray从节点连接成功: $MA_CURRENT_IP -> $HEAD_IP:6379" >> "$LOG_PATH/omni_cli.log"
                break
            else
                echo "Ray从节点连接失败，等待5秒重试..." >> "$LOG_PATH/omni_cli.log"
                sleep 5
                cost_time=$((cost_time + 5))
            fi
            done

            echo "PREFILL从节点启动成功，日志路径: $LOG_PATH"
        fi
    else
        echo "单机组P场景"
        RANK_TABLE_FILE_PATH="$RANK_TABLE_PATH/local_ranktable_"${MA_CURRENT_IP}"_${device_collection}.json"
    fi
    # 等待全局Ranktable文件生成
    wait_for_file "$PREFILL_RANKTABLE_SAVE_PATH/global_ranktable_merge.json"

    # 复制Ranktable到本地目录
    cp -r "$PREFILL_RANKTABLE_SAVE_PATH"/* "$RANK_TABLE_PATH"/
    cp -r "$PREFILL_RANKTABLE_SAVE_PATH"/* "$RANK_TABLE_PATH_P"/
    echo "已复制Ranktable到本地目录: $RANK_TABLE_PATH"

    # 打印环境变量（调试用）
    echo "======== PREFILL节点环境变量 ========"
    # env | grep -E '^(ROLE|HOST_IP|MODEL_LEN_MAX_PREFILL|PREFILL_|RAY_|KV_|GPU_UTIL)' | sort
    echo "======================================"
else
    # generate ranktable
    echo_with_time "------------> generate local ranktable for ${CURRENT_STARTUP_ROLE}"
    python3 "${TOOL_DIR}"/ranktable/local_rank_table_generator.py
    echo_with_time "------------> generate local ranktable for ${CURRENT_STARTUP_ROLE} finished"
fi

echo_with_time "------------> check all environment variable for ${CURRENT_STARTUP_ROLE}"
env
echo_with_time "------------> check all environment variable for ${CURRENT_STARTUP_ROLE} finished"

echo_with_time "------------> start Omni service"
export TOOL_DIR="${TOOL_DIR}"

cd "${SCRIPT_DIR}" || exit
if [ ! ${NODE_IPS} ] || [ $((${PREFILL_RANK} % $NUM_PER_PREFILL_POD)) -eq 0 ]; then
    echo "start to exec pd_run.sh"
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
        --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" \
        --log-dir "${LOG_PATH}" 2>&1 | tee "${LOG_PATH}"/run_prefill.log &
fi