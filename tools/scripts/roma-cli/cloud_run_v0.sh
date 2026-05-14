#!/bin/bash

required_vars=(
    "MODEL_PATH"
    "MODEL_NAME"
    "START_SCRIPT"
    "SERVICE_NAME"
    "BUCKET_PATH"
    "MOUNT_PATH"
)

BASE_LOG_PATH_I="/home/ma-user/logs"
PROXY_PORT_I=7000
SERVED_MODEL_NAME_I="roma-model"   # changed by SERVED_MODEL_NAME env, used for vllm serve --served-model-name
CODE_PATH="/workspace/omniinfer"
NIC_NAME="eth0"
YAML_FILE="server_profiles.yml"

TIME_OUT="3000s"
IP_LIST=""
LOCAL_IP=""
PREFILL_SERVER_LIST=()
DECODE_SERVER_LIST=()
NODE_INDEX=-1
NODE_NAME="default_name"
NODE_ROLE="default"
LOG_PATH=""
RANKTABLE_SAVE_PATH=""
PROXY_IP=""
NODE_IPS=""
P_COUNT=0
D_COUNT=0
TMP_SCRIPTS_PATH="/home/ma-user/tmp_scripts"
ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMNI_CACHE_MEMMAP_PATH=/dev/shm/omni_cache
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 0 Read environment variables
get_env() {
    for var in "${required_vars[@]}"; do [ -z "${!var}" ] && echo "error: $var not set" && exit 1; done && echo "check env ok"
    CLOUD_SCRIPT_PATH="$START_SCRIPT"
    CLOUD_SCRIPT_PATH=$(dirname "$CLOUD_SCRIPT_PATH")
    BASE_LOG_PATH="$MOUNT_PATH/$BUCKET_PATH/omni-elb/logs/$MODEL_NAME/$SERVICE_NAME"
    export CLOUD_SCRIPT_PATH=${CLOUD_SCRIPT_PATH}
    export MODEL_PATH=${MODEL_PATH}
    export BASE_LOG_PATH=${BASE_LOG_PATH:-$BASE_LOG_PATH_I}
    export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$SERVED_MODEL_NAME_I}
    export PROXY_PORT=${PROXY_PORT:-$PROXY_PORT_I}
}

get_node_info() {
    cd ${CLOUD_SCRIPT_PATH}
    node_count=$(python analyze_nodes.py "$YAML_FILE" --quick)
    P_COUNT=$(echo "$node_count" | grep -oP '\d+(?=p)')
    D_COUNT=$(echo "$node_count" | grep -oP '\d+(?=d)')

    # 1. IP
    IP_LIST=$(python3 ${CLOUD_SCRIPT_PATH}/get_node_ip_list.py --mode all)
    LOCAL_IP=${POD_IP}
    IFS=',' read -ra ip_list <<< "$IP_LIST"
    PROXY_IP=${ip_list[0]} # PROXY
    ip_list=($(printf '%s\n' "${ip_list[@]}" | sort))
    count=${#ip_list[@]}

    if [[ $((P_COUNT + D_COUNT)) -ne $count ]]; then
        echo "error: node count not match"
        exit 1
    fi

    index=0
    for ip in "${ip_list[@]}"; do
        if [ $index -lt $P_COUNT ]; then
            PREFILL_SERVER_LIST+=("$ip")
        else
            DECODE_SERVER_LIST+=("$ip")
        fi
        index=$((index+1));
    done

    if [[ " ${PREFILL_SERVER_LIST[@]} " =~ " $LOCAL_IP " ]]; then
        NODE_INDEX=$(printf '%s\n' "${PREFILL_SERVER_LIST[@]}" | grep -n "^${LOCAL_IP}$" | cut -d: -f1)
        NODE_INDEX=$((NODE_INDEX-1))
        NODE_NAME="p${NODE_INDEX}"
        NODE_ROLE="prefill"
    else
        NODE_INDEX=$(printf '%s\n' "${DECODE_SERVER_LIST[@]}" | grep -n "^${LOCAL_IP}$" | cut -d: -f1)
        NODE_INDEX=$((NODE_INDEX-1))
        NODE_NAME="d${NODE_INDEX}"
        NODE_ROLE="decode"
    fi

    # Generate NODE_IPS string, format like "p0:ip1;p1:ip2;...;d0:ip5;d1:ip6;..."
    for i in "${!PREFILL_SERVER_LIST[@]}"; do
        node_name="p${i}"
        node_ip="${PREFILL_SERVER_LIST[$i]}"
        if [ -n "$NODE_IPS" ]; then
            NODE_IPS="${NODE_IPS};"
        fi
        NODE_IPS="${NODE_IPS}${node_name}:${node_ip}"
    done
    for i in "${!DECODE_SERVER_LIST[@]}"; do
        node_name="d${i}"
        node_ip="${DECODE_SERVER_LIST[$i]}"
        if [ -n "$NODE_IPS" ]; then
            NODE_IPS="${NODE_IPS};"
        fi
        NODE_IPS="${NODE_IPS}${node_name}:${node_ip}"
    done
}

gen_run_script() {
    GLOBAL_RANK_TABLE_FILE_PATH="${RANKTABLE_SAVE_PATH}/global/global_ranktable_merge.json"
    if [ $D_COUNT -gt 1 ]; then
        RANK_TABLE_FILE_PATH_D=($(ls ${RANKTABLE_SAVE_PATH}/global/local_*merge.json))
    else
        RANK_TABLE_FILE_PATH_D=($(ls ${RANKTABLE_SAVE_PATH}/decode/local_*.json))
    fi

    if [ $P_COUNT -gt 1 ]; then
        RANK_TABLE_FILE_PATH_P=($(ls ${RANKTABLE_SAVE_PATH}/global/prefill/local_*merge.json))
    else
        RANK_TABLE_FILE_PATH_P=($(ls ${RANKTABLE_SAVE_PATH}/prefill/local_*.json))
    fi

    echo "===== print cloud info ========="
    echo "MODEL_PATH: $MODEL_PATH"
    echo "LOG_PATH: $LOG_PATH"
    echo "CLOUD_SCRIPT_PATH: $CLOUD_SCRIPT_PATH"
    echo "RANK_TABLE_FILE_PATH_D: ${RANK_TABLE_FILE_PATH_D};"
    echo "RANK_TABLE_FILE_PATH_P: ${RANK_TABLE_FILE_PATH_P};"
    echo "GLOBAL_RANK_TABLE_FILE_PATH: ${GLOBAL_RANK_TABLE_FILE_PATH};"
    echo "PROXY_PORT: $PROXY_PORT"
    echo "SERVED_MODEL_NAME: $SERVED_MODEL_NAME"
    echo "=====     end print    ========="

    cd ${TMP_SCRIPTS_PATH}
    python analyze_nodes.py "$YAML_FILE" \
      --node-ips "$NODE_IPS" \
      --model-path "$MODEL_PATH" \
      --service-name "$SERVED_MODEL_NAME" \
      --log-path "$BASE_LOG_PATH" \
      --nic-name "$NIC_NAME" \
      --global-rank-table "$GLOBAL_RANK_TABLE_FILE_PATH" \
      --rank-table-d "$RANK_TABLE_FILE_PATH_D" \
      --rank-table-p "$RANK_TABLE_FILE_PATH_P" \
      --proxy-port "$PROXY_PORT" \
      --output "$TMP_SCRIPTS_PATH/omni_cli_config.sh" \
      --run
}

handle_dir() {
    LOG_PATH="$BASE_LOG_PATH/$NODE_NAME"
    mkdir -p ${LOG_PATH}

    NFS_BASE="${CLOUD_SCRIPT_PATH:-/dev/nfs/}"
    NFS_PATH="${NFS_BASE%/}"
    crc_result=$(echo -n "$IP_LIST" | cksum | awk '{print $1}')
    MIDDLE_DIR="ranktable_dir"
    RANKTABLE_SAVE_PATH="${NFS_PATH}/${MIDDLE_DIR}/${crc_result}"

    mkdir -p ${TMP_SCRIPTS_PATH}
    \cp -f "${CLOUD_SCRIPT_PATH}/analyze_nodes.py" "${TMP_SCRIPTS_PATH}/"
    \cp -f "${CLOUD_SCRIPT_PATH}/server_profiles.yml" "${TMP_SCRIPTS_PATH}/"
}

run_server() {
    node_script="omni_start_${NODE_NAME}.sh"
    node_script_path="${TMP_SCRIPTS_PATH}/$node_script"

    # Wait for node_script to exist
    while [[ ! -f "$node_script_path" ]]; do
        echo "Waiting for $node_script_path to be created..."
        sleep 30
    done

    sed -i 's|^[[:space:]]*set -euo pipefail|# set -euo pipefail|' "$node_script_path"
    echo "start server: bash $node_script_path"
    bash "$node_script_path"

    if [[ "$LOCAL_IP" == "$PROXY_IP" ]]; then
        echo "starting c..."
        bash "${TMP_SCRIPTS_PATH}/omni_start_c.sh"
    fi
}

check_files_exist() {
    local arr=("$@")
    echo "check arr: ${arr[@]}"
    for file in "${arr[@]}"; do
        echo "start check file:"
        echo $file
        if [[ ! -f "$file" ]]; then
            echo "File $file does not exist." >&2
            return 1
        fi
    done
    return 0
}

wait_local_ranktable_ready() {
    eval "$1"
    eval "$2"
    echo "P: ${RESULT_ENTRIES_P[@]}"
    echo "D: ${RESULT_ENTRIES_D[@]}"
    while true; do
        if check_files_exist "${RESULT_ENTRIES_P[@]}"; then
            break
        fi
        echo "Waiting for prefill local_ranktable.json to be created..."
        sleep 10
    done

    while true; do
        if check_files_exist "${RESULT_ENTRIES_D[@]}"; then
            break
        fi
        echo "Waiting for decode local_ranktable.json to be created..."
        sleep 10
    done
}

global_ranktable_generate() {
    if [ -f "$RANKTABLE_SAVE_PATH/global/global_ranktable_merge.json" ]; then
        echo "global_ranktable_merge.json exit, skip generate."
        return 0
    fi

    declare -a RESULT_ENTRIES_P=()
    RANKTABLE_SUFFIX=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | awk '$1=$1' | tr -d ',')
    for ip in "${PREFILL_SERVER_LIST[@]}"; do
        entry="${RANKTABLE_SAVE_PATH}/prefill/local_ranktable_${ip}_${RANKTABLE_SUFFIX}.json"
        RESULT_ENTRIES_P+=("$entry")
    done

    declare -a RESULT_ENTRIES_D=()
    for ip in "${DECODE_SERVER_LIST[@]}"; do
        entry="${RANKTABLE_SAVE_PATH}/decode/local_ranktable_${ip}_${RANKTABLE_SUFFIX}.json"
        RESULT_ENTRIES_D+=("$entry")
    done

    PREFILL_RANKTABLE_LIST=$(IFS=,; echo "${RESULT_ENTRIES_P[*]}")
    DECODE_RANKTABLE_LIST=$(IFS=,; echo "${RESULT_ENTRIES_D[*]}")

    echo "PREFILL_RANKTABLE_LIST: $PREFILL_RANKTABLE_LIST"
    echo "DECODE_RANKTABLE_LIST: $DECODE_RANKTABLE_LIST"

    p_l=$(declare -p RESULT_ENTRIES_P)
    d_l=$(declare -p RESULT_ENTRIES_D)

    if timeout ${TIME_OUT} bash -c "$(declare -f wait_local_ranktable_ready check_files_exist); wait_local_ranktable_ready '$p_l' '$d_l'"; then
        echo "global ranktable ready."
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "error: wait_local_ranktable_ready time out"
        fi
        exit 1
    fi

    prefill_ranktable_list=${PREFILL_RANKTABLE_LIST}
    prefill_ranktable_list=$(echo "$prefill_ranktable_list" | awk '$1=$1' | tr ',' ' ')
    decode_ranktable_list=${DECODE_RANKTABLE_LIST}
    decode_ranktable_list=$(echo "$decode_ranktable_list" | awk '$1=$1' | tr ',' ' ')

    # rm -rf ${RANKTABLE_SAVE_PATH}/global
    mkdir -p ${RANKTABLE_SAVE_PATH}/global
    if [ $D_COUNT -gt 1 ]; then
        python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
            --mode merge-local \
            --local-ranktable-list ${decode_ranktable_list} \
            --save-dir ${RANKTABLE_SAVE_PATH}/global

        decode_local_ranktable_merge=$(ls ${RANKTABLE_SAVE_PATH}/global/local*merge.json | tr '\n' ' ')
    else
        decode_local_ranktable_merge="${decode_ranktable_list}"
    fi

    if [ $P_COUNT -gt 1 ]; then
        mkdir -p ${RANKTABLE_SAVE_PATH}/global/prefill
        python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
            --mode merge-local \
            --local-ranktable-list ${prefill_ranktable_list} \
            --save-dir ${RANKTABLE_SAVE_PATH}/global/prefill

        prefill_local_ranktable_merge=$(ls ${RANKTABLE_SAVE_PATH}/global/prefill/local*merge.json | tr '\n' ' ')
    else
        prefill_local_ranktable_merge="${prefill_ranktable_list}"
    fi

    api_server_files=$(ls ${RANKTABLE_SAVE_PATH}/prefill/local_ranktable*host.json | head -1)
    python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
        --mode merge-all \
        --api-server-list ${api_server_files} \
        --prefill-server-list ${prefill_local_ranktable_merge} \
        --decode-server-list ${decode_local_ranktable_merge} \
        --save-dir ${RANKTABLE_SAVE_PATH}/global
}

local_ranktable_generate() {
    if [ -f "$RANKTABLE_SAVE_PATH/global/global_ranktable_merge.json" ]; then
        echo "global_ranktable_merge.json exit, skip generate."
        return 0
    fi
    case "$NODE_ROLE" in
        prefill)
            echo "python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
                --mode gen \
                --prefill-server-list \"${ASCEND_RT_VISIBLE_DEVICES}\" \
                --api-server \
                --save-dir ${RANKTABLE_SAVE_PATH}/prefill \
                --ip ${LOCAL_IP}"
            python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
                --mode gen \
                --prefill-server-list "${ASCEND_RT_VISIBLE_DEVICES}" \
                --api-server \
                --save-dir ${RANKTABLE_SAVE_PATH}/prefill \
                --ip ${LOCAL_IP}
            ;;
        decode)
            echo "python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
                --mode gen \
                --decode-server-list \"${ASCEND_RT_VISIBLE_DEVICES}\" \
                --save-dir ${RANKTABLE_SAVE_PATH}/decode \
                --ip ${LOCAL_IP}"
            python3 ${CODE_PATH}/tools/scripts/pd_ranktable_tools.py \
                --mode gen \
                --decode-server-list "${ASCEND_RT_VISIBLE_DEVICES}" \
                --save-dir ${RANKTABLE_SAVE_PATH}/decode \
                --ip ${LOCAL_IP}
            ;;
        c) echo " C node, do nothing." ;;
    esac
}

wait_global_ranktable_ready() {
    echo "NODE_NAME: $NODE_NAME"
    while [ ! -f "$RANKTABLE_SAVE_PATH/global/global_ranktable_merge.json" ]; do
        echo "Waiting for global_ranktable_merge.json to be created..."
        sleep 8
    done
}

echo "===== main ====="
get_env
get_node_info
handle_dir
bash "${CLOUD_SCRIPT_PATH}/clear.sh"

log_file="${LOG_PATH}/cloud_run.log"
rm -rf $log_file
exec 3>&1
exec 1> >(tee -a "$log_file") 2>&1

export NODE_NAME=${NODE_NAME}
export RANKTABLE_SAVE_PATH=${RANKTABLE_SAVE_PATH}
mkdir -p ${RANKTABLE_SAVE_PATH} ${RANKTABLE_SAVE_PATH}/prefill ${RANKTABLE_SAVE_PATH}/decode


echo "===== before local_ranktable_generate ====="
local_ranktable_generate
echo "===== after local_ranktable_generate ====="

echo "===== before wait global ranktable ready ====="
if [ "$NODE_NAME" = "p0" ]; then
    echo "p0 start generate global ranktable"
    global_ranktable_generate
fi

if timeout ${TIME_OUT} bash -c "$(declare -f wait_global_ranktable_ready); wait_global_ranktable_ready"; then
    echo "global ranktable ready."
else
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "error: wait_global_ranktable_ready time out"
    fi
    exit 1
fi
echo "===== after wait global ranktable ready ====="

gen_run_script
run_server

while :
do
    sleep 1000
done