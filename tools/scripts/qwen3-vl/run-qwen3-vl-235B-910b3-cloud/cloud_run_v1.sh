#!/bin/bash
# set -euo pipefail

# pre set
TP=4
MODEL_LEN_MAX=168960 # 165K
NUM_SPECULATIVE_TOKENS=0
SERVED_MODEL_NAME="qwen"
GPU_UTIL=0.9
GLOO_SOCKET_IFNAME=eth0
TP_SOCKET_IFNAME=eth0
VLLM_PLUGINS="omni-npu,omni_custom_models,omni_npu_patches"
OMNI_NPU_VLLM_PATCHES=ALL
ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPU_PER_NODE=8

BASE_LOG_PATH="/home/ma-user/work/*"
SCRIPTS_PATH="/home/ma-user/work/*"
MODEL_PATH="/home/ma-user/work/*"
LOCAL_PATH="/home/ma-user/work/"


BASE_API_PORT=9000
PROXY_PORT=7000
PROXY_TYPE="aggregation"


# replace bucket name if BUCKET_PATH env variable exists
replace_bucket_name() {
    if [ -n "$BUCKET_PATH" ]; then
        BASE_LOG_PATH=$(echo "$BASE_LOG_PATH" | sed 's|\*|'"$BUCKET_PATH"'|g')
        SCRIPTS_PATH=$(echo "$SCRIPTS_PATH" | sed 's|\*|'"$BUCKET_PATH"'|g')
        MODEL_PATH=$(echo "$MODEL_PATH" | sed 's|\*|'"$BUCKET_PATH"'|g')
    fi
}


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

replace_log_dir_name() {
    if [ -n "$LOG_SELF_PATH" ]; then 
        BASE_LOG_PATH="${BASE_LOG_PATH/log_mix/$LOG_SELF_PATH}"
    fi
    
    # 使用固定的时间戳
    BASE_LOG_PATH="${BASE_LOG_PATH}${TIMESTAMP}/"
}

# local var
IP_LIST=""
LOCAL_IP=""
LOG_PATH=""
SERVER_OFFSET=0
NODE_INDEX=-1
NODE_NAME="default_name"
NUM_SERVERS=1
DP=1
MASTER_IP=""
MASTER_PORT=8100
NODE_NUM=1
PROXY_API_LIST=""

# 1 get role & ip list
get_node_info() {
    # 1. IP
    IP_LIST=$(python3 ${SCRIPTS_PATH}/get_node_ip_list.py --mode all)
	LOCAL_IP=${POD_IP}
    IFS=',' read -ra ip_list <<< "$IP_LIST"
    NODE_NUM=${#ip_list[@]}
    index=0
    for ip in "${ip_list[@]}"; do
        if [ "$ip" = "$LOCAL_IP" ]; then
            SERVER_OFFSET=$((index * NPU_PER_NODE))
            NODE_INDEX=$index
            NODE_NAME="node_${index}"
            break
        fi
        index=$((index + 1))
    done
    MASTER_IP=${ip_list[0]}

    # 2. num servers
    NUM_SERVERS=$((NPU_PER_NODE / TP))
    DP=$((NODE_NUM * NUM_SERVERS))

    # 3. proxy api list
    api_list=()
    for ip in "${ip_list[@]}"; do
        for ((i=0; i<NUM_SERVERS; i++)); do
            api_list+=("${ip}:$((BASE_API_PORT + i))")
        done
    done
    IFS=","
    PROXY_API_LIST="${api_list[*]}"
}

# 2 log path
make_log_path() {
    LOG_PATH="$BASE_LOG_PATH/$NODE_NAME"
    mkdir -p ${LOG_PATH}
}

# 3. print var
print_all_variables() {
    echo "===== start print var ======"
    echo "BASE_LOG_PATH: $BASE_LOG_PATH"
    echo "ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
    echo "NPU_PER_NODE: $NPU_PER_NODE"
    echo "LOG_PATH: $LOG_PATH"
    echo "SCRIPTS_PATH: $SCRIPTS_PATH"
    echo "MODEL_PATH: $MODEL_PATH"
    echo "NUM_SERVERS: $NUM_SERVERS"
    echo "DP: $DP"
    echo "TP: $TP"
    echo "MODEL_LEN_MAX: $MODEL_LEN_MAX"
    echo "NUM_SPECULATIVE_TOKENS: $NUM_SPECULATIVE_TOKENS"
    echo "SERVED_MODEL_NAME: $SERVED_MODEL_NAME"
    echo "GPU_UTIL: $GPU_UTIL"
    echo "MASTER_IP: $MASTER_IP"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "BASE_API_PORT: $BASE_API_PORT"
    echo "PROXY_PORT: $PROXY_PORT"
    echo "PROXY_TYPE: $PROXY_TYPE"
    echo "PROXY_API_LIST: $PROXY_API_LIST"
    echo "IP_LIST: $IP_LIST"
    echo "LOCAL_IP: $LOCAL_IP"
    echo "SERVER_OFFSET: $SERVER_OFFSET"
    echo "NODE_INDEX: $NODE_INDEX"
    echo "NODE_NAME: $NODE_NAME"
    echo "===================================="
}

# export
export_env() {
    export VLLM_NO_KERNEL_CACHE=1
    export DEVICE_TYPE=npu
    export ASCEND_GLOBAL_LOG_LEVEL=3
    # export VLLM_LOGGING_LEVEL=DEBUG
    export HCCL_OP_RETRY_ENABLE="L0:0,L1:0,L2:0"
    export TASK_QUEUE_ENABLE=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE="256"
    export HCCL_OP_EXPANSION_MODE="AIV"
    export SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER=16384

    export GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
    export TP_SOCKET_IFNAME=$TP_SOCKET_IFNAME
    export VLLM_PLUGINS=$VLLM_PLUGINS
    export OMNI_NPU_VLLM_PATCHES=$OMNI_NPU_VLLM_PATCHES
    export HCCL_IF_IP=${LOCAL_IP}
    export ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES

    export PROXY_PORT=$PROXY_PORT
    export PROXY_TYPE=$PROXY_TYPE
    export PROXY_API_LIST=$PROXY_API_LIST
    export MODEL_PATH=$MODEL_PATH
    export DP=$DP
    export LOG_PATH=$LOG_PATH
    export SCRIPTS_PATH=$SCRIPTS_PATH
    export PYTHONHASHSEED=123

    export XGRAMMAR_DISABLE_TORCH_COMPILE=1
    export TORCH_COMPILE_DISABLE=1
    export VLLM_USE_TRITON_FLASH_ATTN=0
}

# 5. run server
run_v1(){

    EXTRA_ARGS='--enable-expert-parallel --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 16384 --max-num-seqs 512 --distributed-executor-backend mp --swap_space 64.0 --disable-log-requests --enable-prompt-tokens-details --compilation-config {"level": 3, "cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4,8,16,32,48,72,96,128,256,384,448,512], "backend":"eager", "compile_sizes":[4,8,16,32,48,72,96,128,256,384,448,512]} --limit-mm-per-prompt {"image":2048} --media-io-kwargs {"video":{"fps":2,"num_frames":-1}} --allowed-local-media-path '"${LOCAL_PATH}"

    local mtp_args=""
    if [ "$NUM_SPECULATIVE_TOKENS" -ne 0 ]; then
        mtp_args="--enable-mtp"
    fi
    python ${SCRIPTS_PATH}/mix_start_servers.py \
    --num-servers "${NUM_SERVERS}" \
    --max-port-attempts 1 \
    --num-dp "${DP}" \
    --server-offset "${SERVER_OFFSET}" \
    --model-path "${MODEL_PATH}" \
    --master-ip "${MASTER_IP}" \
    --master-port "${MASTER_PORT}" \
    --max-model-len "${MODEL_LEN_MAX}" \
    --base-api-port "${BASE_API_PORT}" \
    --tp "${TP}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --gpu-util "${GPU_UTIL}" \
    $mtp_args \
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" \
    --log-dir "$LOG_PATH" \
    --extra-args "${EXTRA_ARGS}" >> $log_file &
}

# 6. run proxy
run_c0(){
    if [ "$NODE_INDEX" -eq 0 ]; then
        bash ${SCRIPTS_PATH}/run_proxy.sh
    fi
}

# 7. tmp nginx
run_nginx(){
    python ${SCRIPTS_PATH}/vllm_proxy.py "${PROXY_API_LIST}"
    echo "python ${SCRIPTS_PATH}/vllm_proxy.py $PROXY_SERVERS"
    sudo mkdir -p /etc/nginx/conf.d/
    sleep 1
    sudo pkill -9 nginx 2>/dev/null
    sleep 1
    sudo \cp ${SCRIPTS_PATH}/nginx.conf /usr/local/nginx/conf/
    sleep 1
    sudo \cp ${SCRIPTS_PATH}/vllm_proxy.conf /etc/nginx/conf.d/
    sleep 1
    sudo /usr/sbin/nginx
    echo "nginx started!"
}


source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh

replace_bucket_name
replace_log_dir_name
get_node_info
make_log_path

log_file="${LOG_PATH}/omni_run.log"
rm -rf $log_file
exec 3>&1
exec 1> >(tee -a "$log_file") 2>&1

print_all_variables
export_env

run_v1

run_c0
# if [ "$NODE_INDEX" -eq 0 ]; then
#     run_nginx
# fi

while :
do
    sleep 1000
done