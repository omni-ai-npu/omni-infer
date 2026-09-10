source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh

USER_EXTRA_ARGS=("$@")

rm -rf ~/.cache/huggingface/modules/transformers_modules.tokenization_sophon_fast

cp -f /mnt/bucket-910c-6055/zhangqiang/npu_shared_offload_region.py /workspace/omniinfer/omni/v1/kv_offload/cpu

export API_PORT=${BASE_API_PORT:-9000}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_AICPU_PATH=${ASCEND_AICPU_PATH:-/usr/local/Ascend/latest}
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/latest}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
export ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-/usr/local/Ascend/latest/opp}
export ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/latest}
export PYTHONHASHSEED=${PYTHONHASHSEED:-1234}
export AUTO_USE_UC_MEMORY=${AUTO_USE_UC_MEMORY:-1}
export CODE_PATH=${CODE_PATH:-/workspace/omniinfer}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-2}
export LOG_PATH=${LOG_PATH:-"/home/ma-user/scripts/prefill/log"}
LOG_PATH=${LOG_PATH}/${POD_IP}
rm -rf ${LOG_PATH}
mkdir -p "${LOG_PATH}"
export ASCEND_PROCESS_LOG_PATH=${LOG_PATH}/ascend
rm -rf ${ASCEND_PROCESS_LOG_PATH}
mkdir -p "${ASCEND_PROCESS_LOG_PATH}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! all_ip_list=$(python3 "$SCRIPTS_DIR/get_node_ip_list.py"); then
    echo "ERROR: failed to get node IP list" >&2
    exit 1
fi
if [[ -z "$all_ip_list" ]]; then
    echo "ERROR: empty node IP list" >&2
    exit 1
fi

IFS=';' read -ra ip_list <<< "$all_ip_list"
IFS=',' read -ra first_instance_ip_list <<< "${ip_list[0]}"
PREFILL_POD_NUM_PER_INSTANCE=${#first_instance_ip_list[@]}
echo "env_PREFILL_POD_NUM_PER_INSTANCE is ${PREFILL_POD_NUM_PER_INSTANCE}" >> ${LOG_PATH}/server.log 2>&1

IFS=',;' read -ra arr <<< "$all_ip_list"
cur_pod_ip=${POD_IP}
echo "cur_pod_ip is ${cur_pod_ip}" >> ${LOG_PATH}/server.log 2>&1
for index in "${!arr[@]}"; do
    if [[ "${arr[$index]}" == "$cur_pod_ip" ]]; then
        if [[ $(($index % $PREFILL_POD_NUM_PER_INSTANCE)) == 0 ]]; then
            HOST_IP=${cur_pod_ip}
        else
            HOST_IP=${arr[$((index / PREFILL_POD_NUM_PER_INSTANCE * PREFILL_POD_NUM_PER_INSTANCE))]}
        fi
        break
    fi
done

DECODE_INSTANCE_NUM=${DECODE_INSTANCE_NUM:-1}
IFS=';' read -ra head_ip_list <<< "$all_ip_list"
instance_num=${#head_ip_list[@]}
SERVER_IP_LIST=""
P_NODE_LIST=""
for index in "${!head_ip_list[@]}"; do
    instance_ip_str="${head_ip_list[$index]}"
    if [[ -z "$instance_ip_str" ]]; then
        echo "ERROR: empty IP group in node IP list: $all_ip_list" >&2
        exit 1
    fi
    IFS=',' read -ra instance_ip_list <<< "$instance_ip_str"
    if [[ $index -ge $((instance_num - DECODE_INSTANCE_NUM)) ]]; then
        for j in "${!instance_ip_list[@]}"; do
            SERVER_IP_LIST+="${instance_ip_list[$j]},"
        done
    else
        for j in "${!instance_ip_list[@]}"; do
            P_NODE_LIST+="${instance_ip_list[$j]},"
        done
        P_NODE_LIST="${P_NODE_LIST%,};"
    fi
done
if [[ "$SERVER_IP_LIST" == *, ]]; then
    SERVER_IP_LIST="${SERVER_IP_LIST%,}"
fi

export GLOBAL_DECODE_SERVER_IP_LIST=${SERVER_IP_LIST}

export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}

export LCCL_DETERMINISTIC=${LCCL_DETERMINISTIC:-0}
export LCCL_PARALLEL=${LCCL_PARALLEL:-0}
export LOCAL_DECODE_SERVER_IP_LIST=${LOCAL_DECODE_SERVER_IP_LIST:-}
export MASTER_PORT=${MASTER_PORT:-8000}

export OMNI_USE_DSV3=${OMNI_USE_DSV3:-1}
export OMNI_REUSE_PREFILLED_TOKENS=${OMNI_REUSE_PREFILLED_TOKENS:-1}
export OMNI_SKIP_DECODE_TOKENIZE=${OMNI_SKIP_DECODE_TOKENIZE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

export ROLE=${ROLE:-prefill}
export SERVER_OFFSET=${SERVER_OFFSET:-0}
export SHLVL=${SHLVL:-1}

export SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export TNG_HOST_COPY=${TNG_HOST_COPY:-1}
export TOKENIZER_PROC_POOL=${TOKENIZER_PROC_POOL:-0}
export TOOLCHAIN_HOME=${TOOLCHAIN_HOME:-/usr/local/Ascend/latest/toolkit}
export TP_SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export USING_LCCL_COM=${USING_LCCL_COM:-0}
export VLLM_ENABLE_MC2=${VLLM_ENABLE_MC2:-1}
export VLLM_LLMDATADIST_ZMQ_PORT=${VLLM_LLMDATADIST_ZMQ_PORT:-5568}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-fork}
export HYBRID_ATTN_GROUP_SIZE=${HYBRID_ATTN_GROUP_SIZE:-16}

export VLLM_DP_RANK=${VLLM_DP_RANK:-0}
export VLLM_DP_MASTER_IP=${HOST_IP}
export VLLM_DP_MASTER_PORT=${MASTER_PORT}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
export HCCL_OP_RETRY_ENABLE=${HCCL_OP_RETRY_ENABLE:-"L0:0, L1:0, L2:0"}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-32000}
export OMNI_NPU_VLLM_PATCHES=${OMNI_NPU_VLLM_PATCHES:-"ALL"}
export VLLM_PLUGINS=${VLLM_PLUGINS:-"omni-npu,omni_npu_patches,omni_pangu_models,omni_custom_models"}
export VLLM_ALL2ALL_BACKEND=${VLLM_ALL2ALL_BACKEND:-"naive"}
npu=${npu:-16}
export NUM_DIE_PER_MACH=${npu}
export PREFILL_POD_NUM=${PREFILL_INSTANCE_NUM:-1}
export DECODE_POD_NUM=${DECODE_INSTANCE_NUM:-1}
long_prefill_token_threshold=${long_prefill_token_threshold:-1024}
if [[ -n "${KV_EVENTS_CONFIG}" ]]; then
    export KV_EVENTS_CONFIG
fi

export VLLM_DP_SIZE=${dp:-1}
tp=${tp:-$((npu * PREFILL_POD_NUM_PER_INSTANCE))}
hccl_port_wait_time_out=${hccl_port_wait_time_out:-200}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

if [[ -n "${npu}" ]]; then
    temp=()
    for ((i=0; i<${npu}; i++)); do
        temp+=($i)
    done
    export ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${temp[*]}")
    export PREFILL_SERVER_LIST=${ASCEND_RT_VISIBLE_DEVICES}
else
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export PREFILL_SERVER_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
fi

if [[ $kv_offload == "true" ]]; then
    kv_transfer_config='{
      "kv_connector":"MultiConnector",
      "kv_role":"kv_producer",
      "kv_connector_extra_config":{
        "connectors": [
          {
            "kv_connector": "__KV_CONNECTOR__",
            "kv_role": "kv_producer",
            "kv_rank": __KV_RANK__,
            "kv_parallel_size": __KV_PARALLEL_SIZE__,
            "kv_buffer_device": "npu",
            "kv_port": 5568
          },
          {
            "kv_connector": "NPUOffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
              "cpu_bytes_to_use": __CPU_BYTES_TO_USE__,
              "eviction_policy": "arc"
            }
          }
        ]
      }
    }'
    kv_transfer_config="${kv_transfer_config//__CPU_BYTES_TO_USE__/${cpu_bytes_to_use:-0}}"
    if [[ $hugepage_enabled == "true" ]]; then
        SETUP_HUGETLBFS="/workspace/omniinfer/tools/deploy/start_server/setup_hugetlbfs.sh"
        KV_OFFLOAD_SETUP_LOG="${LOG_PATH}/kv_offload_setup.log"
        if ! MAP_SIZE_BYTES="${cpu_bytes_to_use}" \
            bash "${SETUP_HUGETLBFS}" > "${KV_OFFLOAD_SETUP_LOG}" 2>&1; then
            echo "[ERROR] KV offload HugePage setup failed; see ${KV_OFFLOAD_SETUP_LOG}" >&2
            exit 1
        fi
    else
        rm -f -- /dev/shm/vllm_offload_*.mmap
    fi
else
    kv_transfer_config='{"kv_buffer_device":"npu","kv_connector":"__KV_CONNECTOR__","kv_parallel_size":__KV_PARALLEL_SIZE__,"kv_role":"kv_producer","kv_rank":__KV_RANK__,"kv_port":5568}'
fi

export ENABLE_OVERWRITE_REQ_IDS=${ENABLE_OVERWRITE_REQ_IDS:-0}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-100}

echo "MODEL_EXTRA_CFG_PATH is ${MODEL_EXTRA_CFG_PATH}, reasoning_parser is '${reasoning_parser}', gpu_util is ${gpu_util}, max_model_len is ${max_model_len}, max_batch_tokens is ${max_num_batched_tokens}, speculative_config is '${speculative_config}', additional_config is '${additional_config}'" >> ${LOG_PATH}/server.log 2>&1

if [ ! -e "/usr/local/Ascend/latest" ]; then
    mkdir -p /usr/local/Ascend/latest
    ln -sf /usr/local/Ascend/ascend-toolkit/latest/* /usr/local/Ascend/latest
    echo "Link created successfully."
else
    echo "Link already exists or target missing"
fi

if [[ "${proc_bind_enabled}" == "true" ]]; then
    echo "proc_bind_enabled=true, binding CPU cores..." >> ${LOG_PATH}/server.log 2>&1
    chmod +x /workspace/omniinfer/tools/scripts/bind_cpu.sh 2>/dev/null
    /workspace/omniinfer/tools/scripts/bind_cpu.sh >> ${LOG_PATH}/server.log 2>&1
    echo "CPU bind completed." >> ${LOG_PATH}/server.log 2>&1
fi

IFS=',;' read -ra arr <<< "$P_NODE_LIST"
prefill_instance_rank=-1
node_rank=0
for index in "${!arr[@]}"; do
    if [[ $(($index % $PREFILL_POD_NUM_PER_INSTANCE)) == 0 ]]; then
        prefill_instance_rank=$((${prefill_instance_rank} + 1))
        node_rank=0
    else
        ((node_rank++))
    fi
    if [[ "${arr[$index]}" == "$cur_pod_ip" ]]; then
        echo "cur pod index is ${index}" >> ${LOG_PATH}/server.log 2>&1
        kv_transfer_config="${kv_transfer_config//__KV_RANK__/$((prefill_instance_rank))}"
        break
    fi
done

echo "start to check hccl_port" >> ${LOG_PATH}/server.log 2>&1
count=0
for hccl_port in $(seq 32000 32015); do
    while netstat -naltp | grep :$hccl_port; do
        echo "hccl_port ${hccl_port} has been used." >> ${LOG_PATH}/server.log 2>&1
        netstat -naltp  | grep :$hccl_port >> ${LOG_PATH}/port.log 2>&1
        count=$((count + 1))
        if [ $count -ge ${hccl_port_wait_time_out} ]; then
            echo "Timeout: Waiting for hccl_port ${hccl_port} to be released timed out." >> ${LOG_PATH}/server.log 2>&1
            exit 1
        fi
        sleep 3
    done
    echo "current checked hccl_port is ${hccl_port}" >> ${LOG_PATH}/server.log 2>&1
done

cmd=(
    vllm serve "${MODEL_PATH}" \
    --host "${cur_pod_ip}" \
    --trust-remote-code \
    --gpu-memory-utilization ${gpu_util:-0.95} \
    --tensor-parallel-size ${tp} \
    --data-parallel-size ${VLLM_DP_SIZE} \
    --data-parallel-size-local 1 \
    --data-parallel-address ${HOST_IP} \
    --data-parallel-rpc-port ${MASTER_PORT} \
    --port ${API_PORT} \
    --served-model-name "${MODEL_NAME:-pangu_ultra_moe}" \
    --enable-expert-parallel
)
if [[ "${ENABLE_OMNI_CACHE:-0}" == "1" ]]; then
    export ENABLE_OMNI_CACHE=1
    export ENABLE_HOST_MAPPING=0
    export DISABLE_GATHER_SELECTION=1
    export OMNI_CACHE_MMAP_FILE="${OMNI_CACHE_PREFILL_MMAP_FILE:-omni_cache_p}"
    export OMNI_CACHE_MMAP_PATH="/dev/shm/${OMNI_CACHE_MMAP_FILE}"
    export OMNI_CACHE_PACKED_HBM=1
    export OMNI_CACHE_LAYER_BYTES="${OMNI_CACHE_LAYER_BYTES:-88046829568}" # 82GB
    export MAP_SIZE_BYTES="${MAP_SIZE_BYTES:-1610612736000}" # 1500GB
    export NUM_DIE_PER_MACH="${NUM_DIE_PER_MACH:-16}"
    export BASE_PORT="${BASE_PORT:-16077}"
    export ZMQ_BASE_PORT="${ZMQ_BASE_PORT:-16555}"
    export OMNI_CACHE_MLA_SWA_DEBUG="${OMNI_CACHE_MLA_SWA_DEBUG:-1}"
    export NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-${OMNI_CACHE_PREFILL_NUM_GPU_BLOCKS_OVERRIDE:-480000}}"
    export P_NODE_LIST="${P_NODE_LIST:-${HOST_IP}}"
    export ROLE=prefill
    export VLLM_PLUGINS="omni-npu,omni_custom_models,omni_pangu_models,omni_npu_patches"

    export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:=$(( OMNI_CACHE_LAYER_BYTES * HYBRID_ATTN_GROUP_SIZE ))}"
    EXTRA_ARGS="${EXTRA_ARGS} --kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES}"

    SETUP_HUGETLBFS_SH="/workspace/omniinfer/components/omni-cache/tools/setup/setup_hugetlbfs_2MB.sh"
    MAP_SIZE_BYTES="${MAP_SIZE_BYTES}" OMNI_FILE="${OMNI_CACHE_MMAP_FILE}" bash "${SETUP_HUGETLBFS_SH}"

    python -c "from omni_cache.connector import register_connectors; register_connectors()"

    KV_CONNECTOR="OmniCacheConnector"
    KV_PARALLEL_SIZE=1
    EXTRA_ARGS="${EXTRA_ARGS} --num-gpu-blocks-override ${NUM_GPU_BLOCKS_OVERRIDE}"
else
    export ENABLE_OMNI_CACHE=0
    export ENABLE_HOST_MAPPING=0
    KV_CONNECTOR="${KV_CONNECTOR:-LLMDataDistConnector}"
    KV_PARALLEL_SIZE=$((PREFILL_POD_NUM + 1))
fi
kv_transfer_config="${kv_transfer_config//__KV_CONNECTOR__/$KV_CONNECTOR}"
kv_transfer_config="${kv_transfer_config//__KV_PARALLEL_SIZE__/$KV_PARALLEL_SIZE}"
cmd+=(--kv-transfer-config "${kv_transfer_config}")

if [[ "${ENABLE_KV_EVENT}" == "true" ]]; then
    base_api_port=${API_PORT}
    port_offset=0
    kv_rank=${prefill_instance_rank}
    api_port=$(( base_api_port + port_offset + $((kv_rank * 10)) + node_rank ))
    export ENDPOINT_PORT=$((api_port + 100))
    echo "ENDPOINT_PORT is ${ENDPOINT_PORT}" >> ${LOG_PATH}/server.log 2>&1
    KV_EVENTS_CONFIG='{"enable_kv_cache_events":true,"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'${ENDPOINT_PORT}'"}'
    cmd+=(--kv-events-config "${KV_EVENTS_CONFIG}")
fi
cmd+=("${USER_EXTRA_ARGS[@]}")
wait_ray_head() {
    local ray_head_ip="$HOST_IP"
    local ray_head_port="${2:-6379}"
    local timeout="${time_out:-300}"
    local interval=5
    local elapsed=0

    echo "等待Ray主节点就绪: $ray_head_ip:$ray_head_port（超时时间: $timeout 秒）" >> ${LOG_PATH}/ray.log 2>&1
    while true; do
        if ray status --address="$ray_head_ip:$ray_head_port" &> /dev/null; then
            echo "Ray主节点已就绪: $ray_head_ip:$ray_head_port" >> ${LOG_PATH}/ray.log 2>&1
            return 0
        fi
        if [ $elapsed -ge $timeout ]; then
            echo "错误：等待Ray主节点超时（$timeout 秒）: $ray_head_ip:$ray_head_port" >> ${LOG_PATH}/ray.log 2>&1
            exit 1
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done
}

if [ ${PREFILL_POD_NUM_PER_INSTANCE} -gt 1 ]; then
    echo "PREFILL_POD_NUM_PER_INSTANCE is ${PREFILL_POD_NUM_PER_INSTANCE}" >> ${LOG_PATH}/server.log 2>&1
    IFS=',;' read -ra arr <<< "$P_NODE_LIST"
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    for index in "${!arr[@]}"; do
        if [[ "${arr[$index]}" == "$cur_pod_ip" ]]; then
            echo "cur pod index is ${index}" >> ${LOG_PATH}/server.log 2>&1
            if [[ $(($index % $PREFILL_POD_NUM_PER_INSTANCE)) == 0 ]]; then
                echo "cur pod is master" >> ${LOG_PATH}/server.log 2>&1
                ray stop --force &> /dev/null
                ray start --head --port=${ray_master_port:-6379} --num-gpus=${npu} >> ${LOG_PATH}/ray.log 2>&1
                echo "${cmd[@]}" >> ${LOG_PATH}/server.log 2>&1
                wait_ray_head "$cur_pod_ip" ${ray_master_port:-6379}
                cmd+=(--distributed-executor-backend ray)
                "${cmd[@]}" >> ${LOG_PATH}/server.log 2>&1 &
            else
                echo "cur pod is slave" >> ${LOG_PATH}/server.log 2>&1
                master_ip=${arr[$((index / PREFILL_POD_NUM_PER_INSTANCE * PREFILL_POD_NUM_PER_INSTANCE))]}
                ray stop --force &> /dev/null
                wait_ray_head "$master_ip" ${ray_master_port:-6379}
                ray start --address="${master_ip}:${ray_master_port:-6379}" --num-gpus=${npu} >> ${LOG_PATH}/ray.log 2>&1
            fi
            break
        fi
    done

else
    echo "${cmd[@]}" >> ${LOG_PATH}/server.log 2>&1
    "${cmd[@]}" >> ${LOG_PATH}/server.log 2>&1 &
fi

