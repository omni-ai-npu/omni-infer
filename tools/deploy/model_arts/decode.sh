source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/set_env.sh

USER_EXTRA_ARGS=("$@")

export LOG_PATH=${LOG_PATH:-"/home/ma-user/scripts/decode/log"}
LOG_PATH=${LOG_PATH}/${POD_IP}
rm -rf ${LOG_PATH}
mkdir -p "${LOG_PATH}"
export ASCEND_PROCESS_LOG_PATH=${LOG_PATH}/ascend
mkdir -p "${ASCEND_PROCESS_LOG_PATH}"

rm -rf ~/.cache/huggingface/modules/transformers_modules.tokenization_sophon_fast
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${MODEL_PATH}', trust_remote_code=True)" >> ${LOG_PATH}/server.log 2>&1

rm -rf /workspace/omniinfer/tools/scripts/static_kernel_compile_outputs
rm -rf /usr/local/Ascend/cann/opp/static_kernel
rm -rf graph_cache/

export API_PORT=${BASE_API_PORT:-9100}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-"AIV"}
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
IFS=',' read -ra last_instance_ip_list <<< "${ip_list[-1]}"
DECODE_POD_NUM_PER_INSTANCE=${#last_instance_ip_list[@]}
echo "env_DECODE_POD_NUM_PER_INSTANCE is ${DECODE_POD_NUM_PER_INSTANCE}" >> ${LOG_PATH}/server.log 2>&1

DECODE_INSTANCE_NUM=${DECODE_INSTANCE_NUM:-1}
cur_pod_ip=${POD_IP}
SERVER_IP_LIST=""
IFS=';' read -ra head_ip_list <<< "$all_ip_list"
instance_num=${#head_ip_list[@]}
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
            if [[ "${instance_ip_list[$j]}" == "$cur_pod_ip" ]]; then
                pod_id=$j
                if [[ $(($j % $DECODE_POD_NUM_PER_INSTANCE)) == 0 ]]; then
                    MASTER_IP=${cur_pod_ip}
                else
                    MASTER_IP=${instance_ip_list[$((j / DECODE_POD_NUM_PER_INSTANCE * DECODE_POD_NUM_PER_INSTANCE))]}
                fi
                break
            fi
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
if [[ -z "${MASTER_IP:-}" ]]; then
    echo "ERROR: MASTER_IP not found, cur_pod_ip=$cur_pod_ip not in decode IP list" >&2
    exit 1
fi

export LOCAL_DECODE_SERVER_IP_LIST=${SERVER_IP_LIST}
export GLOBAL_DECODE_SERVER_IP_LIST=${SERVER_IP_LIST}

export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
export LCCL_DETERMINISTIC=${LCCL_DETERMINISTIC:-0}
export LCCL_PARALLEL=${LCCL_PARALLEL:-0}
export MASTER_PORT=${MASTER_PORT:-8000}
export OMNI_REUSE_PREFILLED_TOKENS=${OMNI_REUSE_PREFILLED_TOKENS:-1}
export OMNI_SKIP_DECODE_TOKENIZE=${OMNI_SKIP_DECODE_TOKENIZE:-1}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export ROLE=${ROLE:-decode}
export SHLVL=${SHLVL:-1}
export SOCKET_IFNAME=${SOCKET_IFNAME:-enp23s0f3}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export TNG_HOST_COPY=${TNG_HOST_COPY:-1}
export TOKENIZER_PROC_POOL=${TOKENIZER_PROC_POOL:-0}
export TOOLCHAIN_HOME=${TOOLCHAIN_HOME:-/usr/local/Ascend/latest/toolkit}
export TP_SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export USING_LCCL_COM=${USING_LCCL_COM:-0}
export OMNI_LLMDATADIST_ZMQ_PORT=${OMNI_LLMDATADIST_ZMQ_PORT:-5668}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-fork}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
export HCCL_OP_RETRY_ENABLE=${HCCL_OP_RETRY_ENABLE:-"L0:0, L1:0, L2:0"}
export OMNI_NPU_PATCHES_DIR=${OMNI_NPU_PATCHES_DIR:-"pangu_v2_hybrid_vl"}
export OMNI_NPU_VLLM_PATCHES=${OMNI_NPU_VLLM_PATCHES:-"ALL"}
export VLLM_PLUGINS=${VLLM_PLUGINS:-"omni-npu,omni_npu_patches,omni_pangu_models,omni_custom_models"}

export HYBRID_ATTN_GROUP_SIZE=${HYBRID_ATTN_GROUP_SIZE:-16}

npu=${npu:-16}
export NUM_DIE_PER_MACH=${npu}

export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-32000}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-fork}
export DISABLE_GATHER_SELECTION=${DISABLE_GATHER_SELECTION:-1}

server_model_name="$MODEL_NAME"

export PREFILL_POD_NUM=${PREFILL_INSTANCE_NUM:-1}
export DECODE_POD_NUM=${DECODE_INSTANCE_NUM:-1}
dtype=${dtype:-bfloat16}

tp=${tp:-1}
max_model_len=${max_model_len:-64000}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export ENABLE_OVERWRITE_REQ_IDS=${ENABLE_OVERWRITE_REQ_IDS:-0}

max_num_seqs=${max_num_seqs:-4}
hccl_port_wait_time_out=${hccl_port_wait_time_out:-200}

if [ ${npu} ]; then
    temp=()
    for ((i=0; i<${npu}; i++)); do
        temp+=($i)
    done
    ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${temp[*]}")
    export DECODE_SERVER_LIST=${ASCEND_RT_VISIBLE_DEVICES}
    VLLM_DP_SIZE=$((${npu}*${DECODE_POD_NUM_PER_INSTANCE}/${tp}))
    num_servers=$((${npu} / ${tp}))
    export VLLM_DP_SIZE
    echo "env_DECODE_SERVER_LIST is ${DECODE_SERVER_LIST}, VLLM_DP_SIZE is ${VLLM_DP_SIZE}, num_servers is ${num_servers}" >> ${LOG_PATH}/server.log 2>&1
else
    export DECODE_SERVER_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    export VLLM_DP_SIZE=64
    num_servers=16
fi

export SERVER_OFFSET=$((${pod_id}*$((num_servers))))

if [ ${kv_transfer_config[@]+x} ]; then
    IFS=';' read -ra arr <<< "$P_NODE_LIST"

    new_kv_transfer_config=$(echo "$kv_transfer_config" | sed "s/\"kv_parallel_size\":[0-9]*/\"kv_parallel_size\":$((${PREFILL_POD_NUM} + 1))/")
    kv_transfer_config=$(echo "$new_kv_transfer_config" | sed "s/\"kv_rank\":[0-9]*/\"kv_rank\":${#arr[@]}/")
    echo "env_kv_transfer_config is ${kv_transfer_config}" >> ${LOG_PATH}/server.log 2>&1
else
    kv_transfer_config='{"kv_buffer_device":"npu", "kv_connector":"LLMDataDistConnector", "kv_parallel_size":1, "kv_role":"kv_consumer"}'
fi
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-2200}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-3000}

main() {
    if [ ! -e "/usr/local/Ascend/latest" ]; then
        mkdir -p /usr/local/Ascend/latest
        ln -sf /usr/local/Ascend/ascend-toolkit/latest/* /usr/local/Ascend/latest
        echo "Link created successfully."
    else
        echo "Link already exists or target missing"
    fi

    if [ ! -d ${LOG_PATH} ]; then
        mkdir -p ${LOG_PATH}
    fi

    if [[ "${proc_bind_enabled}" == "true" ]]; then
        echo "proc_bind_enabled=true, binding CPU cores..." >> ${LOG_PATH}/server.log 2>&1
        chmod +x /workspace/omniinfer/tools/scripts/bind_cpu.sh 2>/dev/null
        /workspace/omniinfer/tools/scripts/bind_cpu.sh >> ${LOG_PATH}/server.log 2>&1
        echo "CPU bind completed." >> ${LOG_PATH}/server.log 2>&1
    fi

    echo "start to check hccl_port" >> ${LOG_PATH}/server.log 2>&1
    count=0
    for hccl_port in $(seq 32000 32015); do
        while netstat -naltp  | grep :$hccl_port; do
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


    for rank in $(seq 0 $((num_servers - 1))); do
        export VLLM_DP_RANK=$((rank + ${SERVER_OFFSET}))
        export VLLM_DP_MASTER_PORT=${MASTER_PORT}
        export VLLM_DP_MASTER_IP=${MASTER_IP}
        echo "VLLM_DP_RANK is ${VLLM_DP_RANK}" >> ${LOG_PATH}/server.log 2>&1
        able_port=$((${API_PORT}+rank))
        start=$((rank * ${tp}))
        end=$(((rank + 1) * ${tp}))
        devices=$(seq -s, $start $((end - 1)))
        export ASCEND_RT_VISIBLE_DEVICES="$devices"
        cmd=(
            vllm serve "${MODEL_PATH}" \
            --host "${cur_pod_ip}" \
            --trust-remote-code \
            --gpu-memory-utilization ${gpu_util:-0.92} \
            --tensor-parallel-size ${tp} \
            --data-parallel-size ${dp:-${VLLM_DP_SIZE}} \
            --data-parallel-size-local ${local_dp:-1} \
            --data-parallel-address ${VLLM_DP_MASTER_IP} \
            --data-parallel-rpc-port ${MASTER_PORT} \
            --data-parallel-rank ${VLLM_DP_RANK} \
            --port ${able_port} \
            --dtype ${dtype:-"bfloat16"} \
            --served-model-name ${server_model_name} \
            --max-model-len ${max_model_len} \
            --enable-expert-parallel \
            --max-num-seqs ${max_num_seqs} \
            --max-num-batched-tokens ${max_num_batched_tokens:-2048} \
            --no-disable-hybrid-kv-cache-manager
        )

    if [[ "${ENABLE_OMNI_CACHE:-1}" == "1" ]]; then
        echo "env_ENABLE_OMNI_CACHE is ${ENABLE_OMNI_CACHE}" >> ${LOG_PATH}/server.log 2>&1
        export ENABLE_OMNI_CACHE=1
        export ENABLE_HOST_MAPPING="${ENABLE_HOST_MAPPING:-0}"
        export OMNI_CACHE_MMAP_FILE="${OMNI_CACHE_DECODE_MMAP_FILE:-omni_cache}"
        export OMNI_CACHE_MMAP_PATH="/dev/hugepages/${OMNI_CACHE_MMAP_FILE}"
        export OMNI_CACHE_LAYER_BYTES="${OMNI_CACHE_LAYER_BYTES:-27917287424}" # 48GB
        export MAP_SIZE_BYTES="${MAP_SIZE_BYTES:-549755813888}" # 1000GB
        export NUM_DIE_PER_MACH="${NUM_DIE_PER_MACH:-16}"
        export BASE_PORT="${BASE_PORT:-16077}"
        export ZMQ_BASE_PORT="${ZMQ_BASE_PORT:-16555}"
        export OMNI_CACHE_MLA_SWA_DEBUG="${OMNI_CACHE_MLA_SWA_DEBUG:-1}"
        export ENABLE_OMNI_CACHE_DSA_SPLIT="${ENABLE_OMNI_CACHE_DSA_SPLIT:-0}"
        export OMNI_CACHE_DSA_MMAP_FILE="${OMNI_CACHE_DSA_MMAP_FILE:-omni_cache_decode_dsa}"
        export OMNI_CACHE_DSA_MMAP_PATH="/dev/hugepages/${OMNI_CACHE_DSA_MMAP_FILE}"
        export ROLE=decode
        export DISABLE_GATHER_SELECTION=1
        export OMNI_CACHE_LOCAL_DP_SIZE=16

        export USE_OMNI_INPUT_BATCH="${USE_OMNI_INPUT_BATCH:-0}"

        if [[ "${ENABLE_OMNI_CACHE_DSA_SPLIT}" == "1" ]]; then
            OMNI_CACHE_DSA_MAP_SIZE_BYTES="${OMNI_CACHE_DSA_MAP_SIZE_BYTES:-$((MAP_SIZE_BYTES * 80 / 100))}"
            PRIMARY_PAGES=$(( (MAP_SIZE_BYTES + (2 * 1024 * 1024) - 1) / (2 * 1024 * 1024) ))
            DSA_PAGES=$(( (OMNI_CACHE_DSA_MAP_SIZE_BYTES + (2 * 1024 * 1024) - 1) / (2 * 1024 * 1024) ))
            DSA_TOTAL_PAGES=$(( PRIMARY_PAGES + DSA_PAGES ))
            MAP_SIZE_BYTES="${OMNI_CACHE_DSA_MAP_SIZE_BYTES}" OMNI_FILE="${OMNI_CACHE_DSA_MMAP_FILE}" \
            bash "${SETUP_HUGETLBFS_SH}" "${DSA_TOTAL_PAGES}"
        fi

        python -c "from omni_cache.connector import register_connectors; register_connectors()"

        KV_PARALLEL_SIZE=$((VLLM_DP_SIZE + 1))
        new_kv_transfer_config=$(echo "$kv_transfer_config" | sed "s/\"kv_parallel_size\":[0-9]*/\"kv_parallel_size\":$((KV_PARALLEL_SIZE))/")
        kv_transfer_config="${new_kv_transfer_config}"
        cmd+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
    else
        export ENABLE_OMNI_CACHE=0
        export ENABLE_HOST_MAPPING=0
    fi
    cmd+=(--kv-transfer-config "${kv_transfer_config}")
    cmd+=("${USER_EXTRA_ARGS[@]}")
    echo "${cmd[@]}" >> ${LOG_PATH}/server_${rank}.log 2>&1 &
    "${cmd[@]}" >> ${LOG_PATH}/server_${rank}.log 2>&1 &
    done
    wait
}

main
