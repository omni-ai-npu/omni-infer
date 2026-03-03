#!/bin/bash
IP_ADDRESSES=$(hostname -I | tr ' ' '\n' | grep -v '^127\.0\.0\.1$' | grep -v '10.244*' | grep -v '172.17*')

export HCCL_IF_BASE_PORT=59000
sysctl -w net.ipv4.ip_local_reserved_ports=59000-59015

export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1000
export GLOO_SOCKET_IFNAME="enp23s0f3"
export TP_SOCKET_IFNAME="enp23s0f3"
export HCCL_SOCKET_IFNAME="enp23s0f3"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_LOGGING_LEVEL="INFO"
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export VLLM_WORKER_MULTIPROC_METHOD="fork"
export HCCL_CONNECT_TIMEOUT=180
export HCCL_EXEC_TIMEOUT=180
export TASK_QUEUE_ENABLE=1

model=/path/to/model # Set this to the path of the model checkpoint
ADDRESS1=XXX # Set this to the IP address of the first machine
ADDRESS2=XXX # Set this to the IP address of the second machine

rm -rf /root/.cache/vllm/torch_compile_cache/

# patch
export VLLM_PLUGINS="omni-npu,omni_pangu_models,omni_npu_patches"
export OMNI_NPU_PATCHES_DIR="pangu_sink_swa_mla"
export OMNI_NPU_VLLM_PATCHES="ALL"

log_file="./pangu505b.log"

# Ensure log directory exists
mkdir -p "$(dirname "$log_file")"
# Ensure log file exists
if [ ! -f "$log_file" ]; then
    touch "$log_file"
fi

COMPILATION_CONFIG='{"level": 3, "cudagraph_mode":"FULL", "cudagraph_capture_sizes":[64], "backend":"eager", "compile_sizes":[64]}'

BASE_CMD="vllm serve "$model" \
    --served-model-name pangu505b \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --no-enable-prefix-caching \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.86 \
    --trust-remote-code \
    --tensor-parallel-size 16 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --enable-expert-parallel \
    --data-parallel-address $ADDRESS1 \
    --data-parallel-rpc-port 13345 \
    --compilation-config '$COMPILATION_CONFIG'"


if [ "$IP_ADDRESSES" = "$ADDRESS2" ]; then
    BASE_CMD="$BASE_CMD --headless --data-parallel-start-rank 1"
fi

eval $BASE_CMD > "$log_file" 2>&1 &