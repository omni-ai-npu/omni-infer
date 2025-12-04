#!/bin/bash

set -ex

# user setting
# define four machines and master IP (can be specified by environment variables, default values are given)
# master IP is the IP of the MACHINE1
MACHINE1_HOSTNAME="${MACHINE1_HOSTNAME:-node1}"
MACHINE2_HOSTNAME="${MACHINE2_HOSTNAME:-node2}"
MACHINE3_HOSTNAME="${MACHINE3_HOSTNAME:-node3}"
MACHINE4_HOSTNAME="${MACHINE4_HOSTNAME:-node4}"
MASTER_IP="${MACHINE1_IP}"

HCCL_IF_IP="${LOCAL_IP}"
export HCCL_IF_IP

export GLOO_SOCKET_IFNAME="bond1"
export TP_SOCKET_IFNAME="bond1"

MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-R1-Quant-OmniInfer}"
########################################################

# node args (automatically determine data parallel start rank etc. based on hostname)
case "$(hostname)" in
  "$MACHINE1_HOSTNAME")
    VLLM_ARGS=(
    )
    ;;
  "$MACHINE2_HOSTNAME")
    sleep 10
    VLLM_ARGS=(
      --data-parallel-start-rank 8
      --headless
    )
    ;;
  "$MACHINE3_HOSTNAME")
    sleep 10
    VLLM_ARGS=(
      --data-parallel-start-rank 16
      --headless
    )
    ;;
  "$MACHINE4_HOSTNAME")
    sleep 10
    VLLM_ARGS=(
      --data-parallel-start-rank 24
      --headless
    )
    ;;
  *)
    echo "hostname '$(hostname)' is not in the predefined node list, please check the MACHINE*_HOSTNAME environment variable settings!"
    exit 1
    ;;
esac

export HCCL_BUFFSIZE=200
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=120
export ASCEND_GLOBAL_LOG_LEVEL=3

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export OMNI_USE_DSV3=1
export USING_LCCL_COM=0
export VLLM_ENABLE_MC2=0

MODEL_EXTRA_CFG_PATH="$(realpath ../../tests/test_config/test_config_pd_hybrid_a2.json)"
export MODEL_EXTRA_CFG_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_PLATFORM="A2"

ADDITIONAL_CONFIG='{"graph_model_compile_config": {"level":1}}'

export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

export ROLE="decode" # use for throw_dequant config
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0

export TNG_HOST_COPY=1
export AUTO_USE_UC_MEMORY=1
export TASK_QUEUE_ENABLE=2
export ENABLE_OVERWRITE_REQ_IDS=1

export OMNI_PD_HYBRID=1


# use vllm cli
vllm serve $MODEL_PATH \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 32 \
  --data-parallel-size-local 8 \
  --data-parallel-address $MASTER_IP \
  --data-parallel-rpc-port 9001 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --additional-config "$ADDITIONAL_CONFIG" \
  "${VLLM_ARGS[@]}"