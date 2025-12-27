# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

export GLOO_SOCKET_IFNAME=enp23s0f3
export TP_SOCKET_IFNAME=enp23s0f3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0
export OMNI_USE_PANGU=1

export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_RDMA_TIMEOUT=5
export HCCL_DETERMINISTIC=False
export ASCEND_GLOBAL_LOG_LEVEL=3
export CPU_AFFINITY_CONF=2
export VLLM_LOGGING_LEVEL=INFO
export HCCL_BUFFSIZE=1000
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
# chunk prefill
export FORCE_ENABLE_CHUNK_PREFILL=1
export VLLM_ENABLE_SLEEP_MODE=1

python start_api_servers.py \
        --num-servers 4 \
        --model-path /data/models/Pangu72B \
        --master-ip 0.0.0.0 \
        --tp 4 \
        --num-dp 4 \
        --master-port 6660 \
        --served-model-name Pangu72B \
        --log-dir apiserverlog \
        --extra-args "--max-num-batched-tokens 2048 --enforce-eager  --enable-expert-parallel --disable-log-requests --max-num-seqs 16 --long-prefill-token-threshold 2048 --enable-reasoning --reasoning-parser pangu --enable-auto-tool-choice --tool-call-parser pangu"\
        --base-api-port 8001 \
        --gpu-util 0.95 \
        --max-model-len 131072 \
        --additional-config '{"graph_model_compile_config":{"level":1, "use_ge_graph_cached":true, "decode_gear_list": [16]}, "enable_hybrid_graph_mode": false, "expert_parallel_size": 16, "expert_tensor_parallel_size": 1}'

