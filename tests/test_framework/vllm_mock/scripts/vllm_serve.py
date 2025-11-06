# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import runpy
import sys


def run_vllm_serve(tp=1, dp=1, model="/home/kc/models/DeepSeek-V2-Lite"):
    # Define the parameters you want to pass to vllm serve
    additional_config = None
    additional_config = \
'{"graph_model_compile_config": \
{"level":1, "use_ge_graph_cached":false, "block_num_floating_range":50}, \
"enable_hybrid_graph_mode": true}'

#    additional_config = \
#'{"graph_model_compile_config": \
#{"level":1, "use_ge_graph_cached":false, "block_num_floating_range":50}}'                             

    #additional_config = '{"enable_hybrid_graph_mode": true}'                             
    params = [
        "vllm.entrypoints.openai.api_server",
        "--port", "8089",
        "--model", model,
        "--enable-expert-parallel",
        "--max_num_seqs", "4",
        "--max_model_len", "4096",
        "--tensor_parallel_size", f"{tp}",
        "--data_parallel_size", f"{dp}",
        "--gpu_memory_utilization", "0.6",
        "--trust_remote_code",
        "--served-model-name", "qwen",
        "--dtype", "bfloat16",
        "--distributed-executor-backend", "mp",
        "--block_size", "512",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--additional-config", additional_config,
    ]
	#'--speculative-config', '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}',

    # Set sys.argv to include the script name and all parameters
    sys.argv = params

    # Run the vllm serve command using runpy
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == "__main__":
    # Set environment variables
    os.environ['VLLM_ENABLE_MC2'] = '0'
    os.environ['VLLM_USE_V1'] = '1'
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    #os.environ['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] = "1"
    os.environ['HCCL_CONNECT_TIMEOUT'] = "3600"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    os.environ["USING_LCCL_COM"] = "0"

    os.environ["MOE_DISPATCH_COMBINE"] = "1"
    os.environ["ASCEND_PLATFORM"] = "A3"
    # os.environ["RANDOM_MODE"] = "1"
    # os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
    os.environ["HCCL_INTRA_PCIE_ENABLE"] = "0"
    os.environ["HCCL_BUFFSIZE"] = "1000"
    os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
    
    #run_vllm_serve(tp=8, dp=1, model="/home/ma-user/work/models/Qwen/Qwen3-Next")
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "0"
    os.environ["PROFILING_SAVE_PATH"] = "/tmp/omni_logs"
    #os.environ["TNG_LOG_LEVEL"] = "0"

    os.environ['OMNI_USE_QWEN'] = '1'  # Enable custom model support
    os.environ['VLLM_LOGGING_LEVEL'] = 'INFO'
    os.environ['TNG_HOST_COPY'] = '1'
    os.environ['TASK_QUEUE_ENABLE'] = '2'
    os.environ['CPU_AFFINITY_CONF'] = '2'

    run_vllm_serve(tp=8, dp=1, model="/data/model/Qwen3-Next-80B-A3B-Instruct-Mini/")
    #run_vllm_serve(tp=8, dp=1, model="/data/model/Qwen3-Next-80B-A3B-Instruct/")
    
