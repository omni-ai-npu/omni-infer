# PR 标题&描述
## 标题：【需求合入】/【Bugfix】/【任务合入】 | 【标签N】 问题描述：xxx
【标签N】：可以是如下其中的一个或者多个
- **[CI/Build]**：构建或 CI 改进
- **[Doc]**：文档修改或改进
- **[Model]**：新增或改进模型（**模型名称必须出现在标题中**）
- **[Frontend]**：前端相关变更（如 OpenAI API Server、LLM 类等）
- **[Kernel]**：CAAN 或其他计算内核变更
- **[Core]**：核心逻辑变更（如 LLMEngine、Scheduler、NPU Model Runner、NPU worker 等）
  

## 描述：
### 【Bugfix】：【缺陷描述】【缺陷链接】
### 【需求合入】 ：【需求描述】【需求链接】
### 【任务合入】：【任务描述】【任务链接】
### 【Test Plan】：测试命令    
### 【Test Result】：测试结果

<br>

# 样例 -【bugfix】:
## 标题：【Bugfix】 | 【Core】 问题描述：反思截断测试问题，修改合入
## 描述：
### 【缺陷描述】：【release_v0.8.0】【A3】【Pangu-718B_8816】【6P8-1D16】反思截断测试问题：3、设置全局变量0，think_budget设置0，流式输出没有思考内容，期望输出全部思考内容
### 【缺陷链接】：https://e.gitee.com/omniai/issues/table?issue=IDH3TV 

### 【Test Plan】：测试命令  
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
        export ASCEND_GLOBAL_LOG_LEVEL=3
        model="/linziming/iter_0015075_pp1_final_hf_128f_fix_resize/"
        log_file="/data/linziming/log/encode_server.log"

        export GLOO_SOCKET_IFNAME=enp23s0f3
        export HCCL_INTRA_ROCE_ENABLE=1
        export HCCL_INTRA_PCIE_ENABLE=0
        export VLLM_WORKER_MULTIPROC_METHOD=fork
        #export VLLM_LOGGING_LEVEL=DEBUG

        export OMNI_NPU_VLLM_PATCHES="ALL"
        export VLLM_PLUGINS="omni-npu,omni_npu_patches,omni_pangu_models"
        export OMNI_NPU_PATCHES_DIR="openpangu_vl"

        vllm serve \
            "$model" \
            --served-model-name pangu \
            --port 8101 \
            --dtype bfloat16 \
            --max-model-len 82000 \
            --max-num-batched-tokens 82000 \
            --max-num-seqs 16 \
            --no-enable-chunked-prefill \
            --no-enable-prefix-caching \
            --mm-processor-cache-gb 0 \
            --distributed-executor-backend mp \
            --gpu-memory-utilization 0.01 \
            --trust-remote-code \
            --allowed-local-media-path / \
            --tensor-parallel-size 4 \
            --data-parallel-size 1 \
            --enable-expert-parallel \
            --allowed-local-media-path / \
            --ec-transfer-config '{"ec_connector": "ECNetworkConnector", "ec_role": "ec_producer", "ec_port": 16789, "ec_connector_extra_config":{"ec_cache_max_gb": 20}}' \
            --enforce-eager &> "${log_file}"
    
### 【Test Result】：测试结果
        pd测试结果
        026-04-09 17:31:07,574 - benchmark_utils - INFO - 所有请求耗时: 76.689 s
        2026-04-09 17:31:07,575 - benchmark_utils - INFO - 请求吞吐: 13.35 requests/s
        2026-04-09 17:31:07,575 - benchmark_utils - INFO - 输出tokens总吞吐: 13.35 tokens/s
        2026-04-09 17:31:07,575 - benchmark_utils - INFO - 输入+输出tokens总吞吐: 948.04 tokens/s
        2026-04-09 17:31:07,576 - benchmark_utils - INFO - 首tokens时延TP90: 2609.000 ms
        2026-04-09 17:31:07,576 - benchmark_utils - INFO - 首tokens时延TP95: 2665.000 ms
        2026-04-09 17:31:07,576 - benchmark_utils - INFO - 首tokens时延TP99: 2857.000 ms
        2026-04-09 17:31:07,576 - benchmark_utils - INFO - 最大首tokens时延: 3195.000 ms
        2026-04-09 17:31:07,576 - benchmark_utils - INFO - 平均首tokens时延: 2345.000 ms
        2026-04-09 17:31:07,578 - benchmark_utils - INFO - 增量时延TP90: 0.000 ms
        2026-04-09 17:31:07,578 - benchmark_utils - INFO - 增量时延TP95: 0.000 ms
        2026-04-09 17:31:07,578 - benchmark_utils - INFO - 增量时延TP99: 0.000 ms
        2026-04-09 17:31:07,578 - benchmark_utils - INFO - 最大增量时延: 0.000 ms
        2026-04-09 17:31:07,578 - benchmark_utils - INFO - 平均增量时延: -0.000 ms
        2026-04-09 17:31:07,578 - benchmark_utils - INFO - 端到端请求时延TP90: 2.609 s
        2026-04-09 17:31:07,579 - benchmark_utils - INFO - 端到端请求时延TP95: 2.665 s
        2026-04-09 17:31:07,579 - benchmark_utils - INFO - 端到端请求时延TP99: 2.857 s
        2026-04-09 17:31:07,579 - benchmark_utils - INFO - 最大端到端请求时延: 3.195 s
        2026-04-09 17:31:07,579 - benchmark_utils - INFO - 平均端到端请求时延: 2.345 s

        epd测试结果
        026-04-09 19:31:21,136 - benchmark_utils - INFO - 所有请求耗时: 70.475 s
        2026-04-09 19:31:21,137 - benchmark_utils - INFO - 请求吞吐: 14.53 requests/s
        2026-04-09 19:31:21,137 - benchmark_utils - INFO - 输出tokens总吞吐: 14.53 tokens/s
        2026-04-09 19:31:21,137 - benchmark_utils - INFO - 输入+输出tokens总吞吐: 1031.63 tokens/s
        2026-04-09 19:31:21,138 - benchmark_utils - INFO - 首tokens时延TP90: 2441.000 ms
        2026-04-09 19:31:21,138 - benchmark_utils - INFO - 首tokens时延TP95: 2546.000 ms
        2026-04-09 19:31:21,138 - benchmark_utils - INFO - 首tokens时延TP99: 2841.000 ms
        2026-04-09 19:31:21,138 - benchmark_utils - INFO - 最大首tokens时延: 3632.000 ms
        2026-04-09 19:31:21,138 - benchmark_utils - INFO - 平均首tokens时延: 2146.000 ms
        2026-04-09 19:31:21,140 - benchmark_utils - INFO - 增量时延TP90: 0.000 ms
        2026-04-09 19:31:21,140 - benchmark_utils - INFO - 增量时延TP95: 0.000 ms
        2026-04-09 19:31:21,140 - benchmark_utils - INFO - 增量时延TP99: 0.000 ms
        2026-04-09 19:31:21,140 - benchmark_utils - INFO - 最大增量时延: 0.000 ms
        2026-04-09 19:31:21,140 - benchmark_utils - INFO - 平均增量时延: -0.000 ms
        2026-04-09 19:31:21,140 - benchmark_utils - INFO - 端到端请求时延TP90: 2.441 s
        2026-04-09 19:31:21,141 - benchmark_utils - INFO - 端到端请求时延TP95: 2.546 s
        2026-04-09 19:31:21,141 - benchmark_utils - INFO - 端到端请求时延TP99: 2.841 s
        2026-04-09 19:31:21,141 - benchmark_utils - INFO - 最大端到端请求时延: 3.632 s
        2026-04-09 19:31:21,141 - benchmark_utils - INFO - 平均端到端请求时延: 2.146 s



# 样例 -【需求合入】
## 标题：【需求合入】 | 【Core】 需求描述：pd分离支持返回apc缓存token数量和缓存命中率
## 描述：
### 【需求描述】 pd分离支持返回apc缓存token数量和缓存命中率
### 【需求链接】https://e.gitee.com/omniai/issues/table?issue=IDEVA3   

###  【 Test Plan】：测试命令  

        1、pd分离支持返回apc缓存token数量和缓存命中率
            默认开启apc
            --no-enable-prefix-caching 则不开启apc
            --enable-prompt-tokens-details 返回apc缓存相关信息



### 【Test Result】：测试结果
        开启命中 
        {"id": "chatcmpl-e0acfe31-a0e6-420c-bf6d-eebbf7e3b759", "object": "chat.completion.chunk", "created": 1776329506, "model": "pangu", "choices": [], "usage": {"prompt_tokens": 1330, "total_tokens": 1332, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 1280, "cached_rate": 0.9668}}}
        开启没命中 
        {"id": "chatcmpl-e0acfe31-a0e6-420c-bf6d-eebbf7e3b759", "object": "chat.completion.chunk", "created": 1776329506, "model": "pangu", "choices": [], "usage": {"prompt_tokens": 1330, "total_tokens": 1332, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0, "cached_rate": 0.0}}}
        没开启
        {"id": "chatcmpl-e0acfe31-a0e6-420c-bf6d-eebbf7e3b759", "object": "chat.completion.chunk", "created": 1776329506, "model": "pangu", "choices": [], "usage": {"prompt_tokens": 1330, "total_tokens": 1332, "completion_tokens": 2, "prompt_tokens_details": null}}



# 样例 -【任务合入】
      ---同 “样例 -【需求合入】”
    【任务】--- 可以是 专项代码review等活动