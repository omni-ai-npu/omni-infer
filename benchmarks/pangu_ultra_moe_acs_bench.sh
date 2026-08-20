#!/bin/bash
#
# acs-bench 性能压测脚本（pangu_ultra_moe / OpenAI Chat 后端）
#
# 功能：通过 acs-bench prof 对 vLLM 服务执行长文本（8K 输入 / 2K 输出）
#       性能压测，开启投机推理（spec decode, 3 个 spec token），采用限速请求
#       注入（request-rate 4/s，burstiness 100），warmup 2000 请求后统计指标。
#
# 前置条件：
#   1. 已安装 acs-bench：pip show acs-bench
#   2. 待测服务已启动且可访问（base_url 指向服务 /v1 地址）
#   3. 数据集 claw_all_openai.json 已生成（参考 benchmarks/README.md）
#
# 用法：
#   bash benchmarks/pangu_ultra_moe_acs_bench.sh
#   或通过环境变量覆盖默认值：
#   BASE_URL=http://x.x.x.x:7000/v1 INPUT_PATH=/data/claw_all_openai.json \
#       bash benchmarks/pangu_ultra_moe_acs_bench.sh
set -e

BASE_URL=${BASE_URL:-http://172.0.0.1:7000/v1}
INPUT_PATH=${INPUT_PATH:-/your/datasets/path/claw_all_openai.json}

acs-bench prof \
    --model-args "[{\"model_name\": \"pangu_ultra_moe\", \"base_url\": \"${BASE_URL}\"}]" \
    --dataset-type CustomOpenAIChat \
    --input-path "${INPUT_PATH}" \
    --concurrency-backend threading-pool \
    --backend openai-chat \
    --generation-config '{"chat_template_kwargs": {"thinking": false}}' \
    --epochs 1 \
    --num-requests 10155 \
    --concurrency 1024 \
    --input-length 8192 \
    --output-length 2000 \
    --trust-remote-code \
    --use-spec-decode \
    --num-spec-tokens 3 \
    --request-rate 4 \
    --temperature 1.0 \
    --top-p 0.8 \
    --top-k 100 \
    --ignore-eos False \
    --timeout 7200 \
    --burstiness 100 \
    --random-seed -1 \
    --warmup 2000
