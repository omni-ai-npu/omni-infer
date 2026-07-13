#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.


set -euo pipefail

INPUT_PATH=""
OUTPUT_PATH=""

# =========================
# 解析参数
# =========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# =========================
# 参数检查
# =========================
if [[ -z "$INPUT_PATH" || -z "$OUTPUT_PATH" ]]; then
    echo "Usage:"
    echo "./gpt_oss_mxfp4_to_int8.sh --input-path <path-fp4> --output-path <path-int8>"
    exit 1
fi

PATH_FP4="$INPUT_PATH"
PATH_INT8="$OUTPUT_PATH"
PATH_BF16="${PATH_FP4}-bf16"

echo "=============================="
echo "FP4 path  : $PATH_FP4"
echo "BF16 path : $PATH_BF16"
echo "INT8 path : $PATH_INT8"
echo "=============================="

# =========================
# 创建目录
# =========================
mkdir -p "$PATH_BF16"
mkdir -p "$PATH_INT8"

# =========================
# 复制非 safetensors 文件
# =========================
echo "Copying config/tokenizer files..."

find "$PATH_FP4" -maxdepth 1 -type f ! -name "*.safetensors" -exec cp {} "$PATH_BF16" \;
find "$PATH_FP4" -maxdepth 1 -type f ! -name "*.safetensors" -exec cp {} "$PATH_INT8" \;

# =========================
# FP4 -> BF16
# =========================
echo "Running fp4_cast_bf16..."

python fp4_cast_bf16.py \
    --input-fp4-hf-path "$PATH_FP4" \
    --output-bf16-hf-path "$PATH_BF16" \
    --gpt-oss

# =========================
# BF16 -> INT8
# =========================
echo "Running quantization..."

python quant_gptoss.py \
    --input-bf16-hf-path "$PATH_BF16" \
    --output-path "$PATH_INT8" \
    --device "cpu"

# =========================
# 删除临时目录
# =========================
echo "Removing temp bf16 directory..."

rm -rf "$PATH_BF16"

echo "================================"
echo "All steps finished successfully."
echo "Output: $PATH_INT8"
echo "================================"