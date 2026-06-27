#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Script to run magic number detection for omni-cache

set -e

# 记录开始时间
START_TIME=$(date +%s)

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MAGIC_NUM_DIR="$SCRIPT_DIR/magic_num"
TARGET_PATH="${PROJECT_ROOT}/omni_cache"

# Default values
STRICT_MODE=false
DIFF_MODE=false
OUTPUT_DIR="$SCRIPT_DIR/htmlcov_magic_num"
CONFIG_FILE="magic_rule.yaml"

usage() {
    echo "Usage: $0 [OPTIONS] [TARGET_PATH]"
    echo ""
    echo "Semgrep 魔鬼数字检测 - HTML 报告生成器"
    echo ""
    echo "Options:"
    echo "  --strict          严格模式：0, 1, -1 也作为 WARNING"
    echo "  --diff            增量检测：只检测当前分支未提交的变更文件"
    echo "  --output DIR      HTML 输出目录 (默认: tests/htmlcov_magic_num)"
    echo "  --config FILE     Semgrep 规则配置文件 (默认: magic_rule.yaml)"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # 默认模式检测 omni_cache 目录"
    echo "  $0 --strict                     # 严格模式"
    echo "  $0 --diff                       # 增量检测（当前分支未提交的变更）"
    echo "  $0 --strict --diff              # 严格模式 + 增量检测"
    echo "  $0 /path/to/project             # 指定检测目标"
    echo ""
    echo "检测 5 类魔鬼数字："
    echo "  - 赋值语句 (magic-number-assign)"
    echo "  - 比较运算 (magic-number-compare)"
    echo "  - 加减乘除运算 (magic-number-arithmetic)"
    echo "  - 数组索引 (magic-number-index)"
    echo "  - 函数参数 (magic-number-func-arg)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --diff)
            DIFF_MODE=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            TARGET_PATH="$1"
            shift
            ;;
    esac
done

# Check if target path exists
if [ ! -d "$TARGET_PATH" ] && [ ! -f "$TARGET_PATH" ]; then
    echo "[ERROR] 目标路径不存在: $TARGET_PATH"
    exit 1
fi

# Check if semgrep is installed
if ! command -v semgrep &> /dev/null; then
    echo "[INFO] semgrep 未安装，正在安装..."
    pip install semgrep
fi

# Check if config file exists
CONFIG_PATH="${MAGIC_NUM_DIR}/${CONFIG_FILE}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] 规则文件不存在: $CONFIG_PATH"
    exit 1
fi

# Build command
cmd_args=(
    python3 "${MAGIC_NUM_DIR}/generate_html_report.py"
    "$TARGET_PATH"
    --config "$CONFIG_PATH"
    --output "$OUTPUT_DIR"
)

if [ "$STRICT_MODE" = true ]; then
    cmd_args+=(--strict)
fi

if [ "$DIFF_MODE" = true ]; then
    cmd_args+=(--diff)
fi

# Print info
echo "============================================================"
echo "Semgrep 魔鬼数字检测"
echo "============================================================"
echo "  目标路径: $TARGET_PATH"
echo "  规则文件: $CONFIG_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  严格模式: $STRICT_MODE"
echo "  增量检测: $DIFF_MODE"
echo "============================================================"

# Run detection
"${cmd_args[@]}"

# 计算耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "[INFO] 报告已生成: ${OUTPUT_DIR}/index.html"
echo "[INFO] 耗时: ${MINUTES}分${SECONDS}秒"