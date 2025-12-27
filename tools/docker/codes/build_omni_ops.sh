#!/bin/bash

set -e

DEFAULT_OMNI_OPS_PATH="/workspace/dist/codes/omni-ops"
DEFAULT_NPU_PLATFORM="910C"

# 解析命令行参数
OMNI_OPS_PATH=""
npu_platform="$DEFAULT_NPU_PLATFORM"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --omni-ops-path)
            OMNI_OPS_PATH="$2"
            shift 2
            ;;
        --npu-platform)
            npu_platform="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# 使用默认值（未传参时）
if [ -z "$OMNI_OPS_PATH" ]; then
    OMNI_OPS_PATH="$DEFAULT_OMNI_OPS_PATH"
fi

# 检查路径是否存在
if [[ ! -d "$OMNI_OPS_PATH" ]]; then
    echo "Error: OPS code path does not exist: $OMNI_OPS_PATH" >&2
    exit 1
fi

# 获取绝对路径
OMNI_OPS_PATH=$(realpath "$OMNI_OPS_PATH")
echo "omni code path: $OMNI_OPS_PATH"

# 设置 ASCEND_PATH
ASCEND_PATH=""
if [ -d /usr/local/Ascend/ascend-toolkit ]; then
    ASCEND_PATH="/usr/local/Ascend/ascend-toolkit"
else
    ASCEND_PATH="/usr/local/Ascend"
fi

if [ -f "$ASCEND_PATH/set_env.sh" ]; then
    source "$ASCEND_PATH/set_env.sh"
fi

# 根据 npu_platform 动态调整 build.sh 参数
cd "$OMNI_OPS_PATH/inference/ascendc"
if [ "$npu_platform" != "$DEFAULT_NPU_PLATFORM" ]; then
    echo "Using custom compute unit: ascend910b (NPU platform: $npu_platform)"
    bash build.sh --disable-check-compatible --compute-unit ascend910b
else
    bash build.sh --disable-check-compatible
fi

cd "$OMNI_OPS_PATH/inference/ascendc/output"
RUN_FILE=$(find . -maxdepth 1 -name "*.run" | head -n1)

if [[ -z "$RUN_FILE" ]]; then
    echo "Error: No *.run package found in output directory!" >&2
    exit 1
fi

chmod +x "$RUN_FILE"
"./$RUN_FILE" --quiet --install-path=$ASCEND_PATH/latest/opp

echo "source $ASCEND_PATH/latest/opp/vendors/omni_custom_ops/bin/set_env.bash" >> ~/.bashrc

echo "Starting ops build_and_install..."
cd "$OMNI_OPS_PATH/inference/ascendc/torch_ops_extension"
bash build_and_install.sh

echo "Success: Custom ops installed successfully."