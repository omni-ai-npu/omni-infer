#!/bin/bash

set -e

DEFAULT_OPS_CODE_PATH="/workspace/dist/codes/cann-recipes-infer"
DEFAULT_NPU_PLATFORM="910C"

OPS_CODE_PATH=""
NPU_PLATFORM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --omni-ops-path)
            OPS_CODE_PATH="$2"
            shift 2
            ;;
        --npu-platform)
            NPU_PLATFORM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$OPS_CODE_PATH" ]]; then
    OPS_CODE_PATH="${DEFAULT_OPS_CODE_PATH}"
fi

if [[ -z "$NPU_PLATFORM" ]]; then
    NPU_PLATFORM="${DEFAULT_NPU_PLATFORM}"
fi

if [[ ! -d "$OPS_CODE_PATH" ]]; then
    echo "Error: OPS code path does not exist: $OPS_CODE_PATH" >&2
    exit 1
fi

OPS_CODE_PATH=$(realpath "$OPS_CODE_PATH")
echo "CANN recipes code path: $OPS_CODE_PATH"
echo "Target NPU platform: $NPU_PLATFORM"

ASCEND_PATH=""
if [ -d /usr/local/Ascend/ascend-toolkit ]; then
    ASCEND_PATH="/usr/local/Ascend/ascend-toolkit"
else
    ASCEND_PATH="/usr/local/Ascend"
fi

if [ -f "$ASCEND_PATH/set_env.sh" ]; then
    source "$ASCEND_PATH/set_env.sh"
fi

BUILD_ARGS="--disable-check-compatible"
if [[ "$NPU_PLATFORM" != "910C" ]]; then
    BUILD_ARGS="$BUILD_ARGS --compute-unit ascend910b"
fi

cd "$OPS_CODE_PATH/ops/ascendc"
bash build.sh $BUILD_ARGS

cd "$OPS_CODE_PATH/ops/ascendc/output"
RUN_FILE=$(find . -maxdepth 1 -name "CANN-custom_ops-*.run" | head -n1)

if [[ -z "$RUN_FILE" ]]; then
    echo "Error: No CANN-custom_ops-*.run package found in output directory!" >&2
    exit 1
fi

chmod +x "$RUN_FILE"
"./$RUN_FILE" --quiet --install-path="$ASCEND_PATH/latest/opp"

echo "source $ASCEND_PATH/latest/opp/vendors/customize/bin/set_env.bash" >> ~/.bashrc

echo "Starting ops build_and_install..."
cd "$OPS_CODE_PATH/ops/ascendc/torch_ops_extension"
bash build_and_install.sh

echo "Success: Custom ops installed successfully."