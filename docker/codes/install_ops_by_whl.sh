#!/bin/bash


DEFAULT_OPS_PATH="/workspace/dist/codes"
OPS_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ops-path)
            OPS_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$OPS_PATH" ]]; then
    OPS_PATH="$DEFAULT_OPS_PATH"
fi

if [[ ! -d "$OPS_PATH" ]]; then
    echo "Error: OPS code path does not exist: $OPS_PATH" >&2
    exit 1
fi

ASCEND_PATH=""
if [ -d /usr/local/Ascend/ascend-toolkit ]; then
    ASCEND_PATH="/usr/local/Ascend/ascend-toolkit"
else
    ASCEND_PATH="/usr/local/Ascend"
fi

cd $OPS_PATH
echo $OPS_PATH

# -------------train-------------------
TRAIN_RUN_FILE=$(find . -maxdepth 2 -name "*omni_training_custom_ops*.run" | head -n1)
TRAIN_WHL_FILE=$(find . -maxdepth 2 -name "*omni_training_ascendc_custom_ops*.whl" | head -n1)
echo "TRAIN_RUN_FILE: $TRAIN_RUN_FILE"
echo "TRAIN_WHL_FILE: $TRAIN_WHL_FILE"

if [[ -n "$TRAIN_RUN_FILE" ]] && [[ -n "$TRAIN_WHL_FILE" ]]; then
    chmod +x "$TRAIN_RUN_FILE"
    "./$TRAIN_RUN_FILE" --quiet --install-path="$ASCEND_PATH"
    if [ -f "$ASCEND_PATH/vendors/omni_training_custom_ops/bin/set_env.bash" ]; then
        echo "source $ASCEND_PATH/vendors/omni_training_custom_ops/bin/set_env.bash" >> ~/.bashrc
    fi
    if [ -f "$ASCEND_PATH/vendors/omni_training_custom_transformer/bin/set_env.bash" ]; then
        echo "source $ASCEND_PATH/vendors/omni_training_custom_transformer/bin/set_env.bash" >> ~/.bashrc
    fi
    pip install "$TRAIN_WHL_FILE" 2>/dev/null
fi


# -------------inference-------------------
INFERENCE_RUN_FILE=$(find . -maxdepth 2 -name "*omni_inference_custom_ops*.run" | head -n1)
INFERENCE_WHL_FILE=$(find . -maxdepth 2 -name "*omni_inference_ascendc_custom_ops*.whl" | head -n1)
echo "INFERENCE_RUN_FILE: $INFERENCE_RUN_FILE"
echo "INFERENCE_WHL_FILE: $INFERENCE_WHL_FILE"

if [[ -n "$INFERENCE_RUN_FILE" ]] && [[ -n "$INFERENCE_WHL_FILE" ]]; then
    chmod +x "$INFERENCE_RUN_FILE"
    "./$INFERENCE_RUN_FILE" --quiet --install-path="$ASCEND_PATH"
    if [ -f "$ASCEND_PATH/vendors/omni_custom_ops/bin/set_env.bash" ]; then
        echo "source $ASCEND_PATH/vendors/omni_custom_ops/bin/set_env.bash" >> ~/.bashrc
    fi
    if [ -f "$ASCEND_PATH/vendors/omni_custom_transformer/bin/set_env.bash" ]; then
        echo "source $ASCEND_PATH/vendors/omni_custom_transformer/bin/set_env.bash" >> ~/.bashrc
    fi
    pip install "$INFERENCE_WHL_FILE" 2>/dev/null
fi


#--------------third----------------
THIRD_RUN_FILE=$(find . -maxdepth 2 -name "CANN-custom_ops*.run" | head -n1)
THIRD_WHL_FILE=$(find . -maxdepth 2 -name "custom_ops*.whl" | head -n1)
echo "THIRD_RUN_FILE: $THIRD_RUN_FILE"
echo "THIRD_WHL_FILE: $THIRD_WHL_FILE"

if [[ -n "$THIRD_RUN_FILE" ]] && [[ -n "$THIRD_WHL_FILE" ]]; then
    chmod +x "$THIRD_RUN_FILE"
    "./$THIRD_RUN_FILE" --quiet --install-path="$ASCEND_PATH"
    echo "source $ASCEND_PATH/latest/opp/vendors/customize/bin/set_env.bash" >> ~/.bashrc
    pip install "$THIRD_WHL_FILE" 2>/dev/null
fi