#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# 通用 OpenPangu VL / OMNI JointFix-MDMixQ W8A8 量化入口。
# MTP、router、shared expert 固定 BF16；所有被量化 Linear 均为统一 W8A8。
# decoder manifest 推荐同时包含长纯文本、图片和音频请求。
set -euo pipefail

MODE="${1:-}"
STEP="${2:-all}"
if [[ "$MODE" != "vl" && "$MODE" != "omni" ]]; then
    echo "用法: $0 <vl|omni> [decoder|vision|audio|finalize|merge|all]"
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${JOINTFIX_ROOT:-$SCRIPT_DIR}"
source "${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

: "${MODEL:?请设置原始 BF16 checkpoint: MODEL=/path/to/model}"

export PYTHONPATH=".:${OMNI_MODELS_ROOT:-$SCRIPT_DIR/../../..}:${PYTHONPATH:-}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-omni_pangu_models}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export PANGU_TORCH_USE_NPU_DSA="${PANGU_TORCH_USE_NPU_DSA:-0}"

run_decoder() {
    : "${CALIB_MANIFEST:?请设置 decoder 多模态 JSONL: CALIB_MANIFEST=/path/to/calib.jsonl}"
    : "${LAYER_DIR:?请设置逐层输出目录: LAYER_DIR=/path/to/layers}"
    [[ ! -e "$LAYER_DIR" ]] || { echo "LAYER_DIR 已存在，请换新目录: $LAYER_DIR"; return 1; }
    python -X faulthandler -u -m jointfix.cli quantize \
        --backend pangu --method jointfix-mdmixq \
        --model "$MODEL" --output "$LAYER_DIR" \
        --calib-data "$CALIB_MANIFEST" --calib-format omni \
        --n-samples "${N_SAMPLES:-32}" --seq-len "${SEQ_LEN:-1024}" \
        --mm-min-pixels "${MIN_PIXELS:-50176}" \
        --mm-max-pixels "${MAX_PIXELS:-401408}" \
        --num-iterations "${NUM_ITERATIONS:-2}" \
        --iter-ab-tol "${ITER_AB_TOL:-0.05}" \
        --num-devices "${NUM_DEVICES:-16}" --device npu \
        --objective output-recon --write-quant gptq \
        --skip-shared-experts --no-finalize
}

run_tower() {
    local modality="$1" manifest="$2"
    : "${MM_ARTIFACTS:?请设置 tower 逐层输出目录: MM_ARTIFACTS=/path/to/mm_layers}"
    [[ -f "$manifest" ]] || { echo "校准 manifest 不存在: $manifest"; return 1; }
    python -X faulthandler -u examples/quantize_omni_multimodal.py calibrate \
        --model "$MODEL" --output "$MM_ARTIFACTS" --modality "$modality" \
        --"${modality}"-manifest "$manifest" \
        --"n-${modality}-samples" "${N_TOWER_SAMPLES:-32}" \
        --sample-rows "${SAMPLE_ROWS:-512}" \
        --forward-token-budget "${FORWARD_TOKEN_BUDGET:-4096}" \
        --gptq-block-size "${GPTQ_BLOCK_SIZE:-128}" --gptq-damp "${GPTQ_DAMP:-0.01}" \
        --device npu
}

run_finalize() {
    : "${LAYER_DIR:?请设置逐层输出目录: LAYER_DIR=/path/to/layers}"
    : "${TEXT_BASE:?请设置 decoder+BF16-MTP checkpoint: TEXT_BASE=/path/to/text_base}"
    [[ ! -e "$TEXT_BASE" ]] || { echo "TEXT_BASE 已存在，请换新目录: $TEXT_BASE"; return 1; }
    python -u examples/quantize_omni_multimodal.py finalize-text \
        --model "$MODEL" --text-artifacts "$LAYER_DIR" --output "$TEXT_BASE"
}

run_merge() {
    : "${MM_ARTIFACTS:?请设置 MM_ARTIFACTS}"
    : "${TEXT_BASE:?请设置 decoder+BF16-MTP checkpoint: TEXT_BASE=/path/to/text_base}"
    : "${FINAL_MODEL:?请设置最终 checkpoint: FINAL_MODEL=/path/to/final_model}"
    [[ ! -e "$FINAL_MODEL" ]] || { echo "FINAL_MODEL 已存在，请换新目录: $FINAL_MODEL"; return 1; }
    local args=(--base-quant "$TEXT_BASE" --artifacts "$MM_ARTIFACTS"
                --output "$FINAL_MODEL" --require-vision)
    if [[ "$MODE" == "omni" ]]; then
        args+=(--require-audio)
    else
        args+=(--no-require-audio)
    fi
    python -u examples/quantize_omni_multimodal.py merge "${args[@]}"
}

case "$STEP" in
    decoder) run_decoder ;;
    vision) : "${VISION_MANIFEST:?请设置 VISION_MANIFEST}"; run_tower vision "$VISION_MANIFEST" ;;
    audio)
        [[ "$MODE" == "omni" ]] || { echo "VL 模型不执行 audio"; exit 2; }
        : "${AUDIO_MANIFEST:?请设置 AUDIO_MANIFEST}"; run_tower audio "$AUDIO_MANIFEST" ;;
    finalize) run_finalize ;;
    merge) run_merge ;;
    all)
        run_decoder
        run_finalize
        : "${VISION_MANIFEST:?请设置 VISION_MANIFEST}"; run_tower vision "$VISION_MANIFEST"
        if [[ "$MODE" == "omni" ]]; then
            : "${AUDIO_MANIFEST:?请设置 AUDIO_MANIFEST}"; run_tower audio "$AUDIO_MANIFEST"
        fi
        run_merge
        ;;
    *) echo "未知步骤: $STEP"; exit 2 ;;
esac
