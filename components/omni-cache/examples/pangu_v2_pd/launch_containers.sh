#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# launch_containers.sh — create and start Docker containers for the
# Pangu V2 hybrid 1P-1D deployment on a single node.
#
# This script only creates containers.  Use launch_pd.sh afterwards to
# start the vLLM / proxy services inside them.
#
# Topology (1 node, 16 NPU cards):
#   P (prefill)  cards 0-7   TP=8
#   D (decode)   cards 8-15  DP=8 TP=1
#   Proxy runs inside the P container.
#
# Usage:
#   DOCKER_IMAGE_ID=<image> bash launch_containers.sh
#   DOCKER_IMAGE_ID=<image> LAUNCH_MODE=all bash launch_containers.sh
#
#   # Create only prefill container
#   DOCKER_IMAGE_ID=<image> LAUNCH_MODE=prefill bash launch_containers.sh
#
#   # Create only decode container
#   DOCKER_IMAGE_ID=<image> LAUNCH_MODE=decode bash launch_containers.sh
#
# Container names (overridable, same defaults as launch_pd.sh):
#   PREFILL_CONTAINER  default omnicache_pangu_p0
#   DECODE_CONTAINER   default omnicache_pangu_d0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Load configuration ──────────────────────────────────────────────────
source "$SCRIPT_DIR/configs/base.sh"
if [[ -n "${CONFIG_PROFILE:-}" ]]; then
    profile_path="$SCRIPT_DIR/configs/${CONFIG_PROFILE}.sh"
    [[ -f "$profile_path" ]] || {
        echo "ERROR: config profile not found: $profile_path" >&2
        exit 1
    }
    source "$profile_path"
fi

# ─── Required: docker image ──────────────────────────────────────────────
if [[ -z "${DOCKER_IMAGE_ID:-}" ]]; then
    echo "ERROR: DOCKER_IMAGE_ID is required" >&2
    echo "  DOCKER_IMAGE_ID=<image> bash $0" >&2
    exit 1
fi

# ─── Container names (match launch_pd.sh defaults) ───────────────────────
: "${PREFILL_CONTAINER:=omnicache_pangu_p0}"
: "${DECODE_CONTAINER:=omnicache_pangu_d0}"

: "${LAUNCH_MODE:=all}"

# ─── Paths (overridable) ─────────────────────────────────────────────────
: "${LOG_PATH:=/data/logs}"
: "${SCRIPTS_PATH:=/data/ldy/PanguV2/92B_service/omni-cache}"
: "${SHM_SIZE:=500g}"

# ─── Docker run base command ─────────────────────────────────────────────
_docker_run() {
    local name="$1"
    shift
    sudo -n docker run -dit \
        --name "$name" \
        --shm-size="$SHM_SIZE" \
        -e LOG_PATH="$LOG_PATH" \
        -e PYTHONHASHSEED=123 \
        --net=host \
        --privileged=true \
        -u root \
        -w /data \
        --device=/dev/davinci_manager \
        --device=/dev/hisi_hdc \
        --device=/dev/devmm_svm \
        --entrypoint=bash \
        -v /data:/data \
        -v /mnt:/mnt \
        -v /tmp:/tmp \
        -v /home:/home \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /etc/hccn.conf:/etc/hccn.conf \
        -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
        -v "$LOG_PATH":"$LOG_PATH" \
        -v "$MODEL_PATH":"$MODEL_PATH" \
        -v "$SCRIPTS_PATH":"$SCRIPTS_PATH" \
        -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
        "$@" \
        "$DOCKER_IMAGE_ID"
}

# ─── Helper: create one container (skip if already running) ──────────────
_create_container() {
    local name="$1"
    shift
    if sudo -n docker ps -q -f "name=^${name}$" | grep -q .; then
        echo "[launch_containers] container '$name' already running — skipping"
        return 0
    fi
    if sudo -n docker ps -aq -f "name=^${name}$" | grep -q .; then
        echo "[launch_containers] container '$name' exists but stopped — removing and recreating"
        sudo -n docker rm "$name"
    fi
    echo "[launch_containers] creating container '$name'..."
    _docker_run "$name" "$@"
}

echo "=== Creating 1P-1D containers ==="
echo "    image:       $DOCKER_IMAGE_ID"
echo "    mode:        $LAUNCH_MODE"
echo "    model_path:  $MODEL_PATH"
echo "    scripts:     $SCRIPTS_PATH"

# ─── Create containers based on LAUNCH_MODE ──────────────────────────────
if [[ "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "prefill" ]]; then
    _create_container "$PREFILL_CONTAINER"
fi

if [[ "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "decode" ]]; then
    _create_container "$DECODE_CONTAINER"
fi

echo "[launch_containers] Done. Containers created:"
[[ "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "prefill" ]] && echo "    $PREFILL_CONTAINER (P)"
[[ "$LAUNCH_MODE" == "all" || "$LAUNCH_MODE" == "decode" ]]  && echo "    $DECODE_CONTAINER (D)"
[[ "$LAUNCH_MODE" == "all" ]] && echo "    Proxy will reuse $PREFILL_CONTAINER (P)"
echo ""
echo "Next: run launch_pd.sh to start services inside these containers."