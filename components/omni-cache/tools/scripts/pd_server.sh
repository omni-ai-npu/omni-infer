#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# pd_server.sh — Start prefill or decode server with correct Ascend environment.
#
# Usage:
#   bash tools/scripts/pd_server.sh prefill [extra_env_vars...]
#   bash tools/scripts/pd_server.sh decode [extra_env_vars...]
#
# Examples:
#   bash tools/scripts/pd_server.sh prefill DUMP_KV_TENSOR=1
#   bash tools/scripts/pd_server.sh decode
#
# The script handles the common issue where docker restart clears the
# Ascend CANN environment variables. It sources set_env.sh before
# launching the server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ascend CANN environment setup
ASCEND_SET_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"
if [[ ! -f "$ASCEND_SET_ENV" ]]; then
    ASCEND_SET_ENV="/usr/local/Ascend/cann/set_env.sh"
fi
if [[ ! -f "$ASCEND_SET_ENV" ]]; then
    echo "ERROR: Cannot find Ascend set_env.sh" >&2
    exit 1
fi

ROLE="${1:-prefill}"
shift || true

# Source Ascend environment first
source "$ASCEND_SET_ENV"

# Build extra environment from remaining args
EXTRA_ENV=""
for arg in "$@"; do
    EXTRA_ENV="$EXTRA_ENV $arg"
done

# Launch the appropriate server
case "$ROLE" in
    prefill)
        cd "$REPO_ROOT"
        eval "${EXTRA_ENV} bash examples/pangu_v2_pd/launch_prefill.sh"
        ;;
    decode)
        cd "$REPO_ROOT"
        eval "${EXTRA_ENV} bash examples/pangu_v2_pd/launch_decode.sh"
        ;;
    *)
        echo "ERROR: Unknown role '$ROLE'. Use 'prefill' or 'decode'." >&2
        exit 1
        ;;
esac