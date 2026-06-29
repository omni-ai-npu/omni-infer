#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Quick KV verification: send requests + compare baseline vs omnicache.
#
# Usage:
#   1. Start PD servers with mock schedule + deterministic compute:
#      ENABLE_OMNI_CACHE=0 OMNI_MOCK_SCHEDULE=1 ... bash launch_pd.sh
#
#   2. Run baseline:
#      bash tools/scripts/quick_verify.sh send baseline
#
#   3. Restart PD servers with ENABLE_OMNI_CACHE=1
#
#   4. Run omnicache:
#      bash tools/scripts/quick_verify.sh send omnicache
#
#   5. Compare:
#      bash tools/scripts/quick_verify.sh compare
set -euo pipefail
cd "$(dirname "$0")/.."

# Must match the server's --max-num-seqs.  No hardcoded default —
# the caller is responsible for setting N to the correct value.
if [[ -z "${N:-}" ]]; then
    echo "ERROR: N is required — must match server --max-num-seqs"
    exit 1
fi
MAX_TOKENS="${MAX_TOKENS:-5}"
ID_PREFIX="${ID_PREFIX:-mytest}"
URL="${URL:-http://localhost:7077/v1/chat/completions}"
MODEL="${MODEL:-deepseek}"
RESP_DIR="${RESP_DIR:-/tmp/kv_responses}"
DUMP_DIR="${DUMP_DIR:-/tmp/kv_dumps_step}"
ANALYZE="${ANALYZE:-0}"

ACTION="${1:-}"

case "$ACTION" in
send)
    MODE="${2:-}"
    if [[ -z "$MODE" ]]; then
        echo "Usage: $0 send <baseline|omnicache>"
        exit 1
    fi
    echo "=== Sending $N concurrent requests (mode=$MODE) ==="
    python3 tools/kv_dump/run_kv_verification.py send \
        -n "$N" --max-tokens "$MAX_TOKENS" --ignore-eos \
        --id-prefix "$ID_PREFIX" --mode "$MODE" \
        --url "$URL" --model "$MODEL" \
        --responses-dir "$RESP_DIR"
    echo "=== Done: responses saved to $RESP_DIR/$MODE ==="
    ;;
compare)
    EXTRA=()
    [[ "$ANALYZE" == "1" ]] && EXTRA+=(--analyze)
    python3 tools/kv_dump/run_kv_verification.py compare \
        --id-prefix "$ID_PREFIX" \
        --dump-dir "$DUMP_DIR" \
        --responses-dir "$RESP_DIR" \
        "${EXTRA[@]}"
    ;;
*)
    echo "Usage: $0 {send <baseline|omnicache>|compare}"
    echo ""
    echo "  send <mode>    Send concurrent requests (mode=baseline|omnicache)"
    echo "  compare        Compare KV dumps + logP across runs"
    echo ""
    echo "Env vars (defaults):"
    echo "  N=8                batch size"
    echo "  MAX_TOKENS=5       tokens per request"
    echo "  ID_PREFIX=mytest   deterministic request id prefix"
    echo "  URL=http://localhost:7077/v1/chat/completions"
    echo "  MODEL=deepseek"
    echo "  ANALYZE=0          set to 1 for HTML analysis report"
    exit 1
    ;;
esac
