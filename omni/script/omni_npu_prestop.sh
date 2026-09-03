#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# OMNI-DUMP preStop driver script.
#
# Purpose:
#   Drive the last-chance forensics collection when a pod/instance is being
#   terminated: collect node-level hardware state (npu-smi) once, send the
#   forensics signal (SIGUSR1) to every instrumented omni-npu process, then
#   wait briefly for their dump packages to land. It never blocks the
#   termination flow (always exits 0).
#
# Usage:
#   - Deployed inside the omni_npu wheel (omni_npu/script/); mount it as the
#     Kubernetes preStop hook of the serving container, e.g.
#       lifecycle:
#         preStop:
#           exec: { command: ["/bin/sh", "-c", ".../omni_npu_prestop.sh"] }
#   - Can also be run manually on a node to trigger an on-demand dump of all
#     live instrumented processes: OMNI_DUMP_DIR=/var/log/omni-npu/dump ./omni_npu_prestop.sh
#   - Deployment inputs are OMNI_DUMP_DIR (dump root, defaults to
#     /var/log/omni-npu/dump) and PATH (must resolve npu-smi if hardware
#     collection is wanted). Head variables below are tuning knobs fixed at
#     release time, not deployment configuration.
#
# Upstream / downstream:
#   - Upstream (who calls this): Kubernetes preStop hook on pod termination,
#     or an operator/SRE running it by hand.
#   - Downstream (what this relies on / produces):
#     * Reads pidfiles under $OMNI_DUMP_DIR/pids/omni_npu_*.pid, written by
#       omni_npu.diagnostics.dump.exit_dump at process startup (api/engine/
#       worker roles), with proc_start recorded to guard against pid reuse.
#     * Sends SIGUSR1 to each registered process; the in-process handler
#       (installed by exit_dump) writes dump_<role>_<pid>_*.json into
#       $OMNI_DUMP_DIR, which this script waits for (bounded by WAIT_SEC).
#     * Invokes npu-smi (bounded by NPU_SMI_TIMEOUT) and writes hw_<ts>.txt
#       into $OMNI_DUMP_DIR; dump packages reference it via hardware.node_ref.

NPU_SMI_TIMEOUT=3
WAIT_SEC=15

DUMP_DIR="${OMNI_DUMP_DIR:-/var/log/omni-npu/dump}"
PID_DIR="$DUMP_DIR/pids"

mkdir -p "$DUMP_DIR" 2>/dev/null

# --- 0. node-level hardware pre-collection (failure must not block) --------
collect_hw() {
    command -v npu-smi >/dev/null 2>&1 || return 0
    ts=$(date +%s)
    raw="$DUMP_DIR/.hw_$ts.raw"
    npu-smi info >"$raw" 2>&1 &
    smi_pid=$!
    (
        sleep "$NPU_SMI_TIMEOUT"
        kill "$smi_pid" 2>/dev/null
    ) &
    watcher=$!
    wait "$smi_pid" 2>/dev/null
    smi_rc=$?
    kill "$watcher" 2>/dev/null
    if [ "$smi_rc" -ne 0 ] || [ ! -s "$raw" ]; then
        rm -f "$raw"
        return 0
    fi
    # Raw npu-smi text is the most readable form; consumers reference it
    # by filename via the hardware.node_ref section of each dump package.
    mv "$raw" "$DUMP_DIR/hw_$ts.txt"
}
collect_hw

# --- helpers ----------------------------------------------------------------
proc_start_of() {
    # Start time (jiffies, field 22 of /proc/<pid>/stat); empty when unknown.
    stat=$(cat "/proc/$1/stat" 2>/dev/null) || {
        echo ""
        return 0
    }
    rest="${stat##*) }"
    # shellcheck disable=SC2086
    set -- $rest
    if [ "$#" -ge 20 ]; then
        echo "${20}"
    else
        echo ""
    fi
}

is_ours() {
    # $1 = pid, $2 = recorded proc_start. Guards against pid reuse.
    kill -0 "$1" 2>/dev/null || return 1
    [ -n "$2" ] || return 0
    current=$(proc_start_of "$1")
    [ -z "$current" ] && return 0
    [ "$current" = "$2" ]
}

# --- 1-3. enumerate pidfiles, validate, signal ------------------------------
REF="$DUMP_DIR/.prestop_ref_$$"
touch "$REF"
PIDS=""
for pf in "$PID_DIR"/omni_npu_*.pid; do
    [ -f "$pf" ] || continue
    pid=$(sed -n 's/^pid=//p' "$pf")
    recorded=$(sed -n 's/^proc_start=//p' "$pf")
    [ -n "$pid" ] || continue
    is_ours "$pid" "$recorded" || continue
    kill -s USR1 "$pid" 2>/dev/null || continue
    PIDS="$PIDS $pid"
done

# --- 4. wait for a dump package newer than the signal time ------------------
deadline=$(($(date +%s) + WAIT_SEC))
for pid in $PIDS; do
    while [ "$(date +%s)" -lt "$deadline" ]; do
        found=$(find "$DUMP_DIR" -maxdepth 1 -name "dump_*_${pid}_*.json" \
            -newer "$REF" 2>/dev/null | head -n 1)
        [ -n "$found" ] && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
done
rm -f "$REF"

exit 0
