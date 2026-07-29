# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Process-level hardware probe: device memory view and /proc/self/status.

Node-level hardware (npu-smi) is collected by the prestop script only; this
module is read-only and must never spawn a subprocess.
"""
import threading
import time
from pathlib import Path

from omni_npu.diagnostics.dump import constants

_PROC_STATUS_KEYS = ("VmRSS", "VmHWM", "Threads", "FDSize")
_DEFAULT_PROC_STATUS = "/proc/self/status"


def _import_torch():
    import torch  # noqa: PLC0415 - lazy: NPU-free hosts have no torch

    return torch


def _collect_device_mem():
    """Probe device memory in a disposable timed thread.

    A corrupted NPU runtime can make torch import or mem_get_info hang
    forever (not just raise), which would wedge the whole dump and the
    OnceGuard behind it; on timeout the thread is abandoned.
    """
    result = {}

    def probe():
        try:
            torch = _import_torch()
            free, total = torch.npu.mem_get_info()
            result["value"] = {"free_bytes": int(free), "total_bytes": int(total)}
        except Exception as e:  # noqa: BLE001
            result["value"] = f"ERROR:{type(e).__name__}:{e}"

    worker = threading.Thread(target=probe, daemon=True, name="omni-dump-hw-probe")
    worker.start()
    worker.join(constants.DEVICE_MEM_PROBE_TIMEOUT_SEC)
    return result.get("value", "ERROR:timeout:device memory probe did not return")


def _collect_proc_status(path):
    try:
        result = {}
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition(":")
            if key in _PROC_STATUS_KEYS:
                result[key] = value.strip()
        return result
    except Exception as e:  # noqa: BLE001
        return f"ERROR:{type(e).__name__}:{e}"


def collect_process_hw(proc_status_path=_DEFAULT_PROC_STATUS, include_device_mem=True):
    """Collect the process-level hardware section; sub-probes fail independently.

    The device probe is skipped on teardown paths (atexit): the device state
    is untrustworthy there and a hung probe would stall process exit.
    """
    if include_device_mem:
        device_mem = _collect_device_mem()
    else:
        device_mem = constants.SKIPPED
    return {
        "device_mem": device_mem,
        "proc_status": _collect_proc_status(proc_status_path),
    }


def find_hw_reference(dump_dir):
    """Best-effort pointer to the newest node-level hw_* file, or None."""
    try:
        files = [f for f in Path(dump_dir).glob("hw_*") if f.is_file()]
        if not files:
            return None
        newest = max(files, key=lambda f: f.stat().st_mtime)
        return {
            "hardware_ref": newest.name,
            "age_sec": round(time.time() - newest.stat().st_mtime, 1),
        }
    except OSError:
        return None
