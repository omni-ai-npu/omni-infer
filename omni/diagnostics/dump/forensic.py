# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exit-forensics primitives: atomic write, dump packaging, once-guard, stale cleanup.

stdlib-only. This module must never spawn subprocesses.
"""
import datetime
import json
import logging
import os
import threading
import time
from pathlib import Path

from omni_npu.diagnostics.dump import constants

logger = logging.getLogger(__name__)


def atomic_write_text(path, content):
    """Write via tmp file + fsync + os.replace so readers never see a partial file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return path


def _dump_prefix(role, rank):
    if role == constants.ROLE_WORKER:
        return f"rank{rank}"
    return role


def stack_file_path(dump_dir, pid=None):
    pid = os.getpid() if pid is None else pid
    return Path(dump_dir) / constants.RAW_SUBDIR / f"dump_{pid}_stack.txt"


def merge_dump(
    dump_dir,
    *,
    role,
    trigger,
    rank=None,
    stack_text=None,
    stats=None,
    hardware=None,
):
    """Write a dump package as a single human-readable JSON file and return its path.

    atexit-triggered dumps mark the stack section as "skipped": business frames are
    already unwound at that point, so a stack would carry no diagnostic value.
    """
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    ts = time.time()

    if trigger == constants.TRIGGER_ATEXIT:
        stack = constants.STACK_SKIPPED
    else:
        # A list of lines renders one frame per line under indent=2,
        # keeping the most-read section human-readable.
        stack = (stack_text or "").splitlines()

    doc = {
        "schema": constants.DUMP_SCHEMA,
        "trigger": trigger,
        "role": role,
        "rank": rank,
        "pid": pid,
        "ts": ts,
        # UTC so timestamps correlate unambiguously across nodes and timezones.
        "ts_iso": datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat(),
        "stack": stack,
        "stats": stats,
        "hardware": hardware,
    }

    json_path = dump_dir / f"dump_{_dump_prefix(role, rank)}_{pid}_{int(ts)}.json"
    atomic_write_text(json_path, json.dumps(doc, ensure_ascii=False, indent=2))
    return json_path


class OnceGuard:
    """Run the full collection exactly once per process exit; late triggers are skipped."""

    def __init__(self):
        self._lock = threading.Lock()
        self._done = False

    def enter(self, trigger, collect_fn):
        """Return True and run collect_fn on the first entry; later entries return False.

        The flag is set before running the collection so that a hung
        collection never blocks the decision for other triggers: they skip
        immediately instead of queueing on the lock.
        """
        with self._lock:
            if self._done:
                logger.info("omni-dump: full dump already taken, skip trigger=%s", trigger)
                return False
            self._done = True
        collect_fn()
        return True


def cleanup_stale(dump_dir, pid):
    """Remove artifacts left by a previous process that had the same pid.

    Only this pid's dump packages, stats snapshot and stack file are removed;
    other pids' artifacts and hw_* snapshots are kept.
    """
    dump_dir = Path(dump_dir)
    if not dump_dir.is_dir():
        return
    patterns = (
        f"dump_*_{pid}_*.json",
        f"{constants.RAW_SUBDIR}/dump_*_{pid}_stats.json",
        f"{constants.RAW_SUBDIR}/dump_{pid}_stack.txt",
    )
    for pattern in patterns:
        for f in dump_dir.glob(pattern):
            try:
                f.unlink()
            except OSError as e:
                logger.warning("omni-dump: cleanup of %s failed: %r", f, e)
