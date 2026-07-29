# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Periodic stats snapshot thread (EngineCore role only).

The overwrite-style snapshot file is the only artifact that survives SIGKILL,
so it answers "how far had inference progressed" after a hard kill.
"""
import json
import logging
import os
import threading
from pathlib import Path

from omni_npu.diagnostics.dump import constants, forensic

logger = logging.getLogger(__name__)


def stats_file_path(dump_dir, pid=None):
    pid = os.getpid() if pid is None else pid
    return Path(dump_dir) / constants.RAW_SUBDIR / f"dump_engine_{pid}_stats.json"


class StatsFlushThread:
    def __init__(self, dump_dir, snapshot_fn, interval_sec=constants.STATS_FLUSH_INTERVAL_SEC):
        self._path = stats_file_path(dump_dir)
        self._snapshot_fn = snapshot_fn
        self._interval = interval_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="omni-dump-flush", daemon=True)

    def start(self):
        Path(self._path.parent).mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def stop(self, timeout=2.0):
        self._stop.set()
        self._thread.join(timeout)

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                snapshot = self._snapshot_fn()
                forensic.atomic_write_text(
                    self._path, json.dumps(snapshot, ensure_ascii=False, indent=2)
                )
            except Exception as e:  # noqa: BLE001 - one failed flush must not kill the thread
                logger.warning("omni-dump: stats flush failed: %r", e)
