# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Collection thread: sleeps on select(), woken by signal bytes on the wakeup fd.

The wakeup fd is process-global (any Python-handled signal writes its number
there, including vLLM's own SIGTERM/SIGINT handlers), so the thread drains the
pipe completely and only reacts to the forensics signal.
"""
import logging
import os
import select
import threading

from omni_npu.diagnostics.dump import constants

logger = logging.getLogger(__name__)


class DumpThread:
    def __init__(self, wakeup_read_fd, collect_fn, signum=constants.DUMP_SIGNAL):
        self._wakeup_r = wakeup_read_fd
        self._collect_fn = collect_fn
        self._signum = int(signum)
        self._stop_r, self._stop_w = os.pipe()
        os.set_blocking(self._wakeup_r, False)
        os.set_blocking(self._stop_r, False)
        self._thread = threading.Thread(target=self._run, name="omni-dump", daemon=True)

    def start(self):
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def stop(self, timeout=2.0):
        try:
            os.write(self._stop_w, b"x")
        except OSError:
            pass
        self._thread.join(timeout)
        for fd in (self._stop_r, self._stop_w):
            try:
                os.close(fd)
            except OSError:
                pass

    def _run(self):
        while True:
            ready, _, _ = select.select([self._wakeup_r, self._stop_r], [], [])
            if self._stop_r in ready:
                return
            if not self._drain_has_dump_signal():
                continue
            try:
                self._collect_fn()
            except Exception:  # noqa: BLE001 - one failed collection must not kill the thread
                logger.exception("omni-dump: collection failed")

    def _drain_has_dump_signal(self):
        """Read the wakeup pipe until EAGAIN; report whether the dump signal was seen."""
        found = False
        while True:
            try:
                data = os.read(self._wakeup_r, 4096)
            except BlockingIOError:
                return found
            except OSError:
                return found
            if not data:
                return found
            if self._signum in data:
                found = True
