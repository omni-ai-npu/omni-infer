# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the OMNI-DUMP periodic stats snapshot thread."""
import json
import os
import time

import pytest

from omni.diagnostics.dump import stats_flush

pytestmark = pytest.mark.unit


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class TestPeriodicFlush:
    def test_snapshot_is_written_periodically_and_atomically(self, tmp_path):
        counter = {"n": 0}

        def snapshot_fn():
            counter["n"] += 1
            return {"engine_step_count": counter["n"]}

        thread = stats_flush.StatsFlushThread(tmp_path, snapshot_fn, interval_sec=0.02)
        thread.start()
        try:
            path = stats_flush.stats_file_path(tmp_path)
            assert wait_until(lambda: path.exists())
            assert path.name == f"dump_engine_{os.getpid()}_stats.json"
            assert path.parent.name == "raw"

            first = json.loads(path.read_text(encoding="utf-8"))
            assert wait_until(
                lambda: json.loads(path.read_text(encoding="utf-8")) != first
            ), "the snapshot file must be overwritten on the next period"
            assert json.loads(path.read_text(encoding="utf-8"))["engine_step_count"] > 0
        finally:
            thread.stop()

    def test_snapshot_failure_keeps_thread_alive(self, tmp_path):
        state = {"raised": False}

        def snapshot_fn():
            if not state["raised"]:
                state["raised"] = True
                raise RuntimeError("injected snapshot failure")
            return {"ok": True}

        thread = stats_flush.StatsFlushThread(tmp_path, snapshot_fn, interval_sec=0.02)
        thread.start()
        try:
            path = stats_flush.stats_file_path(tmp_path)
            assert wait_until(lambda: path.exists())
        finally:
            thread.stop()


class TestStop:
    def test_stop_joins_and_writes_no_more(self, tmp_path):
        thread = stats_flush.StatsFlushThread(tmp_path, lambda: {"x": 1}, interval_sec=0.02)
        thread.start()
        path = stats_flush.stats_file_path(tmp_path)
        assert wait_until(lambda: path.exists())

        thread.stop(timeout=2.0)
        assert not thread.is_alive()
        mtime = path.stat().st_mtime_ns
        time.sleep(0.08)
        assert path.stat().st_mtime_ns == mtime
