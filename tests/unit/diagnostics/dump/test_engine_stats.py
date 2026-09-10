# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for OMNI-DUMP EngineCore runtime stats."""
import logging
import threading
import time

import pytest

from omni_npu.diagnostics.dump import engine_stats

pytestmark = pytest.mark.unit


class FakeSchedulerOutput:
    def __init__(self, tokens=0, num_scheduled=None, new_reqs=None):
        self.total_num_scheduled_tokens = tokens
        self.num_scheduled_tokens = num_scheduled if num_scheduled is not None else {}
        self.scheduled_new_reqs = new_reqs if new_reqs is not None else []


@pytest.fixture(autouse=True)
def _fresh_state():
    engine_stats.reset()
    yield
    engine_stats.reset()


class TestMicroTick:
    def test_dispatch_then_complete_updates_snapshot(self):
        engine_stats.on_dispatch(FakeSchedulerOutput(tokens=8, num_scheduled={"r1": 8}))
        engine_stats.on_complete()
        snap = engine_stats.snapshot()
        assert snap["last_dispatch_ts"] is not None
        assert snap["last_complete_ts"] >= snap["last_dispatch_ts"]
        assert snap["engine_step_count"] == 1
        assert len(snap["step_durations_recent"]) == 1
        assert snap["step_durations_recent"][0] >= 0.0

    def test_stuck_worker_signature(self):
        engine_stats.on_dispatch(FakeSchedulerOutput(tokens=8))
        engine_stats.on_complete()
        engine_stats.on_dispatch(FakeSchedulerOutput(tokens=8))  # never completes
        first = engine_stats.snapshot()
        time.sleep(0.02)
        second = engine_stats.snapshot()
        assert first["last_dispatch_ts"] > first["last_complete_ts"]
        assert second["since_last_complete_sec"] > first["since_last_complete_sec"]

    def test_tick_swallows_bad_scheduler_output(self):
        engine_stats.on_dispatch(object())  # lacks every expected attribute
        snap = engine_stats.snapshot()
        assert snap["last_dispatch_ts"] is not None

    def test_swallowed_failure_logs_warning(self, caplog, capture_logger):
        class ExplodingOutput:
            @property
            def total_num_scheduled_tokens(self):
                raise RuntimeError("renamed upstream")

        with capture_logger(engine_stats.logger):
            engine_stats.on_dispatch(ExplodingOutput())
        assert any(
            r.levelno == logging.WARNING and "on_dispatch" in r.message
            for r in caplog.records
        ), "swallowed hot-path failure must surface as WARNING"


class TestConcurrentSnapshot:
    def test_reader_never_breaks_while_writer_ticks(self):
        stop = threading.Event()
        errors = []

        def writer():
            while not stop.is_set():
                engine_stats.on_dispatch(FakeSchedulerOutput(tokens=4, num_scheduled={"r": 4}))
                engine_stats.on_complete()

        def reader():
            try:
                for _ in range(2000):
                    snap = engine_stats.snapshot()
                    assert isinstance(snap["engine_step_count"], int)
                    assert isinstance(snap["since_last_complete_sec"], float)
                    assert isinstance(snap["step_durations_recent"], list)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start()
        r.start()
        r.join()
        stop.set()
        w.join()
        assert errors == []

    def test_snapshot_returns_independent_copy(self):
        engine_stats.on_dispatch(FakeSchedulerOutput(tokens=4))
        snap = engine_stats.snapshot()
        snap["engine_step_count"] = 999
        snap["dispatched_batch"]["total_num_scheduled_tokens"] = 999
        fresh = engine_stats.snapshot()
        assert fresh["engine_step_count"] != 999
        assert fresh["dispatched_batch"]["total_num_scheduled_tokens"] != 999


class TestSchedulerDirectRead:
    def test_reads_live_scheduler_attributes(self):
        class FakeKVManager:
            usage = 0.42

        class FakeScheduler:
            running = ["a", "b"]
            waiting = ["c"]
            kv_cache_manager = FakeKVManager()

        state = engine_stats.collect_scheduler_state(FakeScheduler())
        assert state["num_running"] == 2
        assert state["num_waiting"] == 1
        assert state["kv_cache_usage"] == 0.42

    def test_missing_attributes_degrade_per_field(self):
        class BrokenKVManager:
            @property
            def usage(self):
                raise RuntimeError("renamed upstream")

        class PartialScheduler:  # no `waiting` attribute at all
            running = ["a"]
            kv_cache_manager = BrokenKVManager()

        state = engine_stats.collect_scheduler_state(PartialScheduler())
        assert state["num_running"] == 1
        assert str(state["num_waiting"]).startswith("ERROR:")
        assert str(state["kv_cache_usage"]).startswith("ERROR:")


class TestBatchSummary:
    def test_reports_raw_batch_facts(self):
        engine_stats.on_dispatch(
            FakeSchedulerOutput(tokens=100, num_scheduled={"r1": 96, "r2": 4}, new_reqs=["r1"])
        )
        batch = engine_stats.snapshot()["dispatched_batch"]
        assert batch["total_num_scheduled_tokens"] == 100
        assert batch["num_reqs"] == 2
        assert batch["new_reqs"] == 1

    def test_no_phase_inference(self):
        # A phase label cannot be derived reliably from SchedulerOutput
        # (chunked-prefill continuations, mixed batches, spec decode), so the
        # summary must stick to raw facts.
        engine_stats.on_dispatch(FakeSchedulerOutput(tokens=4, num_scheduled={"r1": 4}))
        assert "phase" not in engine_stats.snapshot()["dispatched_batch"]
