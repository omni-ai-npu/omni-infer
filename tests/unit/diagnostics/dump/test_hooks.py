# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the vllm-free role hooks (worker / engine / api)."""
import pytest

from omni.diagnostics.dump import engine_stats, hooks

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_stats():
    engine_stats.reset()
    yield
    engine_stats.reset()


class FakeWorker:
    def __init__(self, local_rank=None, rank=None):
        if local_rank is not None:
            self.local_rank = local_rank
        if rank is not None:
            self.rank = rank


class TestEnableGate:
    def test_disabled_env_mounts_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "0")
        hooks.on_worker_init(FakeWorker(local_rank=0), dump_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_install_failure_never_raises(
            self, tmp_path, monkeypatch, caplog, capture_logger):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "1")
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        with capture_logger(hooks.logger):
            hooks.on_worker_init(FakeWorker(local_rank=0), dump_dir=blocker)
        assert any("install failed" in r.message for r in caplog.records)


class TestWorkerRank:
    def test_rank_from_worker_attribute(self):
        assert hooks.worker_rank(FakeWorker(local_rank=3)) == 3

    def test_rank_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "7")
        assert hooks.worker_rank(FakeWorker()) == 7

    def test_rank_unresolvable_is_none(self, monkeypatch):
        for env in ("LOCAL_RANK", "RANK", "RANK_ID"):
            monkeypatch.delenv(env, raising=False)
        assert hooks.worker_rank(FakeWorker()) is None


class FakeSchedulerOutput:
    total_num_scheduled_tokens = 8
    num_scheduled_tokens = {"r1": 8}
    scheduled_new_reqs = []


class TestEngineExecutorWrap:
    def test_sync_executor_ticks_dispatch_and_complete(self):
        class FakeExecutor:
            def execute_model(self, scheduler_output):
                return "output"

        executor = FakeExecutor()
        hooks.wrap_executor(executor)
        assert executor.execute_model(FakeSchedulerOutput()) == "output"
        snap = engine_stats.snapshot()
        assert snap["engine_step_count"] == 1
        assert snap["last_complete_ts"] >= snap["last_dispatch_ts"]

    def test_future_executor_completes_on_callback(self):
        callbacks = []

        class FakeFuture:
            def add_done_callback(self, fn):
                callbacks.append(fn)

        class FakeExecutor:
            def execute_model(self, scheduler_output):
                return FakeFuture()

        executor = FakeExecutor()
        hooks.wrap_executor(executor)
        executor.execute_model(FakeSchedulerOutput())

        snap = engine_stats.snapshot()
        assert snap["last_dispatch_ts"] is not None
        assert snap["engine_step_count"] == 0, "complete must wait for the future"

        callbacks[0](None)
        assert engine_stats.snapshot()["engine_step_count"] == 1


class TestEngineStatsFn:
    def test_merges_micro_tick_and_scheduler_state(self):
        class FakeKVManager:
            usage = 0.5

        class FakeScheduler:
            running = ["a"]
            waiting = []
            kv_cache_manager = FakeKVManager()

        engine_stats.on_dispatch(FakeSchedulerOutput())
        engine_stats.on_complete()
        stats = hooks.build_engine_stats_fn(FakeScheduler())()
        assert stats["engine_step_count"] == 1
        assert stats["scheduler"]["num_running"] == 1
        assert stats["scheduler"]["kv_cache_usage"] == 0.5


class TestApiStatsFn:
    def test_reads_in_flight_requests(self):
        class FakeOutputProcessor:
            request_states = {"a": 1, "b": 2}

        class FakeAsyncLLM:
            output_processor = FakeOutputProcessor()

        assert hooks.build_api_stats_fn(FakeAsyncLLM())() == {"in_flight_reqs": 2}

    def test_missing_attribute_degrades(self):
        stats = hooks.build_api_stats_fn(object())()
        assert str(stats["in_flight_reqs"]).startswith("ERROR:")
