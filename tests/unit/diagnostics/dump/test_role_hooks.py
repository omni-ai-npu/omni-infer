# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the role-hook entry points and hooks plumbing.

test_hooks.py covers the stats builders (build_stats_fn / wrap_executor /
worker_rank). This module covers the on_init entry points of all three roles and
the shared hooks helpers (enable gate, default_dump_dir, guarded_install),
without performing a real forensics install: exit_dump.install / guarded_install
are replaced by recorders so only the hook wiring is under test.
"""
import pytest

from omni_npu.diagnostics.dump import constants, engine_stats, exit_dump, hooks

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_stats():
    engine_stats.reset()
    yield
    engine_stats.reset()


@pytest.fixture
def record_guarded_install(monkeypatch):
    """Capture hooks.guarded_install calls made by the role hooks."""
    calls = []
    monkeypatch.setattr(
        hooks, "guarded_install",
        lambda **kwargs: calls.append(kwargs))
    return calls


# --------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------

class TestHooksCommon:
    def test_enabled_default_true(self, monkeypatch):
        monkeypatch.delenv("OMNI_DUMP_ENABLE", raising=False)
        assert hooks.enabled() is True

    def test_enabled_respects_disable(self, monkeypatch):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "0")
        assert hooks.enabled() is False

    def test_default_dump_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("OMNI_DUMP_DIR", "/custom/dump/path")
        assert hooks.default_dump_dir() == "/custom/dump/path"

    def test_default_dump_dir_fallback(self, monkeypatch):
        monkeypatch.delenv("OMNI_DUMP_DIR", raising=False)
        assert hooks.default_dump_dir() == "/var/log/omni-npu/dump"

    def test_guarded_install_forwards_and_defaults_dir(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            exit_dump, "install",
            lambda **kwargs: captured.update(kwargs) or "installed")
        monkeypatch.delenv("OMNI_DUMP_DIR", raising=False)

        def stats_fn():
            return {}

        hooks.guarded_install(
            role=constants.ROLE_ENGINE, dump_dir=None, rank=3, stats_fn=stats_fn)
        assert captured["role"] == constants.ROLE_ENGINE
        # dump_dir=None falls back to the default via default_dump_dir()
        assert captured["dump_dir"] == "/var/log/omni-npu/dump"
        assert captured["rank"] == 3
        assert captured["stats_fn"] is stats_fn

    def test_guarded_install_uses_explicit_dir(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(
            exit_dump, "install", lambda **kwargs: captured.update(kwargs))
        hooks.guarded_install(role=constants.ROLE_WORKER, dump_dir=tmp_path)
        assert captured["dump_dir"] == tmp_path

    def test_guarded_install_swallows_failure(self, monkeypatch, caplog, capture_logger):
        def boom(**kwargs):
            raise RuntimeError("install exploded")

        monkeypatch.setattr(exit_dump, "install", boom)
        # must not raise
        with capture_logger(hooks.logger):
            hooks.guarded_install(role=constants.ROLE_WORKER, dump_dir="/x")
        assert any("install failed" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# worker role
# --------------------------------------------------------------------------

class FakeWorker:
    def __init__(self, local_rank=None, rank=None):
        if local_rank is not None:
            self.local_rank = local_rank
        if rank is not None:
            self.rank = rank


class TestWorkerOnInit:
    def test_installs_with_worker_rank(
            self, tmp_path, monkeypatch, record_guarded_install):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "1")
        hooks.on_worker_init(FakeWorker(local_rank=5), dump_dir=tmp_path)
        assert len(record_guarded_install) == 1
        call = record_guarded_install[0]
        assert call["role"] == constants.ROLE_WORKER
        assert call["dump_dir"] == tmp_path
        assert call["rank"] == 5

    def test_disabled_does_not_install(
            self, tmp_path, monkeypatch, record_guarded_install):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "0")
        hooks.on_worker_init(FakeWorker(local_rank=0), dump_dir=tmp_path)
        assert record_guarded_install == []

    def test_worker_rank_non_numeric_env_returned_verbatim(self, monkeypatch):
        for env in ("LOCAL_RANK", "RANK", "RANK_ID"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("RANK", "not-a-number")
        assert hooks.worker_rank(FakeWorker()) == "not-a-number"


# --------------------------------------------------------------------------
# api role
# --------------------------------------------------------------------------

class FakeAsyncLLM:
    class _OutputProcessor:
        request_states = {"a": 1, "b": 2}

    output_processor = _OutputProcessor()


class TestApiOnInit:
    def test_installs_api_role_with_working_stats_fn(
            self, tmp_path, monkeypatch, record_guarded_install):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "1")
        hooks.on_api_init(FakeAsyncLLM(), dump_dir=tmp_path)
        assert len(record_guarded_install) == 1
        call = record_guarded_install[0]
        assert call["role"] == constants.ROLE_API
        assert call["dump_dir"] == tmp_path
        assert call["stats_fn"]() == {"in_flight_reqs": 2}

    def test_disabled_does_not_install(
            self, tmp_path, monkeypatch, record_guarded_install):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "0")
        hooks.on_api_init(FakeAsyncLLM(), dump_dir=tmp_path)
        assert record_guarded_install == []


# --------------------------------------------------------------------------
# engine role
# --------------------------------------------------------------------------

class FakeSchedulerOutput:
    total_num_scheduled_tokens = 8
    num_scheduled_tokens = {"r1": 8}
    scheduled_new_reqs = []


class FakeExecutor:
    def execute_model(self, scheduler_output):
        return "output"


class FakeScheduler:
    running = ["a"]
    waiting = []

    class _KV:
        usage = 0.5

    kv_cache_manager = _KV()


class FakeEngineCore:
    def __init__(self, executor=None, scheduler=None):
        self.model_executor = executor
        self.scheduler = scheduler


class TestEngineOnInit:
    def test_wraps_executor_and_installs_with_merged_stats(
            self, tmp_path, monkeypatch, record_guarded_install):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "1")
        executor = FakeExecutor()
        core = FakeEngineCore(executor=executor, scheduler=FakeScheduler())
        hooks.on_engine_init(core, dump_dir=tmp_path)

        # execute_model was replaced by the ticking wrapper
        assert executor.execute_model.__name__ == "wrapped"
        assert executor.execute_model(FakeSchedulerOutput()) == "output"
        assert engine_stats.snapshot()["engine_step_count"] == 1

        assert len(record_guarded_install) == 1
        call = record_guarded_install[0]
        assert call["role"] == constants.ROLE_ENGINE
        stats = call["stats_fn"]()
        assert stats["scheduler"]["num_running"] == 1
        assert stats["scheduler"]["kv_cache_usage"] == 0.5

    def test_no_executor_still_installs(
            self, tmp_path, monkeypatch, record_guarded_install):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "1")
        core = FakeEngineCore(executor=None, scheduler=FakeScheduler())
        hooks.on_engine_init(core, dump_dir=tmp_path)
        assert len(record_guarded_install) == 1
        assert record_guarded_install[0]["role"] == constants.ROLE_ENGINE

    def test_executor_wrap_failure_degrades_to_warning_but_installs(
            self, tmp_path, monkeypatch, record_guarded_install, caplog, capture_logger):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "1")

        class BadExecutor:
            @property
            def execute_model(self):
                raise RuntimeError("cannot read execute_model")

        core = FakeEngineCore(executor=BadExecutor(), scheduler=None)
        with capture_logger(hooks.logger):
            hooks.on_engine_init(core, dump_dir=tmp_path)
        assert any("executor wrap failed" in r.message for r in caplog.records)
        # install still proceeds despite the wrap failure
        assert len(record_guarded_install) == 1
        assert record_guarded_install[0]["role"] == constants.ROLE_ENGINE

    def test_disabled_does_not_install(
            self, tmp_path, monkeypatch, record_guarded_install):
        monkeypatch.setenv("OMNI_DUMP_ENABLE", "0")
        core = FakeEngineCore(executor=FakeExecutor(), scheduler=FakeScheduler())
        hooks.on_engine_init(core, dump_dir=tmp_path)
        assert record_guarded_install == []
