# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""In-process unit tests for exit_dump.install().

The sibling test_exit_dump.py drives install() through real child processes,
which validates end-to-end signal / atexit / crash behaviour but is invisible
to coverage (child processes are not measured). These tests wire the machinery
up *inside the test process* so every branch of install() - role selection,
both collection triggers, stats-error isolation, the already-installed guard and
the partial-install rollback - is exercised and measured directly.

Each test fully restores process-global signal / faulthandler / wakeup-fd state
via the autouse _isolate_process fixture, and the real atexit registration is
intercepted (captured, never registered) so nothing fires at interpreter exit.
"""
import faulthandler
import json
import os
import signal
import time

import pytest

from omni.diagnostics.dump import constants, exit_dump

pytestmark = pytest.mark.unit


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _teardown_install(inst):
    """Stop the threads and drop the process-global forensics wiring."""
    try:
        faulthandler.unregister(constants.DUMP_SIGNAL)
    except Exception:  # noqa: BLE001 - best effort teardown
        pass
    faulthandler.disable()
    try:
        signal.set_wakeup_fd(-1)
    except (ValueError, OSError):
        pass
    if inst is not None:
        if inst.collector is not None:
            inst.collector.stop()
        if inst.flusher is not None:
            inst.flusher.stop()
    exit_dump._installed = None


@pytest.fixture(autouse=True)
def _isolate_process():
    """Snapshot and restore SIGUSR1 disposition, wakeup fd and faulthandler."""
    orig_handler = signal.getsignal(constants.DUMP_SIGNAL)
    exit_dump._installed = None
    yield
    exit_dump._installed = None
    try:
        signal.set_wakeup_fd(-1)
    except (ValueError, OSError):
        pass
    try:
        faulthandler.unregister(constants.DUMP_SIGNAL)
    except Exception:  # noqa: BLE001
        pass
    faulthandler.disable()
    # restore pytest's own faulthandler (stderr) for later tests
    try:
        faulthandler.enable()
    except Exception:  # noqa: BLE001
        pass
    signal.signal(constants.DUMP_SIGNAL, orig_handler)


@pytest.fixture
def atexit_hooks(monkeypatch):
    """Capture atexit registrations instead of really registering them."""
    hooks = []

    def _capture(fn):
        hooks.append(fn)
        return fn

    monkeypatch.setattr(exit_dump.atexit, "register", _capture)
    return hooks


def _dump_jsons(dump_dir):
    return sorted(dump_dir.glob(f"dump_*_{os.getpid()}_*.json"))


class TestInstallWorker:
    def test_wires_threads_pidfile_and_stack_file(self, tmp_path, atexit_hooks):
        inst = exit_dump.install(role="worker", dump_dir=tmp_path, rank=0)
        try:
            assert exit_dump._installed is inst
            assert inst.role == "worker"
            assert inst.rank == 0
            assert inst.collector.is_alive()
            assert inst.flusher is None, "workers carry no stats flusher"

            pidfiles = list((tmp_path / "pids").glob("omni_npu_worker_*.pid"))
            assert len(pidfiles) == 1
            content = pidfiles[0].read_text()
            assert f"pid={os.getpid()}" in content
            assert "role=worker" in content
            assert "rank=0" in content

            stack_file = tmp_path / "raw" / f"dump_{os.getpid()}_stack.txt"
            assert stack_file.exists()
            assert len(atexit_hooks) == 1
        finally:
            _teardown_install(inst)

    def test_atexit_hook_writes_package_and_removes_pidfile(
            self, tmp_path, atexit_hooks):
        inst = exit_dump.install(role="worker", dump_dir=tmp_path, rank=4)
        try:
            atexit_hooks[0]()  # simulate interpreter shutdown
            jsons = _dump_jsons(tmp_path)
            assert len(jsons) == 1
            doc = json.loads(jsons[0].read_text())
            assert doc["trigger"] == "atexit"
            assert doc["role"] == "worker"
            assert doc["rank"] == 4
            # atexit dumps skip the stack (frames already unwound) and the
            # device probe (unreliable during teardown).
            assert doc["stack"] == constants.STACK_SKIPPED
            assert doc["stats"] is None
            assert doc["hardware"]["device_mem"] == constants.SKIPPED
            assert "proc_status" in doc["hardware"]
            # the pidfile is unlinked by the atexit hook
            assert list((tmp_path / "pids").glob("*.pid")) == []
        finally:
            _teardown_install(inst)

    def test_real_signal_triggers_collection(self, tmp_path, atexit_hooks):
        # Exercises the full wakeup path: _noop handler + set_wakeup_fd byte +
        # DumpThread drain + signal-trigger collection, without a subprocess.
        inst = exit_dump.install(role="worker", dump_dir=tmp_path, rank=2)
        try:
            os.kill(os.getpid(), constants.DUMP_SIGNAL)
            assert wait_until(lambda: len(_dump_jsons(tmp_path)) == 1)
            doc = json.loads(_dump_jsons(tmp_path)[0].read_text())
            assert doc["trigger"] == "signal"
            assert doc["role"] == "worker"
            assert doc["rank"] == 2
        finally:
            _teardown_install(inst)


class TestInstallEngine:
    def test_engine_starts_flusher_and_signal_collect_carries_stats(
            self, tmp_path, atexit_hooks):
        def stats_fn():
            return {"engine_step_count": 5}

        inst = exit_dump.install(
            role="engine", dump_dir=tmp_path, stats_fn=stats_fn)
        try:
            assert inst.flusher is not None
            assert inst.flusher.is_alive()
            # invoke the collector's signal-trigger closure directly so the
            # signal collection path is measured deterministically.
            inst.collector._collect_fn()
            jsons = _dump_jsons(tmp_path)
            assert len(jsons) == 1
            doc = json.loads(jsons[0].read_text())
            assert doc["trigger"] == "signal"
            assert doc["role"] == "engine"
            assert doc["stats"] == {"engine_step_count": 5}
            assert "device_mem" in doc["hardware"]
        finally:
            _teardown_install(inst)

    def test_stats_fn_failure_is_isolated_into_the_dump(
            self, tmp_path, atexit_hooks):
        def boom():
            raise RuntimeError("stats boom")

        inst = exit_dump.install(role="engine", dump_dir=tmp_path, stats_fn=boom)
        try:
            inst.collector._collect_fn()
            doc = json.loads(_dump_jsons(tmp_path)[0].read_text())
            assert str(doc["stats"]["error"]).startswith("RuntimeError")
        finally:
            _teardown_install(inst)

    def test_engine_without_stats_fn_has_no_flusher(self, tmp_path, atexit_hooks):
        inst = exit_dump.install(role="engine", dump_dir=tmp_path, stats_fn=None)
        try:
            assert inst.flusher is None
        finally:
            _teardown_install(inst)


class TestInstallApi:
    def test_api_role_installs_under_running_loop(self, tmp_path, atexit_hooks):
        import asyncio

        async def amain():
            inst = exit_dump.install(
                role="api", dump_dir=tmp_path,
                stats_fn=lambda: {"in_flight_reqs": 2})
            try:
                assert inst.role == "api"
                assert inst.collector.is_alive()
                assert inst.flusher is None, "flusher is engine-only"
                inst.collector._collect_fn()
                jsons = _dump_jsons(tmp_path)
                assert len(jsons) == 1
                doc = json.loads(jsons[0].read_text())
                assert doc["role"] == "api"
                assert doc["stats"] == {"in_flight_reqs": 2}
            finally:
                loop = asyncio.get_running_loop()
                try:
                    loop.remove_signal_handler(constants.DUMP_SIGNAL)
                except (ValueError, NotImplementedError, RuntimeError):
                    pass
                _teardown_install(inst)

        asyncio.run(amain())


class TestInstallGuards:
    def test_second_install_is_ignored(self, tmp_path, caplog, capture_logger):
        sentinel = object()
        exit_dump._installed = sentinel
        try:
            with capture_logger(exit_dump.logger):
                got = exit_dump.install(role="worker", dump_dir=tmp_path)
            assert got is sentinel
            assert any("already installed" in r.message for r in caplog.records)
            # nothing new was wired: no pidfile dir populated
            assert not list(tmp_path.glob("**/*.pid"))
        finally:
            exit_dump._installed = None

    def test_partial_install_failure_rolls_back(
            self, tmp_path, atexit_hooks, monkeypatch):
        # Force a failure AFTER the wakeup fd is armed and the collector thread
        # is started (flusher construction is the next step for an engine with a
        # stats_fn). The rollback must unwind faulthandler, the wakeup fd and the
        # collector, and leave the module un-installed.
        class BoomFlusher:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("flusher init boom")

        monkeypatch.setattr(exit_dump.stats_flush, "StatsFlushThread", BoomFlusher)

        with pytest.raises(RuntimeError, match="flusher init boom"):
            exit_dump.install(
                role="engine", dump_dir=tmp_path, stats_fn=lambda: {"x": 1})

        assert exit_dump._installed is None, "a failed install must not register"
        # wakeup fd was reset by the rollback
        assert signal.set_wakeup_fd(-1) == -1

    def test_api_install_failure_removes_loop_signal_handler(
            self, tmp_path, atexit_hooks, monkeypatch):
        # The api branch registers an asyncio signal handler before building the
        # collector; a failure there must exercise the api-specific rollback
        # (loop.remove_signal_handler) rather than the set_wakeup_fd path.
        import asyncio

        class BoomThread:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("collector init boom")

        monkeypatch.setattr(exit_dump.dump_thread, "DumpThread", BoomThread)

        async def amain():
            with pytest.raises(RuntimeError, match="collector init boom"):
                exit_dump.install(
                    role="api", dump_dir=tmp_path, stats_fn=lambda: {})
            assert exit_dump._installed is None
            loop = asyncio.get_running_loop()
            # the rollback already removed our handler, so a second removal is a
            # no-op (returns False) - proof the api rollback branch ran.
            assert loop.remove_signal_handler(constants.DUMP_SIGNAL) is False

        asyncio.run(amain())
