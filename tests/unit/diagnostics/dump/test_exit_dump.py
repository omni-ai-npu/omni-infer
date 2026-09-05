# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Integration tests for exit_dump.install, run against real child processes."""
import json
import os
import signal
import time

import pytest

from tests.unit.diagnostics.dump.procutil import Child, dump_jsons, wait_until

pytestmark = pytest.mark.unit


@pytest.fixture
def child_factory(tmp_path):
    children = []

    def factory(role, scenario):
        c = Child(role, tmp_path, scenario, tmp_path)
        children.append(c)
        return c

    yield factory
    for c in children:
        c.kill_if_alive()


class TestSignalCollection:
    def test_worker_collects_and_keeps_running(self, tmp_path, child_factory):
        child = child_factory("worker", "block")
        os.kill(child.pid, signal.SIGUSR1)
        assert wait_until(lambda: len(dump_jsons(tmp_path, child.pid)) == 1)

        doc = json.loads(dump_jsons(tmp_path, child.pid)[0].read_text())
        assert doc["trigger"] == "signal"
        assert doc["role"] == "worker"
        assert doc["rank"] == 0
        assert "most recent call first" in "\n".join(doc["stack"])
        assert doc["stats"] is None
        assert "proc_status" in doc["hardware"]
        assert child.proc.poll() is None, "the dump signal must not terminate the process"

        child.release()
        assert child.proc.wait(timeout=5) == 0

    def test_engine_collects_with_stats(self, tmp_path, child_factory):
        child = child_factory("engine", "block")
        os.kill(child.pid, signal.SIGUSR1)
        assert wait_until(lambda: len(dump_jsons(tmp_path, child.pid)) == 1)
        doc = json.loads(dump_jsons(tmp_path, child.pid)[0].read_text())
        assert doc["role"] == "engine"
        assert doc["stats"] == {"engine_step_count": 5}
        child.release()

    def test_api_collects_via_event_loop(self, tmp_path, child_factory):
        child = child_factory("api", "api")
        os.kill(child.pid, signal.SIGUSR1)
        assert wait_until(lambda: len(dump_jsons(tmp_path, child.pid)) == 1)
        doc = json.loads(dump_jsons(tmp_path, child.pid)[0].read_text())
        assert doc["role"] == "api"
        assert doc["stats"] == {"in_flight_reqs": 2}
        assert "most recent call first" in "\n".join(doc["stack"])
        assert child.proc.poll() is None
        child.release()


class TestSigtermUntouched:
    def test_handler_unchanged_and_default_termination(self, tmp_path, child_factory):
        child = child_factory("worker", "getsignal")
        assert "HANDLER_UNCHANGED=True" in child.lines

        os.kill(child.pid, signal.SIGTERM)
        rc = child.proc.wait(timeout=5)
        assert rc == -signal.SIGTERM, "default SIGTERM disposition must stay in force"
        assert dump_jsons(tmp_path, child.pid) == []


class TestAtexit:
    def test_normal_exit_writes_atexit_package(self, tmp_path, child_factory):
        child = child_factory("worker", "exit0")
        assert child.proc.wait(timeout=5) == 0
        files = dump_jsons(tmp_path, child.pid)
        assert len(files) == 1
        doc = json.loads(files[0].read_text())
        assert doc["trigger"] == "atexit"
        assert doc["stack"] == "skipped"
        assert doc["hardware"]["device_mem"] == "skipped", (
            "the device probe is unreliable during teardown and must be skipped"
        )
        assert "proc_status" in doc["hardware"]

    def test_repeated_signals_then_exit_collect_once(self, tmp_path, child_factory):
        child = child_factory("worker", "block")
        for _ in range(3):
            os.kill(child.pid, signal.SIGUSR1)
            time.sleep(0.05)
        assert wait_until(
            lambda: len(dump_jsons(tmp_path, child.pid)) == 1, timeout=15.0
        )
        child.release()
        assert child.proc.wait(timeout=5) == 0
        files = dump_jsons(tmp_path, child.pid)
        assert len(files) == 1, "one exit must yield exactly one full dump"
        assert json.loads(files[0].read_text())["trigger"] == "signal"


class TestPidfile:
    def test_pidfile_lifecycle(self, tmp_path, child_factory):
        child = child_factory("worker", "block")
        pidfiles = list((tmp_path / "pids").glob("omni_npu_*.pid"))
        assert len(pidfiles) == 1
        content = pidfiles[0].read_text()
        assert f"pid={child.pid}" in content
        assert "role=worker" in content
        assert "rank=0" in content
        assert "proc_start=" in content

        child.release()
        assert child.proc.wait(timeout=5) == 0
        assert list((tmp_path / "pids").glob("omni_npu_*.pid")) == []


class TestCrash:
    def test_segv_leaves_faulthandler_stack(self, tmp_path, child_factory):
        child = child_factory("worker", "segv")
        child.release()
        rc = child.proc.wait(timeout=5)
        assert rc != 0
        stack_file = tmp_path / "raw" / f"dump_{child.pid}_stack.txt"
        assert stack_file.exists()
        assert "Segmentation fault" in stack_file.read_text(errors="replace")
