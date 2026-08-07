# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for the prestop driver script."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.unit.diagnostics.dump.procutil import Child, dump_jsons, wait_until

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).parents[4] / "omni" / "script" / "omni_npu_prestop.sh"


@pytest.fixture
def fast_script(tmp_path):
    """Copy of the script with test-friendly head variables."""
    text = SCRIPT.read_text()
    # date +%s truncates to whole seconds, so the effective wait lies in
    # [WAIT_SEC-1, WAIT_SEC]; keep assertions below that lower bound.
    text = text.replace("WAIT_SEC=15", "WAIT_SEC=3")
    # 2s keeps headroom for stub startup under full-suite load while still
    # killing the 5s hanging stub well before the test deadline.
    text = text.replace("NPU_SMI_TIMEOUT=3", "NPU_SMI_TIMEOUT=2")
    copy = tmp_path / "prestop.sh"
    copy.write_text(text)
    copy.chmod(0o755)
    return copy


@pytest.fixture
def child_factory(tmp_path):
    children = []

    def factory(role, scenario="block"):
        c = Child(role, tmp_path, scenario, tmp_path)
        children.append(c)
        return c

    yield factory
    for c in children:
        c.kill_if_alive()


def _path_without_npu_smi():
    """Inherited PATH minus any directory holding a real npu-smi.

    The script's only deployment inputs are OMNI_DUMP_DIR and PATH, and it
    finds npu-smi via `command -v`. On NPU machines the driver's real npu-smi
    would otherwise leak into every test: the "missing tool" case would find
    it, and its ~2s runtime would break the no-wait timing assertions. Tests
    that want an npu-smi prepend a stub explicitly.
    """
    kept = [
        d
        for d in os.environ.get("PATH", "").split(":")
        if not (d and os.access(os.path.join(d, "npu-smi"), os.X_OK))
    ]
    return ":".join(kept)


def run_script(script, dump_dir, path_prepend=None):
    env = dict(os.environ, OMNI_DUMP_DIR=str(dump_dir), PATH=_path_without_npu_smi())
    if path_prepend is not None:
        env["PATH"] = f"{path_prepend}:{env['PATH']}"
    start = time.monotonic()
    proc = subprocess.run(
        ["/bin/sh", str(script)], env=env, capture_output=True, text=True, timeout=30
    )
    return proc, time.monotonic() - start


def npu_smi_stub(tmp_path, body):
    stub_dir = tmp_path / "stub-bin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "npu-smi"
    stub.write_text(f"#!/bin/sh\n{body}\n")
    stub.chmod(0o755)
    return stub_dir


def write_pidfile(dump_dir, pid, proc_start=""):
    pids = Path(dump_dir) / "pids"
    pids.mkdir(parents=True, exist_ok=True)
    (pids / f"omni_npu_worker_{pid}.pid").write_text(
        f"pid={pid}\nrole=worker\nrank=0\nproc_start={proc_start}\n"
    )


class TestFullChain:
    def test_hw_signal_and_wait(self, tmp_path, fast_script, child_factory):
        worker = child_factory("worker")
        engine = child_factory("engine")
        stub = npu_smi_stub(tmp_path, 'echo "NPU 0 OK temp 45C"')

        proc, elapsed = run_script(fast_script, tmp_path, path_prepend=stub)

        assert proc.returncode == 0
        hw_files = list(tmp_path.glob("hw_*.txt"))
        assert len(hw_files) == 1
        assert "NPU 0 OK" in hw_files[0].read_text()
        # The script may return before slow collection finishes; poll.
        assert wait_until(lambda: len(dump_jsons(tmp_path, worker.pid)) == 1, timeout=10)
        assert wait_until(lambda: len(dump_jsons(tmp_path, engine.pid)) == 1, timeout=10)
        assert elapsed < 10


class TestHwPreCollection:
    def test_missing_npu_smi_does_not_block(self, tmp_path, fast_script, child_factory):
        worker = child_factory("worker")
        proc, _ = run_script(fast_script, tmp_path)
        assert proc.returncode == 0
        assert list(tmp_path.glob("hw_*.txt")) == []
        assert wait_until(lambda: len(dump_jsons(tmp_path, worker.pid)) == 1, timeout=10)

    def test_hanging_npu_smi_is_killed(self, tmp_path, fast_script, child_factory):
        worker = child_factory("worker")
        stub = npu_smi_stub(tmp_path, "sleep 5")
        proc, elapsed = run_script(fast_script, tmp_path, path_prepend=stub)
        assert proc.returncode == 0
        assert list(tmp_path.glob("hw_*.txt")) == []
        assert wait_until(lambda: len(dump_jsons(tmp_path, worker.pid)) == 1, timeout=10)
        assert elapsed < 10


class TestPidValidation:
    def test_dead_pid_is_skipped(self, tmp_path, fast_script):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        write_pidfile(tmp_path, dead.pid)
        proc, elapsed = run_script(fast_script, tmp_path)
        assert proc.returncode == 0
        assert elapsed < 1.5, "a dead pid must be skipped without waiting"

    @pytest.mark.skipif(sys.platform != "linux", reason="needs /proc")
    def test_pid_reuse_is_skipped(self, tmp_path, fast_script):
        keeper = subprocess.Popen(["sleep", "30"])
        try:
            write_pidfile(tmp_path, keeper.pid, proc_start="1")  # wrong start time
            proc, elapsed = run_script(fast_script, tmp_path)
            assert proc.returncode == 0
            assert elapsed < 1.5, "a reused pid must not be signalled or awaited"
        finally:
            keeper.kill()
            keeper.wait()


class TestWaiting:
    @pytest.fixture
    def deaf_process(self, tmp_path):
        proc = subprocess.Popen(["/bin/sh", "-c", 'trap "" USR1; sleep 30'])
        time.sleep(0.1)
        write_pidfile(tmp_path, proc.pid)
        yield proc
        proc.kill()
        proc.wait()

    def test_no_dump_times_out_but_exits_zero(self, tmp_path, fast_script, deaf_process):
        proc, elapsed = run_script(fast_script, tmp_path)
        assert proc.returncode == 0, "prestop must never block the termination flow"
        assert elapsed >= 1.8, "the script must wait out the full window"
        assert dump_jsons(tmp_path, deaf_process.pid) == []

    def test_stale_dump_is_not_mistaken_for_completion(
        self, tmp_path, fast_script, deaf_process
    ):
        stale = tmp_path / f"dump_rank0_{deaf_process.pid}_100.json"
        stale.write_text("{}")
        old = time.time() - 3600
        os.utime(stale, (old, old))

        proc, elapsed = run_script(fast_script, tmp_path)
        assert proc.returncode == 0
        assert elapsed >= 1.8, "a pre-existing dump must not satisfy the wait"
