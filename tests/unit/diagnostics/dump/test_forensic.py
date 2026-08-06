# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for OMNI-DUMP forensic primitives."""
import json
import threading
import time

import pytest

from omni.diagnostics.dump import constants, forensic

pytestmark = pytest.mark.unit


class TestAtomicWrite:
    def test_write_then_read_back(self, tmp_path):
        target = tmp_path / "out.json"
        forensic.atomic_write_text(target, "hello world")
        assert target.read_text(encoding="utf-8") == "hello world"
        assert list(tmp_path.iterdir()) == [target], "no tmp file left behind"

    def test_replace_failure_leaves_no_partial_target(self, tmp_path, monkeypatch):
        target = tmp_path / "out.json"

        def boom(src, dst):
            raise OSError("injected replace failure")

        monkeypatch.setattr(forensic.os, "replace", boom)
        with pytest.raises(OSError):
            forensic.atomic_write_text(target, "x" * 4096)
        assert not target.exists(), "readers must never see a partial file"

    def test_replace_failure_keeps_old_content(self, tmp_path, monkeypatch):
        target = tmp_path / "out.json"
        forensic.atomic_write_text(target, "old-complete-content")

        def boom(src, dst):
            raise OSError("injected replace failure")

        monkeypatch.setattr(forensic.os, "replace", boom)
        with pytest.raises(OSError):
            forensic.atomic_write_text(target, "new-content")
        assert target.read_text(encoding="utf-8") == "old-complete-content"


class TestMergeDumpSchema:
    def test_signal_dump_full_schema(self, tmp_path):
        json_path = forensic.merge_dump(
            tmp_path,
            role="worker",
            rank=0,
            trigger=constants.TRIGGER_SIGNAL,
            stack_text="Thread 0x01 (most recent call first):\n  fake frame",
            stats={"since_last_complete_sec": 1.5},
            hardware={"proc_status": {"VmRSS": "1 kB"}},
        )
        assert json_path.exists()
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        assert doc["schema"] == constants.DUMP_SCHEMA
        assert doc["trigger"] == "signal"
        assert doc["role"] == "worker"
        assert doc["rank"] == 0
        assert isinstance(doc["pid"], int)
        assert isinstance(doc["ts"], float)
        assert doc["stack"] == [
            "Thread 0x01 (most recent call first):",
            "  fake frame",
        ]
        assert doc["stats"] == {"since_last_complete_sec": 1.5}
        assert doc["hardware"] == {"proc_status": {"VmRSS": "1 kB"}}

    def test_json_is_the_only_artifact(self, tmp_path):
        json_path = forensic.merge_dump(
            tmp_path, role="worker", rank=0, trigger="signal", stack_text="s"
        )
        assert not json_path.with_suffix(".txt").exists()
        assert list(tmp_path.iterdir()) == [json_path]

    def test_json_is_indented_for_humans(self, tmp_path):
        json_path = forensic.merge_dump(
            tmp_path, role="engine", trigger="signal", stack_text="s"
        )
        assert json_path.read_text(encoding="utf-8").count("\n") > 5

    def test_filename_contains_role_pid(self, tmp_path):
        import os as _os

        pid = _os.getpid()
        p_worker = forensic.merge_dump(
            tmp_path, role="worker", rank=3, trigger="signal", stack_text="s"
        )
        p_engine = forensic.merge_dump(tmp_path, role="engine", trigger="signal", stack_text="s")
        p_api = forensic.merge_dump(tmp_path, role="api", trigger="signal", stack_text="s")
        assert p_worker.name.startswith("dump_rank3_")
        assert p_engine.name.startswith("dump_engine_")
        assert p_api.name.startswith("dump_api_")
        for p in (p_worker, p_engine, p_api):
            # the prestop script locates packages via: dump_*_${pid}_*.json
            assert f"_{pid}_" in p.name
            assert p.suffix == ".json"


class TestTriggerTiering:
    def test_atexit_dump_skips_stack(self, tmp_path):
        json_path = forensic.merge_dump(
            tmp_path,
            role="engine",
            trigger=constants.TRIGGER_ATEXIT,
            stack_text="should be ignored",
            stats={"engine_step_count": 7},
        )
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        assert doc["stack"] == constants.STACK_SKIPPED
        assert doc["stats"] == {"engine_step_count": 7}

    def test_signal_dump_keeps_stack(self, tmp_path):
        json_path = forensic.merge_dump(
            tmp_path, role="engine", trigger=constants.TRIGGER_SIGNAL, stack_text="real stack"
        )
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        assert doc["stack"] == ["real stack"]


class TestOnceGuard:
    def test_concurrent_enter_collects_exactly_once(self):
        guard = forensic.OnceGuard()
        calls = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            guard.enter("signal", lambda: calls.append(1))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1

    def test_late_enter_skipped_with_return_value(self):
        guard = forensic.OnceGuard()
        assert guard.enter("signal", lambda: None) is True
        assert guard.enter("atexit", lambda: None) is False

    def test_hung_collection_does_not_block_later_triggers(self):
        guard = forensic.OnceGuard()
        started = threading.Event()
        release = threading.Event()

        def hung_collect():
            started.set()
            release.wait()

        t = threading.Thread(
            target=lambda: guard.enter("atexit", hung_collect), daemon=True
        )
        t.start()
        assert started.wait(2)

        begin = time.monotonic()
        assert guard.enter("signal", lambda: None) is False
        assert time.monotonic() - begin < 1.0, "a hung collection must not block the decision"
        release.set()
        t.join(2)


class TestCleanupStale:
    def test_cleanup_removes_own_pid_keeps_others(self, tmp_path):
        pid = 4242
        (tmp_path / "raw").mkdir()
        own = [
            tmp_path / f"dump_rank0_{pid}_100.json",
            tmp_path / "raw" / f"dump_engine_{pid}_stats.json",
            tmp_path / "raw" / f"dump_{pid}_stack.txt",
        ]
        keep = [
            tmp_path / "dump_rank1_9999_100.json",
            tmp_path / "raw" / "dump_9999_stack.txt",
            tmp_path / "hw_100.txt",
        ]
        for f in own + keep:
            f.write_text("x")
        forensic.cleanup_stale(tmp_path, pid)
        for f in own:
            assert not f.exists(), f"{f.name} should have been removed"
        for f in keep:
            assert f.exists(), f"{f.name} must not be touched"

    def test_cleanup_missing_dir_is_noop(self, tmp_path):
        forensic.cleanup_stale(tmp_path / "not-exist", 1)
