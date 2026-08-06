# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for OMNI-DUMP process-level hardware probe."""
import os
import subprocess
import time

import pytest

from omni.diagnostics.dump import hardware_probe

pytestmark = pytest.mark.unit

FAKE_PROC_STATUS = """Name:\tpython
VmHWM:\t  200000 kB
VmRSS:\t  100000 kB
Threads:\t12
FDSize:\t256
Uid:\t500
"""


class TestProcessProbeIsolation:
    def test_torch_failure_does_not_break_proc_status(self, tmp_path, monkeypatch):
        proc_file = tmp_path / "status"
        proc_file.write_text(FAKE_PROC_STATUS)

        def no_torch():
            raise ImportError("torch not installed")

        monkeypatch.setattr(hardware_probe, "_import_torch", no_torch)
        hw = hardware_probe.collect_process_hw(proc_status_path=proc_file)
        assert str(hw["device_mem"]).startswith("ERROR:")
        assert hw["proc_status"] == {
            "VmRSS": "100000 kB",
            "VmHWM": "200000 kB",
            "Threads": "12",
            "FDSize": "256",
        }

    def test_proc_failure_does_not_break_device_mem(self, tmp_path, monkeypatch):
        class FakeNpu:
            @staticmethod
            def mem_get_info():
                return (1024, 4096)

        class FakeTorch:
            npu = FakeNpu()

        monkeypatch.setattr(hardware_probe, "_import_torch", lambda: FakeTorch())
        hw = hardware_probe.collect_process_hw(proc_status_path=tmp_path / "missing")
        assert hw["device_mem"] == {"free_bytes": 1024, "total_bytes": 4096}
        assert str(hw["proc_status"]).startswith("ERROR:")


class TestHwReference:
    def test_references_newest_hw_file(self, tmp_path):
        now = time.time()
        for name, age in (("hw_100.txt", 300), ("hw_200.txt", 200), ("hw_300.txt", 5)):
            f = tmp_path / name
            f.write_text("NPU 0 OK")
            os.utime(f, (now - age, now - age))
        ref = hardware_probe.find_hw_reference(tmp_path)
        assert ref["hardware_ref"] == "hw_300.txt"
        assert 0 <= ref["age_sec"] < 60

    def test_no_hw_file_yields_none(self, tmp_path):
        assert hardware_probe.find_hw_reference(tmp_path) is None
        assert hardware_probe.find_hw_reference(tmp_path / "missing") is None


class TestHangProtection:
    def test_hanging_mem_get_info_times_out(self, tmp_path, monkeypatch):
        import threading

        class HangingNpu:
            @staticmethod
            def mem_get_info():
                threading.Event().wait()  # never returns

        class HangingTorch:
            npu = HangingNpu()

        monkeypatch.setattr(hardware_probe, "_import_torch", lambda: HangingTorch())
        monkeypatch.setattr(
            hardware_probe.constants, "DEVICE_MEM_PROBE_TIMEOUT_SEC", 0.2
        )
        proc_file = tmp_path / "status"
        proc_file.write_text(FAKE_PROC_STATUS)

        start = time.monotonic()
        hw = hardware_probe.collect_process_hw(proc_status_path=proc_file)
        assert time.monotonic() - start < 2.0, "a hung device probe must not hang the dump"
        assert str(hw["device_mem"]).startswith("ERROR:timeout")
        assert hw["proc_status"]["VmRSS"] == "100000 kB"

    def test_device_mem_can_be_skipped(self, tmp_path):
        proc_file = tmp_path / "status"
        proc_file.write_text(FAKE_PROC_STATUS)
        hw = hardware_probe.collect_process_hw(
            proc_status_path=proc_file, include_device_mem=False
        )
        assert hw["device_mem"] == "skipped"


class TestNeverSpawns:
    def test_all_probe_paths_spawn_nothing(self, tmp_path, monkeypatch):
        def tripwire(*args, **kwargs):
            raise AssertionError("hardware probe must never spawn a subprocess")

        monkeypatch.setattr(subprocess, "Popen", tripwire)
        monkeypatch.setattr(os, "system", tripwire)
        if hasattr(os, "posix_spawn"):
            monkeypatch.setattr(os, "posix_spawn", tripwire)

        proc_file = tmp_path / "status"
        proc_file.write_text(FAKE_PROC_STATUS)
        hardware_probe.collect_process_hw(proc_status_path=proc_file)
        hardware_probe.find_hw_reference(tmp_path)
