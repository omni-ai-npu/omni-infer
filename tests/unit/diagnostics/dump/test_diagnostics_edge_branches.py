# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Edge-case exception branches not reached by the primary module test suites.

These are the degrade-not-crash paths whose whole point is to swallow an OS-level
failure; the happy paths are covered elsewhere, so this file only forces the
failure branches (a failing unlink during stale cleanup, an OSError while
scanning for hw_* references) plus the real torch import shim.
"""
import pathlib

import pytest

from omni.diagnostics.dump import forensic, hardware_probe

pytestmark = pytest.mark.unit


class TestForensicCleanupFailure:
    def test_unlink_failure_is_logged_not_raised(
            self, tmp_path, monkeypatch, caplog, capture_logger):
        pid = 4243
        (tmp_path / "raw").mkdir()
        target = tmp_path / f"dump_rank0_{pid}_100.json"
        target.write_text("x")

        real_unlink = pathlib.Path.unlink

        def flaky_unlink(self, *args, **kwargs):
            if self.name == target.name:
                raise OSError("permission denied")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "unlink", flaky_unlink)
        # must not raise even though the matching file cannot be removed
        with capture_logger(forensic.logger):
            forensic.cleanup_stale(tmp_path, pid)
        assert any("cleanup of" in r.message for r in caplog.records)


class TestHwReferenceFailure:
    def test_glob_oserror_degrades_to_none(self, tmp_path, monkeypatch):
        def boom(self, pattern):
            raise OSError("fs unavailable")

        monkeypatch.setattr(pathlib.Path, "glob", boom)
        assert hardware_probe.find_hw_reference(tmp_path) is None


class TestImportTorchShim:
    def test_import_torch_returns_the_module(self):
        # On NPU/CI torch is installed; the shim simply returns it. This covers
        # the lazy-import body that every other test monkeypatches away.
        torch = pytest.importorskip("torch")
        assert hardware_probe._import_torch() is torch
