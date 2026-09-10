# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared child-process helpers for exit_dump and prestop tests."""
import os
import subprocess
import sys
import time
from pathlib import Path

CHILD = Path(__file__).with_name("exit_dump_child.py")
SRC = Path(__file__).parents[4] / "src"


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def dump_jsons(dump_dir, pid):
    return sorted(Path(dump_dir).glob(f"dump_*_{pid}_*.json"))


class Child:
    """A child process that installed exit forensics and waits for a stop file."""

    def __init__(self, role, dump_dir, scenario, tmp_path):
        self.stop_file = tmp_path / "stop"
        env = dict(os.environ, PYTHONPATH=str(SRC))
        self.proc = subprocess.Popen(
            [sys.executable, str(CHILD), role, str(dump_dir), scenario, str(self.stop_file)],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        self.lines = []
        while True:
            line = self.proc.stdout.readline().strip()
            self.lines.append(line)
            if line == "READY" or not line:
                break
        assert "READY" in self.lines, f"child failed to start: {self.lines}"

    @property
    def pid(self):
        return self.proc.pid

    def release(self):
        self.stop_file.write_text("")

    def kill_if_alive(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.proc.stdout.close()
