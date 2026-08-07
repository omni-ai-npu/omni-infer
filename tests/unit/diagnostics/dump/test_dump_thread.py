# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the OMNI-DUMP collection thread."""
import os
import signal
import time

import pytest

from omni_npu.diagnostics.dump import constants, dump_thread

pytestmark = pytest.mark.unit


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def pipe_fds():
    r, w = os.pipe()
    os.set_blocking(r, False)
    os.set_blocking(w, False)
    yield r, w
    for fd in (r, w):
        try:
            os.close(fd)
        except OSError:
            pass


@pytest.fixture
def collector():
    class Collector:
        def __init__(self):
            self.calls = []
            self.raise_once = False

        def __call__(self):
            if self.raise_once:
                self.raise_once = False
                raise RuntimeError("injected collect failure")
            self.calls.append(time.monotonic())

    return Collector()


def start_thread(pipe_fds, collector):
    thread = dump_thread.DumpThread(pipe_fds[0], collector)
    thread.start()
    return thread


class TestWakeup:
    def test_signal_byte_triggers_collect_once(self, pipe_fds, collector):
        thread = start_thread(pipe_fds, collector)
        try:
            os.write(pipe_fds[1], bytes([constants.DUMP_SIGNAL]))
            assert wait_until(lambda: len(collector.calls) == 1)
            time.sleep(0.05)
            assert len(collector.calls) == 1
        finally:
            thread.stop()

    def test_signal_storm_is_drained_without_losing_the_dump_signal(
        self, pipe_fds, collector
    ):
        noise = bytes([signal.SIGTERM]) * 5000
        os.write(pipe_fds[1], noise + bytes([constants.DUMP_SIGNAL]))
        thread = start_thread(pipe_fds, collector)
        try:
            assert wait_until(lambda: len(collector.calls) == 1)
            time.sleep(0.05)
            assert len(collector.calls) == 1, "one batch must collect exactly once"
        finally:
            thread.stop()

    def test_non_dump_signals_are_discarded(self, pipe_fds, collector):
        thread = start_thread(pipe_fds, collector)
        try:
            os.write(pipe_fds[1], bytes([signal.SIGTERM, signal.SIGINT]))
            time.sleep(0.1)
            assert collector.calls == []
            # the thread must be back on select and still responsive
            os.write(pipe_fds[1], bytes([constants.DUMP_SIGNAL]))
            assert wait_until(lambda: len(collector.calls) == 1)
        finally:
            thread.stop()


class TestLifecycle:
    def test_stop_joins_promptly(self, pipe_fds, collector):
        thread = start_thread(pipe_fds, collector)
        start = time.monotonic()
        thread.stop(timeout=2.0)
        assert time.monotonic() - start < 2.0
        assert not thread.is_alive()

    def test_collect_failure_keeps_thread_alive(self, pipe_fds, collector):
        collector.raise_once = True
        thread = start_thread(pipe_fds, collector)
        try:
            os.write(pipe_fds[1], bytes([constants.DUMP_SIGNAL]))
            time.sleep(0.1)
            assert collector.calls == []
            os.write(pipe_fds[1], bytes([constants.DUMP_SIGNAL]))
            assert wait_until(lambda: len(collector.calls) == 1)
        finally:
            thread.stop()
