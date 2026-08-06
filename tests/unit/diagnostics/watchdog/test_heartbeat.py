# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for omni.diagnostics.watchdog.heartbeat.

Covers the progress API, sleep exemption and stalled-engine detection.
Pure CPU; no NPU needed.
"""

import pytest

from omni.diagnostics.watchdog import heartbeat


@pytest.fixture(autouse=True)
def _reset_heartbeat():
    heartbeat.reset()
    yield
    heartbeat.reset()


# ----- progress -----

def test_mark_progress_makes_recent():
    heartbeat.mark_progress(0)
    snap = heartbeat.snapshot()
    assert 0 in snap and snap[0] < 1.0


def test_snapshot_lists_all_engines():
    heartbeat.mark_progress(0)
    heartbeat.mark_progress(1)
    snap = heartbeat.snapshot()
    assert set(snap.keys()) == {0, 1}
    assert all(v >= 0 for v in snap.values())


def test_reset_clears_everything():
    heartbeat.mark_progress(0)
    heartbeat.mark_sleeping(0, True)
    heartbeat.mark_initialized(1)
    heartbeat.reset()
    assert heartbeat.snapshot() == {}
    assert heartbeat.is_sleeping(0) is False


# ----- sleep -----

def test_mark_sleeping_toggles_flag():
    heartbeat.mark_sleeping(0, True)
    assert heartbeat.is_sleeping(0) is True
    heartbeat.mark_sleeping(0, False)
    assert heartbeat.is_sleeping(0) is False


def test_wake_resets_progress_ts():
    heartbeat._last_progress[0] = 1.0
    heartbeat._sleeping[0] = True
    heartbeat.mark_sleeping(0, False)
    assert heartbeat.is_sleeping(0) is False
    assert heartbeat.snapshot()[0] < 1.0


def test_progress_clears_sleeping_flag():
    heartbeat.mark_sleeping(0, True)
    heartbeat.mark_progress(0)
    assert heartbeat.is_sleeping(0) is False


# ----- initialization -----

def test_mark_initialized_resets_progress():
    heartbeat.mark_initialized(0)
    assert heartbeat.snapshot()[0] < 1.0


# ----- stalled_engines -----

HANG = 10.0
NOW = 100.0


def test_stalled_when_frozen():
    heartbeat._last_progress[0] = 80.0
    assert heartbeat.stalled_engines(HANG, now=NOW) == [0]


def test_not_stalled_when_recent_progress():
    heartbeat._last_progress[0] = 99.5
    assert heartbeat.stalled_engines(HANG, now=NOW) == []


def test_sleeping_engine_is_exempt():
    heartbeat._last_progress[0] = 80.0
    heartbeat._sleeping[0] = True
    assert heartbeat.stalled_engines(HANG, now=NOW) == []


def test_busy_since_catches_first_step_hang():
    heartbeat.mark_initialized(0)
    heartbeat._last_progress[0] = 1.0
    heartbeat._busy_since[0] = 80.0
    assert heartbeat.stalled_engines(HANG, now=NOW) == [0]


# ----- busy_since (request-side push) -----

def test_mark_busy_stamps_known_engines():
    heartbeat.mark_initialized(0)  # registers engine 0, clears busy_since
    assert 0 not in heartbeat._busy_since
    heartbeat.mark_busy()
    assert 0 in heartbeat._busy_since


def test_mark_busy_ignores_unknown_engines():
    heartbeat.mark_busy()  # nothing registered yet
    assert heartbeat._busy_since == {}


def test_multi_engine_reports_only_hung():
    heartbeat._last_progress[0] = 99.0
    heartbeat._last_progress[1] = 80.0
    assert heartbeat.stalled_engines(HANG, now=NOW) == [1]


def test_empty_returns_empty():
    assert heartbeat.stalled_engines(HANG, now=NOW) == []
