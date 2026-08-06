# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""In-process engine progress heartbeat.

`OmniNpuStatLogger` calls `mark_progress()` whenever an engine produces output.
If an engine hangs it stops producing output and the timestamp freezes.

Two consumers share this single source of truth:
  1. Metrics: `engine_iterations` counter tracks scheduling progress.
  2. Watchdog: `stalled_engines()` is read from the patched `check_health` to
     flip `/health` to unhealthy when an engine is stalled while serving traffic.

All state lives in the API-server process, so a module-level singleton is enough.
`monotonic` clock is used to avoid system-time skew.
"""

import threading
import time

# engine_idx -> monotonic timestamp of last observed progress
_last_progress: dict[int, float] = {}
# engine_idx -> timestamp when the current batch of in-flight work started.
# Stamped on the 0 -> >0 in-flight edge from the patched OutputProcessor.add_request
# so that "first step of a new request hangs before any progress is recorded" is
# still detected, without depending on a health probe polling the idle->busy edge.
_busy_since: dict[int, float] = {}
# engine_idx -> sleep flag; sleeping engines are exempt from hang detection.
_sleeping: dict[int, bool] = {}
_lock = threading.Lock()


def mark_progress(engine_idx: int = 0) -> None:
    """Record that an engine just made progress."""
    with _lock:
        _last_progress[engine_idx] = time.monotonic()
        _sleeping[engine_idx] = False  # making progress => awake


def snapshot() -> dict[int, float]:
    """Return effective seconds-since-progress for every known engine.

    Effective time uses the same basis as stalled_engines -- max(last_progress,
    busy_since) -- so the seconds reported for a hang line up with the threshold
    the stall was actually judged against. (Reporting raw now - last_progress
    would overstate a first-request-after-idle hang: it would show the long age
    since the engine last produced, not how long this in-flight batch is stuck.)
    """
    now = time.monotonic()
    with _lock:
        return {
            idx: now - max(ts, _busy_since.get(idx, ts))
            for idx, ts in _last_progress.items()
        }


def reset() -> None:
    """Clear all heartbeat state. Intended for tests only."""
    with _lock:
        _last_progress.clear()
        _busy_since.clear()
        _sleeping.clear()


def mark_busy() -> None:
    """Stamp the busy-since timestamp on the 0 -> >0 in-flight edge.

    Called request-side from the patched OutputProcessor.add_request when a
    request arrives while the engine was idle (no unfinished requests). It marks
    when the current batch of work actually started, so a hang on the very first
    step of a new request is detected even if record() is never called -- and,
    unlike polling in check_health, without depending on a health probe firing
    during the idle->busy transition.

    The current deployment has one local engine per API server, so the edge is
    applied to every known engine.
    """
    now = time.monotonic()
    with _lock:
        for idx in _last_progress:
            _busy_since[idx] = now


def mark_initialized(engine_idx: int = 0) -> None:
    """Mark one engine as freshly initialized and reset only its heartbeat.

    A just-initialized engine is in a clean state: making progress right now,
    not sleeping, and with no in-flight batch under way. So we stamp its
    last_progress and drop any leftover sleep / busy-since state for it.

    This is deliberately per-engine, not a global reset(): other engines in the
    same process may already be running, and reset() (test-only) would wipe them.
    """
    with _lock:
        now = time.monotonic()
        _last_progress[engine_idx] = now
        _sleeping[engine_idx] = False      # not sleeping right after init
        _busy_since.pop(engine_idx, None)  # no in-flight batch yet; drop stale start


def mark_sleeping(engine_idx: int, sleeping: bool) -> None:
    """Mark an engine as sleeping / waking up.

    Waking up refreshes the progress timestamp to avoid an immediate stall verdict.
    """
    with _lock:
        _sleeping[engine_idx] = sleeping
        if not sleeping:
            _last_progress[engine_idx] = time.monotonic()


def is_sleeping(engine_idx: int = 0) -> bool:
    with _lock:
        return _sleeping.get(engine_idx, False)


def stalled_engines(hang_sec: float, now: float | None = None) -> list[int]:
    """Return engine indexes whose heartbeat has been frozen longer than hang_sec.

    An engine is reported when:
      - effective elapsed time since progress exceeds hang_sec, where effective
        time is max(last_progress, busy_since);
      - it is not sleeping.

    This is a pure time check and intentionally does NOT look at in-flight work.
    An idle engine with no traffic has a stale heartbeat and will show up here --
    that is expected, not a hang. The "is there actually work in flight?" gate
    (in_flight > 0) is applied by the caller (check_health); a reported engine
    only means a hang when combined with in_flight > 0. Keeping the two concerns
    separate is intentional -- do not add an in_flight check in here.
    """
    now = time.monotonic() if now is None else now
    out: list[int] = []
    with _lock:
        for idx, ts in _last_progress.items():
            eff = max(ts, _busy_since.get(idx, ts))
            if (now - eff) <= hang_sec:
                continue
            if _sleeping.get(idx, False):
                continue
            out.append(idx)
    return out
