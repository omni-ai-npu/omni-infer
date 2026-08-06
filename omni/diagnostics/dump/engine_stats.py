# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""EngineCore runtime stats: micro tick around execute_model plus dump-time
direct reads of live scheduler state.

Concurrency model: single writer (the engine main thread calls on_dispatch /
on_complete), multiple readers (dump and stats-flush threads call snapshot).
Scalar reads/writes are atomic under the GIL, so no lock is needed.
"""
import logging
import time
from collections import deque

from omni_npu.diagnostics.dump import constants

logger = logging.getLogger(__name__)

_last_dispatch_ts = None
_last_complete_ts = None
_step_count = 0
_step_durations = deque(maxlen=constants.STEP_DURATION_RING_SIZE)
_dispatched_batch = {}


def reset():
    global _last_dispatch_ts, _last_complete_ts, _step_count, _dispatched_batch
    _last_dispatch_ts = None
    _last_complete_ts = None
    _step_count = 0
    _step_durations.clear()
    _dispatched_batch = {}


def on_dispatch(scheduler_output):
    """Record dispatch time and a summary of the batch being sent to workers.

    Runs on the engine hot loop: memory writes only, never raises.
    """
    global _last_dispatch_ts, _dispatched_batch
    now = time.time()
    _last_dispatch_ts = now
    try:
        tokens = getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
        # Raw facts only: a phase label cannot be derived reliably from
        # SchedulerOutput (chunked-prefill continuations are not in
        # scheduled_new_reqs; batches mix prefill and decode; spec decode
        # schedules >1 token per request).
        _dispatched_batch = {
            "total_num_scheduled_tokens": tokens,
            "num_reqs": len(num_scheduled) if num_scheduled is not None else 0,
            "new_reqs": len(new_reqs) if new_reqs is not None else 0,
        }
    except Exception as e:  # noqa: BLE001 - never raise into the engine loop
        logger.warning("omni-dump: on_dispatch stats update failed, stats degraded: %r", e)


def on_complete():
    """Record completion time; a growing dispatch/complete gap means workers are stuck."""
    global _last_complete_ts, _step_count
    now = time.time()
    try:
        if _last_dispatch_ts is not None:
            _step_durations.append(now - _last_dispatch_ts)
        _last_complete_ts = now
        _step_count += 1
    except Exception as e:  # noqa: BLE001 - never raise into the engine loop
        logger.warning("omni-dump: on_complete stats update failed, stats degraded: %r", e)


def snapshot():
    """Return an independent copy of the micro-tick state (reader side)."""
    now = time.time()
    since = (now - _last_complete_ts) if _last_complete_ts is not None else 0.0
    return {
        "last_dispatch_ts": _last_dispatch_ts,
        "last_complete_ts": _last_complete_ts,
        "since_last_complete_sec": float(since),
        "engine_step_count": _step_count,
        "step_durations_recent": list(_step_durations),
        "dispatched_batch": dict(_dispatched_batch),
    }


def collect_scheduler_state(scheduler):
    """Read live scheduler attributes at dump time.

    Each field degrades independently to "ERROR:<field>" so an upstream vLLM
    rename never voids the whole stats section.
    """
    state = {}
    try:
        state["num_running"] = len(scheduler.running)
    except Exception:  # noqa: BLE001
        state["num_running"] = "ERROR:running"
    try:
        state["num_waiting"] = len(scheduler.waiting)
    except Exception:  # noqa: BLE001
        state["num_waiting"] = "ERROR:waiting"
    try:
        state["kv_cache_usage"] = float(scheduler.kv_cache_manager.usage)
    except Exception:  # noqa: BLE001
        state["kv_cache_usage"] = "ERROR:kv_cache_usage"
    return state
