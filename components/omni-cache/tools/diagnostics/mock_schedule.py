# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Mock scheduling for deterministic KV dump comparison.

Activated by OMNI_MOCK_SCHEDULE=1. Monkey-patches vllm Scheduler to:
1. Sort running by request_id at every schedule() step
2. Let add_request go to waiting normally (triggers KV pull)
3. In schedule(), sweep waiting -> staging (intercept, KV already pulled)
4. Flush staging -> waiting only when running is empty AND batch is full
   vLLM then moves waiting -> running with its normal allocation strategy.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler

ENV_VAR = "OMNI_MOCK_SCHEDULE"
_MODULE_NAME = "vllm.v1.core.sched.scheduler"

_orig_add_request = None
_orig_schedule = None
_orig_has_requests = None

_PATCH_TARGETS = {
    "vllm.v1.core.sched.scheduler",
    "vllm.v1.core.sched.async_scheduler",
}


def _get_staging(scheduler: "Scheduler") -> list:
    staging = getattr(scheduler, "_mock_staging", None)
    if staging is None:
        staging = []
        scheduler._mock_staging = staging
    return staging


def _patched_add_request(self: "Scheduler", request) -> None:
    # Normal path: register + add to waiting (triggers KV pull)
    self.requests[request.request_id] = request
    _orig_add_request(self, request)


def _patched_has_requests(self: "Scheduler") -> bool:
    staging = getattr(self, "_mock_staging", None)
    if staging:
        return True
    return _orig_has_requests(self)


def _patched_schedule(self: "Scheduler"):
    self.running.sort(key=lambda r: r.request_id)

    staging = _get_staging(self)

    # Sweep all waiting requests into staging (intercept before running).
    # KV is already pulled by now — add_request triggered it earlier.
    while True:
        try:
            req = self.waiting.pop_request()
        except Exception:
            break
        staging.append(req)

    # Release a full batch only when the previous batch has finished.
    if staging and not self.running:
        active = [r for r in staging if not r.is_finished()]
        capacity = getattr(self, "max_num_running_reqs", 1)
        if len(active) >= capacity:
            active.sort(key=lambda r: r.request_id)
            for req in active:
                self.waiting.add_request(req)
                if self.log_stats:
                    from vllm.v1.engine import EngineCoreEventType
                    req.record_event(EngineCoreEventType.QUEUED)
            staging.clear()

    return _orig_schedule(self)


def install() -> None:
    """Called from plugin.py:register(). No-op — actual install via import hook."""
    pass


def install_in_worker() -> None:
    """vllm.general_plugins entry point. Patches Scheduler in-place."""
    import sys as _sys
    if not int(os.environ.get(ENV_VAR, "0")):
        return
    global _orig_add_request, _orig_schedule, _orig_has_requests
    if _orig_add_request is not None:
        return
    _try_patch(_sys)


def _apply_patch(mod) -> None:
    """Patch Scheduler (or AsyncScheduler) on the given module."""
    global _orig_add_request, _orig_schedule, _orig_has_requests
    cls = getattr(mod, "Scheduler", None) or getattr(mod, "AsyncScheduler", None)
    if cls is None:
        return
    _orig_add_request = cls.add_request
    _orig_schedule = cls.schedule
    _orig_has_requests = cls.has_requests
    cls.add_request = _patched_add_request
    cls.schedule = _patched_schedule
    cls.has_requests = _patched_has_requests


def _try_patch(_sys=None) -> None:
    global _orig_add_request
    if _orig_add_request is not None:
        return
    if _sys is None:
        import sys as _sys
    for name in _PATCH_TARGETS:
        mod = _sys.modules.get(name)
        if mod is not None:
            _apply_patch(mod)
            return


# Register a meta-path hook that intercepts the Scheduler module import
# and patches the class synchronously — before any code gets a reference.
import sys as __sys

if int(os.environ.get(ENV_VAR, "0")):
    # Try immediate patch (module may already be loaded).
    _try_patch(__sys)

    if _orig_add_request is None:
        # Daemon poll: scheduler module may not be importable yet.
        # Poll every 0.1s until it appears, then patch synchronously.
        import threading as __threading
        import time as __time

        def __poll():
            while _orig_add_request is None:
                __time.sleep(0.1)
                _try_patch(__sys)

        __threading.Thread(target=__poll, daemon=True).start()
