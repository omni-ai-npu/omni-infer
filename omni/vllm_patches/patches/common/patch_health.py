# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hang-detection watchdog patch for AsyncLLM health checks.

The watchdog flips `/health` (and SageMaker `/ping`) to 503 when an engine is
stalled while serving traffic, without killing the server. Recovery is automatic
once the engine resumes making progress.

Timestamping lives in `omni_npu.diagnostics.watchdog.heartbeat` and is driven by
`OmniNpuStatLogger`.

Two patches are registered here and applied automatically by the patch manager:
  - `HealthHangPatch` replaces `AsyncLLM.check_health`.
  - `HealthAttachRouterPatch` registers an app-level exception handler that turns
    `EngineHangError` into a 503 JSON response.
"""

from http import HTTPStatus

from fastapi.responses import JSONResponse

from vllm.logger import init_logger
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import OutputProcessor
import vllm.entrypoints.serve.instrumentator.health as health_mod

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.diagnostics.watchdog import heartbeat

logger = init_logger(__name__)


class EngineHangError(Exception):
    """Signal that one or more engines are hung.

    Not inheriting from `EngineDeadError` and not setting `errored` keeps the
    server alive and lets it recover automatically.
    """

    def __init__(self, engines: list[int], stalled_sec: dict[int, float], in_flight: int):
        self.engines = engines
        self.stalled_sec = stalled_sec
        self.in_flight = in_flight
        super().__init__(f"engine hang: {engines}")


def _hang_config() -> float:
    """Hang-detection threshold in seconds."""
    return float(envs.OMNI_HEALTH_HANG_SEC)


# Capture the community check_health at import time (before the patch manager
# swaps it) so the patched version can still call the original -- same capture
# pattern as _orig_add_request below.
_orig_check_health = AsyncLLM.check_health


@register_patch("HealthHangPatch", AsyncLLM)
class HealthHangPatch(VLLMPatch):
    _attr_names_to_apply = ["check_health"]

    async def check_health(self):
        # Run the community check first: it re-raises for a dead engine and runs
        # any upstream checks vLLM may add. We only layer hang detection on top,
        # so we await it (execute it -- a bare call would just build an unawaited
        # coroutine and skip the checks) but do NOT return it: our own check
        # below still needs to run, and we must not depend on its return value.
        await _orig_check_health(self)

        # Current deployments route one local engine per API server, so the
        # aggregate unfinished-request count is the right proxy for "has work".
        # busy_since is stamped request-side (patched OutputProcessor.add_request),
        # so check_health only reads heartbeat state here -- it never writes it.
        in_flight = self.output_processor.get_num_unfinished_requests()

        hang_sec = _hang_config()
        stalled = heartbeat.stalled_engines(hang_sec)

        # Log only in_flight here, not `stalled`: stalled_engines is a pure time
        # check that ignores in_flight, so it can be non-empty for a healthy idle
        # engine (in_flight=0), which would look alarming in the log. A real hang
        # (stalled AND in_flight>0) is reported by the warning below.
        logger.debug("check_health: in_flight=%d", in_flight)

        if stalled and in_flight > 0:
            snap = heartbeat.snapshot()
            stalled_sec = {idx: snap.get(idx) for idx in stalled}
            logger.warning(
                "engine hang detected: engines=%s stalled_sec=%s in_flight=%d",
                stalled, stalled_sec, in_flight,
            )
            raise EngineHangError(stalled, stalled_sec, in_flight)


async def _engine_hang_handler(request, exc):
    return JSONResponse(
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        content={
            "fault message": "engine hang",
            "engines": exc.engines,
            "in_flight": exc.in_flight,
        },
    )


_orig_attach_router = health_mod.attach_router


def _attach_router(app):
    _orig_attach_router(app)
    app.add_exception_handler(EngineHangError, _engine_hang_handler)


@register_patch("HealthAttachRouterPatch", health_mod)
class HealthAttachRouterPatch(VLLMPatch):
    _attr_names_to_apply = ["attach_router"]

    attach_router = _attach_router


_orig_add_request = OutputProcessor.add_request


@register_patch("BusySinceAddRequestPatch", OutputProcessor)
class BusySinceAddRequestPatch(VLLMPatch):
    """Stamp busy_since on the request-side 0 -> >0 in-flight edge.

    Detecting the edge here, rather than polling in check_health, means hang
    detection no longer depends on a health probe firing during the idle->busy
    transition: once a request is in flight, a single later probe is enough.
    """

    _attr_names_to_apply = ["add_request"]

    def add_request(self, *args, **kwargs):
        # request_states is still empty at entry -> this request starts a fresh
        # busy window for the previously idle engine.
        if self.get_num_unfinished_requests() == 0:
            heartbeat.mark_busy()
        return _orig_add_request(self, *args, **kwargs)
