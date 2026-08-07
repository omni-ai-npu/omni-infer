# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import time
from typing import Any

from vllm.v1.request import Request
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.core.sched.scheduler import Scheduler

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("SchedulerResetPrefixCachePatch", Scheduler)
class SchedulerResetPrefixCachePatch(VLLMPatch):
    _attr_names_to_apply = [
        "_free_request",
        "_free_blocks",
        "delayed_free_req_ids",
        "has_delayed_free_requests",
        "has_requests",
    ]

    # modified method
    def _free_request(self: Scheduler, request: Request) -> dict[str, Any] | None:
        assert request.is_finished(), "request must be finished before freeing scheduler state."

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)
        else: # patch here
            # The connector is holding the blocks for an async transfer.
            # Track the request id so the engine stays awake polling
            # `connector.get_finished()` until the transfer completes.
            self.delayed_free_req_ids.add(request_id)

        return kv_xfer_params

    # modified method
    def _free_blocks(self: Scheduler, request: Request):
        assert request.is_finished(), "request must be finished before freeing KV cache blocks."
        self.kv_cache_manager.free(request)
        self.delayed_free_req_ids.discard(request.request_id) # patch here
        del self.requests[request.request_id]

    # added property
    @property
    def delayed_free_req_ids(self: Scheduler) -> set[str]:
        if not hasattr(self, "_delayed_free_req_ids"):
            setattr(self, "_delayed_free_req_ids", set())
        return self._delayed_free_req_ids

    # added method
    def has_delayed_free_requests(self: Scheduler) -> bool:
        """Returns True if any request still holds blocks pending the KV
        connector finishing an async outbound transfer."""
        return len(self.delayed_free_req_ids) > 0

    # modified method
    def has_requests(self: Scheduler) -> bool:
        """Override `SchedulerInterface.has_requests` to also stay alive while
        any request is waiting for the KV connector to complete an async
        outbound transfer. Without this, the engine can sleep on the input
        queue right after a prefill (PD-disagg prefiller), never polling
        `connector.get_finished()` to receive the decode-side acknowledgement,
        and the held blocks become unreclaimable until the connector's own
        timeout fires."""
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or self.has_delayed_free_requests()
        )


@register_patch("EngineCoreResetPrefixCachePatch", EngineCoreProc)
class EngineCoreResetPrefixCachePatch(VLLMPatch):
    _attr_names_to_apply = ["_process_engine_step"]

    # modified method
    def _process_engine_step(self: EngineCoreProc) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        # If no model execution happened but there are waiting requests
        # (e.g., WAITING_FOR_REMOTE_KVS), yield the GIL briefly to allow
        # background threads (like NIXL handshake) to make progress.
        # Without this, the tight polling loop can starve background threads.
        # if not model_executed and self.scheduler.has_unfinished_requests():
        if not model_executed and (
            self.scheduler.has_unfinished_requests()
            or self.scheduler.has_delayed_free_requests() # patch here
        ):
            time.sleep(0.001)

        return model_executed
