# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from vllm.v1.core.sched.utils import check_stop

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


@register_patch("PanguV2SchedulerPatch", Scheduler)
class PanguV2SchedulerPatch(VLLMPatch):
    _attr_names_to_apply = [
        "_update_request_with_output",
    ]

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:

        # PD disaggregation: the prefill (kv_producer) node must stop after
        # exactly `original_max_tokens` (1 token) via FINISHED_LENGTH_CAPPED
        kv_transfer_config = getattr(self.vllm_config, "kv_transfer_config", None)
        is_p_node = bool(
            kv_transfer_config is not None
            and getattr(kv_transfer_config, "kv_role", None) == "kv_producer"
        )

        # Get start_token_id and end_token_id for reasoning end detection.
        # vLLM 0.25.1 initializes these IDs on reasoning_config.
        start_token_id = end_token_id = None
        reasoning_config = getattr(self.vllm_config, "reasoning_config", None)
        if (
            not is_p_node
            and envs.OMNI_ENABLE_MAX_TOKENS_EXCLUDE_REASONING
            and reasoning_config is not None
            and reasoning_config.enabled
        ):
            start_token_ids = reasoning_config.reasoning_start_token_ids
            end_token_ids = reasoning_config.reasoning_end_token_ids
            if start_token_ids:
                start_token_id = start_token_ids[-1]
            if end_token_ids:
                end_token_id = end_token_ids[-1]

        if not hasattr(request, "content_generated"):
            request.reasoning_ended = end_token_id is None
            request.content_generated = (
                request.num_output_tokens if request.reasoning_ended else 0
            )
            request._original_max_tokens = request.max_tokens
            request._reasoning_started = start_token_id is None or (
                start_token_id in (request.prompt_token_ids or [])
                or start_token_id in request.output_token_ids
            )

        original_max_tokens = request._original_max_tokens

        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            was_reasoning_ended = request.reasoning_ended
            if end_token_id is not None and output_token_id == end_token_id:
                request.reasoning_ended = True
                request._reasoning_started = True
            if start_token_id is not None and output_token_id == start_token_id:
                request._reasoning_started = True
            if was_reasoning_ended:
                request.content_generated += 1

            if not request._reasoning_started:
                # Reasoning hasn't started yet (and we can still detect its
                # start): bound total output normally.
                request.max_tokens = original_max_tokens
            elif not request.reasoning_ended:
                # Reasoning in progress: lift the cap so check_stop's
                # max_tokens comparison can't trigger on it.
                request.max_tokens = self.max_model_len
            else:
                # Reasoning ended: only content tokens should count. check_stop compares
                # the real (total) request.num_output_tokens against request.max_tokens.
                request.max_tokens = original_max_tokens + (
                    request.num_output_tokens - request.content_generated
                )

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                request.max_tokens = original_max_tokens
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped
