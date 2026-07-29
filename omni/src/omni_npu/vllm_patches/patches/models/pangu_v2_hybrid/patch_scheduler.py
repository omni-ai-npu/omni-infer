# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.patches.common.patch_user_repetition_detection import (
    check_stop,
)

logger = init_logger(__name__)


@register_patch("PanguV2SchedulerPatch", Scheduler)
class PanguV2SchedulerPatch(VLLMPatch):
    _attr_names_to_apply = [
        "_update_request_with_output",
    ]

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:

        # With drafter enabled, input_fits_in_drafter may fail inconsistently
        # across DP groups near max_model_len due to different batch
        # compositions. MTP/EAGLE/EAGLE3 must execute the drafter on all DP
        # groups; partial execution causes service hang in MoE + expert
        # parallel + DP mode. Terminate requests early to avoid this.
        #
        # The *3 margin also prevents a skip-one-step issue in async scheduling
        # near max_model_len: num_computed_tokens is updated one step behind
        # (in _update_after_schedule), which can cause the scheduler to skip
        # a step and re-schedule the request in the next step with stale spec
        # token placeholders.
        #
        # The *3 margin accounts for the async scheduling pipeline:
        #
        #   Token p is generated when computing token p-1 (position lag = 1).
        #   The last valid schedule covers positions
        #     [M-1-2N, M-1-N] (1+N tokens: 1 real + N spec).
        #   At that point, update_request_with_output is processing
        #     tokens on positions [M-1-3N, M-1-2N].
        #   Positions referenced: M = max_model_len, N = num_speculative_tokens.
        #
        # Without the *3 margin, the drafter's input_fits_in_drafter check
        # sees max_seq_len that has already advanced past the boundary,
        # causing some DP groups to skip the drafter while others run it.
        num_early_skip_tokens = 0 \
            if self.vllm_config.speculative_config is None \
            else self.vllm_config.speculative_config.num_speculative_tokens * 3

        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len - num_early_skip_tokens)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped
