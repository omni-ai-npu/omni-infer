# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

from omni_npu.vllm_patches.usefull_patch.models.pangu_v2_base import patch_scheduler as sched_mod


def _request(**kwargs):
    values = {
        "output_token_ids": [],
        "prompt_token_ids": [],
        "num_output_tokens": 0,
        "max_tokens": 8,
    }
    values.update(kwargs)
    request = SimpleNamespace(**values)

    def append_output_token_ids(token_id):
        request.output_token_ids.append(token_id)
        request.num_output_tokens = len(request.output_token_ids)

    request.append_output_token_ids = append_output_token_ids
    return request


def _scheduler(enabled=True, start_ids=None, end_ids=None, kv_role=None):
    return SimpleNamespace(
        max_model_len=64,
        vllm_config=SimpleNamespace(
            kv_transfer_config=(
                SimpleNamespace(kv_role=kv_role) if kv_role is not None else None
            ),
            reasoning_config=SimpleNamespace(
                enabled=enabled,
                reasoning_start_token_ids=start_ids,
                reasoning_end_token_ids=end_ids,
            ),
        ),
    )


def test_scheduler_seeds_content_generated_from_existing_output_tokens():
    request = _request(num_output_tokens=3, max_tokens=8)
    scheduler = _scheduler(enabled=False)

    with patch.object(sched_mod.envs, "OMNI_ENABLE_MAX_TOKENS_EXCLUDE_REASONING", False), \
            patch.object(sched_mod, "check_stop", return_value=False):
        new_tokens, stopped = sched_mod.PanguV2SchedulerPatch._update_request_with_output(
            scheduler, request, [11]
        )

    assert new_tokens == [11]
    assert stopped is False
    assert request.reasoning_ended is True
    assert request.content_generated == 4
    assert request._original_max_tokens == 8


def test_scheduler_marks_reasoning_started_from_output_token_ids():
    request = _request(output_token_ids=[7], prompt_token_ids=[1, 2], max_tokens=4)
    scheduler = _scheduler(start_ids=[7], end_ids=[9])

    with patch.object(sched_mod.envs, "OMNI_ENABLE_MAX_TOKENS_EXCLUDE_REASONING", True), \
            patch.object(sched_mod, "check_stop", return_value=False):
        sched_mod.PanguV2SchedulerPatch._update_request_with_output(
            scheduler, request, [3]
        )

    assert request.content_generated == 0
    assert request._reasoning_started is True
    assert request.reasoning_ended is False
