# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shared helpers for NPU MoE apply_experts (CodeCheck dups)."""

from typing import NamedTuple

import torch


class ApplyExpertsState(NamedTuple):
    experts: object
    shared_experts: object
    moe_parallel_config: object
    hidden_states: torch.Tensor
    expert_tokens: object
    avg_tokens_per_expert: object
    pertoken_scale: object


def unpack_apply_experts_state(layer, prepare_permute_result):
    shared = layer._shared_experts
    return ApplyExpertsState(
        experts=layer.routed_experts,
        shared_experts=None if shared is None else shared._layer,
        moe_parallel_config=layer.moe_config.moe_parallel_config,
        hidden_states=prepare_permute_result.hidden_states_sorted_by_experts,
        expert_tokens=prepare_permute_result.expert_tokens,
        avg_tokens_per_expert=getattr(
            prepare_permute_result, "avg_tokens_per_expert", None
        ) or [0],
        pertoken_scale=prepare_permute_result.dynamic_scale,
    )
