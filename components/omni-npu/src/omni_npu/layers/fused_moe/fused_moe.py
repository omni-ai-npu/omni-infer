# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional

import torch
import torch_npu

from vllm.platforms import current_platform


def _apply_experts(layer: torch.nn.Module, h: torch.Tensor, expert_tokens: torch.Tensor, pertoken_scale=None):
    from omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize import PreparePermuteResult

    return layer.quant_method.apply_experts(
        layer=layer,
        prepare_permute_result=PreparePermuteResult(
            hidden_states_sorted_by_experts=h,
            expert_tokens=expert_tokens,
            dynamic_scale=pertoken_scale,
        ),
        activation=getattr(layer, "activation", "silu"),
    )


def fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(gating_output, k=topk)

    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids, row_idx


def grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None
):
    gating_output = gating_output.float()
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.unsqueeze(0)
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores,
                                        k=topk,
                                        dim=-1,
                                        sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_ids = topk_ids.int()
    # adapt add row_idx
    row_idx = torch.arange(topk_ids.numel(), device=topk_ids.device, dtype=topk_ids.dtype)
    row_idx = row_idx.reshape(topk_ids.shape[1], topk_ids.shape[0]).transpose(1, 0).contiguous()
    # adapt end

    return topk_weights, topk_ids, row_idx


def fused_experts_tp(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
):
    row_idx = torch.arange(topk_ids.numel(), device=current_platform.device_type,
                           dtype=torch.int32).view(-1, x.shape[0]).transpose(0, 1)
    num_tokens, hidden_size = x.shape
    n_routed_experts = layer.global_num_experts
    sorted_tokens, expanded_src_to_dst_row, expanded_expert_idx = \
        torch_npu.npu_moe_init_routing(x, row_idx, topk_ids, num_tokens)
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, n_routed_experts).to(torch.int64)
    if layer.quant_config is None:
        pertoken_scale = None
    else:
        sorted_tokens, pertoken_scale = torch_npu.npu_dynamic_quant(sorted_tokens)

    out = _apply_experts(layer, sorted_tokens, expert_tokens, pertoken_scale)
    
    return torch_npu.npu_moe_finalize_routing(out, None, None, None, topk_weights,
                                              expanded_src_to_dst_row, topk_ids).to(torch.bfloat16)