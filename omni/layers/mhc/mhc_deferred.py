# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shared deferred-MHC launch glue for the attention custom ops."""

import torch
from vllm.forward_context import get_forward_context


def run_attention_mhc_deferred(
    attn_forward,
    forward_context,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    residual: torch.Tensor,
    layer_name: str,
    mhc_layer_name: str,
    task_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch next-layer MHC work between attention and the attention epilog."""
    layer = forward_context.no_compile_layers[layer_name]
    mhc_module = forward_context.no_compile_layers[mhc_layer_name]
    deferred_results: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _launch_mhc() -> None:
        deferred_results.append(
            mhc_module.launch_fused_split_sinkhorn(residual, task_key)
        )

    layer.pre_epilog_callback = _launch_mhc
    try:
        output = attn_forward(hidden_states, cos, sin, layer_name)
    finally:
        layer.pre_epilog_callback = None

    if deferred_results:
        h_post, h_res = deferred_results[0]
    else:
        # Prefill-only paths do not run the decode epilog callback.
        h_post, h_res = mhc_module.launch_fused_split_sinkhorn(
            residual, task_key
        )
    return output, h_post, h_res


def call_attention_mhc_deferred(
    attn_forward,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    residual: torch.Tensor,
    layer_name: str,
    mhc_layer_name: str,
    task_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve the current forward context and launch deferred MHC work."""
    return run_attention_mhc_deferred(
        attn_forward,
        get_forward_context(),
        hidden_states,
        cos,
        sin,
        residual,
        layer_name,
        mhc_layer_name,
        task_key,
    )


def attention_mhc_deferred_fake(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    residual: torch.Tensor,
    layer_name: str,
    mhc_layer_name: str,
    task_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_module = get_forward_context().no_compile_layers[mhc_layer_name]
    residual = residual.reshape(
        -1, mhc_module.num_stream, mhc_module.hidden_size
    )
    h_post = torch.empty_like(residual[..., 0], dtype=torch.float32)
    h_res = torch.empty_like(
        residual[..., :mhc_module.num_stream], dtype=torch.float32
    )
    return torch.empty_like(hidden_states), h_post, h_res
