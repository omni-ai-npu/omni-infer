# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import omni_custom_ops
import torch
import torch.nn.functional as F
import torch_npu
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def npu_fused_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    activation: str | None = "silu",
    max_query_len: int = -1,
    inplace: bool = False,
):
    if weight.dim() == 3:
        weight = weight.squeeze(1)
    weight = weight.transpose(0, 1).contiguous()

    return torch.ops.custom.npu_ai_infra_fused_causal_conv1d(
        x=x,
        weight=weight,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        activation=activation,
        max_query_len=max_query_len,
        conv_mode=0,
        inplace=inplace,
        residual_connection=0,
    )


def custom_depthwise_conv1d(x_t: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
    """
    A PyTorch-native 1D depthwise convolution that bypasses the AI_CORE.
    x_t: [batch, seqlen, dim] (Optimized contiguous layout)
    weight: [dim, width] or [dim, 1, width]
    bias: [dim] or None

    Returns: [batch, seqlen - width + 1, dim]
    """
    batch, seqlen, dim = x_t.shape
    width = weight.shape[-1]
    out_len = seqlen - width + 1

    # 1. Reshape weight to [width, dim] for native broadcasting
    w = weight.view(dim, width).transpose(0, 1).contiguous()
    x_t = x_t.contiguous()

    # 2. Unroll the sliding window (Vector Core MACs)
    out_t = x_t[:, 0:out_len, :] * w[0]
    for i in range(1, width):
        out_t += x_t[:, i : i + out_len, :] * w[i]

    # 3. Add bias if present
    if bias is not None:
        out_t += bias

    # Return in [batch, out_len, dim] layout to avoid redundant transposes downstream
    return out_t


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    dtype_in = x.dtype
    x = x.to(weight.dtype)

    conv_dim = weight.shape[0]
    if x.shape[-1] != conv_dim and x.shape[0] == conv_dim:
        # Handle legacy [dim, seqlen] input by transposing to [seqlen, dim]
        x = x.transpose(0, 1)

    x_for_cat = x.unsqueeze(0) 
    batch, seqlen, dim = x_for_cat.shape
    width = weight.shape[-1]
    state_len = width - 1

    if initial_states is None:
        pad_tensor = torch.empty((1, state_len, dim), dtype=x.dtype, device=x.device)
        pad_tensor[...] = 0
        x_padded_native = torch.cat([pad_tensor, x_for_cat], dim=1)
    else:
        if initial_states.shape[1] == dim:
            initial_t = initial_states.transpose(1, 2)
        else:
            initial_t = initial_states
            
        initial_t = initial_t[:, -state_len:, :]
        x_padded_native = torch.cat([initial_t, x_for_cat], dim=1)

    out_t = custom_depthwise_conv1d(x_padded_native, weight, bias)

    if return_final_states:
        if final_states_out is not None:
            final_states = x_padded_native[:, -state_len:, :].to(dtype_in)
            if final_states_out.shape[1] == dim:
                final_states_out[:, :, -state_len:].copy_(final_states.transpose(1, 2))
            else:
                final_states_out[:, -state_len:].copy_(final_states)

    out = out_t.squeeze(0) 
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    **kwargs
):
    """
    Combined Update + Ref function. 
    Guarantees shape alignment regardless of incoming legacy wrapper logic.
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    if x.dim() == 2:
        x_t = x.unsqueeze(1) 
    elif x.dim() == 3:
        x_t = x
    dim = x_t.shape[2]
    width = weight.shape[-1]
    state_len = width - 1

    batch_indices = torch.clamp(conv_state_indices, 0)
    state_slice = conv_state[batch_indices] 

    if state_slice.shape[1] == dim:
        state_t = state_slice.transpose(1, 2)

    else:
        state_t = state_slice
    store_state_len = state_t.shape[1]

    if num_accepted_tokens is not None:
        basic_range = torch.arange(state_len, dtype=num_accepted_tokens.dtype, device=num_accepted_tokens.device)
        batch_size = conv_state_indices.numel()
        batch_start = torch.arange(batch_size, dtype=num_accepted_tokens.dtype, device=num_accepted_tokens.device)
        selected_index = (
            batch_start[:, None] * state_t.shape[1]
            + num_accepted_tokens[:batch_size, None]
            + basic_range[None, :]
            - 1
        )
        state_t = state_t.view(-1, state_t.shape[-1])[selected_index].reshape(batch_size, state_len, -1)

    x_new_t = torch.cat([state_t, x_t], dim=1).to(weight.dtype)

    new_state_t = x_new_t[:, -store_state_len:, :]
    if state_slice.shape[1] == dim:
        new_state = new_state_t.transpose(1, 2)
    else:
        new_state = new_state_t

    torch_npu.npu_scatter_nd_update_(conv_state, conv_state_indices.unsqueeze(1), new_state)

    out_t = custom_depthwise_conv1d(x_new_t, weight, bias)

    out = out_t.squeeze(1)
    if activation is not None:
        out = F.silu(out)

    return out.to(x.dtype)


def causal_conv1d_fn_bubble(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor, 
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = -1,
    metadata=None,
    validate_data=False,
    seq_lens: torch.Tensor | None = None, 
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    bias = bias.contiguous() if bias is not None else None

    if seq_lens is not None:
        seqlens_list = seq_lens.tolist()
    else:
        seqlens_tensor = query_start_loc[1:] - query_start_loc[:-1]
        seqlens_list = seqlens_tensor.tolist()

    splits = torch.split(x, seqlens_list, dim=0)
    out_ref_b = []

    for i, x_s in enumerate(splits):
        if cache_indices[i] == pad_slot_id:
            continue

        state_slice = conv_states[cache_indices[i]].unsqueeze(0)

        out_b, _ = causal_conv1d_ref(
            x_s,
            weight,
            bias,
            activation=activation,
            return_final_states=True,
            final_states_out=state_slice,
            initial_states=state_slice if has_initial_state[i] else None
        )
        out_ref_b.append(out_b)

    out_ref_tensor = torch.cat(out_ref_b, dim=0)
    return out_ref_tensor


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    seqlens_list: list[int],
    cache_indices_list: list[int],
    has_initial_state_list: list[bool],
    activation: str | None = "silu",
    pad_slot_id: int = -1,
    metadata=None,
    validate_data=False,
):
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    bias = bias.contiguous() if bias is not None else None

    splits = torch.split(x, seqlens_list, dim=0)
    out_ref_b = []

    for i, x_s in enumerate(splits):
        if cache_indices_list[i] == pad_slot_id or seqlens_list[i] == 0:
            continue

        state_slice = conv_states[cache_indices_list[i]].unsqueeze(0)

        initial_s = state_slice if has_initial_state_list[i] else None

        out_b, _ = causal_conv1d_ref(
            x_s,
            weight,
            bias,
            activation=activation,
            return_final_states=True,
            final_states_out=state_slice,
            initial_states=initial_s
        )
        out_ref_b.append(out_b)

    out_ref_tensor = torch.cat(out_ref_b, dim=0)
    return out_ref_tensor
