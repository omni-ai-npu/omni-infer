# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Tensor utility functions for prefill cache operations.

This module contains tensor manipulation utilities used by PrefillOmniCache.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def padding_kv_cache(tensor, sum_total_len: int, query_mask, sum_query_len: int):
    """Pad KV cache tensor to total length with query positions.

    Args:
        tensor: Input tensor to pad
        sum_total_len: Total length after padding
        query_mask: Mask for query positions
        sum_query_len: Length of query portion

    Returns:
        Padded tensor
    """
    import torch

    result = torch.zeros(
        (sum_total_len, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device
    )
    result[query_mask] = tensor[:sum_query_len]
    return result


def nd_to_nz(
    tensor,
    sum_total_len: int,
    query_mask,
    sum_query_len: int,
    block_size: int,
    nz_size: int,
    is_hybrid_attn: bool,
    batch_token_indices=None
):
    """Convert ND format tensor to NZ format.

    Args:
        tensor: Input tensor in ND format
        sum_total_len: Total length for padding
        query_mask: Mask for query positions
        sum_query_len: Length of query portion
        block_size: Block size for reshaping
        nz_size: NZ dimension size
        is_hybrid_attn: Whether hybrid attention is enabled
        batch_token_indices: Token indices for batch (optional)

    Returns:
        Tensor in NZ format
    """
    import torch

    # First apply padding
    tensor = padding_kv_cache(tensor, sum_total_len, query_mask, sum_query_len)
    global_num_tokens = tensor.size(0)
    tensor = tensor.view(
        global_num_tokens // block_size,
        block_size,
        -1,
        nz_size
    )
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    tensor = tensor.view(global_num_tokens, -1)

    if not is_hybrid_attn and batch_token_indices is not None:
        tensor = tensor[batch_token_indices]

    return tensor
