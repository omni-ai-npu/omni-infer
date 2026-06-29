# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Prefill metadata utilities for OmniCache.

This module provides functions for building volatile metadata during prefill operations,
including slot mapping and attention metadata preparation.
"""

from typing import List, Optional, Tuple

import torch
import os

from omni_cache.cache.utils.ops import pad_inputs, generate_full_block_slot


def get_volatile_metadata(
    cache,
    query_lens_list: List[int],
    seq_lens_list: List[int],
    graph_pad_size: int,
    pad_slot_id: int,
    orig_slot_mapping: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build volatile metadata for prefill operations.

    This function prepares slot mappings and volatile table for prefill
    attention operations, handling padding and block alignment.

    Args:
        cache: The PrefillOmniCache instance.
        query_lens_list: List of query lengths per request.
        seq_lens_list: List of sequence lengths per request.
        graph_pad_size: Padding size for graph compilation.
        pad_slot_id: Slot ID value for padding positions.
        orig_slot_mapping: Original slot mapping tensor.

    Returns:
        Tuple of (volatile_table, slot_mapping) or (None, None) if DSA not enabled.
    """
    orig_shape = orig_slot_mapping.shape

    # Handle attention parallelism padding
    attn_sp_size = int(os.environ.get("ATTN_SP_SIZE", "1"))
    if attn_sp_size > 1:
        orig_slot_mapping = pad_inputs(
            orig_slot_mapping,
            query_lens_list,
            attn_sp_size * 2,
            pad_slot_id
        )
    else:
        orig_slot_mapping = generate_full_block_slot(
            orig_slot_mapping, query_lens_list, cache.block_size
        )

    # Calculate padding and total lengths
    query_lens_tensor = torch.tensor(query_lens_list)
    padding_lens_tensor = (
        cache.block_size - query_lens_tensor % cache.block_size
    ) % cache.block_size
    total_lens_list = (query_lens_tensor + padding_lens_tensor).tolist()

    cache.sum_total_len = sum(total_lens_list)
    cache.sum_query_len = sum(query_lens_list)
    cache.query_mask = torch.zeros(cache.sum_total_len, dtype=torch.bool)

    start = 0
    for ql, tl in zip(query_lens_list, total_lens_list):
        cache.query_mask[start:start + ql] = 1
        start += tl

    # Compute token indices and slot mapping
    blocks, slots = (
        orig_slot_mapping // cache.block_size,
        orig_slot_mapping % cache.block_size
    )
    ranks, rank_slots = (
        slots // cache.rank_block_size,
        slots % cache.node_block_size
    )

    token_idx = torch.nonzero(
        (ranks == cache.tp_rank) & (orig_slot_mapping > 0),
        as_tuple=True
    )[0]
    slot_mapping = blocks[token_idx] * cache.node_block_size + rank_slots[token_idx]
    num_tokens = token_idx.shape[0]

    cache.batch_slots_cpu[:num_tokens].copy_(slot_mapping, non_blocking=True)
    cache.batch_token_indices = token_idx

    if os.environ.get("ENABLE_DSA", "0") != "1":
        return None, None

    # Build slot mapping for DSA mode
    max_num_blocks_per_req = cache.volatile_table.shape[1]
    slot_mapping = []
    for i, (q_len, seq_len) in enumerate(zip(query_lens_list, seq_lens_list)):
        start = (i * max_num_blocks_per_req + 1) * cache.block_size
        slot_mapping.append(
            cache.arange[seq_len - q_len:seq_len] + start
        )

    if graph_pad_size > 0:
        padding = torch.full(
            (graph_pad_size,),
            fill_value=pad_slot_id,
            dtype=cache.arange.dtype,
            device=cache.arange.device
        )
        slot_mapping.append(padding)

    slot_mapping = torch.cat(slot_mapping, dim=0)

    if slot_mapping.shape != orig_shape:
        raise RuntimeError(
            f"Slot mapping shape mismatch! {slot_mapping.shape=}, {orig_shape=}."
        )

    return cache.volatile_table, slot_mapping
