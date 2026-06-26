# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Attention metadata builder for decode operations.

This module provides functions to build fake attention metadata for decode
operations in hybrid attention mode.
"""

from typing import TYPE_CHECKING

import os

import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backends.utils import (
        AttentionMetadataBuilder,
        CommonAttentionMetadata,
    )


def fake_build_attention(
    cache,
    builder: "AttentionMetadataBuilder",
    common_attn_metadata: "CommonAttentionMetadata",
    draft_index: int = -1
) -> "NPUMetadata":
    """Build fake attention metadata for decode operations.

    This function constructs NPUMetadata for decode attention operations,
    preparing block tables and slot mappings with host mapping support.

    Args:
        cache: The DecodeOmniCache instance.
        builder: Attention metadata builder with buffers.
        common_attn_metadata: Common attention metadata.
        draft_index: Draft index for MTP (-1 for normal decode).

    Returns:
        NPUMetadata for decode attention operations.
    """
    from vllm.v1.attention.backends.utils import (
        AttentionCGSupport,
        AttentionMetadataBuilder,
        CommonAttentionMetadata,
        split_decodes_and_prefills
    )
    from vllm.attention.backends.utils import PAD_SLOT_ID
    from omni_npu.attention.backends.attention import NPUMetadata
    import numpy as np
    from omni_cache.cache.decode.static_utils import torch_to_numpy_zero_copy

    num_actual_tokens = common_attn_metadata.num_actual_tokens
    max_query_len = common_attn_metadata.max_query_len
    max_seq_len = common_attn_metadata.max_seq_len
    query_start_loc = common_attn_metadata.query_start_loc
    query_cumlens = common_attn_metadata.query_cumlens
    seq_lens = common_attn_metadata.seq_lens
    slot_mapping = common_attn_metadata.slot_mapping
    num_prefills = common_attn_metadata.num_prefills
    num_decodes = common_attn_metadata.num_decodes
    num_decode_tokens = common_attn_metadata.num_decode_tokens

    seq_lens_cpu = torch_to_numpy_zero_copy(seq_lens)
    query_start_loc_cpu = torch_to_numpy_zero_copy(query_start_loc)

    block_table = common_attn_metadata.block_table_tensor
    num_reqs = common_attn_metadata.num_reqs
    cur_metadata_grp_id = cache.metadata_grp_id

    # Import here to avoid circular dependency
    from omni_cache.gather_selection import get_block_table_np
    from omni_cache.cache.core.constants import ENABLE_HOST_MAPPING

    # Prepare block_table and slot_mapping
    if ENABLE_HOST_MAPPING:
        blk_table_np, slot_mapping_np = get_block_table_np(
            cache.runner, cache, cache.runner.graph_block_tables, cur_metadata_grp_id,
            block_table, num_reqs, num_actual_tokens
        )
    else:
        raise RuntimeError("ENABLE_HOST_MAPPING must be enabled for H2D transfer.")

    if getattr(cache, "req_ids_update_buffer", None) is None:
        cache.req_ids_update_buffer = {}

    if seq_lens_cpu[0] > 1 and len(cache.req_ids_update_buffer) > 0:
        from .sliding_window_attention import post_process_fake_metadata
        block_table_np, slot_mapping_np = post_process_fake_metadata(
            cache, num_reqs, slot_mapping_np, block_table_np,
            common_attn_metadata, cur_metadata_grp_id,
            query_start_loc_cpu, seq_lens_cpu, draft_index
        )
        block_table.copy_(
            cache.blk_table_buffers[cur_metadata_grp_id][:block_table.shape[0]],
            non_blocking=True
        )
        slot_mapping.copy_(
            cache.slot_mapping_buffers[cur_metadata_grp_id][:slot_mapping.shape[0]],
            non_blocking=True
        )

    attn_metadata = NPUMetadata(
        num_actual_tokens=num_actual_tokens,
        block_tables=block_table,
        query_start_loc=common_attn_metadata.query_start_loc,
        query_cumlens=query_cumlens,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        slot_mapping=slot_mapping,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
    )

    return attn_metadata
