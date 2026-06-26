# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Fake metadata compression utilities for strided attention.

This module provides functions to build compressed attention metadata
for decode operations with strided KV cache compression patterns.
"""

import os
from typing import TYPE_CHECKING

import torch

from omni_cache.cache.utils.ops import torch_to_numpy_zero_copy

# disable triton for quick fix
ENABLE_HOST_MAPPING = os.getenv("ENABLE_HOST_MAPPING", "1") == "1"

if ENABLE_HOST_MAPPING:
    try:
        import triton  # noqa: F401
    except ImportError:
        triton = None
    from omni_cache.cache.device_backend.ascend.ops.triton_ops import (
        build_fake_block_table_kernel_compress,
    )
else:
    triton = None
    build_fake_block_table_kernel_compress = None


def _next_power_of_2(n: int) -> int:
    """Python fallback for triton.next_power_of_2."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

if TYPE_CHECKING:
    from vllm.v1.attention.backends.utils import (
        AttentionCGSupport,
        AttentionMetadataBuilder,
        CommonAttentionMetadata,
    )


def compute_compress_outlen(B: int, T: int, ratio: int) -> int:
    """Compute output length for compressed attention.

    Args:
        B: Batch size (number of requests)
        T: Total number of tokens
        ratio: Compression ratio

    Returns:
        Compressed output length
    """
    return min(T, T // ratio + B)


def _extract_metadata_params(cache, common_attn_metadata):
    """Extract and normalize metadata parameters.

    Extracts common parameters from metadata and converts CPU tensors
    to numpy arrays for zero-copy access.

    Args:
        cache: The DecodeOmniCache instance
        common_attn_metadata: Common attention metadata

    Returns:
        Tuple of extracted parameters and numpy arrays
    """
    num_reqs = common_attn_metadata.num_reqs
    num_actual_tokens = common_attn_metadata.num_actual_tokens
    max_query_len = common_attn_metadata.max_query_len
    max_seq_len = common_attn_metadata.max_seq_len
    query_start_loc = common_attn_metadata.query_start_loc
    seq_lens = common_attn_metadata.seq_lens
    block_table_tensor = common_attn_metadata.block_table_tensor

    block_size = cache.kv_cache_config.kv_cache_groups[cache.metadata_grp_id].kv_cache_spec.block_size
    max_model_len = cache.vllm_config.model_config.max_model_len
    hbm_block_pool = cache.hbm_buffer_block_table_pool[cache.metadata_grp_id]

    seq_lens_cpu = torch_to_numpy_zero_copy(common_attn_metadata.seq_lens_cpu)
    query_start_loc_cpu = torch_to_numpy_zero_copy(common_attn_metadata.query_start_loc_cpu)

    return (
        num_reqs, num_actual_tokens, max_query_len, max_seq_len,
        query_start_loc, seq_lens, block_table_tensor,
        block_size, max_model_len, hbm_block_pool,
        seq_lens_cpu, query_start_loc_cpu
    )


def _fix_padding_metadata(common_attn_metadata, seq_lens_cpu, seq_lens, query_start_loc, block_table_tensor):
    """Fix metadata for padded sequences.

    Identifies padding positions and adjusts tensors accordingly.

    Args:
        common_attn_metadata: Common attention metadata
        seq_lens_cpu: Sequence lengths on CPU (numpy array)
        seq_lens: Sequence lengths tensor
        query_start_loc: Query start locations tensor
        block_table_tensor: Block table tensor
    """
    if not common_attn_metadata.attn_metadata or getattr(common_attn_metadata, "cmp_metadata_fixed", False):
        return

    first_pad_pos = getattr(common_attn_metadata, "first_pad_pos", None)
    if first_pad_pos is None:
        is_pad = seq_lens_cpu == 0
        if is_pad.any():
            first_pad_pos = is_pad.nonzero(as_tuple=True)[0][0]
        else:
            first_pad_pos = -1
        setattr(common_attn_metadata, "first_pad_pos", first_pad_pos)

    if first_pad_pos != -1:
        seq_lens[first_pad_pos:].fill_(0)
        query_start_loc[first_pad_pos + 1 :] = query_start_loc[first_pad_pos]
        block_table_tensor[first_pad_pos:].fill_(0)

    setattr(common_attn_metadata, "cmp_metadata_fixed", True)


def _should_skip_fake_build(cache) -> bool:
    """Check if fake block table build should be skipped.

    Determines skip conditions based on environment variables
    and metadata group ID.

    Args:
        cache: The DecodeOmniCache instance

    Returns:
        True if should skip, False otherwise
    """
    if int(os.getenv("DISABLE_C128_MAPPING", "0")):
        skip_conditions = [3, 4]
    else:
        skip_conditions = [3]

    return cache.metadata_grp_id in skip_conditions


def _build_fake_block_table(
    cache, block_table_tensor, num_reqs, block_size, max_model_len, hbm_block_pool
) -> None:
    """Build fake block table contents using Triton kernel.

    Fills the block table tensor with fake contents from HBM buffer pool.

    Args:
        cache: The DecodeOmniCache instance
        block_table_tensor: Block table tensor to fill
        num_reqs: Number of requests
        block_size: Block size
        max_model_len: Maximum model length
        hbm_block_pool: HBM buffer pool dictionary
    """
    if getattr(cache, "req_ids_update_buffer", None) is None:
        cache.req_ids_update_buffer = {}

    actual_num_reqs = min(num_reqs, len(cache.req_ids_update_buffer))
    if actual_num_reqs <= 0:
        return

    # Pre-calculate select_len
    if cache.metadata_grp_id == 2:
        token_per_blc = block_size * 3
        max_block_need = (max_model_len + token_per_blc - 1) // token_per_blc
    else:
        token_per_blc = block_size * 128
        max_block_need = (max_model_len + token_per_blc - 1) // token_per_blc + 6

    select_len = min(max_block_need, block_table_tensor.shape[1])
    real_indices_tensor = cache.id_to_idx_table[:actual_num_reqs]

    if not ENABLE_HOST_MAPPING:
        raise RuntimeError("fake_build_compress requires ENABLE_HOST_MAPPING=1")

    BLOCK_SIZE = (
        triton.next_power_of_2(select_len) if triton is not None
        else _next_power_of_2(select_len)
    )
    build_fake_block_table_kernel_compress[(actual_num_reqs,)](
        block_table_tensor,
        hbm_block_pool['block_table_ts'],
        real_indices_tensor,
        block_table_tensor.stride(0),
        block_table_tensor.stride(1),
        hbm_block_pool['req_offset'],
        select_len,
        BLOCK_SIZE=BLOCK_SIZE
    )


def _prepare_block_table_buffer(builder, block_table_tensor) -> int:
    """Prepare block table buffer for compression.

    Ensures the builder has a properly sized block table buffer.

    Args:
        builder: Attention metadata builder
        block_table_tensor: Block table tensor

    Returns:
        Length of the block table (without extra blocks)
    """
    len_table = block_table_tensor.shape[1] - builder.num_extra_blocks

    if builder.block_table_buf is None:
        assert len_table > 0, f"Block table length should be positive, but got {len_table}."
        builder.block_table_buf = torch.zeros(
            builder.max_num_reqs, len_table, dtype=torch.int32, device=builder.device)

    assert builder.block_table_buf.shape[1] + builder.num_extra_blocks == block_table_tensor.shape[1], \
        f"{builder.block_table_buf.shape=}, {block_table_tensor.shape=}"

    return len_table


def _check_apc_hits(builder, common_attn_metadata, seq_lens, query_start_loc, cur_slot_mapping):
    """Check for APC (prefix caching) hits and update slot mapping.

    Identifies sequences with prefix cache hits and marks their
    first slots with -1.

    Args:
        builder: Attention metadata builder
        common_attn_metadata: Common attention metadata
        seq_lens: Sequence lengths tensor
        query_start_loc: Query start locations tensor
        cur_slot_mapping: Current slot mapping tensor
    """
    num_prefills = common_attn_metadata.num_prefills
    if not builder.vllm_config.cache_config.enable_prefix_caching or num_prefills <= 0:
        return

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    computed_lens = builder.computed_lens_buffer[:seq_lens.shape[0]]
    is_apc_hit_mask = (computed_lens > 1) & (query_lens > 1)

    if not torch.any(is_apc_hit_mask):
        return

    num_new_slots_per_req = (seq_lens // builder.stride) - (computed_lens // builder.stride)
    slot_offsets = torch.cumsum(num_new_slots_per_req, dim=0, dtype=torch.long)
    slot_offsets = torch.cat((torch.tensor([0], device=slot_offsets.device), slot_offsets[:-1]))
    target_indices = slot_offsets[is_apc_hit_mask]
    valid_indices_mask = target_indices < cur_slot_mapping.shape[0]
    final_indices_to_update = target_indices[valid_indices_mask]

    if final_indices_to_update.numel() > 0:
        cur_slot_mapping[final_indices_to_update] = -1


def fake_build_compress(
    cache,
    builder: "AttentionMetadataBuilder",
    common_attn_metadata: "CommonAttentionMetadata",
) -> "StridedCompressAttentionMetadata":
    """Build compressed attention metadata with fake block table contents.

    This function is used in hybrid attention mode to prepare fake metadata
    for strided KV cache compression. It modifies block table tensors and
    builds the compressed attention metadata structure.

    Args:
        cache: The DecodeOmniCache instance
        builder: Attention metadata builder with buffers
        common_attn_metadata: Common attention metadata

    Returns:
        StridedCompressAttentionMetadata for compressed attention
    """
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills
    from vllm.v1.kv_cache_interface import StridedCompressKVCacheSpec
    from vllm.attention.backends.utils import PAD_SLOT_ID

    (
        StridedCompressAttentionMetadata,
        _layout_and_computed_lens_kernel,
        _scatter_copy_and_locate_kernel,
    ) = cache._load_stride_compress_backend()

    # Step 1: Extract metadata parameters
    (
        num_reqs, num_actual_tokens, max_query_len, max_seq_len,
        query_start_loc, seq_lens, block_table_tensor,
        block_size, max_model_len, hbm_block_pool,
        seq_lens_cpu, query_start_loc_cpu
    ) = _extract_metadata_params(cache, common_attn_metadata)

    # Step 2: Fix padding metadata if needed
    _fix_padding_metadata(
        common_attn_metadata, seq_lens_cpu,
        seq_lens, query_start_loc, block_table_tensor
    )

    # Step 3: Split decodes and prefills
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=builder.reorder_batch_threshold,
        )
    )

    # Step 4: Build fake block table contents (if not skipped)
    if seq_lens_cpu[0] > 1 and not _should_skip_fake_build(cache):
        _build_fake_block_table(
            cache, block_table_tensor, num_reqs,
            block_size, max_model_len, hbm_block_pool
        )

    # Step 5: Prepare block table buffer
    len_table = _prepare_block_table_buffer(builder, block_table_tensor)

    assert num_reqs == block_table_tensor.shape[0], f"{num_reqs=}, {block_table_tensor.shape=}"

    # Step 6: Get buffer views
    cur_n1 = builder.n1_buf[:num_reqs]
    cur_m = builder.m_buf[:num_reqs]
    cur_starts = builder.starts_buf[:num_reqs]
    cur_computed_lens = builder.computed_lens_buffer[:num_reqs]
    cur_kv_states = builder.kv_states_table[:num_reqs]
    cur_score_states = builder.score_states_table[:num_reqs]
    cur_block_table = builder.block_table_buf[:num_reqs, :len_table]

    # Step 7: Compute layout kernel
    _layout_and_computed_lens_kernel[(1,)](
        seq_lens,
        query_start_loc,
        cur_n1,
        cur_m,
        cur_starts,
        cur_computed_lens,
        stride=builder.stride,
        B=num_reqs,
        BLOCK_SIZE=1024,
    )

    # Step 8: Scatter copy kernel
    L = compute_compress_outlen(num_reqs, num_actual_tokens, builder.stride)
    cur_slot_mapping = builder.slot_mapping_buf[:L]
    cur_slot_mapping.fill_(PAD_SLOT_ID)

    _scatter_copy_and_locate_kernel[(num_reqs,)](
        query_start_loc,
        seq_lens,
        block_table_tensor,
        cur_n1,
        cur_m,
        cur_starts,
        cur_kv_states,
        cur_score_states,
        cur_block_table,
        cur_slot_mapping,
        block_size_const=builder.block_size,
        num_state_blocks=builder.num_state_blocks,
        num_extra_blocks=builder.num_extra_blocks,
        copy_len_table=len_table,
        compressor_block_size=builder.compressor_block_size,
        compressor_len_table=builder.kv_states_table.shape[1],
        N_remain=builder.stride * builder.coff,
        stride_bt_b=block_table_tensor.stride(0),
        stride_bt_l=block_table_tensor.stride(1),
        stride_kv_b=builder.kv_states_table.stride(0),
        stride_kv_l=builder.kv_states_table.stride(1),
        stride_score_b=builder.score_states_table.stride(0),
        stride_score_l=builder.score_states_table.stride(1),
        stride_buf_b=builder.block_table_buf.stride(0),
        stride_buf_l=builder.block_table_buf.stride(1),
        BLOCK_SIZE=128,
    )

    # Step 9: Check for APC hits
    _check_apc_hits(builder, common_attn_metadata, seq_lens, query_start_loc, cur_slot_mapping)

    # Step 10: Build and return metadata
    cmp_metadata = StridedCompressAttentionMetadata(
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_tables=builder.block_table_buf[:num_reqs],
        slot_mapping=cur_slot_mapping,
        kv_states_block_table=builder.kv_states_table[:num_reqs],
        score_states_block_table=builder.score_states_table[:num_reqs],
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        tiling_meta=builder.tiling_meta,
        computed_lens=builder.computed_lens_buffer[:seq_lens.shape[0]],
    )

    return cmp_metadata