# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Utility functions for fake metadata operations.

This module provides helper functions used by both compress and postprocess
modules for fake metadata handling.
"""


def compute_compress_outlen(B: int, T: int, ratio: int) -> int:
    """Compute output length for compressed attention.

    Args:
        B: Batch size (number of requests)
        T: Total number of tokens
        ratio: Compression ratio

    Returns:
        Compressed output length (min of T and T//ratio + B)
    """
    return min(T, T // ratio + B)


def is_dummy_run(seq_len: int, metadata_grp_id: int, skip_conditions: list) -> bool:
    """Check if this is a dummy run that should skip processing.

    Args:
        seq_len: Current sequence length
        metadata_grp_id: Current metadata group ID
        skip_conditions: List of group IDs to skip

    Returns:
        True if this run should be skipped
    """
    if seq_len <= 1:
        return True
    if metadata_grp_id in skip_conditions:
        return True
    return False


def calculate_max_block_need(
    max_model_len: int,
    block_size: int,
    metadata_grp_id: int,
    extra_blocks: int = 6
) -> int:
    """Calculate maximum number of blocks needed.

    Args:
        max_model_len: Maximum model length
        block_size: Size of each block
        metadata_grp_id: Metadata group ID (affects token multiplier)
        extra_blocks: Extra blocks to add for safety margin

    Returns:
        Maximum number of blocks needed
    """
    if metadata_grp_id == 2:
        token_per_blc = block_size * 3
        max_block_need = (max_model_len + token_per_blc - 1) // token_per_blc
    else:
        token_per_blc = block_size * 128
        max_block_need = (max_model_len + token_per_blc - 1) // token_per_blc + extra_blocks

    return max_block_need
