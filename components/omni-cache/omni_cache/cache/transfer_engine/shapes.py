# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Cache shape calculation utilities for H2D/D2H operations.

This module provides unified cache shape calculation functions for both
decode and prefill operations.
"""

import math
from typing import Tuple

import torch

from vllm.logger import init_logger
from omni_cache.cache.core.constants import (
    SIZE_BYTES_PER_LAYER_DECODE,
    SIZE_BYTES_PER_LAYER_PREFILL,
    LOCAL_DP_SIZE,
    PRE_CALC_PREFILL_BLOCK_NUM,
)
from omni_cache.cache.utils.ops import divide_or_raise

logger = init_logger("vllm.v1.omni")



_HUGE_PAGE_SIZE = 2 * 1024 * 1024


def _align_num_blocks(num_blocks, num_layers, block_size, head_total, itemsize):
    """Return num_blocks rounded DOWN so that num_layers * num_blocks *
    block_size * head_total * itemsize is a multiple of the 2 MiB huge page.

    Keeps per-rank host-cache strides 2 MiB-aligned so NPU MMU aliasing and
    aclrtMemcpyAsync both burst at maximum width, and OX's implicit per-rank
    offset (num_blocks * num_layers * block_bytes) stays in sync with
    Python's (size_total) with no extra knob.
    """
    unit = num_layers * block_size * head_total * itemsize
    if unit <= 0 or num_blocks <= 0:
        return num_blocks
    gcd_v = math.gcd(unit, _HUGE_PAGE_SIZE)
    if gcd_v == 0:
        return num_blocks
    step = _HUGE_PAGE_SIZE // gcd_v
    if step <= 1:
        return num_blocks
    aligned = (num_blocks // step) * step
    return aligned if aligned > 0 else num_blocks


def calc_cache_shape_for_decode(
    num_layers: int,
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
) -> Tuple[Tuple[int, ...], int]:
    """Calculate cache shape for decode operations.

    Args:
        num_layers: Number of layers in the model.
        block_size: Size of each cache block.
        head_size: Size of each attention head.
        dtype: Data type of the cache tensors.

    Returns:
        Tuple of (cache_shape, num_blocks) where cache_shape is
        `(LOCAL_DP_SIZE, num_layers, num_blocks, block_size, head_size)`.
        The leading die axis allows per-die slicing in the memory pool.
        num_blocks is the per-die block count (total blocks / LOCAL_DP_SIZE).
    """
    itemize = dtype.itemsize
    numel_per_layer = divide_or_raise(SIZE_BYTES_PER_LAYER_DECODE, itemize)
    numel_per_block = block_size * head_size
    num_blocks_decode = (numel_per_layer // numel_per_block) // LOCAL_DP_SIZE
    num_blocks_decode = _align_num_blocks(
        num_blocks_decode, num_layers, block_size, head_size, itemize,
    )

    d_shape = (
        LOCAL_DP_SIZE,
        num_layers,
        num_blocks_decode,
        block_size,
        head_size,
    )
    logger.debug(
        f"**DecodeOmniCache**: {num_layers=}; {d_shape=}; {num_blocks_decode=}"
    )

    return d_shape, num_blocks_decode


def calc_cache_shape_for_prefill(
    num_layers: int,
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
) -> Tuple[Tuple[int, ...], int]:
    """Calculate cache shape for prefill operations.

    Args:
        num_layers: Number of layers in the model.
        block_size: Size of each cache block.
        head_size: Size of each attention head.
        dtype: Data type of the cache tensors.

    Returns:
        Tuple of (cache_shape, num_blocks_prefill) where cache_shape is a tuple
        of dimensions (num_layers, num_blocks, block_size, head_size).
    """
    itemize = dtype.itemsize
    numel_per_layer = divide_or_raise(SIZE_BYTES_PER_LAYER_PREFILL, itemize)
    numel_per_block = block_size * head_size
    num_blocks_prefill = numel_per_layer // numel_per_block  # floor division
    num_blocks_prefill = _align_num_blocks(
        num_blocks_prefill, num_layers, block_size, head_size, itemize,
    ) - 1 # safe boundary

    if PRE_CALC_PREFILL_BLOCK_NUM != num_blocks_prefill:
        logger.debug(f"<<< {PRE_CALC_PREFILL_BLOCK_NUM=} not equal to {num_blocks_prefill=}")

    p_shape = (
        num_layers,
        num_blocks_prefill,
        block_size,
        head_size
    )

    return p_shape, num_blocks_prefill


def calculate_kv_xsfer_params(shape, num_blocks: int, dp_local_rank: int) -> Tuple[int, int]:
    """Calculate KV transfer parameters.

    Args:
        shape: Cache shape tuple.
        num_blocks: Number of blocks per rank.
        dp_local_rank: Data parallel local rank.

    Returns:
        Tuple of (block_len_dtype, dp_offset).
    """
    block_len_dtype = math.prod(shape[3:]) * shape[1]
    logger.debug(f"<<<<<< {block_len_dtype=}, {shape=}")
    dp_offset = dp_local_rank * num_blocks
    return block_len_dtype, dp_offset