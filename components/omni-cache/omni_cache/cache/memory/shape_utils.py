# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shape calculation utilities for KV cache memory pool."""

from typing import List, Tuple, Union
import torch
import os
from vllm.config.model import ModelConfig
from vllm.distributed.parallel_state import get_tp_group

from .constants import ENABLE_C8_INDEXER, SHAPE_RANK_PREFILL, SHAPE_RANK_DECODE


def compute_head_size_ratio(
    shape: tuple,
    is_hybrid_attn: bool,
    tp_world_size: int,
    enable_dsa: bool = False,
    vllm_config=None,
) -> List[int]:
    """Compute head size ratio based on attention type.

    Args:
        shape: Input shape tuple.
        is_hybrid_attn: Whether using hybrid attention.
        tp_world_size: Tensor parallel world size.
        enable_dsa: Whether DSA is enabled.

    Returns:
        List of head size ratios.
    """
    _kv, _rope, _idx = None, None, None
    if vllm_config is not None:
        from omni_cache.cache.utils.support import (
            _pangu_kv_head_dims_from_hf,
        )
        _kv, _rope, _idx = _pangu_kv_head_dims_from_hf(
            vllm_config.model_config.hf_config
        )
    if enable_dsa and not is_hybrid_attn:
        head_size_ratio = (
            [_kv, _rope, _idx] if _kv is not None else [512, 64, 128]
        )
    elif is_hybrid_attn:
        if ENABLE_C8_INDEXER:
            head_size_ratio = [512, 8]
        else:
            head_size_ratio = [shape[-1]]
    elif ModelConfig.is_deepseek_mla and not is_hybrid_attn:
        head_size_ratio = [_kv, _rope] if _kv is not None else [512, 64]
    else:
        head_size_ratio = [shape[-1], shape[-1]]

    return [h // tp_world_size for h in head_size_ratio]


def extract_dimensions(shape: tuple) -> Tuple[int, int, int, int]:
    """Extract dimensions from shape tuple.

    Args:
        shape: Input shape (4D for prefill, 5D for decode).

    Returns:
        Tuple of (num_ranks, num_layers, num_blocks, block_size).

    Raises:
        ValueError: If shape is unsupported.
    """
    if len(shape) == SHAPE_RANK_PREFILL:
        num_ranks = 1
        num_layers = shape[0]
        num_blocks = shape[1]
        block_size = shape[2]
    elif len(shape) == SHAPE_RANK_DECODE:
        num_ranks = shape[0]
        num_layers = shape[1]
        num_blocks = shape[2]
        block_size = shape[3]
    else:
        raise ValueError("Unsupported shape for ram cache")

    return num_ranks, num_layers, num_blocks, block_size


def compute_tensor_shapes(
    base_shape: tuple,
    head_size_ratio: List[int],
    head_total_slices: int
) -> List[tuple]:
    """Compute tensor shapes for each head component.

    Args:
        base_shape: Base shape (block_num, block_size, head_dim).
        head_size_ratio: Head size ratios.
        head_total_slices: Total head slices.

    Returns:
        List of shapes for each tensor component.
    """
    shapes = []
    for ratio in head_size_ratio:
        dim_size = base_shape[-1] * ratio // head_total_slices
        shapes.append((*base_shape[:-1], dim_size))
    return shapes


def calculate_offsets_and_sizes(
    shapes: List[tuple],
    element_size: int,
    num_layers: int
) -> Tuple[List[int], List[int], List[int], int]:
    """Calculate byte offsets and sizes for tensor components.

    Args:
        shapes: List of tensor shapes.
        element_size: Size of each element in bytes.
        num_layers: Number of layers.

    Returns:
        Tuple of (sizes, offsets, layer_sizes, layer_size_bytes).
    """
    sizes = []
    offsets = []
    layer_sizes = []
    current_offset = 0

    for shape in shapes:
        block_elements = shape[1] * shape[2]
        num_elements = shape[0] * block_elements
        layer_elements = num_elements

        layer_sizes.append(layer_elements)
        sizes.append(num_elements)
        offsets.append(current_offset)
        current_offset += num_elements * element_size

    return sizes, offsets, layer_sizes, current_offset
