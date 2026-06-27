# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Utility functions for block table operations in gather selection.

This module provides utility functions for working with block tables,
including conversion to NumPy arrays and buffer management.
"""

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache


def get_block_table_np(
    runner,
    omni_cache: "DecodeOmniCache",
    blk_table,
    kv_cache_gid: int,
    blk_table_tensor,
    num_reqs_padded: int,
    num_tokens_padded: int
) -> Tuple:
    """获取块表的 NumPy 表示。

    Args:
        runner: Model runner 实例
        omni_cache: DecodeOmniCache 实例
        blk_table: 块表对象
        kv_cache_gid: KV cache 组 ID
        blk_table_tensor: 块表张量
        num_reqs_padded: 填充后的请求数量
        num_tokens_padded: 填充后的 token 数量

    Returns:
        Tuple of (blk_table_np, slot_mapping_np)
    """
    import torch
    from omni_cache.cache.decode.static_utils import torch_to_numpy_zero_copy

    if not hasattr(omni_cache, 'blk_table_buffers'):
        omni_cache.blk_table_buffers = {}

    if kv_cache_gid not in omni_cache.blk_table_buffers:
        omni_cache.blk_table_buffers[kv_cache_gid] = torch.empty(
            (runner.vllm_config.scheduler_config.max_num_reqs,
             blk_table_tensor.shape[1]),
            dtype=torch.int32,
            pin_memory=True
        )

    buf = torch_to_numpy_zero_copy(omni_cache.blk_table_buffers[kv_cache_gid])
    buf[:num_reqs_padded] = blk_table.get_numpy_array()[:num_reqs_padded]
    blk_table_np = buf[:num_reqs_padded]

    if not hasattr(omni_cache, 'slot_mapping_buffers'):
        omni_cache.slot_mapping_buffers = {}

    if kv_cache_gid not in omni_cache.slot_mapping_buffers:
        max_len = runner.vllm_config.scheduler_config.max_num_seqs * (omni_cache.num_spec_token + 1)
        omni_cache.slot_mapping_buffers[kv_cache_gid] = torch.empty(
            max_len, dtype=torch.int64, pin_memory=True
        )

    buf = torch_to_numpy_zero_copy(omni_cache.slot_mapping_buffers[kv_cache_gid])
    if num_tokens_padded > buf.shape[0]:
        raise ValueError(f"num_tokens_padded({num_tokens_padded}) > buffer size({buf.shape[0]})")
    buf.fill(-1)
    slot_mapping_np = buf[:num_tokens_padded]

    return blk_table_np, slot_mapping_np


__all__ = ["get_block_table_np"]