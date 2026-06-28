# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Selection buffer management for OmniCache.

This module provides the SelectionBuffers class which manages all
tensor buffers related to gather selection operations, including
buffer initialization and top-k index pre-computation.
"""

import os
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache


class SelectionBuffers:
    """管理 Selection 相关的缓冲区。

    负责初始化和管理 gather selection 操作所需的所有张量缓冲区，
    包括 KV cache、块表、状态缓冲区等。
    """

    def __init__(self, omni_cache: "DecodeOmniCache"):
        """Initialize selection buffers for the given cache instance.

        Args:
            omni_cache: The DecodeOmniCache instance to initialize buffers for.
        """
        self.omni_cache = omni_cache
        self._initialize_buffers()

    def _initialize_buffers(self) -> None:
        """初始化所有 selection 相关的缓冲区。

        分配和初始化 gather selection 操作所需的全部张量缓冲区，
        包括 KV cache、块表、状态缓冲区以及预计算的 top-k 索引。
        """
        import custom_ops

        omni_cache = self.omni_cache

        # Resolve a DSA layer name for checking per-layer component count.
        dsa_layer = None
        for group in omni_cache.kv_cache_config.kv_cache_groups:
            from vllm.v1.kv_cache_interface import DSAAttentionSpec

            if isinstance(group.kv_cache_spec, DSAAttentionSpec):
                dsa_layer = group.layer_names[0]
                break

        # 计算参数
        batch_size = omni_cache.runner.max_num_reqs

        headnum = 1
        s_block_size = omni_cache.block_size
        k_rope_sz = (
            omni_cache.k_rope_size if dsa_layer is not None and len(omni_cache.host_cache_pa[dsa_layer]) == 3 else 1
        )
        kvcache_sz = omni_cache.kvcache_size

        s_max_block_num = omni_cache.s_max_block_num
        state_size = omni_cache.selection_state_size

        # 分配张量缓冲区
        omni_cache.selection_k_rope = torch.zeros(
            [omni_cache.num_layers, s_max_block_num * batch_size * headnum, s_block_size, k_rope_sz],
            dtype=torch.bfloat16,
            device=omni_cache.device,
        ).contiguous()

        omni_cache.selection_kv_cache = torch.zeros(
            [omni_cache.num_layers, s_max_block_num * batch_size * headnum, s_block_size, kvcache_sz],
            dtype=torch.bfloat16,
            device=omni_cache.device,
        ).contiguous()

        if not int(os.getenv("USE_OMNI_INPUT_BATCH", "0")):
            omni_cache.selection_kv_block_table = (
                torch.arange(
                    batch_size * headnum * s_max_block_num,
                    dtype=torch.int32,
                    device=omni_cache.device,
                )
                .view(batch_size, headnum * s_max_block_num)
                .contiguous()
            )

            omni_cache.selection_kv_block_status = -torch.ones(
                [omni_cache.num_layers, batch_size, state_size],
                dtype=torch.int32,
                device=omni_cache.device,
            ).contiguous()

            # 工作缓冲区
            omni_cache.selection_kv_block_table_buffer = torch.empty(
                [batch_size, s_max_block_num],
                dtype=torch.int32,
                device=omni_cache.device,
            ).contiguous()

            omni_cache.selection_kv_block_status_buffer = (
                torch.empty(
                    [omni_cache.num_layers, batch_size, state_size],
                    dtype=torch.int32,
                    device=omni_cache.device,
                )
                .contiguous()
                .view(omni_cache.num_layers, batch_size, state_size)
            )

        omni_cache.index_buffer = torch.empty(batch_size, dtype=torch.long, device=omni_cache.device)

        omni_cache.reuse_rate = torch.zeros(omni_cache.num_layers, dtype=torch.float64, device=omni_cache.device)

        omni_cache.record_smooth_alpha = 0.7


def initialize_selection_buffers(omni_cache: "DecodeOmniCache") -> None:
    """初始化 selection 缓冲区的便捷函数。

    Args:
        omni_cache: The DecodeOmniCache instance to initialize buffers for.
    """
    SelectionBuffers(omni_cache)


__all__ = ["SelectionBuffers", "initialize_selection_buffers"]
