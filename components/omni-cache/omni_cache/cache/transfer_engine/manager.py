# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Transfer manager for unified H2D/D2H operations.

This module provides TransferManager, a unified manager for H2D (Host-to-Device)
and D2H (Device-to-Host) transfer operations for both decode and prefill caches.
"""

from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache
    from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache
    from vllm.v1.kv_cache_interface import KVCacheConfig


class TransferManager:
    """Unified manager for H2D/D2H transfer operations.

    This class manages all host-device transfer operations, including:
    - Buffer allocation and management
    - Stream creation and synchronization
    - Async operation coordination
    - Thread pool management

    Args:
        cache: The parent cache instance (PrefillOmniCache or DecodeOmniCache)
        max_num_batched_tokens: Maximum number of batched tokens
        max_num_seqs: Maximum number of sequences
    """

    def __init__(self, cache, max_num_batched_tokens: Optional[int] = None, max_num_seqs: Optional[int] = None):
        from .buffers import TransferBuffers, StreamManager, ThreadPoolManager

        self.cache = cache
        self.buffers = TransferBuffers(cache)
        self.streams = StreamManager(cache.device)
        self.thread_pool = ThreadPoolManager()

    def initialize_prefill(
        self,
        kv_cache_config: "KVCacheConfig",
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_model_len: int
    ) -> None:
        """Initialize all resources for prefill operations.

        This replaced (methods removed):
        - prefill_omni_cache.py:_init_cpu_buffers()
        - prefill_omni_cache.py:_init_token_indices()
        - prefill_omni_cache.py:_init_streams_and_pools()
        - prefill_omni_cache.py:_init_prefix_buffer() (partial, streams only)

        Args:
            kv_cache_config: KV cache configuration
            max_num_batched_tokens: Maximum number of batched tokens
            max_num_seqs: Maximum number of sequences
            max_model_len: Maximum model length
        """
        self.thread_pool.initialize(self.cache, max_workers=1)

        # Initialize CPU buffers (now N-staged off num_stages_layer_copy)
        self.buffers.initialize_cpu_buffers(kv_cache_config, max_num_batched_tokens, max_model_len)

        # Initialize token indices
        self.buffers.initialize_token_indices(max_num_batched_tokens)

        # Initialize streams
        self.streams.initialize_prefill_streams(self.cache)

    def initialize_decode(self) -> None:
        """Initialize all resources for decode operations.

        This replaces decode_omni_cache.py:__init__ (H2D related parts)
        """
        # Initialize streams
        self.streams.initialize_decode_streams(self.cache)

    def sync_h2d_prefill(self, prefix_meta, attn_names: list[str]):
        """Perform H2D synchronization for prefill.

        Args:
            prefix_meta: PrefixCopyMeta for chunked prefill
            attn_names: List of layer names to transfer

        Returns:
            None
        """
        from .synchronize import synchronize_h2d_prefill
        return synchronize_h2d_prefill(self.cache, prefix_meta, attn_names)

    def sync_d2h_prefill(self, attn_names: list[str], attn_metadatas: list[str], kv_event: torch.npu.Event):
        """Perform D2H synchronization for prefill.

        Args:
            attn_names: List of layer names
            attn_metadatas: List of attention metadata objects
            kv_event: Event signaling KV computation completion

        Returns:
            None
        """
        from .synchronize import synchronize_d2h_prefill
        return synchronize_d2h_prefill(self.cache, attn_names, attn_metadatas, kv_event)

    def sync_d2h_hybrid(self, layer_name_list: List[str], attn_metadata_list: List, kv_event: torch.npu.Event):
        """Perform D2H synchronization for hybrid attention.

        Args:
            layer_name_list: List of layer names
            attn_metadata_list: List of attention metadata
            kv_event: Event signaling KV computation completion

        Returns:
            None
        """
        from .synchronize import synchronize_d2h_hybrid
        return synchronize_d2h_hybrid(self.cache, layer_name_list, attn_metadata_list, kv_event)

    def sync_h2d_decode(self, batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes):
        """Perform H2D synchronization for decode.

        Args:
            batch_device_mem: Device memory batch
            batch_device_max: Device memory max sizes
            batch_host_mem: Host memory batch
            batch_host_sizes: Host memory sizes

        Returns:
            None
        """
        from .synchronize import synchronize_h2d_decode
        return synchronize_h2d_decode(self.cache, batch_device_mem, batch_device_max,
                                      batch_host_mem, batch_host_sizes)
