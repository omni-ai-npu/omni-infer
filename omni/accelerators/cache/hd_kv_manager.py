# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import os
from collections import defaultdict
from dataclasses import dataclass
from typing_extensions import override

from vllm.logger import init_logger
from vllm.utils import sha256, cdiv
from vllm.config import get_layers_from_vllm_config
from vllm.attention import AttentionType
from vllm.attention.layer import Attention
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request
from .kv_cache_manager import get_manager_for_kv_cache_spec, OmniKVCacheManager, OmniKVCacheBlocks
from .omni_cache import PrefillOmniCache, DecodeOmniCache


logger = init_logger("vllm.v1.omni")


class HostDeviceKVCacheManager(OmniKVCacheManager):

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        assert len(kv_cache_config.kv_cache_groups) == 1, (
            "HostDeviceKVCacheManager does not support hybrid models with more than 1 "
            "kv cache group")

        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = kv_cache_spec.block_size
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        # calculate number of blocks for host cache
        num_layers = len(kv_cache_config.kv_cache_groups[0].layer_names)
        num_kv_heads = kv_cache_spec.num_kv_heads
        head_size = kv_cache_spec.head_size
        dtype = kv_cache_spec.dtype

        if os.getenv("ROLE") == "prefill":
            num_host_blocks = PrefillOmniCache.calc_cache_shape_for_prefill(
                num_layers=num_layers,
                block_size=self.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype
            )[1]
            self.block_pools: list[BlockPool] = [
                BlockPool(num_host_blocks, enable_caching, enable_kv_cache_events),
            ]
            logger.warning(f"**HostDeviceKVCacheManager**: For prefill, {num_host_blocks} blocks are available for host cache.")
        else:
            num_host_blocks = DecodeOmniCache.calc_cache_shape_for_decode(
                num_layers=num_layers,
                block_size=self.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype
            )[1]
            self.block_pools: list[BlockPool] = [
                BlockPool(self.num_gpu_blocks, enable_caching, enable_kv_cache_events),
                BlockPool(num_host_blocks, enable_caching, enable_kv_cache_events),
            ]
            logger.warning(f"**HostDeviceKVCacheManager**: For decode, {num_host_blocks} blocks are available for host cache and {self.num_gpu_blocks} blocks for device cache.")

        self.hybrid_managers: list[SingleTypeKVCacheManager] = []
        for block_pool in self.block_pools:
            self.hybrid_managers.append(
                get_manager_for_kv_cache_spec(
                    kv_cache_spec=kv_cache_spec,
                    use_eagle=self.use_eagle,
                    num_kv_cache_groups=1,
                    caching_hash_fn=self.caching_hash_fn,
                    block_pool=block_pool,
                )
            )

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHashType]] = defaultdict(list)

    # @override
    # def get_computed_blocks(self,
    #                         request: Request) -> tuple[OmniKVCacheBlocks, int]:
    #     kv_cache_blocks, num_computed_tokens = super().get_computed_blocks(request)
    #     blocks: list[KVCacheBlock] = kv_cache_blocks.blocks
    #     multi_group_blocks = OmniKVCacheBlocks([blocks for _ in range(len(self.hybrid_managers))])
    #     return multi_group_blocks, num_computed_tokens
