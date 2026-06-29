# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

"""
Add MomeManager class and update spec_manager_map for Pangu V2 hybrid attention.
"""

from vllm.v1.kv_cache_interface import KVCacheSpec, SinkFullAttentionSpec
from vllm.v1.core import single_type_kv_cache_manager
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashList, KVCacheBlock, BlockHashListWithBlockSize

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

from .patch_kv_cache_interface import (
    MomeSpec,
    DSAAttentionSpec,
    ShareKVSlidingWindowSpec,
    SinkMLAAttentionSpec
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec
)
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator


# MomeManager class
class MomeManager(SingleTypeKVCacheManager):
    """
    Manager for Mome KV cache (similar to Mamba).

    In each block, it always stores representations of k tokens.
    """

    def __init__(
        self,
        kv_cache_spec: MomeSpec,
        block_pool: BlockPool,
        enable_caching: bool = False,
        kv_cache_group_id: int = 0,
        **kwargs
    ) -> None:
        SingleTypeKVCacheManager.__init__(
            self,
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            **kwargs
        )
        self.kernel_size = kv_cache_spec.kernel_size
        self.num_extra_reserved_blocks = kv_cache_spec.num_extra_reserved_blocks

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list, ...]:
        """
        For prefix caching, Mome is similar to Mamba. It searches backwards
        until ONE block is hit, and immediately returns that block.
        """

        assert isinstance(kv_cache_spec, MomeSpec), (
            "MomeManager can only be used for mome groups"
        )
        assert dcp_world_size == 1, "DCP not support mome now."
        assert pcp_world_size == 1, "PCP not support mome now."
        computed_blocks: tuple[list, ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break  # we just need the last match - early stopping

        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Mome computes convolution with respect to previous `kernel_size - 1`
        tokens, so it's very similar to sliding window in KV alloc/free,
        with window_size = kernel_size.
        """
        num_retained = self.kernel_size - 1 \
                + self.num_extra_reserved_blocks * self.block_size
        return max(0, num_computed_tokens - num_retained)


class ShareKVSlidingWindowManager(SlidingWindowManager):
    def __init__(
        self,
        kv_cache_spec: ShareKVSlidingWindowSpec,
        block_pool: BlockPool,
        **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.num_extra_reserved_blocks = kv_cache_spec.num_extra_reserved_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        num_retained = self.sliding_window - 1 \
                + self.num_extra_reserved_blocks * self.block_size
        return max(0, num_computed_tokens - num_retained)


class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)


@register_patch("HybridKVCacheCoordinatorPatch", HybridKVCacheCoordinator)
class HybridKVCacheCoordinatorPatch(VLLMPatch):
    
    _attr_names_to_apply = ["find_longest_cache_hit"]
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, kv_cache_spec.block_size
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        while True:
            curr_hit_length = hit_length

            for spec, group_ids, manager_cls in self.attention_groups:
                is_full_attn = isinstance(spec, FullAttentionSpec)

                # Full attention: reuse cached blocks (downward-closed property)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if is_full_attn and cached_blocks is not None:
                    # For full attention, we only need to compute the cache hit
                    # length once. Starting from the second iteration, if the
                    # curr_hit_length is reduced by other groups, we can simply
                    # keep the first (curr_hit_length // block_size) blocks from
                    # the last iteration.
                    num_blocks = curr_hit_length // spec.block_size
                    curr_hit_length = num_blocks * spec.block_size
                    for group_id in group_ids:
                        blocks = hit_blocks_by_group[group_id]
                        assert blocks is not None
                        del blocks[num_blocks:]
                else:
                    hit_blocks = manager_cls.find_longest_cache_hit(
                        block_hashes=_get_block_hashes(spec),
                        max_length=curr_hit_length,
                        kv_cache_group_ids=group_ids,
                        block_pool=self.block_pool,
                        kv_cache_spec=spec,
                        use_eagle=self.use_eagle and is_full_attn, # patch here, for apply apc with mtp
                        alignment_tokens=self.lcm_block_size,
                    )
                    curr_hit_length = len(hit_blocks[0]) * spec.block_size
                    for group_id, blocks in zip(group_ids, hit_blocks):
                        hit_blocks_by_group[group_id] = blocks

            if curr_hit_length < hit_length:
                hit_length = curr_hit_length
            else:
                break

        return tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        ), hit_length


# Update spec_manager_map and add MomeManager to the module
original_spec_manager_map = dict(single_type_kv_cache_manager.spec_manager_map)
original_spec_manager_map[DSAAttentionSpec] = FullAttentionManager
original_spec_manager_map[ShareKVSlidingWindowSpec] = ShareKVSlidingWindowManager
original_spec_manager_map[MomeSpec] = MomeManager
original_spec_manager_map[SinkMLAAttentionSpec] = SinkFullAttentionManager


# Create a patch to update spec_manager_map
@register_patch("SingleTypeKVCacheManagerPatch", single_type_kv_cache_manager)
class SingleTypeKVCacheManagerPatch(VLLMPatch):
    """Patch to add MomeManager and update spec_manager_map"""

    _attr_names_to_apply = ["spec_manager_map", "MomeManager", "ShareKVSlidingWindowManager"]

    # Patch start
    spec_manager_map: dict = original_spec_manager_map
    MomeManager: type = MomeManager
    ShareKVSlidingWindowManager: type = ShareKVSlidingWindowManager
    # patch end
