# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Keep hybrid APC hits on a prefix every KV-cache group can serve.

vLLM 0.25.1's hybrid fixed-point can report a length that Full Attention does
not hold: with MTP it drops the last FA block, Mome/Mamba then revives a longer
hit, and ``is_simple_hybrid`` exits before FA is capped. The connector path
makes it worse — the scheduler uses ``max(per_group_hits)`` as the common
prefix, which ``LLMDataDistConnector`` cannot fill in.

This patch keeps the stock loop, with two changes:

1. do not early-exit after the first simple-hybrid iteration;
2. when FA is skipped as downward-closed, cap ``curr_hit_length`` to the
   tokens those blocks actually hold, so a later Mome revive is pulled back.

``find_longest_cache_hit_per_group`` then just repeats that common length.

TODO: remove once vLLM tracks a per-group hit length in the fixed-point loop
(upstream #50344; v0.27.2+).
"""

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def find_longest_cache_hit(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Stock hybrid lookup, with FA held-token cap and no simple-hybrid exit."""

    def _get_block_hashes(kv_cache_spec: KVCacheSpec):
        if kv_cache_spec.block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes, self.hash_block_size, kv_cache_spec.block_size
        )

    num_groups = len(self.kv_cache_config.kv_cache_groups)
    hit_length = max_cache_hit_length
    longest_hit_length = 0
    hit_blocks_by_group = [None] * num_groups
    eagle_verified: set[int] = set()

    while True:
        curr_hit_length = hit_length

        for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(
            self.attention_groups
        ):
            cached_blocks = hit_blocks_by_group[group_ids[0]]
            if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                ### adapt start
                # Cap to tokens FA actually holds. Stock only aligns curr_hit_length,
                # so a later Mome revive can leave hit_length past these blocks.
                curr_hit_length = min(
                    curr_hit_length,
                    len(cached_blocks) * spec.block_size,
                )
                ### adapt end
                curr_hit_length = (
                    curr_hit_length // spec.block_size * spec.block_size
                )
                continue

            drop_eagle_block = use_eagle and idx not in eagle_verified
            _max_length = curr_hit_length
            if drop_eagle_block:
                _max_length = min(
                    curr_hit_length + spec.block_size, max_cache_hit_length
                )
            hit_blocks = manager_cls.find_longest_cache_hit(
                block_hashes=_get_block_hashes(spec),
                max_length=_max_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=drop_eagle_block,
                alignment_tokens=self.scheduler_block_size,
            )
            _new_hit_length = len(hit_blocks[0]) * spec.block_size
            if drop_eagle_block:
                eagle_verified.add(idx)
            elif _new_hit_length < curr_hit_length:
                eagle_verified.clear()
            curr_hit_length = _new_hit_length
            for group_id, blocks in zip(group_ids, hit_blocks):
                hit_blocks_by_group[group_id] = blocks
            longest_hit_length = max(longest_hit_length, curr_hit_length)

        if curr_hit_length >= hit_length:
            break
        hit_length = curr_hit_length
        # adapt start
        # Removed stock `if is_simple_hybrid: break`. FA+Mome is 2 groups, so
        # that exit skipped the second pass where the FA cap above takes effect.
        # adapt end

    first_group = self.attention_groups[0]
    if isinstance(first_group.spec, FullAttentionSpec):
        num_blocks = hit_length // first_group.spec.block_size
        for group_id in first_group.group_ids:
            if (blks := hit_blocks_by_group[group_id]) is not None:
                del blks[num_blocks:]

    self.num_uncached_common_prefix_tokens = longest_hit_length - hit_length
    return tuple(
        blocks if blocks is not None else [] for blocks in hit_blocks_by_group
    ), hit_length


def find_longest_cache_hit_per_group(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Return the common hybrid hit as a per-group tuple for the scheduler."""
    ### adapt start
    # Stock looks up each group independently; the scheduler then uses
    # max(per_group_hits). Repeat the common length instead.
    blocks, hit_length = find_longest_cache_hit(
        self, block_hashes, max_cache_hit_length
    )
    return blocks, (hit_length,) * len(blocks)
    ### adapt end


@register_patch("HybridAPCConnectorHitPatch", HybridKVCacheCoordinator)
class HybridAPCConnectorHitPatch(VLLMPatch):
    _attr_names_to_apply = [
        "find_longest_cache_hit",
        "find_longest_cache_hit_per_group",
    ]

    find_longest_cache_hit = find_longest_cache_hit
    find_longest_cache_hit_per_group = find_longest_cache_hit_per_group
