# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Repeat the common hybrid APC hit length for every KV-cache group.

``find_longest_cache_hit_per_group`` repeats that common length for the
scheduler connector path.

TODO: remove once vLLM tracks a per-group hit length in the fixed-point loop
(upstream #50344; v0.27.2+).
"""

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def find_longest_cache_hit_per_group(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Return the common hybrid hit as a per-group tuple for the scheduler."""
    # adapt start
    # Stock looks up each group independently; the scheduler then uses
    # max(per_group_hits). Repeat the common length instead.
    blocks, hit_length = self.find_longest_cache_hit(
        block_hashes, max_cache_hit_length
    )
    return blocks, (hit_length,) * len(blocks)
    # adapt end


@register_patch("HybridAPCConnectorHitPatch", HybridKVCacheCoordinator)
class HybridAPCConnectorHitPatch(VLLMPatch):
    _attr_names_to_apply = ["find_longest_cache_hit_per_group"]

    find_longest_cache_hit_per_group = find_longest_cache_hit_per_group
