# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Keep hybrid APC hits consistent when using the Omni KV connector.

vLLM 0.25.1 handles a hybrid model plus a KV connector by looking up every
KV-cache group independently and treating ``max(per_group_hits)`` as the
locally computed prefix for *all* groups.  That is only valid for connectors
which can restore holes in a lagging group.  ``LLMDataDistConnector`` does not
provide that contract (and its producer side never loads external blocks), so
resuming at the maximum can consume missing Full-Attention or MoME state and
silently change model output.

For the pinned vLLM 0.25.1 integration, always use the existing hybrid
fixed-point/common lookup for this connector entry point.  APC remains enabled
and the largest prefix valid for every cache group is still reused.

TODO: Remove this patch when upgrading to vLLM v0.27.2rc0 or any later release
containing upstream fix #50344.  For production, target v0.27.2 or later.
"""

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def find_longest_cache_hit_per_group(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Return the largest prefix valid for every hybrid cache group."""
    common_result = self.find_longest_cache_hit(
        block_hashes, max_cache_hit_length
    )
    common_blocks, common_hit_length = common_result[:2]

    return common_blocks, (common_hit_length,) * len(common_blocks)


@register_patch("HybridAPCConnectorHitPatch", HybridKVCacheCoordinator)
class HybridAPCConnectorHitPatch(VLLMPatch):
    _attr_names_to_apply = ["find_longest_cache_hit_per_group"]

    find_longest_cache_hit_per_group = find_longest_cache_hit_per_group
