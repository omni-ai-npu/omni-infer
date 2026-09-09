# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the high_throughout hybrid APC per-group coordinator patch.

The ``find_longest_cache_hit`` FA-cap patch was removed; the scheduler still
uses ``find_longest_cache_hit_per_group``, which repeats the common hybrid hit
length for every KV-cache group.
"""

import pytest

from omni_npu.vllm_patches.patches.models.high_throughout import (
    patch_hybrid_kv_cache_coordinator as hybrid_mod,
)


class _FakeCoordinator:
    def __init__(self, blocks, hit_length):
        self._blocks = blocks
        self._hit_length = hit_length

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        return self._blocks, self._hit_length


@pytest.mark.unit
def test_per_group_repeats_the_common_hit_length():
    blocks = (["g0b0", "g0b1"], ["g1b0"])
    coordinator = _FakeCoordinator(blocks, hit_length=128)

    out_blocks, hit_lengths = hybrid_mod.find_longest_cache_hit_per_group(
        coordinator, ["h0", "h1"], 256
    )

    assert out_blocks is blocks
    assert hit_lengths == (128, 128)


@pytest.mark.unit
def test_per_group_zero_hit_is_preserved():
    coordinator = _FakeCoordinator(([], []), hit_length=0)
    blocks, hit_lengths = hybrid_mod.find_longest_cache_hit_per_group(
        coordinator, [], 256
    )
    assert blocks == ([], [])
    assert hit_lengths == (0, 0)


@pytest.mark.unit
def test_connector_patch_targets_hybrid_coordinator():
    assert hybrid_mod.HybridAPCConnectorHitPatch._target is (
        hybrid_mod.HybridKVCacheCoordinator
    )
    assert hybrid_mod.HybridAPCConnectorHitPatch._attr_names_to_apply == [
        "find_longest_cache_hit_per_group"
    ]
    assert (
        hybrid_mod.HybridAPCConnectorHitPatch.find_longest_cache_hit_per_group
        is hybrid_mod.find_longest_cache_hit_per_group
    )
