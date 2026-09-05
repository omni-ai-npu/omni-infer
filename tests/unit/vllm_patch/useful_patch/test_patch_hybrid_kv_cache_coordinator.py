# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the hybrid APC coordinator patch on vLLM 0.25.1.

The stock simple-hybrid loop lets Mome revive a length past the EAGLE-dropped
Full Attention prefix. The patch caps FA to the blocks it holds and keeps
iterating; ``per_group`` repeats that one length.

The file can run without installing ``omni_npu``::

    python tests/unit/vllm_patch/useful_patch/\
        test_patch_hybrid_kv_cache_coordinator.py
"""

import importlib.util
import sys
import types
import unittest
from collections import namedtuple
from pathlib import Path

_SpecGroup = namedtuple(
    "SpecGroup", ("spec", "group_ids", "manager_cls", "use_eagle")
)


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_PATCH_PATH = (
    REPO_ROOT
    / "omni/vllm_patches/usefull_patch/models/pangu_v2_base/patch_hybrid_kv_cache_coordinator.py"
)
HYBRID_PATCH_PATH = (
    REPO_ROOT
    / "omni/vllm_patches/usefull_patch/models/high_throughout/patch_hybrid_kv_cache_coordinator.py"
)

NULL_BLOCK = "NULL"
BLOCK_SIZE = 16


def _exec_patch(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_patch_modules():
    try:
        from omni_npu.vllm_patches.usefull_patch.models.pangu_v2_base import (
            patch_hybrid_kv_cache_coordinator as loaded_base,
        )
        from omni_npu.vllm_patches.usefull_patch.models.high_throughout import (
            patch_hybrid_kv_cache_coordinator as loaded_hybrid,
        )

        return loaded_base, loaded_hybrid
    except Exception:  # noqa: BLE001 - use a direct source load on dev machines
        pass

    saved_modules = {}

    def put(name, *, is_package=False, **attrs):
        saved_modules[name] = sys.modules.get(name)
        module = types.ModuleType(name)
        if is_package:
            module.__path__ = []
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules[name] = module
        return module

    class _StubFullAttentionSpec:
        pass

    class KVCacheSpec:
        pass

    class HybridKVCacheCoordinator:
        pass

    class BlockHashListWithBlockSize:
        def __init__(self, block_hashes, hash_block_size, block_size):
            self.block_hashes = block_hashes
            self.scale = block_size // hash_block_size

        def __len__(self):
            return len(self.block_hashes) // self.scale

        def __getitem__(self, index):
            return self.block_hashes[(index + 1) * self.scale - 1]

    put("vllm", is_package=True)
    put("vllm.v1", is_package=True)
    put("vllm.v1.core", is_package=True)
    put(
        "vllm.v1.core.kv_cache_coordinator",
        HybridKVCacheCoordinator=HybridKVCacheCoordinator,
    )
    put(
        "vllm.v1.core.kv_cache_utils",
        BlockHashListWithBlockSize=BlockHashListWithBlockSize,
    )
    put(
        "vllm.v1.kv_cache_interface",
        FullAttentionSpec=_StubFullAttentionSpec,
        KVCacheSpec=KVCacheSpec,
    )

    core = types.ModuleType("omni_npu.vllm_patches.core")

    class VLLMPatch:
        pass

    def register_patch(_name, target):
        def decorate(cls):
            cls._target = target
            return cls

        return decorate

    core.VLLMPatch = VLLMPatch
    core.register_patch = register_patch

    saved_modules["omni_npu.vllm_patches.core"] = sys.modules.get(
        "omni_npu.vllm_patches.core"
    )
    sys.modules["omni_npu.vllm_patches.core"] = core
    try:
        loaded_base = _exec_patch(
            BASE_PATCH_PATH, "_hybrid_apc_connector_patch_under_test"
        )
        loaded_hybrid = _exec_patch(
            HYBRID_PATCH_PATH, "_hybrid_apc_find_longest_hit_patch_under_test"
        )
        return loaded_base, loaded_hybrid
    finally:
        for name, saved in reversed(saved_modules.items()):
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


base_mod, hybrid_mod = _load_patch_modules()
FullAttentionSpec = hybrid_mod.FullAttentionSpec


class FakeBlockPool:
    def __init__(self, cached_per_group):
        self.cached_per_group = cached_per_group
        self.null_block = NULL_BLOCK

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        index = int(block_hash[1:])
        blocks = []
        for gid in kv_cache_group_ids:
            if index not in self.cached_per_group[gid]:
                return None
            blocks.append(f"g{gid}b{index}")
        return blocks


class _CacheHitManager:
    _scan_reverse = False

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
    ):
        computed = tuple([] for _ in kv_cache_group_ids)
        return cls._scan_hits(
            computed,
            block_hashes,
            max_length,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
        )

    @classmethod
    def _scan_hits(
        cls,
        computed,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
    ):
        block_count = max_length // kv_cache_spec.block_size
        indices = (
            range(block_count - 1, -1, -1) if cls._scan_reverse else range(block_count)
        )
        for index in indices:
            cached = block_pool.get_cached_block(
                block_hashes[index], kv_cache_group_ids
            )
            if cached is None:
                if cls._scan_reverse:
                    continue
                break
            for blocks, block in zip(computed, cached):
                if cls._scan_reverse:
                    blocks.extend([block_pool.null_block] * index)
                blocks.append(block)
            if cls._scan_reverse:
                break
        if drop_eagle_block and computed[0] and not cls._scan_reverse:
            for blocks in computed:
                blocks.pop()
        return computed


class FullManager(_CacheHitManager):
    pass


class MomeManager(_CacheHitManager):
    _scan_reverse = True


class _FASpec(FullAttentionSpec):
    """Avoid the frozen dataclass ``__init__`` when the real vLLM spec is loaded."""

    def __init__(self, block_size=BLOCK_SIZE):
        object.__setattr__(self, "block_size", block_size)


class _OtherSpec:
    def __init__(self, block_size=BLOCK_SIZE):
        self.block_size = block_size


class FakeCoordinator:
    """``groups`` is ``(spec, manager_cls, cached_indices, use_eagle)``."""

    def __init__(self, groups, block_size=BLOCK_SIZE, num_hashes=16):
        self.hash_block_size = block_size
        self.scheduler_block_size = block_size
        self.num_uncached_common_prefix_tokens = 0
        self.block_hashes = [f"h{i}" for i in range(num_hashes)]
        self.block_pool = FakeBlockPool(
            [set(cached) for _spec, _manager, cached, _eagle in groups]
        )
        self.kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=[object() for _ in groups]
        )
        self.attention_groups = [
            _SpecGroup(spec, [gid], manager, use_eagle)
            for gid, (spec, manager, _cached, use_eagle) in enumerate(groups)
        ]

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        return hybrid_mod.find_longest_cache_hit(
            self, block_hashes, max_cache_hit_length
        )


class HybridAPCConnectorHitPatchTest(unittest.TestCase):
    @staticmethod
    def _call_per_group(coordinator, max_cache_hit_length=256):
        return base_mod.find_longest_cache_hit_per_group(
            coordinator, coordinator.block_hashes, max_cache_hit_length
        )

    def test_simple_hybrid_mtp_caps_to_the_fa_prefix(self):
        """FA+Mome, EAGLE on FA: Mome revives 144, the patch must report 128."""
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(9), True),
                # Pangu MTP flags every group; Mome still does not pop.
                (_OtherSpec(), MomeManager, [7, 8], True),
            ]
        )

        blocks, hit_lengths = self._call_per_group(coordinator)

        self.assertEqual(hit_lengths, (128, 128))
        self.assertEqual(blocks[0], [f"g0b{i}" for i in range(8)])
        self.assertEqual(blocks[1], [NULL_BLOCK] * 7 + ["g1b7"])
        self.assertEqual(coordinator.num_uncached_common_prefix_tokens, 16)

    def test_agreed_prefix_is_kept_without_mtp(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(8), False),
                (_OtherSpec(), MomeManager, [7], False),
            ]
        )

        blocks, hit_lengths = self._call_per_group(coordinator)

        self.assertEqual(hit_lengths, (128, 128))
        self.assertEqual(blocks[0], [f"g0b{i}" for i in range(8)])
        self.assertEqual(blocks[1], [NULL_BLOCK] * 7 + ["g1b7"])

    def test_zero_hit_is_preserved(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, [], False),
                (_OtherSpec(), MomeManager, [], False),
            ]
        )

        blocks, hit_lengths = self._call_per_group(coordinator)

        self.assertEqual(blocks, ([], []))
        self.assertEqual(hit_lengths, (0, 0))

    def test_repeats_the_common_length_for_every_group(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(4), False),
                (_OtherSpec(), MomeManager, [3], False),
                (_OtherSpec(), MomeManager, [3], False),
            ]
        )

        _blocks, hit_lengths = self._call_per_group(coordinator)

        self.assertEqual(hit_lengths, (64, 64, 64))

    def test_direct_entry_point_returns_one_length(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(8), False),
                (_OtherSpec(), MomeManager, [7], False),
            ]
        )

        blocks, hit_length = hybrid_mod.find_longest_cache_hit(
            coordinator, coordinator.block_hashes, 256
        )

        self.assertEqual(hit_length, 128)
        self.assertEqual(len(blocks), 2)

    def test_patch_targets_active_hybrid_coordinator(self):
        self.assertIs(
            base_mod.HybridAPCConnectorHitPatch._target,
            base_mod.HybridKVCacheCoordinator,
        )
        self.assertEqual(
            base_mod.HybridAPCConnectorHitPatch._attr_names_to_apply,
            ["find_longest_cache_hit_per_group"],
        )
        self.assertIs(
            hybrid_mod.HybridAPCFindLongestCacheHitPatch._target,
            hybrid_mod.HybridKVCacheCoordinator,
        )
        self.assertEqual(
            hybrid_mod.HybridAPCFindLongestCacheHitPatch._attr_names_to_apply,
            ["find_longest_cache_hit"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
