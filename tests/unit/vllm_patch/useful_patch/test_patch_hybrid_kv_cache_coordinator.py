# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for hybrid APC hit reconciliation on vLLM 0.25.1.

The file can run directly on a development machine without installing
``omni_npu``::

    python tests/unit/vllm_patch/useful_patch/\
        test_patch_hybrid_kv_cache_coordinator.py
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PATCH_PATH = (
    REPO_ROOT
    / "omni/vllm_patches/usefull_patch/patch_hybrid_kv_cache_coordinator.py"
)


def _load_patch_module():
    try:
        from omni_npu.vllm_patches.usefull_patch import (
            patch_hybrid_kv_cache_coordinator as mod,
        )

        return mod
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

    class HybridKVCacheCoordinator:
        @staticmethod
        def find_longest_cache_hit_per_group(*_args, **_kwargs):
            raise NotImplementedError

    put("vllm", is_package=True)
    put("vllm.v1", is_package=True)
    put("vllm.v1.core", is_package=True)
    put(
        "vllm.v1.core.kv_cache_coordinator",
        HybridKVCacheCoordinator=HybridKVCacheCoordinator,
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
        spec = importlib.util.spec_from_file_location(
            "_hybrid_apc_connector_patch_under_test", PATCH_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, saved in reversed(saved_modules.items()):
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


patch_mod = _load_patch_module()


class FakeCoordinator:
    def __init__(self, common_blocks=None, common_hit=128):
        if common_blocks is None:
            common_blocks = (["fa-common"], ["mome-common"])
        self.common_blocks = common_blocks
        self.common_hit = common_hit
        self.common_calls = 0
        self.common_args = None
        self.num_uncached_common_prefix_tokens = 0

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        self.common_calls += 1
        self.common_args = (block_hashes, max_cache_hit_length)
        self.num_uncached_common_prefix_tokens = 64
        return self.common_blocks, self.common_hit


class HybridAPCConnectorHitPatchTest(unittest.TestCase):
    @staticmethod
    def _call(coordinator, block_hashes=None, max_cache_hit_length=256):
        if block_hashes is None:
            block_hashes = ["h0", "h1"]
        return patch_mod.find_longest_cache_hit_per_group(
            coordinator, block_hashes, max_cache_hit_length
        )

    def test_uses_common_lookup_directly(self):
        coordinator = FakeCoordinator()

        blocks, hit_lengths = self._call(coordinator)

        self.assertEqual(blocks, coordinator.common_blocks)
        self.assertEqual(hit_lengths, (128, 128))
        self.assertEqual(coordinator.common_calls, 1)

    def test_repeats_common_hit_for_every_group(self):
        common_blocks = (["g0"], ["g1"], ["g2"])
        coordinator = FakeCoordinator(common_blocks=common_blocks, common_hit=64)

        blocks, hit_lengths = self._call(coordinator)

        self.assertEqual(blocks, common_blocks)
        self.assertEqual(hit_lengths, (64, 64, 64))

    def test_zero_common_hit_is_preserved(self):
        coordinator = FakeCoordinator(common_blocks=([], []), common_hit=0)

        blocks, hit_lengths = self._call(coordinator)

        self.assertEqual(blocks, ([], []))
        self.assertEqual(hit_lengths, (0, 0))

    def test_forwards_lookup_arguments(self):
        coordinator = FakeCoordinator()

        self._call(
            coordinator,
            block_hashes=["hash0", "hash1", "hash2"],
            max_cache_hit_length=384,
        )

        self.assertEqual(
            coordinator.common_args,
            (["hash0", "hash1", "hash2"], 384),
        )

    def test_patch_targets_active_hybrid_coordinator(self):
        self.assertIs(
            patch_mod.HybridAPCConnectorHitPatch._target,
            patch_mod.HybridKVCacheCoordinator,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
