# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import unittest
from types import SimpleNamespace

import torch

from omni_npu.attention.backends.dsa import NPUDSAMetadataBuilder
from omni_npu.v1.layers.attention.npu_pangu import _get_slot_mapping_2d


class TestGetSlotMapping2d(unittest.TestCase):
    def test_fast_path_returns_existing_slot_mapping_2d(self):
        cached = torch.tensor([[0, 0], [1, 1]])

        def _should_not_be_called(*_args, **_kwargs):
            raise AssertionError("get_slot_mapping_2d must not be called when attribute is set")

        meta = SimpleNamespace(slot_mapping_2d=cached, get_slot_mapping_2d=_should_not_be_called)
        self.assertIs(_get_slot_mapping_2d(meta), cached)

    def test_default_layer_idx_invokes_zero_arg_callback(self):
        """Mirrors MLA's lambda which accepts no arguments."""
        sentinel = torch.tensor([[1, 2]])
        meta = SimpleNamespace(slot_mapping_2d=None, get_slot_mapping_2d=lambda: sentinel)
        self.assertIs(_get_slot_mapping_2d(meta), sentinel)

    def test_explicit_layer_idx_is_passed_through(self):
        """DSA's closure expects layer_idx; ensure it gets forwarded."""
        seen = {}

        def cb(layer_idx):
            seen["layer_idx"] = layer_idx
            return torch.tensor([[3, 4]])

        meta = SimpleNamespace(slot_mapping_2d=None, get_slot_mapping_2d=cb)
        out = _get_slot_mapping_2d(meta, layer_idx=7)
        self.assertEqual(seen["layer_idx"], 7)
        self.assertTrue(torch.equal(out, torch.tensor([[3, 4]])))

    def test_returns_none_when_metadata_has_neither_attribute(self):
        meta = SimpleNamespace()
        self.assertIsNone(_get_slot_mapping_2d(meta))


class TestDSALazySlotMapping2d(unittest.TestCase):
    """Direct coverage for NPUDSAMetadataBuilder._lazy_slot_mapping_2d."""

    @staticmethod
    def _minimal_builder(block_size=16):
        b = NPUDSAMetadataBuilder.__new__(NPUDSAMetadataBuilder)
        b.kv_cache_spec = SimpleNamespace(block_size=block_size)
        return b

    @staticmethod
    def _metadata(slots, first_layer_idx=-1):
        class Metadata(SimpleNamespace):
            pass

        return Metadata(
            slot_mapping=slots,
            first_layer_idx=first_layer_idx,
            slot_mapping_cache=None,
        )

    def test_default_layer_idx_recomputes_into_cache(self):
        # MLA's zero-arg lambda path: layer_idx defaults to -1, closure must
        # recompute and write through the cache; returned tensor is the cache.
        b = self._minimal_builder(block_size=16)
        meta = self._metadata(torch.tensor([0, 5, 16, 17], dtype=torch.long))
        inner = b._lazy_slot_mapping_2d(meta)
        out = inner()
        expect = torch.stack([meta.slot_mapping // 16, meta.slot_mapping % 16], dim=-1)
        self.assertTrue(torch.equal(out, expect))
        self.assertIs(out, meta.slot_mapping_cache)

    def test_first_layer_idx_populates_cache(self):
        # DSA first-layer path: writes cache and returns it.
        b = self._minimal_builder(block_size=16)
        meta = self._metadata(torch.tensor([0, 5, 16, 17], dtype=torch.long), first_layer_idx=0)
        inner = b._lazy_slot_mapping_2d(meta)
        out = inner(0)
        expect = torch.stack([meta.slot_mapping // 16, meta.slot_mapping % 16], dim=-1)
        self.assertTrue(torch.equal(out, expect))
        self.assertIs(out, meta.slot_mapping_cache)

    def test_non_first_layer_idx_returns_existing_cache(self):
        # Non-first DSA layer reads back the cache populated by the first layer.
        b = self._minimal_builder(block_size=16)
        meta = self._metadata(torch.tensor([0, 5, 16, 17], dtype=torch.long), first_layer_idx=0)
        inner = b._lazy_slot_mapping_2d(meta)
        inner(0)  # populate cache
        seeded = meta.slot_mapping_cache
        out = inner(3)
        self.assertIs(out, seeded)


if __name__ == "__main__":
    unittest.main()
