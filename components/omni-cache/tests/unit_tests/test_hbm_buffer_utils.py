# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for hbm_buffer_utils.py module.

Tests HBM buffer construction utilities for decode cache, covering:
- Helper functions: _derive_attention_head_sizes, _is_mamba_spec,
  _is_sliding_window_spec, _req_offset_for_spec
- Allocator functions: _alloc_attention, _alloc_mamba
- Main entry: construct_hbm_buffer
- All spec dispatch paths: default attention, SlidingWindow, DSA, Mamba, MomeSpec,
  ShareKVSlidingWindow
"""

import sys
import types

import pytest
import torch
from unittest.mock import Mock

from omni_cache.cache.decode.hbm_buffer_utils import (
    _derive_attention_head_sizes,
    _is_mamba_spec,
    _is_sliding_window_spec,
    _req_offset_for_spec,
    _alloc_attention,
    _alloc_mamba,
    construct_hbm_buffer,
)


def _make_default_spec_with_attrs(block_size=16, dtype=torch.float16,
                                  separate_kvscale=None, **extra_attrs):
    """Create a simple namespace-based spec that only has the attributes we set."""
    class _SimpleSpec:
        pass
    spec = _SimpleSpec()
    spec.block_size = block_size
    spec.dtype = dtype
    spec.separate_kvscale = separate_kvscale
    for k, v in extra_attrs.items():
        setattr(spec, k, v)
    return spec


# ---------------------------------------------------------------------------
# Lightweight spec stubs that mirror the real vllm / Pangu V2 class hierarchy.
# These are injected into sys.modules so that the isinstance() checks inside
# hbm_buffer_utils.py work correctly without requiring a full vllm install.
# ---------------------------------------------------------------------------

class _StubSlidingWindowSpec:
    """Minimal stub matching vllm.v1.kv_cache_interface.SlidingWindowSpec."""
    def __init__(self, block_size=16, sliding_window=256, dtype=torch.float16,
                 num_kv_heads=1, head_size=128):
        self.block_size = block_size
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size


class _StubMambaSpec:
    """Minimal stub matching vllm.v1.kv_cache_interface.MambaSpec."""
    def __init__(self, block_size=1, shapes=None, dtypes=None):
        self.block_size = block_size
        self.shapes = shapes or [64]
        self.dtypes = dtypes or [torch.float16]


class _StubDSAAttentionSpec:
    """Minimal stub matching DSAAttentionSpec."""
    def __init__(self, block_size=16, dtype=torch.float16, head_sizes=None):
        self.block_size = block_size
        self.dtype = dtype
        self.head_sizes = head_sizes  # e.g. [128] for indexer


class _StubMomeSpec(_StubMambaSpec):
    """Minimal stub matching MomeSpec (extends MambaSpec)."""
    def __init__(self, block_size=1, kernel_size=7, num_spec_tokens=1,
                 shapes=None, dtypes=None):
        super().__init__(block_size=block_size, shapes=shapes, dtypes=dtypes)
        self.kernel_size = kernel_size
        self.num_spec_tokens = num_spec_tokens

    @property
    def num_total_tokens(self):
        return self.kernel_size - 1 + self.num_spec_tokens


class _StubShareKVSlidingWindowSpec(_StubSlidingWindowSpec):
    """Minimal stub matching ShareKVSlidingWindowSpec (extends SlidingWindowSpec)."""
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inject_spec_classes():
    """Inject spec stubs into sys.modules so hbm_buffer_utils dynamic
    isinstance() imports resolve to our stubs.

    This fixture injects:
      - vllm.v1.kv_cache_interface.{SlidingWindowSpec, MambaSpec, DSAAttentionSpec}
      - omni_npu.vllm_patches...patch_kv_cache_interface.{MomeSpec, ShareKVSlidingWindowSpec, DSAAttentionSpec}
    """
    import vllm.v1.kv_cache_interface as kvif

    saved = {}
    # Save originals
    for name in ("SlidingWindowSpec", "MambaSpec", "DSAAttentionSpec"):
        saved[name] = getattr(kvif, name, None)

    # Inject into vllm.v1.kv_cache_interface
    kvif.SlidingWindowSpec = _StubSlidingWindowSpec
    kvif.MambaSpec = _StubMambaSpec
    kvif.DSAAttentionSpec = _StubDSAAttentionSpec

    # Build omni_npu patch module path
    patch_path = "omni_npu.vllm_patches.patches.models.pangu_v2_hybrid.patch_kv_cache_interface"
    if patch_path not in sys.modules:
        mod = types.ModuleType(patch_path)
        mod.__path__ = []
        sys.modules[patch_path] = mod
    else:
        mod = sys.modules[patch_path]

    saved_pangu = {}
    for name in ("MomeSpec", "ShareKVSlidingWindowSpec", "DSAAttentionSpec"):
        saved_pangu[name] = getattr(mod, name, None)

    mod.MomeSpec = _StubMomeSpec
    mod.ShareKVSlidingWindowSpec = _StubShareKVSlidingWindowSpec
    mod.DSAAttentionSpec = _StubDSAAttentionSpec

    yield

    # Restore
    for name, val in saved.items():
        if val is None:
            if hasattr(kvif, name):
                delattr(kvif, name)
        else:
            setattr(kvif, name, val)

    for name, val in saved_pangu.items():
        if val is None:
            if hasattr(mod, name):
                delattr(mod, name)
        else:
            setattr(mod, name, val)


def _make_decode_cache(specs_and_layers, *, num_max_batch_pool=4,
                       max_model_len=2048, block_bytes=16384,
                       num_blocks=100, device="cpu"):
    """Build a mock DecodeOmniCache with the given kv_cache_groups.

    Args:
        specs_and_layers: list of (spec, layer_names) tuples.
    """
    cache = Mock()
    cache.num_max_batch_pool = num_max_batch_pool
    cache.num_blocks = num_blocks
    cache.device = device
    cache.block_bytes = block_bytes
    cache.vllm_config = Mock()
    cache.vllm_config.model_config = Mock()
    cache.vllm_config.model_config.max_model_len = max_model_len

    groups = []
    for spec, layer_names in specs_and_layers:
        group = Mock()
        group.layer_names = layer_names
        group.kv_cache_spec = spec
        groups.append(group)

    config = Mock()
    config.kv_cache_groups = groups
    cache.kv_cache_config = config
    return cache


# ===================================================================
# Tests for _is_mamba_spec / _is_sliding_window_spec
# ===================================================================

class TestIsSpecHelpers:
    """Tests for _is_mamba_spec and _is_sliding_window_spec."""

    def test_mamba_spec_true(self, inject_spec_classes):
        spec = _StubMambaSpec()
        assert _is_mamba_spec(spec) is True

    def test_mome_spec_is_mamba(self, inject_spec_classes):
        """MomeSpec extends MambaSpec, so _is_mamba_spec should return True."""
        spec = _StubMomeSpec()
        assert _is_mamba_spec(spec) is True

    def test_non_mamba_returns_false(self, inject_spec_classes):
        spec = _StubSlidingWindowSpec()
        assert _is_mamba_spec(spec) is False

    def test_plain_mock_returns_false(self):
        spec = Mock()
        assert _is_mamba_spec(spec) is False

    def test_sliding_window_spec_true(self, inject_spec_classes):
        spec = _StubSlidingWindowSpec()
        assert _is_sliding_window_spec(spec) is True

    def test_share_kv_sliding_window_is_sw(self, inject_spec_classes):
        """ShareKVSlidingWindowSpec extends SlidingWindowSpec."""
        spec = _StubShareKVSlidingWindowSpec()
        assert _is_sliding_window_spec(spec) is True

    def test_non_sw_returns_false(self, inject_spec_classes):
        spec = _StubMambaSpec()
        assert _is_sliding_window_spec(spec) is False

    def test_plain_mock_sw_returns_false(self):
        spec = Mock()
        assert _is_sliding_window_spec(spec) is False


# ===================================================================
# Tests for _derive_attention_head_sizes
# ===================================================================

class TestDeriveAttentionHeadSizes:
    """Tests for _derive_attention_head_sizes."""

    def test_share_kv_sw_large_head_dim(self, inject_spec_classes):
        """ShareKVSlidingWindowSpec with head_dim >= 576 splits into [512, 64]."""
        spec = _StubShareKVSlidingWindowSpec()
        result = _derive_attention_head_sizes(spec, head_dim=576)
        assert result == [512, 64]

    def test_share_kv_sw_head_dim_768(self, inject_spec_classes):
        spec = _StubShareKVSlidingWindowSpec()
        result = _derive_attention_head_sizes(spec, head_dim=768)
        assert result == [512, 64]

    def test_share_kv_sw_small_head_dim(self, inject_spec_classes):
        """ShareKVSlidingWindowSpec with head_dim < 576 returns [head_dim]."""
        spec = _StubShareKVSlidingWindowSpec()
        result = _derive_attention_head_sizes(spec, head_dim=256)
        assert result == [256]

    def test_sliding_window_spec(self, inject_spec_classes):
        """Plain SlidingWindowSpec returns [head_dim]."""
        spec = _StubSlidingWindowSpec()
        result = _derive_attention_head_sizes(spec, head_dim=128)
        assert result == [128]

    def test_default_spec(self, inject_spec_classes):
        """Unknown spec type returns [head_dim]."""
        spec = Mock()
        result = _derive_attention_head_sizes(spec, head_dim=64)
        assert result == [64]


# ===================================================================
# Tests for _req_offset_for_spec
# ===================================================================

class TestReqOffsetForSpec:
    """Tests for _req_offset_for_spec."""

    def test_sliding_window_with_sw(self, inject_spec_classes):
        """SWA spec: ceil(sw / block_size) + 2 (one for window, one for leading null block)."""
        spec = _StubSlidingWindowSpec(block_size=128, sliding_window=256)
        result = _req_offset_for_spec(spec, max_model_len=2048)
        # ceil(256/128) + 2 = 2 + 2 = 4
        assert result == 4

    def test_sliding_window_sw_not_aligned(self, inject_spec_classes):
        spec = _StubSlidingWindowSpec(block_size=128, sliding_window=300)
        result = _req_offset_for_spec(spec, max_model_len=2048)
        # ceil(300/128) + 2 = 3 + 2 = 5
        assert result == 5

    def test_sliding_window_no_sw_attr(self, inject_spec_classes):
        """SWA spec without sliding_window attr falls back to full sequence."""
        spec = _StubSlidingWindowSpec(block_size=128, sliding_window=256)
        del spec.sliding_window
        result = _req_offset_for_spec(spec, max_model_len=2048)
        # ceil(2048/128) = 16
        assert result == 16

    def test_default_full_attention(self, inject_spec_classes):
        """Non-SWA spec: ceil(max_model_len / block_size)."""
        spec = _make_default_spec_with_attrs(block_size=128, dtype=torch.float16)
        result = _req_offset_for_spec(spec, max_model_len=2048)
        # ceil(2048/128) = 16
        assert result == 16

    def test_default_with_mock_spec(self):
        """Plain Mock (not SWA): full sequence."""
        spec = Mock()
        spec.block_size = 64
        result = _req_offset_for_spec(spec, max_model_len=1024)
        # ceil(1024/64) = 16
        assert result == 16

    def test_minimum_offset_is_one(self, inject_spec_classes):
        """Even with tiny max_model_len, offset is at least 1."""
        spec = _StubSlidingWindowSpec(block_size=128, sliding_window=1)
        result = _req_offset_for_spec(spec, max_model_len=1)
        # ceil(1/128) + 2 = 1 + 2 = 3
        assert result >= 1


# ===================================================================
# Tests for _alloc_attention
# ===================================================================

class TestAllocAttention:
    """Tests for _alloc_attention."""

    def _make_cache_obj(self, block_bytes=16384, with_vllm_config=False):
        cache = Mock()
        cache.block_bytes = block_bytes
        # Mock vllm_config to avoid Mock truthy issues
        if with_vllm_config:
            mock_vllm_config = Mock()
            mock_model_config = Mock()
            mock_hf_config = Mock()
            # Set specific return values to avoid Mock truthy issues
            mock_hf_config.index_head_dim = None
            mock_hf_config.indexer_head_dim = None
            mock_model_config.hf_config = mock_hf_config
            mock_vllm_config.model_config = mock_model_config
            cache.vllm_config = mock_vllm_config
        else:
            # Explicitly set vllm_config to None to avoid Mock truthy issues
            cache.vllm_config = None
        return cache

    def test_basic_attention_alloc(self, inject_spec_classes):
        """Default attention spec allocates correctly."""
        cache_obj = self._make_cache_obj()
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16)

        data, raw_view = _alloc_attention(cache_obj, spec, num_blocks=5, device="cpu")

        # data is a tuple of head views (single head for default)
        assert isinstance(data, tuple)
        assert len(data) == 1
        # Shape: (num_blocks, block_size, head_dim)
        head_view = data[0]
        assert head_view.shape[0] == 5
        assert head_view.shape[1] == 16
        # raw_view shape: (num_blocks, block_stride)
        assert raw_view.shape[0] == 5

    def test_attention_with_head_sizes(self, inject_spec_classes):
        """Attention spec with explicit head_sizes attribute."""
        cache_obj = self._make_cache_obj()
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16,
                                             head_sizes=[64, 32])

        data, raw_view = _alloc_attention(cache_obj, spec, num_blocks=3, device="cpu")

        assert isinstance(data, tuple)
        assert len(data) == 2
        assert data[0].shape == (3, 16, 64)
        assert data[1].shape == (3, 16, 32)

    def test_attention_with_kvscale(self, inject_spec_classes):
        """Attention spec with separate_kvscale returns scale tensor."""
        cache_obj = self._make_cache_obj()
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16,
                                             separate_kvscale=torch.int8)

        result, raw_view = _alloc_attention(cache_obj, spec, num_blocks=4, device="cpu")

        # result is (data_tuple, scale_tensor)
        assert isinstance(result, tuple)
        assert len(result) == 2
        data, scale = result
        assert isinstance(data, tuple)
        assert scale.shape == (4, 16)
        assert scale.dtype == torch.int8

    def test_dsa_attention_alloc(self, inject_spec_classes):
        """DSA spec allocates only indexer component."""
        cache_obj = self._make_cache_obj()
        spec = _StubDSAAttentionSpec(block_size=16, dtype=torch.float16,
                                     head_sizes=[128])

        data, raw_view = _alloc_attention(cache_obj, spec, num_blocks=5, device="cpu")

        assert isinstance(data, tuple)
        assert len(data) == 1
        # Shape: (num_blocks, block_size, indexer_head_dim)
        assert data[0].shape == (5, 16, 128)

    def test_dsa_attention_no_head_sizes(self, inject_spec_classes):
        """DSA spec without head_sizes falls back to INDEXER_HEAD_DIM."""
        cache_obj = self._make_cache_obj()
        spec = _StubDSAAttentionSpec(block_size=16, dtype=torch.float16,
                                     head_sizes=None)

        data, raw_view = _alloc_attention(cache_obj, spec, num_blocks=3, device="cpu")

        # Falls back to INDEXER_HEAD_DIM = 128
        assert data[0].shape == (3, 16, 128)

    def test_zero_initialized(self, inject_spec_classes):
        """All allocated tensors should be zero-initialized."""
        cache_obj = self._make_cache_obj()
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16)

        data, raw_view = _alloc_attention(cache_obj, spec, num_blocks=2, device="cpu")

        assert torch.all(data[0] == 0)
        assert torch.all(raw_view == 0)


# ===================================================================
# Tests for _alloc_mamba
# ===================================================================

class TestAllocMamba:
    """Tests for _alloc_mamba."""

    def _make_cache_obj(self, block_bytes=16384):
        cache = Mock()
        cache.block_bytes = block_bytes
        return cache

    def test_basic_mamba_alloc(self, inject_spec_classes):
        """Plain MambaSpec allocation."""
        cache_obj = self._make_cache_obj()
        spec = _StubMambaSpec(block_size=1, shapes=[64], dtypes=[torch.float16])

        data, raw_view = _alloc_mamba(cache_obj, spec, num_blocks=4, device="cpu")

        assert isinstance(data, tuple)
        assert len(data) == 1
        # block_size=1, shape[0]=64 -> (num_blocks, 1, 64)
        assert data[0].shape == (4, 1, 64)
        assert raw_view.shape[0] == 4

    def test_mome_spec_uses_num_total_tokens(self, inject_spec_classes):
        """MomeSpec uses num_total_tokens as block_size."""
        cache_obj = self._make_cache_obj()
        spec = _StubMomeSpec(kernel_size=7, num_spec_tokens=1,
                             shapes=[64], dtypes=[torch.float16])
        # num_total_tokens = 7 - 1 + 1 = 7

        data, raw_view = _alloc_mamba(cache_obj, spec, num_blocks=4, device="cpu")

        assert data[0].shape == (4, 7, 64)

    def test_mamba_multi_shape(self, inject_spec_classes):
        """MambaSpec with multiple shapes (tuple and int)."""
        cache_obj = self._make_cache_obj()
        spec = _StubMambaSpec(block_size=1, shapes=[64, (32,)], dtypes=[torch.float16])

        data, raw_view = _alloc_mamba(cache_obj, spec, num_blocks=3, device="cpu")

        assert len(data) == 2
        assert data[0].shape == (3, 1, 64)
        assert data[1].shape == (3, 1, 32)

    def test_mamba_zero_initialized(self, inject_spec_classes):
        cache_obj = self._make_cache_obj()
        spec = _StubMambaSpec(block_size=1, shapes=[64], dtypes=[torch.float16])

        data, raw_view = _alloc_mamba(cache_obj, spec, num_blocks=2, device="cpu")

        assert torch.all(data[0] == 0)


# ===================================================================
# Tests for construct_hbm_buffer
# ===================================================================

class TestConstructHbmBuffer:
    """Tests for the main construct_hbm_buffer function."""

    def test_default_attention_group(self, inject_spec_classes):
        """Default (non-SWA, non-mamba) attention group."""
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0", "layers.1"])],
                                   num_max_batch_pool=4, max_model_len=128)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        assert len(hbm_pool) == 1
        assert len(bt_pool) == 1

        group_cache = hbm_pool[0]
        assert "layers.0" in group_cache
        assert "layers.1" in group_cache

        # group_cache[layer] is the data tuple (head views)
        layer_data = group_cache["layers.0"]
        assert isinstance(layer_data, tuple)

        # Block table metadata
        bt = bt_pool[0]
        assert bt["kind"] == "attention"
        assert bt["block_size"] == 16
        assert bt["req_offset"] == (128 + 16 - 1) // 16  # ceil(128/16) = 8
        assert "block_table" in bt
        assert "block_table_ts" in bt
        assert bt["block_table_ts"].dtype == torch.int32

    def test_sliding_window_group(self, inject_spec_classes):
        """SlidingWindowSpec group with correct req_offset."""
        spec = _StubSlidingWindowSpec(block_size=128, sliding_window=256,
                                      dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0"])],
                                   num_max_batch_pool=4, max_model_len=2048)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        bt = bt_pool[0]
        assert bt["kind"] == "attention"
        # req_offset = ceil(256/128) + 2 = 4
        assert bt["req_offset"] == 4
        # num_blocks = 4 * 4 + 1 = 17
        # block_table = range(1, 17)
        assert bt["block_table"] == list(range(1, 17))

    def test_mamba_group(self, inject_spec_classes):
        """MambaSpec group."""
        spec = _StubMambaSpec(block_size=1, shapes=[64], dtypes=[torch.float16])

        cache = _make_decode_cache([(spec, ["layers.0"])],
                                   num_max_batch_pool=4, max_model_len=2048)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        bt = bt_pool[0]
        assert bt["kind"] == "mamba"
        assert bt["req_offset"] == 1
        assert bt["block_size"] == 1
        # block_table = range(max_num_reqs) = range(4)
        assert bt["block_table"] == [0, 1, 2, 3]
        # num_blocks = 4 * 1 + 1 = 5
        layer_data = hbm_pool[0]["layers.0"]
        assert layer_data[0].shape[0] == 5

    def test_mome_group(self, inject_spec_classes):
        """MomeSpec group uses num_total_tokens as block_size."""
        spec = _StubMomeSpec(kernel_size=7, num_spec_tokens=1,
                             shapes=[64], dtypes=[torch.float16])

        cache = _make_decode_cache([(spec, ["layers.0"])],
                                   num_max_batch_pool=4, max_model_len=2048)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        bt = bt_pool[0]
        assert bt["kind"] == "mamba"
        # MomeSpec uses num_total_tokens as block_size
        assert bt["block_size"] == 7  # 7 - 1 + 1

    def test_dsa_group(self, inject_spec_classes):
        """DSAAttentionSpec group uses num_blocks from decode_cache."""
        spec = _StubDSAAttentionSpec(block_size=16, dtype=torch.float16,
                                     head_sizes=[128])

        cache = _make_decode_cache([(spec, ["layers.0"])],
                                   num_max_batch_pool=4, max_model_len=2048,
                                   num_blocks=200)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        # DSA uses decode_cache.num_blocks directly
        layer_data = hbm_pool[0]["layers.0"]
        assert layer_data[0].shape[0] == 200

    def test_multiple_groups(self, inject_spec_classes):
        """Multiple kv_cache_groups of different types."""
        mamba_spec = _StubMambaSpec(block_size=1, shapes=[64], dtypes=[torch.float16])
        swa_spec = _StubSlidingWindowSpec(block_size=128, sliding_window=256,
                                          dtype=torch.float16)
        full_spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16)

        cache = _make_decode_cache([
            (mamba_spec, ["mamba.0"]),
            (swa_spec, ["swa.0"]),
            (full_spec, ["full.0"]),
        ], num_max_batch_pool=4, max_model_len=2048)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        assert len(hbm_pool) == 3
        assert bt_pool[0]["kind"] == "mamba"
        assert bt_pool[1]["kind"] == "attention"
        assert bt_pool[2]["kind"] == "attention"

    def test_empty_layers(self):
        """Group with empty layer_names produces empty group_cache."""
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16)

        cache = _make_decode_cache([(spec, [])])

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        assert len(hbm_pool) == 1
        assert len(hbm_pool[0]) == 0

    def test_block_table_ts_is_int32_tensor(self, inject_spec_classes):
        """block_table_ts should be a torch.int32 tensor on the correct device."""
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0"])])

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        ts = bt_pool[0]["block_table_ts"]
        assert isinstance(ts, torch.Tensor)
        assert ts.dtype == torch.int32
        assert ts.device.type == "cpu"

    def test_reserved_zero_slot_in_attention(self, inject_spec_classes):
        """Attention groups have +1 reserved zero slot in num_blocks.
        block_table starts from 1 (skipping the zero slot)."""
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0"])],
                                   num_max_batch_pool=4, max_model_len=128)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        req_offset = (128 + 16 - 1) // 16  # = 8
        num_blocks = 4 * req_offset + 1  # = 33
        # block_table = range(1, num_blocks) -> skips 0
        assert bt_pool[0]["block_table"][0] == 1
        assert len(bt_pool[0]["block_table"]) == num_blocks - 1

    def test_reserved_zero_slot_in_mamba(self, inject_spec_classes):
        """Mamba groups also have +1 reserved zero slot."""
        spec = _StubMambaSpec(block_size=1, shapes=[64], dtypes=[torch.float16])

        cache = _make_decode_cache([(spec, ["layers.0"])],
                                   num_max_batch_pool=4)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        # num_blocks = 4 * 1 + 1 = 5
        layer_data = hbm_pool[0]["layers.0"]
        assert layer_data[0].shape[0] == 5
        # Mamba block_table = range(max_num_reqs) = [0,1,2,3] (no skip)
        assert bt_pool[0]["block_table"] == [0, 1, 2, 3]


# ===================================================================
# Edge case tests
# ===================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_very_small_max_model_len(self, inject_spec_classes):
        """Very small max_model_len should still produce valid buffers."""
        spec = _make_default_spec_with_attrs(block_size=128, dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0"])],
                                   num_max_batch_pool=1, max_model_len=1)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        assert len(hbm_pool) == 1
        layer_data = hbm_pool[0]["layers.0"]
        assert layer_data[0].shape[0] >= 1

    def test_large_block_size(self, inject_spec_classes):
        """block_size larger than max_model_len still works (offset = 1)."""
        spec = _make_default_spec_with_attrs(block_size=4096, dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0"])],
                                   num_max_batch_pool=2, max_model_len=512)

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        # req_offset = ceil(512/4096) = 1
        assert bt_pool[0]["req_offset"] == 1

    def test_multiple_layers_share_buffer(self, inject_spec_classes):
        """Multiple layer_names in one group share the same buffer shape."""
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0", "layers.1", "layers.2"])])

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        group = hbm_pool[0]
        for layer_name in ["layers.0", "layers.1", "layers.2"]:
            layer_data = group[layer_name]
            assert layer_data[0].shape[0] == layer_data[0].shape[0]

    def test_hbm_buffer_pool_raw_stored(self, inject_spec_classes):
        """block_pool should contain hbm_buffer_pool_raw with raw views."""
        spec = _make_default_spec_with_attrs(block_size=16, dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0"])])

        hbm_pool, bt_pool = construct_hbm_buffer(cache)

        assert "hbm_buffer_pool_raw" in bt_pool[0]
        raw_dict = bt_pool[0]["hbm_buffer_pool_raw"]
        assert "layers.0" in raw_dict
        assert raw_dict["layers.0"].shape[0] > 0