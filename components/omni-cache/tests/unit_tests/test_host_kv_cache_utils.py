# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for host_kv_cache_utils.py module.

Tests host-side KV cache parsing utilities for decode cache, covering:
- _import_spec_types (spec type detection)
- _resolve_attention_head_sizes (head-size dispatch by spec subtype)
- _host_cache_view (strided attention views)
- _host_cache_view_mamba (byte-packed Mamba/MoME state views)
- parse_pa_kv_cache (main entry point, all dispatch paths)
"""

import sys
import types
from dataclasses import dataclass, field

import pytest
import torch
from unittest.mock import Mock

from omni_cache.cache.decode.host_kv_cache_utils import (
    _import_spec_types,
    _resolve_attention_head_sizes,
    _host_cache_view,
    parse_pa_kv_cache,
)

# ---------------------------------------------------------------------------
# Spec stubs matching the vllm / Pangu V2 class hierarchy so that isinstance
# checks inside host_kv_cache_utils work correctly in tests.
#
# We extend the conftest's AttentionSpec (a frozen dataclass) for the
# attention-type specs that need isinstance dispatch.
# ---------------------------------------------------------------------------

def _get_conftest_attention_spec():
    """Get the AttentionSpec class injected by conftest into vllm."""
    import vllm.v1.kv_cache_interface as kvif
    return kvif.AttentionSpec


def _get_conftest_full_attention_spec():
    import vllm.v1.kv_cache_interface as kvif
    return kvif.FullAttentionSpec


class _StubMambaSpec:
    """Matches vllm.v1.kv_cache_interface.MambaSpec.

    shapes must be a list of tuples (or ints that can be passed to prod()).
    The real MambaSpec uses tuples like (64,) or (1024,).
    """
    def __init__(self, block_size=1, shapes=None, dtypes=None):
        self.block_size = block_size
        self.shapes = shapes or [(64,)]  # list of tuples for prod()
        self.dtypes = dtypes or [torch.float16]


class _StubMomeSpec(_StubMambaSpec):
    """Matches MomeSpec (extends MambaSpec)."""
    def __init__(self, block_size=1, kernel_size=7, num_spec_tokens=1,
                 shapes=None, dtypes=None):
        super().__init__(block_size=block_size, shapes=shapes, dtypes=dtypes)
        self.kernel_size = kernel_size
        self.num_spec_tokens = num_spec_tokens

    @property
    def num_total_tokens(self):
        return self.kernel_size - 1 + self.num_spec_tokens


# SlidingWindowSpec, DSAAttentionSpec, ShareKVSlidingWindowSpec, MLAAttentionSpec
# must extend AttentionSpec for isinstance() dispatch. We create them as
# dataclasses that inherit from the conftest's AttentionSpec.


def _make_sliding_window_spec_cls():
    """Create a SlidingWindowSpec dataclass that extends AttentionSpec."""
    AttentionSpec = _get_conftest_attention_spec()

    @dataclass(frozen=True)
    class SlidingWindowSpec(AttentionSpec):
        sliding_window: int = 256
        separate_kvscale: object = False

    return SlidingWindowSpec


def _make_dsa_attention_spec_cls():
    """Create a DSAAttentionSpec dataclass that extends AttentionSpec."""
    AttentionSpec = _get_conftest_attention_spec()

    @dataclass(frozen=True)
    class DSAAttentionSpec(AttentionSpec):
        head_sizes: object = None
        separate_kvscale: object = False

    return DSAAttentionSpec


def _make_share_kv_sliding_window_spec_cls():
    """Create a ShareKVSlidingWindowSpec that extends SlidingWindowSpec."""
    SlidingWindowSpec = _make_sliding_window_spec_cls()
    return type("ShareKVSlidingWindowSpec", (SlidingWindowSpec,), {})


def _make_mla_attention_spec_cls():
    """Create a MLAAttentionSpec dataclass that extends AttentionSpec."""
    AttentionSpec = _get_conftest_attention_spec()

    @dataclass(frozen=True)
    class MLAAttentionSpec(AttentionSpec):
        head_size_v: int | None = None
        separate_kvscale: object = False

    return MLAAttentionSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inject_spec_classes():
    """Inject spec stubs into sys.modules so dynamic isinstance() imports
    inside host_kv_cache_utils resolve correctly."""
    import vllm.v1.kv_cache_interface as kvif

    # Create spec classes that properly extend AttentionSpec
    SlidingWindowSpec = _make_sliding_window_spec_cls()
    DSAAttentionSpec = _make_dsa_attention_spec_cls()
    ShareKVSlidingWindowSpec = _make_share_kv_sliding_window_spec_cls()
    MLAAttentionSpec = _make_mla_attention_spec_cls()

    saved = {}
    for name in ("SlidingWindowSpec", "MambaSpec", "DSAAttentionSpec",
                 "MLAAttentionSpec"):
        saved[name] = getattr(kvif, name, None)

    kvif.SlidingWindowSpec = SlidingWindowSpec
    kvif.MambaSpec = _StubMambaSpec
    kvif.DSAAttentionSpec = DSAAttentionSpec
    kvif.MLAAttentionSpec = MLAAttentionSpec

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
    mod.ShareKVSlidingWindowSpec = ShareKVSlidingWindowSpec
    mod.DSAAttentionSpec = DSAAttentionSpec

    yield

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


def _make_decode_cache(specs_and_layers, *, num_blocks=100, device="cpu"):
    """Build a mock DecodeOmniCache with the given kv_cache_groups."""
    cache = Mock()
    cache.num_blocks = num_blocks
    cache.device = device
    cache.dsa_host_pool = None  # Explicitly set to None to avoid Mock auto-creation

    groups = []
    for spec, layer_names in specs_and_layers:
        group = Mock()
        group.layer_names = layer_names
        group.kv_cache_spec = spec
        groups.append(group)

    config = Mock()
    config.kv_cache_groups = groups
    cache.kv_cache_config = config

    # Set up vllm_config with proper hf_config to avoid Mock truthy issues
    mock_hf_config = Mock()
    mock_hf_config.kv_lora_rank = None
    mock_hf_config.qk_rope_head_dim = None
    mock_model_config = Mock()
    mock_model_config.hf_config = mock_hf_config
    mock_vllm_config = Mock()
    mock_vllm_config.model_config = mock_model_config
    cache.vllm_config = mock_vllm_config

    return cache


def _make_host_cache(num_layers, num_blocks, block_size, head_dim,
                     dtype=torch.float16):
    """Create a host cache tensor with the standard 4D layout."""
    return [torch.randn(num_layers, num_blocks, block_size, head_dim,
                        dtype=dtype)]


# ===================================================================
# Tests for _import_spec_types
# ===================================================================

class TestImportSpecTypes:

    def test_returns_dict_with_spec_types(self, inject_spec_classes):
        types = _import_spec_types()
        assert types is not None
        assert "AttentionSpec" in types
        assert "MambaSpec" in types
        assert "SlidingWindowSpec" in types
        assert "DSAAttentionSpec" in types
        assert "MomeSpec" in types
        assert "ShareKVSlidingWindowSpec" in types
        assert "MLAAttentionSpec" in types


# ===================================================================
# Tests for _resolve_attention_head_sizes
# ===================================================================

class TestResolveAttentionHeadSizes:

    def _get_injected_spec_cls(self, name):
        """Get a spec class that was injected by the fixture."""
        types = _import_spec_types()
        return types[name]

    def test_dsa_with_head_sizes_attr(self, inject_spec_classes):
        DSAAttentionSpec = self._get_injected_spec_cls("DSAAttentionSpec")
        spec = DSAAttentionSpec(head_size=704, head_sizes=[576, 128])
        host_cache = [torch.randn(1, 1, 1, 704)]
        result = _resolve_attention_head_sizes(spec, host_cache)
        assert result == [576, 128]

    def test_dsa_without_head_sizes(self, inject_spec_classes):
        DSAAttentionSpec = self._get_injected_spec_cls("DSAAttentionSpec")
        spec = DSAAttentionSpec(head_size=704, head_sizes=None)
        host_cache = [torch.randn(1, 1, 1, 704)]
        result = _resolve_attention_head_sizes(spec, host_cache)
        # Falls back to INDEXER_HEAD_DIM=128: [704-128, 128] = [576, 128]
        assert result == [576, 128]

    def test_share_kv_sw_large_head(self, inject_spec_classes):
        ShareKVSlidingWindowSpec = self._get_injected_spec_cls("ShareKVSlidingWindowSpec")
        spec = ShareKVSlidingWindowSpec(head_size=576)
        host_cache = [torch.randn(1, 1, 1, 576)]
        result = _resolve_attention_head_sizes(spec, host_cache)
        assert result == [512, 64]

    def test_sliding_window_large_head(self, inject_spec_classes):
        SlidingWindowSpec = self._get_injected_spec_cls("SlidingWindowSpec")
        spec = SlidingWindowSpec(head_size=576)
        host_cache = [torch.randn(1, 1, 1, 576)]
        result = _resolve_attention_head_sizes(spec, host_cache)
        assert result == [512, 64]

    def test_sliding_window_small_head(self, inject_spec_classes):
        SlidingWindowSpec = self._get_injected_spec_cls("SlidingWindowSpec")
        spec = SlidingWindowSpec(head_size=128)
        host_cache = [torch.randn(1, 1, 1, 128)]
        result = _resolve_attention_head_sizes(spec, host_cache)
        assert result == [128]

    def test_mla_with_head_size_v(self, inject_spec_classes):
        MLAAttentionSpec = self._get_injected_spec_cls("MLAAttentionSpec")
        spec = MLAAttentionSpec(head_size=512, head_size_v=64)
        host_cache = [torch.randn(1, 1, 1, 512)]
        result = _resolve_attention_head_sizes(spec, host_cache)
        assert result == [512, 64]

    def test_mla_without_head_size_v(self, inject_spec_classes):
        MLAAttentionSpec = self._get_injected_spec_cls("MLAAttentionSpec")
        spec = MLAAttentionSpec(head_size=128, head_size_v=None)
        host_cache = [torch.randn(1, 1, 1, 128)]
        result = _resolve_attention_head_sizes(spec, host_cache)
        assert result == [128]

    def test_default_full_attention(self, inject_spec_classes):
        FullAttentionSpec = self._get_injected_spec_cls("FullAttentionSpec")
        spec = FullAttentionSpec(block_size=16, head_size=128, dtype=torch.float16)
        host_cache = [torch.randn(1, 1, 1, 128)]
        result = _resolve_attention_head_sizes(spec, host_cache)
        assert result == [128]


# ===================================================================
# Tests for _host_cache_view
# ===================================================================

class TestHostCacheView:

    def test_single_component_view(self, inject_spec_classes):
        host_cache = _make_host_cache(num_layers=2, num_blocks=100,
                                       block_size=128, head_dim=64)
        views = _host_cache_view(host_cache, head_sizes=[64],
                                  block_size=128, layer_idx=0,
                                  dtype=torch.float16)
        assert len(views) == 1
        assert views[0].shape == (100, 128, 64)

    def test_two_component_view(self, inject_spec_classes):
        host_cache = _make_host_cache(num_layers=2, num_blocks=50,
                                       block_size=128, head_dim=576)
        views = _host_cache_view(host_cache, head_sizes=[512, 64],
                                  block_size=128, layer_idx=0,
                                  dtype=torch.float16)
        assert len(views) == 2
        assert views[0].shape == (50, 128, 512)
        assert views[1].shape == (50, 128, 64)

    def test_views_share_storage(self, inject_spec_classes):
        host_cache = _make_host_cache(num_layers=1, num_blocks=10,
                                       block_size=16, head_dim=64)
        views = _host_cache_view(host_cache, head_sizes=[64],
                                  block_size=16, layer_idx=0,
                                  dtype=torch.float16)
        # Writing to the view should modify the host cache
        views[0][0, 0, 0] = 42.0
        assert host_cache[0][0, 0, 0, 0].item() == pytest.approx(42.0)

    def test_different_layer_indices_produce_views(self, inject_spec_classes):
        """Verify that different layer_idx values produce views with the
        correct shape (they address different slices of the host cache)."""
        host_cache = _make_host_cache(num_layers=3, num_blocks=10,
                                       block_size=16, head_dim=64)
        views0 = _host_cache_view(host_cache, head_sizes=[64],
                                  block_size=16, layer_idx=0,
                                  dtype=torch.float16)
        views2 = _host_cache_view(host_cache, head_sizes=[64],
                                  block_size=16, layer_idx=2,
                                  dtype=torch.float16)
        # Both should have the same shape but be distinct views
        assert views0[0].shape == (10, 16, 64)
        assert views2[0].shape == (10, 16, 64)
        # Verify they address different storage: writing to one should not
        # affect the other.
        orig_v0 = views0[0][0, 0, 0].item()
        orig_v2 = views2[0][0, 0, 0].item()
        views0[0][0, 0, 0] = orig_v0 + 1.0
        # views2 must remain unchanged
        assert views2[0][0, 0, 0].item() == orig_v2


# ===================================================================
# Tests for parse_pa_kv_cache
# ===================================================================

class TestParsePaKvCache:

    def _spec(self, name):
        """Get a spec class injected by the fixture via _import_spec_types."""
        types = _import_spec_types()
        return types[name]

    def test_attention_spec_dispatch(self, inject_spec_classes):
        """AttentionSpec spec goes through the attention path."""
        AttentionSpec = self._spec("AttentionSpec")
        spec = AttentionSpec(block_size=128, head_size=64, dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0", "layers.1"])],
                                   num_blocks=100)
        host_cache = _make_host_cache(num_layers=2, num_blocks=100,
                                       block_size=128, head_dim=64)

        result = parse_pa_kv_cache(cache, host_cache)

        assert "layers.0" in result
        assert "layers.1" in result
        # Each layer maps to a tuple of head views
        assert isinstance(result["layers.0"], tuple)
        assert len(result["layers.0"]) == 1
        assert result["layers.0"][0].shape == (100, 128, 64)

    def test_full_attention_spec_dispatch(self, inject_spec_classes):
        """FullAttentionSpec is a subclass of AttentionSpec, so it takes the
        attention path."""
        FullAttentionSpec = self._spec("FullAttentionSpec")
        spec = FullAttentionSpec(block_size=128, head_size=64, dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0"])], num_blocks=50)
        host_cache = _make_host_cache(num_layers=1, num_blocks=50,
                                       block_size=128, head_dim=64)

        result = parse_pa_kv_cache(cache, host_cache)

        assert "layers.0" in result
        assert isinstance(result["layers.0"], tuple)

    def test_sliding_window_dispatch(self, inject_spec_classes):
        """SlidingWindowSpec dispatches through the attention path with
        [nope, rope] split for head_size >= 576."""
        SlidingWindowSpec = self._spec("SlidingWindowSpec")
        spec = SlidingWindowSpec(block_size=128, head_size=576,
                                  dtype=torch.float16)

        cache = _make_decode_cache([(spec, ["layers.0"])], num_blocks=50)
        host_cache = _make_host_cache(num_layers=1, num_blocks=50,
                                       block_size=128, head_dim=576)

        result = parse_pa_kv_cache(cache, host_cache)

        assert "layers.0" in result
        assert isinstance(result["layers.0"], tuple)
        assert len(result["layers.0"]) == 2
        assert result["layers.0"][0].shape == (50, 128, 512)
        assert result["layers.0"][1].shape == (50, 128, 64)

    def test_dsa_dispatch(self, inject_spec_classes):
        """DSAAttentionSpec returns [nope_and_rope, indexer] split."""
        DSAAttentionSpec = self._spec("DSAAttentionSpec")
        spec = DSAAttentionSpec(block_size=128, head_size=704,
                                 dtype=torch.float16, head_sizes=[576, 128])

        cache = _make_decode_cache([(spec, ["layers.0"])], num_blocks=50)
        host_cache = _make_host_cache(num_layers=1, num_blocks=50,
                                       block_size=128, head_dim=704)

        result = parse_pa_kv_cache(cache, host_cache)

        assert "layers.0" in result
        assert isinstance(result["layers.0"], tuple)
        assert len(result["layers.0"]) == 2

    def test_mamba_dispatch(self, inject_spec_classes):
        """MambaSpec dispatches through the mamba byte-packed path."""
        MambaSpec = self._spec("MambaSpec")
        spec = MambaSpec(block_size=1, shapes=[(64,)], dtypes=[torch.float16])

        cache = _make_decode_cache([(spec, ["layers.0"])], num_blocks=10)
        # For mamba: host_cache shape is (num_layers, num_blocks, block_size, head_dim)
        # The byte width must be enough for the packed state: 64 * 2 = 128 bytes
        # Need head_dim large enough to hold the state bytes as float16 elements
        host_cache = [torch.randn(1, 10, 128, 64, dtype=torch.float16)]

        result = parse_pa_kv_cache(cache, host_cache)

        assert "layers.0" in result
        assert isinstance(result["layers.0"], tuple)

    def test_multiple_groups(self, inject_spec_classes):
        """Multiple KV cache groups of different types."""
        AttentionSpec = self._spec("AttentionSpec")
        attn_spec = AttentionSpec(block_size=64, head_size=32, dtype=torch.float32)

        SlidingWindowSpec = self._spec("SlidingWindowSpec")
        sw_spec = SlidingWindowSpec(block_size=64, head_size=32,
                                     dtype=torch.float32)

        cache = _make_decode_cache([
            (attn_spec, ["group0.layer0", "group0.layer1"]),
            (sw_spec, ["group1.layer0", "group1.layer1"]),
        ], num_blocks=50)

        host_cache = _make_host_cache(num_layers=4, num_blocks=50,
                                       block_size=64, head_dim=32,
                                       dtype=torch.float32)

        result = parse_pa_kv_cache(cache, host_cache)

        assert "group0.layer0" in result
        assert "group0.layer1" in result
        assert "group1.layer0" in result
        assert "group1.layer1" in result

    def test_empty_layers(self, inject_spec_classes):
        """Empty layer_names produces empty dict for that group."""
        AttentionSpec = self._spec("AttentionSpec")
        spec = AttentionSpec(block_size=128, head_size=64, dtype=torch.float16)

        cache = _make_decode_cache([(spec, [])], num_blocks=100)
        host_cache = _make_host_cache(num_layers=0, num_blocks=100,
                                       block_size=128, head_dim=64)

        result = parse_pa_kv_cache(cache, host_cache)
        assert result == {}

    def test_attention_with_kvscale(self, inject_spec_classes):
        """AttentionSpec with separate_kvscale goes through the kvscale path."""
        # Use SlidingWindowSpec which supports separate_kvscale field
        SlidingWindowSpec = self._spec("SlidingWindowSpec")
        spec = SlidingWindowSpec(block_size=128, head_size=64,
                                  dtype=torch.float16,
                                  separate_kvscale=torch.int8)

        cache = _make_decode_cache([(spec, ["layers.0"])], num_blocks=100)
        host_cache = [
            torch.randn(1, 100 * 128, 64, dtype=torch.float16),
            torch.randint(-128, 127, (100 * 128 * 64,), dtype=torch.int8),
        ]

        result = parse_pa_kv_cache(cache, host_cache)

        assert "layers.0" in result
        assert isinstance(result["layers.0"], tuple)
        assert len(result["layers.0"]) == 2  # data + scale


# ===================================================================
# Edge case tests
# ===================================================================

class TestParsePaKvCacheEdgeCases:

    def _spec(self, name):
        types = _import_spec_types()
        return types[name]

    def test_large_block_size(self, inject_spec_classes):
        AttentionSpec = self._spec("AttentionSpec")
        spec = AttentionSpec(block_size=4096, head_size=128, dtype=torch.float32)

        cache = _make_decode_cache([(spec, ["layers.0"])], num_blocks=10)
        host_cache = _make_host_cache(num_layers=1, num_blocks=10,
                                       block_size=4096, head_dim=128,
                                       dtype=torch.float32)

        result = parse_pa_kv_cache(cache, host_cache)
        assert "layers.0" in result
        assert result["layers.0"][0].shape == (10, 4096, 128)