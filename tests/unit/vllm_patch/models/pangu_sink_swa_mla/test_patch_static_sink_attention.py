# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import pytest
import torch

from omni_npu.vllm_patches.patches.models.pangu_sink_swa_mla import patch_static_sink_attention as patch_mod
from omni_npu.vllm_patches.patches.models.pangu_sink_swa_mla.patch_static_sink_attention import (
    StaticSinkAttentionClassPatch,
)
from vllm.config import CacheConfig

class _FakeForwardContext:
    def __init__(self, virtual_engine: int = 0):
        self.virtual_engine = virtual_engine


# =============================================================================
# Tests for StaticSinkAttentionClassPatch.__init__
# =============================================================================
class _MockAttentionBackend:
    """Mock attention backend for testing."""

    @classmethod
    def get_builder_cls(cls):
        return _MockAttentionBuilder

    @classmethod
    def get_impl_cls(cls):
        return object


class _MockAttentionBuilder:
    """Mock attention builder for testing."""

    def __init__(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device


@pytest.mark.parametrize(
    ("block_size", "sink_len", "num_heads", "head_size", "scale", "has_cache_config"),
    [
        (16, 64, 8, 128, 0.125, True),
        (128, 128, 8, 128, 0.125, True),
        (16, 32, 4, 64, 0.0625, False),  # no cache_config, use default block_size=16
    ],
    ids=[
        "basic_config",
        "production_config",
        "no_cache_config",
    ],
)
def test_init(monkeypatch, block_size, sink_len, num_heads, head_size, scale, has_cache_config):
    """Test __init__ correctly initializes attributes with various configurations.

    Key change: CustomOp.__init__ is called BEFORE Attention.__init__ to fix
    the bug where CustomOp.__init__ overwrites buffers.
    """
    call_order = []

    # Mock get_attn_backend to avoid vLLM config requirement
    monkeypatch.setattr(
        patch_mod,
        "get_attn_backend",
        lambda *args: _MockAttentionBackend,
    )

    # Mock CustomOp.__init__
    def _fake_customop_init(self):
        call_order.append("CustomOp")

    monkeypatch.setattr(patch_mod.CustomOp, "__init__", _fake_customop_init)

    # Mock Attention.__init__
    def _fake_attention_init(self, num_heads, head_size, scale, cache_config, attn_backend, **kwargs):
        call_order.append("Attention")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale

    monkeypatch.setattr(patch_mod.Attention, "__init__", _fake_attention_init)

    # Prepare cache_config
    if has_cache_config:
        cache_config = CacheConfig(
            block_size=block_size,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )
        expected_block_size = block_size
    else:
        cache_config = None
        expected_block_size = 16  # default

    init_method = StaticSinkAttentionClassPatch.__init__

    class MockStaticSinkAttention:
        pass

    mock_instance = MockStaticSinkAttention()
    init_method(
        mock_instance,
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        sink_len=sink_len,
        cache_config=cache_config,
    )

    # Verify CustomOp.__init__ is called BEFORE Attention.__init__
    assert call_order == ["CustomOp", "Attention"], (
        f"Expected call order ['CustomOp', 'Attention'], got {call_order}"
    )

    # Verify attributes
    assert mock_instance.sink_len == sink_len
    assert mock_instance.block_size == expected_block_size
    assert mock_instance.sink_populated is False
    assert mock_instance.sink_key is None
    assert mock_instance.sink_value is None


# =============================================================================
# Tests for StaticSinkAttentionClassPatch.forward_native
# =============================================================================

@pytest.mark.parametrize(
    ("sink_key_shape", "sink_value_shape", "sink_populated", "kv_cache", "expect_populate_called", "expect_assertion"),
    [
        # sink_key/sink_value not prepared -> assertion
        (None, None, False, "valid", False, True),
        # already populated -> skip populate
        ("valid", "valid", True, "valid", False, False),
        # not populated, valid kv_cache -> call populate
        ("valid", "valid", False, "valid", True, False),
        # not populated, empty kv_cache -> skip populate
        ("valid", "valid", False, "empty", False, False),
        # not populated, None kv_cache -> skip populate
        ("valid", "valid", False, "none", False, False),
    ],
    ids=[
        "sink_kv_not_prepared",
        "already_populated",
        "valid_kv_cache",
        "empty_kv_cache",
        "none_kv_cache",
    ],
)
def test_forward_native(
    monkeypatch,
    sink_key_shape,
    sink_value_shape,
    sink_populated,
    kv_cache,
    expect_populate_called,
    expect_assertion,
):
    """Test forward_native handles various scenarios with tensor shapes."""
    scatter_calls = []

    def _fake_scatter_nd_update(cache, indices, values):
        scatter_calls.append(True)

    monkeypatch.setattr(
        patch_mod.torch_npu,
        "npu_scatter_nd_update_",
        _fake_scatter_nd_update,
    )
    monkeypatch.setattr(
        patch_mod,
        "get_forward_context",
        lambda: _FakeForwardContext(),
    )

    def _fake_attention_forward(self, query, key, value, output_shape):
        return torch.tensor(42.0)

    monkeypatch.setattr(patch_mod.Attention, "forward", _fake_attention_forward)

    # production config
    block_size = 128
    sink_len = 128
    num_blocks = 2641
    num_heads = 8
    head_dim = 128

    class MockStaticSinkAttention:
        pass

    mock_instance = MockStaticSinkAttention()
    mock_instance.block_size = block_size
    mock_instance.sink_len = sink_len
    mock_instance.sink_populated = sink_populated

    # sink_key/sink_value shape: [sink_len, num_heads, head_dim]
    if sink_key_shape == "valid":
        mock_instance.sink_key = torch.randn(sink_len, num_heads, head_dim)
        mock_instance.sink_value = torch.randn(sink_len, num_heads, head_dim)
    else:
        mock_instance.sink_key = None
        mock_instance.sink_value = None

    # kv_cache shape: [num_blocks, block_size, head_dim]
    if kv_cache == "valid":
        mock_instance.kv_cache = [[
            torch.zeros(num_blocks, block_size, head_dim),
            torch.zeros(num_blocks, block_size, head_dim),
        ]]
    elif kv_cache == "empty":
        mock_instance.kv_cache = [[]]
    else:
        mock_instance.kv_cache = [None]

    populate_sink_kv = StaticSinkAttentionClassPatch.populate_sink_kv
    mock_instance.populate_sink_kv = lambda k, v: populate_sink_kv(mock_instance, k, v)

    forward_native = StaticSinkAttentionClassPatch.forward_native

    query = torch.randn(2, 4096)
    key = torch.randn(2, 1024)
    value = torch.randn(2, 1024)
    output_shape = torch.Size([2, 4096])

    if expect_assertion:
        with pytest.raises(AssertionError, match="sink_key and sink_value have not been prepared"):
            forward_native(mock_instance, query, key, value, output_shape)
    else:
        result = forward_native(mock_instance, query, key, value, output_shape)
        assert result.item() == 42.0

        if expect_populate_called:
            assert len(scatter_calls) == 2
            assert mock_instance.sink_populated is True
        else:
            # sink_populated should remain unchanged if populate not called
            assert mock_instance.sink_populated == sink_populated


# =============================================================================
# Tests for StaticSinkAttentionClassPatch.populate_sink_kv
# =============================================================================

@pytest.mark.parametrize(
    ("block_size", "sink_len", "num_blocks", "num_heads", "head_dim"),
    [
        (128, 128, 2641, 8, 128),
        (16, 4, 100, 8, 64),
        (32, 8, 200, 4, 128),
    ],
    ids=[
        "production_config",
        "small_config",
        "medium_config",
    ],
)
def test_populate_sink_kv(monkeypatch, block_size, sink_len, num_blocks, num_heads, head_dim):
    """Test populate_sink_kv correctly updates k/v cache with tensor shapes."""
    scatter_calls = []

    def _fake_scatter_nd_update(cache, indices, values):
        scatter_calls.append({
            "cache": cache,
            "indices": indices.clone().cpu(),
            "values": values,
        })

    monkeypatch.setattr(
        patch_mod.torch_npu,
        "npu_scatter_nd_update_",
        _fake_scatter_nd_update,
    )

    class MockStaticSinkAttention:
        pass

    mock_instance = MockStaticSinkAttention()
    mock_instance.block_size = block_size
    mock_instance.sink_len = sink_len
    mock_instance.sink_populated = False
    # shape: [sink_len, num_heads, head_dim]
    mock_instance.sink_key = torch.randn(sink_len, num_heads, head_dim)
    mock_instance.sink_value = torch.randn(sink_len, num_heads, head_dim)

    # kv_cache shape: [num_blocks, block_size, head_dim]
    k_cache = torch.zeros(num_blocks, block_size, head_dim)
    v_cache = torch.zeros(num_blocks, block_size, head_dim)

    populate_sink_kv = StaticSinkAttentionClassPatch.populate_sink_kv
    populate_sink_kv(mock_instance, k_cache, v_cache)

    # Verify sink_populated is set to True
    assert mock_instance.sink_populated is True

    # Verify scatter_nd_update was called twice (for k and v cache)
    assert len(scatter_calls) == 2

    # Verify the indices are correct (block_size to sink_len + block_size)
    expected_indices = torch.arange(
        block_size,
        sink_len + block_size,
        dtype=torch.long,
    ).view(-1, 1)
    assert torch.equal(scatter_calls[0]["indices"].cpu(), expected_indices.cpu())
    assert torch.equal(scatter_calls[1]["indices"].cpu(), expected_indices.cpu())

    # Verify slot mapping range
    indices = scatter_calls[0]["indices"]
    assert indices[0].item() == block_size
    assert indices[-1].item() == sink_len + block_size - 1
    assert len(indices) == sink_len

    # Verify values are sink_key and sink_value
    assert torch.equal(scatter_calls[0]["values"].cpu(), mock_instance.sink_key.cpu())
    assert torch.equal(scatter_calls[1]["values"].cpu(), mock_instance.sink_value.cpu())