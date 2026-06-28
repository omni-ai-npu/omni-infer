# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for NPU rotary embedding layers."""

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch

from vllm import platforms

from omni_npu.platform import NPUPlatform
from vllm.platforms import current_platform
from vllm.config import (
    VllmConfig,
)
platforms.current_platform = NPUPlatform()

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from omni_npu.layers.rotary_embedding.common import apply_rotary_emb_full_dim
from omni_npu.layers.rotary_embedding.deepseek_scaling_rope import (
    NPUDeepseekScalingRotaryEmbedding,
)
from omni_npu.layers.rotary_embedding.linear_scaling_rope import (
    NPULinearScalingRotaryEmbedding,
)
from omni_npu.layers.rotary_embedding.llama3_rope import (
    NPULlama3RotaryEmbedding,
)
from omni_npu.layers.rotary_embedding.rotary_embedding_torch_npu import (
    NPURotaryEmbedding,
)
from omni_npu.layers.rotary_embedding.yarn_scaling_rope import (
    NPUYaRNScalingRotaryEmbedding,
)
from omni_npu.layers.rotary_embedding.mrope import NPUMRotaryEmbedding


@pytest.fixture(scope="module", autouse=True)
def _module_default_device_npu():
    prev_device = (
        torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
    )
    torch.set_default_device("npu")
    yield
    torch.set_default_device(prev_device)


def _require_npu() -> torch.device:
    try:
        import torch_npu
    except Exception:
        pytest.skip("torch_npu is not available")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU is not available")
    return torch.device("npu")


def _reference_forward_full_dim(
    layer,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos = layer.cos_cached.index_select(0, positions)
    sin = layer.sin_cached.index_select(0, positions)

    query_shape = query.shape
    query = query.view(num_tokens, -1, layer.head_size)
    query_rot = query[..., : layer.rotary_dim]
    query_rot = apply_rotary_emb_full_dim(query_rot, cos, sin, layer.is_neox_style)
    query = query_rot.reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, layer.head_size)
    key_rot = key[..., : layer.rotary_dim]
    key_rot = apply_rotary_emb_full_dim(key_rot, cos, sin, layer.is_neox_style)
    key = key_rot.reshape(key_shape)
    return query, key


def _reference_forward_partial_dim(
    layer,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = layer.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, layer.head_size)
    query_rot = query[..., : layer.rotary_dim]
    query_pass = query[..., layer.rotary_dim :]
    query_rot = ApplyRotaryEmb.forward_static(query_rot, cos, sin, layer.is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, layer.head_size)
    key_rot = key[..., : layer.rotary_dim]
    key_pass = key[..., layer.rotary_dim :]
    key_rot = ApplyRotaryEmb.forward_static(key_rot, cos, sin, layer.is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


class _CountingCosSinCache:
    def __init__(self, value: torch.Tensor):
        self.value = value
        self.calls = 0

    def __getitem__(self, positions):
        self.calls += 1
        return self.value[positions]


def test_mrope_prefill_reuses_cached_cos_sin(default_vllm_config):
    layer = NPUMRotaryEmbedding.__new__(NPUMRotaryEmbedding)
    layer.head_size = 4
    layer.rotary_dim = 2
    layer.mrope_section = None
    layer.mrope_interleaved = True
    layer._match_cos_sin_cache_dtype = lambda query: None
    layer.apply_rotary_emb = lambda x, cos, sin: x + cos.unsqueeze(1)

    cache = _CountingCosSinCache(torch.ones(4, 4))
    layer.cos_sin_cache = cache
    metadata = SimpleNamespace()
    positions = torch.tensor([0, 1])
    query = torch.zeros(2, 4)
    key = torch.zeros(2, 4)

    first_q, first_k = layer.forward_oot(
        positions, query, key, is_prefill=True, attn_metadata=metadata
    )
    second_q, second_k = layer.forward_oot(
        positions, query, key, is_prefill=True, attn_metadata=metadata
    )

    assert cache.calls == 1
    assert hasattr(metadata, "_npu_cached_cos_sin")
    assert torch.equal(first_q, second_q)
    assert torch.equal(first_k, second_k)


@pytest.mark.parametrize(
    "layer_factory",
    [
        lambda: NPURotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
        ),
        lambda: NPULinearScalingRotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            scaling_factors=2.0,
            dtype=torch.float32,
        ),
        lambda: NPUYaRNScalingRotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            scaling_factor=2.0,
            dtype=torch.float32,
        ),
        lambda: NPULlama3RotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
            scaling_factor=2.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            orig_max_position=32,
        ),
        lambda: NPUDeepseekScalingRotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            scaling_factor=2.0,
            dtype=torch.float32,
        ),
    ],
)
def test_get_cos_sin_matches_cache(default_vllm_config, layer_factory):
    device = _require_npu()
    layer = layer_factory().to(device)
    positions = torch.tensor([0, 3, 7], device=device)
    offsets = torch.tensor([1, 1, 1], device=device)

    cos, sin = layer.get_cos_sin(positions, offsets)
    expected_pos = positions + offsets
    expected_cos = layer.cos_cached[expected_pos].view(-1, 1, 1, layer.cos_cached.shape[-1])
    expected_sin = layer.sin_cached[expected_pos].view(-1, 1, 1, layer.sin_cached.shape[-1])

    assert torch.allclose(cos, expected_cos)
    assert torch.allclose(sin, expected_sin)


@pytest.mark.parametrize(
    "layer_factory",
    [
        lambda: NPULinearScalingRotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            scaling_factors=2.0,
            dtype=torch.float32,
        ),
        lambda: NPUYaRNScalingRotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            scaling_factor=2.0,
            dtype=torch.float32,
        ),
        lambda: NPULlama3RotaryEmbedding(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
            scaling_factor=2.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            orig_max_position=32,
        ),
    ],
)
def test_get_cos_sin_no_offsets_matches_cache(default_vllm_config, layer_factory):
    device = _require_npu()
    layer = layer_factory().to(device)
    positions = torch.tensor([0, 3, 7], device=device)

    cos, sin = layer.get_cos_sin(positions)
    expected_cos = layer.cos_cached[positions].view(-1, 1, 1, layer.cos_cached.shape[-1])
    expected_sin = layer.sin_cached[positions].view(-1, 1, 1, layer.sin_cached.shape[-1])

    assert torch.allclose(cos, expected_cos)
    assert torch.allclose(sin, expected_sin)


def test_linear_scaling_get_cos_sin_with_offsets_uses_cache_slice(default_vllm_config):
    device = _require_npu()
    layer = NPULinearScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=True,
        scaling_factors=[1.0, 2.0],
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([0, 3, 7], device=device)

    offset = layer.scaling_factor_to_offset[2.0]
    offsets = torch.full_like(positions, offset)
    cos, sin = layer.get_cos_sin(positions, offsets)
    expected_pos = positions + offset
    expected_cos = layer.cos_cached[expected_pos].view(-1, 1, 1, layer.cos_cached.shape[-1])
    expected_sin = layer.sin_cached[expected_pos].view(-1, 1, 1, layer.sin_cached.shape[-1])

    assert torch.allclose(cos, expected_cos)
    assert torch.allclose(sin, expected_sin)


def test_linear_scaling_multi_factor_cache_offsets(default_vllm_config):
    device = _require_npu()
    layer = NPULinearScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=4,
        base=10000,
        is_neox_style=True,
        scaling_factors=[1.0, 2.0, 4.0],
        dtype=torch.float32,
    ).to(device)

    expected_offset_1 = 0
    expected_offset_2 = 4
    expected_offset_4 = 12
    assert layer.scaling_factor_to_offset[1.0] == expected_offset_1
    assert layer.scaling_factor_to_offset[2.0] == expected_offset_2
    assert layer.scaling_factor_to_offset[4.0] == expected_offset_4
    assert layer.cos_cached.shape[0] == 4 + 8 + 16
    assert layer.sin_cached.shape[0] == 4 + 8 + 16


def test_linear_scaling_get_cos_sin_multi_factor_offsets_match_cache(default_vllm_config):
    device = _require_npu()
    layer = NPULinearScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=4,
        base=10000,
        is_neox_style=True,
        scaling_factors=[1.0, 2.0, 4.0],
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([0, 1, 3], device=device)

    for scaling_factor in (1.0, 4.0):
        offset = layer.scaling_factor_to_offset[scaling_factor]
        offsets = torch.full_like(positions, offset)
        cos, sin = layer.get_cos_sin(positions, offsets)
        expected_pos = positions + offset
        expected_cos = layer.cos_cached[expected_pos].view(-1, 1, 1, layer.cos_cached.shape[-1])
        expected_sin = layer.sin_cached[expected_pos].view(-1, 1, 1, layer.sin_cached.shape[-1])
        assert torch.allclose(cos, expected_cos)
        assert torch.allclose(sin, expected_sin)


def test_linear_scaling_get_cos_sin_defaults_to_first_factor(default_vllm_config):
    device = _require_npu()
    layer = NPULinearScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=4,
        base=10000,
        is_neox_style=True,
        scaling_factors=[2.0, 4.0],
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([0, 2, 3], device=device)

    cos, sin = layer.get_cos_sin(positions)
    expected_cos = layer.cos_cached[positions].view(-1, 1, 1, layer.cos_cached.shape[-1])
    expected_sin = layer.sin_cached[positions].view(-1, 1, 1, layer.sin_cached.shape[-1])

    assert torch.allclose(cos, expected_cos)
    assert torch.allclose(sin, expected_sin)


@pytest.mark.parametrize(
    "layer_factory",
    [
        lambda: NPULinearScalingRotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            scaling_factors=2.0,
            dtype=torch.float32,
        ),
        lambda: NPUYaRNScalingRotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            scaling_factor=2.0,
            dtype=torch.float32,
        ),
        lambda: NPULlama3RotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
            scaling_factor=2.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            orig_max_position=32,
        ),
    ],
)
def test_forward_oot_rotary_dim_128_matches_reference(default_vllm_config, layer_factory):
    device = _require_npu()
    torch.manual_seed(0)
    layer = layer_factory().to(device)
    positions = torch.tensor([0, 2, 5], device=device)
    query = torch.randn((3, 128), device=device)
    key = torch.randn((3, 128), device=device)
    query_ref = query.clone()
    key_ref = key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_full_dim(layer, positions, query_ref, key_ref)

    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_k, exp_k, atol=1e-4, rtol=1e-4)


def test_forward_oot_rotary_dim_lt_head_size_matches_reference(default_vllm_config):
    device = _require_npu()
    torch.manual_seed(1)
    layer = NPURotaryEmbedding(
        head_size=8,
        rotary_dim=4,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([0, 2, 5], device=device)
    query = torch.randn((3, 8), device=device)
    key = torch.randn((3, 8), device=device)
    query_ref = query.clone()
    key_ref = key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_partial_dim(layer, positions, query_ref, key_ref)

    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_k, exp_k, atol=1e-4, rtol=1e-4)


def test_forward_oot_rotary_dim_lt_head_size_gptj_matches_reference(default_vllm_config):
    device = _require_npu()
    torch.manual_seed(4)
    layer = NPURotaryEmbedding(
        head_size=8,
        rotary_dim=4,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=False,
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([0, 2, 5], device=device)
    query = torch.randn((3, 8), device=device)
    key = torch.randn((3, 8), device=device)
    query_ref = query.clone()
    key_ref = key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_partial_dim(layer, positions, query_ref, key_ref)

    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_k, exp_k, atol=1e-4, rtol=1e-4)


def test_forward_oot_small_ops_matches_reference(default_vllm_config):
    device = _require_npu()
    torch.manual_seed(2)
    layer = NPURotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([1, 3, 7], device=device)
    query = torch.randn((3, 64), device=device)
    key = torch.randn((3, 64), device=device)
    query_ref = query.clone()
    key_ref = key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_full_dim(layer, positions, query_ref, key_ref)

    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_k, exp_k, atol=1e-4, rtol=1e-4)


def test_forward_oot_with_cos_sin_fused_matches_forward_oot(default_vllm_config):
    device = _require_npu()
    torch.manual_seed(6)
    layer = NPURotaryEmbedding(
        head_size=128,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([0, 2, 5], device=device)
    query = torch.randn((3, 128), device=device)
    key = torch.randn((3, 128), device=device)
    query_ref = query.clone()
    key_ref = key.clone()

    cos, sin = layer.get_cos_sin(positions)
    exp_q, exp_k = _reference_forward_partial_dim(
        layer, positions, query_ref, key_ref
    )

    def _mock_apply_rotary_pos_emb(query_in, key_in, cos_in, sin_in, layout):
        assert layout == "TND"
        assert cos_in.ndim == 3
        assert sin_in.ndim == 3
        assert cos_in.shape == (3, 1, 64)
        assert sin_in.shape == (3, 1, 64)
        q_embed = exp_q.view(3, 1, 128)
        k_embed = exp_k.view(3, 1, 128)
        return q_embed, k_embed

    with patch(
        "omni_npu.layers.rotary_embedding.rotary_embedding_torch_npu."
        "torch_npu.npu_apply_rotary_pos_emb",
        side_effect=_mock_apply_rotary_pos_emb,
    ) as mock_apply:
        out_q, out_k = layer.forward_oot(
            positions, query, key, cos=cos, sin=sin
        )

    mock_apply.assert_called_once()
    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_k, exp_k, atol=1e-4, rtol=1e-4)


def test_forward_oot_with_cos_sin_non_fused_falls_back_to_forward_oot(default_vllm_config):
    device = _require_npu()
    torch.manual_seed(7)
    layer = NPURotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([1, 3, 7], device=device)
    query = torch.randn((3, 64), device=device)
    key = torch.randn((3, 64), device=device)
    query_ref = query.clone()
    key_ref = key.clone()

    cos, sin = layer.get_cos_sin(positions)
    out_q, out_k = layer.forward_oot(
        positions, query, key, cos=cos, sin=sin
    )
    exp_q, exp_k = layer.forward_oot(positions, query_ref, key_ref)

    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_k, exp_k, atol=1e-4, rtol=1e-4)


def test_forward_oot_64_128_uses_fused_ops(default_vllm_config):
    device = _require_npu()
    layer = NPURotaryEmbedding(
        head_size=128,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([0, 2, 5], device=device)
    query = torch.randn((3, 128), device=device)
    key = torch.randn((3, 128), device=device)
    expected_q = torch.randn((3, 128), device=device)
    expected_k = torch.randn((3, 128), device=device)

    with patch.object(
        layer,
        "_forward_fused_ops",
        return_value=(expected_q, expected_k),
    ) as mock_fused:
        out_q, out_k = layer.forward_oot(positions, query, key)

    mock_fused.assert_called_once_with(
        positions, query, key, cos=None, sin=None
    )
    assert out_q is expected_q
    assert out_k is expected_k


def test_forward_fused_ops_squeezes_4d_cos_sin(default_vllm_config):
    device = _require_npu()
    layer = NPURotaryEmbedding(
        head_size=128,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(device)
    positions = torch.tensor([0, 2, 5], device=device)
    query = torch.randn((3, 128), device=device)
    key = torch.randn((3, 128), device=device)
    cos = torch.randn((3, 1, 1, 64), device=device)
    sin = torch.randn((3, 1, 1, 64), device=device)

    def _mock_apply_rotary_pos_emb(query_in, key_in, cos_in, sin_in, layout):
        assert layout == "TND"
        assert cos_in.shape == (3, 1, 64)
        assert sin_in.shape == (3, 1, 64)
        return query_in, key_in

    with patch(
        "omni_npu.layers.rotary_embedding.rotary_embedding_torch_npu."
        "torch_npu.npu_apply_rotary_pos_emb",
        side_effect=_mock_apply_rotary_pos_emb,
    ) as mock_apply:
        out_q, out_k = layer._forward_fused_ops(
            positions, query, key, cos=cos, sin=sin
        )

    mock_apply.assert_called_once()
    assert out_q.shape == (3, 128)
    assert out_k.shape == (3, 128)


def test_cached_cos_sin_mixin_branches_shape(default_vllm_config):
    device = _require_npu()
    layer = NPURotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(device)

    cos_int, sin_int = layer.get_cos_sin(4)
    positions = torch.tensor([0, 2, 3], device=device)
    cos_pos, sin_pos = layer.get_cos_sin(positions)

    assert cos_int.shape[-1] == layer.rotary_dim // 2
    assert sin_int.shape[-1] == layer.rotary_dim // 2
    assert cos_pos.shape[-1] == layer.rotary_dim
    assert sin_pos.shape[-1] == layer.rotary_dim

@pytest.mark.parametrize(
    "layer_factory",
    [
        lambda: NPULinearScalingRotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=False,
            scaling_factors=2.0,
            dtype=torch.float32,
        ),
        lambda: NPUYaRNScalingRotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=False,
            scaling_factor=2.0,
            dtype=torch.float32,
        ),
        lambda: NPULlama3RotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=False,
            dtype=torch.float32,
            scaling_factor=2.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            orig_max_position=32,
        ),
    ],
)
def test_forward_oot_rotary_dim_128_gptj_matches_reference(default_vllm_config, layer_factory):
    device = _require_npu()
    torch.manual_seed(3)
    layer = layer_factory().to(device)
    positions = torch.tensor([0, 1, 4], device=device)
    query = torch.randn((3, 128), device=device)
    key = torch.randn((3, 128), device=device)
    query_ref = query.clone()
    key_ref = key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_full_dim(layer, positions, query_ref, key_ref)

    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_k, exp_k, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "layer_factory",
    [
        lambda: NPULinearScalingRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            scaling_factors=2.0,
            dtype=torch.float32,
        ),
        lambda: NPUYaRNScalingRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            scaling_factor=2.0,
            dtype=torch.float32,
        ),
        lambda: NPULlama3RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=32,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
            scaling_factor=2.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            orig_max_position=32,
        ),
    ],
)
def test_forward_oot_key_none_matches_reference(default_vllm_config, layer_factory):
    device = _require_npu()
    torch.manual_seed(5)
    layer = layer_factory().to(device)
    positions = torch.tensor([0, 2, 5], device=device)
    query = torch.randn((3, 64), device=device)
    query_ref = query.clone()

    out_q, out_k = layer.forward_oot(positions, query, key=None)
    exp_q, _ = _reference_forward_full_dim(layer, positions, query_ref, query_ref)

    assert out_k is None
    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
