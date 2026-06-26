# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for NPU Deepseek scaling rotary embedding."""

import pytest
import torch

import omni_npu.layers.rotary_embedding.deepseek_scaling_rope as deepseek_scaling_rope_module
from vllm import platforms
from vllm.platforms import current_platform
from vllm.model_executor.layers.rotary_embedding.common import (
    rotate_gptj,
    rotate_neox,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
)

from omni_npu.platform import NPUPlatform
from omni_npu.layers.rotary_embedding.deepseek_scaling_rope import (
    NPUDeepseekScalingRotaryEmbedding,
)

torch.set_default_device("npu")
platforms.current_platform = NPUPlatform()


def _require_npu() -> torch.device:
    try:
        import torch_npu
    except Exception:
        pytest.skip("torch_npu is not available")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU is not available")
    return torch.device("npu")


def _expected_inv_freq(layer: NPUDeepseekScalingRotaryEmbedding) -> torch.Tensor:
    dim = layer.rotary_dim
    device = current_platform.device_type
    base_pow = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
    freq_extra = 1.0 / (layer.base ** base_pow)
    freq_inter = 1.0 / (layer.scaling_factor * (layer.base ** base_pow))

    low, high = yarn_find_correction_range(
        layer.beta_fast,
        layer.beta_slow,
        dim,
        layer.base,
        layer.max_position_embeddings,
    )
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(
        low, high, dim // 2, dtype=torch.float32
    ).to(device=device)
    return freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask


def _expected_forward(
    layer: NPUDeepseekScalingRotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.add(positions, offsets) if offsets is not None else positions
    cos_sin = layer.cos_sin_cache[pos]
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_rot = query[..., : layer.rotary_dim]
    key_rot = key[..., : layer.rotary_dim]
    if layer.rotary_dim < layer.head_size:
        query_pass = query[..., layer.rotary_dim :]
        key_pass = key[..., layer.rotary_dim :]

    if layer.is_neox_style:
        cos = cos.repeat(1, 1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        rotate_fn = rotate_neox
    else:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        rotate_fn = rotate_gptj

    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin

    if layer.rotary_dim < layer.head_size:
        query_out = torch.cat((query_rot, query_pass), dim=-1)
        key_out = torch.cat((key_rot, key_pass), dim=-1)
        return query_out, key_out
    return query_rot, key_rot


def _patch_fake_npu_apply_rotary_pos_emb(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_npu_apply_rotary_pos_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary_mode: str = "half",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del rotary_mode
        query_embed = query * cos + rotate_neox(query) * sin
        key_embed = key * cos + rotate_neox(key) * sin
        return query_embed, key_embed

    monkeypatch.setattr(
        deepseek_scaling_rope_module.torch_npu,
        "npu_apply_rotary_pos_emb",
        _fake_npu_apply_rotary_pos_emb,
    )


@pytest.fixture(autouse=True)
def _mock_npu_apply_rotary_pos_emb(monkeypatch: pytest.MonkeyPatch):
    _patch_fake_npu_apply_rotary_pos_emb(monkeypatch)


def test_set_cos_sin_cache_matches_formula(default_vllm_config):
    device = _require_npu()
    layer = NPUDeepseekScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=True,
        scaling_factor=2.0,
        dtype=torch.float32,
    ).to(device)

    expected_inv_freq = _expected_inv_freq(layer)
    assert torch.allclose(layer.inv_freq, expected_inv_freq, atol=1e-6, rtol=1e-6)

    t = torch.arange(
        layer.max_position_embeddings * layer.scaling_factor,
        device=current_platform.device_type,
        dtype=torch.float32,
    )
    freqs = torch.outer(t, expected_inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    expected_cos = (emb.cos() * layer.mscale).to(layer.cos_cached.dtype)
    expected_sin = (emb.sin() * layer.mscale).to(layer.sin_cached.dtype)

    assert layer.cos_cached.shape == expected_cos.shape
    assert layer.sin_cached.shape == expected_sin.shape
    assert torch.allclose(layer.cos_cached, expected_cos, atol=1e-5, rtol=1e-5)
    assert torch.allclose(layer.sin_cached, expected_sin, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("is_neox_style", [True, False])
def test_forward_oot_full_rotary_dim_matches_expected(
    default_vllm_config, is_neox_style: bool,
):
    device = _require_npu()
    torch.manual_seed(0)
    layer = NPUDeepseekScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=is_neox_style,
        scaling_factor=2.0,
        dtype=torch.float32,
    ).to(device)

    positions = torch.tensor([0, 2, 5], device=device)
    query = torch.randn((3, 1, 8), device=device)
    key = torch.randn((3, 1, 8), device=device)
    query_ref = query.clone()
    key_ref = key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _expected_forward(layer, positions, query_ref, key_ref)

    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_k, exp_k, atol=1e-4, rtol=1e-4)


def test_forward_oot_with_offsets_matches_expected(default_vllm_config):
    device = _require_npu()
    torch.manual_seed(1)
    layer = NPUDeepseekScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        scaling_factor=2.0,
        dtype=torch.float32,
    ).to(device)

    positions = torch.tensor([0, 2, 4], device=device)
    offsets = torch.tensor([1, 1, 2], device=device)
    query = torch.randn((3, 1, 8), device=device)
    key = torch.randn((3, 1, 8), device=device)
    query_ref = query.clone()
    key_ref = key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key, offsets=offsets)
    exp_q, exp_k = _expected_forward(layer, positions, query_ref, key_ref, offsets=offsets)

    assert torch.allclose(out_q, exp_q, atol=1e-4, rtol=1e-4)
    assert torch.allclose(out_k, exp_k, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("is_neox_style", [True, False])
def test_forward_oot_partial_rotary_dim_keeps_passthrough(
    default_vllm_config,
    is_neox_style: bool,
):
    device = _require_npu()
    torch.manual_seed(2)
    layer = NPUDeepseekScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=4,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=is_neox_style,
        scaling_factor=2.0,
        dtype=torch.float32,
    ).to(device)

    positions = torch.tensor([0, 3, 7], device=device)
    query = torch.randn((3, 1, 8), device=device)
    key = torch.randn((3, 1, 8), device=device)
    query_tail = query[..., layer.rotary_dim :].clone()
    key_tail = key[..., layer.rotary_dim :].clone()

    out_q, out_k = layer.forward_oot(positions, query, key)

    assert torch.allclose(out_q[..., layer.rotary_dim :], query_tail, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_k[..., layer.rotary_dim :], key_tail, atol=1e-6, rtol=1e-6)
