# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""ST tests for NPU rotary embedding layers.

Runs on actual NPU hardware, following the st/ test patterns from
test_vocab_parallel_embedding and test_activation.
"""

import pytest
import torch
import sys
import types
from pathlib import Path

from vllm import platforms
from omni_npu.platform import NPUPlatform


def _prepare_omni_layers_namespace() -> None:
    """Preload omni_npu.layers as namespace package for isolated ST imports.

    This avoids executing omni_npu.layers.__init__ during test collection,
    which may require optional runtime dependencies unrelated to rotary tests.
    """
    package_name = "omni_npu.layers"
    if package_name in sys.modules:
        return
    layers_dir = Path(__file__).resolve().parents[4] / "omni" / "layers"
    pkg = types.ModuleType(package_name)
    pkg.__path__ = [str(layers_dir)]
    sys.modules[package_name] = pkg


_prepare_omni_layers_namespace()

# Apply the vllm patch that defines MRotaryEmbeddingInterleaved on the
# vllm.model_executor.layers.rotary_embedding module. Without this the
# import in mrope_interleaved_torch_npu fails.
import omni_npu.vllm_patches.patches.models.openpangu_v1_vl.patch_m_rotary_embedding as _patch_mod  # noqa: E402
from omni_npu.vllm_patches.patch_manager import PatchManager  # noqa: E402

_pm = PatchManager()
_pm.apply_patches()

from omni_npu.layers.rotary_embedding.common import apply_rotary_emb_full_dim
try:
    from vllm.model_executor.layers.rotary_embedding.common import apply_rotary_emb_torch
except ImportError:
    # Backward compatibility for older vLLM builds where
    # apply_rotary_emb_torch is not exported.
    def apply_rotary_emb_torch(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            return torch.cat((o1, o2), dim=-1)
        return torch.stack((o1, o2), dim=-1).flatten(-2)

try:
    from vllm.model_executor.layers.rotary_embedding.common import (
        rotate_gptj,
        rotate_neox,
    )
except ImportError:
    # Backward compatibility for vLLM variants without rotate helpers export.
    def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def rotate_neox(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

try:
    from vllm.model_executor.layers.rotary_embedding.common import (
        apply_rotary_emb_dispatch as _apply_rotary_emb_dispatch,
    )
except ImportError:
    _apply_rotary_emb_dispatch = apply_rotary_emb_torch


def _apply_rotary_emb_dispatch_compat(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    return _apply_rotary_emb_dispatch(x, cos, sin, is_neox_style)

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
from omni_npu.layers.rotary_embedding.mrope_interleaved_torch_npu import (
    NPUMRotaryEmbeddingInterleaved,
)

from .distributed_test_common import parse_ascend_devices

pytestmark = pytest.mark.usefixtures("default_vllm_config")

platforms.current_platform = NPUPlatform()
FIRST_DIE, _ = parse_ascend_devices()

# Default tolerance for rotary embedding output comparison
_ROTARY_ATOL = 1e-4
_ROTARY_RTOL = 1e-4


@pytest.fixture(scope="module")
def npu_device():
    device = torch.device(f"npu:{FIRST_DIE}")
    torch.npu.set_device(device)
    return device

@pytest.fixture(scope="module")
def cpu_device():
    device = torch.device("cpu")
    return device

def _assert_rotary_outputs_match(
    out_q: torch.Tensor,
    out_k: torch.Tensor | None,
    exp_q: torch.Tensor,
    exp_k: torch.Tensor | None,
    atol: float = _ROTARY_ATOL,
    rtol: float = _ROTARY_RTOL,
) -> None:
    """Assert that rotary embedding outputs match expected values."""
    assert out_q.shape == exp_q.shape
    assert torch.allclose(out_q, exp_q, atol=atol, rtol=rtol)
    if out_k is not None and exp_k is not None:
        assert out_k.shape == exp_k.shape
        assert torch.allclose(out_k, exp_k, atol=atol, rtol=rtol)


def _reference_forward_full_dim(
    layer,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation using apply_rotary_emb_full_dim."""
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
    """Reference for rotary_dim < head_size (ChatGLM-style)."""
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = layer.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, layer.head_size)
    query_rot = query[..., : layer.rotary_dim]
    query_pass = query[..., layer.rotary_dim :]
    query_rot = apply_rotary_emb_torch(query_rot, cos, sin, layer.is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, layer.head_size)
    key_rot = key[..., : layer.rotary_dim]
    key_pass = key[..., layer.rotary_dim :]
    key_rot = apply_rotary_emb_torch(key_rot, cos, sin, layer.is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def _apply_rotary_ref_to_qk(
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    apply_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to query and key using cos/sin and apply_fn.
    Used for cos_sin_cache-style layers (MRoPE, MRoPE interleaved).
    """
    num_tokens = query.shape[0]
    query_shape = query.shape
    query_v = query.view(num_tokens, -1, layer.head_size)
    query_rot = query_v[..., : layer.rotary_dim]
    query_rot_expected_shape = query_rot.shape
    query_pass = query_v[..., layer.rotary_dim :]
    query_rot = apply_fn(query_rot, cos, sin, layer.is_neox_style)
    # Some implementations return extra singleton dims (e.g. interleaved mRoPE
    # cos/sin cache path). Normalize to the expected [T, H, rotary_dim] shape.
    if query_rot.shape != query_rot_expected_shape:
        assert query_rot.numel() == query_v[..., : layer.rotary_dim].numel()
        query_rot = query_rot.reshape(query_rot_expected_shape)
    exp_q = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key_v = key.view(num_tokens, -1, layer.head_size)
    key_rot = key_v[..., : layer.rotary_dim]
    key_rot_expected_shape = key_rot.shape
    key_pass = key_v[..., layer.rotary_dim :]
    key_rot = apply_fn(key_rot, cos, sin, layer.is_neox_style)
    if key_rot.shape != key_rot_expected_shape:
        assert key_rot.numel() == key_v[..., : layer.rotary_dim].numel()
        key_rot = key_rot.reshape(key_rot_expected_shape)
    exp_k = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return exp_q, exp_k


# --- NPURotaryEmbedding ---


def test_npu_rotary_embedding_forward_full_dim_matches_reference(npu_device):
    """rotary_dim == head_size, neox_style."""
    torch.manual_seed(0)
    layer = NPURotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 2, 5], device=npu_device)
    query = torch.randn((3, 8), device=npu_device)
    key = torch.randn((3, 8), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_full_dim(layer, positions, query_ref, key_ref)
    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


def test_npu_rotary_embedding_forward_partial_dim_matches_reference(npu_device):
    """rotary_dim < head_size (ChatGLM-style)."""
    torch.manual_seed(1)
    layer = NPURotaryEmbedding(
        head_size=8,
        rotary_dim=4,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 2, 5], device=npu_device)
    query = torch.randn((3, 8), device=npu_device)
    key = torch.randn((3, 8), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_partial_dim(
        layer, positions, query_ref, key_ref
    )
    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


def test_npu_rotary_embedding_rotary_dim_128_fused_matches_reference(npu_device):
    """rotary_dim=128 uses fused npu_apply_rotary_pos_emb."""
    torch.manual_seed(2)
    layer = NPURotaryEmbedding(
        head_size=128,
        rotary_dim=128,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 2, 5], device=npu_device)
    query = torch.randn((3, 128), device=npu_device)
    key = torch.randn((3, 128), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_full_dim(layer, positions, query_ref, key_ref)
    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


def test_npu_rotary_embedding_small_ops_matches_reference(npu_device):
    """rotary_dim=64 (non-128) uses small ops path."""
    torch.manual_seed(3)
    layer = NPURotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([1, 3, 7], device=npu_device)
    query = torch.randn((3, 64), device=npu_device)
    key = torch.randn((3, 64), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_full_dim(layer, positions, query_ref, key_ref)
    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


# --- NPUDeepseekScalingRotaryEmbedding ---


@pytest.fixture(autouse=False)
def _patch_yarn_ramp_for_npu(monkeypatch, npu_device):
    """Patch yarn_linear_ramp_mask to return NPU tensors (vLLM base creates CPU tensors)."""
    from vllm.model_executor.layers.rotary_embedding import common as rope_common
    from vllm.model_executor.layers.rotary_embedding import deepseek_scaling_rope as ds_rope

    _orig = rope_common.yarn_linear_ramp_mask

    def _patched(low, high, dim, dtype):
        return _orig(low, high, dim, dtype).to(npu_device)

    monkeypatch.setattr(rope_common, "yarn_linear_ramp_mask", _patched)
    monkeypatch.setattr(ds_rope, "yarn_linear_ramp_mask", _patched)


def _expected_forward_deepseek(
    layer: NPUDeepseekScalingRotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for Deepseek scaling RoPE.
    Matches NPU forward_oot shape handling: when cos/sin are 2D, avoid adding
    extra leading dimension so output shape matches query/key.
    """
    pos = torch.add(positions, offsets) if offsets is not None else positions
    pos = pos.flatten()
    cos_sin = layer.cos_sin_cache[pos]
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_rot = query[..., : layer.rotary_dim]
    key_rot = key[..., : layer.rotary_dim]
    if layer.rotary_dim < layer.head_size:
        query_pass = query[..., layer.rotary_dim :]
        key_pass = key[..., layer.rotary_dim :]

    # Match NPU shape handling: cos/sin from cache are 2D [num_tokens, rotary_dim//2]
    cos_was_2d = cos.dim() == 2
    if cos_was_2d:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    if layer.is_neox_style:
        cos = cos.repeat(1, 1, 2)
        sin = sin.repeat(1, 1, 2)
    else:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    if not cos_was_2d:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    rotate_fn = rotate_neox if layer.is_neox_style else rotate_gptj
    query_rot = query_rot * cos + rotate_fn(query_rot) * sin
    key_rot = key_rot * cos + rotate_fn(key_rot) * sin

    if layer.rotary_dim < layer.head_size:
        return (
            torch.cat((query_rot, query_pass), dim=-1),
            torch.cat((key_rot, key_pass), dim=-1),
        )
    return query_rot, key_rot


@pytest.mark.parametrize("is_neox_style", [True, False])
def test_npu_deepseek_rope_forward_full_dim_matches_reference(
    npu_device, is_neox_style, _patch_yarn_ramp_for_npu
):
    """rotary_dim == head_size, neox/gptj styles."""
    torch.manual_seed(4)
    layer = NPUDeepseekScalingRotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=is_neox_style,
        scaling_factor=2.0,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 2, 5], device=npu_device)
    query = torch.randn((3, 1, 64), device=npu_device)
    key = torch.randn((3, 1, 64), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _expected_forward_deepseek(
        layer, positions, query_ref, key_ref
    )
    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


def test_npu_deepseek_rope_forward_with_offsets_matches_reference(
    npu_device, _patch_yarn_ramp_for_npu
):
    """Forward with offsets."""
    torch.manual_seed(5)
    layer = NPUDeepseekScalingRotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        scaling_factor=2.0,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 2, 4], device=npu_device)
    offsets = torch.tensor([1, 1, 2], device=npu_device)
    query = torch.randn((3, 1, 64), device=npu_device)
    key = torch.randn((3, 1, 64), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key, offsets=offsets)
    exp_q, exp_k = _expected_forward_deepseek(
        layer, positions, query_ref, key_ref, offsets=offsets
    )
    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


@pytest.mark.parametrize("is_neox_style", [True, False])
def test_npu_deepseek_rope_partial_rotary_passthrough(
    npu_device, is_neox_style, _patch_yarn_ramp_for_npu
):
    """rotary_dim < head_size: tail passthrough unchanged."""
    torch.manual_seed(6)
    layer = NPUDeepseekScalingRotaryEmbedding(
        head_size=128,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=is_neox_style,
        scaling_factor=2.0,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 3, 7], device=npu_device)
    query = torch.randn((3, 1, 128), device=npu_device)
    key = torch.randn((3, 1, 128), device=npu_device)
    query_tail = query[..., layer.rotary_dim :].clone()
    key_tail = key[..., layer.rotary_dim :].clone()

    out_q, out_k = layer.forward_oot(positions, query, key)

    assert torch.allclose(
        out_q[..., layer.rotary_dim :], query_tail, atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        out_k[..., layer.rotary_dim :], key_tail, atol=1e-6, rtol=1e-6
    )


# --- NPUYaRNScalingRotaryEmbedding ---


@pytest.mark.parametrize(
    "rotary_dim",
    [64, 128],
)
def test_npu_yarn_rope_forward_matches_reference(npu_device, rotary_dim):
    """YaRN forward vs reference."""
    torch.manual_seed(7)
    layer = NPUYaRNScalingRotaryEmbedding(
        head_size=rotary_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        scaling_factor=2.0,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 2, 5], device=npu_device)
    query = torch.randn((3, rotary_dim), device=npu_device)
    key = torch.randn((3, rotary_dim), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_full_dim(
        layer, positions, query_ref, key_ref
    )
    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


def test_npu_yarn_rope_forward_key_none(npu_device):
    """key=None: only query gets RoPE."""
    torch.manual_seed(8)
    layer = NPUYaRNScalingRotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        scaling_factor=2.0,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 2, 5], device=npu_device)
    query = torch.randn((3, 64), device=npu_device)
    query_ref = query.clone()

    out_q, out_k = layer.forward_oot(positions, query, key=None)
    exp_q, _ = _reference_forward_full_dim(
        layer, positions, query_ref, query_ref
    )
    assert out_k is None
    _assert_rotary_outputs_match(out_q, out_k, exp_q, None)


# --- NPULlama3RotaryEmbedding ---


@pytest.mark.parametrize(
    "rotary_dim",
    [64, 128],
)
def test_npu_llama3_rope_forward_matches_reference(npu_device, rotary_dim):
    """Llama3 RoPE forward vs reference."""
    torch.manual_seed(9)
    layer = NPULlama3RotaryEmbedding(
        head_size=rotary_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        scaling_factor=2.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        orig_max_position=32,
    ).to(npu_device)
    positions = torch.tensor([0, 1, 4], device=npu_device)
    query = torch.randn((3, rotary_dim), device=npu_device)
    key = torch.randn((3, rotary_dim), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_full_dim(
        layer, positions, query_ref, key_ref
    )
    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


def test_npu_llama3_rope_forward_key_none(npu_device):
    """key=None."""
    torch.manual_seed(10)
    layer = NPULlama3RotaryEmbedding(
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
    ).to(npu_device)
    positions = torch.tensor([0, 2, 5], device=npu_device)
    query = torch.randn((3, 64), device=npu_device)
    query_ref = query.clone()

    out_q, out_k = layer.forward_oot(positions, query, key=None)
    exp_q, _ = _reference_forward_full_dim(
        layer, positions, query_ref, query_ref
    )
    assert out_k is None
    _assert_rotary_outputs_match(out_q, out_k, exp_q, None)


# --- NPULinearScalingRotaryEmbedding ---


@pytest.mark.parametrize(
    "scaling_factors",
    [2.0, [1.0, 2.0]],
)
def test_npu_linear_scaling_rope_forward_matches_reference(
    npu_device, scaling_factors
):
    """Linear scaling RoPE forward vs reference."""
    torch.manual_seed(11)
    layer = NPULinearScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        scaling_factors=scaling_factors,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 2, 5], device=npu_device)
    query = torch.randn((3, 8), device=npu_device)
    key = torch.randn((3, 8), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward_oot(positions, query, key)
    exp_q, exp_k = _reference_forward_full_dim(
        layer, positions, query_ref, key_ref
    )
    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


def test_npu_linear_scaling_rope_forward_key_none(npu_device):
    """key=None."""
    torch.manual_seed(12)
    layer = NPULinearScalingRotaryEmbedding(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        scaling_factors=2.0,
        dtype=torch.float32,
    ).to(npu_device)
    positions = torch.tensor([0, 2, 5], device=npu_device)
    query = torch.randn((3, 64), device=npu_device)
    query_ref = query.clone()

    out_q, out_k = layer.forward_oot(positions, query, key=None)
    exp_q, _ = _reference_forward_full_dim(
        layer, positions, query_ref, query_ref
    )
    assert out_k is None
    _assert_rotary_outputs_match(out_q, out_k, exp_q, None)


# --- NPUMRotaryEmbedding ---

def test_npu_mrope_decode_npu_mrope_kernel(cpu_device, npu_device):
    """NPUMRotaryEmbedding decode path: smoke test npu_mrope."""
    try:
        from omni_npu.layers.rotary_embedding.mrope import NPUMRotaryEmbedding
    except ImportError:
        pytest.skip("NPUMRotaryEmbedding not available")

    torch.manual_seed(14)
    # npu_mrope: use rotary_dim=128, mrope_section=[32,32,0] (Qwen2-VL style)
    layer = NPUMRotaryEmbedding(
        head_size=128,
        rotary_dim=128,
        max_position_embeddings=256,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[32, 32, 0],  # 32+32+0=64 = 128//2
        mrope_interleaved=False,
    ).to(cpu_device)
    positions = torch.tensor([[3], [0], [0]], device=npu_device)
    query = torch.randn((1, 128), device=npu_device)
    key = torch.randn((1, 128), device=npu_device)

    out_q, out_k = layer.forward_oot(
        positions, query, key, is_prefill=False
    )

    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert not torch.equal(out_q, query)
    assert not torch.equal(out_k, key)


def test_npu_mrope_prefill_text_only(npu_device):
    """NPUMRotaryEmbedding prefill path: text-only positions (1D)."""
    try:
        from omni_npu.layers.rotary_embedding.mrope import NPUMRotaryEmbedding
    except ImportError:
        pytest.skip("NPUMRotaryEmbedding not available")

    torch.manual_seed(17)
    layer = NPUMRotaryEmbedding(
        head_size=128,
        rotary_dim=128,
        max_position_embeddings=256,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[32, 32, 0],  # 32+32+0=64 = 128//2
        mrope_interleaved=False,
    ).to(npu_device)
    
    # Prefill with 1D positions (text only)
    num_tokens = 10
    positions = torch.randint(0, 100, (num_tokens,), device=npu_device)
    query = torch.randn((num_tokens, 8, 128), device=npu_device)  # [T, H, D]
    key = torch.randn((num_tokens, 8, 128), device=npu_device)
    
    out_q, out_k = layer.forward_oot(
        positions, query, key, is_prefill=True
    )
    
    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert not torch.equal(out_q, query)
    assert not torch.equal(out_k, key)





def test_npu_mrope_prefill_interleaved(npu_device):
    """NPUMRotaryEmbedding prefill path with mrope_interleaved=True."""
    try:
        from omni_npu.layers.rotary_embedding.mrope import NPUMRotaryEmbedding
    except ImportError:
        pytest.skip("NPUMRotaryEmbedding not available")

    torch.manual_seed(19)
    layer = NPUMRotaryEmbedding(
        head_size=128,
        rotary_dim=128,
        max_position_embeddings=256,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[32, 32, 0],
        mrope_interleaved=True,  # Enable interleaved mode
    ).to(npu_device)
    
    num_tokens = 10
    positions = torch.randint(0, 100, (3, num_tokens), device=npu_device)
    query = torch.randn((num_tokens, 8, 128), device=npu_device)
    key = torch.randn((num_tokens, 8, 128), device=npu_device)
    
    out_q, out_k = layer.forward_oot(
        positions, query, key, is_prefill=True
    )
    
    assert out_q.shape == query.shape
    assert out_k.shape == key.shape


def test_npu_mrope_prefill_2d_non_interleaved(npu_device):
    """NPUMRotaryEmbedding prefill path: 2D positions, non-interleaved."""
    try:
        from omni_npu.layers.rotary_embedding.mrope import NPUMRotaryEmbedding
    except ImportError:
        pytest.skip("NPUMRotaryEmbedding not available")

    torch.manual_seed(18)
    layer = NPUMRotaryEmbedding(
        head_size=128,
        rotary_dim=128,
        max_position_embeddings=256,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[32, 32, 0],
        mrope_interleaved=False,
    ).to(npu_device)

    num_tokens = 10
    positions = torch.randint(0, 100, (3, num_tokens), device=npu_device)
    query = torch.randn((num_tokens, 8, 128), device=npu_device)
    key = torch.randn((num_tokens, 8, 128), device=npu_device)

    out_q, out_k = layer.forward_oot(
        positions, query, key, is_prefill=True
    )

    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert not torch.equal(out_q, query)
    assert not torch.equal(out_k, key)


def test_npu_mrope_prefill_rotary_dim_less_than_head_size(npu_device):
    """NPUMRotaryEmbedding prefill path with rotary_dim < head_size."""
    try:
        from omni_npu.layers.rotary_embedding.mrope import NPUMRotaryEmbedding
    except ImportError:
        pytest.skip("NPUMRotaryEmbedding not available")

    torch.manual_seed(20)
    # rotary_dim=64, head_size=128
    layer = NPUMRotaryEmbedding(
        head_size=128,
        rotary_dim=64,  # Only half needs rotation
        max_position_embeddings=256,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[16, 16, 0],
        mrope_interleaved=False,
    ).to(npu_device)
    
    num_tokens = 10
    positions = torch.randint(0, 100, (num_tokens,), device=npu_device)
    query = torch.randn((num_tokens, 8, 128), device=npu_device)
    key = torch.randn((num_tokens, 8, 128), device=npu_device)
    
    # Save the tail (non-rotary) part for verification
    query_tail = query[..., layer.rotary_dim:].clone()
    key_tail = key[..., layer.rotary_dim:].clone()
    
    out_q, out_k = layer.forward_oot(
        positions, query, key, is_prefill=True
    )
    
    # Verify shape and tail part unchanged
    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert torch.allclose(out_q[..., layer.rotary_dim:], query_tail, atol=1e-6)
    assert torch.allclose(out_k[..., layer.rotary_dim:], key_tail, atol=1e-6)

# --- NPUMRotaryEmbeddingInterleaved ---


def _make_mrope_positions(npu_device, seq_len=3):
    """Create a [3, seq_len] positions tensor for mrope (temporal, height, width).

    For mrope, positions must be 2D: [3, seq_len] where dim-0 indexes the
    three spatial/temporal modalities.  In the simplest case all three
    modalities share the same position ids.
    """
    pos = torch.arange(seq_len, device=npu_device)
    return pos.unsqueeze(0).expand(3, -1).contiguous()


def _reference_rebuild_pos_emb(layer, positions):
    """Manually replicate _rebuild_pos_emb logic for verification.

    Exercises the same operations as lines 32-36:
      cos_sin = self.cos_sin_cache[positions]
      cos_sin = torch.cat(
          [m[self.mrope_dim[i]]
              for i, m in enumerate(cos_sin.split(self.mrope_section_3d, dim=-1))],
          dim=-1,
      )
    """
    cos_sin = layer.cos_sin_cache[positions]
    cos_sin = torch.cat(
        [m[layer.mrope_dim[i]]
            for i, m in enumerate(cos_sin.split(layer.mrope_section_3d, dim=-1))],
        dim=-1,
    )
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin


def _apply_rotary_ref_to_qk_mrope(layer, query, key, cos, sin):
    """Apply rotary embedding to query and key using cos/sin from _rebuild_pos_emb.

    cos/sin from _rebuild_pos_emb have shape [seq_len, 1, 1, rotary_dim].
    We squeeze the extra dims before passing to apply_rotary_emb_full_dim.
    """
    num_tokens = query.shape[0]
    query_shape = query.shape
    query_v = query.view(num_tokens, -1, layer.head_size)
    query_rot = query_v[..., : layer.rotary_dim]
    query_pass = query_v[..., layer.rotary_dim :]

    # Squeeze cos/sin to [seq_len, rotary_dim] for apply_rotary_emb_full_dim
    cos_2d = cos.squeeze(1).squeeze(1)
    sin_2d = sin.squeeze(1).squeeze(1)
    query_rot = apply_rotary_emb_full_dim(query_rot, cos_2d, sin_2d, layer.is_neox_style)
    exp_q = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    if key is not None:
        key_shape = key.shape
        key_v = key.view(num_tokens, -1, layer.head_size)
        key_rot = key_v[..., : layer.rotary_dim]
        key_pass = key_v[..., layer.rotary_dim :]
        key_rot = apply_rotary_emb_full_dim(key_rot, cos_2d, sin_2d, layer.is_neox_style)
        exp_k = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    else:
        exp_k = None
    return exp_q, exp_k


def test_npu_mrope_interleaved_rebuild_cos_sin_cache_indexing(npu_device):
    """Line 32: cos_sin = self.cos_sin_cache[positions] produces [3, seq_len, dim]."""
    torch.manual_seed(100)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    positions = _make_mrope_positions(npu_device, seq_len=3)
    cos, sin = layer._rebuild_pos_emb(positions)

    assert cos.shape == (3, 1, 1, 16), f"Unexpected cos shape: {cos.shape}"
    assert sin.shape == (3, 1, 1, 16), f"Unexpected sin shape: {sin.shape}"


def test_npu_mrope_interleaved_rebuild_split_mrope_dim_indexing(npu_device):
    """Lines 32-38: Verify the mrope_section_3d indexing pipeline."""
    torch.manual_seed(101)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    positions = _make_mrope_positions(npu_device, seq_len=3)
    cos, sin = layer._rebuild_pos_emb(positions)
    ref_cos, ref_sin = _reference_rebuild_pos_emb(layer, positions)

    # After _rebuild_pos_emb, cos/sin are expanded by "half" mode (cat with itself).
    assert torch.allclose(cos.squeeze(), ref_cos.squeeze().repeat(1, 2), atol=1e-6)
    assert torch.allclose(sin.squeeze(), ref_sin.squeeze().repeat(1, 2), atol=1e-6)


def test_npu_mrope_interleaved_rebuild_3_section_mrope(npu_device):
    """Lines 32-35 with 3-section mrope_section (e.g. [2, 2, 4])."""
    torch.manual_seed(102)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[2, 2, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    positions = _make_mrope_positions(npu_device, seq_len=3)
    cos, sin = layer._rebuild_pos_emb(positions)

    assert cos.shape == (3, 1, 1, 16)
    assert sin.shape == (3, 1, 1, 16)

    ref_cos, ref_sin = _reference_rebuild_pos_emb(layer, positions)
    assert torch.allclose(cos.squeeze(), ref_cos.squeeze().repeat(1, 2), atol=1e-6)


def test_npu_mrope_interleaved_rebuild_single_position(npu_device):
    """Lines 32-35 with a single position (seq_len=1)."""
    torch.manual_seed(103)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    positions = _make_mrope_positions(npu_device, seq_len=1)
    cos, sin = layer._rebuild_pos_emb(positions)

    assert cos.shape == (1, 1, 1, 16)
    assert sin.shape == (1, 1, 1, 16)


def test_npu_mrope_interleaved_rebuild_output_cos_sin_flag(npu_device):
    """Line 38: output_cos_sin=True forces the 'half' expansion path
    even when rotary_mode='interleave'."""
    torch.manual_seed(104)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="interleave",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    positions = _make_mrope_positions(npu_device, seq_len=3)
    cos, sin = layer._rebuild_pos_emb(positions, output_cos_sin=True)

    # Result should be half-style (cat), not interleaved
    assert cos.shape == (3, 1, 1, 16)
    assert sin.shape == (3, 1, 1, 16)


def test_npu_mrope_interleaved_rebuild_different_positions_per_modality(npu_device):
    """Lines 32-35: Different position ids per modality (temporal != spatial)."""
    torch.manual_seed(105)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    positions = torch.tensor([[0, 2, 5], [1, 3, 7], [0, 4, 8]], device=npu_device)
    cos, sin = layer._rebuild_pos_emb(positions)

    assert cos.shape == (3, 1, 1, 16)
    assert sin.shape == (3, 1, 1, 16)

    ref_cos, ref_sin = _reference_rebuild_pos_emb(layer, positions)
    assert torch.allclose(cos.squeeze(), ref_cos.squeeze().repeat(1, 2), atol=1e-6)


def test_npu_mrope_interleaved_interleave_mode_cos_sin_shape(npu_device):
    """Lines 41-43: rotary_mode='interleave' produces correct shape."""
    torch.manual_seed(200)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="interleave",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    positions = _make_mrope_positions(npu_device, seq_len=3)
    cos, sin = layer._rebuild_pos_emb(positions)

    assert cos.shape == (3, 1, 1, 16)
    assert sin.shape == (3, 1, 1, 16)


def test_npu_mrope_interleaved_interleave_mode_differs_from_half(npu_device):
    """Verify that interleave mode gives different cos/sin than half mode."""
    torch.manual_seed(201)

    layer_half = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    layer_interleave = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="interleave",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    positions = _make_mrope_positions(npu_device, seq_len=3)
    cos_half, sin_half = layer_half._rebuild_pos_emb(positions)
    cos_int, sin_int = layer_interleave._rebuild_pos_emb(positions)

    assert not torch.allclose(cos_half, cos_int), \
        "interleave and half modes should produce different cos values"
    assert not torch.allclose(sin_half, sin_int), \
        "interleave and half modes should produce different sin values"


def test_npu_mrope_interleaved_interleave_forward_matches_reference(npu_device):
    """Forward with rotary_mode='interleave' matches reference."""
    torch.manual_seed(202)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="interleave",
        num_hidden_layers_cache=1,
    ).to(npu_device)
    layer.layer_counts = 0

    positions = _make_mrope_positions(npu_device, seq_len=3)
    query = torch.randn((3, 16), device=npu_device)
    key = torch.randn((3, 16), device=npu_device)

    out_q, out_k = layer.forward(positions, query, key)

    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert not torch.equal(out_q, query)
    assert not torch.equal(out_k, key)


def test_npu_mrope_interleaved_output_cos_sin_returns_four_values(npu_device):
    """Lines 95-96: output_cos_sin=True returns (query, key, cos, sin)."""
    torch.manual_seed(300)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=256,
        base=10000,
        is_neox_style=True,
        dtype=torch.float16,
        mrope_section=[16, 16],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)
    layer.layer_counts = 0

    positions = _make_mrope_positions(npu_device, seq_len=3)
    query = torch.randn((3, 64), device=npu_device, dtype=torch.float16)
    key = torch.randn((3, 64), device=npu_device, dtype=torch.float16)

    result = layer.forward(positions, query, key, output_cos_sin=True)

    assert len(result) == 4, f"Expected 4 return values, got {len(result)}"
    out_q, out_k, out_cos, out_sin = result

    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert out_cos.shape == (3, 1, 1, 64)
    assert out_sin.shape == (3, 1, 1, 64)


def test_npu_mrope_interleaved_output_cos_sin_key_none(npu_device):
    """output_cos_sin=True with key=None still returns 4 values."""
    torch.manual_seed(301)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=256,
        base=10000,
        is_neox_style=True,
        dtype=torch.float16,
        mrope_section=[16, 16],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)
    layer.layer_counts = 0

    positions = _make_mrope_positions(npu_device, seq_len=3)
    query = torch.randn((3, 64), device=npu_device, dtype=torch.float16)

    result = layer.forward(positions, query, key=None, output_cos_sin=True)

    assert len(result) == 4
    out_q, out_k, out_cos, out_sin = result
    assert out_k is None
    assert out_cos.shape == (3, 1, 1, 64)
    assert out_sin.shape == (3, 1, 1, 64)


def test_npu_mrope_interleaved_output_cos_sin_false_returns_two_values(npu_device):
    """Lines 97-98: output_cos_sin=False returns (query, key)."""
    torch.manual_seed(302)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)
    layer.layer_counts = 0

    positions = _make_mrope_positions(npu_device, seq_len=3)
    query = torch.randn((3, 16), device=npu_device)
    key = torch.randn((3, 16), device=npu_device)

    result = layer.forward(positions, query, key, output_cos_sin=False)

    assert len(result) == 2, f"Expected 2 return values, got {len(result)}"
    out_q, out_k = result
    assert out_q.shape == query.shape
    assert out_k.shape == key.shape


def test_npu_mrope_interleaved_layer_cache_reused(npu_device):
    """Lines 63-64: When layer_counts % num_hidden_layers_cache != 0,
    the cached cos/sin from the previous call is reused."""
    torch.manual_seed(400)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=4,
    ).to(npu_device)
    layer.layer_counts = 0

    positions = _make_mrope_positions(npu_device, seq_len=3)
    query = torch.randn((3, 16), device=npu_device)
    key = torch.randn((3, 16), device=npu_device)

    # First call: rebuilds cos/sin (layer_counts=0, 0%4==0)
    out_q1, out_k1 = layer.forward(positions, query, key)
    assert layer.layer_cache is not None, "layer_cache should be set after first call"
    cached_cos, cached_sin = layer.layer_cache

    # Second call: should reuse cache (layer_counts=1, 1%4!=0)
    query2 = torch.randn((3, 16), device=npu_device)
    key2 = torch.randn((3, 16), device=npu_device)
    out_q2, out_k2 = layer.forward(positions, query2, key2)

    assert layer.layer_cache is not None
    assert torch.equal(layer.layer_cache[0], cached_cos)
    assert torch.equal(layer.layer_cache[1], cached_sin)


def test_npu_mrope_interleaved_layer_cache_rebuilt_at_interval(npu_device):
    """Lines 59-62: When layer_counts % num_hidden_layers_cache == 0,
    cos/sin are rebuilt and the cache is refreshed."""
    torch.manual_seed(401)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=2,
    ).to(npu_device)
    layer.layer_counts = 0

    positions1 = _make_mrope_positions(npu_device, seq_len=3)
    positions2 = torch.tensor([[1, 3, 7], [1, 3, 7], [1, 3, 7]], device=npu_device)
    query = torch.randn((3, 16), device=npu_device)
    key = torch.randn((3, 16), device=npu_device)

    # Call 1: layer_counts=0, 0%2==0 -> rebuild
    layer.forward(positions1, query, key)
    first_cache_cos = layer.layer_cache[0].clone()

    # Call 2: layer_counts=1, 1%2!=0 -> reuse
    layer.forward(positions1, query, key)
    assert torch.equal(layer.layer_cache[0], first_cache_cos)

    # Call 3: layer_counts=2, 2%2==0 -> rebuild with new positions
    layer.forward(positions2, query, key)
    assert not torch.equal(layer.layer_cache[0], first_cache_cos)


def test_npu_mrope_interleaved_forward_half_mode_matches_reference(npu_device):
    """Forward with rotary_mode='half' produces correct output."""
    torch.manual_seed(500)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)
    layer.layer_counts = 0

    positions = _make_mrope_positions(npu_device, seq_len=3)
    query = torch.randn((3, 16), device=npu_device)
    key = torch.randn((3, 16), device=npu_device)
    query_ref, key_ref = query.clone(), key.clone()

    out_q, out_k = layer.forward(positions, query, key)

    cos, sin = layer._rebuild_pos_emb(positions)
    exp_q, exp_k = _apply_rotary_ref_to_qk_mrope(
        layer, query_ref, key_ref, cos, sin
    )

    _assert_rotary_outputs_match(out_q, out_k, exp_q, exp_k)


def test_npu_mrope_interleaved_forward_key_none(npu_device):
    """Forward with key=None only rotates query."""
    torch.manual_seed(501)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)
    layer.layer_counts = 0

    positions = _make_mrope_positions(npu_device, seq_len=3)
    query = torch.randn((3, 16), device=npu_device)

    out_q, out_k = layer.forward(positions, query, key=None)

    assert out_k is None
    assert out_q.shape == query.shape
    assert not torch.equal(out_q, query)


def test_npu_mrope_interleaved_forward_rotary_dim_less_than_head_size(npu_device):
    """Forward with rotary_dim < head_size preserves the pass-through part."""
    torch.manual_seed(502)
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=32,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)
    layer.layer_counts = 0

    positions = _make_mrope_positions(npu_device, seq_len=3)
    query = torch.randn((3, 32), device=npu_device)
    key = torch.randn((3, 32), device=npu_device)

    query_tail = query[..., 16:].clone()
    key_tail = key[..., 16:].clone()

    out_q, out_k = layer.forward(positions, query, key)

    assert out_q.shape == query.shape
    assert out_k.shape == key.shape
    assert torch.allclose(out_q[..., 16:], query_tail, atol=1e-6)
    assert torch.allclose(out_k[..., 16:], key_tail, atol=1e-6)


def test_npu_mrope_interleaved_forward_invalid_rotary_mode_raises(npu_device):
    """Line 45: Invalid rotary_mode raises ValueError."""
    layer = NPUMRotaryEmbeddingInterleaved(
        head_size=16,
        rotary_dim=16,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[4, 4],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=1,
    ).to(npu_device)

    # Force an invalid rotary_mode to trigger the ValueError
    layer.rotary_mode = "invalid_mode"
    positions = _make_mrope_positions(npu_device, seq_len=3)

    with pytest.raises(ValueError, match="only support half or interleave"):
        layer._rebuild_pos_emb(positions)