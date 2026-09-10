# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression tests for the vLLM 0.25.1 interleaved MRoPE patch."""

import numpy as np
import torch

from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbeddingInterleaved as VllmMRotaryEmbeddingInterleaved,
)

from omni_npu.vllm_patches.patches.models.openpangu_v1_vl import (
    patch_m_rotary_embedding as patch_mod,
)
from omni_npu.vllm_patches.patch_manager import PatchManager


def test_patch_preserves_vllm_class_identity():
    """OOT dispatch keys off this exact class object and class name."""
    assert patch_mod.rotary_embedding.MRotaryEmbeddingInterleaved is (
        VllmMRotaryEmbeddingInterleaved
    )
    assert patch_mod.MRotaryEmbeddingInterleavedPatch._target is (
        VllmMRotaryEmbeddingInterleaved
    )
    assert patch_mod.RotaryEmbeddingModulePatch._attr_names_to_apply == [
        "get_rope_wrapper"
    ]


def test_interleaved_dimension_mapping_is_inherited_from_vllm_0251():
    mapping = patch_mod.MRotaryEmbeddingInterleaved.get_mrope_interleaved_id_list

    assert mapping(2, 2, 0) == [0, 1, 0, 1]
    assert mapping(2, 3, 3, force_last=True) == [0, 1, 2, 1, 2, 1, 2, 0]


def test_composite_patch_keeps_the_existing_registration_name():
    assert PatchManager.registered_patches["rotary_embeddingPatch"] is (
        patch_mod.RotaryEmbeddingModulePatch
    )
    assert "MRotaryEmbeddingInterleavedPatch" not in (
        PatchManager.registered_patches
    )


def test_non_mrope_path_delegates_to_vllm_0251(monkeypatch):
    sentinel = object()
    calls = []

    def original(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(patch_mod, "_orig_get_rope", original)

    result = patch_mod.RotaryEmbeddingModulePatch.get_rope_wrapper(
        head_size=128,
        rotary_dim=64,
        max_position=4096,
        base=10000,
        rope_scaling=None,
        dtype=torch.bfloat16,
        partial_rotary_factor=0.5,
    )

    assert result is sentinel
    assert calls == [
        (
            128,
            4096,
            True,
            {
                "rope_theta": 10000,
                "rope_type": "default",
                "partial_rotary_factor": 0.5,
                "rope_dim": 64,
            },
            torch.bfloat16,
            None,
        )
    ]


def test_decode_positions_match_uncached_arange():
    out = np.full((3, 8), -1, dtype=np.int64)

    patch_mod.MRotaryEmbeddingPositionPatch.get_next_input_positions_tensor(
        out=out,
        out_offset=2,
        mrope_position_delta=7,
        context_len=11,
        num_new_tokens=4,
    )

    expected = np.arange(18, 22, dtype=np.int64)
    np.testing.assert_array_equal(out[:, 2:6], np.broadcast_to(expected, (3, 4)))


def test_oot_subclass_is_fully_initialized(monkeypatch, default_vllm_config):
    """Reproduce the class-identity condition behind the missing _parameters."""
    target = patch_mod.MRotaryEmbeddingInterleaved
    patched_init = patch_mod.MRotaryEmbeddingInterleavedPatch.__dict__["__init__"]

    class FakeNPUMRotaryEmbeddingInterleaved(target):
        pass

    monkeypatch.setattr(target, "__init__", patched_init)
    monkeypatch.setitem(
        op_registry_oot,
        target.__name__,
        FakeNPUMRotaryEmbeddingInterleaved,
    )

    layer = target(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=[1, 1, 2],
        mrope_interleaved=True,
        rotary_mode="half",
        num_hidden_layers_cache=3,
    )

    assert isinstance(layer, FakeNPUMRotaryEmbeddingInterleaved)
    assert hasattr(layer, "_parameters")
    assert layer.rotary_mode == "half"
    assert layer.num_hidden_layers_cache == 3
