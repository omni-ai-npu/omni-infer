# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Unit tests for omni_cache/attention/backends/mome_ext.py.

These tests use the real vLLM common metadata and the real omni-npu MoME
metadata builder. They only shim unavailable local NPU runtime imports and the
OmniCache active decode-cache seam.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid import (
    PanguNewKVCacheSpecsPatch,
)
from omni_cache.tests.attention.backends.metadata_builder_utils import (
    load_extension_module,
    make_common_attn_metadata,
    make_decode_cache,
    make_vllm_config,
)

PanguNewKVCacheSpecsPatch.apply()

from vllm.v1.kv_cache_interface import MomeSpec


MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "attention" / "backends" / "mome_ext.py"
)


def _load_module(monkeypatch):
    return load_extension_module(monkeypatch, MODULE_PATH, "test_mome_ext")


def _make_mome_spec():
    return MomeSpec(
        block_size=16,
        shapes=((8,), (8,), (8,)),
        dtypes=(torch.bfloat16, torch.bfloat16, torch.bfloat16),
        kernel_size=3,
        num_spec_tokens=1,
    )


def _make_builder(module, vllm_config):
    return module.NPUMomeAttentionMetadataBuilderExt(
        _make_mome_spec(),
        ["layer0"],
        vllm_config,
        torch.device("cpu"),
    )


def test_build_rewrites_decode_cache_indices(monkeypatch):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    monkeypatch.setenv("ENABLE_HOST_MAPPING", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        module, "get_active_decode_cache", lambda: make_decode_cache([2, 4, 6])
    )
    builder = _make_builder(module, make_vllm_config())
    common_meta = make_common_attn_metadata(
        first_column=[11, 22, 33], query_lens=[1, 1, 1]
    )

    metadata = builder.build(0, common_meta)

    assert metadata.cache_indices.tolist() == [3, 5, 7]
    assert metadata.decode.cache_indices.tolist() == [3, 5, 7]
    assert metadata.prefill is None


def test_update_block_table_rewrites_decode_only(monkeypatch):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    monkeypatch.setenv("ENABLE_HOST_MAPPING", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        module, "get_active_decode_cache", lambda: make_decode_cache([2, 4])
    )
    builder = _make_builder(module, make_vllm_config())
    metadata = builder.build(
        0, make_common_attn_metadata(first_column=[11, 22], query_lens=[1, 1])
    )

    updated = builder.update_block_table(
        metadata,
        torch.tensor([[91, 0], [92, 0]], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
    )

    assert updated.cache_indices.tolist() == [3, 5]
    assert updated.decode.cache_indices.tolist() == [3, 5]


def test_update_block_table_preserves_prefill_only_batch(monkeypatch):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    monkeypatch.setenv("ENABLE_HOST_MAPPING", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "get_active_decode_cache", lambda: None)
    builder = _make_builder(module, make_vllm_config())
    metadata = builder.build(
        0, make_common_attn_metadata(first_column=[11, 22], query_lens=[3, 5])
    )

    updated = builder.update_block_table(
        metadata,
        torch.tensor([[91, 0], [92, 0]], dtype=torch.int32),
        torch.tensor([], dtype=torch.int32),
    )

    assert updated.cache_indices.tolist() == [91, 92]
    assert updated.decode is None
    assert updated.prefill.cache_indices.tolist() == [91, 92]


def test_build_skips_when_no_active_decode_cache(monkeypatch):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    monkeypatch.setenv("ENABLE_HOST_MAPPING", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "get_active_decode_cache", lambda: None)
    builder = _make_builder(module, make_vllm_config())

    metadata = builder.build(
        0, make_common_attn_metadata(first_column=[11, 22], query_lens=[1, 1])
    )

    assert metadata.cache_indices.tolist() == [11, 22]
    assert metadata.decode.cache_indices.tolist() == [11, 22]


def test_build_uses_sequential_dummy_lanes_when_lane_mapping_missing(
    monkeypatch,
):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    monkeypatch.setenv("ENABLE_HOST_MAPPING", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        module, "get_active_decode_cache", lambda: make_decode_cache(None)
    )
    monkeypatch.setattr(module, "resolve_batch_idx_reqs", lambda cache, num_reqs: None)
    builder = _make_builder(module, make_vllm_config())

    metadata = builder.build(
        0, make_common_attn_metadata(first_column=[11, 22], query_lens=[1, 1])
    )

    assert metadata.cache_indices.tolist() == [1, 2]
    assert metadata.decode.cache_indices.tolist() == [1, 2]


def test_build_preserves_prefill_only_batch(monkeypatch):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    monkeypatch.setenv("ENABLE_HOST_MAPPING", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "get_active_decode_cache", lambda: None)
    builder = _make_builder(module, make_vllm_config())

    metadata = builder.build(
        0, make_common_attn_metadata(first_column=[11, 22], query_lens=[3, 5])
    )

    assert metadata.cache_indices.tolist() == [11, 22]
    assert metadata.decode is None
    assert metadata.prefill.cache_indices.tolist() == [11, 22]


def test_build_skips_when_prefix_caching_enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    monkeypatch.setenv("ENABLE_HOST_MAPPING", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        module, "get_active_decode_cache", lambda: make_decode_cache([2, 4])
    )
    builder = _make_builder(module, make_vllm_config(enable_prefix_caching=True))

    metadata = builder.build(
        0, make_common_attn_metadata(first_column=[11, 22], query_lens=[1, 1])
    )

    assert metadata.cache_indices[:, 0].tolist() == [11, 22]
    assert metadata.decode.cache_indices[:, 0].tolist() == [11, 22]
