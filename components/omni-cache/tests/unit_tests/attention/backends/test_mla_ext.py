# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for omni_cache/attention/backends/mla_ext.py.

These tests intentionally use the real vLLM metadata dataclasses and the real
omni-npu MLA metadata builder. The only test double is OmniCache's active
decode-cache lookup.
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
    patch_current_vllm_config,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

PanguNewKVCacheSpecsPatch.apply()

from vllm.v1.kv_cache_interface import ShareKVSlidingWindowSpec


MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "attention" / "backends" / "mla_ext.py"
)


def _load_module(monkeypatch):
    return load_extension_module(monkeypatch, MODULE_PATH, "test_mla_ext")


def _patch_missing_platform_device(monkeypatch):
    import omni_npu.attention.backends.mla as npu_mla

    if not getattr(npu_mla.current_platform, "device_type", None):
        monkeypatch.setattr(
            npu_mla.current_platform, "device_type", "cpu", raising=False
        )


def _make_mla_spec(block_size: int = 128, sliding_window: int = 1536):
    return ShareKVSlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        sliding_window=sliding_window,
    )


def _make_builder(monkeypatch, module, vllm_config):
    patch_current_vllm_config(monkeypatch, vllm_config)
    _patch_missing_platform_device(monkeypatch)
    return module.NPUMLAMetadataBuilderExt(
        _make_mla_spec(),
        ["layer0"],
        vllm_config,
        torch.device("cpu"),
    )


def test_build_rewrites_all_tokens_for_multi_token_decode(monkeypatch):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        module, "get_active_decode_cache", lambda: make_decode_cache([0])
    )
    builder = _make_builder(
        monkeypatch, module, make_vllm_config(num_speculative_tokens=1)
    )
    common_meta = make_common_attn_metadata(
        seq_lens=[1475],
        query_lens=[2],
        slot_mapping=[9999, 9998],
        max_blocks=16,
    )

    metadata = builder.build(0, common_meta)

    assert metadata.prefill is None
    assert metadata.slot_mapping.tolist() == [1601, 1602]
    assert common_meta.block_table_tensor[0, :16].tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        1,
        0,
        0,
        0,
    ]
    assert metadata.decode.block_table[0, :16].tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        1,
        0,
        0,
        0,
    ]


def test_build_rewrites_single_token_and_preserves_padded_tail(
    monkeypatch,
):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        module, "get_active_decode_cache", lambda: make_decode_cache([0])
    )
    builder = _make_builder(
        monkeypatch, module, make_vllm_config(num_speculative_tokens=1)
    )
    common_meta = make_common_attn_metadata(
        seq_lens=[1473],
        query_lens=[1],
        slot_mapping=[9999, PAD_SLOT_ID],
        max_blocks=16,
    )

    metadata = builder.build(0, common_meta)

    assert metadata.prefill is None
    assert metadata.slot_mapping.tolist() == [1600, PAD_SLOT_ID]


def test_build_rewrites_nonsequential_lane_decode_batch(monkeypatch):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "get_active_decode_cache",
        lambda: make_decode_cache([0, 2, 4, 1, 3, 5]),
    )
    builder = _make_builder(
        monkeypatch, module, make_vllm_config(num_speculative_tokens=1)
    )
    common_meta = make_common_attn_metadata(
        seq_lens=[1473, 1476, 1480, 1482, 1487, 1490],
        query_lens=[2, 1, 2, 2, 1, 2],
        slot_mapping=[-9] * 10,
        max_blocks=16,
    )

    metadata = builder.build(0, common_meta)

    assert metadata.prefill is None
    assert metadata.slot_mapping.tolist() == [
        1599,
        1600,
        4931,
        8262,
        8263,
        3272,
        3273,
        6606,
        9936,
        9937,
    ]
    assert common_meta.block_table_tensor[:, 11].tolist() == [12, 38, 64, 25, 51, 77]
    assert metadata.decode.block_table[:, 11].tolist() == [12, 38, 64, 25, 51, 77]


def test_build_routes_unresolved_lane_to_pad_slot(monkeypatch):
    monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
    module = _load_module(monkeypatch)
    monkeypatch.setattr(
        module, "get_active_decode_cache", lambda: make_decode_cache([0, -1])
    )
    builder = _make_builder(
        monkeypatch, module, make_vllm_config(num_speculative_tokens=1)
    )
    common_meta = make_common_attn_metadata(
        seq_lens=[1473, 1473],
        query_lens=[1, 1],
        slot_mapping=[9999, 9998],
        max_blocks=16,
    )

    metadata = builder.build(0, common_meta)

    assert metadata.prefill is None
    assert metadata.slot_mapping.tolist() == [1600, PAD_SLOT_ID]
    assert common_meta.block_table_tensor[0, :14].tolist() != [0] * 14
    assert common_meta.block_table_tensor[1].tolist() == [0] * 16
    assert metadata.decode.block_table[1].tolist() == [0] * 16
