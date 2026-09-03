# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 KV reshape injection: binding, and delegation to the backend."""

from __future__ import annotations

import pytest

from omni_npu.vllm_patches.usefull_patch.common import patch_mrv2_attn_utils as patch_mod
from omni_npu.worker.npu import attn_utils as npu_attn_utils


@pytest.fixture
def applied():
    cls = patch_mod.MRv2ReshapeKvCachePatch
    target = cls._target
    saved = getattr(target, "_reshape_kv_cache")
    owners = dict(getattr(target, "_omni_npu_applied_patches", {}))
    target._omni_npu_applied_patches = {}
    try:
        cls.apply()
        yield target
    finally:
        target._reshape_kv_cache = saved
        target._omni_npu_applied_patches = owners


class _Spec:
    page_size_bytes = 4


class _Backend:
    def __init__(self, hook=None):
        if hook is not None:
            self.reshape_kv_cache = hook


class _Group:
    def __init__(self, backend, layer_names=("layer.0",)):
        self.backend = backend
        self.layer_names = list(layer_names)
        self.kv_cache_spec = _Spec()


def test_reshape_kv_cache_is_replaced(applied):
    assert applied._reshape_kv_cache is npu_attn_utils.reshape_kv_cache


def test_mrv1_reshape_helper_is_untouched(applied):
    """MRv1 imports _reshape_attention_kv_cache from here, a different function."""
    assert (
        applied._reshape_attention_kv_cache.__module__
        == "vllm.v1.worker.gpu.attn_utils"
    )


def _run(monkeypatch, group, raw, upstream_result):
    monkeypatch.setattr(
        npu_attn_utils, "_ORIGINAL", lambda **kwargs: dict(upstream_result)
    )
    return npu_attn_utils.reshape_kv_cache(
        attn_groups=[group],
        kv_cache_raw_tensors={"layer.0": raw},
        cache_dtype="auto",
        kernel_block_sizes=[128],
        shared_kv_cache_layers={},
    )


def test_backend_hook_overrides_the_upstream_result(monkeypatch):
    import torch

    raw = torch.zeros(8, dtype=torch.int8)
    seen = {}

    def hook(raw_tensor, num_blocks, spec):
        seen.update(num_blocks=num_blocks, spec=spec)
        return "backend-view"

    out = _run(monkeypatch, _Group(_Backend(hook)), raw, {"layer.0": "upstream"})

    assert out["layer.0"] == "backend-view"
    assert seen["num_blocks"] == 2  # 8 bytes / page_size_bytes=4


def test_without_the_hook_the_upstream_result_stands(monkeypatch):
    import torch

    out = _run(
        monkeypatch, _Group(_Backend()), torch.zeros(8, dtype=torch.int8),
        {"layer.0": "upstream"},
    )

    assert out["layer.0"] == "upstream"


def test_backend_hook_rejects_a_partial_cache_page(monkeypatch):
    import torch

    raw = torch.zeros(6, dtype=torch.int8)

    with pytest.raises(ValueError, match="not divisible"):
        _run(
            monkeypatch,
            _Group(_Backend(lambda *args: None)),
            raw,
            {"layer.0": "upstream"},
        )


def test_shared_layer_with_a_backend_hook_raises(monkeypatch):
    import torch

    monkeypatch.setattr(npu_attn_utils, "_ORIGINAL", lambda **kwargs: {})
    with pytest.raises(RuntimeError, match="shared"):
        npu_attn_utils.reshape_kv_cache(
            attn_groups=[_Group(_Backend(lambda *a: None))],
            kv_cache_raw_tensors={"layer.0": torch.zeros(8, dtype=torch.int8)},
            cache_dtype="auto",
            kernel_block_sizes=[128],
            shared_kv_cache_layers={"layer.0": "layer.1"},
        )
