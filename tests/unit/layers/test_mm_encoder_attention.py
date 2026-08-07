# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def npu_mm_encoder_attention_module(monkeypatch):
    registry = {}

    class MMEncoderAttention:
        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float | None,
            num_kv_heads: int | None = None,
        ):
            self.num_heads = num_heads
            self.head_size = head_size
            self.scale = scale
            self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
            self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        @classmethod
        def register_oot(cls, oot_cls):
            oot_cls.name = cls.__name__
            registry[cls.__name__] = oot_cls
            return oot_cls

        def maybe_reshape_qkv_to_4d(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            bsz: int,
            q_len: int,
            kv_len: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            query = query.view(bsz, q_len, self.num_heads, self.head_size)
            key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
            value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)
            return query, key, value

        def forward_native(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return query, key, value, cu_seqlens, max_seqlen

    fake_module = types.ModuleType(
        "vllm.model_executor.layers.attention.mm_encoder_attention"
    )
    fake_module.MMEncoderAttention = MMEncoderAttention
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        fake_module,
    )

    module_name = "omni_npu.layers.attention.mm_encoder_attention"
    sys.modules.pop(module_name, None)
    module_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "omni_npu"
        / "layers"
        / "attention"
        / "mm_encoder_attention.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module, registry


@pytest.mark.unit
def test_mm_encoder_attention_registers_oot_class(npu_mm_encoder_attention_module):
    module, registry = npu_mm_encoder_attention_module

    assert registry["MMEncoderAttention"] is module.NPUMMEncoderAttention
    assert issubclass(
        module.NPUMMEncoderAttention,
        sys.modules[
            "vllm.model_executor.layers.attention.mm_encoder_attention"
        ].MMEncoderAttention,
    )


@pytest.mark.unit
def test_mm_encoder_attention_forward_oot_prefers_fia_path(
    npu_mm_encoder_attention_module,
):
    module, _ = npu_mm_encoder_attention_module
    attn = module.NPUMMEncoderAttention(num_heads=8, head_size=64, scale=0.125)
    query = torch.randn(2, 4, 512)
    key = torch.randn(2, 4, 512)
    value = torch.randn(2, 4, 512)
    expected = torch.randn_like(query)

    with patch.object(attn, "_forward_fia", return_value=expected) as mock_fia:
        result = attn.forward_oot(query, key, value)

    mock_fia.assert_called_once_with(query, key, value)
    assert torch.equal(result, expected)


@pytest.mark.unit
def test_mm_encoder_attention_forward_oot_uses_native_for_packed_inputs(
    npu_mm_encoder_attention_module,
):
    module, _ = npu_mm_encoder_attention_module
    attn = module.NPUMMEncoderAttention(num_heads=8, head_size=64, scale=0.125)
    query = torch.randn(2, 4, 512)
    key = torch.randn(2, 4, 512)
    value = torch.randn(2, 4, 512)
    cu_seqlens = torch.tensor([0, 2, 4])
    max_seqlen = torch.tensor(2)
    expected = torch.randn_like(query)

    with patch.object(attn, "forward_native", return_value=expected) as mock_native:
        result = attn.forward_oot(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    mock_native.assert_called_once_with(query, key, value, cu_seqlens, max_seqlen)
    assert torch.equal(result, expected)


@pytest.mark.unit
def test_mm_encoder_attention_forward_native_delegates_to_upstream(
    npu_mm_encoder_attention_module,
):
    module, _ = npu_mm_encoder_attention_module
    attn = module.NPUMMEncoderAttention(num_heads=2, head_size=4, scale=0.25)
    query = torch.randn(1, 3, 8)
    key = torch.randn(1, 4, 8)
    value = torch.randn(1, 4, 8)
    cu_seqlens = torch.tensor([0, 2, 4])
    max_seqlen = torch.tensor(2)

    result = attn.forward_native(
        query,
        key,
        value,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )

    assert result == (query, key, value, cu_seqlens, max_seqlen)


@pytest.mark.unit
def test_mm_encoder_attention_fia_rejects_non_npu_tensors(
    npu_mm_encoder_attention_module,
    monkeypatch,
):
    module, _ = npu_mm_encoder_attention_module
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    attn = module.NPUMMEncoderAttention(num_heads=2, head_size=4, scale=0.25)

    class TensorOnCPU:
        device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="FIA path requires NPU tensors"):
        attn._forward_fia(TensorOnCPU(), object(), object())


@pytest.mark.unit
def test_mm_encoder_attention_fia_uses_npu_kernel(
    npu_mm_encoder_attention_module,
    monkeypatch,
):
    module, _ = npu_mm_encoder_attention_module
    calls = {}

    fake_torch_npu = types.ModuleType("torch_npu")

    def npu_fused_infer_attention_score_v2(**kwargs):
        calls.update(kwargs)
        return (torch.ones(2, 3, 2, 4),)

    fake_torch_npu.npu_fused_infer_attention_score_v2 = (
        npu_fused_infer_attention_score_v2
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    class TensorOnNPU:
        device = types.SimpleNamespace(type="npu")

        def __init__(self, shape: tuple[int, ...], dim: int):
            self._shape = shape
            self._dim = dim

        def size(self, dim: int | None = None):
            if dim is None:
                return self._shape
            return self._shape[dim]

        def dim(self) -> int:
            return self._dim

    attn = module.NPUMMEncoderAttention(
        num_heads=2,
        head_size=4,
        scale=0.5,
        num_kv_heads=1,
    )

    def maybe_reshape_qkv_to_4d(query, key, value, bsz, q_len, kv_len):
        assert (bsz, q_len, kv_len) == (2, 3, 5)
        return (
            torch.randn(2, 3, 2, 4),
            torch.randn(2, 5, 1, 4),
            torch.randn(2, 5, 1, 4),
        )

    monkeypatch.setattr(attn, "maybe_reshape_qkv_to_4d", maybe_reshape_qkv_to_4d)

    result = attn._forward_fia(
        TensorOnNPU((2, 3, 8), 3),
        TensorOnNPU((2, 5, 4), 3),
        object(),
    )

    assert result.shape == (2, 3, 8)
    assert calls["num_query_heads"] == 2
    assert calls["num_key_value_heads"] == 1
    assert calls["input_layout"] == "BSND"
    assert calls["softmax_scale"] == 0.5
    assert calls["actual_seq_qlen"] == [3, 3]
    assert calls["actual_seq_kvlen"] == [5, 5]
