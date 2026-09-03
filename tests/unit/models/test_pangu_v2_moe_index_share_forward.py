# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cover IndexShare-related decoder/model forward after dropping layer-arg buffer.

The Index Share adaptation stores topk in attention metadata instead of threading
``topk_indices_buffer`` through OpenPanguV2DecoderLayer / OpenPanguV2Model.
These tests pin the no-buffer call sites that show up in branch diff coverage.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from omni_npu.v1.models.pangu import pangu_v2_moe as pangu_moe


HIDDEN = 4
TOKENS = 2


def _make_decoder_layer(*, use_mhc: bool = False):
    layer = pangu_moe.OpenPanguV2DecoderLayer.__new__(pangu_moe.OpenPanguV2DecoderLayer)
    layer.use_mhc = use_mhc
    layer.side_stream = None
    layer.layer_idx = 0
    layer.first_k_dense_replace = 0
    layer.hidden_size = HIDDEN
    layer.mhc_num_stream = 1
    layer.use_post_norm = False
    layer.self_attn = MagicMock(
        return_value=torch.ones(TOKENS, HIDDEN),
        prefix="model.layers.0.self_attn",
        pre_epilog_callback=None,
    )
    layer.input_layernorm = MagicMock(side_effect=lambda x: x)
    layer.post_attention_layernorm = MagicMock(side_effect=lambda x: x)
    layer.pre_mlp_layernorm = MagicMock(side_effect=lambda x: x)
    layer.post_mlp_layernorm = MagicMock(side_effect=lambda x: x)
    layer.mlp = MagicMock(side_effect=lambda x: x)
    layer.attn_mhc_module = MagicMock()
    layer.mlp_mhc_module = MagicMock()
    layer.block_post_layernorm = MagicMock(side_effect=lambda x: x)
    layer._tail_refs = (None, None, True)
    layer.mhc_sandwich_norm_post_pre = MagicMock(
        side_effect=lambda hs, residual, h_post, h_res, *args, **kwargs: (
            hs, residual, h_post, h_res, kwargs.get("sk_event")
        )
    )
    return layer


def _make_model(monkeypatch, layer, *, enable_mhc_multistream):
    model = pangu_moe.OpenPanguV2Model.__new__(pangu_moe.OpenPanguV2Model)
    model.use_mhc = True
    model.hidden_size = HIDDEN
    model.mhc_num_stream = 1
    model.need_tp_padding = False
    model.start_layer = 0
    model.end_layer = 1
    model.embed_tokens = MagicMock(return_value=torch.randn(TOKENS, HIDDEN))
    model.norm = MagicMock(side_effect=lambda x: x)
    model.layers = [layer]

    layer.self_attn = SimpleNamespace(
        rotary_emb=SimpleNamespace(
            cos_cached=torch.zeros(8, 2),
            sin_cached=torch.zeros(8, 2),
        )
    )
    monkeypatch.setattr(
        pangu_moe,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        pangu_moe,
        "model_extra_config",
        SimpleNamespace(
            parall_config=SimpleNamespace(ena_seq_parallel=False),
            operator_opt_config=SimpleNamespace(
                enable_mhc_multistream=enable_mhc_multistream,
                use_mhc_fusion_op=False,
            ),
        ),
    )
    return model


@pytest.mark.unit
def test_decoder_layer_forward_calls_attn_without_topk_buffer():
    """forward() must call self_attn(cos,sin) only and return a 5-tuple."""
    layer = _make_decoder_layer(use_mhc=False)
    hidden = torch.randn(TOKENS, HIDDEN)
    cos = torch.randn(TOKENS, 2)
    sin = torch.randn(TOKENS, 2)

    out = layer.forward(hidden, None, None, None, cos, sin, None)

    layer.self_attn.assert_called_once_with(hidden, cos, sin)
    assert len(out) == 5
    assert torch.equal(out[0], torch.ones(TOKENS, HIDDEN))


@pytest.mark.unit
def test_decoder_layer_forward_naive_without_topk_buffer():
    """_forward_naive() returns only hidden_states after IndexShare refactor."""
    layer = _make_decoder_layer(use_mhc=False)
    hidden = torch.randn(TOKENS, HIDDEN)
    cos = torch.randn(TOKENS, 2)
    sin = torch.randn(TOKENS, 2)

    out = layer._forward_naive(hidden, cos, sin)

    layer.self_attn.assert_called_once_with(hidden, cos, sin)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (TOKENS, HIDDEN)


@pytest.mark.unit
def test_model_forward_naive_mhc_multistream_path(monkeypatch):
    """Cube-side MHC multistream path calls _forward_naive without buffer."""
    layer = MagicMock()
    layer._forward_naive = MagicMock(return_value=torch.ones(TOKENS, 1, HIDDEN))
    model = _make_model(monkeypatch, layer, enable_mhc_multistream=True)
    model.merge_mhc_module = MagicMock(
        mhc_pre=MagicMock(
            return_value=(torch.ones(TOKENS, 1, HIDDEN), None, None)
        )
    )

    positions = torch.zeros(TOKENS, dtype=torch.long)
    out = model.forward(
        input_ids=torch.zeros(TOKENS, dtype=torch.long),
        positions=positions,
        intermediate_tensors=None,
    )

    layer._forward_naive.assert_called_once()
    args, _kwargs = layer._forward_naive.call_args
    assert len(args) == 3  # hidden_states, cos, sin — no topk buffer
    assert out.shape == (TOKENS, 1, HIDDEN)


@pytest.mark.unit
def test_model_forward_threaded_path_without_topk_buffer(monkeypatch):
    """Default threaded MHC path calls layer.forward without buffer."""
    layer = MagicMock()
    layer.mhc_head = MagicMock(
        return_value=(torch.ones(TOKENS, 1, HIDDEN), None, None, None, None)
    )
    layer.return_value = (
        torch.ones(TOKENS, 1, HIDDEN), None, None, None, None
    )
    model = _make_model(monkeypatch, layer, enable_mhc_multistream=False)

    positions = torch.zeros(TOKENS, dtype=torch.long)
    out = model.forward(
        input_ids=torch.zeros(TOKENS, dtype=torch.long),
        positions=positions,
        intermediate_tensors=None,
    )

    layer.assert_called_once()
    args, kwargs = layer.call_args
    # hidden, residual, h_post, h_res, cos, sin, sk_event — no topk buffer
    assert len(args) == 7
    assert "topk_indices_buffer" not in kwargs
    assert out.shape == (TOKENS, 1, HIDDEN)
