# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from omni_npu.layers.rotary_embedding.common import get_cos_sin
from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit


def test_cached_get_cos_sin_is_value_identical_to_direct_lookup():
    """The cached RoPE API must not change ordinary language RoPE values."""
    cos_cached = torch.randn(16, 8, dtype=torch.float32)
    sin_cached = torch.randn(16, 8, dtype=torch.float32)
    positions = torch.tensor([0, 3, 7, 15], dtype=torch.long)

    cos, sin = get_cos_sin(cos_cached, sin_cached, positions)

    expected_cos = cos_cached.index_select(0, positions)
    expected_sin = sin_cached.index_select(0, positions)
    assert torch.equal(cos.view_as(expected_cos), expected_cos)
    assert torch.equal(sin.view_as(expected_sin), expected_sin)


def test_model_forward_uses_rotary_get_cos_sin(monkeypatch):
    """VL MRoPE only exposes get_cos_sin; model forward must use that API."""
    tokens = 3
    hidden_size = 4
    positions = torch.arange(tokens, dtype=torch.long).repeat(3, 1)
    hidden_states = torch.randn(tokens, hidden_size)
    cos = torch.randn(tokens, 1, 1, hidden_size)
    sin = torch.randn(tokens, 1, 1, hidden_size)

    rotary_emb = SimpleNamespace(get_cos_sin=MagicMock(return_value=(cos, sin)))
    layer = MagicMock()
    layer.self_attn = SimpleNamespace(rotary_emb=rotary_emb)
    layer.mhc_head.return_value = (hidden_states, None, None, None, None)
    layer.return_value = (hidden_states, None, None, None, None)

    model = SimpleNamespace()
    model.use_mhc = False
    model.need_tp_padding = False
    model.cos_cached = None
    model.sin_cached = None
    model.start_layer = 0
    model.end_layer = 1
    model.layers = [layer]
    model.embed_tokens = MagicMock(return_value=hidden_states)
    model.config = SimpleNamespace(index_topk=2)

    monkeypatch.setattr(
        model_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        model_mod,
        "model_extra_config",
        SimpleNamespace(parall_config=SimpleNamespace(ena_seq_parallel=False)),
    )
    monkeypatch.setattr(model_mod, "high_throughout", lambda: False)

    output = model_mod.OpenPanguV2Model.forward(
        model,
        input_ids=torch.zeros(tokens, dtype=torch.long),
        positions=positions,
        intermediate_tensors=None,
    )

    rotary_emb.get_cos_sin.assert_called_once_with(positions)
    layer.assert_called_once()
    layer_args = layer.call_args.args
    assert layer_args[4] is cos
    assert layer_args[5] is sin
    assert output is hidden_states
