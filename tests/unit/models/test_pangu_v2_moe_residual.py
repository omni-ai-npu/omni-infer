# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit


class _IdentityAttn:
    prefix = "attn"

    def __call__(self, hs, _cos, _sin):
        return hs


def _identity(x):
    return x


def _return_last_arg(*args, **_kwargs):
    return args[-1]


def _bare_decoder_layer(*, use_mhc: bool = False):
    """Uninitialized OpenPanguV2DecoderLayer with identity attn/mlp/norm."""
    layer = model_mod.OpenPanguV2DecoderLayer.__new__(
        model_mod.OpenPanguV2DecoderLayer
    )
    layer.use_mhc = use_mhc
    layer.use_post_norm = False
    layer.side_stream = None
    layer.hidden_size = 4
    layer.mhc_num_stream = 1
    layer.input_layernorm = _identity
    layer.post_attention_layernorm = _identity
    layer.pre_mlp_layernorm = _identity
    layer.post_mlp_layernorm = _identity
    layer.self_attn = _IdentityAttn()
    layer.mlp = _identity
    return layer


def test_mhc_head_reuses_hidden_states_as_residual():
    """mhc_head residual is the input tensor, not a clone (line 1701)."""
    layer = _bare_decoder_layer(use_mhc=False)
    hidden_states = torch.randn(2, 4)

    out, residual, h_post, h_res, sk_event = layer.mhc_head(hidden_states)

    assert residual is hidden_states
    assert torch.equal(out, hidden_states)
    assert h_post is None
    assert h_res is None
    assert sk_event is None


def test_mhc_head_reuses_residual_when_mhc_enabled():
    layer = _bare_decoder_layer(use_mhc=True)
    hidden_states = torch.randn(2, 4)
    mixed = torch.randn(2, 4)
    layer.attn_mhc_module = MagicMock()
    layer.attn_mhc_module.mhc_pre.return_value = (mixed, "h_post", "h_res")
    layer.attn_mhc_module.mhc_sinkhorn.return_value = "h_res_sk"

    out, residual, h_post, h_res, sk_event = layer.mhc_head(hidden_states)

    assert residual is hidden_states
    assert torch.equal(out, mixed)
    assert h_post == "h_post"
    assert h_res == "h_res_sk"
    assert sk_event is None


def test_forward_naive_reuses_hidden_states_as_residual():
    """_forward_naive attn/FFN residuals alias the live tensors (lines 2074, 2106)."""
    layer = _bare_decoder_layer(use_mhc=False)
    hidden_states = torch.ones(2, 4)
    cos = torch.zeros(2, 2)
    sin = torch.zeros(2, 2)

    out = layer._forward_naive(hidden_states, cos, sin)

    # identity attn/mlp + residual add: hs -> 2*hs -> 4*hs
    torch.testing.assert_close(out, hidden_states * 4)


def test_forward_naive_passes_uncloned_residual_into_mhc_post():
    layer = _bare_decoder_layer(use_mhc=True)
    hidden_states = torch.randn(2, 4)
    attn_residuals = []
    mlp_residuals = []

    layer.attn_mhc_task_key = "attn_key"
    layer.mlp_mhc_task_key = "mlp_key"
    layer.attn_mhc_module = MagicMock()
    layer.mlp_mhc_module = MagicMock()
    layer.attn_mhc_module.mhc_pre.return_value = (hidden_states, "attn_post", "attn_res")
    layer.mlp_mhc_module.mhc_pre.return_value = (hidden_states, "mlp_post", "mlp_res")

    def attn_post(hs, h_post, residual, h_res):
        attn_residuals.append(residual)
        return hs

    def mlp_post(hs, h_post, residual, h_res):
        mlp_residuals.append(residual)
        return hs

    layer.attn_mhc_module.mhc_post.side_effect = attn_post
    layer.mlp_mhc_module.mhc_post.side_effect = mlp_post

    with patch.object(
        model_mod, "maybe_register_mhc_task", side_effect=_return_last_arg
    ), patch.object(
        model_mod, "resolve_mhc_h_res", side_effect=_return_last_arg
    ):
        out = layer._forward_naive(
            hidden_states,
            torch.zeros(2, 2),
            torch.zeros(2, 2),
        )

    assert attn_residuals == [hidden_states]
    assert attn_residuals[0] is hidden_states
    assert mlp_residuals == [hidden_states]
    assert mlp_residuals[0] is hidden_states
    assert out is hidden_states


def test_dense_mlp_sk_scope(monkeypatch):
    monkeypatch.setattr(model_mod, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(model_mod, "MergedColumnParallelLinear", lambda *a, **k: _identity)
    monkeypatch.setattr(model_mod, "RowParallelLinear", lambda *a, **k: _identity)
    monkeypatch.setattr(model_mod, "SiluAndMul", lambda: _identity)
    mlp = model_mod.OpenPanguV2MLP(4, 8, "silu", prefix="model.layers.0.mlp")
    x = torch.ones(2, 4)
    assert torch.equal(mlp.forward(x), x)
    assert mlp._fuse_dense_mlp is True
    assert mlp._sk_dense_name == "dense_mlp_0"


def test_mhc_post_pre_sk_scope():
    layer = _bare_decoder_layer(use_mhc=True)
    layer.use_mhc_fusion_op = False
    layer.layer_idx = 0
    hidden = torch.ones(2, 4)
    post_mhc = MagicMock()
    post_mhc.mhc_post.return_value = hidden
    pre_mhc = MagicMock()
    pre_mhc.mhc_pre.return_value = (hidden, "h_post", "h_res")
    pre_mhc.mhc_sinkhorn.return_value = "h_res"

    out, residual, h_post, h_res, sk_event = layer.mhc_sandwich_norm_post_pre(
        hidden, hidden, None, None, _identity, post_mhc, None, pre_mhc, _identity,
        is_model_tail=True,
    )

    post_mhc.mhc_post.assert_called_once()
    assert out is hidden
    assert residual is None
    assert h_post == "h_post"
    assert h_res == "h_res"
    assert sk_event is None
