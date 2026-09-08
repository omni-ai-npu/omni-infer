# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the ``return_h_in_f32`` output contract.

``mhc_sandwich_norm_post_pre`` normally returns a bf16 tensor. When the next
block needs an fp32 copy of the hidden states (``return_h_in_f32=True``) it
must instead return a ``{"hidden_states_bf16", "hidden_states_fp32"}`` dict.
Both the fusion-op path and the eager path have to honour that contract.

The fusion operator is mocked; only the call contract and the output shape are
asserted, never numerics (the operator package is replaceable).
"""

from types import SimpleNamespace

import pytest
import torch

from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit


def _identity(x):
    return x


def _bare_layer(*, use_mhc, use_mhc_fusion_op):
    layer = model_mod.OpenPanguV2DecoderLayer.__new__(
        model_mod.OpenPanguV2DecoderLayer
    )
    layer.use_mhc = use_mhc
    layer.use_mhc_fusion_op = use_mhc_fusion_op
    layer.side_stream = None
    layer.hidden_size = 4
    layer.mhc_num_stream = 1
    layer.layer_idx = 0
    return layer


def _pre_mhc_module():
    return SimpleNamespace(
        phi_weight_pre=torch.ones(4),
        branch_alpha_pre=torch.ones(4),
        branch_beta_pre=torch.zeros(4),
        norm_eps=1e-6,
        hc_eps=1e-6,
    )


def test_fusion_path_returns_bf16_and_fp32_dict(monkeypatch):
    """Fusion op path: the third operator output is exposed as hidden_states_fp32."""
    layer = _bare_layer(use_mhc=True, use_mhc_fusion_op=True)
    hidden_states = torch.ones(2, 4, dtype=torch.bfloat16)
    residual = torch.ones(2, 4, dtype=torch.bfloat16)

    fused_hidden = torch.zeros(2, 4, dtype=torch.bfloat16)
    fused_residual = torch.zeros(2, 4, dtype=torch.bfloat16)
    fused_fp32 = torch.zeros(2, 4, dtype=torch.float32)
    calls = {}

    def fake_fusion(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return fused_hidden, fused_residual, fused_fp32

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(
            npu_ai_infra_mhc_sandwich_norm_post_preonly_v2=fake_fusion
        ),
        raising=False,
    )

    pre_mhc = _pre_mhc_module()
    out, out_residual, h_post, h_res, sk_event = layer.mhc_sandwich_norm_post_pre(
        hidden_states,
        residual,
        None,
        None,
        SimpleNamespace(weight_fp32=torch.ones(4)),
        SimpleNamespace(),
        None,
        pre_mhc,
        SimpleNamespace(weight_fp32=torch.ones(4)),
        is_model_tail=True,
        return_h_in_f32=True,
    )

    # The operator is asked for the fp32 copy explicitly.
    assert calls["kwargs"]["return_h_in_f32"] is True
    # gamma_2 is None when there is no block norm module.
    assert calls["kwargs"]["gamma_2"] is None
    assert calls["kwargs"]["norm_eps"] == pre_mhc.norm_eps
    # Both dtypes are handed back under the documented keys.
    assert set(out) == {"hidden_states_bf16", "hidden_states_fp32"}
    assert out["hidden_states_bf16"] is fused_hidden
    assert out["hidden_states_fp32"] is fused_fp32
    assert out["hidden_states_fp32"].dtype == torch.float32
    # Model tail drops the mhc carry-over state.
    assert out_residual is fused_residual
    assert h_post is None and h_res is None and sk_event is None


def test_fusion_path_forwards_block_norm_weight(monkeypatch):
    """gamma_2 is the block-norm fp32 weight when a block norm module exists."""
    layer = _bare_layer(use_mhc=True, use_mhc_fusion_op=True)
    block_weight = torch.ones(4, dtype=torch.float32)
    calls = {}

    def fake_fusion(*args, **kwargs):
        calls["kwargs"] = kwargs
        return (
            torch.zeros(2, 4, dtype=torch.bfloat16),
            torch.zeros(2, 4, dtype=torch.bfloat16),
            torch.zeros(2, 4, dtype=torch.float32),
        )

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(
            npu_ai_infra_mhc_sandwich_norm_post_preonly_v2=fake_fusion
        ),
        raising=False,
    )

    layer.mhc_sandwich_norm_post_pre(
        torch.ones(2, 4, dtype=torch.bfloat16),
        torch.ones(2, 4, dtype=torch.bfloat16),
        None,
        None,
        SimpleNamespace(weight_fp32=torch.ones(4)),
        SimpleNamespace(),
        SimpleNamespace(weight_fp32=block_weight),
        _pre_mhc_module(),
        SimpleNamespace(weight_fp32=torch.ones(4)),
        is_model_tail=True,
        return_h_in_f32=True,
    )

    assert calls["kwargs"]["gamma_2"] is block_weight


def test_eager_path_returns_bf16_and_fp32_dict():
    """Non-fusion path derives the fp32 copy from the post-norm hidden states."""
    layer = _bare_layer(use_mhc=False, use_mhc_fusion_op=False)
    hidden_states = torch.ones(2, 4, dtype=torch.bfloat16)
    residual = torch.ones(2, 4, dtype=torch.bfloat16)

    out, out_residual, h_post, h_res, sk_event = layer.mhc_sandwich_norm_post_pre(
        hidden_states,
        residual,
        None,
        None,
        _identity,
        SimpleNamespace(),
        None,
        _pre_mhc_module(),
        _identity,
        is_model_tail=True,
        return_h_in_f32=True,
    )

    assert set(out) == {"hidden_states_bf16", "hidden_states_fp32"}
    # hidden + residual, then pre-norm (identity here).
    assert torch.equal(
        out["hidden_states_bf16"], torch.full((2, 4), 2.0, dtype=torch.bfloat16)
    )
    assert out["hidden_states_fp32"].dtype == torch.float32
    assert torch.equal(
        out["hidden_states_fp32"], out["hidden_states_bf16"].to(torch.float32)
    )
    # is_model_tail=True clears the residual chain.
    assert out_residual is None
    assert h_post is None and h_res is None and sk_event is None


def test_eager_path_without_flag_returns_plain_tensor():
    """Without the flag the return type stays a tensor (no dict wrapping)."""
    layer = _bare_layer(use_mhc=False, use_mhc_fusion_op=False)

    out, out_residual, _, _, _ = layer.mhc_sandwich_norm_post_pre(
        torch.ones(2, 4, dtype=torch.bfloat16),
        torch.ones(2, 4, dtype=torch.bfloat16),
        None,
        None,
        _identity,
        SimpleNamespace(),
        None,
        _pre_mhc_module(),
        _identity,
        is_model_tail=False,
    )

    assert isinstance(out, torch.Tensor)
    # Not a tail layer: the summed hidden states carry over as the residual.
    assert torch.equal(out_residual, torch.full((2, 4), 2.0, dtype=torch.bfloat16))
