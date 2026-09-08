# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Tests for the core primitives.

These guard the low-level quantizers: if behaviour drifts unexpectedly, catch
it here first.
"""
import warnings

import pytest
import torch

from jointfix.core.primitives import (
    UNIVERSAL_SKIP_PATTERNS,
    gptq_quantize,
    int8_fake_quantize,
    rtn_quantize,
    select_write_quantize,
    should_quantize,
)


def test_rtn_quantize_range_and_shape():
    torch.manual_seed(0)
    W = torch.randn(4, 8)
    q, scale = rtn_quantize(W)
    assert q.dtype == torch.int8
    assert q.min() >= -128 and q.max() <= 127
    assert scale.shape == (4, 1)
    assert scale.dtype == torch.bfloat16
    # dequant within one quant step of the original
    deq = q.float() * scale.float()
    step = scale.float()  # per-row LSB
    assert torch.all((W - deq).abs() <= step + 1e-5)


def test_int8_fake_quantize_idempotent():
    torch.manual_seed(1)
    W = torch.randn(6, 16)
    once = int8_fake_quantize(W)
    twice = int8_fake_quantize(once)
    # quantizing an already-on-grid tensor is a fixed point
    assert torch.allclose(once, twice, atol=1e-5)


def test_int8_fake_quantize_matches_rtn_dequant():
    torch.manual_seed(2)
    W = torch.randn(5, 10)
    fake = int8_fake_quantize(W)
    q, scale = rtn_quantize(W)
    deq = q.float() * scale.float()
    # same quantizer; only difference is rtn's bf16 scale rounding
    assert torch.allclose(fake, deq, atol=2e-2)


def test_gptq_falls_back_to_rtn_when_rank_deficient():
    torch.manual_seed(3)
    W = torch.randn(4, 8)
    X = torch.randn(3, 8)  # N=3 < h_in=8 -> rank-deficient -> RTN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gq, gs = gptq_quantize(W, X, tag="test")
    rq, rs = rtn_quantize(W)
    assert torch.equal(gq, rq)
    assert torch.equal(gs, rs)


def test_gptq_runs_when_full_rank():
    torch.manual_seed(4)
    W = torch.randn(4, 8)
    X = torch.randn(32, 8)  # N=32 >= h_in=8 -> real GPTQ
    gq, gs = gptq_quantize(W, X, tag="test")
    assert gq.dtype == torch.int8
    assert gq.min() >= -128 and gq.max() <= 127
    assert gs.shape == (4, 1)
    deq = gq.float() * gs.float()
    assert torch.isfinite(deq).all()


def test_select_write_quantize_dispatch():
    torch.manual_seed(5)
    W = torch.randn(4, 8)
    X = torch.randn(32, 8)
    # rtn path
    a_q, a_s = select_write_quantize(W, X, write_quant="rtn")
    r_q, r_s = rtn_quantize(W)
    assert torch.equal(a_q, r_q)
    # None forces rtn
    n_q, _ = select_write_quantize(W, None, write_quant="gptq")
    assert torch.equal(n_q, r_q)
    # unknown -> error
    with pytest.raises(ValueError):
        select_write_quantize(W, X, write_quant="bogus")


def test_should_quantize_skip_patterns():
    W2 = torch.randn(4, 8)
    assert should_quantize("model.layers.0.self_attn.o_proj.weight", W2,
                           UNIVERSAL_SKIP_PATTERNS) is True
    assert should_quantize("lm_head.weight", W2, UNIVERSAL_SKIP_PATTERNS) is False
    assert should_quantize("model.layers.0.mlp.gate.weight", W2,
                           UNIVERSAL_SKIP_PATTERNS) is False  # router gate
    assert should_quantize("model.layers.0.mlp.gate_proj.weight", W2,
                           UNIVERSAL_SKIP_PATTERNS) is True   # NOT the router
    assert should_quantize("o_proj.bias", W2, UNIVERSAL_SKIP_PATTERNS) is False
    assert should_quantize("o_proj.weight", torch.randn(8),
                           UNIVERSAL_SKIP_PATTERNS) is False  # 1D
