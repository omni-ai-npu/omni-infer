# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Tests for methods.smooth_search — pure joint (a,b) math.

Highlight: test_output_recon_batched_matches_single cross-checks the batched
objective against the single one (they must be equivalent).
"""
import pytest
import torch

from jointfix.methods.smooth_search import (
    batched_weight_dW_term,
    batched_weight_dW_term_multi_s,
    compute_output_recon_objective,
    compute_output_recon_objective_batched_ab,
    compute_output_recon_objective_batched_ab_distributed,
    hessian_channel_weight,
    make_smooth_scale,
    outlier_channel_weight,
    select_channel_weight,
)


# ── smooth scale ──────────────────────────────────────────────────────────────
def test_make_smooth_scale_identity_at_zero():
    x = torch.rand(8) + 0.5
    w = torch.rand(8) + 0.5
    s = make_smooth_scale(x, w, a=0.0, b=0.0)
    assert torch.allclose(s, torch.ones(8), atol=1e-6)


def test_make_smooth_scale_formula():
    x = torch.tensor([4.0, 9.0])
    w = torch.tensor([2.0, 3.0])
    # s = x^0.5 / w^1 = [2/2, 3/3] = [1, 1]
    s = make_smooth_scale(x, w, a=0.5, b=1.0)
    assert torch.allclose(s, torch.ones(2), atol=1e-6)


# ── channel weighting ─────────────────────────────────────────────────────────
def test_outlier_weight_unit_when_no_outlier():
    v = torch.tensor([2.0, 3.0, 5.0])
    w_c = outlier_channel_weight(v, v, gamma=1.0)  # p99.9 == median
    assert torch.allclose(w_c, torch.ones(3), atol=1e-3)


def test_hessian_weight_alpha0_is_uniform():
    E_x2 = torch.rand(5) + 0.1
    E_w2 = torch.rand(5) + 0.1
    w_c = hessian_channel_weight(E_x2, E_w2, alpha=0.0)
    assert torch.allclose(w_c, torch.ones(5), atol=1e-6)


def test_hessian_weight_alpha1_mean_one():
    E_x2 = torch.rand(16) + 0.1
    E_w2 = torch.rand(16) + 0.1
    w_c = hessian_channel_weight(E_x2, E_w2, alpha=1.0)
    assert abs(w_c.mean().item() - 1.0) < 1e-5


def test_select_channel_weight_dispatch():
    p, m = torch.rand(4) + 1, torch.rand(4) + 1
    ex2, ew2 = torch.rand(4) + 1, torch.rand(4) + 1
    out = select_channel_weight(p, m, ex2, ew2, channel_weight="outlier", gamma=1.0)
    hes = select_channel_weight(p, m, ex2, ew2, channel_weight="hessian", hessian_alpha=1.0)
    assert out.shape == (4,) and hes.shape == (4,)
    with pytest.raises(ValueError):
        select_channel_weight(p, m, ex2, ew2, channel_weight="bogus")


# ── weight-error term ─────────────────────────────────────────────────────────
def test_dW_term_multi_s_matches_single():
    torch.manual_seed(0)
    W_stacked = torch.randn(3, 6, 8)               # [N, out, in]
    s_batch = torch.rand(4, 8) + 0.5               # [B, in]
    multi = batched_weight_dW_term_multi_s(W_stacked, s_batch, ab_chunk=2)
    for i in range(s_batch.shape[0]):
        single = batched_weight_dW_term(W_stacked, s_batch[i])
        assert torch.allclose(multi[i], single, atol=1e-5)


def test_dW_term_chunking_invariant():
    torch.manual_seed(1)
    W_stacked = torch.randn(5, 4, 8)
    s = torch.rand(8) + 0.5
    full = batched_weight_dW_term(W_stacked, s, batch_size=0)
    chunked = batched_weight_dW_term(W_stacked, s, batch_size=2)
    assert torch.allclose(full, chunked, atol=1e-5)


# ── output-reconstruction objective ───────────────────────────────────────────
def _setup_recon():
    torch.manual_seed(7)
    in_f, tok = 8, 12
    X = torch.randn(tok, in_f) * 0.1
    weights = [torch.randn(6, in_f) * 0.1, torch.randn(4, in_f) * 0.1]
    Y_refs = [X.float() @ W.float().T for W in weights]   # BF16-ref ground truth
    x_stat = X.abs().amax(0) + 1e-3
    w_stat = torch.rand(in_f) + 0.5
    return weights, Y_refs, X, x_stat, w_stat


def test_output_recon_batched_matches_single():
    weights, Y_refs, X, x_stat, w_stat = _setup_recon()
    a_batch = torch.tensor([0.0, 0.3, 0.5, 0.7])
    b_batch = torch.tensor([0.0, 0.7, 0.5, 0.3])
    batched = compute_output_recon_objective_batched_ab(
        weights, Y_refs, X, x_stat, w_stat, a_batch, b_batch, ab_chunk=2)
    for i in range(a_batch.shape[0]):
        single = compute_output_recon_objective(
            weights, Y_refs, X, x_stat, w_stat,
            a=a_batch[i].item(), b=b_batch[i].item())
        # pow (single) vs exp/log (batched) scale construction -> tiny drift
        assert batched[i].item() == pytest.approx(single, rel=1e-3, abs=1e-6)


def test_output_recon_nonnegative():
    weights, Y_refs, X, x_stat, w_stat = _setup_recon()
    j = compute_output_recon_objective(weights, Y_refs, X, x_stat, w_stat, a=0.5, b=0.5)
    assert j >= 0.0


def test_distributed_objective_matches_single():
    """
    The distributed objective (weights split across 2 mock devices, partials
    summed) equals the single-device batched objective.
    """
    weights, Y_refs, X, x_stat, w_stat = _setup_recon()
    cpu = torch.device("cpu")
    groups, yref_groups = [[], []], [[], []]
    for i, (W, Y) in enumerate(zip(weights, Y_refs)):
        groups[i % 2].append(W)
        yref_groups[i % 2].append(Y)
    a = torch.tensor([0.0, 0.3, 0.5, 0.7])
    b = torch.tensor([0.0, 0.7, 0.5, 0.3])

    dist = compute_output_recon_objective_batched_ab_distributed(
        groups, yref_groups, [cpu, cpu], X, x_stat, w_stat, a, b, ab_chunk=2)
    single = compute_output_recon_objective_batched_ab(
        weights, Y_refs, X, x_stat, w_stat, a, b, ab_chunk=2)

    assert torch.allclose(dist, single, rtol=1e-4, atol=1e-5)
