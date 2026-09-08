# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for joint_grid_search_ab — the two-stage (a,b) search driver."""
import pytest
import torch

from jointfix.methods.jointfix import JointSearchConfig, joint_grid_search_ab


def _make_stats(seed=0):
    torch.manual_seed(seed)
    in_f, tok = 8, 16
    X = torch.randn(tok, in_f) * 0.1
    weights = [torch.randn(6, in_f) * 0.1, torch.randn(4, in_f) * 0.1]
    Y_refs = [X.float() @ W.float().T for W in weights]   # BF16 ground truth
    x_stat = X.abs().amax(0) + 1e-3
    w_stat = torch.maximum(weights[0].abs().amax(0), weights[1].abs().amax(0)) + 1e-3
    stats = {
        "amax": x_stat,
        "X_sample": X,
        "E_x2": (X ** 2).mean(0) + 1e-6,
        "p99_9": x_stat.clone(),
        "median": x_stat.clone() * 0.5,
        "Y_refs": Y_refs,
    }
    return weights, stats, w_stat


def test_driver_never_worse_than_baseline():
    weights, stats, w_stat = _make_stats()
    _, trace = joint_grid_search_ab(weights, stats, w_stat, JointSearchConfig())
    # safety net guarantees the search never loses to s=1
    assert trace["J_final"] <= trace["J_at_zero"] + 1e-9


def test_driver_scale_and_ab_valid():
    weights, stats, w_stat = _make_stats()
    s, trace = joint_grid_search_ab(weights, stats, w_stat, JointSearchConfig())
    assert s.shape == (8,)
    assert torch.all(s > 0)
    assert 0.0 <= trace["a"] <= 1.0
    assert 0.0 <= trace["b"] <= 1.0
    assert trace["n_stage2_candidates"] == 25   # 5x5 box (radius .2 / step .1)


def test_driver_batched_equals_unbatched():
    weights, stats, w_stat = _make_stats()
    _, t_b = joint_grid_search_ab(weights, stats, w_stat,
                                  JointSearchConfig(batched_ab_enabled=True))
    _, t_u = joint_grid_search_ab(weights, stats, w_stat,
                                  JointSearchConfig(batched_ab_enabled=False))
    assert t_b["J_final"] == pytest.approx(t_u["J_final"], rel=1e-3, abs=1e-9)


def test_driver_warm_start_skips_stage1():
    weights, stats, w_stat = _make_stats()
    _, trace = joint_grid_search_ab(weights, stats, w_stat,
                                    JointSearchConfig(), warm_start=(0.5, 0.5))
    assert trace["t_stage1_s"] == 0.0
    assert trace["J_final"] <= trace["J_at_zero"] + 1e-9


def test_driver_weight_error_not_implemented():
    weights, stats, w_stat = _make_stats()
    with pytest.raises(NotImplementedError):
        joint_grid_search_ab(weights, stats, w_stat,
                             JointSearchConfig(objective="weight-error"))
