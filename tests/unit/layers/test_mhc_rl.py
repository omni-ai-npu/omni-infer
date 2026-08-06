# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _FakeReplicatedLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=False, **_kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(output_size))
        else:
            self.bias = None

    def forward(self, hidden_states):
        return torch.nn.functional.linear(hidden_states, self.weight, self.bias), None


def _install_mhc_rl_stubs(monkeypatch, on_ascend950=False, use_batch_invariant_op=False):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo_root / "omni"))

    torch_npu_mod = types.ModuleType("torch_npu")
    torch_npu_mod.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_mod)

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.__spec__ = importlib.machinery.ModuleSpec(
        "transformers", loader=None
    )
    transformers_mod.PretrainedConfig = SimpleNamespace
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda _name=None: SimpleNamespace(warning=lambda *_args: None)
    model_executor_mod = types.ModuleType("vllm.model_executor")
    layers_mod = types.ModuleType("vllm.model_executor.layers")
    linear_mod = types.ModuleType("vllm.model_executor.layers.linear")
    linear_mod.ReplicatedLinear = _FakeReplicatedLinear
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear_mod)

    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(repo_root / "omni")]
    omni_layers_pkg = types.ModuleType("omni_npu.layers")
    omni_layers_pkg.__path__ = [str(repo_root / "omni" / "layers")]
    omni_mhc_pkg = types.ModuleType("omni_npu.layers.mhc")
    omni_mhc_pkg.__path__ = [str(repo_root / "omni" / "layers" / "mhc")]
    omni_v1_pkg = types.ModuleType("omni_npu.v1")
    omni_v1_pkg.__path__ = [str(repo_root / "omni" / "v1")]
    utils_mod = types.ModuleType("omni_npu.v1.utils")
    utils_mod.on_ascend950 = lambda: on_ascend950

    omni_model_config_pkg = types.ModuleType("omni_npu.model_config")
    omni_model_config_pkg.__path__ = [str(repo_root / "omni" / "model_config")]
    omni_model_config_loader_pkg = types.ModuleType("omni_npu.model_config.config_loader")
    omni_model_config_loader_pkg.__path__ = [
        str(repo_root / "omni" / "model_config" / "config_loader")
    ]
    loader_mod = types.ModuleType("omni_npu.model_config.config_loader.loader")
    loader_mod.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(use_batch_invariant_op=use_batch_invariant_op)
    )

    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    monkeypatch.setitem(sys.modules, "omni_npu.layers", omni_layers_pkg)
    monkeypatch.setitem(sys.modules, "omni_npu.layers.mhc", omni_mhc_pkg)
    monkeypatch.setitem(sys.modules, "omni_npu.v1", omni_v1_pkg)
    monkeypatch.setitem(sys.modules, "omni_npu.v1.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "omni_npu.model_config", omni_model_config_pkg)
    monkeypatch.setitem(
        sys.modules, "omni_npu.model_config.config_loader", omni_model_config_loader_pkg
    )
    monkeypatch.setitem(
        sys.modules, "omni_npu.model_config.config_loader.loader", loader_mod
    )


def _import_mhc_rl(monkeypatch, on_ascend950=False, use_batch_invariant_op=False):
    _install_mhc_rl_stubs(
        monkeypatch,
        on_ascend950=on_ascend950,
        use_batch_invariant_op=use_batch_invariant_op,
    )
    monkeypatch.delitem(sys.modules, "omni_npu.layers.mhc.mhc_rl", raising=False)
    module = importlib.import_module("omni_npu.layers.mhc.mhc_rl")
    return importlib.reload(module)


def _cfg():
    return SimpleNamespace(
        mhc_num_stream=2,
        hidden_size=3,
        rms_norm_eps=1e-5,
        mhc_recur_norm=3,
        mhc_use_gamma=True,
    )


def _init_mhc_params(mhc):
    with torch.no_grad():
        mhc.phi.weight.fill_(0.1)
        mhc.norm_gamma.fill_(1.0)
        if hasattr(mhc, "branch_alpha"):
            mhc.branch_alpha.fill_(0.5)
            mhc.branch_beta.fill_(0.1)
        if hasattr(mhc, "branch_alpha_pre"):
            mhc.branch_alpha_pre.fill_(0.7)
            mhc.branch_beta_pre.fill_(0.1)


@pytest.mark.unit
def test_mhc_pre_pre_only_matches_torch_reference(monkeypatch):
    module = _import_mhc_rl(monkeypatch)
    mhc = module.NPUmHCRL(_cfg(), pre_only=True, prefix="test")

    hidden_states = torch.tensor(
        [[1.0, -2.0, 3.0, 4.0, -5.0, 6.0], [-1.0, 2.0, -3.0, -4.0, 5.0, -6.0]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        mhc.phi.weight.copy_(
            torch.tensor(
                [
                    [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
                    [-0.3, 0.2, -0.1, 0.6, -0.5, 0.4],
                ],
                dtype=torch.float32,
            )
        )
        mhc.norm_gamma.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0, 1.0, 0.25]))
        mhc.branch_alpha_pre.fill_(0.7)
        mhc.branch_beta_pre.copy_(torch.tensor([0.1, -0.2]))

    output, h_post, h_res = mhc.mhc_pre(hidden_states)

    flat = hidden_states.view(-1, mhc.hidden_size * mhc.num_stream).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + mhc.hc_eps)
    mixes = torch.nn.functional.linear(
        flat * rsqrt * mhc.norm_gamma.view(1, -1),
        mhc.phi.weight,
    )
    hpre_weight = torch.sigmoid(
        mixes * mhc.branch_alpha_pre + mhc.branch_beta_pre.view(1, mhc.num_stream)
    ) + mhc.hc_eps
    expected = torch.sum(
        hpre_weight.view(-1, mhc.num_stream, 1)
        * flat.view(-1, mhc.num_stream, mhc.hidden_size),
        dim=1,
    )

    assert torch.allclose(output, expected)
    assert h_post is None
    assert h_res is None


@pytest.mark.unit
def test_pre_only_sinkhorn_and_post_short_circuit(monkeypatch):
    module = _import_mhc_rl(monkeypatch)
    mhc = module.NPUmHCRL(_cfg(), pre_only=True)

    h_res = torch.randn(2, 2, 2)
    residual = torch.randn(2, 6)

    assert mhc.mhc_sinkhorn(h_res) is h_res
    assert mhc.mhc_post(torch.randn(2, 3), None, residual, None) is residual


@pytest.mark.unit
def test_custom_op_path_forwards_expected_arguments(monkeypatch):
    module = _import_mhc_rl(monkeypatch, on_ascend950=False)
    mhc = module.NPUmHCRL(_cfg(), pre_only=False)
    _init_mhc_params(mhc)

    hidden_states = torch.randn(4, 6)
    pre_output = torch.full((4, 3), 2.0)
    h_post = torch.full((4, 2), 3.0)
    h_res = torch.full((4, 2, 2), 4.0)
    calls = {}

    def fake_pre(hidden, weight, branch_alpha, branch_beta, **kwargs):
        calls["pre"] = (hidden, weight, branch_alpha, branch_beta, kwargs)
        return pre_output, h_post, h_res, None, None, None

    def fake_sinkhorn(tensor, **kwargs):
        calls["sinkhorn"] = (tensor, kwargs)
        return tensor + 1.0, None, None

    def fake_post(residual, h_res_arg, hidden_arg, h_post_arg):
        calls["post"] = (residual, h_res_arg, hidden_arg, h_post_arg)
        return residual + 2.0

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(
            npu_manifold_constrained_hyper_connection_pre=fake_pre,
            npu_sinkhorn=fake_sinkhorn,
            npu_ai_infra_manifold_constrained_hyper_connection_post=fake_post,
        ),
        raising=False,
    )

    pre_result, pre_h_post, pre_h_res = mhc.mhc_pre(hidden_states)
    sinkhorn_result = mhc.mhc_sinkhorn(pre_h_res)
    post_result = mhc.mhc_post(pre_result, pre_h_post, hidden_states, sinkhorn_result)

    pre_hidden, weight, branch_alpha, branch_beta, pre_kwargs = calls["pre"]
    assert pre_hidden.shape == (4, 2, 3)
    assert torch.equal(weight, mhc.phi.weight * mhc.norm_gamma)
    assert branch_alpha is mhc.branch_alpha
    assert branch_beta is mhc.branch_beta
    assert pre_kwargs["gamma"] is None
    assert pre_kwargs["norm_eps"] == mhc.hc_eps
    assert pre_kwargs["hc_eps"] == mhc.hc_eps
    assert pre_kwargs["out_flag"] == 0
    assert torch.equal(pre_result, pre_output)
    assert torch.equal(pre_h_post, h_post)
    assert torch.equal(pre_h_res, h_res)

    sinkhorn_tensor, sinkhorn_kwargs = calls["sinkhorn"]
    assert torch.equal(sinkhorn_tensor, h_res)
    assert sinkhorn_kwargs == {
        "eps": mhc.hc_eps,
        "num_iters": mhc.mhc_recur_norm,
        "out_flag": 0,
    }
    assert torch.equal(sinkhorn_result, h_res + 1.0)

    post_residual, post_h_res, post_hidden, post_h_post = calls["post"]
    assert post_residual.shape == (4, 2, 3)
    assert torch.equal(post_h_res, sinkhorn_result)
    assert torch.equal(post_hidden, pre_output)
    assert torch.equal(post_h_post, h_post)
    assert post_result.shape == (4, 6)


@pytest.mark.unit
def test_ascend950_path_uses_torch_npu_ops(monkeypatch):
    module = _import_mhc_rl(monkeypatch, on_ascend950=True)
    mhc = module.NPUmHCRL(_cfg(), pre_only=False)
    _init_mhc_params(mhc)

    hidden_states = torch.randn(2, 6)
    pre_output = torch.full((2, 3), 5.0)
    h_post = torch.full((2, 2), 6.0)
    h_res = torch.full((2, 2, 2), 7.0)
    calls = {}

    def fake_mhc_pre(hidden, weight, branch_alpha, branch_beta, **kwargs):
        calls["pre"] = (hidden, weight, branch_alpha, branch_beta, kwargs)
        return pre_output, h_post, h_res, None, None, None

    def fake_mhc_sinkhorn(tensor, **kwargs):
        calls["sinkhorn"] = (tensor, kwargs)
        return tensor + 2.0, None, None

    def fake_mhc_post(residual, h_res_arg, hidden_arg, h_post_arg):
        calls["post"] = (residual, h_res_arg, hidden_arg, h_post_arg)
        return residual + 3.0

    monkeypatch.setattr(module.torch_npu, "npu_mhc_pre", fake_mhc_pre, raising=False)
    monkeypatch.setattr(
        module.torch_npu, "npu_mhc_sinkhorn", fake_mhc_sinkhorn, raising=False
    )
    monkeypatch.setattr(module.torch_npu, "npu_mhc_post", fake_mhc_post, raising=False)

    pre_result, pre_h_post, pre_h_res = mhc.mhc_pre(hidden_states)
    sinkhorn_result = mhc.mhc_sinkhorn(pre_h_res)
    post_result = mhc.mhc_post(pre_result, pre_h_post, hidden_states, sinkhorn_result)

    assert calls["pre"][0].shape == (2, 2, 3)
    assert calls["pre"][4]["out_flag"] == 0
    assert torch.equal(calls["pre"][4]["gamma"], mhc.norm_gamma.view(mhc.num_stream, -1))
    assert calls["sinkhorn"][1] == {
        "eps": mhc.hc_eps,
        "num_iters": mhc.mhc_recur_norm,
        "out_flag": 0,
    }
    assert calls["post"][0].shape == (2, 2, 3)
    assert torch.equal(calls["post"][1], sinkhorn_result)
    assert torch.equal(calls["post"][2], pre_output)
    assert torch.equal(calls["post"][3], h_post)
    assert post_result.shape == (2, 6)


@pytest.mark.unit
def test_batch_invariant_pre_path_forwards_expected_arguments(monkeypatch):
    module = _import_mhc_rl(monkeypatch, on_ascend950=False, use_batch_invariant_op=True)
    mhc = module.NPUmHCRL(_cfg(), pre_only=False)
    _init_mhc_params(mhc)

    hidden_states = torch.randn(4, 6)
    pre_output = torch.full((4, 3), 2.0)
    h_post = torch.full((4, 2), 3.0)
    h_res = torch.full((4, 2, 2), 4.0)
    calls = {}

    def fake_pre(hidden, weight, branch_alpha, branch_beta, **kwargs):
        calls["pre"] = (hidden, weight, branch_alpha, branch_beta, kwargs)
        return pre_output, h_post, h_res, None, None, None

    def fake_sinkhorn(tensor, **kwargs):
        calls["sinkhorn"] = (tensor, kwargs)
        return tensor + 1.0, None, None

    def fake_post(residual, h_res_arg, hidden_arg, h_post_arg):
        calls["post"] = (residual, h_res_arg, hidden_arg, h_post_arg)
        return residual + 2.0

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(
            npu_manifold_constrained_hyper_connection_pre=fake_pre,
            npu_sinkhorn=fake_sinkhorn,
            npu_ai_infra_manifold_constrained_hyper_connection_post=fake_post,
        ),
        raising=False,
    )

    pre_result, pre_h_post, pre_h_res = mhc.mhc_pre(hidden_states)
    sinkhorn_result = mhc.mhc_sinkhorn(pre_h_res)
    post_result = mhc.mhc_post(pre_result, pre_h_post, hidden_states, sinkhorn_result)

    pre_hidden, weight, branch_alpha, branch_beta, pre_kwargs = calls["pre"]
    assert pre_hidden.shape == (4, 2, 3)
    assert torch.equal(weight, mhc.phi.weight)
    assert branch_alpha is mhc.branch_alpha
    assert branch_beta is mhc.branch_beta
    assert torch.equal(
        pre_kwargs["gamma"], mhc.norm_gamma.view(mhc.num_stream, mhc.hidden_size)
    )
    assert pre_kwargs["norm_eps"] == mhc.hc_eps
    assert pre_kwargs["hc_eps"] == mhc.hc_eps
    assert pre_kwargs["out_flag"] == 1
    assert torch.equal(pre_result, pre_output)
    assert torch.equal(pre_h_post, h_post)
    assert torch.equal(pre_h_res, h_res)

    sinkhorn_tensor, sinkhorn_kwargs = calls["sinkhorn"]
    assert torch.equal(sinkhorn_tensor, h_res)
    assert sinkhorn_kwargs == {
        "eps": mhc.hc_eps,
        "num_iters": mhc.mhc_recur_norm,
        "out_flag": 1,
    }
    assert torch.equal(sinkhorn_result, h_res + 1.0)

    post_residual, post_h_res, post_hidden, post_h_post = calls["post"]
    assert post_residual.shape == (4, 2, 3)
    assert torch.equal(post_h_res, sinkhorn_result)
    assert torch.equal(post_hidden, pre_output)
    assert torch.equal(post_h_post, h_post)
    assert post_result.shape == (4, 6)
