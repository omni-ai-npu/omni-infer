# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def _stub_mhc_side_stream(
    module, monkeypatch, main_stream, side_stream, ready_event, done_event
):
    stream_names = []

    def fake_named_stream(name):
        stream_names.append(name)
        return side_stream

    monkeypatch.setattr(module, "named_stream", fake_named_stream)
    monkeypatch.setattr(
        module.torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: main_stream,
            Event=MagicMock(side_effect=[ready_event, done_event]),
            stream=lambda _stream: nullcontext(),
        ),
        raising=False,
    )
    return stream_names


def _install_mhc_rl_stubs(
    monkeypatch,
    on_ascend950=False,
    use_batch_invariant_op=False,
    enable_mhc_multistream=False,
    quant_config=None,
    enable_precision_strong_consistency=False,
):
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
    static_forward_context = {}
    config_mod = types.ModuleType("vllm.config")

    def _get_current_vllm_config():
        return SimpleNamespace(
            quant_config=quant_config,
            compilation_config=SimpleNamespace(
                static_forward_context=static_forward_context
            ),
        )

    config_mod.get_current_vllm_config = _get_current_vllm_config
    forward_context_mod = types.ModuleType("vllm.forward_context")

    def _get_forward_context():
        return SimpleNamespace(
            no_compile_layers=static_forward_context,
            additional_kwargs={},
        )

    def _is_forward_context_available():
        return True

    forward_context_mod.get_forward_context = _get_forward_context
    forward_context_mod.is_forward_context_available = (
        _is_forward_context_available
    )
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda _name=None: SimpleNamespace(warning=lambda *_args: None)
    model_executor_mod = types.ModuleType("vllm.model_executor")
    layers_mod = types.ModuleType("vllm.model_executor.layers")
    linear_mod = types.ModuleType("vllm.model_executor.layers.linear")
    linear_mod.ReplicatedLinear = _FakeReplicatedLinear
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.config", config_mod)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context_mod)
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
        dtype=torch.bfloat16,
        operator_opt_config=SimpleNamespace(
            use_batch_invariant_op=use_batch_invariant_op,
            enable_mhc_multistream=enable_mhc_multistream,
            use_mhc_fusion_op=False,
            enable_precision_strong_consistency=enable_precision_strong_consistency,
        ),
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


def _import_mhc_rl(
    monkeypatch,
    on_ascend950=False,
    use_batch_invariant_op=False,
    enable_mhc_multistream=False,
    quant_config=None,
    enable_precision_strong_consistency=False,
):
    _install_mhc_rl_stubs(
        monkeypatch,
        on_ascend950=on_ascend950,
        use_batch_invariant_op=use_batch_invariant_op,
        enable_mhc_multistream=enable_mhc_multistream,
        quant_config=quant_config,
        enable_precision_strong_consistency=enable_precision_strong_consistency,
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
def test_named_side_stream_aliases_share_one_stream(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "layer_utils_side_stream_test",
        repo_root / "omni" / "layers" / "utils.py",
    )
    layer_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layer_utils)

    streams = [object()]
    created = []

    def create_stream():
        stream = streams[len(created)]
        created.append(stream)
        return stream

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(Stream=create_stream), raising=False
    )

    canonical = layer_utils.named_stream(layer_utils.SIDE_STREAM_NAME)

    assert layer_utils.SIDE_STREAM_NAME == "side_stream"
    assert layer_utils.CUBE_SIDE_STREAM_NAME == layer_utils.SIDE_STREAM_NAME
    assert layer_utils._SIDE_STREAM_ALIASES == {
        "sub_stream",
        "npu_attention_decode_sub_stream",
    }
    assert layer_utils.named_stream("sub_stream") is canonical
    assert (
        layer_utils.named_stream("npu_attention_decode_sub_stream")
        is canonical
    )
    assert layer_utils.named_stream("side_stream") is canonical
    assert canonical is streams[0]
    assert created == streams


@pytest.mark.unit
def test_bf16_multistream_uses_direct_launch_and_fetch(monkeypatch):
    module = _import_mhc_rl(
        monkeypatch,
        enable_mhc_multistream=True,
        quant_config=None,
    )
    mhc = module.NPUmHCRL(_cfg(), pre_only=False, prefix="layers.0.attn_mhc")

    h_res = torch.randn(2, 2, 2)
    launched = torch.randn(2, 2, 2)
    fetched = torch.randn(2, 2, 2)
    calls = {}

    def fake_launch(prefix, task_key, value):
        calls["launch"] = (prefix, task_key, value)
        return launched

    def fake_fetch(prefix, task_key, value):
        calls["fetch"] = (prefix, task_key, value)
        return fetched

    monkeypatch.setattr(
        torch.ops.vllm, "mhc_direct_launch", fake_launch, raising=False
    )
    monkeypatch.setattr(
        torch.ops.vllm, "mhc_direct_fetch", fake_fetch, raising=False
    )

    launch_result = mhc.maybe_register_sinkhorn(h_res, "layers.0.self_attn.o_proj")
    fetch_result = mhc.resolve_sinkhorn(launch_result, "layers.0.self_attn.o_proj")

    assert mhc.enable_mhc_multistream
    assert mhc.use_direct_mhc_multistream
    assert calls["launch"] == (
        "layers.0.attn_mhc",
        "layers.0.self_attn.o_proj",
        h_res,
    )
    assert calls["fetch"] == (
        "layers.0.attn_mhc",
        "layers.0.self_attn.o_proj",
        launched,
    )
    assert launch_result is launched
    assert fetch_result is fetched


@pytest.mark.unit
def test_fused_multistream_uses_split_launch_and_fetch_custom_ops(monkeypatch):
    module = _import_mhc_rl(
        monkeypatch,
        enable_mhc_multistream=True,
        quant_config=None,
    )
    mhc = module.NPUmHCRL(_cfg(), pre_only=False, prefix="layers.0.mlp_mhc")

    residual = torch.randn(4, 6)
    launched_h_post = torch.randn(4, 2)
    launched_h_res = torch.randn(4, 2, 2)
    fetched_h_post = torch.randn(4, 2)
    fetched_h_res = torch.randn(4, 2, 2)
    calls = {}

    def fake_launch(prefix, task_key, value):
        calls["launch"] = (prefix, task_key, value)
        return launched_h_post, launched_h_res

    def fake_fetch(prefix, task_key, h_post, h_res):
        calls["fetch"] = (prefix, task_key, h_post, h_res)
        return fetched_h_post, fetched_h_res

    monkeypatch.setattr(
        torch.ops.vllm, "mhc_fused_split_launch", fake_launch, raising=False
    )
    monkeypatch.setattr(
        torch.ops.vllm, "mhc_fused_split_fetch", fake_fetch, raising=False
    )

    launch_result = mhc.launch_fused_split_sinkhorn(
        residual, "layers.0.mlp.experts"
    )
    fetch_result = mhc.resolve_fused_split_sinkhorn(
        *launch_result, "layers.0.mlp.experts"
    )

    assert calls["launch"] == (
        "layers.0.mlp_mhc",
        "layers.0.mlp.experts",
        residual,
    )
    assert calls["fetch"] == (
        "layers.0.mlp_mhc",
        "layers.0.mlp.experts",
        launched_h_post,
        launched_h_res,
    )
    assert launch_result == (launched_h_post, launched_h_res)
    assert fetch_result == (fetched_h_post, fetched_h_res)


@pytest.mark.unit
def test_fused_multistream_disabled_stays_synchronous(monkeypatch):
    module = _import_mhc_rl(
        monkeypatch,
        enable_mhc_multistream=False,
        quant_config=None,
    )
    mhc = module.NPUmHCRL(_cfg(), pre_only=False, prefix="layers.0.mlp_mhc")

    residual = torch.randn(4, 6)
    split_h_post = torch.randn(4, 2)
    split_h_res = torch.randn(4, 2, 2)
    sinkhorn_h_res = torch.randn(4, 2, 2)
    calls = []

    def fake_split(value):
        calls.append(("split", value))
        return split_h_post, split_h_res

    def fake_sinkhorn(value):
        calls.append(("sinkhorn", value))
        return sinkhorn_h_res

    monkeypatch.setattr(mhc, "mhc_pre_split_post_res", fake_split)
    monkeypatch.setattr(mhc, "mhc_sinkhorn", fake_sinkhorn)

    launched = mhc.launch_fused_split_sinkhorn(residual, "mlp_key")
    resolved = mhc.resolve_fused_split_sinkhorn(*launched, "mlp_key")

    assert [call[0] for call in calls] == ["split", "sinkhorn"]
    assert calls[0][1] is residual
    assert calls[1][1] is split_h_res
    assert launched[0] is split_h_post
    assert launched[1] is sinkhorn_h_res
    assert resolved[0] is split_h_post
    assert resolved[1] is sinkhorn_h_res


@pytest.mark.unit
def test_fused_split_fake_outputs_match_kernel_metadata(monkeypatch):
    module = _import_mhc_rl(
        monkeypatch,
        enable_mhc_multistream=True,
        quant_config=None,
    )
    mhc = module.NPUmHCRL(_cfg(), pre_only=False, prefix="layers.0.mlp_mhc")
    residual = torch.randn(4, 6, dtype=torch.bfloat16)

    h_post, h_res = module.mhc_fused_split_launch_fake(
        mhc.prefix, "layers.0.mlp.experts", residual
    )

    assert h_post.shape == (4, mhc.num_stream)
    assert h_res.shape == (4, mhc.num_stream, mhc.num_stream)
    assert h_post.dtype == torch.float32
    assert h_res.dtype == torch.float32


@pytest.mark.unit
def test_direct_custom_ops_launch_fetch_and_missing_launch_fallback(monkeypatch):
    module = _import_mhc_rl(monkeypatch)
    main_stream = MagicMock()
    side_stream = MagicMock()
    ready_event = MagicMock()
    done_event = MagicMock()
    h_res = MagicMock()
    sinkhorn_result = MagicMock()
    fallback = MagicMock()
    fallback_result = MagicMock()
    mhc_module = MagicMock()
    mhc_module.mhc_sinkhorn.side_effect = [
        sinkhorn_result,
        fallback_result,
    ]
    forward_context = SimpleNamespace(
        no_compile_layers={"layers.0.attn_mhc": mhc_module},
        additional_kwargs={},
    )

    monkeypatch.setattr(module, "get_forward_context", lambda: forward_context)
    stream_names = _stub_mhc_side_stream(
        module, monkeypatch, main_stream, side_stream, ready_event, done_event
    )

    launched = module.mhc_direct_launch(
        "layers.0.attn_mhc", "attn_key", h_res
    )
    fetched = module.mhc_direct_fetch(
        "layers.0.attn_mhc", "attn_key", h_res
    )
    recovered = module.mhc_direct_fetch(
        "layers.0.attn_mhc", "missing_key", fallback
    )

    assert launched is h_res
    assert fetched is sinkhorn_result
    assert recovered is fallback_result
    assert stream_names == [module.SIDE_STREAM_NAME]
    ready_event.record.assert_called_once_with(main_stream)
    ready_event.wait.assert_called_once_with(side_stream)
    h_res.record_stream.assert_called_once_with(side_stream)
    done_event.record.assert_called_once_with()
    main_stream.wait_event.assert_called_once_with(done_event)
    sinkhorn_result.record_stream.assert_called_once_with(main_stream)
    assert forward_context.additional_kwargs[module.MHC_DIRECT_PENDING_KEY] == {}
    sinkhorn_calls = mhc_module.mhc_sinkhorn.call_args_list
    assert len(sinkhorn_calls) == 2
    assert sinkhorn_calls[0].args[0] is h_res
    assert sinkhorn_calls[1].args[0] is fallback


@pytest.mark.unit
def test_fused_split_custom_ops_launch_fetch_and_missing_event(monkeypatch):
    module = _import_mhc_rl(monkeypatch)
    main_stream = MagicMock()
    side_stream = MagicMock()
    ready_event = MagicMock()
    done_event = MagicMock()
    residual = MagicMock()
    h_post = MagicMock()
    raw_h_res = MagicMock()
    sinkhorn_h_res = MagicMock()
    mhc_module = MagicMock()
    mhc_module.mhc_pre_split_post_res.return_value = (h_post, raw_h_res)
    mhc_module.mhc_sinkhorn.return_value = sinkhorn_h_res
    forward_context = SimpleNamespace(
        no_compile_layers={"layers.0.mlp_mhc": mhc_module},
        additional_kwargs={},
    )

    monkeypatch.setattr(module, "get_forward_context", lambda: forward_context)
    stream_names = _stub_mhc_side_stream(
        module, monkeypatch, main_stream, side_stream, ready_event, done_event
    )

    launched = module.mhc_fused_split_launch(
        "layers.0.mlp_mhc", "mlp_key", residual
    )
    fetched = module.mhc_fused_split_fetch(
        "layers.0.mlp_mhc", "mlp_key", *launched
    )
    missing_h_post = MagicMock()
    missing_h_res = MagicMock()
    missing = module.mhc_fused_split_fetch(
        "layers.0.mlp_mhc",
        "missing_key",
        missing_h_post,
        missing_h_res,
    )

    assert launched[0] is h_post
    assert launched[1] is sinkhorn_h_res
    assert fetched[0] is h_post
    assert fetched[1] is sinkhorn_h_res
    assert missing[0] is missing_h_post
    assert missing[1] is missing_h_res
    assert stream_names == [module.SIDE_STREAM_NAME]
    ready_event.record.assert_called_once_with(main_stream)
    ready_event.wait.assert_called_once_with(side_stream)
    residual.record_stream.assert_called_once_with(side_stream)
    mhc_module.mhc_pre_split_post_res.assert_called_once_with(residual)
    mhc_module.mhc_sinkhorn.assert_called_once_with(raw_h_res)
    done_event.record.assert_called_once_with()
    main_stream.wait_event.assert_called_once_with(done_event)
    h_post.record_stream.assert_called_once_with(main_stream)
    sinkhorn_h_res.record_stream.assert_called_once_with(main_stream)
    missing_h_post.record_stream.assert_not_called()
    missing_h_res.record_stream.assert_not_called()
    assert (
        forward_context.additional_kwargs[module.MHC_FUSED_SPLIT_PENDING_KEY]
        == {}
    )


@pytest.mark.unit
def test_custom_op_fake_helpers_preserve_metadata_contract(monkeypatch):
    module = _import_mhc_rl(monkeypatch)
    value = torch.randn(2, 2)
    h_post = torch.randn(2, 2)
    h_res = torch.randn(2, 2, 2)

    assert module.mhc_direct_launch_fake("holder", "key", value) is value
    assert module.mhc_direct_fetch_fake("holder", "key", value) is value
    fetched_h_post, fetched_h_res = module.mhc_fused_split_fetch_fake(
        "holder", "key", h_post, h_res
    )
    assert fetched_h_post.shape == h_post.shape
    assert fetched_h_res.shape == h_res.shape


@pytest.mark.unit
def test_quantized_multistream_keeps_cube_side_task_path(monkeypatch):
    module = _import_mhc_rl(
        monkeypatch,
        enable_mhc_multistream=True,
        quant_config=object(),
    )
    mhc = module.NPUmHCRL(_cfg(), pre_only=False, prefix="layers.0.mlp_mhc")

    h_res = torch.randn(2, 2, 2)
    registered = torch.randn(2, 2, 2)
    resolved = torch.randn(2, 2, 2)
    calls = {}

    def fake_register(prefix, task_key, value):
        calls["register"] = (prefix, task_key, value)
        return registered

    def fake_resolve(mhc_module, task_key, value):
        calls["resolve"] = (mhc_module, task_key, value)
        return resolved

    monkeypatch.setattr(module, "maybe_register_mhc_task", fake_register)
    monkeypatch.setattr(module, "resolve_mhc_h_res", fake_resolve)

    register_result = mhc.maybe_register_sinkhorn(h_res, "layers.0.mlp.experts")
    resolve_result = mhc.resolve_sinkhorn(register_result, "layers.0.mlp.experts")

    assert not mhc.use_direct_mhc_multistream
    assert calls["register"] == (
        "layers.0.mlp_mhc",
        "layers.0.mlp.experts",
        h_res,
    )
    assert calls["resolve"] == (
        mhc,
        "layers.0.mlp.experts",
        registered,
    )
    assert register_result is registered
    assert resolve_result is resolved



@pytest.mark.unit
def test_process_weights_after_loading_prepares_fusion_slices(monkeypatch):
    module = _import_mhc_rl(monkeypatch)
    mhc = module.NPUmHCRL(_cfg(), pre_only=False, prefix="fusion")

    with torch.no_grad():
        mhc.phi.weight.copy_(
            torch.arange(mhc.phi.weight.numel(), dtype=torch.float32).view_as(
                mhc.phi.weight
            )
        )
        mhc.norm_gamma.copy_(
            torch.arange(1, mhc.norm_gamma.numel() + 1, dtype=torch.float32)
        )
        mhc.branch_alpha.copy_(torch.tensor([0.1, 0.2, 0.3]))
        mhc.branch_beta.copy_(
            torch.arange(mhc.branch_beta.numel(), dtype=torch.float32)
        )

    mhc.process_weights_after_loading()

    expected = mhc.phi.weight * mhc.norm_gamma
    assert torch.equal(mhc.phi_weight, expected)
    assert torch.equal(mhc.phi_weight_pre, expected[:mhc.num_stream])
    assert torch.equal(mhc.phi_weight_post_res, expected[mhc.num_stream:])
    assert torch.equal(mhc.branch_alpha_pre, mhc.branch_alpha[0:1])
    assert torch.equal(mhc.branch_alpha_post_res, mhc.branch_alpha[1:])
    assert torch.equal(
        mhc.branch_beta_pre, mhc.branch_beta[:mhc.num_stream]
    )
    assert torch.equal(
        mhc.branch_beta_post_res, mhc.branch_beta[mhc.num_stream:]
    )

    # Derived weights are non-persistent buffers: they move with Module.to()
    # but stay out of the state_dict / checkpoint loading path.
    assert mhc._buffers["phi_weight"] is mhc.phi_weight
    assert "phi_weight" not in mhc.state_dict()
    assert "phi_weight_pre" not in mhc.state_dict()

    first_phi_weight = mhc.phi_weight
    mhc.process_weights_after_loading()
    assert mhc.phi_weight is first_phi_weight

    # In-place weight update bumps _version: refresh must reuse the same
    # storage so captured graphs keep a valid pointer.
    with torch.no_grad():
        mhc.norm_gamma.add_(1.0)
    mhc.process_weights_after_loading()
    assert mhc.phi_weight is first_phi_weight
    torch.testing.assert_close(
        mhc.phi_weight, mhc.phi.weight * mhc.norm_gamma
    )


@pytest.mark.unit
def test_mhc_fusion_ops_forward_expected_arguments(monkeypatch):
    module = _import_mhc_rl(monkeypatch)
    post_mhc = module.NPUmHCRL(_cfg(), pre_only=False, prefix="post")
    pre_mhc = module.NPUmHCRL(_cfg(), pre_only=False, prefix="pre")
    pre_mhc.process_weights_after_loading()

    residual = torch.randn(4, 6)
    h_post = torch.randn(4, 2)
    h_res = torch.randn(4, 2, 2)
    hidden_states = torch.randn(4, 3)
    split_h_post = torch.randn(4, 2)
    split_h_res = torch.randn(4, 2, 2)
    fused_hidden = torch.randn(4, 3)
    fused_residual = torch.randn(4, 6)
    fused_hidden_fp32 = torch.randn(4, 3, dtype=torch.float32)
    post_weight = torch.randn(3, dtype=torch.float32)
    pre_weight = torch.randn(3, dtype=torch.float32)
    block_weight = torch.randn(6, dtype=torch.float32)
    calls = {}

    def fake_split(residual_arg, phi, alpha, beta, **kwargs):
        calls["split"] = (residual_arg, phi, alpha, beta, kwargs)
        return split_h_post, split_h_res

    def fake_fusion(*args, **kwargs):
        calls["fusion"] = (args, kwargs)
        return fused_hidden, fused_residual, fused_hidden_fp32

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(
            npu_ai_infra_mhc_pre_split_post_res=fake_split,
            npu_ai_infra_mhc_sandwich_norm_post_preonly_v2=fake_fusion,
        ),
        raising=False,
    )

    split_result = pre_mhc.mhc_pre_split_post_res(residual)
    assert split_result[0] is split_h_post
    assert split_result[1] is split_h_res
    split_args = calls["split"]
    assert split_args[0].shape == (4, pre_mhc.num_stream, pre_mhc.hidden_size)
    torch.testing.assert_close(
        split_args[0], residual.reshape(4, pre_mhc.num_stream, pre_mhc.hidden_size)
    )
    assert split_args[1] is pre_mhc.phi_weight_post_res
    assert split_args[2] is pre_mhc.branch_alpha_post_res
    assert split_args[3] is pre_mhc.branch_beta_post_res
    assert split_args[4] == {"norm_eps": pre_mhc.hc_eps}

    hidden_result, residual_result = (
        post_mhc.mhc_sandwich_norm_post_preonly(
            hidden_states,
            residual,
            h_post,
            h_res,
            SimpleNamespace(weight_fp32=post_weight),
            pre_mhc,
            SimpleNamespace(weight_fp32=pre_weight),
            SimpleNamespace(weight_fp32=block_weight),
            return_h_in_f32=True,
        )
    )
    fusion_args, fusion_kwargs = calls["fusion"]
    assert fusion_args[0] is hidden_states
    assert fusion_args[1].shape == (
        4, post_mhc.num_stream, post_mhc.hidden_size
    )
    torch.testing.assert_close(
        fusion_args[1],
        residual.reshape(4, post_mhc.num_stream, post_mhc.hidden_size),
    )
    assert fusion_args[2] is h_post
    assert fusion_args[3] is h_res
    assert fusion_args[4] is pre_mhc.phi_weight_pre
    assert fusion_args[5] is pre_mhc.branch_alpha_pre
    assert fusion_args[6] is pre_mhc.branch_beta_pre
    assert fusion_args[7] is post_weight
    assert fusion_args[8] is pre_weight
    assert fusion_kwargs == {
        "gamma_2": block_weight,
        "norm_eps": pre_mhc.norm_eps,
        "hc_eps": pre_mhc.hc_eps,
        "return_h_in_f32": True,
    }
    assert hidden_result["hidden_states_bf16"] is fused_hidden
    assert hidden_result["hidden_states_fp32"] is fused_hidden_fp32
    assert residual_result is fused_residual

    post_mhc.use_mhc_fusion_op = True
    assert not post_mhc.can_use_fusion(torch.empty(256, 1))
    post_mhc.process_weights_after_loading()
    assert post_mhc.can_use_fusion(torch.empty(256, 1))
    assert not post_mhc.can_use_fusion(torch.empty(257, 1))



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
def test_mhc_pre_pre_only_uses_rms_norm_for_strong_consistency(monkeypatch):
    module = _import_mhc_rl(
        monkeypatch,
        enable_precision_strong_consistency=True,
    )
    mhc = module.NPUmHCRL(_cfg(), pre_only=True)
    _init_mhc_params(mhc)

    hidden_states = torch.arange(12, dtype=torch.float16).view(2, 6)
    normalized_hidden_states = torch.full((2, 6), 2.0, dtype=torch.float32)
    calls = {}

    def fake_rms_norm(hidden, gamma, eps):
        calls["rms_norm"] = (hidden, gamma, eps)
        return normalized_hidden_states, None

    monkeypatch.setattr(
        module.torch_npu,
        "npu_rms_norm",
        fake_rms_norm,
        raising=False,
    )

    output, h_post, h_res = mhc.mhc_pre(hidden_states)

    rms_hidden, rms_gamma, rms_eps = calls["rms_norm"]
    assert rms_hidden.dtype == torch.float32
    assert torch.equal(rms_hidden, hidden_states.float())
    assert torch.equal(rms_gamma, mhc.norm_gamma.view(-1))
    assert rms_eps == mhc.hc_eps

    hpre_weight = torch.sigmoid(
        torch.nn.functional.linear(normalized_hidden_states, mhc.phi.weight)
        * mhc.branch_alpha_pre
        + mhc.branch_beta_pre.view(1, mhc.num_stream)
    ) + mhc.hc_eps
    expected = torch.sum(
        hpre_weight.view(-1, mhc.num_stream, 1)
        * hidden_states.float().view(-1, mhc.num_stream, mhc.hidden_size),
        dim=1,
    ).to(hidden_states.dtype)

    assert torch.allclose(output, expected)
    assert output.dtype == hidden_states.dtype
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
