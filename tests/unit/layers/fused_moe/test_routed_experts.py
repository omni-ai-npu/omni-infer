# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for NPURoutedExperts.weight_loader (omni/layers/fused_moe/layer.py)."""
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _ensure_module(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _RoutedExpertsBase:
    @classmethod
    def register_oot(cls, subcls):
        return subcls

    def weight_loader(self, *, param, loaded_weight, weight_name,
                       shard_id, expert_id, return_success=False):
        return True


@pytest.fixture
def layer_module(monkeypatch):
    logger_module = types.ModuleType("vllm.logger")

    def _init_logger(_name):
        return MagicMock()

    logger_module.init_logger = _init_logger
    logger_module.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    distributed_module = _ensure_module(monkeypatch, "vllm.distributed")

    def _get_ep_group():
        return SimpleNamespace(rank=0, rank_in_group=0, world_size=1)

    def _get_tp_rank():
        return 0

    def _get_tp_world_size():
        return 1

    def _all_gather(tensor, dim=0):
        return tensor

    def _all_reduce(tensor):
        return tensor

    distributed_module.get_ep_group = _get_ep_group
    distributed_module.get_tensor_model_parallel_rank = _get_tp_rank
    distributed_module.get_tensor_model_parallel_world_size = _get_tp_world_size
    distributed_module.tensor_model_parallel_all_gather = _all_gather
    distributed_module.tensor_model_parallel_all_reduce = _all_reduce

    vllm_config_module = _ensure_module(monkeypatch, "vllm.config")
    import enum

    class CUDAGraphMode(enum.Enum):
        NONE = 0

    vllm_config_module.CUDAGraphMode = CUDAGraphMode

    torch_utils_module = _ensure_module(monkeypatch, "vllm.utils.torch_utils")

    def _stub_register(op_name, op_func, **kw):
        ns = getattr(torch.ops, "vllm", None)
        if ns is None:
            ns = type("vllm", (), {})
            torch.ops.vllm = ns
        if not hasattr(ns, op_name):
            setattr(ns, op_name, op_func)

    torch_utils_module.direct_register_custom_op = _stub_register

    forward_context_module = _ensure_module(monkeypatch, "vllm.forward_context")

    def _get_forward_context():
        return SimpleNamespace(
            attn_metadata={}, cudagraph_runtime_mode=CUDAGraphMode.NONE)

    forward_context_module.get_forward_context = _get_forward_context

    utils_module = _ensure_module(monkeypatch, "vllm.model_executor.utils")

    def _set_weight_attrs(param, attrs):
        for key, value in attrs.items():
            setattr(param, key, value)

    utils_module.set_weight_attrs = _set_weight_attrs

    fused_moe_pkg = _ensure_module(monkeypatch, "vllm.model_executor.layers.fused_moe")
    fused_moe_pkg.__path__ = []

    import enum as _enum

    class FusedMoeWeightScaleSupported(_enum.Enum):
        TENSOR = "tensor"
        CHANNEL = "channel"
        GROUP = "group"
        BLOCK = "block"

    fused_moe_layer_module = _ensure_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.layer")

    class FusedMoE(_RoutedExpertsBase):
        pass

    fused_moe_layer_module.FusedMoE = FusedMoE

    class UnquantizedFusedMoEMethod(_RoutedExpertsBase):
        pass

    fused_moe_layer_module.UnquantizedFusedMoEMethod = UnquantizedFusedMoEMethod
    fused_moe_layer_module.FusedMoeWeightScaleSupported = FusedMoeWeightScaleSupported
    fused_moe_pkg.FusedMoE = FusedMoE
    fused_moe_pkg.UnquantizedFusedMoEMethod = fused_moe_layer_module.UnquantizedFusedMoEMethod
    fused_moe_pkg.FusedMoeWeightScaleSupported = FusedMoeWeightScaleSupported

    def _fused_moe_make_expert_params_mapping(*_a, **_kw):
        return []

    fused_moe_pkg.fused_moe_make_expert_params_mapping = (
        _fused_moe_make_expert_params_mapping
    )

    routed_experts_module = _ensure_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.routed_experts")
    routed_experts_module.RoutedExperts = _RoutedExpertsBase

    capturer_module = _ensure_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.routed_experts_capturer")

    class _Capturer:
        @classmethod
        def get_instance(cls):
            return None

    capturer_module.RoutedExpertsCapturer = _Capturer

    runner_pkg = _ensure_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.runner")
    runner_pkg.__path__ = []
    moe_runner_module = _ensure_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.runner.moe_runner")

    class MoERunner(_RoutedExpertsBase):
        pass

    moe_runner_module.MoERunner = MoERunner

    shared_fused_moe_module = _ensure_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.shared_fused_moe")

    class SharedFusedMoE(_RoutedExpertsBase):
        pass

    shared_fused_moe_module.SharedFusedMoE = SharedFusedMoE

    fused_moe_config_module = _ensure_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.config")
    fused_moe_config_module.FusedMoEConfig = SimpleNamespace
    fused_moe_config_module.FusedMoEQuantConfig = SimpleNamespace
    fused_moe_config_module.FusedMoEQuantDesc = SimpleNamespace

    def _quant_flags_to_group_shape(*_a, **_kw):
        return (None, None)

    fused_moe_config_module._quant_flags_to_group_shape = (
        _quant_flags_to_group_shape
    )

    base_path = Path(__file__).resolve().parents[4]
    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(base_path / "omni")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    layers_pkg = types.ModuleType("omni_npu.layers")
    layers_pkg.__path__ = [str(base_path / "omni" / "layers")]
    monkeypatch.setitem(sys.modules, "omni_npu.layers", layers_pkg)

    model_config_pkg = _ensure_module(monkeypatch, "omni_npu.model_config")
    model_config_pkg.__path__ = []
    config_loader_pkg = _ensure_module(
        monkeypatch, "omni_npu.model_config.config_loader"
    )
    config_loader_pkg.__path__ = []
    loader_module = _ensure_module(
        monkeypatch, "omni_npu.model_config.config_loader.loader"
    )
    loader_module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            enable_agrs_finalize_metadata_overlap=False,
            enable_prefetch=False,
            shared_expert_multi_stream=False,
            enable_moe_allreduce=False,
            gmm_nz=False,
            router_gating_in_fp32=False,
            enable_precision_strong_consistency=False,
        ),
        parall_config=SimpleNamespace(ena_seq_parallel=False),
    )

    def _attn_decorator(fn=None, **_kw):
        def _wrap(func):
            return func
        if fn is not None:
            return _wrap(fn)
        return _wrap

    plugin_decorators = _ensure_module(monkeypatch, "omni_npu.plugin_decorators")
    plugin_decorators.attn_decorator = _attn_decorator

    v1_pkg = _ensure_module(monkeypatch, "omni_npu.v1")
    v1_pkg.__path__ = []
    v1_utils = _ensure_module(monkeypatch, "omni_npu.v1.utils")

    def _on_ascend950():
        return False

    v1_utils.on_ascend950 = _on_ascend950

    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", SimpleNamespace(), raising=False)
    if not hasattr(torch.npu, "config"):
        torch.npu.config = SimpleNamespace(allow_internal_format=False)
    elif not hasattr(torch.npu.config, "allow_internal_format"):
        torch.npu.config.allow_internal_format = False

    fused_moe_omni_module = _ensure_module(monkeypatch, "omni_npu.layers.fused_moe.fused_moe")
    fused_moe_omni_module.fused_experts_tp = MagicMock()
    method_base_module = _ensure_module(
        monkeypatch, "omni_npu.layers.fused_moe.fused_moe_method_base")

    class _NPUFusedMoEMethodBase:
        pass

    method_base_module.NPUFusedMoEMethodBase = _NPUFusedMoEMethodBase
    prepare_module = _ensure_module(
        monkeypatch, "omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize")
    prepare_module.PreparePermuteOptions = SimpleNamespace
    prepare_module.PreparePermuteResult = SimpleNamespace
    prepare_module.AGRSPreparePermuteResult = SimpleNamespace
    prefetch_module = _ensure_module(monkeypatch, "omni_npu.layers.prefetch")
    prefetch_module.PrefetchManager = type("PrefetchManager", (), {})
    utils_layer_module = _ensure_module(monkeypatch, "omni_npu.layers.utils")

    def _wait_stream(_stream):
        return None

    def _named_stream(_name):
        return SimpleNamespace(wait_stream=_wait_stream)

    utils_layer_module.named_stream = _named_stream

    torch_npu_module = _ensure_module(monkeypatch, "torch_npu")

    def _npu_format_cast(tensor, _fmt):
        return tensor

    def _npu_grouped_matmul(*args, **_kw):
        return [args[0]]

    def _npu_swiglu(hidden):
        return hidden

    def _current_stream():
        return SimpleNamespace(wait_stream=_wait_stream)

    def _get_device_name(_device):
        return "Ascend910C"

    def _npu_get_option(_key):
        return b"disable"

    torch_npu_module.npu_format_cast = _npu_format_cast
    torch_npu_module.npu_grouped_matmul = _npu_grouped_matmul
    torch_npu_module.npu_swiglu = _npu_swiglu
    torch_npu_module.npu = SimpleNamespace(
        current_stream=_current_stream,
        get_device_name=_get_device_name,
    )
    torch_npu_module.Format = SimpleNamespace(FRACTAL_NZ="nz", ND="nd")
    torch_npu_module._C = SimpleNamespace(_npu_getOption=_npu_get_option)

    sys.modules.pop("omni_npu.layers.fused_moe.layer", None)
    module = importlib.import_module("omni_npu.layers.fused_moe.layer")
    importlib.reload(module)
    return module


def _make_experts(module, *, enable_eplb=False, tp_rank=0):
    """Build a bare NPURoutedExperts with the attrs weight_loader reads."""
    experts = module.NPURoutedExperts.__new__(module.NPURoutedExperts)
    experts.moe_config = SimpleNamespace(
        tp_rank=tp_rank,
        moe_parallel_config=SimpleNamespace(enable_eplb=enable_eplb),
    )
    experts.local_num_experts = 2
    experts.quant_method = SimpleNamespace(planner=MagicMock(), moe_layer_idx=0)
    experts.moe_layer_idx = 0

    def _identity_expert_id(eid):
        return eid

    experts._map_global_expert_id_to_local_expert_id = _identity_expert_id
    load_calls = []

    def _load_per_channel_weight_scale(**kwargs):
        load_calls.append(kwargs)

    experts._load_per_channel_weight_scale = _load_per_channel_weight_scale
    return experts, load_calls


def _channel_param(*shape):
    param = torch.nn.Parameter(torch.zeros(*shape))
    setattr(param, "quant_method", "channel")
    setattr(param, "is_weight_transposed", False)
    return param


def _load_aux_weight(experts, *, param, loaded_weight, weight_name, expert_id,
                     shard_id="w1"):
    """Call weight_loader with return_success=True and return the result."""
    return experts.weight_loader(
        param=param,
        loaded_weight=loaded_weight,
        weight_name=weight_name,
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=True,
    )


def _capturing_fused_moe(captured):
    """Build a FusedMoE stub that records constructor kwargs into ``captured``."""
    class _FusedMoE:
        def __init__(self, *a, **kw):
            captured.update(kw)

    return _FusedMoE


@pytest.mark.unit
def test_weight_loader_delegates_non_aux_to_super(layer_module, monkeypatch):
    """Non-aux weight names are forwarded to the native RoutedExperts.weight_loader."""
    module = layer_module
    experts, load_calls = _make_experts(module)
    super_called = []
    # Patch the SUPER class (RoutedExperts stub) so we can observe delegation
    # without stubbing the production override itself.
    super_cls = module.NPURoutedExperts.__mro__[1]

    def _super_weight_loader(_self, **kw):
        super_called.append(kw)
        return True

    monkeypatch.setattr(super_cls, "weight_loader", _super_weight_loader)

    param = _channel_param(2, 4)
    result = _load_aux_weight(
        experts,
        param=param,
        loaded_weight=torch.ones(2, 3, 4),
        weight_name="w13_weight",
        shard_id="0",
        expert_id=0,
    )
    assert result is True
    assert load_calls == []  # aux loader never touched
    assert super_called  # super().weight_loader was invoked
    assert super_called[0]["weight_name"] == "w13_weight"


@pytest.mark.unit
def test_weight_loader_routes_int4_scale_with_correct_shard_dim(layer_module):
    """w13_weight_int4_scale (per-expert 2D, channel last) uses shard_dim=1."""
    module = layer_module
    experts, load_calls = _make_experts(module)
    # param.data shape (E=2, 1, C=4): W4A8 aux layout, channel last.
    param = _channel_param(2, 1, 4)
    _load_aux_weight(
        experts,
        param=param,
        loaded_weight=torch.ones(4),  # per-expert 1D
        weight_name="w13_weight_int4_scale",
        expert_id=0,
    )
    assert len(load_calls) == 1
    call = load_calls[0]
    assert call["shard_id"] == "w1"
    assert call["shard_dim"] == 1  # largest non-expert dim of (1, 4)
    assert call["tp_rank"] == 0


@pytest.mark.unit
def test_weight_loader_routes_weight_bias_uses_largest_dim(layer_module):
    """weight_bias (1D per expert) uses shard_dim=0 (only non-expert dim)."""
    module = layer_module
    experts, load_calls = _make_experts(module)
    param = _channel_param(2, 4)  # (E, C)
    _load_aux_weight(
        experts,
        param=param,
        loaded_weight=torch.ones(4),
        weight_name="w13_weight_bias",
        expert_id=1,
    )
    assert load_calls[0]["shard_dim"] == 0


@pytest.mark.unit
def test_weight_loader_routes_weight_offset_channel_first(layer_module):
    """W8A8 offset (C, 1) layout uses shard_dim=0 (largest non-expert dim)."""
    module = layer_module
    experts, load_calls = _make_experts(module)
    param = _channel_param(2, 4, 1)  # (E, C, 1): channel first
    _load_aux_weight(
        experts,
        param=param,
        loaded_weight=torch.ones(4, 1),
        weight_name="w13_weight_offset",
        expert_id=0,
    )
    assert load_calls[0]["shard_dim"] == 0  # C=4 > 1


@pytest.mark.unit
def test_weight_loader_full_load_3d_uses_non_expert_dims(layer_module):
    """3D loaded_weight (all experts) uses shard_dim as the largest non-expert dim."""
    module = layer_module
    experts, load_calls = _make_experts(module)
    param = _channel_param(2, 1, 8)  # (E, 1, C)
    _load_aux_weight(
        experts,
        param=param,
        loaded_weight=torch.ones(2, 1, 8),  # 3D -> full_load
        weight_name="w13_weight_int4_scale",
        expert_id=0,
    )
    assert load_calls[0]["shard_dim"] == 2  # dims 1,2 -> max is dim 2 (C=8)


@pytest.mark.unit
def test_weight_loader_raises_on_non_channel_quant_method(layer_module):
    """Aux params require quant_method == channel, else ValueError."""
    module = layer_module
    experts, load_calls = _make_experts(module)
    param = torch.nn.Parameter(torch.zeros(2, 4))
    setattr(param, "quant_method", "tensor")  # not channel
    setattr(param, "is_weight_transposed", False)
    with pytest.raises(ValueError, match="quant method must be"):
        experts.weight_loader(
            param=param,
            loaded_weight=torch.ones(4),
            weight_name="w13_weight_int4_scale",
            shard_id="w1",
            expert_id=0,
        )


@pytest.mark.unit
def test_weight_loader_skips_non_local_expert(layer_module):
    """A mapped expert id of -1 makes the loader return False/None without loading."""
    module = layer_module
    experts, load_calls = _make_experts(module)

    def _missing_expert(_eid):
        return -1

    experts._map_global_expert_id_to_local_expert_id = _missing_expert
    param = _channel_param(2, 1, 4)

    assert _load_aux_weight(
        experts,
        param=param,
        loaded_weight=torch.ones(4),
        weight_name="w13_weight_int4_scale",
        expert_id=5,
    ) is False
    assert experts.weight_loader(
        param=param, loaded_weight=torch.ones(4),
        weight_name="w13_weight_int4_scale", shard_id="w1", expert_id=5,
    ) is None
    assert load_calls == []


@pytest.mark.unit
def test_weight_loader_eplb_remaps_expert_id(layer_module):
    """Under eplb, a non-local expert short-circuits before scale loading."""
    module = layer_module
    experts, load_calls = _make_experts(module, enable_eplb=True)

    def _expert_not_on_rank(*_a, **_kw):
        return (False, 0)

    experts.quant_method.planner.is_expert_on_current_rank = _expert_not_on_rank
    param = _channel_param(2, 1, 4)
    assert _load_aux_weight(
        experts,
        param=param,
        loaded_weight=torch.ones(4),
        weight_name="w13_weight_int4_scale",
        expert_id=7,
    ) is False
    assert load_calls == []


@pytest.mark.unit
def test_npu_fused_moe_passes_routed_experts_cls(layer_module):
    """NPUFusedMoE / NPUSharedFusedMoE factories forward routed_experts_cls."""
    module = layer_module
    captured = {}

    module.FusedMoE = _capturing_fused_moe(captured)
    # NPUFusedMoE.__new__ calls FusedMoE(..., routed_experts_cls=NPURoutedExperts)
    fused = module.NPUFusedMoE(gate=object())
    assert captured.get("routed_experts_cls") is module.NPURoutedExperts
    assert captured.get("runner_cls") is module.NPUFusedMoERunner


@pytest.mark.unit
def test_npu_shared_fused_moe_passes_routed_experts_cls(layer_module):
    """NPUSharedFusedMoE also forwards routed_experts_cls + shared_experts."""
    module = layer_module
    captured = {}

    module.FusedMoE = _capturing_fused_moe(captured)
    shared = object()
    module.NPUSharedFusedMoE(gate=object(), shared_experts=shared)
    assert captured.get("routed_experts_cls") is module.NPURoutedExperts
    assert captured.get("runner_cls") is module.NPUFusedMoERunner
    assert captured.get("shared_experts") is shared


@pytest.mark.unit
def test_moe_enable_eplb_reads_layer_attr_then_config(layer_module):
    """_moe_enable_eplb prefers the layer flag, then moe_config, else False."""
    module = layer_module
    assert module._moe_enable_eplb(SimpleNamespace(enable_eplb=True)) is True
    assert module._moe_enable_eplb(
        SimpleNamespace(
            enable_eplb=None,
            moe_config=SimpleNamespace(
                moe_parallel_config=SimpleNamespace(enable_eplb=True)
            ),
        )
    ) is True
    assert module._moe_enable_eplb(SimpleNamespace(enable_eplb=None, moe_config=None)) is False


@pytest.mark.unit
def test_npu_moe_forward_and_shared_dispatch(layer_module, monkeypatch):
    """npu_moe_forward / npu_moe_forward_shared both call _npu_moe_apply."""
    module = layer_module
    hidden = torch.ones(2, 4)
    logits = torch.zeros(2, 3)
    applied = MagicMock(return_value=torch.full((2, 4), 7.0))
    layer = SimpleNamespace(
        shared_experts=None,
        quant_method=SimpleNamespace(apply=applied),
        top_k=1,
        renormalize=False,
        use_grouped_topk=False,
        global_num_experts=3,
        expert_map=None,
        rocm_aiter_fmoe_enabled=False,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        activation="silu",
        apply_router_weight_on_input=False,
        enable_eplb=False,
    )

    def _get_forward_context():
        return SimpleNamespace(no_compile_layers={"moe.0": layer})

    monkeypatch.setattr(
        module,
        "get_forward_context",
        _get_forward_context,
    )

    out = module.npu_moe_forward(hidden, logits, "moe.0")
    assert torch.equal(out, torch.full((2, 4), 7.0))

    layer.shared_experts = object()
    shared_out = module.npu_moe_forward_shared(hidden, logits, "moe.0")
    assert torch.equal(shared_out, torch.full((2, 4), 7.0))
    assert applied.call_count == 2


@pytest.mark.unit
def test_npu_moe_forward_fake_shapes(layer_module):
    module = layer_module
    hidden = torch.ones(3, 5)
    logits = torch.zeros(3, 2)
    fake = module.npu_moe_forward_fake(hidden, logits, "moe.0")
    assert fake.shape == hidden.shape
    shared, fused = module.npu_moe_forward_shared_fake(hidden, logits, "moe.0")
    assert shared.shape == hidden.shape
    assert fused.shape == hidden.shape


@pytest.mark.unit
def test_runner_getattr_delegates_to_routed_experts_and_config(layer_module):
    module = layer_module
    runner = module.NPUFusedMoERunner.__new__(module.NPUFusedMoERunner)
    experts = SimpleNamespace(w13_weight="w13", moe_config=SimpleNamespace(ep_size=8))
    runner._modules = {"routed_experts": experts}

    assert runner.w13_weight == "w13"
    assert runner.ep_size == 8
    with pytest.raises(AttributeError, match="__deepcopy__"):
        _ = runner.__deepcopy__
    with pytest.raises(AttributeError, match="missing"):
        _ = runner.missing


@pytest.mark.unit
def test_runner_shared_experts_unwraps_vllm_container(layer_module, monkeypatch):
    module = layer_module
    runner = module.NPUFusedMoERunner.__new__(module.NPUFusedMoERunner)
    inner = object()

    class SharedExperts:
        def __init__(self, layer):
            self._layer = layer

    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.runner.shared_experts",
        SimpleNamespace(SharedExperts=SharedExperts),
    )
    runner._shared_experts = SharedExperts(inner)
    assert runner.shared_experts is inner

    runner.shared_experts = "plain"
    assert runner._shared_experts == "plain"
    assert runner.shared_experts == "plain"


@pytest.mark.unit
def test_runner_maybe_init_modular_kernel_is_noop(layer_module):
    module = layer_module
    runner = module.NPUFusedMoERunner.__new__(module.NPUFusedMoERunner)
    assert runner.maybe_init_modular_kernel() is None


@pytest.mark.unit
def test_weight_loader_eplb_remaps_local_expert(layer_module):
    """When eplb reports the expert is local, the loader remaps and loads the scale."""
    module = layer_module
    experts, load_calls = _make_experts(module, enable_eplb=True)

    def _expert_on_current_rank(*_a, **_k):
        return (True, 1)

    def _identity_expert_id(eid):
        return eid

    experts.quant_method.planner.is_expert_on_current_rank = _expert_on_current_rank
    experts._map_global_expert_id_to_local_expert_id = _identity_expert_id
    param = _channel_param(4, 1, 4)
    result = _load_aux_weight(
        experts,
        param=param,
        loaded_weight=torch.ones(4),
        weight_name="w13_weight_int4_scale",
        expert_id=7,
    )
    assert result is True
    assert load_calls
    assert load_calls[0]["shard_dim"] == 1

