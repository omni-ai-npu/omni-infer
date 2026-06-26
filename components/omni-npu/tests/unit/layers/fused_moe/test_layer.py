# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
import importlib
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


class _DummyStream:
    def __init__(self):
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other)
        return None


def _ensure_module(monkeypatch: pytest.MonkeyPatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def layer_module(monkeypatch):
    # Mock vllm.logger before any import that depends on it
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: MagicMock()
    logger_module.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    # Ensure vllm.distributed exists in sys.modules with get_dp_group
    if "vllm.distributed" not in sys.modules:
        distributed_module = types.ModuleType("vllm.distributed")
        monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)

    # Supplement missing mocks not provided by conftest.py autouse fixture
    # vllm.config with CUDAGraphMode (needed by prepare_permute_unpermute_finalize.py)
    vllm_config_module = _ensure_module(monkeypatch, "vllm.config")
    import enum
    class CUDAGraphMode(enum.Enum):
        NONE = 0
        FULL = 1
        PIECEWISE = 2
        FULL_AND_PIECEWISE = 3
        FULL_DECODE_ONLY = 4
    vllm_config_module.CUDAGraphMode = CUDAGraphMode

    # vllm.utils.torch_utils with direct_register_custom_op (needed by layer.py)
    torch_utils_module = _ensure_module(monkeypatch, "vllm.utils.torch_utils")
    def _stub_direct_register_custom_op(op_name, op_func, mutates_args=None,
                                         fake_impl=None, target_lib=None,
                                         dispatch_key=None, tags=()):
        ns = getattr(torch.ops, "vllm", None)
        if ns is None:
            ns = type("vllm", (), {})()
            torch.ops.vllm = ns
        if not hasattr(ns, op_name):
            setattr(ns, op_name, op_func)
    torch_utils_module.direct_register_custom_op = _stub_direct_register_custom_op

    # omni_npu.layers.prefetch with PrefetchManager (needed by layer.py)
    prefetch_module = _ensure_module(monkeypatch, "omni_npu.layers.prefetch")
    prefetch_module.PrefetchManager = type("PrefetchManager", (), {})

    # omni_npu.plugin_decorators with attn_decorator (needed by layer.py)
    plugin_decorators_module = _ensure_module(monkeypatch, "omni_npu.plugin_decorators")
    plugin_decorators_module.attn_decorator = (
        lambda fn=None, **kw: (fn if fn is not None else lambda f: f)
    )

    # get_ep_group needs rank_in_group (needed by layer.py weight_loader)
    monkeypatch.setattr(
        sys.modules["vllm.distributed"],
        "get_ep_group",
        lambda: SimpleNamespace(rank=0, rank_in_group=0, world_size=1),
        raising=False,
    )

    # get_dp_group (needed by prepare_permute_unpermute_finalize.py)
    monkeypatch.setattr(
        sys.modules["vllm.distributed"],
        "get_dp_group",
        lambda: SimpleNamespace(world_size=1, rank=0),
        raising=False,
    )

    torch_npu = sys.modules["torch_npu"]
    torch_npu.npu_format_cast = MagicMock(side_effect=lambda tensor, _: tensor)
    torch_npu.npu_grouped_matmul = MagicMock(
        side_effect=lambda inputs, _weights, **kwargs: [inputs[0]]
    )
    torch_npu.npu_swiglu = MagicMock(side_effect=lambda tensor: tensor)
    torch_npu.npu_moe_gating_top_k = MagicMock(
        return_value=(
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int32),
            torch.zeros(2, 1, dtype=torch.int32),
        )
    )
    torch_npu.npu = SimpleNamespace(
        get_device_name=lambda _: "Ascend910C",
        current_stream=lambda: _DummyStream(),
    )
    torch_npu.Format = SimpleNamespace(FRACTAL_NZ="FRACTAL_NZ", ND="ND")
    torch_npu._C = SimpleNamespace(_npu_getOption=lambda _key: b"disable")

    context_holder = SimpleNamespace(attn_metadata={})
    monkeypatch.setattr(
        sys.modules["vllm.distributed"],
        "get_dp_group",
        lambda: SimpleNamespace(
            world_size=1,
            all_gather=lambda tensor, dim=0: tensor,
            reduce_scatter=lambda tensor, dim=0: tensor,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["vllm.distributed"],
        "get_tp_group",
        lambda: SimpleNamespace(all_reduce=lambda x: x),
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["vllm.forward_context"],
        "get_forward_context",
        lambda: context_holder,
        raising=False,
    )

    sys.modules.pop("omni_npu.layers.fused_moe.layer", None)
    module = importlib.import_module("omni_npu.layers.fused_moe.layer")
    importlib.reload(module)
    return module, torch_npu, context_holder


class _DummyStrategy:
    def prepare_permute(self, layer, x, topk_ids):
        return SimpleNamespace(
            hidden_states_sorted_by_experts=x,
            expert_tokens=torch.tensor([x.shape[0]], dtype=torch.int64),
            dynamic_scale=None,
        )

    def prepare_finalize_metadata(self, layer, topk_weights, result):
        return None

    def unpermute_finalize(
        self,
        layer,
        hidden_states,
        topk_ids,
        topk_weights,
        result,
        finalize_params=None,
        finalize_metadata=None,
    ):
        return hidden_states


@contextmanager
def _stream_ctx(_stream):
    yield


def _stub_apply_prefetch_attrs(method, module, monkeypatch):
    method.sub_stream = _DummyStream()
    method.agrs_overlap_stream = _DummyStream()
    method.enable_agrs_finalize_metadata_overlap = True
    method.model_prefetch = MagicMock()
    monkeypatch.setattr(module.torch.npu, "stream", _stream_ctx, raising=False)


@pytest.mark.unit
def test_select_experts_profile_mode(layer_module):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = None
    module.get_ep_group = MagicMock(return_value=SimpleNamespace(rank_in_group=1))

    topk_weights, topk_ids = module.NPUFusedMoE.select_experts(
        router_logits=torch.zeros(2, 4),
        top_k=2,
        use_grouped_topk=False,
        renormalize=False,
    )

    assert topk_weights.shape == (2, 2)
    assert topk_ids.shape == (2, 2)
    assert torch.all(topk_ids < 4)


@pytest.mark.unit
def test_select_experts_grouped_topk_requires_group_args(layer_module):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}

    with pytest.raises(ValueError, match="topk_group is None"):
        module.NPUFusedMoE.select_experts(
            router_logits=torch.zeros(1, 4),
            top_k=1,
            use_grouped_topk=True,
            renormalize=False,
            topk_group=None,
            num_expert_group=2,
        )


@pytest.mark.unit
def test_apply_experts_uses_grouped_matmul_twice(layer_module):
    module, torch_npu, _ = layer_module
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        w13_weight=torch.ones(2, 4, 4),
        w2_weight=torch.ones(2, 4, 4),
    )
    prepare_result = module.PreparePermuteResult(
        hidden_states_sorted_by_experts=torch.ones(3, 4),
        expert_tokens=torch.tensor([2, 1], dtype=torch.int64),
        dynamic_scale=None,
    )

    out = method.apply_experts(layer, prepare_result)
    assert out.shape == (3, 4)
    assert torch_npu.npu_grouped_matmul.call_count == 2


@pytest.mark.unit
def test_maybe_all_reduce_tensor_model_parallel(layer_module):
    module, _, _ = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.moe_parallel_config = SimpleNamespace(use_ep=True)
    module.tensor_model_parallel_all_reduce = MagicMock(return_value=torch.tensor([9.0]))

    x = torch.tensor([1.0])
    assert torch.equal(fused.maybe_all_reduce_tensor_model_parallel(x), x)
    module.tensor_model_parallel_all_reduce.assert_not_called()

    fused.moe_parallel_config.use_ep = False
    y = fused.maybe_all_reduce_tensor_model_parallel(x)
    module.tensor_model_parallel_all_reduce.assert_called_once_with(x)
    assert torch.equal(y, torch.tensor([9.0]))


@pytest.mark.unit
def test_select_experts_custom_routing_function(layer_module):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    custom_fn = MagicMock(
        return_value=(
            torch.full((2, 1), 0.5, dtype=torch.float32),
            torch.ones(2, 1, dtype=torch.int32),
        )
    )

    topk_weights, topk_ids = module.NPUFusedMoE.select_experts(
        router_logits=torch.zeros(2, 4),
        top_k=1,
        use_grouped_topk=False,
        renormalize=True,
        custom_routing_function=custom_fn,
    )

    custom_fn.assert_called_once()
    assert torch.equal(topk_weights, torch.full((2, 1), 0.5, dtype=torch.float32))
    assert torch.equal(topk_ids, torch.ones(2, 1, dtype=torch.int32))


@pytest.mark.unit
def test_apply_slices_and_gathers_on_all2all(layer_module, monkeypatch):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 2
    method.tp_rank = 1
    _stub_apply_prefetch_attrs(method, module, monkeypatch)
    method.select_communication_strategy = MagicMock(return_value=("all2all", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 3.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 4.0))
    monkeypatch.setattr(
        module,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=dim),
    )
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int32),
            )
        ),
    )

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=None,
        shared_experts=None,
    )
    output = method.apply(
        layer=layer,
        hidden_states=torch.arange(12, dtype=torch.float32).view(3, 4),
        router_logits=torch.zeros(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    assert output.shape == (3, 4)
    assert method.apply_prepare_permute.call_args.args[2].shape == (2, 4)


@pytest.mark.unit
def test_apply_agrs_overlaps_experts_while_metadata_stays_on_main_stream(layer_module, monkeypatch):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    _stub_apply_prefetch_attrs(method, module, monkeypatch)
    events = []
    metadata = object()
    method.select_communication_strategy = MagicMock(return_value=("agrs", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.prepare_finalize_metadata = MagicMock(
        side_effect=lambda *args: events.append("metadata") or metadata
    )
    method.apply_experts = MagicMock(
        side_effect=lambda **kwargs: events.append("experts") or torch.full((2, 4), 2.0)
    )

    def _finalize(*args, **kwargs):
        events.append("finalize")
        assert kwargs["finalize_metadata"] is metadata
        return torch.full((2, 4), 3.0)

    method.apply_unpermute_finalize = MagicMock(side_effect=_finalize)
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int32),
            )
        ),
    )

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=None,
        shared_experts=None,
    )
    output = method.apply(
        layer=layer,
        hidden_states=torch.ones(2, 4),
        router_logits=torch.zeros(2, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    assert torch.equal(output, torch.full((2, 4), 3.0))
    assert events == ["experts", "metadata", "finalize"]
    assert method.agrs_overlap_stream.waits


@pytest.mark.unit
def test_apply_with_gate_and_shared_experts_adds_plugin_output(layer_module, monkeypatch):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    _stub_apply_prefetch_attrs(method, module, monkeypatch)
    method.select_communication_strategy = MagicMock(return_value=("agrs", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 2.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 3.0))
    monkeypatch.setattr(module, "named_stream", lambda _name: _DummyStream())
    monkeypatch.setattr(module.torch.npu, "current_stream", lambda: _DummyStream(), raising=False)
    monkeypatch.setattr(module.torch.npu, "stream", _stream_ctx, raising=False)
    monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int32),
            )
        ),
    )

    def _gate(x):
        return torch.zeros(x.shape[0], 2, dtype=torch.float32), None

    shared = lambda x: torch.full_like(x, 5.0)
    shared.gate_up_proj = SimpleNamespace(tp_size=1)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=_gate,
        shared_experts=shared,
    )
    output = method.apply(
        layer=layer,
        hidden_states=torch.ones(2, 4, dtype=torch.float32),
        router_logits=None,
        top_k=1,
        renormalize=False,
    )
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    assert isinstance(output, tuple)
    shared_output, routed_output = output
    assert torch.equal(shared_output, torch.full((2, 4), 5.0))
    assert torch.equal(routed_output, torch.full((2, 4), 8.0))


@pytest.mark.unit
def test_apply_shared_experts_reduce_branch(layer_module, monkeypatch):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 2
    method.tp_rank = 1
    _stub_apply_prefetch_attrs(method, module, monkeypatch)
    method.select_communication_strategy = MagicMock(return_value=("all2all", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 2.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 1.0))
    monkeypatch.setattr(module, "named_stream", lambda _name: _DummyStream())
    monkeypatch.setattr(module.torch.npu, "current_stream", lambda: _DummyStream(), raising=False)
    monkeypatch.setattr(
        module,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=dim),
    )
    all_reduce_mock = MagicMock(return_value=torch.full((2, 4), 6.0))
    monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", all_reduce_mock)
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int32),
            )
        ),
    )

    hidden_states = torch.arange(12, dtype=torch.float32).view(3, 4)
    shared = MagicMock(return_value=torch.full((3, 4), 6.0))
    shared.gate_up_proj = SimpleNamespace(tp_size=2)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=None,
        shared_experts=shared,
    )
    output = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=torch.zeros(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    assert isinstance(output, tuple)
    all_reduce_mock.assert_called_once()
    assert shared.call_count == 1
    assert torch.equal(shared.call_args.args[0], hidden_states)


@pytest.mark.unit
def test_weight_loader_handles_transposed_weights(layer_module, monkeypatch):
    module, _, _ = layer_module
    super_weight_loader = MagicMock(return_value=True)
    monkeypatch.setattr(module.FusedMoE, "weight_loader", super_weight_loader, raising=False)
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.enable_eplb = False

    param = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).view(2, 4, 4))
    setattr(param, "is_weight_transposed", True)
    original = param.data.clone()

    result = fused.weight_loader(
        param=param,
        loaded_weight=torch.zeros(2, 3, 4),
        weight_name="w13_weight",
        shard_id="0",
        expert_id=0,
        return_success=True,
    )

    assert result is True
    assert param.shape == original.shape
    assert super_weight_loader.call_count == 1


@pytest.mark.unit
def test_forward_uses_expert_mask_when_rocm_enabled(layer_module, monkeypatch):
    module, _, _ = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.shared_experts = None
    fused.layer_name = "dummy.layer"
    fused.quant_method = SimpleNamespace(apply=MagicMock(return_value=torch.ones(1, 2)))
    fused.top_k = 2
    fused.renormalize = False
    fused.use_grouped_topk = False
    fused.global_num_experts = 4
    fused.expert_map = torch.tensor([0, 1])
    fused.expert_mask = torch.tensor([1, 0])
    fused.rocm_aiter_fmoe_enabled = True
    fused.topk_group = None
    fused.num_expert_group = None
    fused.custom_routing_function = None
    fused.scoring_func = "softmax"
    fused.routed_scaling_factor = 1.0
    fused.e_score_correction_bias = None
    fused.activation = "silu"
    fused.apply_router_weight_on_input = False
    fused.enable_eplb = False
    monkeypatch.setattr(
        module.torch.ops.vllm,
        "npu_moe_forward",
        lambda hidden_states, router_logits, layer_name: module._npu_moe_apply(
            fused, hidden_states, router_logits
        ),
    )

    out = fused.forward(
        hidden_states=torch.ones(1, 2, dtype=torch.float32),
        router_logits=torch.zeros(1, 4, dtype=torch.float32),
    )

    assert torch.equal(out, torch.ones(1, 2))
    kwargs = fused.quant_method.apply.call_args.kwargs
    assert torch.equal(kwargs["expert_map"], fused.expert_mask)


@pytest.mark.unit
def test_unquantized_method_init_sets_tp_info(layer_module, monkeypatch):
    module, _, _ = layer_module
    monkeypatch.setattr(
        module.UnquantizedFusedMoEMethod,
        "__init__",
        lambda self, moe: setattr(self, "_moe", moe),
        raising=False,
    )
    module.get_tensor_model_parallel_world_size = lambda: 4
    module.get_tensor_model_parallel_rank = lambda: 2

    method = module.NPUUnquantizedFusedMoEMethod(SimpleNamespace())
    assert method.tp_size == 4
    assert method.tp_rank == 2


@pytest.mark.unit
def test_apply_slice_path_without_padding_slices_router_logits(layer_module, monkeypatch):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 2
    method.tp_rank = 1
    _stub_apply_prefetch_attrs(method, module, monkeypatch)
    method.select_communication_strategy = MagicMock(return_value=("all2all", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 2.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 3.0))
    monkeypatch.setattr(
        module,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=dim),
    )
    select_mock = MagicMock(
        return_value=(
            torch.ones(2, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(module.NPUFusedMoE, "select_experts", select_mock)

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=None,
        shared_experts=None,
    )
    router_logits = torch.arange(16, dtype=torch.float32).view(4, 4)
    out = method.apply(
        layer=layer,
        hidden_states=torch.arange(16, dtype=torch.float32).view(4, 4),
        router_logits=router_logits,
        top_k=1,
        renormalize=False,
    )

    assert out.shape == (4, 4)
    assert select_mock.call_args.kwargs["router_logits"].shape == (2, 4)


@pytest.mark.unit
def test_apply_single_card_a5_shared_experts_returns_shared_and_routed(layer_module, monkeypatch):
    """A5 single-card: stream sync + shared expert handling, returns (shared, routed+shared)."""
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    method.on_ascend950 = True
    method.sub_stream = _DummyStream()
    method.model_prefetch = MagicMock()
    method.select_communication_strategy = MagicMock(return_value=("all2all", _DummyStrategy()))
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.full((2, 1), 0.5, dtype=torch.float32),
                torch.ones(2, 1, dtype=torch.int32),
            )
        ),
    )
    routed_mock = MagicMock(return_value=torch.full((2, 4), 2.0))
    monkeypatch.setattr(module, "fused_experts_tp", routed_mock)
    shared = MagicMock(return_value=torch.full((2, 4), 5.0))
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False),
        gate=None,
        shared_experts=shared,
    )

    shared_output, routed_output = method.apply(
        layer=layer,
        hidden_states=torch.ones(2, 4),
        router_logits=torch.zeros(2, 3),
        top_k=1,
        renormalize=False,
    )

    routed_mock.assert_called_once()
    shared.assert_called_once()
    assert torch.equal(shared_output, torch.full((2, 4), 5.0))
    assert torch.equal(routed_output, torch.full((2, 4), 7.0))


@pytest.mark.unit
def test_apply_single_card_a5_no_shared_experts_returns_routed(layer_module, monkeypatch):
    """A5 single-card without shared experts: returns routed output directly."""
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    method.on_ascend950 = True
    method.sub_stream = _DummyStream()
    method.model_prefetch = MagicMock()
    method.select_communication_strategy = MagicMock(return_value=("all2all", _DummyStrategy()))
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.full((2, 1), 0.5, dtype=torch.float32),
                torch.ones(2, 1, dtype=torch.int32),
            )
        ),
    )
    routed_mock = MagicMock(return_value=torch.full((2, 4), 3.0))
    monkeypatch.setattr(module, "fused_experts_tp", routed_mock)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False),
        gate=None,
        shared_experts=None,
    )

    output = method.apply(
        layer=layer,
        hidden_states=torch.ones(2, 4),
        router_logits=torch.zeros(2, 3),
        top_k=1,
        renormalize=False,
    )

    routed_mock.assert_called_once()
    assert not isinstance(output, tuple)
    assert torch.equal(output, torch.full((2, 4), 3.0))


@pytest.mark.unit
def test_apply_single_card_non_a5_returns_routed_directly(layer_module, monkeypatch):
    """Non-A5 single-card: falls back to original fused_experts_tp() return, ignores shared_experts."""
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    method.on_ascend950 = False
    method.sub_stream = _DummyStream()
    method.model_prefetch = MagicMock()
    method.select_communication_strategy = MagicMock(return_value=("all2all", _DummyStrategy()))
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.full((2, 1), 0.5, dtype=torch.float32),
                torch.ones(2, 1, dtype=torch.int32),
            )
        ),
    )
    routed_mock = MagicMock(return_value=torch.full((2, 4), 4.0))
    monkeypatch.setattr(module, "fused_experts_tp", routed_mock)
    shared = MagicMock(return_value=torch.full((2, 4), 9.0))
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False),
        gate=None,
        shared_experts=shared,
    )

    output = method.apply(
        layer=layer,
        hidden_states=torch.ones(2, 4),
        router_logits=torch.zeros(2, 3),
        top_k=1,
        renormalize=False,
    )

    routed_mock.assert_called_once()
    shared.assert_not_called()
    assert not isinstance(output, tuple)
    assert torch.equal(output, torch.full((2, 4), 4.0))


@pytest.mark.unit
def test_process_weights_after_loading_transposes_and_marks(layer_module, monkeypatch):
    module, torch_npu, _ = layer_module
    monkeypatch.setattr(
        module.UnquantizedFusedMoEMethod,
        "process_weights_after_loading",
        lambda self, layer: None,
        raising=False,
    )
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(torch.randn(2, 3, 4)),
        w2_weight=torch.nn.Parameter(torch.randn(2, 5, 6)),
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (2, 4, 3)
    assert layer.w2_weight.shape == (2, 6, 5)
    assert getattr(layer.w13_weight, "is_weight_transposed", False) is True
    assert getattr(layer.w2_weight, "is_weight_transposed", False) is True
    assert torch_npu.npu_format_cast.call_count == 2


@pytest.mark.unit
def test_fused_moe_init_sets_gate_and_strategy_selector(layer_module, monkeypatch):
    module, _, _ = layer_module
    selector_mock = MagicMock()

    def _fake_super_init(self, *args, **kwargs):
        self.quant_method = SimpleNamespace(make_communication_strategy_selector=selector_mock)

    monkeypatch.setattr(module.FusedMoE, "__init__", _fake_super_init, raising=False)
    gate_obj = object()
    fused = module.NPUFusedMoE(gate=gate_obj)

    assert fused.gate is gate_obj
    assert fused.is_internal_router is True
    selector_mock.assert_called_once_with(fused)


@pytest.mark.unit
def test_weight_loader_skips_non_local_expert(layer_module):
    module, _, _ = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.enable_eplb = False
    fused.tp_rank = 0
    fused._map_global_expert_id_to_local_expert_id = lambda eid: -1
    fused._load_per_channel_weight_scale = lambda **kw: pytest.fail(
        "scale loader must not run for non-local expert"
    )

    param = torch.nn.Parameter(torch.zeros(2, 4, dtype=torch.float32))
    setattr(param, "quant_method", "channel")
    setattr(param, "is_weight_transposed", False)

    assert fused.weight_loader(
        param=param,
        loaded_weight=torch.ones(4),
        weight_name="w13_weight_int4_scale",
        shard_id="0",
        expert_id=999,
        return_success=True,
    ) is False
    assert fused.weight_loader(
        param=param,
        loaded_weight=torch.ones(4),
        weight_name="w13_weight_int4_scale",
        shard_id="0",
        expert_id=999,
        return_success=False,
    ) is None


@pytest.mark.unit
def test_weight_loader_handles_non_full_load_transposed_branch(layer_module, monkeypatch):
    module, _, _ = layer_module
    super_weight_loader = MagicMock(return_value=True)
    monkeypatch.setattr(module.FusedMoE, "weight_loader", super_weight_loader, raising=False)
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.enable_eplb = False

    param = torch.nn.Parameter(torch.arange(32, dtype=torch.float32).view(2, 4, 4))
    setattr(param, "is_weight_transposed", True)
    result = fused.weight_loader(
        param=param,
        loaded_weight=torch.zeros(3, 4),
        weight_name="w2_weight",
        shard_id="0",
        expert_id=1,
        return_success=True,
    )

    assert result is True
    assert param.shape == (2, 4, 4)
    assert super_weight_loader.call_count == 1


@pytest.mark.unit
def test_weight_loader_bias_branch_channel_quant(layer_module, monkeypatch):
    module, _, _ = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    fused.enable_eplb = False
    fused.tp_rank = 0
    fused._map_global_expert_id_to_local_expert_id = lambda eid: eid
    fused._transpose_if_needed = lambda p: None
    load_called = {}
    fused._load_per_channel_weight_scale = lambda **kw: load_called.update(kw)

    param = torch.nn.Parameter(torch.zeros(2, 4, dtype=torch.float32))
    setattr(param, "quant_method", "channel")
    setattr(param, "is_weight_transposed", False)

    result = fused.weight_loader(
        param=param,
        loaded_weight=torch.ones(4),
        weight_name="w13_weight_int4_scale",
        shard_id="0",
        expert_id=0,
        return_success=True,
    )

    assert result is True
    assert "shard_dim" in load_called
    assert load_called["shard_dim"] == 1


@pytest.mark.unit
def test_maybe_init_modular_kernel_returns_none(layer_module):
    module, _, _ = layer_module
    fused = module.NPUFusedMoE.__new__(module.NPUFusedMoE)
    assert fused.maybe_init_modular_kernel() is None


@pytest.mark.unit
def test_select_experts_grouped_topk_requires_num_expert_group(layer_module):
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}

    with pytest.raises(ValueError, match="num_expert_group is None"):
        module.NPUFusedMoE.select_experts(
            router_logits=torch.zeros(1, 4),
            top_k=1,
            use_grouped_topk=True,
            renormalize=False,
            topk_group=1,
            num_expert_group=None,
        )


@pytest.mark.unit
def test_select_experts_default_path_with_renormalize(layer_module):
    module, torch_npu, context_holder = layer_module
    context_holder.attn_metadata = {}
    torch_npu.npu_moe_gating_top_k.return_value = (
        torch.tensor([[2.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([[0, 1]], dtype=torch.int32),
    )

    weights, ids = module.NPUFusedMoE.select_experts(
        router_logits=torch.zeros(1, 4),
        top_k=2,
        use_grouped_topk=False,
        renormalize=True,
    )

    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
    assert ids.shape == (1, 2)


@pytest.mark.unit
def test_apply_router_gating_in_fp32_casts_input_to_float32(layer_module, monkeypatch):
    """When router_gating_in_fp32=True, gate input should be cast to float32."""
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    _stub_apply_prefetch_attrs(method, module, monkeypatch)
    method.select_communication_strategy = MagicMock(return_value=("agrs", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 2.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 3.0))
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int32),
            )
        ),
    )

    # Enable router_gating_in_fp32
    monkeypatch.setattr(
        module.model_extra_config.operator_opt_config,
        "router_gating_in_fp32",
        True,
    )

    gate_calls = []
    def _gate(x):
        gate_calls.append(x)
        return torch.zeros(x.shape[0], 2, dtype=torch.float32), None

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=_gate,
        shared_experts=None,
    )
    hidden_states = torch.ones(2, 4, dtype=torch.float16)
    method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=None,
        top_k=1,
        renormalize=False,
    )

    assert len(gate_calls) == 1
    assert gate_calls[0].dtype == torch.float32, (
        f"Expected gate input dtype float32, got {gate_calls[0].dtype}"
    )


@pytest.mark.unit
def test_apply_router_gating_not_fp32_keeps_input_dtype(layer_module, monkeypatch):
    """When router_gating_in_fp32=False, gate input should keep its original dtype."""
    module, _, context_holder = layer_module
    context_holder.attn_metadata = {}
    method = module.NPUUnquantizedFusedMoEMethod.__new__(module.NPUUnquantizedFusedMoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    _stub_apply_prefetch_attrs(method, module, monkeypatch)
    method.select_communication_strategy = MagicMock(return_value=("agrs", _DummyStrategy()))
    method.apply_prepare_permute = MagicMock(
        return_value=module.PreparePermuteResult(
            hidden_states_sorted_by_experts=torch.ones(2, 4),
            expert_tokens=torch.tensor([2], dtype=torch.int64),
            dynamic_scale=None,
        )
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 2.0))
    method.apply_unpermute_finalize = MagicMock(return_value=torch.full((2, 4), 3.0))
    monkeypatch.setattr(
        module.NPUFusedMoE,
        "select_experts",
        MagicMock(
            return_value=(
                torch.ones(2, 1, dtype=torch.float32),
                torch.zeros(2, 1, dtype=torch.int32),
            )
        ),
    )

    # Disable router_gating_in_fp32
    monkeypatch.setattr(
        module.model_extra_config.operator_opt_config,
        "router_gating_in_fp32",
        False,
    )

    gate_calls = []
    def _gate(x):
        gate_calls.append(x)
        return torch.zeros(x.shape[0], 2, dtype=torch.float32), None

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        gate=_gate,
        shared_experts=None,
    )
    hidden_states = torch.ones(2, 4, dtype=torch.float16)
    method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=None,
        top_k=1,
        renormalize=False,
    )

    assert len(gate_calls) == 1
    assert gate_calls[0].dtype == torch.float16, (
        f"Expected gate input dtype float16, got {gate_calls[0].dtype}"
    )