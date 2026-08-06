# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _make_package(monkeypatch: pytest.MonkeyPatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _make_module(monkeypatch: pytest.MonkeyPatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class DummyCompressedTensorsW8A8Int8MoEMethod:
    def __init__(self, weight_quant, parent, moe_config):
        self.weight_quant = weight_quant
        self.moe = parent
        self.moe_config = moe_config
        self.static_input_scales = False


class DummyNPUFusedMoEMethodBase:
    def __init__(self):
        self.communication_strategy_selector = None

    def select_communication_strategy(self, num_tokens: int):
        return self.communication_strategy_selector.select_communication_strategy(
            num_tokens
        )

    def apply_prepare_permute(self, strategy_impl, layer, x, topk_ids, options=None):
        return strategy_impl.prepare_permute(layer, x, topk_ids, options)

    def apply_unpermute_finalize(
        self, strategy_impl, layer, hidden_states, topk_ids, topk_weights, result,
        finalize_params=None, finalize_metadata=None,
    ):
        return strategy_impl.unpermute_finalize(
            layer,
            hidden_states,
            topk_ids,
            topk_weights,
            result,
            finalize_params=finalize_params,
            finalize_metadata=finalize_metadata,
        )


class DummyPreparePermuteResult:
    def __init__(
        self,
        hidden_states_sorted_by_experts,
        expert_tokens,
        dynamic_scale,
        avg_tokens_per_expert=None,
        row_idx_type=0,
    ):
        self.hidden_states_sorted_by_experts = hidden_states_sorted_by_experts
        self.expert_tokens = expert_tokens
        self.dynamic_scale = dynamic_scale
        self.avg_tokens_per_expert = avg_tokens_per_expert
        self.row_idx_type = row_idx_type


class DummyFinalizeParams:
    """Dummy finalize params for testing multi-stream path."""
    pass


class DummyStrategyImpl:
    def __init__(self, prepare_result, final_output=None, finalize_params=None):
        self._prepare_result = prepare_result
        self._final_output = final_output
        self._finalize_params = finalize_params
        self.received_finalize_metadata = None
        self.received_finalize_params = None

    def prepare_permute(self, layer, x, topk_ids, options=None):
        return self._prepare_result

    def prepare_finalize_params(self, layer, topk_ids, result):
        return self._finalize_params

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
        self.received_finalize_params = finalize_params
        self.received_finalize_metadata = finalize_metadata
        if self._final_output is None:
            return hidden_states
        return self._final_output


class DummyFinalizeMetadataStrategy(DummyStrategyImpl):
    def __init__(self, prepare_result, final_output=None, finalize_metadata=None):
        super().__init__(prepare_result, final_output=final_output)
        self.finalize_metadata = finalize_metadata
        self.prepare_finalize_metadata_args = None

    def prepare_finalize_metadata(self, layer, topk_weights, result):
        self.prepare_finalize_metadata_args = (layer, topk_weights, result)
        return self.finalize_metadata


class DummyNPUFusedMoE:
    @staticmethod
    def select_experts(
        router_logits,
        top_k,
        use_grouped_topk=False,
        renormalize=False,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
    ):
        num_tokens = router_logits.shape[0]
        topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32)
        topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int64)
        return topk_weights, topk_ids


class DummyNPUStream:
    def wait_stream(self, stream):
        return None


class DummyStreamContext:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class MockMoELayer(torch.nn.Module):
    def __init__(self, use_ep=True, num_experts=2):
        super().__init__()
        self.layer_name = "test_layer"
        self.moe_config = SimpleNamespace(num_experts=num_experts)
        self.local_num_experts = num_experts
        self.enable_eplb = False
        self.moe_parallel_config = SimpleNamespace(use_ep=use_ep)
        self.quant_config = object()
        self.gate = None
        self.shared_experts = None


def _operator_opt_config(**overrides):
    values = {
        "moe_multi_stream_tune": False,
        "enable_prefetch": False,
        "enable_agrs_finalize_metadata_overlap": False,
        "enable_round_pipeline_comm": False,
        "shared_expert_multi_stream": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", SimpleNamespace(), raising=False)
    monkeypatch.setattr(torch.npu, "config", SimpleNamespace(allow_internal_format=False), raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: DummyNPUStream(), raising=False)
    monkeypatch.setattr(torch.npu, "stream", lambda s: DummyStreamContext(s), raising=False)

    torch_npu_module = _make_module(monkeypatch, "torch_npu")
    torch_npu_module.npu_format_cast = lambda t, fmt: t
    torch_npu_module.npu_grouped_matmul = (
        lambda inputs, weights, **kwargs: [inputs[0].to(kwargs.get("output_dtype", inputs[0].dtype))]
    )
    torch_npu_module.npu_swiglu = lambda x: x
    torch_npu_module.npu_dequant_swiglu_quant = (
        lambda **kwargs: (kwargs["x"].to(torch.int8), torch.ones(kwargs["x"].shape[0]))
    )
    torch_npu_module.npu = SimpleNamespace(
        current_stream=lambda: DummyNPUStream(),
        get_device_name=lambda _: "Ascend910B",
    )

    _make_package(monkeypatch, "vllm")
    logger_module = _make_module(monkeypatch, "vllm.logger")
    logger_module.init_logger = lambda _: SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )

    distributed_module = _make_module(monkeypatch, "vllm.distributed")
    distributed_module.tensor_model_parallel_all_gather = lambda x, dim=0: x
    distributed_module.tensor_model_parallel_all_reduce = lambda x: x
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    distributed_module.get_tensor_model_parallel_rank = lambda: 0
    distributed_module.get_world_group = lambda: SimpleNamespace(
        rank_in_group=0, world_size=1
    )
    distributed_module.get_ep_group = lambda: SimpleNamespace(
        world_size=1, rank_in_group=0, device_group=None
    )

    platforms_module = _make_module(monkeypatch, "vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")

    utils_module = _make_module(monkeypatch, "vllm.model_executor.utils")
    utils_module.set_weight_attrs = (
        lambda param, attrs: [setattr(param, k, v) for k, v in attrs.items()]
    )

    ct_moe_module = _make_module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe",
    )
    ct_moe_module.CompressedTensorsW8A8Int8MoEMethod = (
        DummyCompressedTensorsW8A8Int8MoEMethod
    )

    fused_moe_layer_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.layer"
    )

    class FusedMoeWeightScaleSupported:
        CHANNEL = SimpleNamespace(value="channel")

    fused_moe_layer_module.FusedMoeWeightScaleSupported = FusedMoeWeightScaleSupported

    vllm_config_module = _make_module(monkeypatch, "vllm.config")
    vllm_config_module.get_current_vllm_config = lambda: SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(first_k_dense_replace=0))
    )

    vllm_config_module.ParallelConfig = SimpleNamespace  # Mock class

    # Mock the fused_moe config modules to avoid deep imports
    fused_moe_config_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.config"
    )
    fused_moe_config_module.FusedMoEQuantConfig = SimpleNamespace
    fused_moe_config_module.FusedMoEQuantDesc = SimpleNamespace
    fused_moe_config_module._quant_flags_to_group_shape = lambda *a, **kw: None

    # Mock the fused_moe __init__ to prevent auto-imports
    fused_moe_init_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe"
    )
    fused_moe_init_module.FusedMoE = SimpleNamespace

    npu_layer_module = _make_module(monkeypatch, "omni.layers.fused_moe.layer")
    npu_layer_module.NPUFusedMoE = DummyNPUFusedMoE

    npu_fused_moe_module = _make_module(
        monkeypatch, "omni.layers.fused_moe.fused_moe"
    )
    npu_fused_moe_module.fused_experts_tp = MagicMock(
        return_value=torch.tensor([2.0], dtype=torch.float32)
    )

    npu_base_module = _make_module(
        monkeypatch, "omni.layers.fused_moe.fused_moe_method_base"
    )
    npu_base_module.NPUFusedMoEMethodBase = DummyNPUFusedMoEMethodBase

    prepare_module = _make_module(
        monkeypatch, "omni.layers.fused_moe.prepare_permute_unpermute_finalize"
    )
    prepare_module.PreparePermuteOptions = SimpleNamespace
    prepare_module.PreparePermuteResult = DummyPreparePermuteResult
    prepare_module.AGRSPreparePermuteResult = SimpleNamespace

    # Mock omni.layers.fused_moe.config
    npu_fused_moe_config_module = _make_module(
        monkeypatch, "omni.layers.fused_moe.config"
    )
    npu_fused_moe_config_module.int4_w4a8_moe_quant_config = lambda **kw: SimpleNamespace(**kw)

    utils_layer_module = _make_module(monkeypatch, "omni.layers.utils")
    utils_layer_module.named_stream = lambda name: DummyNPUStream()

    # Mock model_extra_config for v1 models config loader
    model_extra_config = SimpleNamespace(
        parall_config=SimpleNamespace(ena_seq_parallel=False),
        operator_opt_config=_operator_opt_config(),
    )
    config_loader_module = _make_module(
        monkeypatch, "omni.model_config.config_loader.loader"
    )
    config_loader_module.model_extra_config = model_extra_config

    base_path = Path(__file__).resolve().parents[4]
    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(base_path / "omni")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    layers_pkg = types.ModuleType("omni.layers")
    layers_pkg.__path__ = [str(base_path / "omni" / "layers")]
    monkeypatch.setitem(sys.modules, "omni.layers", layers_pkg)
    quant_pkg = types.ModuleType("omni.layers.quantization")
    quant_pkg.__path__ = [str(base_path / "omni" / "layers" / "quantization")]
    monkeypatch.setitem(sys.modules, "omni.layers.quantization", quant_pkg)
    ct_pkg = types.ModuleType("omni.layers.quantization.compressed_tensors")
    ct_pkg.__path__ = [
        str(base_path / "omni" / "layers" / "quantization" / "compressed_tensors")
    ]
    monkeypatch.setitem(
        sys.modules, "omni.layers.quantization.compressed_tensors", ct_pkg
    )

    # Mock omni.v1 chain to avoid pulling in vllm/transformers
    v1_pkg = types.ModuleType("omni.v1")
    v1_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "omni.v1", v1_pkg)
    v1_distributed_pkg = _make_module(monkeypatch, "omni.v1.distributed")
    parallel_state_ext_module = _make_module(
        monkeypatch, "omni.v1.distributed.parallel_state_ext"
    )
    parallel_state_ext_module.get_npu_device_count = lambda: 1

    sys.modules.pop(
        "omni.layers.quantization.compressed_tensors.compressed_tensors_moe", None
    )
    yield


@pytest.fixture
def compressed_moe_module(mock_dependencies):
    module = importlib.import_module(
        "omni.layers.quantization.compressed_tensors.compressed_tensors_moe"
    )
    importlib.reload(module)
    return module


def _make_method(module, layer, has_bias=False):
    weight_quant = MagicMock()
    parent = SimpleNamespace(moe_parallel_config=layer.moe_parallel_config, has_bias=has_bias)
    return module.NPUCompressedTensorsW8A8Int8MoEMethod(weight_quant, parent, layer)


def _make_prepare_result(dynamic_scale=None, row_idx_type=0):
    return DummyPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.ones(3, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([1, 2], dtype=torch.int64),
        dynamic_scale=dynamic_scale,
        avg_tokens_per_expert=[1, 2],
        row_idx_type=row_idx_type,
    )


def test_create_weights_registers_expected_params(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer, has_bias=False)

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )

    assert layer.w13_weight.dtype == torch.int8
    assert layer.w13_weight.shape == (2, 6, 4)
    assert layer.w2_weight.shape == (2, 4, 3)
    assert layer.w13_weight_scale.shape == (2, 6, 1)
    assert layer.w2_weight_scale.shape == (2, 4, 1)
    assert layer.w13_weight_scale.quant_method == "channel"
    assert layer.w2_weight_scale.quant_method == "channel"
    assert layer.w13_input_scale is None
    assert layer.w2_input_scale is None
    assert not hasattr(layer, "w13_bias")
    assert not hasattr(layer, "w2_bias")


def test_create_weights_registers_bias_when_enabled(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer, has_bias=True)

    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )

    assert layer.w13_bias.shape == (2, 6)
    assert layer.w2_bias.shape == (2, 4)


def test_process_weights_after_loading_transpose_and_cast(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer, has_bias=False)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (2, 4, 6)
    assert layer.w2_weight.shape == (2, 3, 4)
    assert layer.w13_weight_scale.dtype == torch.float32
    assert layer.w13_weight_scale.shape == (2, 6)
    assert layer.w2_weight_scale.dtype == torch.bfloat16
    assert layer.w2_weight_scale.shape == (2, 4)


def test_apply_experts_quant_requires_dynamic_scale(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )
    method.process_weights_after_loading(layer)

    with pytest.raises(ValueError, match="dynamic per-token scale"):
        method.apply_experts(layer, _make_prepare_result(dynamic_scale=None))


def test_apply_experts_swigluoai_passes_kernel_kwargs(
    compressed_moe_module, monkeypatch
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    layer.swiglu_limit = 8.0
    layer.glu_alpha = 1.7
    layer.glu_bias = 1.0
    method = _make_method(compressed_moe_module, layer, has_bias=True)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )
    method.process_weights_after_loading(layer)

    grouped_calls = []

    def _fake_grouped_matmul(inputs, weights, **kwargs):
        grouped_calls.append(kwargs)
        if kwargs.get("output_dtype") == torch.int32:
            return [torch.ones(inputs[0].shape[0], 6, dtype=torch.int32)]
        return [torch.ones(inputs[0].shape[0], 4, dtype=torch.bfloat16)]

    dequant_mock = MagicMock(
        return_value=(
            torch.ones(3, 3, dtype=torch.int8),
            torch.ones(3, dtype=torch.float32),
        )
    )
    monkeypatch.setattr(
        compressed_moe_module.torch_npu, "npu_grouped_matmul", _fake_grouped_matmul
    )
    monkeypatch.setattr(
        compressed_moe_module.torch_npu, "npu_dequant_swiglu_quant", dequant_mock
    )

    prepare_result = _make_prepare_result(dynamic_scale=torch.ones(1, 3))
    output = method.apply_experts(layer, prepare_result, activation="swigluoai")

    assert output.dtype == torch.bfloat16
    kwargs = dequant_mock.call_args.kwargs
    assert kwargs["bias"] is not None
    assert kwargs["swiglu_mode"] == 1
    assert kwargs["clamp_limit"] == 8.0
    assert kwargs["glu_alpha"] == 1.7
    assert kwargs["glu_bias"] == 1.0
    assert grouped_calls[1]["output_dtype"] == torch.bfloat16


def test_apply_experts_grouped_finalize_returns_intermediate_and_scale(
    compressed_moe_module, monkeypatch
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )
    method.process_weights_after_loading(layer)

    grouped_calls = []

    def _fake_grouped_matmul(inputs, weights, **kwargs):
        grouped_calls.append(kwargs)
        return [torch.ones(inputs[0].shape[0], 6, dtype=torch.int32)]

    intermediate_h = torch.ones(3, 3, dtype=torch.int8)
    pertoken_scale = torch.ones(3, dtype=torch.float32)
    dequant_mock = MagicMock(return_value=(intermediate_h, pertoken_scale))
    monkeypatch.setattr(
        compressed_moe_module.torch_npu, "npu_grouped_matmul", _fake_grouped_matmul
    )
    monkeypatch.setattr(
        compressed_moe_module.torch_npu, "npu_dequant_swiglu_quant", dequant_mock
    )

    output = method.apply_experts(
        layer,
        _make_prepare_result(dynamic_scale=torch.ones(1, 3)),
        use_grouped_matmul_finalize_routing=True,
    )

    assert output[0] is intermediate_h
    assert output[1] is pertoken_scale
    assert len(grouped_calls) == 1

def test_apply_with_ep_returns_tuple_when_shared_experts_enabled(
    compressed_moe_module, monkeypatch
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    layer.shared_experts = lambda x: torch.full_like(x, 2.0)
    layer.shared_experts.gate_up_proj = SimpleNamespace(
        tp_size=1, weight=torch.zeros(1, dtype=torch.float32)
    )
    layer.shared_experts.down_proj = SimpleNamespace(
        tp_size=1, weight=torch.zeros(1, dtype=torch.float32)
    )

    method = _make_method(compressed_moe_module, layer)
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 3.0))

    monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
    hidden_states = torch.randn(3, 4, dtype=torch.bfloat16)
    router_logits = torch.randn(3, 2, dtype=torch.float32)
    output = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=1,
        renormalize=False,
    )
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    assert isinstance(output, tuple)
    shared_output, routed_output = output
    assert torch.equal(shared_output, torch.full((3, 4), 2.0))
    assert torch.equal(routed_output, torch.full((3, 4), 5.0))


def test_apply_passes_grouped_finalize_flag_for_agrs_decode(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(
            _make_prepare_result(dynamic_scale=torch.ones(3), row_idx_type=1),
            final_output=torch.full((3, 4), 4.0),
        ),
    )
    method.apply_experts = MagicMock(
        return_value=(torch.ones(3, 3, dtype=torch.int8), torch.ones(3))
    )

    output = method.apply(
        layer=layer,
        hidden_states=torch.randn(3, 4, dtype=torch.bfloat16),
        router_logits=torch.randn(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    assert method.apply_experts.call_args.kwargs["use_grouped_matmul_finalize_routing"] is True
    assert torch.equal(output, torch.full((3, 4), 4.0))

def test_apply_shared_experts_tp_gt_1_uses_full_hidden_states(
    compressed_moe_module, monkeypatch
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.tp_size = 2
    method.tp_rank = 1
    method.select_communication_strategy = lambda n: (
        "all2all",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((2, 4), 3.0))
    monkeypatch.setattr(
        compressed_moe_module,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x, x], dim=dim),
    )
    all_reduce_mock = MagicMock(return_value=torch.full((3, 4), 9.0))
    monkeypatch.setattr(
        compressed_moe_module, "tensor_model_parallel_all_reduce", all_reduce_mock
    )

    shared = MagicMock(return_value=torch.full((3, 4), 6.0))
    shared.gate_up_proj = SimpleNamespace(
        tp_size=2, weight=torch.zeros(1, dtype=torch.float32)
    )
    shared.down_proj = SimpleNamespace(
        tp_size=2, weight=torch.zeros(1, dtype=torch.float32)
    )
    layer.shared_experts = shared

    hidden_states = torch.randn(3, 4, dtype=torch.bfloat16)
    router_logits = torch.randn(3, 2, dtype=torch.float32)
    output = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=1,
        renormalize=False,
    )

    assert isinstance(output, tuple)
    shared_output, routed_output = output
    assert shared.call_count == 1
    assert torch.equal(shared.call_args.args[0], hidden_states)
    all_reduce_mock.assert_called_once()
    assert torch.equal(all_reduce_mock.call_args.args[0], torch.full((3, 4), 6.0))
    assert torch.equal(shared_output, torch.full((3, 4), 9.0))
    assert routed_output.shape == (3, 4)


def test_apply_enable_eplb_calls_planner(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.enable_eplb = True
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 1.0))
    method.planner = MagicMock(
        plan=MagicMock(
            return_value=(
                None,
                torch.zeros(3, 1, dtype=torch.int64),
                None,
            )
        )
    )
    method.moe_layer_idx = 1
    method.expert_mapping = {"x": 1}

    output = method.apply(
        layer=layer,
        hidden_states=torch.randn(3, 4, dtype=torch.bfloat16),
        router_logits=torch.randn(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
        enable_eplb=True,
    )

    assert method.planner.plan.call_count == 1
    assert layer.planner is method.planner
    assert layer.moe_layer_idx == 1
    assert output.shape == (3, 4)


def test_supports_eplb_property(compressed_moe_module):
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer)
    assert method.supports_eplb is True

def test_init_enable_best_ep_default_false(compressed_moe_module, monkeypatch):
    """Test that enable_best_ep defaults to False when BEST_EP env var is not set."""
    monkeypatch.delenv("BEST_EP", raising=False)
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer)
    assert method.enable_best_ep is False


def test_init_enable_best_ep_from_env_true(compressed_moe_module, monkeypatch):
    """Test that enable_best_ep is True when BEST_EP='true' (case-insensitive)."""
    monkeypatch.setenv("BEST_EP", "true")
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer)
    assert method.enable_best_ep is True


def test_init_enable_best_ep_from_env_true_uppercase(compressed_moe_module, monkeypatch):
    """Test that enable_best_ep is True when BEST_EP='TRUE' (case-insensitive)."""
    monkeypatch.setenv("BEST_EP", "TRUE")
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer)
    assert method.enable_best_ep is True


def test_init_enable_best_ep_from_env_false(compressed_moe_module, monkeypatch):
    """Test that enable_best_ep is False when BEST_EP is set to non-'true' value."""
    monkeypatch.setenv("BEST_EP", "false")
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer)
    assert method.enable_best_ep is False


def test_init_enable_best_ep_from_env_invalid(compressed_moe_module, monkeypatch):
    """Test that enable_best_ep is False when BEST_EP has invalid value."""
    monkeypatch.setenv("BEST_EP", "invalid_value")
    layer = MockMoELayer(use_ep=True)
    method = _make_method(compressed_moe_module, layer)
    assert method.enable_best_ep is False

def test_apply_enable_eplb_with_best_ep_calls_apply_best_load_balance(
    compressed_moe_module, monkeypatch
):
    """Test that when enable_eplb=True and enable_best_ep=True,
    apply_best_load_balance is called with correct arguments."""
    # Setup model_extra_config mock - directly patch the module
    model_extra_config = SimpleNamespace(
        parall_config=SimpleNamespace(ena_seq_parallel=False),
        operator_opt_config=_operator_opt_config(),
    )
    monkeypatch.setattr(compressed_moe_module, "model_extra_config", model_extra_config)

    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.enable_best_ep = True  # Set the flag
    method.enable_eplb = True
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 1.0))

    # Mock planner with apply_best_load_balance
    apply_best_mock = MagicMock(return_value=torch.zeros(3, 1, dtype=torch.int64))
    method.planner = MagicMock(apply_best_load_balance=apply_best_mock)
    method.moe_layer_idx = 1
    method.expert_mapping = {"x": 1}

    hidden_states = torch.randn(3, 4, dtype=torch.bfloat16)
    router_logits = torch.randn(3, 2, dtype=torch.float32)

    output = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=1,
        renormalize=False,
        enable_eplb=True,
        global_num_experts=8,
    )

    # Verify apply_best_load_balance was called
    assert apply_best_mock.call_count == 1
    call_kwargs = apply_best_mock.call_args.kwargs
    assert call_kwargs["topk_ids"].shape == (3, 1)
    assert call_kwargs["global_num_experts"] == 8
    assert call_kwargs["moe_multi_stream_tune"] is False
    assert output.shape == (3, 4)


def test_apply_enable_eplb_without_best_ep_calls_plan_path(
    compressed_moe_module, monkeypatch
):
    """Test that when enable_eplb=True but enable_best_ep=False,
    the planner.plan path is used instead of apply_best_load_balance."""
    # Setup model_extra_config mock - directly patch the module
    model_extra_config = SimpleNamespace(
        parall_config=SimpleNamespace(ena_seq_parallel=False),
        operator_opt_config=_operator_opt_config(moe_multi_stream_tune=True),
    )
    monkeypatch.setattr(compressed_moe_module, "model_extra_config", model_extra_config)

    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.enable_best_ep = False  # Explicitly set to False
    method.enable_eplb = True
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 1.0))

    # Mock planner with plan method (not apply_best_load_balance)
    plan_mock = MagicMock(
        return_value=(
            None,
            torch.zeros(3, 1, dtype=torch.int64),
            None,
        )
    )
    method.planner = MagicMock(plan=plan_mock)
    method.planner.record_activation = MagicMock()
    method.moe_layer_idx = 2
    method.expert_mapping = {"y": 2}

    hidden_states = torch.randn(3, 4, dtype=torch.bfloat16)
    router_logits = torch.randn(3, 2, dtype=torch.float32)

    output = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=2,
        renormalize=False,
        enable_eplb=True,
    )

    # Verify plan was called (not apply_best_load_balance)
    assert plan_mock.call_count == 1
    assert layer.planner is method.planner
    assert layer.moe_layer_idx == 2
    assert output.shape == (3, 4)


def test_apply_eplb_best_ep_respects_global_num_experts(
    compressed_moe_module, monkeypatch
):
    """Test that apply_best_load_balance receives global_num_experts parameter correctly."""
    model_extra_config = SimpleNamespace(
        parall_config=SimpleNamespace(ena_seq_parallel=False),
        operator_opt_config=_operator_opt_config(moe_multi_stream_tune=True),
    )
    monkeypatch.setattr(compressed_moe_module, "model_extra_config", model_extra_config)

    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.enable_best_ep = True
    method.enable_eplb = True
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 1.0))

    apply_best_mock = MagicMock(return_value=torch.zeros(3, 1, dtype=torch.int64))
    method.planner = MagicMock(apply_best_load_balance=apply_best_mock)
    method.moe_layer_idx = 1
    method.expert_mapping = {"x": 1}

    # Test with different global_num_experts values
    for global_experts in [4, 8, 16]:
        apply_best_mock.reset_mock()
        method.apply(
            layer=layer,
            hidden_states=torch.randn(3, 4, dtype=torch.bfloat16),
            router_logits=torch.randn(3, 2, dtype=torch.float32),
            top_k=1,
            renormalize=False,
            enable_eplb=True,
            global_num_experts=global_experts,
        )
        assert apply_best_mock.call_args.kwargs["global_num_experts"] == global_experts


def test_apply_eplb_best_ep_respects_moe_multi_stream_tune(
    compressed_moe_module, monkeypatch
):
    """Test that apply_best_load_balance receives moe_multi_stream_tune from config."""
    # Test with moe_multi_stream_tune=True - modify the already imported module
    model_extra_config = SimpleNamespace(
        parall_config=SimpleNamespace(ena_seq_parallel=False),
        operator_opt_config=_operator_opt_config(moe_multi_stream_tune=True),
    )
    # Directly patch the model_extra_config in the compressed_moe_module
    monkeypatch.setattr(compressed_moe_module, "model_extra_config", model_extra_config)

    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    method = _make_method(compressed_moe_module, layer)
    method.enable_best_ep = True
    method.enable_eplb = True
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(_make_prepare_result(dynamic_scale=torch.ones(3))),
    )
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 1.0))

    apply_best_mock = MagicMock(return_value=torch.zeros(3, 1, dtype=torch.int64))
    method.planner = MagicMock(apply_best_load_balance=apply_best_mock)
    method.moe_layer_idx = 1
    method.expert_mapping = {"x": 1}

    method.apply(
        layer=layer,
        hidden_states=torch.randn(3, 4, dtype=torch.bfloat16),
        router_logits=torch.randn(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
        enable_eplb=True,
        global_num_experts=8,
    )

    assert apply_best_mock.call_args.kwargs["moe_multi_stream_tune"] is True


def test_init_eplb_sets_redundant_experts(compressed_moe_module, monkeypatch):
    planner_kwargs = {}

    class DummyPlanner:
        def __init__(self, **kwargs):
            planner_kwargs.update(kwargs)

        @staticmethod
        def get_deepseek_v3_moe_layer_idx(prefix, first_k_dense_replace):
            return 3

        def expert_mapping_on_current_layer(self, moe_layer_idx):
            return {"idx": moe_layer_idx}

        def get_num_of_redundant_experts(
            self, moe_layer_idx, num_expert_per_device_origin, rank_device
        ):
            return 1

    omni_pkg = types.ModuleType("omni_placement")
    omni_planner_module = types.ModuleType("omni_placement.omni_planner")
    omni_planner_module.OmniPlanner = DummyPlanner
    monkeypatch.setitem(sys.modules, "omni_placement", omni_pkg)
    monkeypatch.setitem(sys.modules, "omni_placement.omni_planner", omni_planner_module)
    monkeypatch.setattr(
        compressed_moe_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    num_hidden_layers=12,
                    first_k_dense_replace=2,
                )
            )
        ),
    )

    layer = MockMoELayer(use_ep=True, num_experts=2)
    layer.enable_eplb = True
    method = _make_method(compressed_moe_module, layer)
    method.create_weights(
        layer=layer,
        num_experts=layer.local_num_experts,
        hidden_size=4,
        intermediate_size_per_partition=3,
        params_dtype=torch.float16,
        weight_loader="mock_loader",
    )

    assert method.num_of_redundant_experts == 1
    assert layer.w13_weight.shape[0] == 3
    assert planner_kwargs["num_layers"] == 12
    assert planner_kwargs["first_k_dense_replace"] == 2


def _npu_to_cpu(monkeypatch):
    """Redirect torch.ones/torch.zeros device='npu' to 'cpu' for CI without NPU."""
    _orig_ones = torch.ones
    _orig_zeros = torch.zeros

    def _ones(*args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return _orig_ones(*args, **kwargs)

    def _zeros(*args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return _orig_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "ones", _ones)
    monkeypatch.setattr(torch, "zeros", _zeros)


@pytest.mark.unit
def test_w4a8_apply_experts_calls_grouped_matmul_and_swiglu(compressed_moe_module, monkeypatch):
    """Test that W4A8 apply_experts calls gate_up matmul, dequant_swiglu_quant, and down matmul."""
    _npu_to_cpu(monkeypatch)
    module = compressed_moe_module
    torch_npu = sys.modules["torch_npu"]

    gate_up_out = torch.randn(3, 8)
    swiglu_out = torch.randn(3, 4)
    swiglu_scale = torch.ones(3)
    down_out = torch.randn(3, 4)

    torch_npu.npu_grouped_matmul = MagicMock(side_effect=[[gate_up_out], [down_out]])
    torch_npu.npu_dequant_swiglu_quant = MagicMock(return_value=(swiglu_out, swiglu_scale))

    W4A8 = module.NPUCompressedTensorsW4A8Int4MoEMethod
    method = W4A8.__new__(W4A8)

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        w13_weight=torch.randn(2, 4, 4),
        w13_weight_bias=torch.zeros(2, 8),
        w13_weight_int4_scale=torch.ones(2, 1, 8, dtype=torch.int64),
        w2_weight=torch.randn(2, 4, 4),
        w2_weight_bias=torch.zeros(2, 4),
        w2_weight_int4_scale=torch.ones(2, 1, 4, dtype=torch.int64),
    )

    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.randn(3, 4),
        expert_tokens=torch.tensor([2, 1], dtype=torch.int64),
        dynamic_scale=torch.ones(3),
    )

    result = method.apply_experts(layer, prepare_result)

    assert torch_npu.npu_grouped_matmul.call_count == 2
    assert torch_npu.npu_dequant_swiglu_quant.call_count == 1
    assert result is down_out


@pytest.mark.unit
def test_w4a8_apply_experts_finalize_routing_returns_intermediate(compressed_moe_module, monkeypatch):
    """W4A8 apply_experts returns (intermediate_h, pertoken_scale) when use_grouped_matmul_finalize_routing=True, skipping w2 matmul."""
    _npu_to_cpu(monkeypatch)
    module = compressed_moe_module
    torch_npu = sys.modules["torch_npu"]

    gate_up_out = torch.randn(3, 8)
    swiglu_out = torch.randn(3, 4)
    swiglu_scale = torch.ones(3)

    torch_npu.npu_grouped_matmul = MagicMock(return_value=[gate_up_out])
    torch_npu.npu_dequant_swiglu_quant = MagicMock(return_value=(swiglu_out, swiglu_scale))

    W4A8 = module.NPUCompressedTensorsW4A8Int4MoEMethod
    method = W4A8.__new__(W4A8)

    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True),
        w13_weight=torch.randn(2, 4, 4),
        w13_weight_bias=torch.zeros(2, 8),
        w13_weight_int4_scale=torch.ones(2, 1, 8, dtype=torch.int64),
    )

    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.randn(3, 4),
        expert_tokens=torch.tensor([2, 1], dtype=torch.int64),
        dynamic_scale=torch.ones(3),
    )

    result = method.apply_experts(layer, prepare_result, use_grouped_matmul_finalize_routing=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is swiglu_out
    assert result[1] is swiglu_scale
    # w2 matmul should NOT be called (only gate_up matmul)
    assert torch_npu.npu_grouped_matmul.call_count == 1


def test_apply_multistream_finalize_with_moe_multi_stream_tune(compressed_moe_module, monkeypatch):
    """When moe_multi_stream_tune=True and agrs+decode, multi-stream finalize path is used."""
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    finalize_params = DummyFinalizeParams()
    method = _make_method(compressed_moe_module, layer)
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(
            _make_prepare_result(dynamic_scale=torch.ones(3), row_idx_type=1),
            final_output=torch.full((3, 4), 4.0),
            finalize_params=finalize_params,
        ),
    )
    method.apply_experts = MagicMock(
        return_value=(torch.ones(3, 3, dtype=torch.int8), torch.ones(3))
    )

    # Enable moe_multi_stream_tune
    compressed_moe_module.model_extra_config.operator_opt_config.moe_multi_stream_tune = True

    output = method.apply(
        layer=MockMoELayer(use_ep=True),
        hidden_states=torch.randn(3, 4, dtype=torch.bfloat16),
        router_logits=torch.randn(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    # Verify multi-stream path was taken
    assert method.apply_experts.call_count == 1
    assert torch.equal(output, torch.full((3, 4), 4.0))

    # Reset
    compressed_moe_module.model_extra_config.operator_opt_config.moe_multi_stream_tune = False


def test_apply_agrs_finalize_metadata_uses_method_prepare_and_overlaps_experts(
    compressed_moe_module,
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    prepare_result = _make_prepare_result(dynamic_scale=torch.ones(3), row_idx_type=1)
    metadata = object()
    strategy = DummyStrategyImpl(
        prepare_result,
        final_output=torch.full((3, 4), 4.0),
    )
    method = _make_method(compressed_moe_module, layer)
    method.enable_agrs_finalize_metadata_overlap = True
    method.prepare_finalize_metadata = MagicMock(return_value=metadata)
    method.select_communication_strategy = lambda n: ("agrs", strategy)
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 3.0))

    output = method.apply(
        layer=layer,
        hidden_states=torch.randn(3, 4, dtype=torch.bfloat16),
        router_logits=torch.randn(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    method.prepare_finalize_metadata.assert_called_once()
    assert method.prepare_finalize_metadata.call_args.args[0] is strategy
    assert method.prepare_finalize_metadata.call_args.args[1] is layer
    assert method.prepare_finalize_metadata.call_args.args[3] is prepare_result
    assert method.agrs_overlap_stream is not None
    assert strategy.received_finalize_metadata is metadata
    assert torch.equal(output, torch.full((3, 4), 4.0))


def test_apply_agrs_finalize_metadata_uses_strategy_prepare_without_overlap_stream(
    compressed_moe_module,
):
    layer = MockMoELayer(use_ep=True)
    layer.quant_config = object()
    prepare_result = _make_prepare_result(dynamic_scale=torch.ones(3), row_idx_type=1)
    metadata = object()
    strategy = DummyFinalizeMetadataStrategy(
        prepare_result,
        final_output=torch.full((3, 4), 5.0),
        finalize_metadata=metadata,
    )
    method = _make_method(compressed_moe_module, layer)
    method.enable_agrs_finalize_metadata_overlap = True
    method.agrs_overlap_stream = None
    method.select_communication_strategy = lambda n: ("agrs", strategy)
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 3.0))

    output = method.apply(
        layer=layer,
        hidden_states=torch.randn(3, 4, dtype=torch.bfloat16),
        router_logits=torch.randn(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    assert strategy.prepare_finalize_metadata_args[0] is layer
    assert strategy.prepare_finalize_metadata_args[2] is prepare_result
    assert strategy.received_finalize_metadata is metadata
    assert torch.equal(output, torch.full((3, 4), 5.0))


def test_apply_multistream_with_shared_experts(compressed_moe_module, monkeypatch):
    """Multi-stream finalize + shared_experts: verify sync before shared_experts."""
    layer = MockMoELayer(use_ep=True)
    finalize_params = DummyFinalizeParams()
    layer.quant_config = object()

    shared = MagicMock(return_value=torch.full((3, 4), 6.0))
    shared.gate_up_proj = SimpleNamespace(tp_size=1)
    layer.shared_experts = shared

    method = _make_method(compressed_moe_module, layer)
    method.select_communication_strategy = lambda n: (
        "agrs",
        DummyStrategyImpl(
            _make_prepare_result(dynamic_scale=torch.ones(3), row_idx_type=1),
            final_output=torch.full((3, 4), 4.0),
            finalize_params=finalize_params,
        ),
    )
    method.apply_experts = MagicMock(return_value=torch.full((3, 4), 3.0))

    compressed_moe_module.model_extra_config.operator_opt_config.moe_multi_stream_tune = True

    hidden_states = torch.randn(3, 4, dtype=torch.bfloat16)
    output = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=torch.randn(3, 2, dtype=torch.float32),
        top_k=1,
        renormalize=False,
    )

    assert shared.call_count == 1
    assert isinstance(output, tuple)

    compressed_moe_module.model_extra_config.operator_opt_config.moe_multi_stream_tune = False
