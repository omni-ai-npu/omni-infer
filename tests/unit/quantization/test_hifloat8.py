# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


pytestmark = pytest.mark.unit


def _make_module(monkeypatch, name, is_package=False):
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def hifloat8_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo_root / "src"))

    torch_npu = _make_module(monkeypatch, "torch_npu")
    torch_npu.hifloat8 = object()

    _make_module(monkeypatch, "vllm", is_package=True)
    config_module = _make_module(monkeypatch, "vllm.config")
    config_module.get_current_vllm_config = lambda: SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace())
    )

    distributed_module = _make_module(monkeypatch, "vllm.distributed")
    distributed_module.get_tensor_model_parallel_rank = lambda: 0
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    distributed_module.tensor_model_parallel_all_gather = lambda x, dim=0: x
    distributed_module.tensor_model_parallel_all_reduce = lambda x: x

    forward_context_module = _make_module(monkeypatch, "vllm.forward_context")
    forward_context_module._fwctx = SimpleNamespace(additional_kwargs={})
    forward_context_module.get_forward_context = lambda: forward_context_module._fwctx
    forward_context_module.is_forward_context_available = lambda: True

    _make_module(monkeypatch, "vllm.utils", is_package=True)
    torch_utils_module = _make_module(monkeypatch, "vllm.utils.torch_utils")
    torch_utils_module.direct_register_custom_op = lambda **_kw: None

    logger_module = _make_module(monkeypatch, "vllm.logger")
    logger_module.init_logger = lambda _name: MagicMock()

    _make_module(monkeypatch, "vllm.model_executor", is_package=True)
    _make_module(monkeypatch, "vllm.model_executor.layers", is_package=True)

    fused_moe_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe", is_package=True
    )

    class FusedMoE:
        pass

    class FusedMoEMethodBase:
        def __init__(self, *_args, **_kwargs):
            pass

    fused_moe_module.FusedMoE = FusedMoE
    fused_moe_module.FusedMoEMethodBase = FusedMoEMethodBase

    fused_moe_layer_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.layer"
    )
    fused_moe_layer_module.FusedMoeWeightScaleSupported = SimpleNamespace(
        CHANNEL=SimpleNamespace(value="channel")
    )

    linear_module = _make_module(monkeypatch, "vllm.model_executor.layers.linear")

    class LinearBase:
        pass

    class LinearMethodBase:
        pass

    class UnquantizedLinearMethod:
        pass

    linear_module.LinearBase = LinearBase
    linear_module.LinearMethodBase = LinearMethodBase
    linear_module.UnquantizedLinearMethod = UnquantizedLinearMethod

    quantization_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.quantization", is_package=True
    )
    quantization_module.QuantizationMethods = str
    quantization_module.register_quantization_config = lambda _name: lambda cls: cls

    quantization_base_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.quantization.base_config"
    )

    class QuantizationConfig:
        def __init__(self, *_args, **_kwargs):
            pass

    class QuantizeMethodBase:
        pass

    quantization_base_module.QuantizationConfig = QuantizationConfig
    quantization_base_module.QuantizeMethodBase = QuantizeMethodBase

    parameter_module = _make_module(monkeypatch, "vllm.model_executor.parameter")
    parameter_module.ChannelQuantScaleParameter = object
    parameter_module.ModelWeightParameter = object

    utils_module = _make_module(monkeypatch, "vllm.model_executor.utils")
    utils_module.set_weight_attrs = (
        lambda param, attrs: [setattr(param, key, value) for key, value in attrs.items()]
    )

    platforms_module = _make_module(monkeypatch, "vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")

    omni_pkg = _make_module(monkeypatch, "omni_npu", is_package=True)
    omni_pkg.__path__ = [str(repo_root / "omni")]
    layers_pkg = _make_module(monkeypatch, "omni_npu.layers", is_package=True)
    layers_pkg.__path__ = [str(repo_root / "omni" / "layers")]
    quant_pkg = _make_module(monkeypatch, "omni_npu.layers.quantization", is_package=True)
    quant_pkg.__path__ = [str(repo_root / "omni" / "layers" / "quantization")]

    fused_moe_pkg = _make_module(monkeypatch, "omni_npu.layers.fused_moe", is_package=True)
    # Point the package at disk so the (lightweight, torch-only) shared_expert
    # submodule imports for real; the other submodules stay stubbed via sys.modules.
    fused_moe_pkg.__path__ = [str(repo_root / "omni" / "layers" / "fused_moe")]
    fused_moe_config_module = _make_module(monkeypatch, "omni_npu.layers.fused_moe.config")
    fused_moe_config_module.hifloat8_moe_quant_config = (
        lambda **kwargs: SimpleNamespace(**kwargs)
    )

    npu_fused_moe_base_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.fused_moe_method_base"
    )

    class NPUFusedMoEMethodBase:
        def __init__(self, *_args, **_kwargs):
            pass

    npu_fused_moe_base_module.NPUFusedMoEMethodBase = NPUFusedMoEMethodBase

    npu_fused_moe_layer_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.layer"
    )

    class NPUFusedMoE:
        pass

    npu_fused_moe_layer_module.NPUFusedMoE = NPUFusedMoE

    from dataclasses import dataclass, field
    from typing import Callable, Optional

    layer_utils_module = _make_module(monkeypatch, "omni_npu.layers.utils")
    layer_utils_module.named_stream = lambda _name: _DummyStream()
    layer_utils_module.CUBE_SIDE_TASKS_KEY = "cube_side_tasks"
    layer_utils_module.CUBE_SIDE_STREAM_NAME = "cube_side_task"

    @dataclass
    class _CubeSideTask:
        fn: Callable[[], None]
        done_event: Optional[object] = None

    layer_utils_module.CubeSideTask = _CubeSideTask

    model_config_pkg = _make_module(
        monkeypatch, "omni_npu.model_config", is_package=True
    )
    config_loader_pkg = _make_module(
        monkeypatch, "omni_npu.model_config.config_loader", is_package=True
    )
    config_loader_module = _make_module(
        monkeypatch, "omni_npu.model_config.config_loader.loader"
    )
    config_loader_module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            shared_expert_multi_stream=True,
            shared_expert_parallel_schedule="with_finalize",
        ),
    )

    _make_module(monkeypatch, "omni_npu.v1", is_package=True)
    _make_module(monkeypatch, "omni_npu.v1.layers", is_package=True)
    v1_linear_module = _make_module(monkeypatch, "omni_npu.v1.layers.linear")

    class FlashCommLinearMethodBase:
        pass

    v1_linear_module.FlashCommLinearMethodBase = FlashCommLinearMethodBase

    # omni_npu.layers.mhc — load the real cube_side_task_ops module from disk
    # so torch.ops.vllm.cube_side_run / cube_side_wait can be bound to its
    # Python implementations for the runner tests.
    mhc_pkg = _make_module(monkeypatch, "omni_npu.layers.mhc", is_package=True)
    mhc_pkg.__path__ = [str(repo_root / "omni" / "layers" / "mhc")]
    cube_side_ops_name = "omni_npu.layers.mhc.cube_side_task_ops"
    monkeypatch.delitem(sys.modules, cube_side_ops_name, raising=False)
    cube_side_ops_module = importlib.import_module(cube_side_ops_name)
    cube_side_ops_module = importlib.reload(cube_side_ops_module)

    # Bind torch.ops.vllm.cube_side_run / cube_side_wait to the Python impls
    # so Hifloat8 quant methods can call them through the torch.ops dispatch.
    torch_ops_vllm = SimpleNamespace(
        cube_side_run=cube_side_ops_module.cube_side_run,
        cube_side_wait=cube_side_ops_module.cube_side_wait,
    )
    monkeypatch.setattr(torch.ops, "vllm", torch_ops_vllm, raising=False)

    module_name = "omni_npu.layers.quantization.hifloat8"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    # Re-expose constants previously imported directly into hifloat8 — kept
    # accessible via the hifloat8 module to minimise test churn.
    module.CUBE_SIDE_TASKS_KEY = cube_side_ops_module.CUBE_SIDE_TASKS_KEY
    module.CUBE_SIDE_PENDING_KEY = cube_side_ops_module.CUBE_SIDE_PENDING_KEY
    module.CubeSideTask = cube_side_ops_module.CubeSideTask
    return module


@pytest.fixture
def mock_torch_npu(monkeypatch, hifloat8_module):
    ns = SimpleNamespace(
        hifloat8=object(),
        npu_trans_quant_param=MagicMock(side_effect=lambda x: x + 1),
        npu_grouped_matmul=MagicMock(),
        npu_swiglu=MagicMock(side_effect=lambda x: x[:, : x.shape[-1] // 2]),
        npu_dtype_cast=MagicMock(side_effect=lambda x, dtype: x.to(torch.int8)),
        npu_quant_matmul=MagicMock(
            side_effect=lambda x1, x2, scale, pertoken_scale, bias, x1_dtype,
            x2_dtype, output_dtype: torch.zeros(
                x1.shape[0], x2.shape[1], dtype=output_dtype
            )
        ),
    )
    monkeypatch.setattr(hifloat8_module, "torch_npu", ns)
    return ns


class _NoopStreamCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyStream:
    def __init__(self):
        self.wait_stream_calls = []
        self.wait_event_calls = []

    def wait_stream(self, other):
        self.wait_stream_calls.append(other)

    def wait_event(self, event):
        self.wait_event_calls.append(event)


def _patch_npu_streams(monkeypatch):
    """Stub torch.npu.{Event,current_stream,stream} for tests touching side streams."""
    main_stream = _DummyStream()

    class _DummyEvent:
        def __init__(self):
            self.recorded = False
            self.waits = []

        def record(self):
            self.recorded = True

        def wait(self, stream):
            self.waits.append(stream)

    npu_ns = SimpleNamespace(
        Event=_DummyEvent,
        current_stream=lambda: main_stream,
        stream=lambda _stream: _NoopStreamCtx(),
    )
    monkeypatch.setattr(torch, "npu", npu_ns, raising=False)
    return main_stream


def _linear_layer(weight_shape=(2, 3), scale_shape=(2, 1)):
    layer = torch.nn.Module()
    weight = torch.arange(
        weight_shape[0] * weight_shape[1], dtype=torch.int64, device="cpu"
    ).reshape(weight_shape).to(torch.uint8)
    layer.weight = torch.nn.Parameter(
        weight,
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones(scale_shape, dtype=torch.float32, device="cpu")
    )
    layer.orig_dtype = torch.bfloat16
    return layer


def test_linear_process_weights_transforms_scale(hifloat8_module, mock_torch_npu):
    layer = _linear_layer()
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())

    method.process_weights_after_loading(layer)

    mock_torch_npu.npu_trans_quant_param.assert_called_once()
    assert layer.weight.shape == (3, 2)
    assert torch.equal(layer.weight_scale, torch.full_like(layer.weight_scale, 2.0))
    assert layer.weight.requires_grad is False
    assert layer.weight_scale.requires_grad is False


def test_flashcomm_linear_process_weights_transforms_scale(
    hifloat8_module, mock_torch_npu
):
    layer = _linear_layer(weight_shape=(4, 2), scale_shape=(4, 1))
    method = hifloat8_module.Hifloat8FCLinearMethod(quant_config=object())

    method.process_weights_after_loading(layer)

    mock_torch_npu.npu_trans_quant_param.assert_called_once()
    assert layer.weight.shape == (2, 4)
    assert torch.equal(layer.weight_scale, torch.full_like(layer.weight_scale, 2.0))
    assert layer.weight.requires_grad is False
    assert layer.weight_scale.requires_grad is False


def test_moe_apply_experts_flattens_scale_and_uses_swiglu_cast(
    hifloat8_module, mock_torch_npu
):
    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    layer = SimpleNamespace(
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
    )
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
    )

    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
        [torch.zeros(4, 4, dtype=torch.bfloat16)],
    ]

    result = method.apply_experts(layer, prepare_result)

    assert result.shape == (4, 4)
    assert mock_torch_npu.npu_grouped_matmul.call_count == 2
    first_kwargs = mock_torch_npu.npu_grouped_matmul.call_args_list[0].kwargs
    second_kwargs = mock_torch_npu.npu_grouped_matmul.call_args_list[1].kwargs
    assert first_kwargs["per_token_scale"][0].shape == (4,)
    assert second_kwargs["per_token_scale"] is None
    mock_torch_npu.npu_swiglu.assert_called_once()
    mock_torch_npu.npu_dtype_cast.assert_called_once()
    cast_args = mock_torch_npu.npu_dtype_cast.call_args.args
    assert cast_args[0].shape == (4, 4)
    assert cast_args[1] is mock_torch_npu.hifloat8


def test_linear_apply_with_dict_input_skips_dtype_cast(
    hifloat8_module, mock_torch_npu
):
    layer = SimpleNamespace(
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())
    x_hif8 = torch.zeros(4, 3, dtype=torch.int8)

    out = method.apply(layer, x={"x_hif8": x_hif8}, bias=None)

    mock_torch_npu.npu_dtype_cast.assert_not_called()
    mock_torch_npu.npu_quant_matmul.assert_called_once()
    assert mock_torch_npu.npu_quant_matmul.call_args.kwargs["x1"] is x_hif8
    assert out.dtype == layer.orig_dtype


def test_linear_apply_with_tensor_input_casts_to_hifloat8(
    hifloat8_module, mock_torch_npu
):
    layer = SimpleNamespace(
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())
    x = torch.ones(4, 3, dtype=torch.bfloat16)
    bias = torch.zeros(2, dtype=torch.bfloat16)

    method.apply(layer, x=x, bias=bias)

    mock_torch_npu.npu_dtype_cast.assert_called_once()
    cast_args = mock_torch_npu.npu_dtype_cast.call_args.args
    assert torch.equal(cast_args[0], x)
    assert cast_args[1] is mock_torch_npu.hifloat8
    qm_kwargs = mock_torch_npu.npu_quant_matmul.call_args.kwargs
    assert qm_kwargs["bias"].dtype == torch.float32


def test_moe_process_weights_calls_npu_trans_quant_param_for_both_scales(
    hifloat8_module, mock_torch_npu
):
    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros(2, 8, 4, dtype=torch.uint8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros(2, 4, 8, dtype=torch.uint8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.ones(2, 8, 1, dtype=torch.float32), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones(2, 4, 1, dtype=torch.float32), requires_grad=False
        ),
        ensure_moe_quant_config_init=MagicMock(),
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (2, 4, 8)
    assert layer.w2_weight.shape == (2, 8, 4)
    assert mock_torch_npu.npu_trans_quant_param.call_count == 2
    # squeeze(-1).float() => shape (2, 8) and (2, 4) of float32 (here +1 from mock)
    assert layer.w13_weight_scale.shape == (2, 8)
    assert layer.w2_weight_scale.shape == (2, 4)
    layer.ensure_moe_quant_config_init.assert_called_once()


def test_moe_apply_experts_with_routed_experts_cv_returns_tuple(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """Cover apply_experts side-stream branch when shared_expert_parallel_schedule == 'with_routed_experts_cv'."""
    main_stream = _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_routed_experts_cv",
    )

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.shared_experts_stream = _DummyStream()

    shared_experts = SimpleNamespace(
        act_fn=MagicMock(side_effect=lambda x: x),
        down_proj=MagicMock(return_value=torch.full((2, 4), 9.0, dtype=torch.bfloat16)),
    )
    layer = SimpleNamespace(
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
        shared_experts=shared_experts,
    )

    shared_event = torch.npu.Event()
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
        shared_expert_gate_up=torch.ones(2, 4, dtype=torch.bfloat16),
        shared_expert_finished_event=shared_event,
    )

    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
        [torch.zeros(4, 4, dtype=torch.bfloat16)],
    ]

    result = method.apply_experts(layer, prepare_result)

    assert isinstance(result, tuple)
    routed_out, shared_out = result
    assert routed_out.shape == (4, 4)
    assert torch.equal(shared_out, torch.full((2, 4), 9.0, dtype=torch.bfloat16))
    # event waited on the main stream and the shared down_proj ran
    assert shared_event.waits == [main_stream]
    shared_experts.act_fn.assert_called_once()
    shared_experts.down_proj.assert_called_once()
    # main stream waits on the side stream twice (around down_proj and again at end)
    assert main_stream.wait_stream_calls.count(method.shared_experts_stream) >= 2


def test_moe_apply_experts_with_routed_experts_cv_grouped_finalize_routing(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """When use_grouped_matmul_finalize_routing=True, apply_experts returns early but
    shared expert side stream still produces results that we can't return; cover line 508."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_routed_experts_cv",
    )

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.shared_experts_stream = _DummyStream()

    shared_experts = SimpleNamespace(
        act_fn=MagicMock(side_effect=lambda x: x),
        down_proj=MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16)),
    )
    layer = SimpleNamespace(
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
        shared_experts=shared_experts,
    )

    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(4, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=None,
        shared_expert_gate_up=torch.ones(2, 4, dtype=torch.bfloat16),
        shared_expert_finished_event=torch.npu.Event(),
    )

    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
    ]

    intermediate, pertoken_scale = method.apply_experts(
        layer, prepare_result, use_grouped_matmul_finalize_routing=True
    )

    assert intermediate.shape == (4, 4)
    assert pertoken_scale is None
    assert mock_torch_npu.npu_grouped_matmul.call_count == 1


def test_moe_init_records_shared_experts_stream_via_named_stream(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """Cover Hifloat8MoEMethod.__init__ — specifically that shared_experts_stream
    is acquired from named_stream('sub_stream')."""
    captured = []

    def fake_named_stream(name):
        captured.append(name)
        return SimpleNamespace(label=name)

    monkeypatch.setattr(hifloat8_module, "named_stream", fake_named_stream)

    layer = SimpleNamespace(
        moe_config=SimpleNamespace(num_experts=8),
        layer_name="layer_0",
    )
    method = hifloat8_module.Hifloat8MoEMethod(quant_config=object(), layer=layer)

    assert captured == ["sub_stream"]
    assert method.shared_experts_stream.label == "sub_stream"
    assert method.n_routed_experts == 8
    assert method.prefix == "layer_0"


def _build_apply_method(hifloat8_module, monkeypatch):
    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    method.shared_experts_stream = _DummyStream()
    method.select_communication_strategy = MagicMock(
        return_value=("agrs", SimpleNamespace())
    )
    method.apply_prepare_permute = MagicMock(
        return_value=SimpleNamespace(row_idx_type=0)
    )
    method.apply_unpermute_finalize = MagicMock(
        return_value=torch.zeros(2, 4, dtype=torch.bfloat16)
    )
    # Patch NPUFusedMoE.select_experts to avoid requiring full vllm gate.
    fake_npu_moe = SimpleNamespace(
        select_experts=lambda **_kwargs: (
            torch.ones(2, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(hifloat8_module, "NPUFusedMoE", fake_npu_moe)
    return method


def test_moe_apply_with_routed_experts_cv_unpacks_tuple(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """Cover apply lines 430-431: shared_expert_parallel_schedule == 'with_routed_experts_cv'
    consumes a (routed_output, shared_output) tuple from apply_experts."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_routed_experts_cv",
    )

    method = _build_apply_method(hifloat8_module, monkeypatch)
    routed = torch.zeros(2, 4, dtype=torch.bfloat16)
    shared = torch.full((2, 4), 7.0, dtype=torch.bfloat16)
    method.apply_experts = MagicMock(return_value=(routed, shared))

    layer = SimpleNamespace(
        gate=lambda x: (torch.zeros(x.shape[0], 4), None),
        shared_experts=SimpleNamespace(
            gate_up_proj=SimpleNamespace(tp_size=1),
        ),
    )
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)

    out = method.apply(
        layer=layer,
        hidden_states=hidden,
        router_logits=None,
        top_k=1,
        renormalize=False,
    )

    # Tuple (shared_output, routed_output) returned because shared_output is not None.
    assert isinstance(out, tuple)
    shared_out, routed_out = out
    assert torch.equal(shared_out, shared)
    assert torch.equal(routed_out, method.apply_unpermute_finalize.return_value)


def test_moe_apply_default_with_finalize_runs_shared_experts_locally(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """Cover apply lines 433-438, 440 default branch (with_finalize) and 446-450 all-reduce."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_finalize",
    )

    method = _build_apply_method(hifloat8_module, monkeypatch)
    method.apply_experts = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))

    shared_call = MagicMock(return_value=torch.full((2, 4), 5.0, dtype=torch.bfloat16))
    shared_experts = MagicMock(side_effect=shared_call)
    shared_experts.gate_up_proj = SimpleNamespace(tp_size=2)

    all_reduce_calls = []

    def fake_all_reduce(t):
        all_reduce_calls.append(t)
        return t * 2

    monkeypatch.setattr(
        hifloat8_module, "tensor_model_parallel_all_reduce", fake_all_reduce
    )

    layer = SimpleNamespace(
        gate=lambda x: (torch.zeros(x.shape[0], 4), None),
        shared_experts=shared_experts,
    )
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)

    out = method.apply(
        layer=layer,
        hidden_states=hidden,
        router_logits=None,
        top_k=1,
        renormalize=False,
    )

    # tp_size > 1 path: shared_experts(hidden_states) called and result all-reduced.
    shared_experts.assert_called_once()
    assert torch.equal(shared_experts.call_args.args[0], hidden)
    assert len(all_reduce_calls) == 1
    assert torch.equal(all_reduce_calls[0], shared_call.return_value)
    assert isinstance(out, tuple)


def test_moe_apply_default_with_finalize_shared_tp1_uses_x_slice(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """Cover apply line 440: tp_size == 1 branch invokes shared_experts(x_slice)."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_finalize",
    )

    method = _build_apply_method(hifloat8_module, monkeypatch)
    method.apply_experts = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))

    shared_experts = MagicMock(return_value=torch.full((2, 4), 3.0, dtype=torch.bfloat16))
    shared_experts.gate_up_proj = SimpleNamespace(tp_size=1)

    layer = SimpleNamespace(
        gate=lambda x: (torch.zeros(x.shape[0], 4), None),
        shared_experts=shared_experts,
    )
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)

    method.apply(
        layer=layer,
        hidden_states=hidden,
        router_logits=None,
        top_k=1,
        renormalize=False,
    )

    shared_experts.assert_called_once()
    # x_slice == hidden_states when tp_size == 1 (no slicing needed).
    assert torch.equal(shared_experts.call_args.args[0], hidden)


def test_moe_apply_experts_default_with_finalize_no_shared_path(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """When schedule is 'with_finalize' (default), apply_experts must not touch shared_experts."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_finalize",
    )

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.shared_experts_stream = _DummyStream()

    layer = SimpleNamespace(
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
        shared_experts=SimpleNamespace(act_fn=MagicMock(), down_proj=MagicMock()),
    )
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
    )

    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
        [torch.zeros(4, 4, dtype=torch.bfloat16)],
    ]

    result = method.apply_experts(layer, prepare_result)

    assert not isinstance(result, tuple)
    assert result.shape == (4, 4)
    layer.shared_experts.act_fn.assert_not_called()
    layer.shared_experts.down_proj.assert_not_called()


def _set_cube_side_tasks(hifloat8_module, tasks):
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    fwctx.additional_kwargs = {hifloat8_module.CUBE_SIDE_TASKS_KEY: tasks}


def test_linear_apply_runs_registered_cube_side_task(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    layer = SimpleNamespace(
        prefix="layers.0.self_attn.o_proj",
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    _set_cube_side_tasks(hifloat8_module, {layer.prefix: task})
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())

    out = method.apply(layer, x=torch.ones(4, 3, dtype=torch.bfloat16))

    assert flag["called"] is True
    assert task.done_event is not None and task.done_event.recorded
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY] == {}
    mock_torch_npu.npu_quant_matmul.assert_called_once()
    assert out.dtype == layer.orig_dtype


def test_linear_apply_skips_cube_side_task_when_prefix_not_registered(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    _set_cube_side_tasks(hifloat8_module, {"some.other.layer": task})
    layer = SimpleNamespace(
        prefix="layers.0.self_attn.o_proj",
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())

    method.apply(layer, x=torch.ones(4, 3, dtype=torch.bfloat16))

    assert flag["called"] is False
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert "some.other.layer" in fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY]


def test_linear_apply_no_op_when_no_cube_side_tasks_dict(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    fwctx.additional_kwargs = {}
    layer = SimpleNamespace(
        prefix="layers.0.self_attn.o_proj",
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())

    method.apply(layer, x=torch.ones(4, 3, dtype=torch.bfloat16))

    mock_torch_npu.npu_quant_matmul.assert_called_once()


def test_apply_experts_runs_registered_cube_side_task(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    layer = SimpleNamespace(
        layer_name="layers.0.mlp.experts",
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
    )
    _set_cube_side_tasks(hifloat8_module, {layer.layer_name: task})
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
    )
    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
        [torch.zeros(4, 4, dtype=torch.bfloat16)],
    ]

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.apply_experts(layer, prepare_result)

    assert flag["called"] is True
    assert task.done_event is not None and task.done_event.recorded
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY] == {}


def test_apply_experts_skips_cube_side_task_in_grouped_matmul_finalize_routing(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    layer = SimpleNamespace(
        layer_name="layers.0.mlp.experts",
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
    )
    _set_cube_side_tasks(hifloat8_module, {layer.layer_name: task})
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
    )
    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
    ]

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.apply_experts(layer, prepare_result, use_grouped_matmul_finalize_routing=True)

    assert flag["called"] is False
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert layer.layer_name in fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY]


def test_flashcomm_apply_runs_registered_cube_side_task(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)

    # Hifloat8FCLinearMethod.apply imports these lazily; stub the module so the
    # default (x_transform=None) path imports without the real comm extension.
    v1_distributed = types.ModuleType("omni_npu.v1.distributed")
    v1_distributed.__path__ = []
    comm_ext = types.ModuleType("omni_npu.v1.distributed.communication_op_ext")
    comm_ext.layer_parallel_all_gather = lambda x, *a, **k: x
    comm_ext.layer_parallel_all2all_single = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "omni_npu.v1.distributed", v1_distributed)
    monkeypatch.setitem(
        sys.modules, "omni_npu.v1.distributed.communication_op_ext", comm_ext
    )

    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    layer = SimpleNamespace(
        prefix="layers.0.self_attn.o_proj",
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    _set_cube_side_tasks(hifloat8_module, {layer.prefix: task})
    method = hifloat8_module.Hifloat8FCLinearMethod(quant_config=object())

    out = method.apply(layer, x=torch.ones(4, 3, dtype=torch.bfloat16), bias=None)

    # layer_key resolves from layer.prefix; the registered cube-side task fires
    # via torch.ops.vllm.cube_side_run and is awaited via cube_side_wait.
    assert flag["called"] is True
    assert task.done_event is not None and task.done_event.recorded
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY] == {}
    mock_torch_npu.npu_quant_matmul.assert_called_once()
    assert out.dtype == layer.orig_dtype
