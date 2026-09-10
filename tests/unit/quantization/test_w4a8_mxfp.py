# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU unit tests for the W4A8 MXFP dense-linear integration."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


_E8M0_SENTINEL = object()
_E2M1_X2_SENTINEL = object()


def _identity(value, *_args, **_kwargs):
    return value


def _noop(*_args, **_kwargs):
    return None


def _null_execution_context(_label):
    return _NullExecutionContext()


def _return_value_after_key(_key, value):
    return value


def _mock_dynamic_mx_quant(x, dst_type=None, scale_alg=None):
    del dst_type, scale_alg
    return (
        torch.zeros_like(x, dtype=torch.int8),
        torch.zeros(x.shape[0], x.shape[-1] // 32, dtype=torch.uint8),
    )


def _mock_quant_matmul(x, weight, weight_scale, **kwargs):
    del weight_scale
    return torch.zeros(
        x.shape[0],
        weight.shape[-1],
        dtype=kwargs["output_dtype"],
    )


def _mock_swiglu_mx_quant(
    gate_up,
    group_index=None,
    activate_left=True,
    dst_type=None,
    scale_alg=None,
):
    del group_index, activate_left, dst_type, scale_alg
    output_size = gate_up.shape[-1] // 2
    return (
        torch.zeros(
            gate_up.shape[0],
            output_size,
            dtype=torch.int8,
        ),
        torch.zeros(
            gate_up.shape[0],
            output_size // 32,
            dtype=torch.uint8,
        ),
    )


def _mock_grouped_matmul(inputs, weights, **kwargs):
    return [
        torch.zeros(
            inputs[0].shape[0],
            weights[0].shape[-1],
            dtype=kwargs["output_dtype"],
        )
    ]


def _make_mock_torch_npu():
    return SimpleNamespace(
        float4_e2m1fn_x2=_E2M1_X2_SENTINEL,
        float8_e8m0fnu=_E8M0_SENTINEL,
        npu_dynamic_mx_quant=MagicMock(
            side_effect=_mock_dynamic_mx_quant
        ),
        npu_format_cast=MagicMock(side_effect=_identity),
        npu_quant_matmul=MagicMock(side_effect=_mock_quant_matmul),
        npu_swiglu_mx_quant=MagicMock(
            side_effect=_mock_swiglu_mx_quant
        ),
        npu_grouped_matmul=MagicMock(
            side_effect=_mock_grouped_matmul
        ),
    )


@pytest.fixture
def w4_module(monkeypatch):
    module = importlib.import_module(
        "omni_npu.layers.quantization.w4a8_mxfp"
    )
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    # Another test module temporarily replaces vLLM's fused-MoE module while
    # importing omni's package. Restore the real class to keep this fixture
    # independent of test collection and execution order.
    monkeypatch.setattr(module, "RoutedExperts", RoutedExperts)
    monkeypatch.setattr(
        torch,
        "float8_e4m3fn",
        torch.int8,
        raising=False,
    )
    mock_npu = _make_mock_torch_npu()
    monkeypatch.setattr(module, "torch_npu", mock_npu)
    monkeypatch.setattr(
        module,
        "_FLOAT4_E2M1FN_X2_DTYPE",
        _E2M1_X2_SENTINEL,
    )
    monkeypatch.setattr(
        module,
        "_FLOAT8_E8M0FNU_DTYPE",
        _E8M0_SENTINEL,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "cube_side_run",
        lambda _key, x: x,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "cube_side_wait",
        lambda _key, x: x,
        raising=False,
    )
    with patch(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        return_value=0,
    ), patch(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        yield module, mock_npu


def _create_linear_layer(module, method_cls=None):
    method_cls = method_cls or module.W4A8MXFPLinearMethod
    config = module.W4A8MXFPConfig(group_size=32)
    method = method_cls(config)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[96],
        input_size=128,
        output_size=96,
        params_dtype=torch.bfloat16,
        weight_loader=_noop,
    )
    layer.prefix = "model.layers.0.self_attn.q_proj"
    layer.layer_name_inside_block = "self_attn.q_proj"
    return method, layer


class TestW4A8MXFPConfig:

    def test_from_config(self, w4_module):
        module, _ = w4_module
        config = module.W4A8MXFPConfig.from_config(
            {
                "group_size": 64,
                "ignore": ["lm_head"],
            }
        )
        assert config.group_size == 64
        assert config.ignored_layers == ["lm_head"]
        assert config.get_name() == "w4a8_mxfp"

    def test_default_group_size(self, w4_module):
        module, _ = w4_module
        config = module.W4A8MXFPConfig.from_config({})
        assert config.group_size == 32
        assert config.non_moe_quant_method == "w4a8_mxfp"

    def test_mxfp8_non_moe_config(self, w4_module):
        module, _ = w4_module
        config = module.W4A8MXFPConfig.from_config(
            {
                "group_size": 32,
                "ignore": ["lm_head"],
                "non_moe_quant_method": "mxfp8",
            }
        )

        assert config.non_moe_quant_method == "mxfp8"
        assert config._mxfp8_config.ignored_layers == ["lm_head"]

    def test_invalid_group_size(self, w4_module):
        module, _ = w4_module
        with pytest.raises(ValueError, match="must be positive"):
            module.W4A8MXFPConfig(group_size=0)

    def test_invalid_non_moe_quant_method(self, w4_module):
        module, _ = w4_module
        with pytest.raises(ValueError, match="non_moe_quant_method"):
            module.W4A8MXFPConfig(non_moe_quant_method="int8")

    def test_mxfp8_non_moe_dispatch_delegates_to_mxfp8(
        self,
        w4_module,
    ):
        module, _ = w4_module
        config = module.W4A8MXFPConfig(
            non_moe_quant_method="mxfp8",
        )
        expected = object()
        config._mxfp8_config.get_quant_method_custom = MagicMock(
            return_value=expected
        )
        layer = torch.nn.Module()

        result = config.get_quant_method_custom(
            layer,
            "model.layers.0.self_attn.q_proj",
        )

        assert result is expected
        config._mxfp8_config.get_quant_method_custom.assert_called_once_with(
            layer,
            "model.layers.0.self_attn.q_proj",
        )

    def test_mxfp8_non_moe_dispatch_keeps_fused_moe_w4a8(
        self,
        w4_module,
        monkeypatch,
    ):
        module, _ = w4_module

        class DummyRoutedExperts:
            pass

        expected = object()
        w4_moe_method = MagicMock(return_value=expected)
        monkeypatch.setattr(module, "RoutedExperts", DummyRoutedExperts)
        monkeypatch.setattr(
            module,
            "W4A8MXFPMoEMethod",
            w4_moe_method,
        )
        config = module.W4A8MXFPConfig(
            non_moe_quant_method="mxfp8",
        )
        config._mxfp8_config.get_quant_method_custom = MagicMock()
        layer = DummyRoutedExperts()

        result = config.get_quant_method_custom(
            layer,
            "model.layers.1.mlp.experts",
        )

        assert result is expected
        w4_moe_method.assert_called_once_with(config, layer)
        config._mxfp8_config.get_quant_method_custom.assert_not_called()


class TestW4A8MXFPLayout:

    def test_create_checkpoint_layout(self, w4_module):
        module, _ = w4_module
        _, layer = _create_linear_layer(module)
        assert layer.weight.shape == (96, 64)
        assert layer.weight.dtype == torch.uint8
        assert layer.weight_scale.shape == (96, 4)
        assert layer.weight_scale.dtype == torch.uint8

    def test_input_must_fit_paired_scale_layout(self, w4_module):
        module, _ = w4_module
        method = module.W4A8MXFPLinearMethod(
            module.W4A8MXFPConfig(group_size=32)
        )
        with pytest.raises(ValueError, match="2 \\* group_size"):
            method.create_weights(
                torch.nn.Module(),
                input_size_per_partition=96,
                output_partition_sizes=[64],
                input_size=96,
                output_size=64,
                params_dtype=torch.bfloat16,
                weight_loader=_noop,
            )

    def test_process_weights_packs_kernel_layout(self, w4_module):
        module, mock_npu = w4_module
        method, layer = _create_linear_layer(module)
        checkpoint_scale = (
            torch.arange(
                layer.weight_scale.numel(),
                dtype=torch.int64,
            )
            .remainder(256)
            .to(torch.uint8)
            .reshape_as(layer.weight_scale)
        )
        layer.weight_scale.data.copy_(checkpoint_scale)
        method.process_weights_after_loading(layer)

        assert layer.weight.shape == (64, 96)
        assert layer.weight_scale.shape == (2, 96, 2)
        expected_scale = (
            checkpoint_scale.reshape(96, 2, 2)
            .transpose(0, 1)
        )
        assert torch.equal(layer.weight_scale, expected_scale)
        assert layer.weight_scale.stride() == expected_scale.stride()
        assert not layer.weight_scale.is_contiguous()
        assert layer.weight.requires_grad is False
        assert layer.weight_scale.requires_grad is False
        assert getattr(
            layer.weight,
            module._W4A8_MXFP_PACKED_ATTR,
            False,
        )

        _, args, kwargs = mock_npu.npu_format_cast.mock_calls[0]
        assert args[1] == 29
        assert kwargs["customize_dtype"] == torch.float8_e4m3fn
        assert kwargs["input_dtype"] is _E2M1_X2_SENTINEL

        method.process_weights_after_loading(layer)
        assert mock_npu.npu_format_cast.call_count == 1

    def test_rejects_inconsistent_weight_and_scale(self, w4_module):
        module, _ = w4_module
        weight = torch.zeros(8, 64, dtype=torch.uint8)
        scale = torch.zeros(8, 2, dtype=torch.uint8)
        with pytest.raises(ValueError, match="inconsistent"):
            module._pack_w4a8_mxfp_weight(
                weight,
                scale,
                group_size=32,
            )


class TestW4A8MXFPLinear:

    def test_runtime_capability_is_checked_only_at_construction(
        self,
        w4_module,
    ):
        module, _ = w4_module
        capability_check = MagicMock()
        with patch.object(
            module,
            "_require_w4a8_mxfp_runtime",
            capability_check,
        ):
            method, layer = _create_linear_layer(module)
            method.process_weights_after_loading(layer)
            method.apply(
                layer,
                torch.zeros(5, 128, dtype=torch.bfloat16),
            )

        capability_check.assert_called_once_with()

    def test_raw_input_calls_w4a8_kernel(self, w4_module):
        module, mock_npu = w4_module
        method, layer = _create_linear_layer(module)
        method.process_weights_after_loading(layer)

        output = method.apply(
            layer,
            torch.zeros(5, 128, dtype=torch.bfloat16),
        )

        assert output.shape == (5, 96)
        mock_npu.npu_dynamic_mx_quant.assert_called_once()
        _, kwargs = mock_npu.npu_quant_matmul.call_args
        assert kwargs["group_sizes"] == [0, 0, 32]
        assert kwargs["x2_dtype"] is _E2M1_X2_SENTINEL
        assert kwargs["scale_dtype"] is _E8M0_SENTINEL
        assert kwargs["pertoken_scale_dtype"] is _E8M0_SENTINEL

    @pytest.mark.parametrize("container_type", ["tuple", "dict"])
    def test_prequantized_input_skips_dynamic_quant(
        self,
        w4_module,
        container_type,
    ):
        module, mock_npu = w4_module
        method, layer = _create_linear_layer(module)
        method.process_weights_after_loading(layer)
        activation = torch.zeros(5, 128, dtype=torch.int8)
        scale = torch.zeros(5, 4, dtype=torch.uint8)
        if container_type == "tuple":
            inputs = (activation, scale)
        else:
            inputs = {
                "x_mxfp8": activation,
                "pertoken_scale": scale,
            }

        output = method.apply(layer, inputs)

        assert output.dtype == torch.bfloat16
        mock_npu.npu_dynamic_mx_quant.assert_not_called()


class TestW4A8MXFPFCLinear:

    def test_cube_side_hooks_wrap_matmul(
        self,
        w4_module,
        monkeypatch,
    ):
        module, _ = w4_module
        method, layer = _create_linear_layer(
            module,
            module.W4A8MXFPFCLinearMethod,
        )
        method.process_weights_after_loading(layer)
        cube_side_run = MagicMock(side_effect=_return_value_after_key)
        cube_side_wait = MagicMock(side_effect=_return_value_after_key)
        monkeypatch.setattr(torch.ops.vllm, "cube_side_run", cube_side_run)
        monkeypatch.setattr(torch.ops.vllm, "cube_side_wait", cube_side_wait)

        output = method.apply(
            layer,
            torch.zeros(5, 128, dtype=torch.bfloat16),
        )

        assert output.shape == (5, 96)
        assert cube_side_run.call_args.args[0] == layer.prefix
        assert cube_side_wait.call_args.args[0] == layer.prefix

    @pytest.mark.parametrize(
        ("transform_name", "patch_name"),
        [
            ("AllGather", "layer_parallel_all_gather"),
            ("ALL2ALL", "layer_parallel_all2all_single"),
        ],
    )
    def test_communication_transforms_activation_and_scale(
        self,
        w4_module,
        transform_name,
        patch_name,
    ):
        module, _ = w4_module
        method, layer = _create_linear_layer(
            module,
            module.W4A8MXFPFCLinearMethod,
        )
        method.process_weights_after_loading(layer)
        transform = MagicMock(side_effect=_identity)
        patch_target = (
            "omni_npu.v1.distributed.communication_op_ext."
            f"{patch_name}"
        )

        with patch(patch_target, transform):
            output = method.apply(
                layer,
                torch.zeros(5, 128, dtype=torch.bfloat16),
                x_transform=transform_name,
            )

        assert output.shape == (5, 96)
        assert transform.call_count == 2


class _NullExecutionContext:

    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


class _MockProjection:

    def __init__(self, output_size):
        self.output_size = output_size
        self.last_input = None

    def __call__(self, inputs, **kwargs):
        del kwargs
        self.last_input = inputs
        rows = inputs["x_mxfp8"].shape[0]
        return (
            torch.zeros(
                rows,
                self.output_size,
                dtype=torch.bfloat16,
            ),
            None,
        )


class _MockFusedMLP:

    def __init__(self):
        self.gate_up_proj = _MockProjection(256)
        self.down_proj = _MockProjection(128)


class TestW4A8MXFPMlp:

    @pytest.fixture
    def mlp_method(self, w4_module):
        module, _ = w4_module
        method = module.W4A8MXFPMlpMethod(
            module.W4A8MXFPConfig()
        )
        with patch(
            "omni_npu.v1.layers.utils.get_npu_execution_type",
            side_effect=_null_execution_context,
        ):
            yield method

    def test_raw_input_runs_quant_swiglu_and_down(
        self,
        w4_module,
        mlp_method,
    ):
        _, mock_npu = w4_module
        layer = _MockFusedMLP()

        output = mlp_method.apply(
            layer,
            torch.zeros(4, 128, dtype=torch.bfloat16),
        )

        assert output.shape == (4, 128)
        mock_npu.npu_dynamic_mx_quant.assert_called_once()
        mock_npu.npu_swiglu_mx_quant.assert_called_once()
        assert layer.gate_up_proj.last_input["x_mxfp8"].shape == (
            4,
            128,
        )
        assert layer.down_proj.last_input["x_mxfp8"].shape == (
            4,
            128,
        )

    def test_prequantized_input_skips_initial_quant(
        self,
        w4_module,
        mlp_method,
    ):
        _, mock_npu = w4_module
        layer = _MockFusedMLP()
        inputs = {
            "x_mxfp8": torch.zeros(4, 128, dtype=torch.int8),
            "pertoken_scale": torch.zeros(4, 4, dtype=torch.uint8),
        }

        output = mlp_method.apply(layer, inputs)

        assert output.shape == (4, 128)
        mock_npu.npu_dynamic_mx_quant.assert_not_called()
        mock_npu.npu_swiglu_mx_quant.assert_called_once()

    def test_process_weights_after_loading_is_noop(self, mlp_method):
        mlp_method.process_weights_after_loading(object())

    def test_config_dispatches_fused_mlp_method(self, w4_module):
        module, _ = w4_module
        from omni_npu.v1.layers.fused_mlp.layer import FusedMLP

        layer = object.__new__(FusedMLP)
        torch.nn.Module.__init__(layer)
        method = module.W4A8MXFPConfig().get_quant_method_custom(
            layer,
            "model.layers.0.mlp",
        )

        assert isinstance(method, module.W4A8MXFPMlpMethod)


class _MockMoELayer:

    def __init__(self, has_bias=False):
        self.moe_config = SimpleNamespace(
            num_experts=4,
            has_bias=has_bias,
        )
        self.layer_name = "model.layers.0.mlp.experts"


@pytest.fixture
def moe_method_factory(w4_module):
    module, _ = w4_module

    def create(layer=None):
        stream = MagicMock()
        with patch(
            "omni_npu.layers.quantization.mxfp8.named_stream",
            return_value=stream,
        ), patch(
            "omni_npu.layers.quantization.mxfp8."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ), patch(
            "omni_npu.layers.quantization.mxfp8."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ), patch(
            "omni_npu.layers.quantization.mxfp8."
            "get_current_vllm_config",
            return_value=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=SimpleNamespace())
            ),
        ), patch(
            "omni_npu.layers.quantization.mxfp8.on_ascend950",
            return_value=True,
        ):
            return module.W4A8MXFPMoEMethod(
                module.W4A8MXFPConfig(),
                layer or _MockMoELayer(),
            )

    return create


@pytest.fixture
def moe_method(moe_method_factory):
    return moe_method_factory()


def _create_moe_weights(method):
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace()
    method.create_weights(
        layer,
        num_experts=4,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )
    return layer


class TestW4A8MXFPMoE:

    def test_moe_quant_config_distinguishes_mxfp4_weight(
        self,
        w4_module,
    ):
        module, _ = w4_module
        w1_scale = torch.zeros(4, 2, 128, 2, dtype=torch.uint8)
        w2_scale = torch.zeros(4, 1, 128, 2, dtype=torch.uint8)

        config = module.w4a8_mxfp_moe_quant_config(
            w1_scale,
            w2_scale,
        )

        assert config.use_mxfp4_w4a8 is True
        assert config.use_mxfp8_w8a8 is False
        assert config._w1.scale is w1_scale
        assert config._w2.scale is w2_scale

    def test_create_checkpoint_layout(self, moe_method):
        layer = _create_moe_weights(moe_method)

        assert layer.w13_weight.shape == (4, 128, 64)
        assert layer.w2_weight.shape == (4, 128, 32)
        assert layer.w13_weight_scale.shape == (4, 128, 4)
        assert layer.w2_weight_scale.shape == (4, 128, 2)
        assert layer.w13_weight.dtype == torch.uint8
        assert layer.w2_weight.dtype == torch.uint8

    def test_process_weights_packs_gmm_layout(
        self,
        w4_module,
        moe_method,
    ):
        module, mock_npu = w4_module
        layer = _create_moe_weights(moe_method)
        layer.ensure_moe_quant_config_init = MagicMock()
        checkpoint_w13_scale = (
            torch.arange(
                layer.w13_weight_scale.numel(),
                dtype=torch.int64,
            )
            .remainder(256)
            .to(torch.uint8)
            .reshape_as(layer.w13_weight_scale)
        )
        checkpoint_w2_scale = (
            torch.arange(
                layer.w2_weight_scale.numel(),
                dtype=torch.int64,
            )
            .remainder(256)
            .to(torch.uint8)
            .reshape_as(layer.w2_weight_scale)
        )
        layer.w13_weight_scale.data.copy_(checkpoint_w13_scale)
        layer.w2_weight_scale.data.copy_(checkpoint_w2_scale)

        moe_method.process_weights_after_loading(layer)

        assert layer.w13_weight.shape == (4, 64, 128)
        assert layer.w2_weight.shape == (4, 32, 128)
        assert layer.w13_weight_scale.shape == (4, 2, 128, 2)
        assert layer.w2_weight_scale.shape == (4, 1, 128, 2)
        expected_w13_scale = (
            checkpoint_w13_scale.reshape(4, 128, 2, 2)
            .transpose(1, 2)
        )
        expected_w2_scale = (
            checkpoint_w2_scale.reshape(4, 128, 1, 2)
            .transpose(1, 2)
        )
        assert torch.equal(
            layer.w13_weight_scale,
            expected_w13_scale,
        )
        assert torch.equal(
            layer.w2_weight_scale,
            expected_w2_scale,
        )
        assert (
            layer.w13_weight_scale.stride()
            == expected_w13_scale.stride()
        )
        assert (
            layer.w2_weight_scale.stride()
            == expected_w2_scale.stride()
        )
        assert not layer.w13_weight_scale.is_contiguous()
        layer.ensure_moe_quant_config_init.assert_called_once()
        assert getattr(
            layer.w13_weight,
            module._W4A8_MXFP_PACKED_ATTR,
            False,
        )

        moe_method.process_weights_after_loading(layer)
        assert mock_npu.npu_format_cast.call_count == 2
        layer.ensure_moe_quant_config_init.assert_called_once()

    def test_expert_bias_is_rejected(self, moe_method_factory):
        layer_wrapper = _MockMoELayer(has_bias=True)
        method = moe_method_factory(layer_wrapper)
        with pytest.raises(
            NotImplementedError,
            match="does not support expert bias",
        ):
            _create_moe_weights(method)

    @pytest.mark.parametrize(
        ("use_ep", "expected_group_list_type"),
        [(False, 0), (True, 1)],
    )
    def test_apply_experts_uses_antiquant_scale_and_ep_group_list(
        self,
        w4_module,
        moe_method,
        use_ep,
        expected_group_list_type,
    ):
        _, mock_npu = w4_module
        layer = _create_moe_weights(moe_method)
        layer.moe_parallel_config = SimpleNamespace(use_ep=use_ep)
        layer.ensure_moe_quant_config_init = MagicMock()
        moe_method.process_weights_after_loading(layer)
        prepare_result = SimpleNamespace(
            hidden_states_sorted_by_experts=torch.zeros(
                8,
                128,
                dtype=torch.int8,
            ),
            expert_tokens=torch.tensor(
                [2, 2, 2, 2],
                dtype=torch.int64,
            ),
            dynamic_scale=torch.zeros(
                8,
                4,
                dtype=torch.uint8,
            ),
        )

        output = moe_method.apply_experts(
            layer,
            prepare_result,
        )

        assert output.shape == (8, 128)
        assert mock_npu.npu_grouped_matmul.call_count == 2
        first_call = (
            mock_npu.npu_grouped_matmul.call_args_list[0]
        )
        second_call = (
            mock_npu.npu_grouped_matmul.call_args_list[1]
        )
        assert first_call.kwargs["scale"] is None
        assert first_call.kwargs["antiquant_scale"][0] is (
            layer.w13_weight_scale
        )
        assert second_call.kwargs["scale"] is None
        assert second_call.kwargs["antiquant_scale"][0] is (
            layer.w2_weight_scale
        )
        assert first_call.kwargs["group_list_type"] == (
            expected_group_list_type
        )
        assert second_call.kwargs["group_list_type"] == (
            expected_group_list_type
        )
        assert first_call.kwargs["weight_dtype"] is (
            _E2M1_X2_SENTINEL
        )
        mock_npu.npu_swiglu_mx_quant.assert_called_once()
