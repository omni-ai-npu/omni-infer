# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``omni_npu.layers.quantization.mxfp8``.

MXFP8 kernels live on A5 hardware, but CI runs on A2. Every ``torch_npu.*``
kernel used by the module is therefore replaced with a shape-correct Python
stub so the module's Python-level logic (layout packing, parameter
registration, config dispatch, forward glue) can still be exercised.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


# --------------------------------------------------------------------------- #
# torch_npu kernel mocks
# --------------------------------------------------------------------------- #

# Sentinel passed as a dtype kwarg; the real torch_npu.float8_e8m0fnu is not
# available on A2 hosts and the kernels never read its value.
_E8M0_SENTINEL = object()


def _mock_npu_dynamic_mx_quant(x, dst_type=None):
    m, k = x.shape[-2], x.shape[-1]
    x_fp8 = torch.zeros(*x.shape, dtype=torch.int8)
    x_scale = torch.zeros(m, k // 32, dtype=torch.uint8)
    return x_fp8, x_scale


def _mock_npu_quant_matmul(x, w, w_scale, **kwargs):
    # kernel-ready weight layout is (K, N) after process_weights_after_loading.
    m = x.shape[0]
    n = w.shape[-1]
    out_dtype = kwargs.get("output_dtype") or torch.bfloat16
    return torch.zeros(m, n, dtype=out_dtype)


def _mock_npu_grouped_matmul(xs, ws, **kwargs):
    # w layout: (E, K, N) after process_weights_after_loading.
    x = xs[0]
    w = ws[0]
    m = x.shape[0]
    n = w.shape[-1]
    out_dtype = kwargs.get("output_dtype") or torch.bfloat16
    return [torch.zeros(m, n, dtype=out_dtype)]


def _mock_npu_swiglu_mx_quant(gate_up, group_index=None, activate_left=True, dst_type=None):
    m, two_n = gate_up.shape
    n = two_n // 2
    intermediate = torch.zeros(m, n, dtype=torch.int8)
    scale = torch.zeros(m, n // 32, dtype=torch.uint8)
    return intermediate, scale


def _make_mock_torch_npu():
    return SimpleNamespace(
        npu_dynamic_mx_quant=MagicMock(side_effect=_mock_npu_dynamic_mx_quant),
        npu_quant_matmul=MagicMock(side_effect=_mock_npu_quant_matmul),
        npu_grouped_matmul=MagicMock(side_effect=_mock_npu_grouped_matmul),
        npu_swiglu_mx_quant=MagicMock(side_effect=_mock_npu_swiglu_mx_quant),
        float8_e8m0fnu=_E8M0_SENTINEL,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


_orig_finfo = torch.finfo


@pytest.fixture
def mock_torch_npu(monkeypatch):
    monkeypatch.setattr(torch, "float8_e4m3fn", torch.int8)
    monkeypatch.setattr(
        torch,
        "finfo",
        lambda dtype: _orig_finfo(torch.float16) if dtype == torch.int8 else _orig_finfo(dtype),
    )
    ns = _make_mock_torch_npu()
    # The cube_side_{run,wait} custom ops are only registered for the
    # PrivateUse1 (NPU) dispatch key — invoking them on CPU tensors in unit
    # tests raises NotImplementedError. Override just those two attrs on
    # torch.ops.vllm (other vllm ops like dequant_mxfp4 must keep working,
    # so we don't replace the whole namespace).
    monkeypatch.setattr(
        torch.ops.vllm, "cube_side_run", lambda _key, x: x, raising=False,
    )
    monkeypatch.setattr(
        torch.ops.vllm, "cube_side_wait", lambda _key, y: y, raising=False,
    )
    # ModelWeightParameter.__init__ queries the TP group on construction.
    # In a unit-test context no distributed env is initialised, so we stub
    # those accessors at the single import site inside vllm.
    with patch("omni_npu.layers.quantization.mxfp8.torch_npu", ns), patch(
        "omni_npu.layers.quantization.mxfp8._FLOAT8_E8M0FNU_DTYPE",
        _E8M0_SENTINEL,
    ), patch(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        return_value=0,
    ), patch(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        yield ns


@pytest.fixture
def mxfp8_module(mock_torch_npu):
    # Re-import every test so we always see the patched torch_npu binding.
    return importlib.import_module("omni_npu.layers.quantization.mxfp8")


@pytest.fixture
def null_stream_ctx():
    """Context manager that does nothing; replaces get_npu_execution_type."""

    class _Null:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    return lambda _label: _Null()


# --------------------------------------------------------------------------- #
# Mock layer helpers
# --------------------------------------------------------------------------- #


class _MockQuantLinear:
    """Stand-in for an ``Mxfp8FCLinear`` submodule used inside fused MLPs."""

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.layer_name_inside_block = "mlp_layer"

    def __call__(self, x, throw_dequant=False):
        m = x["x_mxfp8"].shape[0] if isinstance(x, dict) else x.shape[0]
        out = torch.zeros(m, self.out_features, dtype=torch.bfloat16)
        return out, None


class _MockSiluAndMul:
    def __call__(self, x, quant_symbol=False):
        # Simulate the fused activation: halves the last dim and returns a dict
        # that looks like quantized output.
        if isinstance(x, dict):
            gate_up = x["x_mxfp8"]
        else:
            gate_up = x
        m, n2 = gate_up.shape
        return {
            "x_mxfp8": torch.zeros(m, n2 // 2, dtype=torch.int8),
            "pertoken_scale": torch.zeros(m, (n2 // 2) // 32, dtype=torch.uint8),
        }


class _MockMLPLayer:
    def __init__(self, hidden=128, inter=256):
        self.gate_up_proj = _MockQuantLinear(hidden, inter * 2)
        self.down_proj = _MockQuantLinear(inter, hidden)
        self.act_fn = _MockSiluAndMul()


# --------------------------------------------------------------------------- #
# Helper function tests
# --------------------------------------------------------------------------- #


class TestModuleHelpers:
    def test_is_layer_ignored_empty(self, mxfp8_module):
        assert mxfp8_module._is_layer_ignored("foo.bar", []) is False

    def test_is_layer_ignored_exact(self, mxfp8_module):
        assert mxfp8_module._is_layer_ignored("lm_head", ["lm_head"]) is True

    def test_is_layer_ignored_substring(self, mxfp8_module):
        assert mxfp8_module._is_layer_ignored(
            "model.layers.0.mlp.gate_up", ["mlp.gate_up"]
        ) is True

    def test_is_layer_ignored_regex(self, mxfp8_module):
        assert mxfp8_module._is_layer_ignored(
            "layers.10.attn", ["re:layers\\.\\d+\\.attn"]
        ) is True

    def test_is_layer_ignored_no_match(self, mxfp8_module):
        assert mxfp8_module._is_layer_ignored(
            "model.lm_head", ["re:layers\\.\\d+", "attn"]
        ) is False

    def test_pack_mxfp8_weight_shape_and_dtype(self, mxfp8_module):
        O, I = 64, 128
        weight = torch.zeros(O, I, dtype=torch.float8_e4m3fn)
        scales = torch.zeros(O, I // 32, dtype=torch.uint8)
        w, s = mxfp8_module._pack_mxfp8_weight(weight, scales)
        assert w.shape == (I, O)
        assert w.dtype == torch.float8_e4m3fn
        assert s.shape == (I // 64, O, 2)
        assert s.dtype == torch.uint8
        assert w.is_contiguous()
        assert s.is_contiguous()

    def test_pack_mxfp8_weight_odd_blocks_raises(self, mxfp8_module):
        weight = torch.zeros(8, 32 * 3, dtype=torch.float8_e4m3fn)
        scales = torch.zeros(8, 3, dtype=torch.uint8)
        with pytest.raises(AssertionError):
            mxfp8_module._pack_mxfp8_weight(weight, scales)

    def test_pack_mxfp8_expert_weight_shape(self, mxfp8_module):
        E, O, I = 4, 64, 128
        weight = torch.zeros(E, O, I, dtype=torch.float8_e4m3fn)
        scales = torch.zeros(E, O, I // 32, dtype=torch.uint8)
        w, s = mxfp8_module._pack_mxfp8_expert_weight(weight, scales)
        assert w.shape == (E, I, O)
        assert s.shape == (E, I // 64, O, 2)

    def test_pack_mxfp8_expert_weight_odd_blocks_raises(self, mxfp8_module):
        weight = torch.zeros(2, 8, 32 * 3, dtype=torch.float8_e4m3fn)
        scales = torch.zeros(2, 8, 3, dtype=torch.uint8)
        with pytest.raises(AssertionError):
            mxfp8_module._pack_mxfp8_expert_weight(weight, scales)

    def test_mxfp8_moe_quant_config(self, mxfp8_module):
        w1 = torch.zeros(2, 4, dtype=torch.uint8)
        w2 = torch.zeros(2, 4, dtype=torch.uint8)
        cfg = mxfp8_module.mxfp8_moe_quant_config(w1, w2, None, None)
        assert cfg._w1.scale is w1
        assert cfg._w2.scale is w2
        assert cfg._a1.scale is None
        assert cfg._a2.scale is None

    def test_create_mxfp8_linear_weights(self, mxfp8_module):
        layer = torch.nn.Module()
        mxfp8_module._create_mxfp8_linear_weights(
            layer,
            output_size_per_partition=64,
            input_size_per_partition=128,
            weight_loader=lambda *a, **k: None,
        )
        assert layer.weight.shape == (64, 128)
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.shape == (64, 128 // 32)
        assert layer.weight_scale.dtype == torch.uint8

    def test_create_mxfp8_linear_weights_input_not_multiple_raises(self, mxfp8_module):
        layer = torch.nn.Module()
        with pytest.raises(AssertionError):
            mxfp8_module._create_mxfp8_linear_weights(
                layer,
                output_size_per_partition=64,
                input_size_per_partition=33,
                weight_loader=lambda *a, **k: None,
            )


# --------------------------------------------------------------------------- #
# Mxfp8Config tests
# --------------------------------------------------------------------------- #


class TestMxfp8Config:
    def test_from_config_with_ignore(self, mxfp8_module):
        cfg = mxfp8_module.Mxfp8Config.from_config({"ignore": ["lm_head"]})
        assert cfg.ignored_layers == ["lm_head"]

    def test_from_config_missing_ignore(self, mxfp8_module):
        cfg = mxfp8_module.Mxfp8Config.from_config({})
        assert cfg.ignored_layers is None

    def test_from_config_empty_ignore_normalises_to_none(self, mxfp8_module):
        cfg = mxfp8_module.Mxfp8Config.from_config({"ignore": []})
        assert cfg.ignored_layers is None

    def test_get_name(self, mxfp8_module):
        assert mxfp8_module.Mxfp8Config.get_name() == "mxfp8"

    def test_get_supported_act_dtypes(self, mxfp8_module):
        assert mxfp8_module.Mxfp8Config.get_supported_act_dtypes() == [torch.bfloat16]

    def test_get_min_capability_raises(self, mxfp8_module):
        with pytest.raises(NotImplementedError):
            mxfp8_module.Mxfp8Config.get_min_capability()

    def test_get_config_filenames_empty(self, mxfp8_module):
        assert mxfp8_module.Mxfp8Config.get_config_filenames() == []

    def test_get_quant_method_raises_without_env(self, mxfp8_module, monkeypatch):
        monkeypatch.delenv("VLLM_PLUGINS", raising=False)
        cfg = mxfp8_module.Mxfp8Config()
        with pytest.raises(NotImplementedError):
            cfg.get_quant_method(torch.nn.Linear(4, 4), "layer")

    def test_get_quant_method_routes_to_custom(self, mxfp8_module, monkeypatch):
        monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
        cfg = mxfp8_module.Mxfp8Config()
        sentinel = object()
        cfg.get_quant_method_custom = lambda layer, prefix: sentinel
        assert cfg.get_quant_method(torch.nn.Linear(4, 4), "x") is sentinel

    def test_dispatch_unknown_layer_returns_none(self, mxfp8_module):
        cfg = mxfp8_module.Mxfp8Config()
        assert cfg.get_quant_method_custom(torch.nn.Module(), "x") is None

    def test_dispatch_linear_base(self, mxfp8_module):
        from vllm.model_executor.layers.linear import LinearBase

        layer = MagicMock(spec=LinearBase)
        cfg = mxfp8_module.Mxfp8Config()
        result = cfg.get_quant_method_custom(layer, "model.layer")
        assert isinstance(result, mxfp8_module.Mxfp8LinearMethod)

    def test_dispatch_linear_base_ignored(self, mxfp8_module):
        from vllm.model_executor.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )

        layer = MagicMock(spec=LinearBase)
        cfg = mxfp8_module.Mxfp8Config(ignored_layers=["lm_head"])
        result = cfg.get_quant_method_custom(layer, "lm_head")
        assert isinstance(result, UnquantizedLinearMethod)

    def test_dispatch_flashcomm_linear(self, mxfp8_module):
        from omni_npu.v1.layers.linear import FlashCommLinearBase

        layer = MagicMock(spec=FlashCommLinearBase)
        cfg = mxfp8_module.Mxfp8Config()
        result = cfg.get_quant_method_custom(layer, "model.layer")
        assert isinstance(result, mxfp8_module.Mxfp8FCLinearMethod)

    def test_dispatch_flashcomm_linear_ignored(self, mxfp8_module):
        from omni_npu.v1.layers.linear import (
            FlashCommLinearBase,
            UnquantizedFlashCommLinearMethod,
        )

        layer = MagicMock(spec=FlashCommLinearBase)
        cfg = mxfp8_module.Mxfp8Config(ignored_layers=["lm_head"])
        result = cfg.get_quant_method_custom(layer, "lm_head")
        assert isinstance(result, UnquantizedFlashCommLinearMethod)

    def test_dispatch_fused_mlp(self, mxfp8_module):
        from omni_npu.v1.layers.fused_mlp.layer import FusedMLP

        layer = MagicMock(spec=FusedMLP)
        cfg = mxfp8_module.Mxfp8Config()
        result = cfg.get_quant_method_custom(layer, "model.mlp")
        assert isinstance(result, mxfp8_module.Mxfp8MlpMethod)

    def test_dispatch_fused_moe(self, mxfp8_module, moe_patches):
        from vllm.model_executor.layers.fused_moe import RoutedExperts

        layer = MagicMock(spec=RoutedExperts)
        layer.moe_config = SimpleNamespace(num_experts=4, has_bias=False)
        layer.layer_name = "model.moe"
        cfg = mxfp8_module.Mxfp8Config()
        result = cfg.get_quant_method_custom(layer, "model.moe")
        assert isinstance(result, mxfp8_module.Mxfp8MoEMethod)


# --------------------------------------------------------------------------- #
# Mxfp8LinearMethod tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def linear_method(mxfp8_module):
    return mxfp8_module.Mxfp8LinearMethod(mxfp8_module.Mxfp8Config())


@pytest.fixture
def linear_layer_created(linear_method):
    layer = torch.nn.Module()
    linear_method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[64, 64],
        input_size=128,
        output_size=128,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    return layer


class TestMxfp8LinearMethod:
    def test_init_stores_config(self, mxfp8_module, linear_method):
        assert isinstance(linear_method.quant_config, mxfp8_module.Mxfp8Config)

    def test_create_weights_registers_parameters(self, linear_layer_created):
        layer = linear_layer_created
        assert layer.input_size_per_partition == 128
        assert layer.output_size_per_partition == 128
        assert layer.orig_dtype == torch.bfloat16
        assert layer.weight.shape == (128, 128)
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert layer.weight_scale.shape == (128, 128 // 32)
        assert layer.weight_scale.dtype == torch.uint8

    def test_process_weights_after_loading_repacks(
        self, linear_method, linear_layer_created
    ):
        linear_method.process_weights_after_loading(linear_layer_created)
        # On-disk (O, I) -> kernel (I, O)
        assert linear_layer_created.weight.shape == (128, 128)
        # Scales (O, I//32) -> (I//64, O, 2)
        assert linear_layer_created.weight_scale.shape == (128 // 64, 128, 2)
        assert linear_layer_created.weight.requires_grad is False
        assert linear_layer_created.weight_scale.requires_grad is False

    def test_apply_shape_and_kernel_calls(
        self, mock_torch_npu, linear_method, linear_layer_created
    ):
        linear_method.process_weights_after_loading(linear_layer_created)
        x = torch.zeros(4, 128, dtype=torch.bfloat16)
        out = linear_method.apply(linear_layer_created, x)
        assert out.shape == (4, 128)
        assert out.dtype == torch.bfloat16
        mock_torch_npu.npu_dynamic_mx_quant.assert_called_once()
        mock_torch_npu.npu_quant_matmul.assert_called_once()
        _, kwargs = mock_torch_npu.npu_quant_matmul.call_args
        assert kwargs["group_sizes"] == [1, 1, 32]
        assert kwargs["scale_dtype"] is _E8M0_SENTINEL
        assert kwargs["pertoken_scale_dtype"] is _E8M0_SENTINEL
        assert kwargs["bias"] is None

    def test_apply_with_bias_casts_to_float32(
        self, mock_torch_npu, linear_method, linear_layer_created
    ):
        linear_method.process_weights_after_loading(linear_layer_created)
        x = torch.zeros(4, 128, dtype=torch.bfloat16)
        bias = torch.zeros(128, dtype=torch.bfloat16)
        linear_method.apply(linear_layer_created, x, bias=bias)
        _, kwargs = mock_torch_npu.npu_quant_matmul.call_args
        assert kwargs["bias"] is not None
        assert kwargs["bias"].dtype == torch.float32

    def test_apply_prequantised_dict_skips_dynamic_quant(
        self, mock_torch_npu, linear_method, linear_layer_created
    ):
        linear_method.process_weights_after_loading(linear_layer_created)
        x = {
            "x_mxfp8": torch.zeros(4, 128, dtype=torch.int8),
            "pertoken_scale": torch.zeros(4, 128 // 32, dtype=torch.uint8),
        }
        out = linear_method.apply(linear_layer_created, x)
        assert out.shape == (4, 128)
        mock_torch_npu.npu_dynamic_mx_quant.assert_not_called()
        mock_torch_npu.npu_quant_matmul.assert_called_once()


# --------------------------------------------------------------------------- #
# Mxfp8FCLinearMethod tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def fc_linear_method(mxfp8_module):
    return mxfp8_module.Mxfp8FCLinearMethod(mxfp8_module.Mxfp8Config())


@pytest.fixture
def fc_layer_created(fc_linear_method):
    layer = torch.nn.Module()
    fc_linear_method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[128],
        input_size=128,
        output_size=128,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    layer.layer_name_inside_block = "mlp_layer"
    return layer


class TestMxfp8FCLinearMethod:
    def test_create_weights_registers_parameters(self, fc_layer_created):
        assert fc_layer_created.weight.shape == (128, 128)
        assert fc_layer_created.weight_scale.shape == (128, 128 // 32)
        assert fc_layer_created.orig_dtype == torch.bfloat16

    def test_process_weights_after_loading_repacks(
        self, fc_linear_method, fc_layer_created
    ):
        fc_linear_method.process_weights_after_loading(fc_layer_created)
        assert fc_layer_created.weight.shape == (128, 128)
        assert fc_layer_created.weight_scale.shape == (128 // 64, 128, 2)

    def test_apply_raw_tensor_path(
        self, mock_torch_npu, fc_linear_method, fc_layer_created
    ):
        fc_linear_method.process_weights_after_loading(fc_layer_created)
        x = torch.zeros(4, 128, dtype=torch.bfloat16)
        out = fc_linear_method.apply(fc_layer_created, x)
        assert out.shape == (4, 128)
        mock_torch_npu.npu_dynamic_mx_quant.assert_called_once()
        mock_torch_npu.npu_quant_matmul.assert_called_once()

    def test_apply_prequantised_dict_skips_dynamic_quant(
        self, mock_torch_npu, fc_linear_method, fc_layer_created
    ):
        fc_linear_method.process_weights_after_loading(fc_layer_created)
        x = {
            "x_mxfp8": torch.zeros(4, 128, dtype=torch.int8),
            "pertoken_scale": torch.zeros(4, 128 // 32, dtype=torch.uint8),
        }
        out = fc_linear_method.apply(fc_layer_created, x)
        assert out.shape == (4, 128)
        mock_torch_npu.npu_dynamic_mx_quant.assert_not_called()
        mock_torch_npu.npu_quant_matmul.assert_called_once()

    def test_apply_x_transform_allgather(
        self, mock_torch_npu, fc_linear_method, fc_layer_created
    ):
        fc_linear_method.process_weights_after_loading(fc_layer_created)
        x = torch.zeros(4, 128, dtype=torch.bfloat16)
        ag = MagicMock(side_effect=lambda t, *a, **k: t)
        a2a = MagicMock()
        with patch(
            "omni_npu.v1.distributed.communication_op_ext.layer_parallel_all_gather", ag
        ), patch(
            "omni_npu.v1.distributed.communication_op_ext.layer_parallel_all2all_single",
            a2a,
        ):
            fc_linear_method.apply(fc_layer_created, x, x_transform="AllGather")
        assert ag.call_count == 2  # once for scale, once for fp8
        a2a.assert_not_called()

    def test_apply_x_transform_all2all(
        self, mock_torch_npu, fc_linear_method, fc_layer_created
    ):
        fc_linear_method.process_weights_after_loading(fc_layer_created)
        x = torch.zeros(4, 128, dtype=torch.bfloat16)
        ag = MagicMock()
        a2a = MagicMock(side_effect=lambda t, *a, **k: t)
        with patch(
            "omni_npu.v1.distributed.communication_op_ext.layer_parallel_all_gather", ag
        ), patch(
            "omni_npu.v1.distributed.communication_op_ext.layer_parallel_all2all_single",
            a2a,
        ):
            fc_linear_method.apply(fc_layer_created, x, x_transform="ALL2ALL")
        assert a2a.call_count == 2
        ag.assert_not_called()


# --------------------------------------------------------------------------- #
# Mxfp8MlpMethod tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def mlp_method(mxfp8_module, null_stream_ctx):
    method = mxfp8_module.Mxfp8MlpMethod(mxfp8_module.Mxfp8Config())
    with patch("omni_npu.v1.layers.utils.get_npu_execution_type", null_stream_ctx):
        yield method


class TestMxfp8MlpMethod:
    def test_init_stores_config(self, mxfp8_module, mlp_method):
        assert isinstance(mlp_method.quant_config, mxfp8_module.Mxfp8Config)

    def test_process_weights_after_loading_is_noop(self, mlp_method):
        sentinel = object()
        # No attribute changes; call must not raise.
        mlp_method.process_weights_after_loading(sentinel)

    def test_apply_quant(self, mock_torch_npu, mlp_method):
        x = torch.zeros(4, 128, dtype=torch.bfloat16)
        x_fp8, x_scale = mlp_method.apply_quant(x)
        assert x_fp8.shape == (4, 128)
        assert x_scale.shape == (4, 128 // 32)
        mock_torch_npu.npu_dynamic_mx_quant.assert_called_once()

    def test_apply_part1_gate_up(self, mlp_method):
        layer = _MockMLPLayer(hidden=128, inter=256)
        x = {
            "x_mxfp8": torch.zeros(4, 128, dtype=torch.int8),
            "pertoken_scale": torch.zeros(4, 128 // 32, dtype=torch.uint8),
        }
        out = mlp_method.apply_part1_gate_up_on_stream(layer, x)
        assert out.shape == (4, 512)  # inter * 2

    def test_apply_part2_activation(self, mlp_method):
        layer = _MockMLPLayer()
        gate_up = torch.zeros(4, 512, dtype=torch.bfloat16)
        out = mlp_method.apply_part2_activation_on_stream(layer, gate_up)
        assert isinstance(out, dict)
        assert "x_mxfp8" in out
        assert out["x_mxfp8"].shape == (4, 256)

    def test_apply_part3_down(self, mlp_method):
        layer = _MockMLPLayer(hidden=128, inter=256)
        x = {
            "x_mxfp8": torch.zeros(4, 256, dtype=torch.int8),
            "pertoken_scale": torch.zeros(4, 256 // 32, dtype=torch.uint8),
        }
        out = mlp_method.apply_part3_down_on_stream(layer, x)
        assert out.shape == (4, 128)

    def test_apply_end_to_end(self, mock_torch_npu, mlp_method):
        layer = _MockMLPLayer(hidden=128, inter=256)
        x = torch.zeros(4, 128, dtype=torch.bfloat16)
        out = mlp_method.apply(layer, x)
        assert out.shape == (4, 128)
        mock_torch_npu.npu_dynamic_mx_quant.assert_called_once()


# --------------------------------------------------------------------------- #
# Mxfp8MoEMethod tests
# --------------------------------------------------------------------------- #


class _MockMoELayer:
    """Minimal stand-in for ``FusedMoE`` used to construct ``Mxfp8MoEMethod``."""

    def __init__(self, num_experts=4, has_bias=False):
        self.moe_config = SimpleNamespace(
            num_experts=num_experts,
            has_bias=has_bias,
        )
        self.layer_name = "test_moe"


@pytest.fixture
def moe_patches(mxfp8_module):
    stream_instance = MagicMock()
    with patch(
        "omni_npu.layers.quantization.mxfp8.named_stream",
        MagicMock(return_value=stream_instance),
    ), patch(
        "omni_npu.layers.quantization.mxfp8.get_tensor_model_parallel_world_size",
        return_value=1,
    ), patch(
        "omni_npu.layers.quantization.mxfp8.get_tensor_model_parallel_rank",
        return_value=0,
    ), patch(
        "omni_npu.layers.quantization.mxfp8.get_current_vllm_config",
        return_value=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace())
        ),
    ):
        yield


@pytest.fixture
def moe_method(mxfp8_module, moe_patches):
    layer = _MockMoELayer()
    return mxfp8_module.Mxfp8MoEMethod(mxfp8_module.Mxfp8Config(), layer)


class TestMxfp8MoEMethod:
    def test_init_basic_attributes(self, mxfp8_module, moe_method):
        assert isinstance(moe_method.quant_config, mxfp8_module.Mxfp8Config)
        assert moe_method.tp_size == 1
        assert moe_method.tp_rank == 0
        assert moe_method.n_routed_experts == 4
        assert moe_method.prefix == "test_moe"
        assert moe_method.num_of_redundant_experts == 0

    def test_create_weights_registers_all_tensors(self, moe_method):
        layer = torch.nn.Module()
        layer.moe_config = SimpleNamespace()
        moe_method.create_weights(
            layer,
            num_experts=4,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        assert layer.w13_weight.shape == (4, 128, 128)  # (E, 2*inter, hidden)
        assert layer.w13_weight.dtype == torch.float8_e4m3fn
        assert layer.w2_weight.shape == (4, 128, 64)  # (E, hidden, inter)
        assert layer.w13_weight_scale.shape == (4, 128, 128 // 32)
        assert layer.w13_weight_scale.dtype == torch.uint8
        assert layer.w2_weight_scale.shape == (4, 128, 64 // 32)
        assert layer.w13_input_scale is None
        assert layer.w2_input_scale is None
        assert not hasattr(layer, "w13_bias")

    def test_create_weights_with_bias(self, mxfp8_module, moe_patches):
        layer_wrap = _MockMoELayer(num_experts=4, has_bias=True)
        method = mxfp8_module.Mxfp8MoEMethod(mxfp8_module.Mxfp8Config(), layer_wrap)
        target = torch.nn.Module()
        method.create_weights(
            target,
            num_experts=4,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        assert target.w13_bias.shape == (4, 128)
        assert target.w13_bias.dtype == torch.bfloat16
        assert target.w2_bias.shape == (4, 128)

    def test_create_weights_hidden_size_assertion(self, moe_method):
        layer = torch.nn.Module()
        layer.moe_config = SimpleNamespace()
        with pytest.raises(AssertionError):
            moe_method.create_weights(
                layer,
                num_experts=4,
                hidden_size=33,  # not a multiple of 32
                intermediate_size_per_partition=64,
                params_dtype=torch.bfloat16,
            )

    def test_get_fused_moe_quant_config(self, moe_method):
        layer = torch.nn.Module()
        layer.w13_weight_scale = torch.zeros(4, 128, 128 // 32, dtype=torch.uint8)
        layer.w2_weight_scale = torch.zeros(4, 128, 64 // 32, dtype=torch.uint8)
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        cfg = moe_method.get_fused_moe_quant_config(layer)
        assert cfg._w1.scale is layer.w13_weight_scale
        assert cfg._w2.scale is layer.w2_weight_scale

    def test_process_weights_after_loading_repacks(self, moe_method):
        layer = torch.nn.Module()
        layer.moe_config = SimpleNamespace()
        moe_method.create_weights(
            layer,
            num_experts=4,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        layer.ensure_moe_quant_config_init = MagicMock()
        moe_method.process_weights_after_loading(layer)
        # (E, N13, K) -> (E, K, N13)
        assert layer.w13_weight.shape == (4, 128, 128)
        # (E, N13, K//32) -> (E, K//64, N13, 2)
        assert layer.w13_weight_scale.shape == (4, 128 // 64, 128, 2)
        # w2: (E, K, I) -> (E, I, K)
        assert layer.w2_weight.shape == (4, 64, 128)
        layer.ensure_moe_quant_config_init.assert_called_once()


class TestMxfp8MoEMethodApplyExperts:
    """Covers the ``apply_experts`` hot path without routing scaffolding."""

    def _build_layer(self, moe_method):
        """Return a layer that has been through create_weights + processing."""
        layer = torch.nn.Module()
        layer.moe_config = SimpleNamespace()
        moe_method.create_weights(
            layer,
            num_experts=4,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        layer.ensure_moe_quant_config_init = MagicMock()
        moe_method.process_weights_after_loading(layer)
        return layer

    def _prepare_permute_result(self, dynamic_scale):
        return SimpleNamespace(
            hidden_states_sorted_by_experts=torch.zeros(
                8, 128, dtype=torch.int8
            ),
            expert_tokens=torch.tensor([2, 2, 2, 2], dtype=torch.int64),
            avg_tokens_per_expert=[2],
            dynamic_scale=dynamic_scale,
            row_idx_type=0,
        )

    def test_apply_experts_with_external_scale(self, mock_torch_npu, moe_method):
        layer = self._build_layer(moe_method)
        scale = torch.zeros(8, 128 // 32, dtype=torch.uint8)  # (M, K//32)
        result = moe_method.apply_experts(
            layer, self._prepare_permute_result(scale), activation="silu",
        )
        # Returns (M, K) in bfloat16 when finalize_routing is False
        assert result.shape == (8, 128)
        assert result.dtype == torch.bfloat16
        # Outside quant was supplied, so npu_dynamic_mx_quant was NOT called.
        mock_torch_npu.npu_dynamic_mx_quant.assert_not_called()
        # Two grouped matmuls + one swiglu
        assert mock_torch_npu.npu_grouped_matmul.call_count == 2
        mock_torch_npu.npu_swiglu_mx_quant.assert_called_once()

    def test_apply_experts_dynamic_scale_none_triggers_quant(
        self, mock_torch_npu, moe_method
    ):
        layer = self._build_layer(moe_method)
        result = moe_method.apply_experts(
            layer, self._prepare_permute_result(dynamic_scale=None),
        )
        assert result.shape == (8, 128)
        mock_torch_npu.npu_dynamic_mx_quant.assert_called_once()

    def test_apply_experts_flattens_3d_input(self, mock_torch_npu, moe_method):
        layer = self._build_layer(moe_method)
        pr = SimpleNamespace(
            hidden_states_sorted_by_experts=torch.zeros(2, 4, 128, dtype=torch.int8),
            expert_tokens=torch.tensor([2, 2, 2, 2], dtype=torch.int64),
            avg_tokens_per_expert=None,
            dynamic_scale=torch.zeros(8, 128 // 32, dtype=torch.uint8),
            row_idx_type=0,
        )
        result = moe_method.apply_experts(layer, pr)
        assert result.shape == (8, 128)

    def test_apply_experts_finalize_routing_returns_intermediate(
        self, mock_torch_npu, moe_method
    ):
        layer = self._build_layer(moe_method)
        pr = self._prepare_permute_result(
            torch.zeros(8, 128 // 32, dtype=torch.uint8)
        )
        result = moe_method.apply_experts(
            layer, pr, use_grouped_matmul_finalize_routing=True
        )
        # Returns (intermediate, pertoken_scale) and skips the second matmul
        assert isinstance(result, tuple)
        intermediate, scale = result
        assert intermediate.shape == (8, 64)
        assert mock_torch_npu.npu_grouped_matmul.call_count == 1


# --------------------------------------------------------------------------- #
# Shared helpers for multistream / cv coverage (mirrors test_hifloat8.py)
# --------------------------------------------------------------------------- #


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


def _set_schedule(mxfp8_module, monkeypatch, *, multi_stream, schedule):
    monkeypatch.setattr(
        mxfp8_module,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                shared_expert_multi_stream=multi_stream,
                shared_expert_parallel_schedule=schedule,
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# apply_experts cv-block tests
# --------------------------------------------------------------------------- #


class TestMxfp8MoEMethodApplyExpertsCV:
    """Covers the ``run_shared_with_cv`` branch in ``apply_experts``."""

    def _build_layer(self, moe_method, shared_experts):
        layer = torch.nn.Module()
        layer.moe_config = SimpleNamespace()
        moe_method.create_weights(
            layer,
            num_experts=4,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        layer.ensure_moe_quant_config_init = MagicMock()
        moe_method.process_weights_after_loading(layer)
        layer.shared_experts = shared_experts
        layer._shared_experts = SimpleNamespace(_layer=shared_experts)
        return layer

    def test_apply_experts_cv_returns_routed_and_shared_tuple(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        main_stream = _patch_npu_streams(monkeypatch)
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=True, schedule="with_routed_experts_cv",
        )
        moe_method.shared_experts_stream = _DummyStream()

        shared_down_out = torch.full((8, 128), 9.0, dtype=torch.bfloat16)
        shared_experts = SimpleNamespace(
            act_fn=MagicMock(side_effect=lambda x: x),
            down_proj=MagicMock(return_value=shared_down_out),
        )
        layer = self._build_layer(moe_method, shared_experts)
        finished_event = torch.npu.Event()
        pr = SimpleNamespace(
            hidden_states_sorted_by_experts=torch.zeros(8, 128, dtype=torch.int8),
            expert_tokens=torch.tensor([2, 2, 2, 2], dtype=torch.int64),
            avg_tokens_per_expert=[2],
            dynamic_scale=torch.zeros(8, 128 // 32, dtype=torch.uint8),
            row_idx_type=0,
            shared_expert_gate_up=torch.zeros(8, 256, dtype=torch.bfloat16),
            shared_expert_finished_event=finished_event,
        )

        result = moe_method.apply_experts(layer, pr)

        # Returns (routed, shared) tuple.
        assert isinstance(result, tuple)
        routed_out, shared_out = result
        assert routed_out.shape == (8, 128)
        assert torch.equal(shared_out, shared_down_out)
        # Side-stream synchronisation actually happened.
        assert finished_event.waits == [main_stream]
        shared_experts.act_fn.assert_called_once()
        shared_experts.down_proj.assert_called_once()
        # apply_experts called npu_dynamic_mx_quant once for the shared
        # activation (no other call: pertoken_scale was supplied).
        mock_torch_npu.npu_dynamic_mx_quant.assert_called_once()
        # Main stream waits on the side stream at least twice (around
        # down_proj launch and at the end).
        assert main_stream.wait_stream_calls.count(moe_method.shared_experts_stream) >= 2

    def test_apply_experts_cv_finalize_routing_skips_w2(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=True, schedule="with_routed_experts_cv",
        )
        moe_method.shared_experts_stream = _DummyStream()

        shared_experts = SimpleNamespace(
            act_fn=MagicMock(side_effect=lambda x: x),
            down_proj=MagicMock(return_value=torch.zeros(8, 128, dtype=torch.bfloat16)),
        )
        layer = self._build_layer(moe_method, shared_experts)
        pr = SimpleNamespace(
            hidden_states_sorted_by_experts=torch.zeros(8, 128, dtype=torch.int8),
            expert_tokens=torch.tensor([2, 2, 2, 2], dtype=torch.int64),
            avg_tokens_per_expert=[2],
            dynamic_scale=torch.zeros(8, 128 // 32, dtype=torch.uint8),
            row_idx_type=0,
            shared_expert_gate_up=torch.zeros(8, 256, dtype=torch.bfloat16),
            shared_expert_finished_event=torch.npu.Event(),
        )

        intermediate, scale = moe_method.apply_experts(
            layer, pr, use_grouped_matmul_finalize_routing=True,
        )
        # Only one grouped_matmul (w13) — w2 was skipped.
        assert mock_torch_npu.npu_grouped_matmul.call_count == 1
        assert intermediate.shape == (8, 64)


# --------------------------------------------------------------------------- #
# Mxfp8MoEMethod.apply schedule tests
# --------------------------------------------------------------------------- #


def _build_apply_method(mxfp8_module, moe_method, monkeypatch):
    """Stub the inner machinery so apply() exercises only the schedule branching."""
    moe_method.tp_size = 1
    moe_method.tp_rank = 0
    moe_method.shared_experts_stream = _DummyStream()
    moe_method.select_communication_strategy = MagicMock(
        return_value=("agrs", SimpleNamespace())
    )
    moe_method.apply_prepare_permute = MagicMock(
        return_value=SimpleNamespace(row_idx_type=0)
    )
    moe_method.apply_unpermute_finalize = MagicMock(
        return_value=torch.zeros(2, 128, dtype=torch.bfloat16)
    )
    fake_npu_moe = SimpleNamespace(
        select_experts=lambda **_kwargs: (
            torch.ones(2, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(mxfp8_module, "NPUFusedMoE", fake_npu_moe)
    return moe_method


class TestMxfp8MoEMethodApply:
    """Covers the .apply schedule branching (cv / with_finalize / synchronous)."""

    def test_apply_cv_unpacks_apply_experts_tuple(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=True, schedule="with_routed_experts_cv",
        )

        moe_method = _build_apply_method(mxfp8_module, moe_method, monkeypatch)
        routed = torch.zeros(2, 128, dtype=torch.bfloat16)
        shared = torch.full((2, 128), 7.0, dtype=torch.bfloat16)
        moe_method.apply_experts = MagicMock(return_value=(routed, shared))

        shared_experts = MagicMock()
        shared_experts.gate_up_proj = SimpleNamespace(tp_size=1)
        layer = SimpleNamespace(
            gate=lambda x: (torch.zeros(x.shape[0], 4), None),
            shared_experts=shared_experts,
        )
        hidden = torch.ones(2, 128, dtype=torch.bfloat16)

        out = moe_method.apply(
            layer=layer,
            hidden_states=hidden,
            router_logits=None,
            top_k=1,
            renormalize=False,
        )
        # Tuple unpacking happened: apply did not also call shared_experts itself.
        shared_experts.assert_not_called()
        assert isinstance(out, tuple)
        shared_out, routed_out = out
        assert torch.equal(shared_out, shared)
        assert torch.equal(routed_out, moe_method.apply_unpermute_finalize.return_value)

    def test_apply_with_finalize_tp_gt1_all_reduces_shared(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=True, schedule="with_finalize",
        )
        moe_method = _build_apply_method(mxfp8_module, moe_method, monkeypatch)
        moe_method.apply_experts = MagicMock(
            return_value=torch.zeros(2, 128, dtype=torch.bfloat16)
        )

        shared_out = torch.full((2, 128), 5.0, dtype=torch.bfloat16)
        shared_experts = MagicMock(return_value=shared_out)
        shared_experts.gate_up_proj = SimpleNamespace(tp_size=2)
        all_reduce_calls = []

        def fake_all_reduce(t):
            all_reduce_calls.append(t)
            return t * 2

        monkeypatch.setattr(
            mxfp8_module, "tensor_model_parallel_all_reduce", fake_all_reduce
        )

        layer = SimpleNamespace(
            gate=lambda x: (torch.zeros(x.shape[0], 4), None),
            shared_experts=shared_experts,
        )
        hidden = torch.ones(2, 128, dtype=torch.bfloat16)
        out = moe_method.apply(
            layer=layer,
            hidden_states=hidden,
            router_logits=None,
            top_k=1,
            renormalize=False,
        )

        # Shared MLP fed full hidden_states (tp>1 path) and all-reduced once.
        shared_experts.assert_called_once()
        assert torch.equal(shared_experts.call_args.args[0], hidden)
        assert len(all_reduce_calls) == 1
        assert torch.equal(all_reduce_calls[0], shared_out)
        assert isinstance(out, tuple)

    def test_apply_with_finalize_tp1_uses_x_slice(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=True, schedule="with_finalize",
        )
        moe_method = _build_apply_method(mxfp8_module, moe_method, monkeypatch)
        moe_method.apply_experts = MagicMock(
            return_value=torch.zeros(2, 128, dtype=torch.bfloat16)
        )

        shared_experts = MagicMock(
            return_value=torch.full((2, 128), 3.0, dtype=torch.bfloat16)
        )
        shared_experts.gate_up_proj = SimpleNamespace(tp_size=1)
        layer = SimpleNamespace(
            gate=lambda x: (torch.zeros(x.shape[0], 4), None),
            shared_experts=shared_experts,
        )
        hidden = torch.ones(2, 128, dtype=torch.bfloat16)
        moe_method.apply(
            layer=layer,
            hidden_states=hidden,
            router_logits=None,
            top_k=1,
            renormalize=False,
        )
        shared_experts.assert_called_once()
        # tp_size==1 → x_slice == hidden_states.
        assert torch.equal(shared_experts.call_args.args[0], hidden)

    def test_apply_synchronous_runs_shared_on_main_stream(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=False, schedule="with_finalize",
        )
        moe_method = _build_apply_method(mxfp8_module, moe_method, monkeypatch)
        moe_method.apply_experts = MagicMock(
            return_value=torch.zeros(2, 128, dtype=torch.bfloat16)
        )

        shared_experts = MagicMock(
            return_value=torch.full((2, 128), 2.0, dtype=torch.bfloat16)
        )
        shared_experts.gate_up_proj = SimpleNamespace(tp_size=1)
        layer = SimpleNamespace(
            gate=lambda x: (torch.zeros(x.shape[0], 4), None),
            shared_experts=shared_experts,
        )
        hidden = torch.ones(2, 128, dtype=torch.bfloat16)
        out = moe_method.apply(
            layer=layer,
            hidden_states=hidden,
            router_logits=None,
            top_k=1,
            renormalize=False,
        )
        # Sync branch hit: shared called, no side-stream wait recorded.
        shared_experts.assert_called_once()
        assert moe_method.shared_experts_stream.wait_stream_calls == []
        assert isinstance(out, tuple)

    def test_apply_non_multistream_with_finalize_tp_gt1_all_reduces_shared(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=False, schedule="with_finalize",
        )
        moe_method = _build_apply_method(mxfp8_module, moe_method, monkeypatch)
        moe_method.apply_experts = MagicMock(
            return_value=torch.zeros(2, 128, dtype=torch.bfloat16)
        )

        shared = torch.full((2, 128), 3.0, dtype=torch.bfloat16)
        reduced_shared = shared * 2
        monkeypatch.setattr(
            mxfp8_module,
            "tensor_model_parallel_all_reduce",
            MagicMock(return_value=reduced_shared),
        )
        shared_experts = MagicMock(return_value=shared)
        shared_experts.gate_up_proj = SimpleNamespace(tp_size=2)
        layer = SimpleNamespace(
            gate=lambda x: (torch.zeros(x.shape[0], 4), None),
            shared_experts=shared_experts,
        )

        shared_out, routed_out = moe_method.apply(
            layer=layer,
            hidden_states=torch.ones(2, 128, dtype=torch.bfloat16),
            router_logits=None,
            top_k=1,
            renormalize=False,
        )

        assert torch.equal(shared_out, reduced_shared)
        assert torch.equal(routed_out, moe_method.apply_unpermute_finalize.return_value)
        mxfp8_module.tensor_model_parallel_all_reduce.assert_called_once()
        assert mxfp8_module.tensor_model_parallel_all_reduce.call_args.args[0] is shared

    def test_apply_non_multistream_with_finalize_custom_model_adds_shared_output(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=False, schedule="with_finalize",
        )
        moe_method = _build_apply_method(mxfp8_module, moe_method, monkeypatch)
        routed = torch.full((2, 128), 1.0, dtype=torch.bfloat16)
        moe_method.apply_unpermute_finalize = MagicMock(return_value=routed)
        moe_method.apply_experts = MagicMock(
            return_value=torch.zeros(2, 128, dtype=torch.bfloat16)
        )

        shared = torch.full((2, 128), 3.0, dtype=torch.bfloat16)
        shared_experts = MagicMock(return_value=shared)
        shared_experts.gate_up_proj = SimpleNamespace(tp_size=1)
        layer = SimpleNamespace(
            gate=lambda x: (torch.zeros(x.shape[0], 4), None),
            shared_experts=shared_experts,
        )

        shared_out, routed_out = moe_method.apply(
            layer=layer,
            hidden_states=torch.ones(2, 128, dtype=torch.bfloat16),
            router_logits=None,
            top_k=1,
            renormalize=False,
        )

        assert torch.equal(shared_out, shared)
        assert torch.equal(routed_out, routed + shared)

    def test_apply_with_finalize_custom_model_adds_shared_output(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=True, schedule="with_finalize",
        )
        moe_method = _build_apply_method(mxfp8_module, moe_method, monkeypatch)
        routed = torch.full((2, 128), 1.0, dtype=torch.bfloat16)
        moe_method.apply_unpermute_finalize = MagicMock(return_value=routed)
        moe_method.apply_experts = MagicMock(
            return_value=torch.zeros(2, 128, dtype=torch.bfloat16)
        )

        shared = torch.full((2, 128), 3.0, dtype=torch.bfloat16)
        shared_experts = MagicMock(return_value=shared)
        shared_experts.gate_up_proj = SimpleNamespace(tp_size=1)
        layer = SimpleNamespace(
            gate=lambda x: (torch.zeros(x.shape[0], 4), None),
            shared_experts=shared_experts,
        )

        shared_out, routed_out = moe_method.apply(
            layer=layer,
            hidden_states=torch.ones(2, 128, dtype=torch.bfloat16),
            router_logits=None,
            top_k=1,
            renormalize=False,
        )

        assert torch.equal(shared_out, shared)
        assert torch.equal(routed_out, routed + shared)

    def test_apply_with_finalize_custom_model_adds_tp_shared_output(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=True, schedule="with_finalize",
        )
        moe_method = _build_apply_method(mxfp8_module, moe_method, monkeypatch)
        routed = torch.full((2, 128), 1.0, dtype=torch.bfloat16)
        moe_method.apply_unpermute_finalize = MagicMock(return_value=routed)
        moe_method.apply_experts = MagicMock(
            return_value=torch.zeros(2, 128, dtype=torch.bfloat16)
        )

        shared = torch.full((2, 128), 3.0, dtype=torch.bfloat16)
        reduced_shared = shared * 2
        monkeypatch.setattr(
            mxfp8_module,
            "tensor_model_parallel_all_reduce",
            MagicMock(return_value=reduced_shared),
        )
        shared_experts = MagicMock(return_value=shared)
        shared_experts.gate_up_proj = SimpleNamespace(tp_size=2)
        layer = SimpleNamespace(
            gate=lambda x: (torch.zeros(x.shape[0], 4), None),
            shared_experts=shared_experts,
        )

        shared_out, routed_out = moe_method.apply(
            layer=layer,
            hidden_states=torch.ones(2, 128, dtype=torch.bfloat16),
            router_logits=None,
            top_k=1,
            renormalize=False,
        )

        assert torch.equal(shared_out, reduced_shared)
        assert torch.equal(routed_out, routed + reduced_shared)
        mxfp8_module.tensor_model_parallel_all_reduce.assert_called_once()
        assert mxfp8_module.tensor_model_parallel_all_reduce.call_args.args[0] is shared

    def test_apply_with_prefetch_runs_moe_and_next_attn_prefetch(
        self, mock_torch_npu, moe_method, mxfp8_module, monkeypatch
    ):
        _patch_npu_streams(monkeypatch)
        _set_schedule(
            mxfp8_module, monkeypatch,
            multi_stream=False, schedule="with_finalize",
        )
        monkeypatch.setattr(
            mxfp8_module.model_extra_config.operator_opt_config,
            "enable_prefetch", True,
            raising=False,
        )
        moe_method = _build_apply_method(mxfp8_module, moe_method, monkeypatch)
        moe_method.apply_experts = MagicMock(
            return_value=torch.zeros(2, 128, dtype=torch.bfloat16)
        )
        moe_method.model_prefetch = SimpleNamespace(prefetch=MagicMock())

        shared_experts = MagicMock(
            return_value=torch.full((2, 128), 2.0, dtype=torch.bfloat16)
        )
        shared_experts.gate_up_proj = SimpleNamespace(tp_size=1)
        layer = SimpleNamespace(
            gate=lambda x: (torch.zeros(x.shape[0], 4), None),
            shared_experts=shared_experts,
        )

        out = moe_method.apply(
            layer=layer,
            hidden_states=torch.ones(2, 128, dtype=torch.bfloat16),
            router_logits=None,
            top_k=1,
            renormalize=False,
        )

        assert isinstance(out, tuple)
        assert moe_method.model_prefetch.prefetch.call_count == 2
        first_call, second_call = moe_method.model_prefetch.prefetch.call_args_list
        assert first_call.args[0] == "moe"
        assert second_call.args[0] == "next_attn"
        assert second_call.kwargs["layer"] is layer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
