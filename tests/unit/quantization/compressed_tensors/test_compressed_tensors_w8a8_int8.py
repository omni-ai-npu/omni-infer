# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from unittest.mock import MagicMock

import pytest
import torch
from compressed_tensors.quantization import QuantizationStrategy

import vllm.model_executor.parameter as parameter_module
from omni_npu.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    NPUCompressedTensorsW8A8Int8,
)


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )


def _create_layer(strategy=QuantizationStrategy.CHANNEL, params_dtype=torch.bfloat16):
    layer = torch.nn.Module()
    scheme = NPUCompressedTensorsW8A8Int8(
        strategy=strategy,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[2, 3],
        input_size_per_partition=4,
        params_dtype=params_dtype,
        weight_loader=lambda *_args, **_kwargs: None,
    )
    return scheme, layer


def test_config_and_min_capability():
    scheme = NPUCompressedTensorsW8A8Int8(
        strategy=QuantizationStrategy.TENSOR,
        is_static_input_scheme=True,
        input_symmetric=False,
    )

    assert scheme.strategy == QuantizationStrategy.TENSOR
    assert scheme.is_static_input_scheme is True
    assert scheme.input_symmetric is False
    assert scheme.get_min_capability() == 75


@pytest.mark.parametrize(
    ("strategy", "params_dtype", "scale_shape", "has_offset"),
    [
        (QuantizationStrategy.TENSOR, torch.float16, (2,), False),
        (QuantizationStrategy.CHANNEL, torch.bfloat16, (5, 1), True),
    ],
)
def test_create_weights(strategy, params_dtype, scale_shape, has_offset):
    scheme, layer = _create_layer(strategy, params_dtype)

    assert scheme.logical_widths == [2, 3]
    assert layer.weight.shape == (5, 4)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == scale_shape
    assert (layer.weight_offset is not None) is has_offset
    assert scheme.empty_out.dtype == params_dtype


@pytest.mark.parametrize("throw_dequant", [False, True])
def test_process_weights_after_loading(monkeypatch, throw_dequant):
    _, layer = _create_layer()
    layer.throw_dequant = throw_dequant
    format_cast = MagicMock(side_effect=lambda weight, _fmt: weight)
    monkeypatch.setattr(torch.ops, "npu", MagicMock(), raising=False)

    import torch_npu

    monkeypatch.setattr(torch_npu, "npu_format_cast", format_cast)
    original_weight = layer.weight.detach().clone()

    scheme = NPUCompressedTensorsW8A8Int8(
        QuantizationStrategy.CHANNEL, False, True
    )
    scheme.process_weights_after_loading(layer)

    assert torch.equal(layer.weight, original_weight.t())
    assert layer.weight_scale.shape == (5,)
    assert layer.weight_offset.shape == (5,)
    assert layer.weight_offset.dtype == torch.float32
    format_cast.assert_called_once()


def test_apply_weights_dynamic_quant_and_squeeze(monkeypatch):
    scheme, layer = _create_layer()
    layer.weight = torch.nn.Parameter(layer.weight.data.t(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        layer.weight_scale.data.view(-1), requires_grad=False
    )

    import torch_npu

    dynamic_quant = MagicMock(
        return_value=(torch.ones(2, 4, dtype=torch.int8), torch.ones(2))
    )
    quant_matmul = MagicMock(return_value=torch.ones(2, 5, dtype=torch.bfloat16))
    monkeypatch.setattr(torch_npu, "npu_dynamic_quant", dynamic_quant)
    monkeypatch.setattr(torch_npu, "npu_quant_matmul", quant_matmul)

    output = scheme.apply_weights(layer, torch.ones(2, 1, 4), bias=torch.ones(5))

    assert output.shape == (2, 1, 5)
    dynamic_quant.assert_called_once()
    assert quant_matmul.call_args.kwargs["output_dtype"] == torch.bfloat16


def test_apply_weights_dict_throw_dequant(monkeypatch):
    scheme, layer = _create_layer()
    layer.throw_dequant = True

    import torch_npu

    quant_matmul = MagicMock(return_value=torch.ones(2, 5, dtype=torch.int32))
    monkeypatch.setattr(torch_npu, "npu_quant_matmul", quant_matmul)
    scale = torch.ones(2)

    output, returned_scale = scheme.apply_weights(
        layer,
        {"x_int8": torch.ones(2, 4, dtype=torch.int8), "pertoken_scale": scale},
        bias=None,
    )

    assert output.dtype == torch.int32
    assert returned_scale is scale
    assert quant_matmul.call_args.kwargs["output_dtype"] == torch.int32
