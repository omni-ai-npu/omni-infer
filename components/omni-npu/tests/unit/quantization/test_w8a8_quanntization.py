# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class MockQuantConfig:
    def __init__(self):
        self.some_config = "test_config"


class MockLinearLayer:
    def __init__(self, in_features, out_features, tp_rank=0):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_scale = torch.nn.Parameter(
            torch.randn(out_features, dtype=torch.float32),
            requires_grad=False,
        )
        self.tp_rank = tp_rank
        self.layer_name_inside_block = "mlp_layer"

    def __call__(self, x, throw_dequant=False):
        if isinstance(x, dict):
            batch = x["x_int8"].shape[0]
            x_scale = x["pertoken_scale"]
        else:
            batch = x.shape[0]
            x_scale = torch.ones(batch, dtype=torch.float32)
        output = torch.randint(
            -127,
            127,
            (batch, self.out_features),
            dtype=torch.int32,
        )
        if throw_dequant:
            return (output, x_scale), None
        return output, x_scale


class MockSiluAndMul:
    def __call__(self, x, quant_symbol=False):
        return x


class MockMLPLayer:
    def __init__(self):
        self.gate_up_proj = MockLinearLayer(128, 256)
        self.down_proj = MockLinearLayer(256, 128)
        self.act_fn = MockSiluAndMul()
        self.layer_name_inside_block = "mlp_layer"


def mock_get_npu_execution_type(stream_label):
    class MockContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None
    return MockContext()

def mock_npu_dynamic_quant(x):
    x_quant = torch.clamp(x, -127, 127).to(dtype=torch.int8)
    x_scale = torch.ones(x.shape[0], dtype=torch.float32)
    return x_quant, x_scale


@pytest.fixture
def mock_dependencies():
    mock_torch_npu = SimpleNamespace(npu_dynamic_quant=mock_npu_dynamic_quant)
    with patch(
        "omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear.torch_npu",
        mock_torch_npu,
    ), patch(
        "omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear.get_npu_execution_type",
        mock_get_npu_execution_type,
    ):
        yield mock_torch_npu


@pytest.fixture
def w8a8_mlp_method(mock_dependencies):
    quant_config = MockQuantConfig()
    # 延迟导入真实类，避免 pytest 收集阶段循环导入
    compressed = importlib.import_module(
        "omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear"
    )
    W8A8Int8MlpMethod = getattr(compressed, "W8A8Int8MlpMethod")
    return W8A8Int8MlpMethod(quant_config)


@pytest.fixture
def mock_mlp_layer():
    return MockMLPLayer()


@pytest.fixture
def dummy_tensor():
    return torch.randn(32, 128, dtype=torch.float32)

class TestW8A8Int8MlpMethod:
    def test_init(self, w8a8_mlp_method):
        assert hasattr(w8a8_mlp_method, 'quant_config')
        assert w8a8_mlp_method.quant_config.some_config == "test_config"

    def test_process_weights_after_loading(self, w8a8_mlp_method, mock_mlp_layer):
        w8a8_mlp_method.process_weights_after_loading(mock_mlp_layer)
        assert mock_mlp_layer.gate_up_proj.weight_scale.dtype == torch.float32
        assert not mock_mlp_layer.gate_up_proj.weight_scale.requires_grad

    def test_apply_quant(self, w8a8_mlp_method, dummy_tensor):
        x_quant, x_scale = w8a8_mlp_method.apply_quant(dummy_tensor)
        assert x_quant.dtype == torch.int8
        assert x_scale.dtype == torch.float32
        assert x_quant.shape == dummy_tensor.shape
        assert x_scale.shape[0] == dummy_tensor.shape[0]

    def test_apply_part1_gate_up_on_stream(self, w8a8_mlp_method, mock_mlp_layer, dummy_tensor):
        x_quant, x_scale = w8a8_mlp_method.apply_quant(dummy_tensor)
        x = {"x_int8": x_quant, "pertoken_scale": x_scale}
        gate_up = w8a8_mlp_method.apply_part1_gate_up_on_stream(
            mock_mlp_layer, x
        )
        assert gate_up[0].dtype == torch.int32
        assert gate_up[0].shape == (dummy_tensor.shape[0], 256)
        assert gate_up[1].shape == (dummy_tensor.shape[0],)

    def test_apply_part2_activation_on_stream(self, w8a8_mlp_method, mock_mlp_layer):
        y_int32 = torch.randint(-127, 127, (32, 256), dtype=torch.int32)
        x_scale = torch.ones(32, dtype=torch.float32)
        x = w8a8_mlp_method.apply_part2_activation_on_stream(
            mock_mlp_layer, (y_int32, x_scale)
        )
        assert x["x_int8"].shape == y_int32.shape
        assert x["pertoken_scale"].shape == x_scale.shape

    def test_apply_part3_down_on_stream(self, w8a8_mlp_method, mock_mlp_layer):
        int_int32 = torch.randint(-127, 127, (32, 256), dtype=torch.int32)
        int_scale = torch.ones(32, dtype=torch.float32)
        x = {"x_int8": int_int32, "pertoken_scale": int_scale}
        output = w8a8_mlp_method.apply_part3_down_on_stream(
            mock_mlp_layer, x
        )
        assert output.shape == (int_int32.shape[0], 128)

    def test_apply_full_flow(self, w8a8_mlp_method, mock_mlp_layer, dummy_tensor):
        output = w8a8_mlp_method.apply(mock_mlp_layer, dummy_tensor)
        assert output.shape == (dummy_tensor.shape[0], 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])