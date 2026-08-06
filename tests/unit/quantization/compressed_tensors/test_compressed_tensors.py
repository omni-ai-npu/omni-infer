# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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


class DummyQuantizationArgs:
    def __init__(self, num_bits, strategy, dynamic=False, symmetric=True):
        self.num_bits = num_bits
        self.strategy = strategy
        self.dynamic = dynamic
        self.symmetric = symmetric

    @classmethod
    def parse_obj(cls, obj):
        return cls(
            num_bits=obj.get("num_bits"),
            strategy=obj.get("strategy"),
            dynamic=obj.get("dynamic", False),
            symmetric=obj.get("symmetric", True),
        )


class DummyQuantizationStrategy:
    TENSOR = SimpleNamespace(value="tensor")
    CHANNEL = SimpleNamespace(value="channel")
    GROUP = SimpleNamespace(value="group")
    TOKEN = SimpleNamespace(value="token")


class DummyCompressedTensorsConfig:
    def __init__(self):
        self.quant_format = "activation"
        self.target_scheme_map = {}
        self.ignore = []
        self.packed_modules_mapping = {}


class DummyCompressedTensorsLinearMethod:
    def __init__(self, config):
        self.config = config


class DummyCompressedTensorsScheme:
    pass


class DummyNPUCompressedTensorsW8A8Int8(DummyCompressedTensorsScheme):
    def __init__(self, strategy, is_static_input_scheme, input_symmetric):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric


class DummyNPUCompressedTensorsW8A8Int8MoEMethod:
    def __init__(self, weight_quant, parent, layer):
        self.weight_quant = weight_quant
        self.parent = parent
        self.layer = layer


class DummyNPUCompressedTensorsW4A8Int4MoEMethod:
    def __init__(self, weight_quant, parent, layer):
        self.weight_quant = weight_quant
        self.parent = parent
        self.layer = layer


class DummyQuantizeMethodBase:
    pass


class DummyLinearMethodBase:
    pass


class DummyUnquantizedLinearMethod(DummyLinearMethodBase):
    pass


class DummyLinearBase(torch.nn.Module):
    pass


class DummyFusedMoE(torch.nn.Module):
    pass


class DummyUnquantizedFlashCommLinearMethod:
    pass


class DummyW8A8Int8FCLinearMethod:
    def __init__(self, config):
        self.config = config


class DummyFlashCommLinearBase(torch.nn.Module):
    pass


class DummyUnquantizedShardedLinearMethod(torch.nn.Module):
    pass


class DummyW8A8Int8ShardedLinearMethod(torch.nn.Module):
    pass


class DummyShardedLinear(torch.nn.Module):
    pass


class DummyFusedMLP(torch.nn.Module):
    pass


class DummyW8A8Int8MlpMethod:
    def __init__(self, config):
        self.config = config


def _make_config_instance(module):
    config = module.NPUCompressedTensorsConfig.__new__(module.NPUCompressedTensorsConfig)
    config.quant_format = "activation"
    config.target_scheme_map = {}
    config.ignore = []
    config.packed_modules_mapping = {}
    return config


@pytest.fixture
def mock_dependencies(monkeypatch: pytest.MonkeyPatch):
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(
            torch, "npu", SimpleNamespace(is_available=lambda: True), raising=False
        )
    if not hasattr(torch.npu, "is_available"):
        torch.npu.is_available = lambda: True

    compressed_tensors_module = _make_package(monkeypatch, "compressed_tensors")
    quant_module = _make_module(monkeypatch, "compressed_tensors.quantization")
    quant_module.QuantizationArgs = DummyQuantizationArgs
    quant_module.QuantizationStrategy = DummyQuantizationStrategy
    compressed_tensors_module.quantization = quant_module

    vllm_module = _make_package(monkeypatch, "vllm")
    model_executor_module = _make_package(monkeypatch, "vllm.model_executor")
    layers_module = _make_package(monkeypatch, "vllm.model_executor.layers")
    quantization_module = _make_package(
        monkeypatch, "vllm.model_executor.layers.quantization"
    )

    def _register_quantization_config(name):
        def _decorator(cls):
            return cls
        return _decorator

    quantization_module.register_quantization_config = _register_quantization_config

    base_config_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.quantization.base_config"
    )
    base_config_module.QuantizeMethodBase = DummyQuantizeMethodBase

    ct_module = _make_module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
    )
    ct_module.QUANTIZATION_SCHEME_MAP_TYPE = dict
    ct_module.CompressedTensorsConfig = DummyCompressedTensorsConfig
    ct_module.CompressedTensorsLinearMethod = DummyCompressedTensorsLinearMethod

    schemes_module = _make_module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes",
    )
    schemes_module.CompressedTensorsScheme = DummyCompressedTensorsScheme

    utils_module = _make_module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.utils",
    )

    def _find_matched_target(layer_name, module, targets, fused_mapping=None):
        if layer_name in targets:
            return layer_name
        if "Linear" in targets:
            return "Linear"
        return next(iter(targets))

    def _is_activation_quantization_format(format_name):
        return format_name == "activation"

    def _should_ignore_layer(layer_name, ignore, fused_mapping=None):
        return layer_name in (ignore or [])

    utils_module.find_matched_target = _find_matched_target
    utils_module.is_activation_quantization_format = _is_activation_quantization_format
    utils_module.should_ignore_layer = _should_ignore_layer

    fused_moe_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.layer"
    )
    fused_moe_module.FusedMoE = DummyFusedMoE

    linear_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.linear"
    )
    linear_module.LinearBase = DummyLinearBase
    linear_module.LinearMethodBase = DummyLinearMethodBase
    linear_module.UnquantizedLinearMethod = DummyUnquantizedLinearMethod

    vllm_module.model_executor = model_executor_module
    model_executor_module.layers = layers_module
    layers_module.quantization = quantization_module

    npu_moe_module = _make_module(
        monkeypatch,
        "omni_npu.layers.quantization.compressed_tensors.compressed_tensors_moe",
    )
    npu_moe_module.NPUCompressedTensorsW8A8Int8MoEMethod = (
        DummyNPUCompressedTensorsW8A8Int8MoEMethod
    )
    npu_moe_module.NPUCompressedTensorsW4A8Int4MoEMethod = (
        DummyNPUCompressedTensorsW4A8Int4MoEMethod
    )

    npu_scheme_module = _make_module(
        monkeypatch,
        "omni_npu.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8",
    )
    npu_scheme_module.NPUCompressedTensorsW8A8Int8 = DummyNPUCompressedTensorsW8A8Int8

    v1_linear_module = _make_module(monkeypatch, "omni_npu.v1.layers.linear")
    v1_linear_module.UnquantizedFlashCommLinearMethod = (
        DummyUnquantizedFlashCommLinearMethod
    )
    v1_linear_module.UnquantizedShardedLinearMethod = DummyUnquantizedShardedLinearMethod
    v1_linear_module.FlashCommLinearBase = DummyFlashCommLinearBase
    v1_linear_module.ShardedLinear = DummyShardedLinear

    v1_ct_linear_module = _make_module(
        monkeypatch,
        "omni_npu.v1.layers.quantization.compressed_tensors.npu_compressed_tensors_linear",
    )
    v1_ct_linear_module.W8A8Int8FCLinearMethod = DummyW8A8Int8FCLinearMethod
    v1_ct_linear_module.W8A8Int8ShardedLinearMethod = DummyW8A8Int8ShardedLinearMethod

    v1_fused_mlp_module = _make_module(monkeypatch, "omni_npu.v1.layers.fused_mlp.layer")
    v1_fused_mlp_module.FusedMLP = DummyFusedMLP

    v1_ct_linear_module.W8A8Int8MlpMethod = DummyW8A8Int8MlpMethod

    base_path = Path(__file__).resolve().parents[4]
    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(base_path / "omni")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    layers_pkg = types.ModuleType("omni_npu.layers")
    layers_pkg.__path__ = [str(base_path / "omni" / "layers")]
    monkeypatch.setitem(sys.modules, "omni_npu.layers", layers_pkg)
    quant_pkg = types.ModuleType("omni_npu.layers.quantization")
    quant_pkg.__path__ = [
        str(base_path / "omni" / "layers" / "quantization")
    ]
    monkeypatch.setitem(sys.modules, "omni_npu.layers.quantization", quant_pkg)
    ct_pkg = types.ModuleType("omni_npu.layers.quantization.compressed_tensors")
    ct_pkg.__path__ = [
        str(
            base_path
            / "omni"
            / "layers"
            / "quantization"
            / "compressed_tensors"
        )
    ]
    monkeypatch.setitem(
        sys.modules, "omni_npu.layers.quantization.compressed_tensors", ct_pkg
    )

    sys.modules.pop(
        "omni_npu.layers.quantization.compressed_tensors.compressed_tensors",
        None,
    )
    yield


@pytest.fixture
def compressed_tensors_module(mock_dependencies):
    module = importlib.import_module(
        "omni_npu.layers.quantization.compressed_tensors.compressed_tensors"
    )
    importlib.reload(module)
    return module


def _make_w8a8_quant(num_bits=8, strategy="tensor", dynamic=False, symmetric=True):
    return DummyQuantizationArgs(
        num_bits=num_bits, strategy=strategy, dynamic=dynamic, symmetric=symmetric
    )

def _make_w4a8_quant(num_bits=dict(), strategy="tensor", dynamic=False, symmetric=True):
    return DummyQuantizationArgs(
        num_bits=num_bits, strategy=strategy, dynamic=dynamic, symmetric=symmetric
    )

class TestNPUCompressedTensorsConfig:
    def test_quantization_scheme_map_from_config(self, compressed_tensors_module):
        config = {
            "config_groups": {
                "group": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 8, "strategy": "tensor"},
                    "input_activations": {
                        "num_bits": 8,
                        "strategy": "token",
                        "dynamic": True,
                        "symmetric": True,
                    },
                }
            }
        }
        target_map = compressed_tensors_module.NPUCompressedTensorsConfig._quantization_scheme_map_from_config(
            config
        )
        assert "Linear" in target_map
        assert target_map["Linear"]["weights"].num_bits == 8
        assert target_map["Linear"]["input_activations"].num_bits == 8

    def test_get_scheme_ignores_layer(self, compressed_tensors_module):
        config = _make_config_instance(compressed_tensors_module)
        config.ignore = ["skip_layer"]
        config.target_scheme_map = {"Linear": {"weights": _make_w8a8_quant(), "input_activations": _make_w8a8_quant(8, "token", True)}}
        scheme = config.get_scheme(layer=DummyLinearBase(), layer_name="skip_layer")
        assert scheme is None

    def test_get_weight_num_bits_dict_match(self, compressed_tensors_module):
        config = _make_config_instance(compressed_tensors_module)
        weight_quant = _make_w8a8_quant(num_bits={"module": 4})
        assert config._get_weight_num_bits("mlp.module", weight_quant) == 4

    def test_get_scheme_from_parts_dynamic_w8a8(self, compressed_tensors_module):
        config = _make_config_instance(compressed_tensors_module)
        weight_quant = _make_w8a8_quant(8, "tensor", False, True)
        input_quant = _make_w8a8_quant(8, "token", True, True)
        scheme = config._get_scheme_from_parts(
            weight_quant=weight_quant,
            input_quant=input_quant,
            quant_format="activation",
            layer_name="Linear",
        )
        assert isinstance(scheme, DummyNPUCompressedTensorsW8A8Int8)

    def test_override_quantization_method(self, compressed_tensors_module, monkeypatch):
        monkeypatch.setattr(torch.npu, "is_available", lambda: True)
        override = compressed_tensors_module.NPUCompressedTensorsConfig.override_quantization_method(
            {"quant_method": "compressed-tensors"}, None
        )
        assert override == compressed_tensors_module.NPU_COMPRESSED_TENSORS

    def test_get_moe_method_returns_npu_method(self, compressed_tensors_module):
        config = _make_config_instance(compressed_tensors_module)
        config.target_scheme_map = {
            "Linear": {
                "weights": _make_w8a8_quant(8, "tensor", False, True),
                "input_activations": _make_w8a8_quant(8, "token", True, True),
            }
        }
        method = config.get_moe_method(DummyFusedMoE())
        assert isinstance(method, DummyNPUCompressedTensorsW8A8Int8MoEMethod)

    def test_get_fc_method_returns_w8a8(self, compressed_tensors_module):
        config = _make_config_instance(compressed_tensors_module)
        config.target_scheme_map = {
            "Linear": {
                "weights": _make_w8a8_quant(8, "tensor", False, True),
                "input_activations": _make_w8a8_quant(8, "token", True, True),
            }
        }
        method = config.get_fc_method(DummyLinearBase(), "Linear")
        assert isinstance(method, DummyW8A8Int8FCLinearMethod)

    def test_get_quant_method_linear_base_sets_scheme(self, compressed_tensors_module):
        config = _make_config_instance(compressed_tensors_module)
        config.target_scheme_map = {
            "Linear": {
                "weights": _make_w8a8_quant(8, "tensor", False, True),
                "input_activations": _make_w8a8_quant(8, "token", True, True),
            }
        }
        layer = DummyLinearBase()
        method = config.get_quant_method(layer, "Linear")
        assert isinstance(method, DummyCompressedTensorsLinearMethod)
        assert hasattr(layer, "scheme")

    def test_get_quant_method_moe_returns_w8a8_method(self, compressed_tensors_module):
        config = _make_config_instance(compressed_tensors_module)
        config.target_scheme_map = {
            "Linear": {
                "weights": _make_w8a8_quant(8, "tensor", False, True),
                "input_activations": _make_w8a8_quant(8, "token", True, True),
            }
        }
        method = config.get_quant_method(DummyFusedMoE(), "moe")
        assert isinstance(method, DummyNPUCompressedTensorsW8A8Int8MoEMethod)

    def test_get_quant_method_moe_returns_w4a8_method(self, compressed_tensors_module):
        config = _make_config_instance(compressed_tensors_module)
        config.target_scheme_map = {
            "Linear": {
                "weights": _make_w4a8_quant({"mlp.experts": 4}, "tensor", False, True),
                "input_activations": _make_w8a8_quant(8, "token", True, True),
            }
        }
        method = config.get_quant_method(DummyFusedMoE(), "moe")
        assert isinstance(method, DummyNPUCompressedTensorsW4A8Int4MoEMethod)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
