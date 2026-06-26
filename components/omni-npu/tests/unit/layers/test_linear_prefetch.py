# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import importlib.machinery
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest
import torch


def _install_linear_stubs(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo_root / "src"))

    torch_npu_mod = types.ModuleType("torch_npu")
    torch_npu_mod.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    torch_npu_mod.npu_format_cast = lambda tensor, _fmt: tensor
    torch_npu_mod.npu_prefetch = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_mod)

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda _name=None: None
    model_executor = types.ModuleType("vllm.model_executor")
    model_layers = types.ModuleType("vllm.model_executor.layers")
    model_quant = types.ModuleType("vllm.model_executor.layers.quantization")
    model_quant.__path__ = []
    quant_base = types.ModuleType("vllm.model_executor.layers.quantization.base_config")
    parameter_mod = types.ModuleType("vllm.model_executor.parameter")
    utils_mod = types.ModuleType("vllm.model_executor.utils")
    dist_mod = types.ModuleType("vllm.distributed")

    class QuantizeMethodBase:
        pass

    class QuantizationConfig:
        def get_quant_method(self, *_args, **_kwargs):
            return None

    class BasevLLMParameter:
        pass

    class GroupCoordinator:
        pass

    quant_base.QuantizationConfig = QuantizationConfig
    quant_base.QuantizeMethodBase = QuantizeMethodBase
    parameter_mod.BasevLLMParameter = BasevLLMParameter
    utils_mod.set_weight_attrs = (
        lambda param, attrs: [setattr(param, key, value) for key, value in attrs.items()]
    )
    dist_mod.divide = lambda value, divisor: value // divisor
    dist_mod.split_tensor_along_last_dim = (
        lambda tensor, num_partitions: torch.chunk(tensor, num_partitions, dim=-1)
    )
    dist_mod.GroupCoordinator = GroupCoordinator

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", model_layers)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.quantization", model_quant
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.base_config",
        quant_base,
    )
    monkeypatch.setitem(sys.modules, "vllm.model_executor.parameter", parameter_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "vllm.distributed", dist_mod)

    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(repo_root / "src" / "omni_npu")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    omni_v1_pkg = types.ModuleType("omni_npu.v1")
    omni_v1_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1", omni_v1_pkg)
    omni_dist_pkg = types.ModuleType("omni_npu.v1.distributed")
    omni_dist_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1" / "distributed")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1.distributed", omni_dist_pkg)
    omni_layers_pkg = types.ModuleType("omni_npu.v1.layers")
    omni_layers_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1" / "layers")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1.layers", omni_layers_pkg)
    omni_models_pkg = types.ModuleType("omni_npu.v1.models")
    omni_models_pkg.__path__ = [str(repo_root / "src" / "omni_npu" / "v1" / "models")]
    monkeypatch.setitem(sys.modules, "omni_npu.v1.models", omni_models_pkg)
    omni_config_loader_pkg = types.ModuleType("omni_npu.model_config.config_loader")
    omni_config_loader_pkg.__path__ = [
        str(repo_root / "src" / "omni_npu" / "v1" / "models" / "config_loader")
    ]
    monkeypatch.setitem(
        sys.modules, "omni_npu.model_config.config_loader", omni_config_loader_pkg
    )

    comm_mod = types.ModuleType("omni_npu.v1.distributed.communication_op_ext")
    comm_mod.layer_parallel_all_reduce = lambda data, *_args: data
    comm_mod.layer_parallel_all_gather = lambda data, *_args: data
    comm_mod.layer_parallel_reduce_scatter = lambda data, *_args: data
    comm_mod.layer_parallel_all2all_single = lambda data, *_args: data
    monkeypatch.setitem(
        sys.modules, "omni_npu.v1.distributed.communication_op_ext", comm_mod
    )

    parallel_state_mod = types.ModuleType("omni_npu.v1.distributed.parallel_state_ext")
    parallel_state_mod.get_layer_transform_type = lambda *_args: "NoOp"
    parallel_state_mod.get_layer_dim = lambda *_args: 0
    parallel_state_mod.get_layer_parallel_world_size = lambda: 1
    parallel_state_mod.get_layer_parallel_rank = lambda: 0
    parallel_state_mod.get_layer_parallel_group = lambda data, *_args: GroupCoordinator()
    monkeypatch.setitem(
        sys.modules, "omni_npu.v1.distributed.parallel_state_ext", parallel_state_mod
    )

    loader_mod = types.ModuleType("omni_npu.model_config.config_loader.loader")
    loader_mod.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(attn_prefetch=0, enable_mlaprolog=False)
    )
    monkeypatch.setitem(sys.modules, "omni_npu.model_config.config_loader.loader", loader_mod)

    utils_v1_mod = types.ModuleType("omni_npu.v1.utils")
    utils_v1_mod.get_last_two_parts = lambda name: ".".join(name.split(".")[-2:])
    utils_v1_mod.ACL_FORMAT_FRACTAL_NZ = 29
    monkeypatch.setitem(sys.modules, "omni_npu.v1.utils", utils_v1_mod)


def _import_linear_module(monkeypatch):
    _install_linear_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, "omni_npu.v1.layers.linear", raising=False)
    module = importlib.import_module("omni_npu.v1.layers.linear")
    return importlib.reload(module)


@pytest.mark.unit
def test_row_parallel_flash_comm_linear_prefetches_next_layer_weights(monkeypatch):
    module = _import_linear_module(monkeypatch)

    output_parallel = torch.randn((2, 4))
    prefetch_calls = []

    def fake_apply(_layer, input_parallel, bias, x_transform, x_dim):
        assert input_parallel.shape == (2, 4)
        assert bias is layer.bias
        assert x_transform == "NoOp"
        assert x_dim == 0
        return output_parallel

    monkeypatch.setattr(
        module.model_extra_config,
        "operator_opt_config",
        SimpleNamespace(attn_prefetch=3, enable_mlaprolog=False),
    )
    monkeypatch.setattr(
        module,
        "layer_parallel_communication_op",
        lambda data, *_args: data,
    )
    monkeypatch.setattr(
        module.torch_npu,
        "npu_prefetch",
        lambda weight, trigger, size: prefetch_calls.append((weight, trigger, size)),
    )

    layer = module.RowParallelFlashCommLinear.__new__(module.RowParallelFlashCommLinear)
    torch.nn.Module.__init__(layer)
    layer.input_is_parallel = True
    layer.quant_method = SimpleNamespace(apply=fake_apply)
    layer.tp_rank = 0
    layer.skip_bias_add = False
    layer.bias = torch.zeros(4)
    layer.x_transform = "NoOp"
    layer.x_dim = 0
    layer.y_transform = "NoOp"
    layer.y_dim = 0
    layer.layer_name_inside_block = "layers.0.self_attn.o_proj"
    layer.return_bias = True

    next_layer = [
        SimpleNamespace(weight=torch.tensor([1.0])),
        SimpleNamespace(weight=torch.tensor([2.0])),
    ]

    output, output_bias = module.RowParallelFlashCommLinear.forward(
        layer,
        torch.randn((2, 4)),
        next_layer=next_layer,
    )

    assert output is output_parallel
    assert output_bias is None
    assert len(prefetch_calls) == 2
    assert prefetch_calls[0][0] is next_layer[0].weight
    assert prefetch_calls[0][1] is output_parallel
    assert prefetch_calls[0][2] == 3 * 1024 * 1024
    assert prefetch_calls[1][0] is next_layer[1].weight
    assert prefetch_calls[1][1] is output_parallel
    assert prefetch_calls[1][2] == 3 * 1024 * 1024
