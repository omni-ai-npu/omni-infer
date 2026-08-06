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
def npu_mome_module(monkeypatch):
    """Import omni.layers.mome.npu_mome with its deps stubbed (no NPU)."""
    repo_root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo_root / "omni"))

    _make_module(monkeypatch, "torch_npu")

    _make_module(monkeypatch, "vllm", is_package=True)
    distributed_module = _make_module(monkeypatch, "vllm.distributed")
    distributed_module.divide = lambda a, b: a // b
    distributed_module.get_tensor_model_parallel_rank = lambda: 0
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1

    logger_module = _make_module(monkeypatch, "vllm.logger")
    logger_module.init_logger = lambda _name: MagicMock()

    _make_module(monkeypatch, "vllm.model_executor", is_package=True)
    _make_module(monkeypatch, "vllm.model_executor.layers", is_package=True)
    _make_module(
        monkeypatch, "vllm.model_executor.layers.quantization", is_package=True
    )
    quant_base_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.quantization.base_config"
    )
    quant_base_module.QuantizationConfig = object

    linear_module = _make_module(monkeypatch, "vllm.model_executor.layers.linear")
    linear_module.BasevLLMParameter = object
    linear_module.ModelWeightParameter = object

    _make_module(monkeypatch, "vllm.model_executor.models", is_package=True)
    models_utils_module = _make_module(monkeypatch, "vllm.model_executor.models.utils")
    models_utils_module.extract_layer_index = lambda _prefix: 0

    omni_pkg = _make_module(monkeypatch, "omni_npu", is_package=True)
    omni_pkg.__path__ = [str(repo_root / "omni")]
    layers_pkg = _make_module(monkeypatch, "omni.layers", is_package=True)
    layers_pkg.__path__ = [str(repo_root / "omni" / "layers")]
    mome_pkg = _make_module(monkeypatch, "omni.layers.mome", is_package=True)
    mome_pkg.__path__ = [str(repo_root / "omni" / "layers" / "mome")]

    _make_module(monkeypatch, "omni.v1", is_package=True)
    v1_utils_module = _make_module(monkeypatch, "omni.v1.utils")
    v1_utils_module.on_ascend950 = lambda: True

    _make_module(monkeypatch, "omni.attention", is_package=True)
    _make_module(monkeypatch, "omni.attention.backends", is_package=True)
    mome_attn_module = _make_module(monkeypatch, "omni.attention.backends.mome")

    class NPUMomeAttentionMetadata:
        pass

    mome_attn_module.NPUMomeAttentionMetadata = NPUMomeAttentionMetadata

    module_name = "omni.layers.mome.npu_mome"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    return module


def _mome_metadata():
    # Values are opaque — forward() only forwards them into the mocked kernel.
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        cache_indices=torch.zeros(4, dtype=torch.int32),
        num_accepted_tokens=None,
        num_computed_tokens=torch.zeros(4, dtype=torch.int32),
        block_idx_first_scheduled_token=torch.zeros(1, dtype=torch.int32),
        block_idx_last_scheduled_token=torch.zeros(1, dtype=torch.int32),
        block_idx_last_computed_token=torch.zeros(1, dtype=torch.int32),
        pad_slot_id=-1,
        max_query_len=4,
        B_size=64,
    )


def _fake_layer():
    return SimpleNamespace(on_ascend950=True, weight=torch.zeros(2, 3))


def test_forward_fused_kernel_inplace_uses_v2(npu_mome_module):
    npu_mome_module.torch_npu.npu_fused_causal_conv1d_v2 = MagicMock(return_value="v2")
    npu_mome_module.torch_npu.npu_fused_causal_conv1d = MagicMock(return_value="v1")

    out = npu_mome_module.ColumnParallelMOME.forward(
        _fake_layer(),
        torch.zeros(4, 3),
        torch.zeros(4, 3),
        _mome_metadata(),
        inplace=True,
    )

    assert out == "v2"
    npu_mome_module.torch_npu.npu_fused_causal_conv1d_v2.assert_called_once()
    npu_mome_module.torch_npu.npu_fused_causal_conv1d.assert_not_called()
    # conv_kwargs are built once and forwarded to the fused kernel.
    kwargs = npu_mome_module.torch_npu.npu_fused_causal_conv1d_v2.call_args.kwargs
    assert kwargs["conv_mode"] == "pangu"
    assert kwargs["run_mode"] == 0
    assert kwargs["residual_connection"] == 1


def test_forward_fused_kernel_non_inplace_uses_v1(npu_mome_module):
    npu_mome_module.torch_npu.npu_fused_causal_conv1d_v2 = MagicMock(return_value="v2")
    npu_mome_module.torch_npu.npu_fused_causal_conv1d = MagicMock(return_value="v1")

    out = npu_mome_module.ColumnParallelMOME.forward(
        _fake_layer(),
        torch.zeros(4, 3),
        torch.zeros(4, 3),
        _mome_metadata(),
        inplace=False,
    )

    assert out == "v1"
    npu_mome_module.torch_npu.npu_fused_causal_conv1d.assert_called_once()
    npu_mome_module.torch_npu.npu_fused_causal_conv1d_v2.assert_not_called()
