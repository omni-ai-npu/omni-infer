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


def _divide(dividend, divisor):
    return dividend // divisor


def _zero(*_args, **_kwargs):
    return 0


def _one(*_args, **_kwargs):
    return 1


def _true(*_args, **_kwargs):
    return True


def _mock_logger(*_args, **_kwargs):
    return MagicMock()


def _module(monkeypatch, name, package=False):
    mod = types.ModuleType(name)
    if package:
        mod.__path__ = []
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def mome_rl(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root))
    _module(monkeypatch, "torch_npu")

    _module(monkeypatch, "vllm", True)
    dist = _module(monkeypatch, "vllm.distributed")
    dist.divide = _divide
    dist.get_tensor_model_parallel_rank = _zero
    dist.get_tensor_model_parallel_world_size = _one
    _module(monkeypatch, "vllm.logger").init_logger = _mock_logger
    _module(monkeypatch, "vllm.model_executor", True)
    _module(monkeypatch, "vllm.model_executor.layers", True)
    _module(monkeypatch, "vllm.model_executor.layers.quantization", True)
    _module(
        monkeypatch, "vllm.model_executor.layers.quantization.base_config"
    ).QuantizationConfig = object
    linear = _module(monkeypatch, "vllm.model_executor.layers.linear")
    linear.BasevLLMParameter = object
    linear.ModelWeightParameter = object
    _module(monkeypatch, "vllm.model_executor.models", True)
    _module(
        monkeypatch, "vllm.model_executor.models.utils"
    ).extract_layer_index = _zero

    omni = _module(monkeypatch, "omni_npu", True)
    omni.__path__ = [str(root / "omni")]
    layers = _module(monkeypatch, "omni_npu.layers", True)
    layers.__path__ = [str(root / "omni" / "layers")]
    mome = _module(monkeypatch, "omni_npu.layers.mome", True)
    mome.__path__ = [str(root / "omni" / "layers" / "mome")]
    _module(monkeypatch, "omni_npu.v1", True)
    _module(monkeypatch, "omni_npu.v1.utils").on_ascend950 = _true
    _module(monkeypatch, "omni_npu.attention", True)
    _module(monkeypatch, "omni_npu.attention.backends", True)
    _module(
        monkeypatch, "omni_npu.attention.backends.mome"
    ).NPUMomeAttentionMetadata = object
    _module(monkeypatch, "omni_npu.model_config", True)
    _module(monkeypatch, "omni_npu.model_config.config_loader", True)
    _module(
        monkeypatch, "omni_npu.model_config.config_loader.loader"
    ).model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            enable_precision_strong_consistency=False
        )
    )

    name = "omni_npu.layers.mome.mome_rl"
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def _meta(actual, max_query_len=1):
    return SimpleNamespace(
        num_actual_tokens=actual,
        query_start_loc=torch.tensor([0, actual], dtype=torch.int32),
        cache_indices=torch.zeros(1, dtype=torch.int32),
        num_accepted_tokens=None,
        num_computed_tokens=torch.zeros(1, dtype=torch.int32),
        block_idx_first_scheduled_token=torch.zeros(1, dtype=torch.int32),
        block_idx_last_scheduled_token=torch.zeros(1, dtype=torch.int32),
        block_idx_last_computed_token=torch.zeros(1, dtype=torch.int32),
        pad_slot_id=-1,
        max_query_len=max_query_len,
        B_size=64,
    )


def _layer():
    return SimpleNamespace(on_ascend950=True, weight=torch.zeros(3, 4))


def test_non_inplace_slices_and_zero_pads_to_input_shape(mome_rl):
    calls = {}

    def fake(conv_x, weight, states, **kwargs):
        calls["x"] = conv_x
        calls["kwargs"] = kwargs
        return torch.full_like(conv_x, 7)

    mome_rl.torch_npu.npu_fused_causal_conv1d = fake
    mome_rl.torch_npu.npu_fused_causal_conv1d_v2 = MagicMock()
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    out = mome_rl.ColumnParallelMOMERL.forward(
        _layer(), x, torch.zeros(1, 2, 4), _meta(4, 2), inplace=False
    )

    assert calls["x"].shape == (4, 4)
    assert torch.equal(calls["x"], x[:4])
    assert calls["kwargs"]["max_query_len"] == 2
    assert out.shape == x.shape
    assert torch.equal(out[:4], torch.full((4, 4), 7.0))
    assert torch.count_nonzero(out[4:]) == 0


def test_non_inplace_without_padding_passes_original_x(mome_rl):
    calls = {}

    def fake(conv_x, weight, states, **kwargs):
        calls["x"] = conv_x
        return conv_x + 1

    mome_rl.torch_npu.npu_fused_causal_conv1d = fake
    mome_rl.torch_npu.npu_fused_causal_conv1d_v2 = MagicMock()
    x = torch.zeros(4, 3)
    out = mome_rl.ColumnParallelMOMERL.forward(
        _layer(), x, torch.zeros(1, 2, 3), _meta(4, 4), inplace=False
    )

    assert calls["x"] is x
    assert out.shape == x.shape
    assert torch.equal(out, torch.ones_like(x))


def test_inplace_passes_only_valid_slice_to_v2(mome_rl):
    calls = {}

    def fake(conv_x, weight, states, **kwargs):
        calls["x"] = conv_x
        conv_x.add_(3)
        return conv_x

    mome_rl.torch_npu.npu_fused_causal_conv1d_v2 = fake
    mome_rl.torch_npu.npu_fused_causal_conv1d = MagicMock()
    x = torch.zeros(6, 4)
    out = mome_rl.ColumnParallelMOMERL.forward(
        _layer(), x, torch.zeros(1, 2, 4), _meta(4, 2), inplace=True
    )

    assert calls["x"].shape == (4, 4)
    assert out.shape == (4, 4)
    assert torch.equal(x[:4], torch.full((4, 4), 3.0))
    assert torch.count_nonzero(x[4:]) == 0
