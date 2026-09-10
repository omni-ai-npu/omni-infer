# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for int4_w4a8_moe_quant_config offset params (omni/layers/fused_moe/config.py)."""
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _ensure_module(monkeypatch, name):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _FusedMoEQuantDesc:
    """Captures all positional args so the test can assert offset/bias wiring."""

    def __init__(self, dtype, shape, scale, *extra):
        self.dtype = dtype
        self.shape = shape
        self.scale = scale
        self.extra = extra  # (None, offset, bias) for w1/w2


class _FusedMoEQuantConfig:
    def __init__(self, _a1=None, _a2=None, _w1=None, _w2=None):
        self._a1 = _a1
        self._w2 = _w2
        self._w1 = _w1
        self._w2 = _w2
        # _quant_flags_to_group_shape returns a_shape whose first elt encodes
        # per_act_token_quant; mirror that so the assert in production code holds.
        self.per_act_token_quant = _a1.shape[0] if _a1 is not None else False


def _quant_flags_to_group_shape(dtype, per_act_token_quant, *_a, **_kw):
    # a_shape[0] carries the per_act_token flag; w_shape is unused by the test.
    return ((per_act_token_quant,), None)


@pytest.fixture
def config_module(monkeypatch):
    fused_moe_config_module = _ensure_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.config")
    fused_moe_config_module.FusedMoEQuantConfig = _FusedMoEQuantConfig
    fused_moe_config_module.FusedMoEQuantDesc = _FusedMoEQuantDesc
    fused_moe_config_module._quant_flags_to_group_shape = _quant_flags_to_group_shape

    base_path = Path(__file__).resolve().parents[4]
    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(base_path / "omni")]
    monkeypatch.setitem(sys.modules, "omni_npu", omni_pkg)
    layers_pkg = types.ModuleType("omni_npu.layers")
    layers_pkg.__path__ = [str(base_path / "omni" / "layers")]
    monkeypatch.setitem(sys.modules, "omni_npu.layers", layers_pkg)
    fused_moe_pkg = types.ModuleType("omni_npu.layers.fused_moe")
    fused_moe_pkg.__path__ = [str(base_path / "omni" / "layers" / "fused_moe")]
    monkeypatch.setitem(sys.modules, "omni_npu.layers.fused_moe", fused_moe_pkg)

    sys.modules.pop("omni_npu.layers.fused_moe.config", None)
    module = importlib.import_module("omni_npu.layers.fused_moe.config")
    importlib.reload(module)
    return module


@pytest.mark.unit
def test_int4_w4a8_moe_quant_config_passes_offsets_to_w1_w2(config_module):
    """w1_offset / w2_offset are forwarded into the w1/w2 FusedMoEQuantDesc."""
    w1_scale = torch.ones(2)
    w2_scale = torch.ones(2)
    w1_bias = torch.zeros(2)
    w2_bias = torch.zeros(2)
    w1_offset = torch.full((2,), 1.0)
    w2_offset = torch.full((2,), 2.0)

    cfg = config_module.int4_w4a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        a1_scale=None,
        a2_scale=None,
        w1_offset=w1_offset,
        w2_offset=w2_offset,
        per_act_token_quant=True,
    )

    # _w1 desc extra = (None, w1_offset, w1_bias)
    assert cfg._w1.extra[1] is w1_offset
    assert cfg._w1.extra[2] is w1_bias
    assert cfg._w2.extra[1] is w2_offset
    assert cfg._w2.extra[2] is w2_bias
    assert cfg._w1.scale is w1_scale
    assert cfg._w2.scale is w2_scale
    assert cfg.per_act_token_quant is True


@pytest.mark.unit
def test_int4_w4a8_moe_quant_config_defaults_offsets_to_none(config_module):
    """When offsets are not supplied they default to None and still wire into the descs (symmetric quant path)."""
    cfg = config_module.int4_w4a8_moe_quant_config(
        w1_scale=torch.ones(2),
        w2_scale=torch.ones(2),
        w1_bias=None,
        w2_bias=None,
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=False,
    )
    assert cfg._w1.extra[1] is None
    assert cfg._w2.extra[1] is None
    assert cfg.per_act_token_quant is False
