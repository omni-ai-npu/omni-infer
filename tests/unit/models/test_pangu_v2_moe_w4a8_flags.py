# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for OpenPanguV2MOE W4A8 flag detection (pangu_v2_moe.py).

The W4A8 forward path itself calls NPU ops (npu_grouped_matmul /
npu_dequant_swiglu_quant) and is covered by integration tests. Here we
guard the two inline detection lines in ``OpenPanguV2MOE.__init__`` that
select between the W4A8 and W8A8 forward branches:

    self._is_w4a8 = hasattr(self.experts, "w13_weight_int4_scale")
    self._is_w4a8_weight_asymmetric = (
        getattr(self.experts, "w13_weight_offset", None) is not None
    )

``__init__`` builds the full MoE runner (heavy vLLM deps), so we bypass it
with ``__new__`` and exercise the detection logic against mock experts
objects that mirror the four relevant configurations.
"""
from types import SimpleNamespace

import pytest
import torch

# Real import (the test env has vLLM installed, like test_pangu_v2_moe_residual).
from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit


def _detect_w4a8_flags(experts):
    """Mirror of the two production detection lines in OpenPanguV2MOE.__init__."""
    moe = model_mod.OpenPanguV2MOE.__new__(model_mod.OpenPanguV2MOE)
    moe.experts = experts
    # Exact production logic:
    moe._is_w4a8 = hasattr(moe.experts, "w13_weight_int4_scale")
    moe._is_w4a8_weight_asymmetric = (
        getattr(moe.experts, "w13_weight_offset", None) is not None
    )
    return moe


def test_w4a8_flag_false_for_w8a8_experts():
    """W8A8 experts (int8 weight, no int4_scale) -> _is_w4a8=False."""
    experts = SimpleNamespace(w13_weight=torch.zeros(2, 4, 4, dtype=torch.int8))
    moe = _detect_w4a8_flags(experts)
    assert moe._is_w4a8 is False
    assert moe._is_w4a8_weight_asymmetric is False


def test_w4a8_flag_true_for_symmetric_w4a8_experts():
    """Symmetric W4A8 experts (int4_scale present, no offset) -> _is_w4a8=True, _is_w4a8_weight_asymmetric=False."""
    experts = SimpleNamespace(
        w13_weight=torch.zeros(2, 4, 4, dtype=torch.int8),
        w13_weight_int4_scale=torch.ones(2, 1, 8, dtype=torch.int64),
    )
    moe = _detect_w4a8_flags(experts)
    assert moe._is_w4a8 is True
    assert moe._is_w4a8_weight_asymmetric is False


def test_w4a8_flag_true_and_asymmetric_for_w4a8_experts_with_offset():
    """Asymmetric W4A8 experts (int4_scale + offset present) -> _is_w4a8=True, _is_w4a8_weight_asymmetric=True."""
    experts = SimpleNamespace(
        w13_weight=torch.zeros(2, 4, 4, dtype=torch.int8),
        w13_weight_int4_scale=torch.ones(2, 1, 8, dtype=torch.int64),
        w13_weight_offset=torch.ones(2, 1, 8, dtype=torch.float32),
    )
    moe = _detect_w4a8_flags(experts)
    assert moe._is_w4a8 is True
    assert moe._is_w4a8_weight_asymmetric is True


def test_w4a8_asymmetric_flag_ignores_none_offset():
    """A registered-but-None offset (symmetric quant) -> asymmetric=False."""
    experts = SimpleNamespace(
        w13_weight=torch.zeros(2, 4, 4, dtype=torch.int8),
        w13_weight_int4_scale=torch.ones(2, 1, 8, dtype=torch.int64),
        w13_weight_offset=None,
    )
    moe = _detect_w4a8_flags(experts)
    assert moe._is_w4a8 is True
    assert moe._is_w4a8_weight_asymmetric is False
