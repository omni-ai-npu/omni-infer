# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for patch_matmul."""

import pytest
import torch
import torch.nn.functional as F

from omni.vllm_patches.patches.common import patch_matmul


@pytest.fixture
def custom_op(monkeypatch):
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_ai_infra_matmul",
        torch.mm,
        raising=False,
    )


@pytest.mark.unit
def test_matmul_npu(custom_op):
    weight = torch.ones(3, 4)

    assert patch_matmul.matmul_npu(torch.ones(2, 3), weight).shape == (2, 4)
    assert patch_matmul.matmul_npu(torch.ones(2, 2, 3), weight).shape == (2, 2, 4)

    with pytest.raises(ValueError):
        patch_matmul.matmul_npu(torch.ones(2, 3), weight, out=torch.empty(2, 4))
    with pytest.raises(ValueError):
        patch_matmul.matmul_npu(torch.ones(2, 3), weight.unsqueeze(0))
    with pytest.raises(ValueError):
        patch_matmul.matmul_npu(torch.ones(3), weight)


@pytest.mark.unit
def test_linear_npu(custom_op):
    output = patch_matmul.linear_npu(
        torch.ones(1, 3),
        torch.ones(2, 3),
        torch.ones(2),
    )

    assert output.shape == (1, 2)


@pytest.mark.unit
def test_matmul_npu_backward(custom_op):
    input_tensor = torch.ones(2, 2, 3, requires_grad=True)
    weight = torch.ones(3, 4, requires_grad=True)

    patch_matmul.matmul_npu(input_tensor, weight).sum().backward()

    assert torch.equal(input_tensor.grad, torch.full_like(input_tensor, 4))
    assert torch.equal(weight.grad, torch.full_like(weight, 4))

    empty_input = torch.empty(2, 0, 3, requires_grad=True)
    empty_weight = torch.ones(3, 4, requires_grad=True)
    output = patch_matmul.matmul_npu(empty_input, empty_weight)
    output.sum().backward()

    assert output.shape == (2, 0, 4)
    assert empty_input.grad.shape == empty_input.shape
    assert torch.equal(empty_weight.grad, torch.zeros_like(empty_weight))


@pytest.mark.unit
def test_custom_op_available(custom_op, monkeypatch):
    assert patch_matmul._custom_op_available() is True

    def raise_error(*args, **kwargs):
        raise RuntimeError

    monkeypatch.setitem(
        patch_matmul._custom_op_available.__globals__,
        "getattr",
        raise_error,
    )
    assert patch_matmul._custom_op_available() is False


@pytest.mark.unit
def test_patch_matmul_skips_without_custom_op(monkeypatch):
    monkeypatch.setattr(patch_matmul, "_custom_op_available", lambda: False)
    original_functions = (torch.matmul, F.linear)

    patch_matmul.patch_matmul()

    assert (torch.matmul, F.linear) == original_functions


@pytest.mark.unit
def test_patch_matmul_replaces_native_functions(monkeypatch):
    monkeypatch.setattr(patch_matmul, "_custom_op_available", lambda: True)
    monkeypatch.setattr(torch, "matmul", torch._C._VariableFunctions.matmul)
    monkeypatch.setattr(F, "linear", torch._C._nn.linear)

    patch_matmul.patch_matmul()

    assert torch.matmul is patch_matmul.matmul_npu
    assert F.linear is patch_matmul.linear_npu


@pytest.mark.unit
def test_patch_matmul_keeps_existing_patches(monkeypatch):
    existing_matmul = object()
    existing_linear = object()
    monkeypatch.setattr(patch_matmul, "_custom_op_available", lambda: True)
    monkeypatch.setattr(torch, "matmul", existing_matmul)
    monkeypatch.setattr(F, "linear", existing_linear)

    patch_matmul.patch_matmul()

    assert torch.matmul is existing_matmul
    assert F.linear is existing_linear
