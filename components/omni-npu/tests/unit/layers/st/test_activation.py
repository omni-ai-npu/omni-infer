# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F

from omni_npu.layers.activation import NPUSiluAndMul
from .distributed_test_common import parse_ascend_devices

FIRST_DEVICE, _ = parse_ascend_devices()


@pytest.fixture(scope="module")
def npu_device():
    device = torch.device(f"npu:{FIRST_DEVICE}")
    torch.npu.set_device(device)
    return device

def _swiglu_ref(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def test_silu_and_mul_matches_swiglu_reference(default_vllm_config, npu_device):
    mod = NPUSiluAndMul().to(npu_device)
    x = torch.randn(2, 8, device=npu_device, dtype=torch.float16)

    out = mod(x)
    expected = _swiglu_ref(x)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3)


def test_silu_and_mul_quant_path_uses_dequant_kernel(default_vllm_config, npu_device):
    mod = NPUSiluAndMul().to(npu_device)
    tokens = 3
    hidden = 4  # output hidden; input last dim = 2 * hidden for SwiGLU
    x_int = torch.randint(
        -4, 5, (tokens, hidden * 2), device=npu_device, dtype=torch.int32
    )

    payload = {
        "x_int8": x_int,
        "out_scale": torch.ones(hidden * 2, device=npu_device, dtype=torch.float32),
        "in_scale": torch.ones(hidden, device=npu_device, dtype=torch.float32),
        "pertoken_scale": torch.ones(tokens, device=npu_device, dtype=torch.float32),
    }

    result = mod(payload, quant_symbol=True)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"x_int8", "pertoken_scale"}

    h = result["x_int8"]
    pertoken_scale = result["pertoken_scale"]

    assert h.shape == (tokens, hidden)
    assert pertoken_scale.shape == (tokens,)

    # Dequantize using returned per-token scale and compare against reference swiglu.
    dequant_out = h.to(torch.float32) * pertoken_scale.view(-1, 1)
    expected = _swiglu_ref(x_int.to(torch.float32))
    assert torch.allclose(dequant_out, expected, atol=0.5, rtol=0.1)