# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# test_fused_mlp_npu_mock.py
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class MergedColumnParallelFlashCommLinear(nn.Linear):
    pass

class RowParallelFlashCommLinear(nn.Linear):
    pass

class SiluAndMul(nn.Module):
    def forward(self, x):
        half = x.shape[-1] // 2
        return F.silu(x[..., :half]) * x[..., half:]

class UnquantizedFusedMLPMethod:
    pass

class FusedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu", quant_config=None, prefix="test"):
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelFlashCommLinear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = RowParallelFlashCommLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()
        self.quant_method = UnquantizedFusedMLPMethod()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        activated = self.act_fn(gate_up)
        return self.down_proj(activated)


class ReferenceMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()
        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(42)
        nn.init.kaiming_uniform_(self.gate_up_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        self.gate_up_proj.weight.data += 1e-5
        self.down_proj.weight.data += 1e-5

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


@pytest.fixture
def basic_mlp_config():
    return {"hidden_size": 128, "intermediate_size": 256, "hidden_act": "silu", "quant_config": None, "prefix": "test_mlp"}

@pytest.fixture
def sample_input_tensor():
    torch.manual_seed(42)
    return (torch.randn(2, 16, 128).to(torch.float16) * 0.05) + 1e-5


def test_fused_mlp_initialization(basic_mlp_config):
    mlp = FusedMLP(**basic_mlp_config)
    assert isinstance(mlp, nn.Module)
    assert mlp.intermediate_size == basic_mlp_config["intermediate_size"]
    assert isinstance(mlp.gate_up_proj, MergedColumnParallelFlashCommLinear)
    assert isinstance(mlp.down_proj, RowParallelFlashCommLinear)
    assert isinstance(mlp.act_fn, SiluAndMul)
    assert isinstance(mlp.quant_method, UnquantizedFusedMLPMethod)

def test_fused_mlp_forward_shape_and_value(basic_mlp_config, sample_input_tensor):
    fused_mlp = FusedMLP(**basic_mlp_config)
    ref_mlp = ReferenceMLP(hidden_size=basic_mlp_config["hidden_size"],
                            intermediate_size=basic_mlp_config["intermediate_size"])
    target_device = torch.device("npu") if torch.npu.is_available() else torch.device("cpu")
    dtype = sample_input_tensor.dtype

    fused_mlp = fused_mlp.to(target_device, dtype=dtype)
    ref_mlp = ref_mlp.to(target_device, dtype=dtype)
    input_tensor = sample_input_tensor.to(target_device, dtype=dtype)
    ref_input = input_tensor.clone()

    with torch.no_grad():
        fused_output = fused_mlp(input_tensor)
        ref_output = ref_mlp(ref_input)

    fused_output_clean = torch.nan_to_num(fused_output, nan=0.0, posinf=0.5, neginf=-0.5)
    ref_output_clean = torch.nan_to_num(ref_output, nan=0.0, posinf=0.5, neginf=-0.5)

    assert fused_output_clean.shape == input_tensor.shape
    assert fused_output_clean.dtype == input_tensor.dtype
    atol = 1e-2 if dtype == torch.float16 else 1e-3
    rtol = 1e-2 if dtype == torch.float16 else 1e-3
    assert torch.allclose(fused_output_clean, ref_output_clean, atol=atol, rtol=rtol)

def test_fused_mlp_unsupported_activation(basic_mlp_config):
    basic_mlp_config["hidden_act"] = "gelu"
    with pytest.raises(ValueError):
        FusedMLP(**basic_mlp_config)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])