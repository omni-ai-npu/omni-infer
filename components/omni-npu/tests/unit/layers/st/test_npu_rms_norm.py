# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional
import pytest
import torch
from omni_npu.layers.npu_rms_norm import NPURMSNorm
from .distributed_test_common import parse_ascend_devices, distributed_worker_pool

FIRST_DIE, _ = parse_ascend_devices()


@pytest.fixture
def npu_device():
    return torch.device(f"npu:{FIRST_DIE}")


def rmsnorm_golden(
    x: torch.Tensor, residual: Optional[torch.Tensor], weight: torch.Tensor, eps: float
):
    """
    reference rmsnorm
    """
    x_f32 = x.float()
    weight_f32 = weight.float()

    if residual is not None:
        res_f32 = residual.float()
        res_out = x_f32 + res_f32
        norm_input = res_out
    else:
        res_out = None
        norm_input = x_f32

    variance = norm_input.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = norm_input * torch.rsqrt(variance + eps)
    out = hidden_states * weight_f32

    return out.to(x.dtype), (res_out.to(x.dtype) if res_out is not None else None)


@pytest.mark.parametrize("load_bias_env", ["0"])
@pytest.mark.parametrize("with_residual", [True, False])
@pytest.mark.parametrize("with_quant", [True, False])
@pytest.mark.parametrize("y_transform", [""])
def test_npurmsnorm_basic(
    default_vllm_config, npu_device, load_bias_env, with_residual, with_quant, y_transform
):
    """
    Runs the NPURMSNorm logic on actual NPU hardware and compares against a Reference.
    """
    hidden_size = 128
    dtype = torch.float16
    eps = 1e-6

    with pytest.MonkeyPatch.context() as m:
        m.setenv("LOAD_RMSNORM_BIAS", load_bias_env)
        norm = NPURMSNorm(hidden_size, eps=eps, dtype=dtype).to(npu_device)

        x = torch.randn(1, 10, hidden_size, dtype=dtype, device=npu_device)
        x_ref = x.clone()

        if with_residual:
            residual = torch.randn(1, 10, hidden_size, dtype=dtype, device=npu_device)
            residual_ref = residual.clone()
        else:
            residual = None
            residual_ref = None

        output = norm(
            x, residual=residual, quant_symbol=with_quant, y_transform=y_transform
        )
        ref_out, ref_residual = rmsnorm_golden(x_ref, residual_ref, norm.weight, eps)

        if with_residual:
            assert isinstance(output, tuple)
            out_x, out_res = output
            assert torch.allclose(
                out_res, ref_residual, atol=1e-3, rtol=1e-3
            ), "Residual update mismatch"

            if with_quant:
                assert isinstance(out_x, dict)
                assert set(out_x.keys()) == {"x_int8", "pertoken_scale"}
                # For now, just ensuring structure and residual correctness is sufficient.
            else:
                assert isinstance(out_x, torch.Tensor)
                assert torch.allclose(
                    out_x, ref_out, atol=1e-3, rtol=1e-3
                ), "RMSNorm output mismatch (Residual path)"
        else:
            assert isinstance(
                output, torch.Tensor
            ), "Output should be tensor when residual is not provided"
            assert torch.allclose(
                output, ref_out, atol=1e-3, rtol=1e-3
            ), "RMSNorm output mismatch (Standard path)"


def _logic_npurmsnorm_ag(device, local_rank, world_size, hidden_size, dtype):
    """
    Logic for testing NPURMSNorm AG in distributed setting.
    """
    device = torch.device(f"npu:{device}")
    eps = 1e-6

    model = NPURMSNorm(hidden_size, eps=eps).to(dtype).to(device)
    torch.nn.init.ones_(model.weight)

    torch.manual_seed(local_rank)
    x = torch.randn(2, 4, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(2, 4, hidden_size, dtype=dtype, device=device)
    x_ref = x.clone()
    residual_ref = residual.clone()

    out_gathered, out_residual = model(
        x, residual=residual, quant_symbol=False, y_transform="AG"
    )

    expected_shape = (world_size * x.shape[0], x.shape[1], x.shape[2])
    assert (
        out_gathered.shape == expected_shape
    ), f"Shape mismatch. Got {out_gathered.shape}, expected {expected_shape}"

    expected_res = (x + residual).to(dtype)
    assert torch.allclose(
        out_residual, expected_res, atol=1e-3, rtol=1e-3
    ), "Residual value mismatch"

    local_ref_out, _ = rmsnorm_golden(x_ref, residual_ref, model.weight, eps)
    start_idx = local_rank * x.shape[0]
    end_idx = (local_rank + 1) * x.shape[0]
    local_slice_in_gathered = out_gathered[start_idx:end_idx]
    assert torch.allclose(
        local_slice_in_gathered, local_ref_out, atol=1e-3, rtol=1e-3
    ), f"Rank {local_rank}: Gathered output's local slice does not match local computation"


def test_npurmsnorm_ag_distributed(distributed_worker_pool):
    """
    Tests NPURMSNorm AG using shared persistent worker pool.
    """
    hidden_size = 128
    dtype = torch.float16
    distributed_worker_pool(_logic_npurmsnorm_ag, hidden_size, dtype)