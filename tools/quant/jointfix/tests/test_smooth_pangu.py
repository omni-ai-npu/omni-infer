# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Tests for the Pangu smooth+quantize seam (methods/_smooth_pangu.py).

Full end-to-end (with absorption) needs a real layer + collectors and is covered
by the on-machine equivalence test. Here we validate the quantize path and
passthrough in isolation: with no collectors, all smoothing is skipped, so this
exercises the output/input-side INT8 dispatch + skip-pattern passthrough + counts.
"""
import argparse

import torch

from jointfix.core.primitives import UNIVERSAL_SKIP_PATTERNS
from jointfix.methods._smooth_pangu import smooth_quantize_pangu_layer
from jointfix.methods._smooth_pangu import _refine_row_scales_for_weight_mse
from jointfix.methods.jointfix import JointSearchConfig, JointFixMethod

_PFX = "model.layers.0"
_MC = {"num_attention_heads": 2, "qk_nope_head_dim": 2, "v_head_dim": 2, "n_routed_experts": 0}


def _layer_tensors(h=8):
    return {
        f"{_PFX}.self_attn.o_proj.weight": torch.randn(h, h),   # output-side
        f"{_PFX}.mlp.gate_proj.weight": torch.randn(h, h),      # input-side
        f"{_PFX}.mlp.down_proj.weight": torch.randn(h, h),      # output-side (dense)
        f"{_PFX}.mlp.gate.weight": torch.randn(4, h),           # router -> skip
        f"{_PFX}.input_layernorm.weight": torch.ones(h),        # 1D -> not quantized
    }


def test_quantize_only_no_collectors():
    lt = _layer_tensors()
    cfg = JointSearchConfig(write_quant="rtn")   # rtn avoids the gptq fallback warning
    out, n_smooth, n_quant, traces = smooth_quantize_pangu_layer(
        lt, {}, _MC, 0, cfg, UNIVERSAL_SKIP_PATTERNS, torch.device("cpu"))

    assert n_smooth == 0          # no collectors -> no smoothing happened
    assert n_quant == 3           # o_proj, gate_proj, down_proj

    for wk in (f"{_PFX}.self_attn.o_proj.weight",
               f"{_PFX}.mlp.gate_proj.weight",
               f"{_PFX}.mlp.down_proj.weight"):
        assert out[wk].dtype == torch.int8
        sk = wk.replace(".weight", ".weight_scale")
        assert out[sk].dtype == torch.bfloat16

    # router (matches "mlp.gate.") -> passthrough, no scale
    assert out[f"{_PFX}.mlp.gate.weight"].dtype != torch.int8
    assert f"{_PFX}.mlp.gate.weight_scale" not in out
    # 1D norm -> passthrough
    assert out[f"{_PFX}.input_layernorm.weight"].dtype != torch.int8


def test_quantize_dequant_close_to_original():
    lt = _layer_tensors()
    cfg = JointSearchConfig(write_quant="rtn")
    out, _, _, _ = smooth_quantize_pangu_layer(
        lt, {}, _MC, 0, cfg, UNIVERSAL_SKIP_PATTERNS, torch.device("cpu"))
    wk = f"{_PFX}.mlp.gate_proj.weight"
    deq = out[wk].float() * out[wk.replace(".weight", ".weight_scale")].float()
    # no smoothing -> dequant within one per-row quant step of the original
    step = out[wk.replace(".weight", ".weight_scale")].float()
    assert torch.all((lt[wk] - deq).abs() <= step + 1e-4)


def test_configure_applies_cli_args():
    """The configure() fix: parsed CLI args must land on JointSearchConfig."""
    m = JointFixMethod()
    assert m.config.num_iterations == 1   # default
    args = argparse.Namespace(num_iterations=2, skip_shared_experts=True,
                              write_quant="rtn", objective="output-recon")
    m.configure(args)
    assert m.config.num_iterations == 2
    assert m.config.skip_shared_experts is True
    assert m.config.write_quant == "rtn"


def test_skip_shared_experts_leaves_them_bf16():
    h = 8
    lt = {
        f"{_PFX}.mlp.shared_experts.down_proj.weight": torch.randn(h, h),
        f"{_PFX}.mlp.shared_experts.gate_proj.weight": torch.randn(h, h),
        f"{_PFX}.mlp.down_proj.weight": torch.randn(h, h),     # dense -> still quantized
    }
    skip = list(UNIVERSAL_SKIP_PATTERNS) + ["mlp.shared_experts"]
    out, _, n_quant, _ = smooth_quantize_pangu_layer(
        lt, {}, _MC, 0, JointSearchConfig(write_quant="rtn"), skip, torch.device("cpu"))

    assert out[f"{_PFX}.mlp.shared_experts.down_proj.weight"].dtype != torch.int8
    assert out[f"{_PFX}.mlp.shared_experts.gate_proj.weight"].dtype != torch.int8
    assert f"{_PFX}.mlp.shared_experts.down_proj.weight_scale" not in out
    assert out[f"{_PFX}.mlp.down_proj.weight"].dtype == torch.int8   # dense still int8
    assert n_quant == 1


def test_mdmixq_scale_refinement_strictly_reduces_weight_error():
    q = torch.tensor([[1, 2, -3], [2, -1, 1]], dtype=torch.int8)
    target = torch.tensor([[0.11, 0.24, -0.34], [0.43, -0.19, 0.18]])
    baseline = torch.tensor([[0.1], [0.2]], dtype=torch.bfloat16)

    refined, metrics = _refine_row_scales_for_weight_mse(q, baseline, target)

    before = (q.float() * baseline.float() - target).square().mean()
    after = (q.float() * refined.float() - target).square().mean()
    assert after < before
    assert metrics["weight_nmse_after_scale_refine"] < metrics["weight_nmse_before_scale_refine"]
