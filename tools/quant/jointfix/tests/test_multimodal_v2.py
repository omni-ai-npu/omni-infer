# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import pytest
import torch

from jointfix.methods.jointfix import JointSearchConfig
from jointfix.multimodal import quant as mm_quant
from jointfix.multimodal.runner import (
    enforce_reconstruction_gate,
    reconstruction_metrics,
)


class _Collector:
    def __init__(self, x):
        self.x = x

    def finalize(self, device):
        x = self.x.to(device)
        return {"X_sample": x, "amax": x.abs().amax(dim=0)}


def test_bound_smooth_scale_clamps_and_rejects_nonfinite():
    scale, info = mm_quant.bound_smooth_scale(
        torch.tensor([0.01, 0.5, 8.0]), minimum=0.25, maximum=4.0, tag="test"
    )
    torch.testing.assert_close(scale, torch.tensor([0.25, 0.5, 4.0]))
    assert info["smooth_scale_clamped_low"] == 1
    assert info["smooth_scale_clamped_high"] == 1

    with pytest.raises(RuntimeError, match="non-finite"):
        mm_quant.bound_smooth_scale(
            torch.tensor([float("inf")]), minimum=0.25, maximum=4.0, tag="bad"
        )


def test_reconstruction_gate_checks_output_and_update():
    layer_input = torch.tensor([[1.0, 2.0]])
    reference = torch.tensor([[2.0, 4.0]])
    metrics = reconstruction_metrics(reference, reference.clone(), layer_input)
    result = enforce_reconstruction_gate(
        metrics,
        tag="exact",
        max_output_rel_rmse=0.01,
        min_output_cosine=0.999,
        max_update_rel_rmse=0.01,
        min_update_cosine=0.999,
    )
    assert result["passed"] is True

    bad = reconstruction_metrics(reference, layer_input.clone(), layer_input)
    with pytest.raises(RuntimeError, match="reconstruction gate failed"):
        enforce_reconstruction_gate(
            bad,
            tag="bad",
            max_output_rel_rmse=0.01,
            min_output_cosine=0.999,
            max_update_rel_rmse=0.01,
            min_update_cosine=0.999,
        )


def test_audio_fc2_is_quantized_without_cross_gelu_smoothing(monkeypatch):
    prefix = "audio_tower.layers.0"
    hidden, ffn = 4, 8
    tensors = {
        f"{prefix}.self_attn_layer_norm.weight": torch.ones(hidden),
        f"{prefix}.self_attn_layer_norm.bias": torch.zeros(hidden),
        f"{prefix}.final_layer_norm.weight": torch.ones(hidden),
        f"{prefix}.final_layer_norm.bias": torch.zeros(hidden),
        f"{prefix}.self_attn.q_proj.weight": torch.randn(hidden, hidden),
        f"{prefix}.self_attn.q_proj.bias": torch.randn(hidden),
        f"{prefix}.self_attn.k_proj.weight": torch.randn(hidden, hidden),
        f"{prefix}.self_attn.k_proj.bias": torch.randn(hidden),
        f"{prefix}.self_attn.v_proj.weight": torch.randn(hidden, hidden),
        f"{prefix}.self_attn.v_proj.bias": torch.randn(hidden),
        f"{prefix}.self_attn.out_proj.weight": torch.randn(hidden, hidden),
        f"{prefix}.self_attn.out_proj.bias": torch.randn(hidden),
        f"{prefix}.fc1.weight": torch.randn(ffn, hidden),
        f"{prefix}.fc1.bias": torch.randn(ffn),
        f"{prefix}.fc2.weight": torch.randn(hidden, ffn),
        f"{prefix}.fc2.bias": torch.randn(hidden),
    }
    original_fc1_bias = tensors[f"{prefix}.fc1.bias"].clone()
    collectors = {
        "attn_in": _Collector(torch.randn(8, hidden)),
        "attn_out": _Collector(torch.randn(8, hidden)),
        "ffn_in": _Collector(torch.randn(8, hidden)),
        "ffn_out": _Collector(torch.randn(8, ffn)),
    }
    calls = []

    def _fixed_search(weights, stats, config, device):
        calls.append(stats["X_sample"].shape[1])
        return torch.full((weights[0].shape[1],), 2.0), {"a": 0.5, "b": 0.5}

    monkeypatch.setattr(mm_quant, "_search_scale", _fixed_search)
    output, traces = mm_quant.quantize_dense_transformer_layer(
        tensors,
        collectors,
        prefix=prefix,
        kind="audio",
        config=JointSearchConfig(gptq_block_size=2),
        device=torch.device("cpu"),
        smooth_scale_min=0.25,
        smooth_scale_max=4.0,
    )

    # Only the three mathematically exact groups search a smooth scale.
    assert calls == [hidden, hidden, hidden]
    torch.testing.assert_close(output[f"{prefix}.fc1.bias"], original_fc1_bias)
    assert traces[f"{prefix}.fc2.weight"]["smoothing"] == "disabled_across_gelu"
    assert output[f"{prefix}.fc2.weight"].dtype == torch.int8
