# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Dense-transformer JointFix + strict block-GPTQ for ViT and Audio Tower.

Unlike the generic deployment RTN path, every INT8 weight handled here must have
real calibration activations.  A failed/undersampled GPTQ block is an error; it
never silently falls back to RTN.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch

from jointfix.methods.jointfix import JointSearchConfig, joint_grid_search_ab


def strict_block_gptq(
    weight: torch.Tensor,
    activations: torch.Tensor,
    *,
    block_size: int = 128,
    damp: float = 0.01,
    tag: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-row INT8 GPTQ using independent input-channel Hessian blocks.

    The stock JointFix GPTQ constructs a full ``h_in x h_in`` Hessian and falls
    back to RTN when ``N < h_in``.  Audio fc2 has h_in=8192, so that fallback is
    guaranteed with practical calibration caches.  This strict variant keeps
    GPTQ's within-block second-order error feedback, needs only N>=block_size,
    and raises on every failure instead of changing algorithms silently.
    """
    if weight.ndim != 2 or activations.ndim != 2:
        raise ValueError(f"strict GPTQ expects 2D W/X, got {weight.shape}, {activations.shape}")
    h_out, h_in = weight.shape
    if activations.shape[1] != h_in:
        raise ValueError(f"{tag}: X width {activations.shape[1]} != W width {h_in}")
    required = min(block_size, h_in)
    if activations.shape[0] < required:
        raise RuntimeError(
            f"{tag}: strict GPTQ needs at least {required} activation rows; "
            f"got {activations.shape[0]} (RTN fallback is disabled)"
        )

    original_device = weight.device
    device = activations.device
    w = weight.to(device=device, dtype=torch.float32).clone()
    x = activations.to(device=device, dtype=torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / 127.0
    quant = torch.empty_like(w, dtype=torch.int8)

    for start in range(0, h_in, block_size):
        end = min(start + block_size, h_in)
        xb = x[:, start:end]
        width = end - start
        if xb.shape[0] < width:
            raise RuntimeError(
                f"{tag}: block [{start}:{end}] has N={xb.shape[0]} < width={width}; "
                "RTN fallback is disabled"
            )
        hessian = (xb.T @ xb) / float(xb.shape[0])
        diag_mean = hessian.diag().mean().clamp(min=1e-8)

        chol_inv = None
        cur_damp = float(damp)
        for _ in range(6):
            h_try = hessian.clone()
            h_try.diagonal().add_(cur_damp * diag_mean)
            try:
                chol = torch.linalg.cholesky(h_try, upper=False)
                h_inv = torch.cholesky_inverse(chol, upper=False)
                chol_inv = torch.linalg.cholesky(h_inv, upper=True)
                break
            except RuntimeError:
                cur_damp *= 2.0
        if chol_inv is None:
            raise RuntimeError(
                f"{tag}: Cholesky failed for GPTQ block [{start}:{end}] after 6 retries; "
                "RTN fallback is disabled"
            )

        wb = w[:, start:end].clone()
        for column in range(width):
            w_col = wb[:, column]
            q_col = (w_col / scale.squeeze(1)).round().clamp(-128, 127)
            q_dequant = q_col * scale.squeeze(1)
            denom = chol_inv[column, column].clamp(min=1e-12)
            error = (w_col - q_dequant) / denom
            quant[:, start + column] = q_col.to(torch.int8)
            if column + 1 < width:
                wb[:, column + 1:] -= (
                    error.unsqueeze(1) * chol_inv[column, column + 1:].unsqueeze(0)
                )

    print(
        f"[GPTQ-MM] {tag} h_out={h_out} h_in={h_in} "
        f"N={activations.shape[0]} block={block_size}",
        flush=True,
    )
    return quant.to(original_device), scale.to(torch.bfloat16).to(original_device)


def _finalize_stats(collector, device) -> dict:
    stats = collector.finalize(device)
    if stats["X_sample"].shape[0] == 0:
        raise RuntimeError("empty activation collector; RTN fallback is disabled")
    return stats


def _search_scale(weights: Iterable[torch.Tensor], stats: dict,
                  config: JointSearchConfig, device) -> Tuple[torch.Tensor, dict]:
    ws = [w.to(device=device, dtype=torch.float32) for w in weights]
    w_stat = torch.stack([w.abs().amax(dim=0) for w in ws]).amax(dim=0)
    stats = dict(stats)
    stats["Y_refs"] = [stats["X_sample"].float() @ w.T for w in ws]
    scale, trace = joint_grid_search_ab(ws, stats, w_stat, config)
    return scale.detach().cpu(), trace


def bound_smooth_scale(scale: torch.Tensor, *, minimum: float, maximum: float,
                       tag: str) -> Tuple[torch.Tensor, dict]:
    """
    Clamp modality smoothing to a conservative, finite range.

    Activation channels can be exactly zero on a small calibration set.  The
    unconstrained ``x_stat**a / w_stat**b`` formula then produces scales close
    to zero and the inverse absorption can inflate a weight row by many orders
    of magnitude.  This guard is modality-specific and intentionally stricter
    than the generic language-model path.
    """
    if not (0.0 < minimum <= 1.0 <= maximum):
        raise ValueError(
            f"invalid smooth scale bounds: minimum={minimum}, maximum={maximum}"
        )
    raw = scale.float()
    if not torch.isfinite(raw).all():
        raise RuntimeError(f"{tag}: non-finite JointFix smooth scale")
    bounded = raw.clamp(min=minimum, max=maximum)
    info = {
        "smooth_scale_raw_min": float(raw.min().item()),
        "smooth_scale_raw_max": float(raw.max().item()),
        "smooth_scale_min": float(bounded.min().item()),
        "smooth_scale_max": float(bounded.max().item()),
        "smooth_scale_clamped_low": int((raw < minimum).sum().item()),
        "smooth_scale_clamped_high": int((raw > maximum).sum().item()),
        "smooth_scale_bound_min": float(minimum),
        "smooth_scale_bound_max": float(maximum),
    }
    if info["smooth_scale_clamped_low"] or info["smooth_scale_clamped_high"]:
        print(
            f"[SMOOTH-CLAMP] {tag} raw=[{info['smooth_scale_raw_min']:.3e},"
            f" {info['smooth_scale_raw_max']:.3e}] -> "
            f"[{info['smooth_scale_min']:.3e}, {info['smooth_scale_max']:.3e}] "
            f"low={info['smooth_scale_clamped_low']} "
            f"high={info['smooth_scale_clamped_high']}",
            flush=True,
        )
    return bounded.cpu(), info


def _divide_norm(tensors: Dict[str, torch.Tensor], norm_base: str,
                 scale: torch.Tensor) -> None:
    for suffix in ("weight", "bias"):
        key = f"{norm_base}.{suffix}"
        if key in tensors:
            tensors[key] /= scale


def _divide_output_rows(tensors: Dict[str, torch.Tensor], linear_base: str,
                        scale: torch.Tensor) -> None:
    tensors[f"{linear_base}.weight"] /= scale.unsqueeze(1)
    bias_key = f"{linear_base}.bias"
    if bias_key in tensors:
        tensors[bias_key] /= scale


def quantize_dense_transformer_layer(
    layer_tensors: Dict[str, torch.Tensor],
    collectors: dict,
    *,
    prefix: str,
    kind: str,
    config: JointSearchConfig,
    device,
    smooth_scale_min: float = 0.25,
    smooth_scale_max: float = 4.0,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Joint-smooth and strict-GPTQ one ViT or Audio transformer layer."""
    tensors = {name: value.clone().float() for name, value in layer_tensors.items()}
    traces: dict = {}
    x_for_weight: Dict[str, torch.Tensor] = {}

    if kind == "vision":
        groups = [
            # collector, consumers, exact upstream absorption
            ("attn_in", [f"{prefix}.attn.qkv"], ("norm", f"{prefix}.norm1")),
            ("attn_out", [f"{prefix}.attn.proj"], ("vision_v", f"{prefix}.attn.qkv")),
            ("ffn_in", [f"{prefix}.mlp.up_proj"], ("norm", f"{prefix}.norm2")),
            # GELU is not homogeneous: GELU(x / s) * s != GELU(x).  Quantize
            # down_proj from its real post-GELU activation without smoothing.
            ("ffn_out", [f"{prefix}.mlp.down_proj"], None),
        ]
    elif kind == "audio":
        groups = [
            ("attn_in", [f"{prefix}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj")],
             ("norm", f"{prefix}.self_attn_layer_norm")),
            ("attn_out", [f"{prefix}.self_attn.out_proj"],
             ("linear", f"{prefix}.self_attn.v_proj")),
            ("ffn_in", [f"{prefix}.fc1"], ("norm", f"{prefix}.final_layer_norm")),
            # Same rule as ViT: do not absorb a scale across GELU.
            ("ffn_out", [f"{prefix}.fc2"], None),
        ]
    else:
        raise ValueError(f"unknown dense transformer kind: {kind}")

    for collector_name, consumers, absorption in groups:
        if collector_name not in collectors:
            raise RuntimeError(f"{prefix}: missing activation collector {collector_name}")
        stats = _finalize_stats(collectors[collector_name], device)
        weight_keys = [f"{base}.weight" for base in consumers]
        missing = [key for key in weight_keys if key not in tensors]
        if missing:
            raise RuntimeError(f"{prefix}: missing weights for JointFix: {missing}")

        if absorption is None:
            traces["+".join(weight_keys)] = {
                "collector": collector_name,
                "consumers": weight_keys,
                "smoothing": "disabled_across_gelu",
                "smooth_scale_min": 1.0,
                "smooth_scale_max": 1.0,
            }
            for key in weight_keys:
                x_for_weight[key] = stats["X_sample"].float()
            continue

        smooth, trace = _search_scale([tensors[key] for key in weight_keys], stats, config, device)
        smooth, bound_info = bound_smooth_scale(
            smooth,
            minimum=smooth_scale_min,
            maximum=smooth_scale_max,
            tag="+".join(weight_keys),
        )
        traces["+".join(weight_keys)] = {
            **trace,
            **bound_info,
            "collector": collector_name,
            "consumers": weight_keys,
            "smoothing": "bounded_exact_absorption",
        }

        absorption_kind, upstream = absorption
        if absorption_kind == "norm":
            _divide_norm(tensors, upstream, smooth)
        elif absorption_kind == "linear":
            _divide_output_rows(tensors, upstream, smooth)
        elif absorption_kind == "vision_v":
            qkv_weight = tensors[f"{upstream}.weight"]
            hidden = qkv_weight.shape[0] // 3
            qkv_weight[2 * hidden:3 * hidden] /= smooth.unsqueeze(1)
            bias_key = f"{upstream}.bias"
            if bias_key in tensors:
                tensors[bias_key][2 * hidden:3 * hidden] /= smooth
        else:
            raise AssertionError(absorption_kind)

        x_smooth = stats["X_sample"].float() / smooth.to(device).unsqueeze(0).clamp(min=1e-12)
        for key in weight_keys:
            tensors[key] *= smooth.unsqueeze(0)
            x_for_weight[key] = x_smooth

    output: Dict[str, torch.Tensor] = {}
    quantized = 0
    for name, original in sorted(layer_tensors.items()):
        if name in x_for_weight:
            qweight, weight_scale = strict_block_gptq(
                tensors[name], x_for_weight[name],
                block_size=config.gptq_block_size,
                damp=config.gptq_damp,
                tag=name,
            )
            output[name] = qweight
            output[name.replace(".weight", ".weight_scale")] = weight_scale
            quantized += 1
        else:
            output[name] = tensors[name].to(original.dtype)

    expected = 4 if kind == "vision" else 6
    if quantized != expected:
        raise RuntimeError(f"{prefix}: expected {expected} strict-GPTQ weights, got {quantized}")
    return output, traces
