# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Pure smooth-search math for the joint (a,b) method.

Single-device, side-effect-free tensor functions: smooth-scale construction,
channel weighting, weight-error term, and the output-reconstruction objective
(single + batched-over-(a,b)).

DECOUPLED from JointSearchConfig on purpose — callers pass explicit scalars
(eps / gamma / channel_weight / hessian_alpha). That keeps this a leaf module
(imports only core.primitives) and makes every function unit-testable without a
model. methods/jointfix.py wires these together in process_layer.

The distributed (multi-device) variants live with the parallel forward
machinery — they need real devices.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from jointfix.core.primitives import int8_fake_quantize


# ─────────────────────────────────────────────────────────────────────────────
# Smooth scale
# ─────────────────────────────────────────────────────────────────────────────
def make_smooth_scale(x_stat: torch.Tensor, w_stat: torch.Tensor,
                      a: float, b: float, eps: float = 1e-12) -> torch.Tensor:
    """
    Per-channel smooth scale: s_c = x_stat_c^a / w_stat_c^b.

    a=b=0 -> s=1 (no smoothing); a=b=0.5 is the SmoothQuant special case.
    """
    return (x_stat.clamp(min=eps) ** a) / (w_stat.clamp(min=eps) ** b)


# ─────────────────────────────────────────────────────────────────────────────
# Channel weighting
# ─────────────────────────────────────────────────────────────────────────────
def outlier_channel_weight(p99_9: torch.Tensor, median: torch.Tensor,
                           gamma: float, eps: float = 1e-8) -> torch.Tensor:
    """w_c = 1 + gamma*(p99.9/median - 1)_+ (the older v2 outlier heuristic)."""
    ratio = p99_9 / (median + eps)
    boost = torch.clamp(ratio - 1.0, min=0.0)
    return 1.0 + gamma * boost


def hessian_channel_weight(E_x2: torch.Tensor, E_w2: torch.Tensor,
                           alpha: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    w_c = (E[X²]_c · E[W²]_c / mean)^α — Hessian-diagonal proxy.

    alpha=0 -> uniform (all ones); alpha=1 -> normalized Hessian (mean 1).
    """
    H_diag = E_x2.float() * E_w2.float() + eps
    H_normalized = H_diag / H_diag.mean().clamp(min=eps)
    return H_normalized.pow(alpha)


def select_channel_weight(p99_9: torch.Tensor, median: torch.Tensor,
                          E_x2: torch.Tensor, E_w2: torch.Tensor,
                          channel_weight: str, gamma: float = 1.0,
                          hessian_alpha: float = 1.0) -> torch.Tensor:
    """Dispatch w_c by `channel_weight` ("outlier" | "hessian")."""
    if channel_weight == "outlier":
        return outlier_channel_weight(p99_9, median, gamma)
    elif channel_weight == "hessian":
        return hessian_channel_weight(E_x2, E_w2, alpha=hessian_alpha)
    else:
        raise ValueError(f"unknown channel_weight: {channel_weight}")


# ─────────────────────────────────────────────────────────────────────────────
# Weight-error term (per-channel dW²)
# ─────────────────────────────────────────────────────────────────────────────
def batched_weight_dW_term(W_stacked: torch.Tensor, s: torch.Tensor,
                           batch_size: int = 0) -> torch.Tensor:
    """
    total_dW_term [h_in] = sum_N mean_out( (fakeq(W·s)/s - W)² ), for a stack
    of weights [N, h_out, h_in] sharing input scale s [h_in]. batch_size>0 chunks
    the N dim for peak-memory control (identical result).
    """
    s_row = s[None, None, :]
    s_clamp = s_row.clamp(min=1e-12)

    if batch_size <= 0 or batch_size >= W_stacked.shape[0]:
        W_smooth = W_stacked * s_row
        W_q = int8_fake_quantize(W_smooth).float()
        dW = W_q / s_clamp - W_stacked
        return (dW ** 2).mean(dim=1).sum(dim=0)

    total = torch.zeros(W_stacked.shape[-1], dtype=W_stacked.dtype,
                        device=W_stacked.device)
    for start in range(0, W_stacked.shape[0], batch_size):
        chunk = W_stacked[start:start + batch_size]
        W_smooth = chunk * s_row
        W_q = int8_fake_quantize(W_smooth).float()
        dW = W_q / s_clamp - chunk
        total += (dW ** 2).mean(dim=1).sum(dim=0)
    return total


def batched_weight_dW_term_multi_s(W_stacked: torch.Tensor, s_batch: torch.Tensor,
                                   ab_chunk: int = 9) -> torch.Tensor:
    """
    dW_term for B different scales in one (chunked) pass.

    W_stacked [N, h_out, h_in], s_batch [B, h_in] -> [B, h_in]. Row i equals
    batched_weight_dW_term(W_stacked, s_batch[i]).
    """
    N, h_out, h_in = W_stacked.shape
    B = s_batch.shape[0]
    out = torch.zeros(B, h_in, dtype=W_stacked.dtype, device=W_stacked.device)

    for start in range(0, B, ab_chunk):
        s_chunk = s_batch[start:start + ab_chunk]
        s_view = s_chunk.view(-1, 1, 1, h_in)
        s_clamp = s_view.clamp(min=1e-12)
        W_smooth = W_stacked.unsqueeze(0) * s_view
        W_q = int8_fake_quantize(W_smooth).float()
        dW = W_q / s_clamp - W_stacked.unsqueeze(0)
        out[start:start + ab_chunk] = (dW ** 2).mean(dim=2).sum(dim=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Output-reconstruction objective  (AWQ-style, extended to joint W+X / W8A8)
# ─────────────────────────────────────────────────────────────────────────────
def compute_output_recon_objective(weights: List[torch.Tensor],
                                   Y_refs: List[torch.Tensor],
                                   X_sample: torch.Tensor,
                                   x_stat: torch.Tensor, w_stat: torch.Tensor,
                                   a: float, b: float, eps: float = 1e-12) -> float:
    """
    J(a,b) = sum_i mean( (fakeq(X/s) @ fakeq(W_i·s)^T - Y_ref_i)² ).

    int8_fake_quantize returns dequantized values, so the s factor cancels:
    fakeq(X/s) @ fakeq(W·s)^T ≈ X @ W^T. No extra s multiplication here.
    """
    s = make_smooth_scale(x_stat, w_stat, a, b, eps)

    X_smooth = X_sample.float() / s.unsqueeze(0).clamp(min=1e-12)
    X_q = int8_fake_quantize(X_smooth).float()

    total_mse = 0.0
    for W, Y_ref in zip(weights, Y_refs):
        W_smooth = W.float() * s.unsqueeze(0)
        W_q = int8_fake_quantize(W_smooth).float()
        Y_fake = X_q @ W_q.T
        total_mse += ((Y_fake - Y_ref) ** 2).mean().item()
    return total_mse


def compute_output_recon_objective_batched_ab(weights: List[torch.Tensor],
                                              Y_refs: List[torch.Tensor],
                                              X_sample: torch.Tensor,
                                              x_stat: torch.Tensor,
                                              w_stat: torch.Tensor,
                                              a_batch: torch.Tensor,
                                              b_batch: torch.Tensor,
                                              eps: float = 1e-12,
                                              ab_chunk: int = 9) -> torch.Tensor:
    """
    Batched output-recon J(a,b) for B candidates -> [B].

    Semantically equivalent to calling compute_output_recon_objective B times
    (tests assert this). Chunks the B dim; iterates weights individually since
    Y_refs may have different shapes.
    """
    device = x_stat.device
    a_t = a_batch.to(device).float()
    b_t = b_batch.to(device).float()
    B = a_t.shape[0]

    x_clamp = x_stat.clamp(min=eps)
    w_clamp = w_stat.clamp(min=eps)
    log_x = torch.log(x_clamp)
    log_w = torch.log(w_clamp)
    log_s = log_x.unsqueeze(0) * a_t.unsqueeze(1) - log_w.unsqueeze(0) * b_t.unsqueeze(1)
    s_batch = torch.exp(log_s)

    X_f = X_sample.float()
    J_total = torch.zeros(B, device=device, dtype=torch.float32)

    for start in range(0, B, ab_chunk):
        end = min(start + ab_chunk, B)
        s_chunk = s_batch[start:end]
        s_chunk_clamp = s_chunk.clamp(min=1e-12)

        X_smooth = X_f.unsqueeze(0) / s_chunk_clamp.unsqueeze(1)
        X_q = int8_fake_quantize(X_smooth).float()

        mse_chunk = torch.zeros(end - start, device=device, dtype=torch.float32)
        for W, Y_ref in zip(weights, Y_refs):
            W_f = W.float()
            W_smooth = W_f.unsqueeze(0) * s_chunk.unsqueeze(1)
            W_q = int8_fake_quantize(W_smooth).float()
            Y_fake = torch.bmm(X_q, W_q.transpose(1, 2))
            Y_ref_f = Y_ref.to(device).float()
            mse_chunk = mse_chunk + ((Y_fake - Y_ref_f.unsqueeze(0)) ** 2).mean(dim=(1, 2))
        J_total[start:end] = mse_chunk

    return J_total


def compute_output_recon_objective_batched_ab_distributed(
        weight_groups, Y_refs_per_dev, devices, X_sample, x_stat, w_stat,
        a_batch, b_batch, eps=1e-12, ab_chunk=9):
    """
    Distributed batched output-recon J for B (a,b) candidates -> [B] on CPU.

    weight_groups[i] / Y_refs_per_dev[i] live on devices[i]. Each device computes
    the MSE for its weight subset (one cross-device reduce instead of B), partials
    summed on CPU. Semantically equal to the single-device batched objective over
    the concatenated weights (tests assert this).
    """
    from concurrent.futures import ThreadPoolExecutor

    n_dev = len(devices)
    B = int(a_batch.shape[0])
    partials = [None] * n_dev

    def _worker(idx):
        dev = devices[idx]
        x_dev = x_stat.to(dev).float()
        w_dev = w_stat.to(dev).float()
        a_t = a_batch.to(dev).float()
        b_t = b_batch.to(dev).float()
        log_s = (torch.log(x_dev.clamp(min=eps)).unsqueeze(0) * a_t.unsqueeze(1)
                 - torch.log(w_dev.clamp(min=eps)).unsqueeze(0) * b_t.unsqueeze(1))
        s_batch = torch.exp(log_s)
        s_clamp = s_batch.clamp(min=1e-12)
        X_f = X_sample.to(dev).float()
        local = torch.zeros(B, device=dev, dtype=torch.float32)
        for start in range(0, B, ab_chunk):
            end = min(start + ab_chunk, B)
            s_chunk = s_batch[start:end]
            X_smooth = X_f.unsqueeze(0) / s_clamp[start:end].unsqueeze(1)
            X_q = int8_fake_quantize(X_smooth).float()
            chunk = torch.zeros(end - start, device=dev, dtype=torch.float32)
            for W, Y_ref in zip(weight_groups[idx], Y_refs_per_dev[idx]):
                W_smooth = W.to(dev).float().unsqueeze(0) * s_chunk.unsqueeze(1)
                W_q = int8_fake_quantize(W_smooth).float()
                Y_fake = torch.bmm(X_q, W_q.transpose(1, 2))
                chunk = chunk + ((Y_fake - Y_ref.to(dev).float().unsqueeze(0)) ** 2).mean(dim=(1, 2))
            local[start:end] = chunk
        partials[idx] = local.cpu()

    if n_dev == 1:
        _worker(0)
    else:
        with ThreadPoolExecutor(max_workers=n_dev) as ex:
            list(ex.map(_worker, range(n_dev)))

    total = torch.zeros(B, dtype=torch.float32)
    for p in partials:
        total = total + p
    return total
