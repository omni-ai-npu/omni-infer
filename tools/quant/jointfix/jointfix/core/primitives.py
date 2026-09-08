# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Quantization primitives — model-agnostic AND method-agnostic.

The bottom of the dependency graph: low-level INT8 quantizers (RTN / GPTQ),
the fake-quant round-trip, and the output-side dispatch.

This module must NOT import from jointfix.methods or jointfix.backends.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Quant layout (which weights get INT8)
# ─────────────────────────────────────────────────────────────────────────────
# Universal skip patterns — apply to LLaMA / Qwen / Mistral and most HF models.
# Model-specific extras (e.g. Pangu's `indexer.*` / `mhc_module.phi`) are supplied
# by the backend via ModelBackend.skip_patterns(), NOT hardcoded here. That keeps
# architecture-specific names out of the method-agnostic core.
UNIVERSAL_SKIP_PATTERNS: List[str] = [
    "embed",
    "q_a_proj",        # MLA low-rank
    "kv_a_proj",       # MLA low-rank
    "kv_b_proj",       # MLA low-rank (input-side)
    "lm_head",
    "shared_head.head",
    "mlp.gate.",       # router gate (scalar logits) — trailing dot ≠ gate_proj
]


def should_quantize(name: str, tensor: torch.Tensor, skip_patterns: List[str]) -> bool:
    """
    True if (name, tensor) is an INT8-quantizable linear weight.

    `skip_patterns` is passed in (backend-provided) rather than read from a module
    global, so each model can extend the universal list without editing core.
    """
    if not name.endswith(".weight"):
        return False
    if tensor.ndim != 2:
        return False
    for pat in skip_patterns:
        if pat in name:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Fake-quant round trip
# ─────────────────────────────────────────────────────────────────────────────
def int8_fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """
    INT8 per-row symmetric fake quantization (round-trip).

    Simulates: x → INT8 quantize → dequantize, matching rtn_quantize exactly.

    Works for both weights [out, in] and activations [tokens, hidden]:
      scale_per_row = amax(|row|) / 127
      q = round(row / scale).clamp(-128, 127)
      x_deq = q * scale

    No external dependencies. All native PyTorch ops (amax, round, clamp).
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x_2d = x.float().reshape(-1, orig_shape[-1])  # [rows, cols]

    scale = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / 127.0
    x_deq = (x_2d / scale).round().clamp(-128, 127) * scale

    return x_deq.reshape(orig_shape).to(orig_dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    rms = xf.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return xf / rms * weight.float()


# ─────────────────────────────────────────────────────────────────────────────
# INT8 quantizers
# ─────────────────────────────────────────────────────────────────────────────
def rtn_quantize(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric RTN INT8. Returns (int8_weight, bf16_scale)."""
    W = W.float()
    scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / 127.0
    q = (W / scale).round_().clamp_(-128, 127).to(torch.int8)
    return q, scale.to(torch.bfloat16)


def gptq_quantize(
    W: torch.Tensor,
    X: torch.Tensor,
    damp: float = 0.01,
    block_size: int = 128,
    scale_dtype: torch.dtype = torch.bfloat16,
    tag: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPTQ INT8 per-row symmetric quantization for output-side weights.

    Falls back to rtn_quantize when:
      - X is empty (dead MoE expert)
      - N < h_in (rank-deficient Hessian)
      - Cholesky fails after 3 retries with doubled damp

    Returns (int8_weight, scale_bfloat16) — same format as rtn_quantize.

    NOTE: emits `[GPTQ-RUN]` on success; RTN fallbacks are silent (grep the [GPTQ-RUN]
    count against the weight count to see how many fell back).
    """
    h_out, h_in = W.shape
    N = X.shape[0]

    # Fallbacks to RTN are SILENT (they fire per-expert on big MoE layers — pure log noise).
    # The positive [GPTQ-RUN] print below is the signal: its absence for a weight means GPTQ
    # fell back. So "did GPTQ run?" is answered by grep -c '[GPTQ-RUN]' vs the weight count.
    if N == 0:                       # dead MoE expert
        return rtn_quantize(W)
    if N < h_in:                     # rank-deficient Hessian
        return rtn_quantize(W)

    # Align W to X's device for GPTQ compute. W typically arrives on CPU (from the
    # layer_tensors dict); X comes from X_sample on the search device (NPU/CUDA).
    # Do all GPTQ math on X's device, then move the final int8_w + scale back to
    # W's original device so the caller's shard-save accounting is unchanged.
    _orig_W_device = W.device
    if W.device != X.device:
        W = W.to(X.device)

    W = W.float().clone()
    X = X.float()

    H = (X.t() @ X) / float(N)
    diag_mean = H.diag().mean().item()
    H.diagonal().add_(damp * diag_mean)

    cur_damp = damp
    Linv = None
    for retry in range(3):
        try:
            L = torch.linalg.cholesky(H, upper=False)
            U = torch.cholesky_inverse(L, upper=False)
            Linv = torch.linalg.cholesky(U, upper=True)
            break
        except RuntimeError:
            if retry == 2:                       # exhausted retries — silent RTN fallback
                return rtn_quantize(W.to(_orig_W_device))
            cur_damp *= 2.0                       # retry with doubled damp (silent)
            H.diagonal().add_(cur_damp * diag_mean)

    if Linv is None:
        return rtn_quantize(W.to(_orig_W_device))

    # GPTQ is now COMMITTED (passed N>=h_in AND Cholesky succeeded). Emit a positive,
    # greppable signal so "did GPTQ actually run?" is answered by presence:
    #   grep -c '\[GPTQ-RUN\]' <log>                 # total real GPTQ executions
    print(f"[GPTQ-RUN] {tag} h_out={h_out} h_in={h_in} N={N}")

    scale_row = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-10) / 127.0

    Q = torch.zeros_like(W, dtype=torch.int8)
    block_size = min(block_size, h_in)

    for blk_start in range(0, h_in, block_size):
        blk_end = min(blk_start + block_size, h_in)
        W_blk = W[:, blk_start:blk_end].clone()
        U_blk = Linv[blk_start:blk_end, blk_start:blk_end]
        err_blk = torch.zeros_like(W_blk)

        for i in range(blk_end - blk_start):
            w_col = W_blk[:, i]
            q_col = (w_col / scale_row.squeeze(1)).round().clamp(-128, 127)
            q_dequant = q_col * scale_row.squeeze(1)
            err_col = (w_col - q_dequant) / U_blk[i, i]
            err_blk[:, i] = err_col

            if i + 1 < blk_end - blk_start:
                W_blk[:, i + 1:] -= err_col.unsqueeze(1) * U_blk[i, i + 1:].unsqueeze(0)

            Q[:, blk_start + i] = q_col.to(torch.int8)

        if blk_end < h_in:
            W[:, blk_end:] -= err_blk @ Linv[blk_start:blk_end, blk_end:]

    return Q.to(_orig_W_device), scale_row.to(scale_dtype).to(_orig_W_device)


def select_write_quantize(
    W: torch.Tensor,
    X_smooth: Optional[torch.Tensor],
    write_quant: str = "gptq",
    gptq_damp: float = 0.01,
    gptq_block_size: int = 128,
    tag: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch quantizer for output-side weights.

    Decoupled from any method config: callers pass loose `write_quant` /
    `gptq_*` values (the smooth method reads them off its JointSearchConfig).

    Args:
        W:        [h_out, h_in] weight already multiplied by smooth scale s.
        X_smooth: [N, h_in] X/s (smoothed input); None forces RTN.
        tag:      weight key for greppable GPTQ run/fallback logging.
    """
    if write_quant == "rtn" or X_smooth is None:
        return rtn_quantize(W)
    elif write_quant == "gptq":
        return gptq_quantize(
            W, X_smooth, damp=gptq_damp, block_size=gptq_block_size, tag=tag,
        )
    else:
        raise ValueError(f"unknown write_quant: {write_quant}")
