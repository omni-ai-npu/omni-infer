# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Torch-only Pangu decoder modules for quantization calibration.

A plain-PyTorch reimplementation of the Pangu decoder layer (DSA attention, MoME,
MHC multi-stream, LightningIndexer, param-sink, MoE) — normal torch modules and
hooks rather than serving-kernel parity. Vendored into jointfix as the Pangu
backend's model implementation; loaded lazily by backends/pangu.py.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


_NPU_DSA_IMPORT_ATTEMPTED = False
_LOGGER = logging.getLogger(__name__)


def _ensure_npu_dsa_custom_ops_registered() -> None:
    # Open-source build: the internal NPU DSA custom-op registration is omitted
    # (it referenced an internal serving-runtime package). On CUDA/CPU these fused
    # ops are unused. To run on Ascend NPU with fused DSA kernels, register them
    # here by importing your own op module.
    return


@dataclass(frozen=True)
class PanguTorchLayerSpec:
    layer_idx: int
    attention_type: str
    is_dsa: bool
    is_moe: bool
    sliding_window: int | None
    hidden_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    intermediate_size: int
    moe_intermediate_size: int | None
    n_routed_experts: int | None
    num_experts_per_tok: int | None
    use_mhc: bool
    use_mome: bool
    mhc_num_stream: int
    mhc_recur_norm: int
    router_sliding_window: int
    has_block_post_layernorm: bool


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _cfg_get(config: dict[str, Any], name: str, default: Any = None) -> Any:
    return config.get(name, default)


def build_pangu_torch_layer_specs(model_dir: str) -> list[PanguTorchLayerSpec]:
    config = _load_json(os.path.join(model_dir, "config.json"))
    num_layers = int(config["num_hidden_layers"])
    dsa_layers = set(config.get("dsa_layers") or [])
    swa_layer_list = list(config.get("swa_layers") or [])
    swa_layers = set(swa_layer_list)
    sliding_windows = config.get("sliding_window_list") or []
    first_moe_layer = int(config.get("first_k_dense_replace", num_layers))
    block_post_layers = set(config.get("block_post_layernorm_idx") or [])
    mhc_num_stream = int(config.get("mhc_num_stream", 1))
    use_mhc = bool(config.get("use_mhc", False)) and mhc_num_stream > 1
    use_mome = bool(config.get("use_mome", False))

    specs = []
    for layer_idx in range(num_layers):
        is_dsa = layer_idx in dsa_layers
        is_swa = layer_idx in swa_layers
        sliding_window = None
        if is_swa:
            pos_in_swa = swa_layer_list.index(layer_idx)
            sliding_window = int(sliding_windows[pos_in_swa])
        specs.append(
            PanguTorchLayerSpec(
                layer_idx=layer_idx,
                attention_type="dsa_attention"
                if is_dsa
                else ("sliding_attention" if is_swa else "full_attention"),
                is_dsa=is_dsa,
                is_moe=layer_idx >= first_moe_layer
                and config.get("n_routed_experts") is not None,
                sliding_window=sliding_window if is_swa else None,
                hidden_size=int(config["hidden_size"]),
                num_attention_heads=int(config["num_attention_heads"]),
                q_lora_rank=int(config["q_lora_rank"]),
                kv_lora_rank=int(config["kv_lora_rank"]),
                qk_nope_head_dim=int(config["qk_nope_head_dim"]),
                qk_rope_head_dim=int(config["qk_rope_head_dim"]),
                v_head_dim=int(config["v_head_dim"]),
                intermediate_size=int(config["intermediate_size"]),
                moe_intermediate_size=(
                    int(config["moe_intermediate_size"])
                    if config.get("moe_intermediate_size") is not None
                    else None
                ),
                n_routed_experts=(
                    int(config["n_routed_experts"])
                    if config.get("n_routed_experts") is not None
                    else None
                ),
                num_experts_per_tok=(
                    int(config["num_experts_per_tok"])
                    if config.get("num_experts_per_tok") is not None
                    else None
                ),
                use_mhc=use_mhc,
                use_mome=use_mome,
                mhc_num_stream=mhc_num_stream,
                mhc_recur_norm=int(config.get("mhc_recur_norm", 1)),
                router_sliding_window=int(config.get("router_sliding_window", 0) or 0),
                has_block_post_layernorm=layer_idx in block_post_layers,
            )
        )
    return specs


class _SafeTensorStore:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        index = _load_json(index_path)
        self.weight_map = index["weight_map"]

    def has_tensor(self, name: str) -> bool:
        return name in self.weight_map

    def get_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        shard_path = os.path.join(self.model_dir, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            return f.get_tensor(name)


class PanguTorchRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


class PanguTorchMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x).float()
        up = self.up_proj(x).float()
        hidden = (F.silu(gate) * up).to(x.dtype)
        return self.down_proj(hidden)


class PanguTorchExpert(PanguTorchMLP):
    """Pangu expert using the same projection layout as its MLP."""


class PanguTorchMoE(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden_size = int(config["hidden_size"])
        intermediate_size = int(config["moe_intermediate_size"])
        n_experts = int(config["n_routed_experts"])
        shared_intermediate = intermediate_size * int(config.get("n_shared_experts", 1))

        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.e_score_correction_bias = (
            nn.Parameter(torch.empty(n_experts, dtype=torch.float32))
            if config.get("router_enable_expert_bias", False)
            else None
        )
        self.experts = nn.ModuleList(
            [PanguTorchExpert(hidden_size, intermediate_size) for _ in range(n_experts)]
        )
        self.shared_experts = PanguTorchMLP(hidden_size, shared_intermediate)
        self.num_experts_per_tok = int(config["num_experts_per_tok"])
        self.norm_topk_prob = bool(config.get("norm_topk_prob", False))
        self.routed_scaling_factor = float(config.get("routed_scaling_factor", 1.0))

    def _router_logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.gate.weight.float())

    def _select_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self._router_logits(x)
        if hasattr(self, "_jointfix_route_collector"):
            self._jointfix_last_router_logits = router_logits.detach()
        if os.getenv("PANGU_TORCH_NPU_TOPK", "0") == "1" and x.device.type == "npu":
            try:
                import torch_npu

                topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                    router_logits.to(torch.float32),
                    k=self.num_experts_per_tok,
                    bias=self.e_score_correction_bias,
                    k_group=1,
                    group_count=1,
                    group_select_mode=1,
                    renorm=0,
                    norm_type=1,
                    routed_scaling_factor=self.routed_scaling_factor,
                    eps=1e-20,
                )
                return topk_ids.to(torch.long), topk_weights.to(torch.float32)
            except (AttributeError, RuntimeError, TypeError) as error:
                _LOGGER.debug("NPU top-k is unavailable; using the torch fallback: %s", error)

        scores = torch.sigmoid(router_logits)
        choice_scores = scores
        if self.e_score_correction_bias is not None:
            choice_scores = choice_scores + self.e_score_correction_bias.float()
        topk_ids = torch.topk(
            choice_scores,
            k=self.num_experts_per_tok,
            dim=-1,
        ).indices
        topk_weights = scores.gather(1, topk_ids)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_ids, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, original_shape[-1])
        topk_ids, topk_weights = self._select_topk(x)

        token_is_text = getattr(self, "_jointfix_token_is_text", None)
        if token_is_text is not None:
            token_is_text = token_is_text.reshape(-1).bool()
            if x.shape[0] % token_is_text.numel() != 0:
                raise ValueError(
                    f"MoE rows {x.shape[0]} are not divisible by modality rows "
                    f"{token_is_text.numel()}"
                )
            token_is_text = token_is_text.repeat_interleave(
                x.shape[0] // token_is_text.numel())
            collector = getattr(self, "_jointfix_route_collector", None)
            logits = getattr(self, "_jointfix_last_router_logits", None)
            if collector is not None and logits is not None:
                collector.update(
                    topk_ids, topk_weights,
                    logits.gather(1, topk_ids), token_is_text,
                )

        routed = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            token_pos, rank_pos = torch.where(topk_ids == expert_idx)
            if token_pos.numel() == 0:
                continue
            if token_is_text is not None:
                expert.down_proj._jointfix_token_is_text = token_is_text.index_select(0, token_pos)
                expert.down_proj._jointfix_priority = topk_weights[token_pos, rank_pos]
            expert_out = expert(x.index_select(0, token_pos))
            routed.index_add_(
                0,
                token_pos,
                expert_out * topk_weights[token_pos, rank_pos].to(expert_out.dtype).unsqueeze(-1),
            )

        if token_is_text is not None:
            self.shared_experts.down_proj._jointfix_token_is_text = token_is_text
            self.shared_experts.down_proj._jointfix_priority = torch.ones_like(
                token_is_text, dtype=torch.float32)
        out = routed + self.shared_experts(x)
        return out.reshape(original_shape)


class PanguTorchMHC(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_stream: int,
        eps: float,
        recur_norm: int,
        *,
        pre_only: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stream = num_stream
        self.recur_norm = recur_norm
        self.hc_eps = 1e-6
        self.pre_only = pre_only
        self.eps = eps if pre_only else self.hc_eps
        input_size = hidden_size * num_stream
        output_size = num_stream if pre_only else num_stream * (num_stream + 2)
        self.phi = nn.Linear(input_size, output_size, bias=False)
        self.norm_gamma = nn.Parameter(torch.empty(input_size))
        if pre_only:
            self.branch_alpha_pre = nn.Parameter(torch.empty(1))
            self.branch_beta_pre = nn.Parameter(torch.empty(num_stream))
        else:
            self.branch_alpha = nn.Parameter(torch.empty(3))
            self.branch_beta = nn.Parameter(torch.empty(num_stream * (num_stream + 2)))

    def mhc_pre(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        dtype = hidden_states.dtype
        x = hidden_states.view(-1, self.num_stream, self.hidden_size)
        flat = x.reshape(-1, self.num_stream * self.hidden_size).float()
        normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(normed * self.norm_gamma.float(), self.phi.weight.float())

        if self.pre_only:
            h_pre = torch.sigmoid(
                mixes * self.branch_alpha_pre.float()
                + self.branch_beta_pre.float().view(1, self.num_stream)
            ) + self.hc_eps
            hidden = torch.sum(h_pre.view(-1, self.num_stream, 1) * x.float(), dim=1)
            return hidden.to(dtype), None, None

        h_pre, h_post, h_res = mixes.split(
            [self.num_stream, self.num_stream, self.num_stream * self.num_stream],
            dim=-1,
        )
        alpha_pre, alpha_post, alpha_res = self.branch_alpha.float().view(-1).split([1, 1, 1])
        beta_pre, beta_post, beta_res = self.branch_beta.float().view(-1).split(
            [self.num_stream, self.num_stream, self.num_stream * self.num_stream]
        )
        h_pre = torch.sigmoid(h_pre * alpha_pre + beta_pre) + self.hc_eps
        h_post = 2 * torch.sigmoid(h_post * alpha_post + beta_post)
        h_res = h_res.view(-1, self.num_stream, self.num_stream)
        h_res = h_res * alpha_res + beta_res.view(self.num_stream, self.num_stream)
        hidden = torch.sum(h_pre.view(-1, self.num_stream, 1) * x.float(), dim=1)
        return hidden.to(dtype), h_post, h_res

    def mhc_sinkhorn(self, h_res: torch.Tensor | None) -> torch.Tensor | None:
        if h_res is None:
            return None
        h_res = h_res.float().softmax(-1) + self.hc_eps
        h_res = h_res / (h_res.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(max(self.recur_norm - 1, 0)):
            h_res = h_res / (h_res.sum(-1, keepdim=True) + self.hc_eps)
            h_res = h_res / (h_res.sum(-2, keepdim=True) + self.hc_eps)
        return h_res

    def mhc_post(
        self,
        hidden_states: torch.Tensor,
        h_post: torch.Tensor | None,
        residual: torch.Tensor,
        h_res: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.pre_only:
            return residual
        if h_post is None or h_res is None:
            raise ValueError("full MHC post requires h_post and h_res")
        dtype = hidden_states.dtype
        residual = residual.view(-1, self.num_stream, self.hidden_size)
        hidden = (
            h_post.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2)
            + torch.sum(h_res.float().unsqueeze(-1) * residual.float().unsqueeze(-2), dim=-3)
        )
        return hidden.to(dtype).reshape(-1, self.num_stream * self.hidden_size)


class PanguTorchMOMEConv(nn.Module):
    def __init__(self, dim: int, kernel_width: int):
        super().__init__()
        self.dim = dim
        self.kernel_width = kernel_width
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_width <= 0:
            return x
        dtype = x.dtype
        original_shape = x.shape
        seq = x.reshape(-1, original_shape[-1]).float()
        weight = self.weight.to(device=seq.device, dtype=torch.float32)
        padded = F.pad(seq.transpose(0, 1).unsqueeze(0), (self.kernel_width - 1, 0))
        conv = F.conv1d(padded, weight, groups=self.dim).squeeze(0).transpose(0, 1)
        if self.kernel_width > 1:
            conv[: self.kernel_width - 1] = 0
        return (conv + seq).to(dtype).reshape(original_shape)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return ((x * cos.float()) + (_rotate_half(x) * sin.float())).to(dtype)


def _apply_rotary_dtype(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class PanguTorchIndexer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.index_topk = int(config["index_topk"])
        self.index_n_heads = int(config["index_n_heads"])
        self.index_head_dim = int(config["index_head_dim"])
        self.wq_b = nn.Linear(
            int(config["q_lora_rank"]),
            self.index_n_heads * self.index_head_dim,
            bias=False,
        )
        self.wk = nn.Linear(int(config["hidden_size"]), self.index_head_dim, bias=False)
        self.k_norm = PanguTorchRMSNorm(self.index_head_dim, float(config["rms_norm_eps"]))
        self.weights_proj = nn.Linear(int(config["hidden_size"]), self.index_n_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.wq_b(q_lora).view(-1, self.index_n_heads, self.index_head_dim)
        k = self.k_norm(self.wk(hidden_states))
        if cos is not None and sin is not None:
            q_pe, q_nope = torch.split(
                q,
                [cos.shape[-1], self.index_head_dim - cos.shape[-1]],
                dim=-1,
            )
            k_pe, k_nope = torch.split(
                k,
                [cos.shape[-1], self.index_head_dim - cos.shape[-1]],
                dim=-1,
            )
            q = torch.cat([_apply_rotary(q_pe, cos, sin), q_nope], dim=-1)
            k = torch.cat([_apply_rotary(k_pe, cos, sin), k_nope], dim=-1)
        weights = self.weights_proj(hidden_states)
        return q, k, weights


def torch_lightning_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    sparse_count: int,
    causal: bool = True,
) -> torch.Tensor:
    """Torch reference for the unquantized LightningIndexer scoring formula."""
    seq_len = q.shape[0]
    topk = min(max(int(sparse_count), 1), seq_len)
    scores = torch.einsum("tgd,sd->tgs", q.float(), k.float())
    scores = torch.relu(scores)
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=1)
    if causal:
        q_pos = torch.arange(seq_len, device=scores.device).unsqueeze(1)
        k_pos = torch.arange(seq_len, device=scores.device).unsqueeze(0)
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    return torch.topk(scores, k=topk, dim=-1).indices


class PanguTorchMLAAttention(nn.Module):
    def __init__(self, config: dict[str, Any], spec: PanguTorchLayerSpec, *, full_layer: bool = False):
        super().__init__()
        hidden_size = spec.hidden_size
        self.num_heads = spec.num_attention_heads
        self.qk_nope_head_dim = spec.qk_nope_head_dim
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.qk_head_dim = spec.qk_nope_head_dim + spec.qk_rope_head_dim
        self.v_head_dim = spec.v_head_dim
        self.q_lora_rank = spec.q_lora_rank
        self.kv_lora_rank = spec.kv_lora_rank
        self.sliding_window = spec.sliding_window
        self.is_dsa = spec.is_dsa
        self.use_mome = bool(full_layer and spec.use_mome and spec.router_sliding_window > 0)
        self.index_topk = int(config.get("index_topk", 0)) if spec.is_dsa else 0
        self.rope_theta = float(config.get("rope_theta", 10000.0))
        self.scaling = self.qk_head_dim ** -0.5

        self.q_a_proj = nn.Linear(hidden_size, spec.q_lora_rank, bias=False)
        self.q_a_layernorm = PanguTorchRMSNorm(spec.q_lora_rank, float(config["rms_norm_eps"]))
        self.q_b_proj = nn.Linear(
            spec.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            spec.kv_lora_rank + spec.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = PanguTorchRMSNorm(spec.kv_lora_rank, float(config["rms_norm_eps"]))
        self.kv_b_proj = nn.Linear(
            spec.kv_lora_rank,
            self.num_heads * (spec.qk_nope_head_dim + spec.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * spec.v_head_dim, hidden_size, bias=False)
        sink_count = int(config.get("param_sink_number", 0))
        self.param_sink_compressed_kv = nn.Parameter(
            torch.empty(sink_count, spec.kv_lora_rank)
        )
        self.param_sink_k_pe = nn.Parameter(torch.empty(sink_count, spec.qk_rope_head_dim))
        self._last_dsa_backend = "none"
        self._last_dsa_fallback_reason = ""
        if self.is_dsa:
            self.indexer = PanguTorchIndexer(config)
        if self.use_mome:
            kernel_width = spec.router_sliding_window
            self.qa_conv = PanguTorchMOMEConv(spec.q_lora_rank, kernel_width)
            self.compresskv_conv = PanguTorchMOMEConv(spec.kv_lora_rank, kernel_width)
            self.o_conv = PanguTorchMOMEConv(self.num_heads * spec.v_head_dim, kernel_width)

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        half_dim = self.qk_rope_head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = k_pos <= q_pos
        if self.sliding_window is not None:
            mask = mask & (k_pos >= q_pos - self.sliding_window + 1)
        return mask

    def _kv_cache_style_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.kv_a_layernorm.eps)
        y = y.to(dtype)
        weight = self.kv_a_layernorm.weight.to(device=x.device, dtype=dtype)
        return y * weight

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = x.shape[0]
        q_lora = self.q_a_proj(x)
        if self.use_mome:
            q_lora = self.qa_conv(q_lora)
        q_lora = self.q_a_layernorm(q_lora)
        q = self.q_b_proj(q_lora).view(seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        kv = self.kv_a_proj_with_mqa(x)
        k_nope_latent, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.use_mome:
            k_nope_latent = self.compresskv_conv(k_nope_latent)
        return q_lora, q_nope, q_pe, k_nope_latent, k_pe

    def _forward_swa_or_full(
        self,
        x: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope_latent: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = x.shape[0]
        kv_up = self.kv_b_proj(self._kv_cache_style_rmsnorm(k_nope_latent)).view(
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, v = torch.split(kv_up, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos, sin = self._cos_sin(seq_len, x.device, q_pe.dtype)
        q_pe = _apply_rotary(q_pe, cos, sin)
        k_pe = _apply_rotary_dtype(k_pe, cos, sin).unsqueeze(1).expand(-1, self.num_heads, -1)
        if self.param_sink_compressed_kv.numel() > 0:
            sink_latent = self.kv_a_layernorm(self.param_sink_compressed_kv)
            sink_kv = self.kv_b_proj(sink_latent).view(
                -1,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            sink_k_nope, sink_v = torch.split(
                sink_kv,
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1,
            )
            sink_k_pe = self.param_sink_k_pe.view(-1, 1, self.qk_rope_head_dim).expand(
                -1,
                self.num_heads,
                -1,
            )
            k_nope = torch.cat([sink_k_nope.to(k_nope.dtype), k_nope], dim=0)
            k_pe = torch.cat([sink_k_pe.to(k_pe.dtype), k_pe], dim=0)
            v = torch.cat([sink_v.to(v.dtype), v], dim=0)

        scores = torch.einsum("thd,shd->hts", q_nope, k_nope)
        scores = scores + torch.einsum("thr,shr->hts", q_pe, k_pe)
        scores = scores * self.scaling
        attn_mask = self._attention_mask(seq_len, x.device)
        sink_count = k_nope.shape[0] - seq_len
        if sink_count > 0:
            sink_mask = torch.ones(
                seq_len,
                sink_count,
                dtype=torch.bool,
                device=x.device,
            )
            attn_mask = torch.cat([sink_mask, attn_mask], dim=-1)
        attn_mask = attn_mask.unsqueeze(0)
        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q_nope.dtype)
        probs = torch.where(attn_mask, probs, torch.zeros_like(probs))
        attn = torch.einsum("hts,shv->thv", probs, v).reshape(seq_len, -1)
        if self.use_mome:
            attn = self.o_conv(attn)
        return self.o_proj(attn)

    def _w_uk_t(self) -> torch.Tensor:
        weight = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        return weight[:, : self.qk_nope_head_dim, :].contiguous()

    def _w_uv(self) -> torch.Tensor:
        weight = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        return weight[:, self.qk_nope_head_dim:, :].transpose(1, 2).contiguous()

    def _npu_dsa_latent(
        self,
        q_nope_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope_latent: torch.Tensor,
        k_pe: torch.Tensor,
        topk_indices: torch.Tensor,
        sink_k_nope: torch.Tensor,
        sink_k_pe: torch.Tensor,
    ) -> torch.Tensor | None:
        self._last_dsa_backend = "torch"
        self._last_dsa_fallback_reason = ""
        if os.getenv("PANGU_TORCH_USE_NPU_DSA", "0") != "1":
            self._last_dsa_fallback_reason = "env_disabled"
            return None
        if q_nope_latent.device.type != "npu" or torch.is_grad_enabled():
            self._last_dsa_fallback_reason = "unsupported_device_or_grad"
            return None
        if not hasattr(torch.ops, "custom"):
            self._last_dsa_fallback_reason = "missing_torch_ops_custom"
            return None
        op = getattr(torch.ops.custom, "npu_ai_infra_sparse_flash_attention_pioneer", None)
        if op is None:
            _ensure_npu_dsa_custom_ops_registered()
            op = getattr(torch.ops.custom, "npu_ai_infra_sparse_flash_attention_pioneer", None)
        if op is None:
            self._last_dsa_fallback_reason = "missing_sparse_attention_op"
            return None

        seq_len = q_nope_latent.shape[0]
        block_size = 128
        num_blocks = (seq_len + block_size - 1) // block_size
        padded_tokens = num_blocks * block_size
        pad_k_nope = torch.zeros(
            (padded_tokens, k_nope_latent.shape[-1]),
            dtype=k_nope_latent.dtype,
            device=k_nope_latent.device,
        )
        pad_k_pe = torch.zeros(
            (padded_tokens, k_pe.shape[-1]),
            dtype=k_pe.dtype,
            device=k_pe.device,
        )
        pad_k_nope[:seq_len] = k_nope_latent
        pad_k_pe[:seq_len] = k_pe
        key_cache = (
            torch.cat([pad_k_nope, pad_k_pe], dim=-1)
            .view(num_blocks, block_size, 1, -1)
            .contiguous()
        )
        value_cache = torch.empty(
            (num_blocks, block_size, 1, k_nope_latent.shape[-1]),
            dtype=k_nope_latent.dtype,
            device=k_nope_latent.device,
        )
        query = torch.cat([q_nope_latent, q_pe], dim=-1).contiguous()
        sink_kv = torch.cat([sink_k_nope, sink_k_pe], dim=-1).unsqueeze(1).contiguous()
        sink_value = sink_k_nope.unsqueeze(1).contiguous()
        query_cumlens = torch.tensor([seq_len], dtype=torch.int32, device=q_nope_latent.device)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=q_nope_latent.device)
        block_table = torch.arange(num_blocks, dtype=torch.int32, device=q_nope_latent.device).view(1, -1)
        try:
            out = op(
                query=query,
                key=key_cache,
                value=value_cache,
                sparse_indices=topk_indices.to(torch.int32).view(seq_len, 1, -1).contiguous(),
                scale_value=self.scaling,
                sparse_block_size=1,
                block_table=block_table,
                actual_seq_lengths_query=query_cumlens,
                actual_seq_lengths_kv=seq_lens,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                attention_mode=2,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                key_sink=sink_kv,
                value_sink=sink_value,
            )[0]
        except RuntimeError as exc:
            self._last_dsa_fallback_reason = f"{type(exc).__name__}: {exc}"
            return None
        self._last_dsa_backend = "npu_ai_infra_sparse_flash_attention_pioneer"
        self._last_dsa_fallback_reason = ""
        return out.view_as(q_nope_latent)

    def _forward_dsa(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope_latent: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = x.shape[0]
        cos, sin = self._cos_sin(seq_len, x.device, q_pe.dtype)
        q_pe = _apply_rotary(q_pe, cos, sin)
        k_pe = _apply_rotary(k_pe, cos, sin)
        q_nope_latent = torch.einsum(
            "thd,hdr->thr",
            q_nope.float(),
            self._w_uk_t().float(),
        ).to(q_nope.dtype)
        k_nope_latent = self.kv_a_layernorm(k_nope_latent)

        indexer_q, indexer_k, indexer_weights = self.indexer(x, q_lora, cos, sin)
        topk_indices = torch_lightning_indexer_topk(
            indexer_q,
            indexer_k,
            indexer_weights,
            self.index_topk,
        )
        sink_k_nope = self.kv_a_layernorm(self.param_sink_compressed_kv).to(
            dtype=q_nope_latent.dtype,
            device=x.device,
        )
        sink_k_pe = self.param_sink_k_pe.to(dtype=q_pe.dtype, device=x.device)
        attn_latent = self._npu_dsa_latent(
            q_nope_latent,
            q_pe,
            k_nope_latent,
            k_pe,
            topk_indices,
            sink_k_nope,
            sink_k_pe,
        )
        if attn_latent is not None:
            attn = torch.einsum("thr,hrv->thv", attn_latent.float(), self._w_uv().float())
            attn = attn.to(x.dtype).reshape(seq_len, -1)
            if self.use_mome:
                attn = self.o_conv(attn)
            return self.o_proj(attn)

        gathered_k_nope = k_nope_latent.index_select(0, topk_indices.reshape(-1)).view(
            seq_len,
            topk_indices.shape[1],
            self.kv_lora_rank,
        )
        gathered_k_pe = k_pe.index_select(0, topk_indices.reshape(-1)).view(
            seq_len,
            topk_indices.shape[1],
            self.qk_rope_head_dim,
        )
        valid = topk_indices <= torch.arange(seq_len, device=x.device).unsqueeze(1)

        if sink_k_nope.numel() > 0:
            gathered_k_nope = torch.cat(
                [sink_k_nope.unsqueeze(0).expand(seq_len, -1, -1), gathered_k_nope],
                dim=1,
            )
            gathered_k_pe = torch.cat(
                [sink_k_pe.unsqueeze(0).expand(seq_len, -1, -1), gathered_k_pe],
                dim=1,
            )
            valid = torch.cat(
                [
                    torch.ones(
                        (seq_len, sink_k_nope.shape[0]),
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    valid,
                ],
                dim=1,
            )

        scores = torch.einsum("qhr,qkr->qhk", q_nope_latent.float(), gathered_k_nope.float())
        scores = scores + torch.einsum("qhr,qkr->qhk", q_pe.float(), gathered_k_pe.float())
        scores = scores * self.scaling
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(valid.unsqueeze(1), probs, torch.zeros_like(probs))
        attn_latent = torch.einsum("qhk,qkr->qhr", probs, gathered_k_nope.float()).to(q_nope_latent.dtype)
        attn = torch.einsum("thr,hrv->thv", attn_latent.float(), self._w_uv().float())
        attn = attn.to(x.dtype).reshape(seq_len, -1)
        if self.use_mome:
            attn = self.o_conv(attn)
        return self.o_proj(attn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        x = hidden_states.reshape(-1, original_shape[-1])
        q_lora, q_nope, q_pe, k_nope_latent, k_pe = self._project_qkv(x)
        if self.is_dsa:
            out = self._forward_dsa(x, q_lora, q_nope, q_pe, k_nope_latent, k_pe)
        else:
            out = self._forward_swa_or_full(x, q_nope, q_pe, k_nope_latent, k_pe)
        return out.reshape(original_shape)


class PanguTorchDecoderLayer(nn.Module):
    def __init__(self, config: dict[str, Any], spec: PanguTorchLayerSpec, *, full_layer: bool = False):
        super().__init__()
        self.spec = spec
        self.full_layer = full_layer
        self.layer_idx = spec.layer_idx
        self.attention_type = spec.attention_type
        self.hidden_size = spec.hidden_size
        self.mhc_num_stream = spec.mhc_num_stream
        self.use_mhc = bool(full_layer and spec.use_mhc)
        self.has_block_post_layernorm = bool(full_layer and spec.has_block_post_layernorm)
        self.input_layernorm = PanguTorchRMSNorm(spec.hidden_size, float(config["rms_norm_eps"]))
        self.self_attn = PanguTorchMLAAttention(config, spec, full_layer=full_layer)
        self.post_attention_layernorm = PanguTorchRMSNorm(
            spec.hidden_size,
            float(config["rms_norm_eps"]),
        )
        self.sandwich_norm = bool(config.get("sandwich_norm", False))
        if self.sandwich_norm:
            self.pre_mlp_layernorm = PanguTorchRMSNorm(
                spec.hidden_size,
                float(config["rms_norm_eps"]),
            )
            self.post_mlp_layernorm = PanguTorchRMSNorm(
                spec.hidden_size,
                float(config["rms_norm_eps"]),
            )
        self.mlp = PanguTorchMoE(config) if spec.is_moe else PanguTorchMLP(
            spec.hidden_size,
            spec.intermediate_size,
        )
        if self.use_mhc:
            self.attn_mhc_module = PanguTorchMHC(
                spec.hidden_size,
                spec.mhc_num_stream,
                float(config["rms_norm_eps"]),
                spec.mhc_recur_norm,
            )
            self.mlp_mhc_module = PanguTorchMHC(
                spec.hidden_size,
                spec.mhc_num_stream,
                float(config["rms_norm_eps"]),
                spec.mhc_recur_norm,
            )
        if self.has_block_post_layernorm:
            block_hidden = spec.hidden_size * spec.mhc_num_stream if self.use_mhc else spec.hidden_size
            self.block_post_layernorm = PanguTorchRMSNorm(
                block_hidden,
                float(config["rms_norm_eps"]),
            )

    def forward(self, hidden_states: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        if self.full_layer:
            return self._forward_full_layer(hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        if self.sandwich_norm:
            hidden_states = self.pre_mlp_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)
        return hidden_states + residual

    def _flatten_hidden_states(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if hidden_states.dim() == 4:
            bsz, seq_len, num_stream, hidden = hidden_states.shape
            return hidden_states.reshape(bsz * seq_len, num_stream, hidden), bsz, seq_len
        is_mhc_stream_layout = (
            hidden_states.dim() == 3
            and self.use_mhc
            and hidden_states.shape[1] == self.mhc_num_stream
            and hidden_states.shape[0] != 1
        )
        if is_mhc_stream_layout:
            return hidden_states, 1, hidden_states.shape[0]
        if hidden_states.dim() == 3:
            bsz, seq_len, hidden = hidden_states.shape
            return hidden_states.reshape(bsz * seq_len, hidden), bsz, seq_len
        if hidden_states.dim() == 2:
            return hidden_states, 1, hidden_states.shape[0]
        raise ValueError(f"Unexpected hidden_states shape: {tuple(hidden_states.shape)}")

    def _unflatten_hidden_states(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        expected_tokens = batch_size * seq_len
        if hidden_states.dim() == 2:
            token_count = hidden_states.shape[0]
            if token_count == expected_tokens:
                return hidden_states.view(batch_size, seq_len, -1)
            if batch_size > 0 and token_count % batch_size == 0:
                return hidden_states.view(batch_size, token_count // batch_size, -1)
            return hidden_states
        if hidden_states.dim() == 3:
            token_count = hidden_states.shape[0]
            if token_count == expected_tokens:
                return hidden_states.view(batch_size, seq_len, hidden_states.shape[1], hidden_states.shape[2])
            if batch_size > 0 and token_count % batch_size == 0:
                return hidden_states.view(
                    batch_size,
                    token_count // batch_size,
                    hidden_states.shape[1],
                    hidden_states.shape[2],
                )
        return hidden_states

    def _forward_full_layer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, batch_size, seq_len = self._flatten_hidden_states(hidden_states)
        if self.use_mhc and hidden_states.dim() == 2:
            if hidden_states.shape[-1] == self.hidden_size * self.mhc_num_stream:
                hidden_states = hidden_states.view(-1, self.mhc_num_stream, self.hidden_size)
            elif hidden_states.shape[-1] == self.hidden_size:
                hidden_states = hidden_states.view(-1, 1, self.hidden_size).repeat(
                    1,
                    self.mhc_num_stream,
                    1,
                )
            else:
                raise ValueError(
                    "MHC full-layer path expected last dim "
                    f"{self.hidden_size} or {self.hidden_size * self.mhc_num_stream}, "
                    f"got {hidden_states.shape[-1]}"
                )

        residual = hidden_states.clone()
        if self.use_mhc:
            hidden_states, h_post, h_res = self.attn_mhc_module.mhc_pre(hidden_states)

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_mhc:
            h_res = self.attn_mhc_module.mhc_sinkhorn(h_res)
            hidden_states = self.attn_mhc_module.mhc_post(hidden_states, h_post, residual, h_res)
        else:
            hidden_states = hidden_states + residual

        residual = hidden_states.clone()
        if self.use_mhc:
            hidden_states, h_post, h_res = self.mlp_mhc_module.mhc_pre(hidden_states)

        if self.sandwich_norm:
            hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        if self.use_mhc:
            h_res = self.mlp_mhc_module.mhc_sinkhorn(h_res)
            hidden_states = self.mlp_mhc_module.mhc_post(hidden_states, h_post, residual, h_res)
        else:
            hidden_states = hidden_states + residual

        if self.has_block_post_layernorm and self.use_mhc:
            hidden_states = self.block_post_layernorm(
                hidden_states.view(-1, self.mhc_num_stream * self.hidden_size)
            )
            hidden_states = hidden_states.view(-1, self.mhc_num_stream, self.hidden_size)
        elif not self.use_mhc and hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, self.hidden_size)

        return self._unflatten_hidden_states(hidden_states, batch_size, seq_len)


def _copy_param(
    module: nn.Module,
    param_name: str,
    tensor: torch.Tensor,
    *,
    preserve_param_dtype: bool = False,
) -> None:
    parts = param_name.split(".")
    target = module
    for part in parts[:-1]:
        target = getattr(target, part)
    param = getattr(target, parts[-1])
    if preserve_param_dtype:
        tensor = tensor.to(device=param.device, dtype=param.dtype)
    param.data = tensor


class _PanguTorchLazyLayer(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        spec: PanguTorchLayerSpec,
        store: _SafeTensorStore,
        *,
        full_layer: bool = False,
    ):
        super().__init__()
        self.config = config
        self.spec = spec
        self.store = store
        self.full_layer = full_layer
        self.attention_type = spec.attention_type
        self._layer: PanguTorchDecoderLayer | None = None

    def _load_tensor(self, suffix: str) -> torch.Tensor:
        return self.store.get_tensor(f"model.layers.{self.spec.layer_idx}.{suffix}")

    def _ensure_loaded(self) -> PanguTorchDecoderLayer:
        if self._layer is not None:
            return self._layer

        layer = PanguTorchDecoderLayer(self.config, self.spec, full_layer=self.full_layer)
        prefix_map = {
            "input_layernorm.weight": "input_layernorm.weight",
            "post_attention_layernorm.weight": "post_attention_layernorm.weight",
            "self_attn.q_a_proj.weight": "self_attn.q_a_proj.weight",
            "self_attn.q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
            "self_attn.q_b_proj.weight": "self_attn.q_b_proj.weight",
            "self_attn.kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight": "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight": "self_attn.o_proj.weight",
            "self_attn.param_sink_compressed_kv": "self_attn.param_sink_compressed_kv",
            "self_attn.param_sink_k_pe": "self_attn.param_sink_k_pe",
        }
        if self.spec.is_dsa:
            prefix_map.update(
                {
                    "self_attn.indexer.wq_b.weight": "self_attn.indexer.wq_b.weight",
                    "self_attn.indexer.wk.weight": "self_attn.indexer.wk.weight",
                    "self_attn.indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
                    "self_attn.indexer.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
                }
            )
        if layer.sandwich_norm:
            prefix_map.update(
                {
                    "pre_mlp_layernorm.weight": "pre_mlp_layernorm.weight",
                    "post_mlp_layernorm.weight": "post_mlp_layernorm.weight",
                }
            )
        if layer.self_attn.use_mome:
            prefix_map.update(
                {
                    "self_attn.qa_conv.weight": "self_attn.qa_conv.weight",
                    "self_attn.compresskv_conv.weight": "self_attn.compresskv_conv.weight",
                    "self_attn.o_conv.weight": "self_attn.o_conv.weight",
                }
            )
        if layer.use_mhc:
            for module_name in ("attn_mhc_module", "mlp_mhc_module"):
                prefix_map.update(
                    {
                        f"{module_name}.branch_alpha": f"{module_name}.branch_alpha",
                        f"{module_name}.branch_beta": f"{module_name}.branch_beta",
                        f"{module_name}.norm_gamma": f"{module_name}.norm_gamma",
                        f"{module_name}.phi.weight": f"{module_name}.phi.weight",
                    }
                )
        if layer.has_block_post_layernorm:
            prefix_map["block_post_layernorm.weight"] = "block_post_layernorm.weight"

        if self.spec.is_moe:
            prefix_map.update(
                {
                    "mlp.gate.weight": "mlp.gate.weight",
                    "mlp.shared_experts.gate_proj.weight": "mlp.shared_experts.gate_proj.weight",
                    "mlp.shared_experts.up_proj.weight": "mlp.shared_experts.up_proj.weight",
                    "mlp.shared_experts.down_proj.weight": "mlp.shared_experts.down_proj.weight",
                }
            )
            if self.store.has_tensor(f"model.layers.{self.spec.layer_idx}.mlp.e_score_correction_bias"):
                _copy_param(
                    layer,
                    "mlp.e_score_correction_bias",
                    self._load_tensor("mlp.e_score_correction_bias"),
                )
            for expert_idx in range(self.spec.n_routed_experts or 0):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    suffix = f"mlp.experts.{expert_idx}.{proj}.weight"
                    prefix_map[suffix] = suffix
        else:
            prefix_map.update(
                {
                    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
                    "mlp.up_proj.weight": "mlp.up_proj.weight",
                    "mlp.down_proj.weight": "mlp.down_proj.weight",
                }
            )

        for suffix, param_name in prefix_map.items():
            preserve_dtype = (
                ".attn_mhc_module." in f".{param_name}."
                or ".mlp_mhc_module." in f".{param_name}."
                or param_name == "mlp.gate.weight"
            )
            _copy_param(
                layer,
                param_name,
                self._load_tensor(suffix),
                preserve_param_dtype=preserve_dtype,
            )

        self._layer = layer
        return layer

    def unload(self) -> None:
        self._layer = None

    def to(self, *args: Any, **kwargs: Any) -> "_PanguTorchLazyLayer":
        self._ensure_loaded().to(*args, **kwargs)
        return self

    def cpu(self) -> "_PanguTorchLazyLayer":
        if self._layer is not None:
            self._layer.cpu()
        return self

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self._ensure_loaded()(hidden_states, **kwargs)

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True):
        return self._ensure_loaded().named_modules(
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate,
        )

    def named_children(self):
        return self._ensure_loaded().named_children()

    def children(self):
        return self._ensure_loaded().children()

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self._ensure_loaded().named_parameters(prefix=prefix, recurse=recurse)

    def parameters(self, recurse: bool = True):
        return self._ensure_loaded().parameters(recurse=recurse)


class PanguTorchCalibModel(nn.Module):
    def __init__(self, model_dir: str, *, full_layer: bool = False):
        super().__init__()
        self.model_dir = model_dir
        self.is_pangu_torch_calib = True
        self.pangu_torch_full_layer = full_layer
        self.config_dict = _load_json(os.path.join(model_dir, "config.json"))
        self.config = SimpleNamespace(**self.config_dict)
        self.config.use_cache = False
        self.seqlen = min(int(_cfg_get(self.config_dict, "max_position_embeddings", 2048)), 2048)
        self.store = _SafeTensorStore(model_dir)
        self.layer_specs = build_pangu_torch_layer_specs(model_dir)

        embed_weight = self.store.get_tensor("model.embed_tokens.weight")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        self.model.layers = nn.ModuleList(
            [
                _PanguTorchLazyLayer(
                    self.config_dict,
                    spec,
                    self.store,
                    full_layer=full_layer,
                )
                for spec in self.layer_specs
            ]
        )
        self.model.norm = PanguTorchRMSNorm(
            int(self.config_dict["hidden_size"]),
            float(self.config_dict["rms_norm_eps"]),
        )
        norm_name = "model.norm.weight"
        if self.store.has_tensor(norm_name):
            self.model.norm.weight.data = self.store.get_tensor(norm_name)
        if (
            full_layer
            and bool(self.config_dict.get("use_mhc", False))
            and self.store.has_tensor("model.merge_mhc_module.phi.weight")
        ):
            self.model.merge_mhc_module = PanguTorchMHC(
                int(self.config_dict["hidden_size"]),
                int(self.config_dict.get("mhc_num_stream", 1)),
                float(self.config_dict["rms_norm_eps"]),
                int(self.config_dict.get("mhc_recur_norm", 1)),
                pre_only=True,
            )
            for suffix in (
                "branch_alpha_pre",
                "branch_beta_pre",
                "norm_gamma",
                "phi.weight",
            ):
                _copy_param(
                    self.model.merge_mhc_module,
                    suffix,
                    self.store.get_tensor(f"model.merge_mhc_module.{suffix}"),
                    preserve_param_dtype=True,
                )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.embed_tokens(input_ids)
        return self.model.layers[0](hidden_states)


def load_pangu_torch_calib_model(
    model_dir: str,
    hf_token: str | None = None,
    *,
    full_layer: bool = False,
) -> PanguTorchCalibModel:
    del hf_token
    return PanguTorchCalibModel(model_dir, full_layer=full_layer)
