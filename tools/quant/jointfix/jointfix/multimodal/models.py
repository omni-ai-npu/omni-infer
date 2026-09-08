# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Torch-only, TP=1 calibration forwards for OpenPangu Omni ViT/Audio layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from jointfix.core.stats import AccumActStats, StatsConfig


def _linear(tensors: Dict[str, torch.Tensor], base: str, device) -> nn.Linear:
    weight = tensors[f"{base}.weight"]
    bias = tensors.get(f"{base}.bias")
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                      device=device, dtype=weight.dtype)
    layer.weight.data.copy_(weight.to(device))
    if bias is not None:
        layer.bias.data.copy_(bias.to(device))
    layer.requires_grad_(False)
    return layer


def _norm(tensors: Dict[str, torch.Tensor], base: str, eps: float, device) -> nn.LayerNorm:
    weight = tensors[f"{base}.weight"]
    layer = nn.LayerNorm(weight.numel(), eps=eps,
                         elementwise_affine=True, device=device, dtype=weight.dtype)
    layer.weight.data.copy_(weight.to(device))
    bias = tensors.get(f"{base}.bias")
    if bias is None:
        layer.bias.data.zero_()
    else:
        layer.bias.data.copy_(bias.to(device))
    layer.requires_grad_(False)
    return layer


def install_collectors(modules: Dict[str, nn.Module], stats_config: StatsConfig):
    collectors, handles = {}, []

    for key, module in modules.items():
        def hook(_module, inputs, collector_key=key):
            x = inputs[0]
            if collector_key not in collectors:
                collectors[collector_key] = AccumActStats(x.shape[-1], stats_config)
            collectors[collector_key].update(x)
        handles.append(module.register_forward_pre_hook(hook))
    return collectors, handles


def _segmented_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         segments: List[int], scale: float) -> torch.Tensor:
    outputs = []
    offset = 0
    for length in segments:
        if length <= 0:
            continue
        qs = q[offset:offset + length].transpose(0, 1).float()
        ks = k[offset:offset + length].transpose(0, 1).float()
        vs = v[offset:offset + length].transpose(0, 1).float()
        scores = torch.matmul(qs, ks.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, vs).transpose(0, 1).to(v.dtype))
        offset += length
    if offset != q.shape[0]:
        raise ValueError(f"attention segments cover {offset} tokens, input has {q.shape[0]}")
    return torch.cat(outputs, dim=0)


def _rotate_interleave(x: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)


class VisionBlock(nn.Module):
    def __init__(self, tensors: Dict[str, torch.Tensor], prefix: str,
                 config: dict, device):
        super().__init__()
        self.prefix = prefix
        self.hidden = int(config["hidden_size"])
        self.num_heads = int(config["num_heads"])
        self.head_dim = self.hidden // self.num_heads
        self.norm1 = _norm(tensors, f"{prefix}.norm1", 1e-6, device)
        self.norm2 = _norm(tensors, f"{prefix}.norm2", 1e-6, device)
        self.qkv = _linear(tensors, f"{prefix}.attn.qkv", device)
        self.proj = _linear(tensors, f"{prefix}.attn.proj", device)
        self.up_proj = _linear(tensors, f"{prefix}.mlp.up_proj", device)
        self.down_proj = _linear(tensors, f"{prefix}.mlp.down_proj", device)

    def collector_modules(self):
        return {
            "attn_in": self.qkv,
            "attn_out": self.proj,
            "ffn_in": self.up_proj,
            "ffn_out": self.down_proj,
        }

    def forward(self, hidden: torch.Tensor, *, cos: torch.Tensor,
                sin: torch.Tensor, segments: List[int]) -> torch.Tensor:
        residual = hidden
        qkv = self.qkv(self.norm1(hidden))
        q, k, v = qkv.reshape(-1, 3, self.num_heads, self.head_dim).unbind(1)
        cos = cos.to(device=q.device, dtype=q.dtype).unsqueeze(1)
        sin = sin.to(device=q.device, dtype=q.dtype).unsqueeze(1)
        q = q * cos + _rotate_interleave(q) * sin
        k = k * cos + _rotate_interleave(k) * sin
        attn = _segmented_attention(q, k, v, segments, self.head_dim ** -0.5)
        attn = attn.reshape(-1, self.hidden)
        hidden = residual + self.proj(attn)
        residual = hidden
        hidden = self.up_proj(self.norm2(hidden))
        hidden = F.gelu(hidden)
        hidden = self.down_proj(hidden)
        return residual + hidden


class AudioLayer(nn.Module):
    def __init__(self, tensors: Dict[str, torch.Tensor], prefix: str,
                 config: dict, device):
        super().__init__()
        self.hidden = int(config["d_model"])
        self.num_heads = int(config["encoder_attention_heads"])
        self.head_dim = self.hidden // self.num_heads
        self.norm1 = _norm(tensors, f"{prefix}.self_attn_layer_norm", 1e-5, device)
        self.norm2 = _norm(tensors, f"{prefix}.final_layer_norm", 1e-5, device)
        self.q_proj = _linear(tensors, f"{prefix}.self_attn.q_proj", device)
        self.k_proj = _linear(tensors, f"{prefix}.self_attn.k_proj", device)
        self.v_proj = _linear(tensors, f"{prefix}.self_attn.v_proj", device)
        self.out_proj = _linear(tensors, f"{prefix}.self_attn.out_proj", device)
        self.fc1 = _linear(tensors, f"{prefix}.fc1", device)
        self.fc2 = _linear(tensors, f"{prefix}.fc2", device)

    def collector_modules(self):
        # q/k/v share the same input; one hook is sufficient for the group.
        return {
            "attn_in": self.q_proj,
            "attn_out": self.out_proj,
            "ffn_in": self.fc1,
            "ffn_out": self.fc2,
        }

    def forward(self, hidden: torch.Tensor, *, segments: List[int]) -> torch.Tensor:
        residual = hidden
        normed = self.norm1(hidden)
        q = self.q_proj(normed).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(normed).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(normed).reshape(-1, self.num_heads, self.head_dim)
        attn = _segmented_attention(q, k, v, segments, self.head_dim ** -0.5)
        attn = attn.reshape(-1, self.hidden)
        hidden = residual + self.out_proj(attn)
        residual = hidden
        hidden = self.fc1(self.norm2(hidden))
        hidden = F.gelu(hidden)
        hidden = self.fc2(hidden)
        return residual + hidden


@dataclass
class VisionState:
    hidden: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    segments: List[int]


def prepare_vision_state(pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                         tensors: Dict[str, torch.Tensor], config: dict,
                         device) -> VisionState:
    patch_weight = tensors["visual.patch_embed.proj.weight"].to(device)
    input_size = patch_weight.numel() // patch_weight.shape[0]
    x = pixel_values.to(device)
    if x.shape[-1] != input_size:
        patch_area = int(config["patch_size"]) ** 2
        x = torch.cat(
            [x.reshape(-1, patch_area), x.reshape(-1, patch_area)], dim=-1
        ).reshape(-1, input_size)
    hidden = x @ patch_weight.reshape(patch_weight.shape[0], -1).T
    if config.get("use_norm_pre", False):
        norm = _norm(tensors, "visual.layernorm_pre",
                     float(config.get("norm_pre_eps", 1e-5)), device)
        hidden = norm(hidden)

    head_dim = int(config["hidden_size"]) // int(config["num_heads"])
    rope = config.get("rope_scaling") or {}
    ratios = rope.get("mrope_section", [4, 6, 6])
    half = head_dim // 2
    unit = half // sum(ratios)
    sizes = [unit * value for value in ratios]
    theta = float(config.get("rope_theta", 10000.0))
    inv = [1.0 / theta ** (torch.arange(size, dtype=torch.float32) / size) for size in sizes]

    frequencies, segments = [], []
    merge = int(config["spatial_merge_size"])
    for t_raw, h_raw, w_raw in grid_thw.tolist():
        t, h, w = int(t_raw), int(h_raw), int(w_raw)
        hpos = torch.arange(h).reshape(h, 1).repeat(1, w)
        wpos = torch.arange(w).reshape(1, w).repeat(h, 1)
        hpos = hpos.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).reshape(-1)
        wpos = wpos.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).reshape(-1)
        hpos = hpos.repeat(t).float()
        wpos = wpos.repeat(t).float()
        # Runtime image/video-chunk path intentionally resets temporal position.
        tpos = torch.zeros_like(hpos)
        frequencies.append(torch.cat([
            torch.einsum("s,d->sd", tpos, inv[0]),
            torch.einsum("s,d->sd", hpos, inv[1]),
            torch.einsum("s,d->sd", wpos, inv[2]),
        ], dim=-1))
        segments.extend([h * w] * t)
    freq = torch.cat(frequencies, dim=0)
    cos_half, sin_half = freq.cos(), freq.sin()
    # torch_npu rotary_mode="interleave": duplicate each frequency adjacently.
    cos = torch.stack((cos_half, cos_half), dim=-1).flatten(-2)
    sin = torch.stack((sin_half, sin_half), dim=-1).flatten(-2)
    return VisionState(hidden.cpu(), cos.cpu(), sin.cpu(), segments)


@dataclass
class AudioState:
    hidden: torch.Tensor
    segments: List[int]


def prepare_audio_state(input_features: torch.Tensor, feature_lens: torch.Tensor,
                        tensors: Dict[str, torch.Tensor], config: dict,
                        device) -> AudioState:
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoderConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniAudioEncoder,
    )

    cfg_dict = dict(config)
    cfg_dict["encoder_layers"] = 0
    cfg_dict["_attn_implementation"] = "eager"
    cfg = Qwen2_5OmniAudioEncoderConfig(**cfg_dict)
    frontend = Qwen2_5OmniAudioEncoder(cfg).to(device=device, dtype=torch.bfloat16).eval()
    frontend.conv1.weight.data.copy_(tensors["audio_tower.conv1.weight"].to(device))
    frontend.conv1.bias.data.copy_(tensors["audio_tower.conv1.bias"].to(device))
    frontend.conv2.weight.data.copy_(tensors["audio_tower.conv2.weight"].to(device))
    frontend.conv2.bias.data.copy_(tensors["audio_tower.conv2.bias"].to(device))

    feature_lens = feature_lens.to(device)
    features = input_features.to(device)
    chunk_num = torch.ceil(feature_lens / (frontend.n_window * 2)).long()
    chunk_lengths = torch.tensor(
        [frontend.n_window * 2] * int(chunk_num.sum().item()),
        dtype=torch.long, device=device,
    )
    tail = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail] = feature_lens % (frontend.n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, frontend.n_window * 2, chunk_lengths)
    chunk_list = features.split(chunk_lengths.tolist(), dim=1)
    padded, mask, mask_after = frontend.padded_and_mask_function(
        chunk_list, chunk_lengths, padding_value=0, padding_side="right"
    )
    hidden = F.gelu(frontend.conv1(padded)) * mask
    hidden = F.gelu(frontend.conv2(hidden)).transpose(1, 2)
    hidden = hidden + frontend.positional_embedding.positional_embedding[
        :hidden.shape[1]
    ].unsqueeze(0).to(device=device, dtype=hidden.dtype)
    hidden = hidden[mask_after]
    segments = ((chunk_lengths - 1) // 2 + 1).tolist()
    del frontend
    return AudioState(hidden.cpu(), [int(v) for v in segments])
