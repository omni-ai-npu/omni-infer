# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

import torch
import torch_npu
from vllm.platforms import current_platform
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import DeepseekScalingRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import (
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
    rotate_gptj,
    rotate_neox,
)

from .common import CachedCosSinMixin


@DeepseekScalingRotaryEmbedding.register_oot
class NPUDeepseekScalingRotaryEmbedding(
    CachedCosSinMixin, DeepseekScalingRotaryEmbedding
):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base,
            is_neox_style, scaling_factor, dtype,
            extrapolation_factor=extrapolation_factor,
            attn_factor=attn_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            mscale=mscale,
            mscale_all_dim=mscale_all_dim
        )
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=current_platform.device_type,
            dtype=dtype
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.rotary_dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2, dtype=torch.float32).to(
            device=device
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len * self.scaling_factor, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False
        )

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation equivalent to forward()."""
        assert key is not None
        self._match_cos_sin_cache_dtype(query)
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
            query_rot = query_rot.unsqueeze(0)
            key_rot = key_rot.unsqueeze(0)
            query_rot, key_rot = torch_npu.npu_apply_rotary_pos_emb(query_rot, key_rot, cos, sin, rotary_mode='half')
            query_rot = query_rot.squeeze(0)
            key_rot = key_rot.squeeze(0)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

            rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
            query_rot = query_rot * cos + rotate_fn(query_rot) * sin
            key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key
