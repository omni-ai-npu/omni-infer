# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Linear scaling rotary embedding implementation.

This module provides the linear scaling method for extending rotary
positional embeddings. Credits to the Reddit user /u/kaiokendev.
"""

import torch
import torch_npu
from vllm.platforms import current_platform
from vllm.model_executor.layers.rotary_embedding.linear_scaling_rope import LinearScalingRotaryEmbedding
from .common import CachedCosSinMixin, apply_rotary_emb_full_dim


@LinearScalingRotaryEmbedding.register_oot
class NPULinearScalingRotaryEmbedding(
    CachedCosSinMixin, LinearScalingRotaryEmbedding
):
    """RotaryEmbedding extended with linear scaling.

    Linear scaling extends the context length by scaling the positions
    linearly.

    Credits to the Reddit user /u/kaiokendev
    """
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factors: list[float] | float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, scaling_factors, dtype)
        self._set_cos_sin_cache(device=current_platform.device_type, dtype=dtype)
        
    def _set_cos_sin_cache(self, device, dtype) -> None:
        """Compute the cos and sin cache separately with linear scaling."""
        inv_freq = self._compute_inv_freq(self.base).npu()
        cos_list: list[torch.Tensor] = []
        sin_list: list[torch.Tensor] = []
        offsets: list[int] = []
        for scaling_factor in self.scaling_factors:
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, device=device, dtype=inv_freq.dtype)
            t = t / scaling_factor

            # Adapt for ascend rope
            if self.is_neox_style:
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
            else:
                freqs = torch.outer(t, inv_freq).float()
                emb = torch.stack((freqs, freqs), dim=-1).reshape(freqs.shape[0], -1)

            if not cos_list:
                offset = 0
            else:
                offset = offsets[-1] + cos_list[-1].shape[0]
            offsets.append(offset)
            cos_list.append(torch.cos(emb).to(dtype=dtype))
            sin_list.append(torch.sin(emb).to(dtype=dtype))

        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        self.register_buffer("cos_cached", torch.cat(cos_list, dim=0), persistent=False)
        self.register_buffer("sin_cached", torch.cat(sin_list, dim=0), persistent=False)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos = self.cos_cached.index_select(0, positions)
        sin = self.sin_cached.index_select(0, positions)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        if key is None:
            query_rot = apply_rotary_emb_full_dim(
                query_rot, cos, sin, self.is_neox_style
            )
            query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
            return query, None

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        if self.rotary_dim == 128 and self.is_neox_style:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            query_rot, key_rot = torch_npu.npu_apply_rotary_pos_emb(
                query_rot, key_rot, cos, sin, "TND"
            )
        else:
            query_rot = apply_rotary_emb_full_dim(
                query_rot, cos, sin, self.is_neox_style
            )
            key_rot = apply_rotary_emb_full_dim(
                key_rot, cos, sin, self.is_neox_style
            )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key
