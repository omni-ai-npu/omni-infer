# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""YaRN (Yet another RoPE extensioN) scaling rotary embedding implementation.

Credits to Peng et al. github.com/jquesnelle/yarn

This module provides the YaRN scaling method for extending rotary positional embeddings.
"""

import torch
import torch_npu

from vllm.model_executor.layers.rotary_embedding import YaRNScalingRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import yarn_get_mscale
from .common import CachedCosSinMixin, apply_rotary_emb_full_dim


@YaRNScalingRotaryEmbedding.register_oot
class NPUYaRNScalingRotaryEmbedding(
    CachedCosSinMixin, YaRNScalingRotaryEmbedding
):
    """RotaryEmbedding extended with YaRN method.

    YaRN (Yet another RoPE extensioN) is a method for extending the context
    length of models using rotary positional embeddings.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

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
        apply_yarn_scaling: bool = True,
        truncate: bool = True,
    ) -> None:

        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, scaling_factor, dtype,
                         extrapolation_factor=extrapolation_factor,
                         attn_factor=attn_factor,
                         beta_fast=beta_fast,
                         beta_slow=beta_slow, 
                         apply_yarn_scaling=apply_yarn_scaling, 
                         truncate=truncate)
        
        self._set_cos_sin_cache()
        

    def _set_cos_sin_cache(self) -> None:
        """Compute the cos and sin cache separately with YaRN scaling."""
        inv_freq = self._compute_inv_freq(self.scaling_factor).npu()
        self.max_len = self.max_position_embeddings * self.scaling_factor
        t = torch.arange(self.max_len, device=inv_freq.device, dtype=torch.float32)

        # Adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            freqs = torch.outer(t, inv_freq).float()
            emb = torch.stack((freqs, freqs), dim=-1).reshape(freqs.shape[0], -1)

        emb_cos = torch.cos(emb) * self.mscale
        emb_sin = torch.sin(emb) * self.mscale

        self.register_buffer("cos_cached", emb_cos.to(dtype=torch.get_default_dtype()), persistent=False)
        self.register_buffer("sin_cached", emb_sin.to(dtype=torch.get_default_dtype()), persistent=False)

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
