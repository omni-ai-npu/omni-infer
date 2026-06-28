# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import math
import torch
import torch_npu

from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding
from .common import CachedCosSinMixin, apply_rotary_emb_full_dim

LOW_FREQ_FACTOR = 1
HIGH_FREQ_FACTOR = 4
OLD_CONTEXT_LEN = 8192
ROPE_ROTARY_FACTOR = 64
SCALE_FACTOR = 8


@Llama3RotaryEmbedding.register_oot
class NPULlama3RotaryEmbedding(CachedCosSinMixin, Llama3RotaryEmbedding):
    """Llama3 Rotary Embedding with linear scaling for NPU devices.

    This implementation provides the Llama3-specific rotary embedding
    application with linear scaling for NPU devices.
    """
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, 
            is_neox_style, dtype, scaling_factor, low_freq_factor,
            high_freq_factor, orig_max_position)
    
        self._set_cos_sin_cache()

    def _set_cos_sin_cache(self) -> None:
        """Compute the cos and sin cache separately with Llama3 scaling."""
        inv_freq = self._compute_inv_freq(self.base).npu()
        t = torch.arange(self.max_position_embeddings, device=inv_freq.device,
                         dtype=inv_freq.dtype)

        # Adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            freqs = torch.outer(t, inv_freq).float()
            emb = torch.stack((freqs, freqs), dim=-1).reshape(freqs.shape[0], -1)

        self.register_buffer("cos_cached", torch.cos(emb).to(dtype=torch.get_default_dtype()), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb).to(dtype=torch.get_default_dtype()), persistent=False)

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
        query_pass = query[..., self.rotary_dim :]
        if key is None:
            query_rot = apply_rotary_emb_full_dim(
                query_rot, cos, sin, self.is_neox_style
            )
            query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
            return query, None

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
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
    
