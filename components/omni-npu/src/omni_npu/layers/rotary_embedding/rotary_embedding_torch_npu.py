# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Torch NPU rotary embedding implementation.

This module provides the basic rotary embedding implementation for NPU devices.
It includes optimizations for different rotary_dim values and uses torch_npu
operations when available.
"""
from typing import Optional

import torch
import torch_npu

from vllm.platforms import current_platform
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.logger import init_logger
logger = init_logger(__name__)

from .common import CachedCosSinMixin, apply_rotary_emb_full_dim

NEOX_ROTARY_COEFF = 2


@RotaryEmbedding.register_oot
class NPURotaryEmbedding(CachedCosSinMixin, RotaryEmbedding):
    """Rotary positional embedding for NPU devices.

    This implementation provides optimized rotary embedding application for NPU
    devices. It automatically selects the appropriate implementation based on
    the rotary_dim value:
    - rotary_dim < head_size: Uses native PyTorch implementation (for ChatGLM-style)
    - rotary_dim != 128: Uses small operations implementation
    - rotary_dim == 128: Uses fused torch_npu.npu_apply_rotary_pos_emb operation
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ):
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                        is_neox_style, dtype)
        self.max_len = self.max_position_embeddings

        self._set_cos_sin_cache(device=current_platform.device_type, dtype=dtype)

    def _set_cos_sin_cache(self, device, dtype) -> None:
        """Compute the cos and sin cache separately.

        Returns:
            Tuple of (cos, sin) tensors, each of shape [max_position_embeddings, rotary_dim]
        """
        inv_freq = self._compute_inv_freq(self.base).npu()
        t = torch.arange(self.max_len, device=device, dtype=inv_freq.dtype)

        # Adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq) # shape: [max_len, rotary_dim/2]
            emb = torch.cat((freqs, freqs), dim=-1) # shape: [max_len, rotary_dim]
        else:
            freqs = torch.outer(t, inv_freq).float() # shape: [max_len, rotary_dim/2]
            emb = torch.stack((freqs, freqs), dim=-1).reshape(freqs.shape[0], -1) # shape: [max_len, rotary_dim]

        self.register_buffer("cos_cached", torch.cos(emb).to(dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb).to(dtype=dtype), persistent=False)

    def _forward_ascend_ops_and_small_ops(
        self,
        position_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward using ascend ops with small operations.

        Used when rotary_dim >= head_size but rotary_dim != 128.
        """
        assert position_ids.dim() == 1
        cos = torch.index_select(self.cos_cached, dim=0, index=position_ids)
        sin = torch.index_select(self.sin_cached, dim=0, index=position_ids)
        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()
        q_embed = apply_rotary_emb_full_dim(query, cos, sin, self.is_neox_style)
        k_embed = apply_rotary_emb_full_dim(key, cos, sin, self.is_neox_style)
        return q_embed.flatten(-2), k_embed.flatten(-2)

    def _forward_fused_ops(
        self,
        position_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward using fused torch_npu npu_apply_rotary_pos_emb operations with TND format.
        Used when rotary_dim == 128.
        """
        if cos is None or sin is None:
            cos = torch.index_select(
                self.cos_cached, dim=0, index=position_ids.view(-1)
            ).unsqueeze(1)
            sin = torch.index_select(
                self.sin_cached, dim=0, index=position_ids.view(-1)
            ).unsqueeze(1)
        else:
            if cos.ndim == 4:
                cos = cos.squeeze(2)
            if sin.ndim == 4:
                sin = sin.squeeze(2)
        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()

        # Use npu_apply_rotary_pos_emb
        q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin, 'TND')

        # Flatten results
        q_embed_flat = q_embed.flatten(1, 2)
        k_embed_flat = k_embed.flatten(1, 2)
        return q_embed_flat, k_embed_flat

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: Optional[torch.Tensor] = None,
        output_cos_sin: Optional[bool] = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation.

        Automatically dispatches to the appropriate implementation based on rotary_dim.
        """
        # Adapt for ChatGLM: dim = head_size / 2
        if self.rotary_dim < self.head_size:
            if self.cos_sin_cache.device != query.device:
                self.cos_sin_cache = self.cos_sin_cache.npu()
            if output_cos_sin: # return query, key, cos_cache, sin_cache
                return self._forward_npu_interleave_rope(positions, query, key)
            elif self.rotary_dim == 64 and self.head_size == 128:
                q_embed, k_embed = self._forward_fused_ops(
                    positions, query, key, cos=cos, sin=sin
                )
            else:
                q_embed, k_embed = super().forward_native(positions, query, key)
        elif self.rotary_dim != 128:
            # torch_npu.npu_apply_rotary_pos_emb last dim must be 128, resort to small ops when not
            q_embed, k_embed = self._forward_ascend_ops_and_small_ops(positions, query, key)
        elif self.is_neox_style:
            q_embed, k_embed = self._forward_fused_ops(positions, query, key)
        else:
            q_embed, k_embed = self._forward_ascend_ops_and_small_ops(positions, query, key)

        return q_embed, k_embed

    def _forward_npu_interleave_rope(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos_cache = torch.cat((cos, cos), dim=1)
        sin_cache = torch.cat((sin, sin), dim=1)
    
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = torch_npu.npu_interleave_rope(
            query_rot.unsqueeze(2),
            cos_cache.unsqueeze(1).unsqueeze(1),
            sin_cache.unsqueeze(1).unsqueeze(1)
        ).squeeze(2)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
     
        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            key_rot = torch_npu.npu_interleave_rope(
                key_rot.unsqueeze(2), 
                cos_cache.unsqueeze(1).unsqueeze(1),
                sin_cache.unsqueeze(1).unsqueeze(1)
            ).squeeze(2)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

        return query, key, cos_cache, sin_cache

