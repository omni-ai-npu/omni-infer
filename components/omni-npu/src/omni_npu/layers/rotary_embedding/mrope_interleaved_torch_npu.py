# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Torch NPU MRotary Embedding Interleaved implementation.

This module provides the basic mrotary embedding interleaved implementation 
for NPU devices. It includes optimizations for different rotary_dim values 
and uses torch_npu operations when available.
"""
from typing import Optional

import torch
import torch_npu

from vllm.logger import init_logger
logger = init_logger(__name__)

from vllm.model_executor.layers.rotary_embedding import MRotaryEmbeddingInterleaved


@MRotaryEmbeddingInterleaved.register_oot
class NPUMRotaryEmbeddingInterleaved(MRotaryEmbeddingInterleaved):
    """Rotary Embedding with Multimodal Sections and Interleaved Support."""

    def _rebuild_pos_emb(
        self,
        positions: torch.Tensor,
        output_cos_sin: Optional[bool] = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleave the rotary embedding"""
        cos_sin = self.cos_sin_cache[positions]
        mrope_section_3d = [1] * len(self.mrope_dim)
        mrope_dim = self.mrope_dim
        cos_sin = torch.cat(
            [m[mrope_dim[i]]
                for i, m in enumerate(cos_sin.split(mrope_section_3d, dim=-1))],
            dim=-1,
        )
        cos, sin = cos_sin.chunk(2, dim=-1)
        from einops import rearrange
        if self.rotary_mode == 'half' or output_cos_sin:
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        elif self.rotary_mode == 'interleave':
            cos = rearrange(torch.stack((cos, cos), dim=-1), "... d two -> ...(d two)", two=2)
            sin = rearrange(torch.stack((sin, sin), dim=-1), "... d two -> ...(d two)", two=2)
        else:
            raise ValueError("only support half or interleave")
        cos = cos.reshape(-1, 1, 1, self.rotary_dim)
        sin = sin.reshape(-1, 1, 1, self.rotary_dim)

        return cos, sin

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        output_cos_sin: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with interleaved rotary embedding."""
        if self.layer_counts % self.num_hidden_layers_cache == 0:
            cos, sin = self._rebuild_pos_emb(positions, output_cos_sin)
            self.layer_cache = (cos, sin)
            self.layer_counts = 0
        else:
            cos, sin = self.layer_cache
        self.layer_counts += 1

        num_tokens = query.shape[0]

        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        if output_cos_sin:
            query_rot = torch_npu.npu_interleave_rope(query_rot.unsqueeze(2), cos, sin).squeeze(2)
        else:
            query_rot = torch_npu.npu_rotary_mul(
                query_rot.unsqueeze(2).contiguous(), cos, sin, rotary_mode=self.rotary_mode
            ).squeeze(2)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(num_tokens, -1)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            if output_cos_sin:
                key_rot = torch_npu.npu_interleave_rope(key_rot.unsqueeze(2), cos, sin).squeeze(2)
            else:
                key_rot = torch_npu.npu_rotary_mul(
                    key_rot.unsqueeze(2).contiguous(), cos, sin, rotary_mode=self.rotary_mode
                ).squeeze(2)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(num_tokens, -1)

        if output_cos_sin:
            return query, key, cos, sin
        else:
            return query, key
