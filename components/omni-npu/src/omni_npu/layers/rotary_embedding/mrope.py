# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

import torch
import torch_npu
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope


@MRotaryEmbedding.register_oot
class NPUMRotaryEmbedding(MRotaryEmbedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        is_prefill: bool = True,
        attn_metadata=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        if is_prefill:
            assert positions.ndim == 1 or positions.ndim == 2
            assert key is not None
            self._match_cos_sin_cache_dtype(query)
            num_tokens = positions.shape[-1]
            if attn_metadata is not None and hasattr(attn_metadata, '_npu_cached_cos_sin'):
                cos, sin = attn_metadata._npu_cached_cos_sin
            else:
                cos_sin = self.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                if positions.ndim == 2:
                    assert self.mrope_section
                    if self.mrope_interleaved:
                        cos = apply_interleaved_rope(cos, self.mrope_section)
                        sin = apply_interleaved_rope(sin, self.mrope_section)
                    else:
                        cos = torch.cat(
                            [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                            dim=-1,
                        )
                        sin = torch.cat(
                            [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                            dim=-1,
                        )
                cos = cos.contiguous()
                sin = sin.contiguous()
                if attn_metadata is not None:
                    attn_metadata._npu_cached_cos_sin = (cos, sin)
            query_shape = query.shape
            query = query.view(num_tokens, -1, self.head_size)
            query_rot = query[..., : self.rotary_dim]
            query_pass = query[..., self.rotary_dim :]
            query_rot = self.apply_rotary_emb(query_rot, cos, sin)
            query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., : self.rotary_dim]
            key_pass = key[..., self.rotary_dim :]
            key_rot = self.apply_rotary_emb(key_rot, cos, sin)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
            return query, key
        else:
            positions = positions[0]
            mrope_section = [0, 0, 0
                            ] if positions.ndim == 1 else self.mrope_section

            if self.cos_sin_cache.device != query.device:  # type: ignore
                self.cos_sin_cache = self.cos_sin_cache.to(  # type: ignore
                    query.device)  # type: ignore

            if self.cos_sin_cache.dtype != query.dtype:  # type: ignore
                self.cos_sin_cache = self.cos_sin_cache.to(  # type: ignore
                    query.dtype)  # type: ignore
            query, key = torch_npu.npu_mrope(positions.contiguous(),
                                            query.contiguous(),
                                            key.contiguous(),
                                            self.cos_sin_cache.contiguous(),
                                            self.head_size,
                                            mrope_section=mrope_section,
                                            rotary_mode='half')

            return query, key
