# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Pangu V2 Model Family - Common MLA Patch
#
# This patch is shared between:
# - openpangu_v2 (pangu_sink_swa_mla)
# - openpangu_ultra_omni
# - Other Pangu V2 variants
#
# The patch provides StaticSinkMultiHeadLatentAttentionWrapper support for MLA models.

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers import mla
from vllm.model_executor.layers.mla import (
    CacheConfig,
    MLAModules,
    MultiHeadLatentAttentionWrapper,
    QuantizationConfig,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("mlaPatch", mla)
class mlaPatch(VLLMPatch):
    _attr_names_to_apply = ['StaticSinkMultiHeadLatentAttentionWrapper']

    # patch start
    class StaticSinkMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
        def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            scale: float,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: int | None,
            kv_lora_rank: int,
            mla_modules: MLAModules,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
            sink_len: int = 0,
            sliding_window: int | None = None,
            qa_conv: None = None,
            compresskv_conv: None = None,
            o_conv: None = None,
        ) -> None:
            CustomOp.__init__(self)
            self.hidden_size = hidden_size
            self.qk_nope_head_dim = qk_nope_head_dim
            self.qk_rope_head_dim = qk_rope_head_dim
            self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            self.v_head_dim = v_head_dim
            self.q_lora_rank = q_lora_rank
            self.kv_lora_rank = kv_lora_rank
            self.num_heads = num_heads
            self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
            self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
            self.q_a_layernorm = mla_modules.q_a_layernorm
            self.q_b_proj = mla_modules.q_b_proj
            self.q_proj = mla_modules.q_proj
            self.kv_a_layernorm = mla_modules.kv_a_layernorm
            self.kv_b_proj = mla_modules.kv_b_proj
            self.rotary_emb = mla_modules.rotary_emb
            self.o_proj = mla_modules.o_proj
            self.indexer = mla_modules.indexer
            self.indexer_rope_emb = mla_modules.indexer_rotary_emb
            self.is_sparse = mla_modules.is_sparse

            self.qa_conv = qa_conv
            self.compresskv_conv = compresskv_conv
            self.o_conv = o_conv

            if self.indexer is not None:
                assert hasattr(self.indexer, "topk_tokens")
                self.topk_tokens = self.indexer.topk_tokens
                self.topk_indices_buffer = mla_modules.topk_indices_buffer
                
            from vllm.model_executor.layers.attention.static_sink_attention import StaticSinkMLAAttention
            self.mla_attn = StaticSinkMLAAttention(
                num_heads=self.num_heads,
                scale=scale,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                kv_b_proj=self.kv_b_proj,
                use_sparse=self.is_sparse,
                indexer=self.indexer,
                sink_len=sink_len,
                sliding_window=sliding_window,
            )

        def forward_native(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            llama_4_scaling: torch.Tensor | None = None,
        ) -> torch.Tensor:
            q_c = None
            kv_lora = None

            if self.q_lora_rank is not None:
                assert self.fused_qkv_a_proj is not None, (
                    "fused_qkv_a_proj is required when q_lora_rank is not None"
                )
                assert self.q_a_layernorm is not None, (
                    "q_a_layernorm is required when q_lora_rank is not None"
                )
                assert self.q_b_proj is not None, (
                    "q_b_proj is required when q_lora_rank is not None"
                )
                qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
                q_c, kv_lora = qkv_lora.split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
                if self.qa_conv is not None:
                    q_c = self.qa_conv(q_c) + q_c
                q_c = self.q_a_layernorm(q_c)
                q = self.q_b_proj(q_c)[0]
            else:
                assert self.kv_a_proj_with_mqa is not None, (
                    "kv_a_proj_with_mqa is required when q_lora_rank is None"
                )
                assert self.q_proj is not None, (
                    "q_proj is required when q_lora_rank is None"
                )
                kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
                q = self.q_proj(hidden_states)[0]

            kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            if self.compresskv_conv is not None:
                kv_c = self.compresskv_conv(kv_c) + kv_c
            kv_c_normed = self.kv_a_layernorm(kv_c)

            q = q.view(-1, self.num_heads, self.qk_head_dim)
            # Add head dim of 1 to k_pe
            k_pe = k_pe.unsqueeze(1)

            if self.rotary_emb is not None:
                q_pe, k_pe = self.rotary_emb(
                    positions, q[..., self.qk_nope_head_dim :], k_pe
                )

                # Detect if using MRotaryEmbeddingInterleaved (openpangu_ultra_omni)
                # Only apply special reshape for MRotaryEmbeddingInterleaved models
                if (
                    "MRotaryEmbeddingInterleaved" in str(type(self.rotary_emb)) 
                ):
                    q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)
                    k_pe = k_pe.unsqueeze(1)

                q[..., self.qk_nope_head_dim :], k_pe = q_pe, k_pe

            if self.indexer and self.is_sparse:
                forward_context = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.mla_attn.layer_name]
                self_kv_cache = self.mla_attn.kv_cache[forward_context.virtual_engine]
                _topk_indices = self.indexer(
                    hidden_states, q_c, positions, self.indexer_rope_emb, self_kv_cache, attn_metadata
                )

            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            )
            if self.o_conv is not None:
                attn_out = self.o_conv(attn_out) + attn_out
            return self.o_proj(attn_out)[0]
    # patch end

    mla.StaticSinkMultiHeadLatentAttentionWrapper = StaticSinkMultiHeadLatentAttentionWrapper
