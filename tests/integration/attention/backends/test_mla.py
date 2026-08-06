# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for NPUAttentionBackendMLAImpl that require NPU hardware.
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
import pytest

from omni.attention.backends.mla import (
    NPUMLAImpl,
    NPUMLAMetadata,
)
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata

from integration.utils.common_utils import (
    skipif_no_npu,
    NPU_AVAILABLE,
)


@pytest.mark.integration
@skipif_no_npu
class TestNPUAttentionBackendMLAImplIntegration(unittest.TestCase):

    def setUp(self):
        if not NPU_AVAILABLE:
            self.skipTest("NPU not available")
        self.impl_cls = NPUMLAImpl
        self.device = "npu"

    def _create_dummy_kv_cache(self, num_blocks=16, block_size=128, kv_lora_rank=128, qk_nope_head_dim=128, v_head_dim=128):
        # Use bfloat16 to satisfy NPU TND layout requirement
        cache_k = torch.randn(num_blocks, block_size, 512, dtype=torch.bfloat16, device=self.device)
        cache_v = torch.randn(num_blocks, block_size, 64, dtype=torch.bfloat16, device=self.device)
        return (cache_k, cache_v)

    def test_forward_success(self):
        # === MLA configuration ===
        q_lora_rank = 256
        kv_lora_rank = 128
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
        v_head_dim = 128
        num_heads = 8
        num_kv_heads = 8

        batch_size = 1
        prompt_len = 1
        total_prefill_tokens = batch_size * prompt_len

        MAX_SEQ_LEN_NPU = 2048  # NPU hard-coded limit for sparse_mode=3
        causal_mask = torch.tril(torch.ones((MAX_SEQ_LEN_NPU, MAX_SEQ_LEN_NPU), dtype=torch.bool, device=self.device))
        original_mask = NPUMLAImpl.SHARE_MASK_TRIL_SPARSE
        NPUMLAImpl.SHARE_MASK_TRIL_SPARSE = causal_mask

        try:
            mock_ctx = MagicMock()
            mock_ctx.batch_descriptor = MagicMock(
                num_reqs=batch_size,
                max_q_len=prompt_len,
                max_seq_len=prompt_len,
                uniform=True,
            )

            with patch('vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size', return_value=64), \
                patch('omni.attention.backends.mla.get_current_vllm_config', return_value=None):

                # Create kv_b_proj and convert to bfloat16
                kv_b_proj = torch.nn.Linear(
                    kv_lora_rank,
                    num_kv_heads * (qk_nope_head_dim + v_head_dim),
                    bias=False,
                ).to(self.device).to(torch.bfloat16)

                for param in kv_b_proj.parameters():
                    param.requires_grad_(False)

                impl = self.impl_cls(
                    num_heads=num_heads,
                    head_size=qk_head_dim,
                    scale=1.0 / (qk_head_dim ** 0.5),
                    num_kv_heads=num_kv_heads,
                    attn_type=AttentionType.DECODER,
                    alibi_slopes=None,
                    sliding_window=None,
                    kv_cache_dtype="auto",
                    logits_soft_cap=None,
                    kv_sharing_target_layer_name=None,
                    q_lora_rank=q_lora_rank,
                    kv_lora_rank=kv_lora_rank,
                    qk_nope_head_dim=qk_nope_head_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    qk_head_dim=qk_head_dim,
                    v_head_dim=v_head_dim,
                    kv_b_proj=kv_b_proj,
                )

                total_nope_dim = num_kv_heads * qk_nope_head_dim
                total_v_dim = num_kv_heads * v_head_dim
                W_UK_raw = kv_b_proj.weight[:total_nope_dim, :]
                W_UV_raw = kv_b_proj.weight[total_nope_dim:, :]
                impl.W_UK = W_UK_raw.t().contiguous().detach()
                impl.W_UV = W_UV_raw.t().contiguous().detach()

                # All tensors in bfloat16
                q = torch.randn(total_prefill_tokens, num_heads, qk_head_dim, dtype=torch.bfloat16, device=self.device)
                k_c_normed = torch.randn(total_prefill_tokens, kv_lora_rank, dtype=torch.bfloat16, device=self.device)
                k_pe = torch.randn(total_prefill_tokens, qk_rope_head_dim, dtype=torch.bfloat16, device=self.device)
                kv_cache = self._create_dummy_kv_cache(
                    num_blocks=16,
                    block_size=128,
                    kv_lora_rank=kv_lora_rank,
                    qk_nope_head_dim=qk_nope_head_dim,
                    v_head_dim=v_head_dim
                )
                output = torch.empty(total_prefill_tokens, num_heads * v_head_dim, dtype=torch.bfloat16, device=self.device)

                seq_lens = [prompt_len]
                cumsum = [0]
                for l in seq_lens:
                    cumsum.append(cumsum[-1] + l)
                cu_seq_lens = torch.tensor([cumsum], dtype=torch.int32, device=self.device)

                chunked_context = None

                block_table = torch.randint(0, 16, (batch_size, prompt_len), dtype=torch.int32, device=self.device)
                query_start_loc = torch.tensor([0, total_prefill_tokens], dtype=torch.int32, device=self.device)
                slot_mapping = torch.arange(total_prefill_tokens, dtype=torch.long, device=self.device)

                prefill_meta = MagicMock()
                prefill_meta.block_table = block_table
                prefill_meta.seq_lens = seq_lens
                prefill_meta.query_start_loc = query_start_loc
                prefill_meta.query_cumlens = query_start_loc[1:]
                prefill_meta.slot_mapping = slot_mapping
                prefill_meta.chunked_context = chunked_context

                attn_metadata = NPUMLAMetadata(
                    prefill=prefill_meta,
                    decode=None,
                    num_prefills=batch_size,
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_actual_tokens=total_prefill_tokens,
                    slot_mapping=slot_mapping,
                    num_reqs=batch_size,
                    max_query_len=prompt_len,
                    max_seq_len=prompt_len,
                    query_start_loc=query_start_loc,
                )

                layer = MagicMock()
                layer._k_scale = None

                result = impl.forward(
                    layer=layer,
                    q=q,
                    k_c_normed=k_c_normed,
                    k_pe=k_pe,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    output=output,
                )

                self.assertEqual(result.shape, (total_prefill_tokens, num_heads * v_head_dim))
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(result).all())

        finally:
            NPUMLAImpl.SHARE_MASK_TRIL_SPARSE = original_mask


if __name__ == '__main__':
    unittest.main()
