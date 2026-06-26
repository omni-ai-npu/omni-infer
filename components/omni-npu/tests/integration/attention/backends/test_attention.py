# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for NPUAttentionBackendImpl that require NPU hardware.
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
import pytest
from omni_npu.attention.backends.attention import (
    NPUAttentionBackendImpl,
    NPUMetadata,
)
from vllm.v1.attention.backend import AttentionType

# Skip all tests if NPU is not available
from integration.utils.common_utils import (
    skipif_no_npu,
    NPU_AVAILABLE,
)


@pytest.mark.integration
@skipif_no_npu
class TestNPUAttentionBackendDefaultImplIntegration(unittest.TestCase):

    def setUp(self):
        if not NPU_AVAILABLE:
            self.skipTest("NPU not available")
        self.impl_cls = NPUAttentionBackendImpl
        self.metadata_cls = NPUMetadata
        self.device = "npu"
        self.dtype = torch.bfloat16

    def test_forward_decode_path_calls_npu_fused_infer_attention_score(self):
        impl = self.impl_cls(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=4,
            attn_type=AttentionType.DECODER,  # ← pure decode
        )

        layer = unittest.mock.MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        # Decode: 2 sequences, each generates 1 token → total 2 tokens
        num_decodes = 2
        num_decode_tokens = num_decodes
        hidden_size = 8 * 128      # 1024
        kv_hidden_size = 4 * 128   # 512
        block_size = 128
        num_blocks = 100

        # Tensors: [2, ...]
        query = torch.randn(num_decode_tokens, hidden_size, device=self.device, dtype=self.dtype)
        key = torch.randn(num_decode_tokens, 4, 128, device=self.device, dtype=self.dtype)
        value = torch.randn(num_decode_tokens, 4, 128, device=self.device, dtype=self.dtype)

        k_cache = torch.zeros(num_blocks, block_size, kv_hidden_size, device=self.device, dtype=self.dtype)
        v_cache = torch.zeros(num_blocks, block_size, kv_hidden_size, device=self.device, dtype=self.dtype)
        kv_cache = (k_cache, v_cache)

        # Each sequence has current length: e.g., [10, 15]
        seq_lens = [10, 15]
        # So slot_mapping: write to position 10 (index 9) and 15 (index 14)
        slot_mapping = torch.tensor([9, 14], dtype=torch.int32, device=self.device)

        # Block tables: [2 sequences, max_blocks]
        max_blocks_per_seq = 10
        block_tables = torch.randint(0, num_blocks, (num_decodes, max_blocks_per_seq), dtype=torch.int32, device=self.device)

        # Output: [2, 8, 128]  ← 注意！NPU 算子返回 TND layout
        output = torch.empty(num_decode_tokens, 8, 128, device=self.device, dtype=self.dtype)

        # Metadata for pure decode
        metadata = MagicMock(
            num_actual_tokens=num_decode_tokens,  # 2
            num_prefills=0,                       # ← 必须为 0！
            num_decode_tokens=num_decode_tokens,  # 2
            num_decodes=num_decodes,              # 2
            seq_lens=seq_lens,                    # [10, 15] → length=2
            query_start_loc=[0, 1, 2],            # cumulative decode qlens per request
            slot_mapping=slot_mapping,            # [9, 14] → length=2
            block_tables=block_tables,            # [2, 10]
            max_query_len=1,                      # decode: 1 token per seq
        )

        # Mock forward context
        mock_ctx = MagicMock()
        mock_ctx.batch_descriptor = MagicMock(
            num_reqs=num_decodes,    # 2
            max_q_len=1,
            max_seq_len=max(seq_lens),
            uniform=True,
        )
        mock_ctx.capturing = False

        with patch('omni_npu.attention.backends.attention.get_forward_context', return_value=mock_ctx):
            result = impl.forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            self.assertIs(result, output)
            torch.npu.synchronize()
            self.assertFalse(torch.isnan(result).any())
            self.assertFalse(torch.isinf(result).any())
if __name__ == '__main__':
    unittest.main()
