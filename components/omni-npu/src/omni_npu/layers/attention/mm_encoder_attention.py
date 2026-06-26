# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import torch

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)


@MMEncoderAttention.register_oot
class NPUMMEncoderAttention(MMEncoderAttention):
    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward_native(query, key, value, cu_seqlens, max_seqlen)

    def _forward_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        if query.device.type != "npu":
            raise RuntimeError("FIA path requires NPU tensors.")

        batch_size, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.maybe_reshape_qkv_to_4d(
            query, key, value, batch_size, q_len, kv_len
        )
        num_kv_heads = key.shape[-2]
        actual_seq_lengths = [q_len] * batch_size
        actual_seq_lengths_kv = [kv_len] * batch_size

        output = torch_npu.npu_fused_infer_attention_score_v2(
            query=query,
            key=key,
            value=value,
            num_query_heads=self.num_heads,
            num_key_value_heads=num_kv_heads,
            input_layout="BSND",
            softmax_scale=self.scale,
            sparse_mode=0,
            atten_mask=None,
            actual_seq_qlen=actual_seq_lengths,
            actual_seq_kvlen=actual_seq_lengths_kv,
        )[0]

        if is_reshaped:
            output = output.reshape(batch_size, q_len, -1)
        return output

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None or max_seqlen is not None:
            return self.forward_native(query, key, value, cu_seqlens, max_seqlen)
        return self._forward_fia(query, key, value)
