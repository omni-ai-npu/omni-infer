# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This patch is used for enable_eplb fix in ParallelConfig and FusedMoE
# Please use this patch by adding VLLM_PLUGINS="omni-npu,omni_npu_patches" OMNI_NPU_VLLM_PATCHES="EPLBParallelConfig,EPLBFusedMoE" before vllm serve

from typing import Optional

import torch
import torch_npu
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.config import VllmConfig, CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.deepseek_v2 import Indexer

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.attention.backends.dsa import NPUDSAMetadata, NPUDSAImpl


@register_patch("DSV32Indexer", Indexer)
class DSV32IndexerPatch(VLLMPatch):
    """
    Patch to modify the DSV32 indexer behavior for compatibility with vLLM.
    """
    _attr_names_to_apply = ['__init__', 'forward', '_apply_indexer']
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        torch.nn.Module.__init__(self)
        self.vllm_config = vllm_config
        self.config = config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.weights_proj = ReplicatedLinear(
            hidden_size, self.n_head, bias=False, quant_config=None, prefix=f"{prefix}.weights_proj"
        )
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.sink_len = getattr(config, 'param_sink_number', 0)
        self.num_sink_blocks = self.sink_len // self.vllm_config.cache_config.block_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions,
        rotary_emb,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        attn_metadata: Optional[NPUDSAMetadata] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        k, _ = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q_pe = q_pe.unsqueeze(0)
        k_pe = k_pe.unsqueeze(0)
        q = torch.cat([q_pe.squeeze(0), q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze((0, 2)), k_nope], dim=-1)

        weights, _ = self.weights_proj(hidden_states)

        if attn_metadata is None:
            # profile run
            # NOTE(Chen): create the max possible flattened_kv. So that
            # profile_run can get correct memory usage.
            _flattened_kv = torch.empty(
                [self.max_total_seq_len, self.head_dim + 4], device=k.device, dtype=torch.bfloat16
            )
            _k_bf16 = _flattened_kv[..., :self.head_dim].contiguous()
            return self.topk_indices_buffer
        
        assert len(kv_cache) >= 3, (f"Expected kv_cache to have at least 3 elements, but got {len(kv_cache)}")
        
        if kv_cache[2] is not None:
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2].view(-1, 1, k.shape[-1]),
                attn_metadata.slot_mapping.view(-1, 1),
                k.view(-1, 1, k.shape[-1])
            )

        bs = q.shape[0]
        self.topk_indices_buffer[:bs] = self._apply_indexer(
            query=q,
            key=kv_cache[2],
            weights=weights,
            attn_metadata=attn_metadata,
        )

        return self.topk_indices_buffer[:bs]

    def _apply_indexer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: NPUDSAMetadata,
    ):
        block_table, actual_seq_lens_key, actual_seq_lens_query = \
            NPUDSAImpl.get_args_from_attn_metadata(attn_metadata)
        block_table = block_table[:, self.num_sink_blocks:]
        actual_seq_lens_key = actual_seq_lens_key - self.sink_len

        return torch.ops.custom.npu_lightning_indexer_enhance(
            query=query,
            key=key,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lens_query,
            actual_seq_lengths_key=actual_seq_lens_key,
            block_table=block_table,
            layout_key="PA_BSND",
            layout_query="TND",
            sparse_count=self.topk_tokens,
            sparse_mode=3,
        )[0].view(-1, self.topk_tokens)
