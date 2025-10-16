from __future__ import annotations

from enum import IntEnum, auto
from typing import Any, Dict, Optional
import os
import torch
import torchair as tng
from torch import nn
from contextlib import nullcontext

import torch_npu
from transformers import PretrainedConfig

from sglang.srt.layers.communicator import LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_attention_dp_size
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    get_bool_env_var,
    get_int_env_var,
)

from omni.adaptors.sglang.layers.rotary_embedding import get_rope
from omni.adaptors.sglang.layers.layernorm import RMSNorm
from omni.adaptors.sglang.layers.linear import (
    RowParallelLinearWithReduceScatter,
)
KVCACHE_NZ_DIM = 16

def stream_context(stream_tag: str, enable_multi_stream: bool = False):
    if enable_multi_stream:
        return tng.scope.npu_stream_switch(stream_tag)
    return nullcontext()


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use MLA but with fused RoPE
    MLA_FUSED_ROPE = auto()

    # Use MLA with fused RoPE kernel for CPU
    MLA_FUSED_ROPE_CPU = auto()


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekMLA(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        layer_scatter_modes: LayerScatterModes = None,
    ) -> None:
        super().__init__()

        self.rope_scaling = rope_scaling
        self.quant_config = quant_config
        self.prefix = prefix

        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.layer_scatter_modes = layer_scatter_modes
        self.quant_symbol = quant_config is not None

        self.mask_length = 2048
        self.attn_mask = ~torch.tril(
            torch.ones(
                (self.mask_length, self.mask_length),
                dtype=torch.bool,
                device=global_server_args_dict["device"],
            )
        )
        self.enable_fused_qkv = os.environ.get("USE_FUSE_QKV_A_PROJ", "0") == "1"
        # For tensor parallel attention
        if self.q_lora_rank is not None:
            if self.enable_fused_qkv:
                self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
                )
            else:
                self.q_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("q_a_proj", prefix),
                )
                self.kv_a_proj_with_mqa = ReplicatedLinear(
                    self.hidden_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("kv_a_proj_with_mqa", prefix),
                )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,  # NPU not support quantization method for kv_b_proj
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinearWithReduceScatter(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb.forward = self.rotary_emb.forward_native

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.attn_mha.kv_b_proj = None

        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0

        self.w_scale_k = None
        self.w_scale_v = None

        self.disable_chunked_prefix_cache = global_server_args_dict[
            "disable_chunked_prefix_cache"
        ]

        self.rocm_fused_decode_mla = get_bool_env_var(
            "SGLANG_ROCM_FUSED_DECODE_MLA", "false"
        )

        # TODO: Design a finer way to determine the threshold
        self.chunked_prefix_cache_threshold = get_int_env_var(
            "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD", 8192
        )

        # which requires self.w_kc and self.w_vc to be packed.
        # If not, we will use torch.bmm and weight shouldn't be packed in this case
        has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")

        is_packed_weight = (
            has_fused_proj
            and hasattr(self.fused_qkv_a_proj_with_mqa.quant_method, "quant_config")
            and self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.get_name()
            in {"awq", "awq_marlin", "moe_wna16"}
        )

        self.qkv_proj_with_rope_is_int8 = (
            has_fused_proj
            and not is_packed_weight
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.int8
        )
        self.qkv_proj_with_rope_is_fp8 = (
            has_fused_proj
            and not is_packed_weight
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.float8_e4m3fn
        )

        self.weight_block_size = None
        self.enable_mla_multi_stream = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        self.enable_mla_multi_stream = forward_batch.can_run_graph
        if (
            forward_batch.is_decode_or_idle and not forward_batch.is_prefill_idle
        ) or forward_batch.is_target_verify:
            return self._forward_decode(
                positions, hidden_states, forward_batch, zero_allocator
            )
        else:
            return self._forward_prefill(
                positions, hidden_states, forward_batch, zero_allocator
            )

    def _forward_prefill(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if self.q_lora_rank is not None:
            if self.enable_fused_qkv:
                q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
                )
                latent_cache = latent_cache.contiguous()
            else:
                q = self.q_a_proj(hidden_states)[0]
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            latent_cache = get_attention_tp_group().all_gather(latent_cache, dim=0)

            q = self.q_a_layernorm(q)
            if self.quant_symbol:
                q_quant, q_scale = torch_npu.npu_dynamic_quant(q)
                # Quantizing before all_gather can reduce communication overhead.
                q_quant = get_attention_tp_group().all_gather(q_quant, dim=0)
                q_scale = get_attention_tp_group().all_gather(q_scale, dim=0)
                q = {'x_int8':q_quant, 'pertoken_scale':q_scale}
            else:
                q = get_attention_tp_group().all_gather(q, dim=0)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        metadata = forward_batch.attn_backend.forward_metadata
        if metadata.cos is None or metadata.sin is None:
            cos, sin = self.rotary_emb.get_cos_sin(positions)
        else:
            cos, sin = metadata.cos, metadata.sin
        q_rope = q_rope.unsqueeze(2)
        q_rope = torch_npu.npu_interleave_rope(q_rope, cos, sin)
        q_rope = q_rope.squeeze(2)

        k_nope_cache, k_rope_cache = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)
        k_nope_cache = k_nope_cache.view(-1, metadata.page_size, 1, self.kv_lora_rank)
        k_rope_cache = k_rope_cache.view(-1, metadata.page_size, 1, self.qk_rope_head_dim)

        out_cache_loc = forward_batch.out_cache_loc.to(torch.int64)
        k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache.unsqueeze(1).unsqueeze(1),
            self.kv_a_layernorm.weight,
            cos,
            sin,
            out_cache_loc,
            k_rope_cache,
            k_nope_cache,
            k_rope_scale=None,
            c_kv_scale=None,
            k_rope_offset=None,
            c_kv_offset=None,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
            is_output_kv=True
        )
        k_rope = k_rope.view(-1, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, metadata.page_size, KVCACHE_NZ_DIM)
        k_nope = k_nope.view(-1, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, metadata.page_size, KVCACHE_NZ_DIM)
        k_rope = k_rope.transpose(1, 3)
        k_nope = k_nope.transpose(1, 3)
        k_rope = k_rope.reshape(-1, self.qk_rope_head_dim).index_select(0, out_cache_loc).contiguous()
        kv_down = k_nope.reshape(-1, self.kv_lora_rank).index_select(0, out_cache_loc).contiguous()
        
        kv_up = self.kv_b_proj(kv_down)[0]
        kv_up = kv_up.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_up, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if forward_batch.is_prefill_idle:
            attn_output = q.new_empty(
                q.shape[0], self.num_local_heads * self.v_head_dim
            )
        else:
            k_rope = k_rope.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                v,
                query_rope=q_rope,
                key_rope=k_rope,
                num_heads=self.num_local_heads,
                input_layout="TND",
                atten_mask=self.attn_mask,
                sparse_mode=3,
                actual_seq_lengths=metadata.seq_lens_list_cumsum,
                actual_seq_lengths_kv=metadata.seq_lens_list_cumsum,
                scale=self.scaling,
                next_tokens=0,
            )
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)

        return output

    def _forward_decode(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        metadata = forward_batch.attn_backend.forward_metadata
        if self.q_lora_rank is not None:
            if self.enable_fused_qkv:
                fused_qkv_a_proj_out = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
                q, latent_cache = fused_qkv_a_proj_out.split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
                )
            else:
                q = self.q_a_proj(hidden_states)[0]
                with stream_context("mla_multi_stream", self.enable_mla_multi_stream):
                    latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
                if self.enable_mla_multi_stream:
                    tng.scope.npu_wait_tensor(q, q)
            # overlap qk norm
            if metadata.norm_res is not None:
                q, _ = self.q_a_layernorm(q, metadata.norm_res)
            else:
                q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0]
        else:
            q = self.q_proj(hidden_states)[0]
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        bsz, _ = q.shape
        q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1)
        q_nope = (
            torch.matmul(q_nope, self.w_kc)
            .transpose(1, 0)
            .view(bsz, 1, self.num_local_heads, self.kv_lora_rank)
        )
        
        with stream_context("mla_multi_stream", self.enable_mla_multi_stream):
            latent_cache = latent_cache.unsqueeze(1).unsqueeze(1)
            cos, sin = metadata.cos, metadata.sin
            k_nope_cache, k_rope_cache = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)
            k_nope_cache = k_nope_cache.view(-1, metadata.page_size, 1, self.kv_lora_rank)
            k_rope_cache = k_rope_cache.view(-1, metadata.page_size, 1, self.qk_rope_head_dim)
            k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache,
                self.kv_a_layernorm.weight,
                cos,
                sin,
                forward_batch.out_cache_loc.to(torch.int64),
                k_rope_cache,
                k_nope_cache,
                k_rope_scale=None,
                c_kv_scale=None,
                k_rope_offset=None,
                c_kv_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ"
            )
            k_nope = k_nope.view(-1, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, metadata.page_size, KVCACHE_NZ_DIM)
            k_rope = k_rope.view(-1, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, metadata.page_size, KVCACHE_NZ_DIM)
            if self.enable_mla_multi_stream:
                tng.scope.npu_wait_tensor(q_rope, k_nope)
            q_rope = torch_npu.npu_interleave_rope(q_rope, cos, sin)
            q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
            q_rope = q_rope.view(bsz, self.num_local_heads, -1)
        attn_ops_scope = tng.ops if forward_batch.can_run_graph else torch.ops.npu
        attn_output, _ = attn_ops_scope.npu_fused_infer_attention_score(
            q_nope, 
            k_nope, 
            k_nope, 
            query_rope=q_rope, 
            key_rope=k_rope,
            num_heads=self.num_local_heads,
            num_key_value_heads=1, 
            input_layout='TND_NTD',
            scale=self.scaling,
            antiquant_mode=0, 
            antiquant_scale=None,
            block_table=metadata.block_kv_indices,
            block_size=metadata.page_size,
            actual_seq_lengths=metadata.actual_seq_lengths,
            actual_seq_lengths_kv=metadata.seq_lens
        )
       
        attn_output = attn_output.view(self.num_local_heads, -1, self.kv_lora_rank)
        attn_output = (
            torch.matmul(attn_output, self.w_vc)
            .transpose(1, 0)
            .reshape(-1, self.num_local_heads * self.v_head_dim)
        )
        output, _ = self.o_proj(attn_output)

        return output

