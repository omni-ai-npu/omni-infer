# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from typing import Any, Optional, Tuple, Dict
import torch
from torch import nn
import torch_npu
import torchair as tng
from transformers import PretrainedConfig
from vllm.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm.attention import Attention
from vllm.utils import supports_dynamo
from vllm.config import CacheConfig, QuantizationConfig, CompilationLevel, get_current_vllm_config
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.platforms import current_platform

from omni.models.common.config.model_config import model_extra_config
from omni.models.common.layers.rotary_embedding import get_rope
from omni.models.common.layers.linear import (
    MergedReplicatedLinear,
    RowParallelLinearWithReduceScatter,
    DP2TPRowParallelLinear,
)
from omni.models.common.layers.layernorm import RMSNorm
from omni.adaptors.vllm.distributed.communication_op import mla_tensor_model_parallel_all_gather
from omni.adaptors.vllm.distributed.parallel_state import (
    get_o_proj_tp_group,
    GroupCoordinator
)
from omni.models.common.config.model_config import model_extra_config
KVCACHE_NZ_DIM = 16


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
            cache_config: Optional[CacheConfig] = None, # type: ignore
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        if num_heads % self.tp_size != 0:
            raise RuntimeError("num_heads % tp_size != 0")
        self.num_local_heads = num_heads // self.tp_size
        self.scale = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_scale = None
        # FA is fully quantized, KVCache is not quantized, and the function is not enabled.
        self.use_faquant = False
        self.quant_symbol = quant_config is not None

        self.merge_qkv = model_extra_config.operator_opt_config.merge_qkv
        if self.q_lora_rank is not None:
            if self.merge_qkv:
                self.qkv_a_proj = MergedReplicatedLinear(self.hidden_size,
                                                         [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                                                         bias=False,
                                                         quant_config=quant_config,
                                                         prefix=f"{prefix}.qkv_a_proj")
            else:
                self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                                 self.q_lora_rank,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_a_proj")
                self.kv_a_proj_with_mqa = ReplicatedLinear(
                    self.hidden_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.kv_a_proj_with_mqa")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)

            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa")

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.kv_b_proj")
        # O projection.
        if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall:
            self.o_proj = ReplicatedLinear(self.num_heads * self.v_head_dim,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.o_proj")
        elif model_extra_config.parall_config.o_proj_tp_size > 1:
            self.o_proj = DP2TPRowParallelLinear(self.num_heads * self.v_head_dim,
                                                 hidden_size,
                                                 tp_size=get_o_proj_tp_group().world_size,
                                                 tp_rank=get_o_proj_tp_group().rank_in_group,
                                                 bias=False,
                                                 input_is_parallel=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.o_proj")
        else:
            self.o_proj = RowParallelLinearWithReduceScatter(self.num_heads * self.v_head_dim,
                                                             self.hidden_size,
                                                             bias=False,
                                                             quant_config=quant_config,
                                                             prefix=f"{prefix}.o_proj")

        rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scale = self.scale * mscale * mscale

        self.is_mla_prolog_init = False
        # we found npu_flash_attention can only works on 128 divisible head_dim, we pad it to target size here
        # and slice the final result to guarantee its functionality.
        self.padding_head_dim = (
            (self.qk_nope_head_dim + self.qk_rope_head_dim - 1) // 128 +
            1) * 128

        cur_vllm_config = get_current_vllm_config()
        self.enable_graph_mode = (cur_vllm_config.npu_compilation_config.level > CompilationLevel.NO_COMPILATION and supports_dynamo())

        self.attn_mask = ~torch.tril(
            torch.ones((2048, 2048), dtype=torch.bool, device=current_platform.device_type)
        )
        self.qk_rope_head_dim_nz = self.qk_rope_head_dim // 16

        self.vllm_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scale,
            use_mla=True,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.is_init = True
        self.W_UK = None
        self.W_UV = None
        # decode use mla absorb
        if model_extra_config.parall_config.dp_size > 1:
            kv_b_proj_weight = self.kv_b_proj.weight.T

            expected_shape = (
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
            )
            if kv_b_proj_weight.shape != expected_shape:
                raise RuntimeError(f"{kv_b_proj_weight.shape} != {expected_shape}")

            kv_b_proj_weight = kv_b_proj_weight.view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            self.W_UK, self.W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            self.W_UK = self.W_UK.permute(1, 2, 0)
            self.W_UV = self.W_UV.transpose(0, 1)
            self.is_init = False
            self.norm_res = {}
            self.actual_seq_lengths = {}
            for batch_size in model_extra_config.operator_opt_config.decode_gear_list:
                self.norm_res[batch_size] = torch.zeros([batch_size * self.tp_size, self.q_lora_rank], dtype=torch.bfloat16, device=current_platform.device_type)
                self.actual_seq_lengths[batch_size] = torch.tensor(list(range(1, batch_size * self.tp_size + 1)), dtype=torch.int64, device=current_platform.device_type)
                torch._dynamo.mark_static(self.norm_res[batch_size])
                torch._dynamo.mark_static(self.actual_seq_lengths[batch_size])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        comm_group: Optional[GroupCoordinator] = None,
    ) -> torch.Tensor:
        if not self.is_init:
            self.W_UK = torch.nn.Parameter(self.W_UK.contiguous(), requires_grad=False)
            self.W_UV = torch.nn.Parameter(self.W_UV.contiguous(), requires_grad=False)
            self.is_init = True
        if attn_metadata is None or attn_metadata.prefill is not None:
            output = self._forward_prefill(positions, hidden_states, kv_cache, attn_metadata, comm_group=comm_group)
        else:
            output = self._forward_decode(
                positions, hidden_states, kv_cache, attn_metadata,
                use_rmsnorm_rope_cache=model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache
            )
        if model_extra_config.operator_opt_config.use_mlaprolog and not self.is_mla_prolog_init:
            self.is_mla_prolog_init = True
            self.q_a_proj.weight = self._process_mla_prolog_weight(self.q_a_proj.weight)
            self.q_b_proj.weight = self._process_mla_prolog_weight(self.q_b_proj.weight)
            self.kv_a_proj_with_mqa.weight = self._process_mla_prolog_weight(self.kv_a_proj_with_mqa.weight)
        return output

    def _process_mla_prolog_weight(self, weight):
        weight.data = torch_npu.npu_format_cast(weight.data, 2)
        weight = torch.nn.Parameter(weight.transpose(0, 1).contiguous(), requires_grad = False)
        weight.data = torch_npu.npu_format_cast(weight.data, 29)
        return weight

    def _forward_prefill(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        comm_group: Optional[GroupCoordinator] = None,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            if self.merge_qkv:
                qkv = self.qkv_a_proj(hidden_states)[0]
                qkv = tensor_model_parallel_all_gather(qkv, dim=0)
                q, latent_cache = torch.split(qkv, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)

                q = self.q_a_layernorm(q)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            else:
                q = self.q_a_proj(hidden_states)[0]
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
                # q = tensor_model_parallel_all_gather(q, dim=0)
                latent_cache = mla_tensor_model_parallel_all_gather(latent_cache, dim=0, comm_group=comm_group)

                q = self.q_a_layernorm(q)
                if self.quant_symbol:
                    q_quant, q_scale = torch_npu.npu_dynamic_quant(q)
                    # Quantizing before all_gather can reduce communication overhead.
                    q_quant = mla_tensor_model_parallel_all_gather(q_quant, dim=0, comm_group=comm_group)
                    q_scale = mla_tensor_model_parallel_all_gather(q_scale, dim=0, comm_group=comm_group)
                    q = {'x_int8':q_quant, 'pertoken_scale':q_scale}
                else:
                    q = mla_tensor_model_parallel_all_gather(q, dim=0, comm_group=comm_group)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = tensor_model_parallel_all_gather(q, dim=0)
            latent_cache = tensor_model_parallel_all_gather(latent_cache, dim=0)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim],  dim=-1)
        # k_pe:BNS,64 kv_a:BNS, 512, kv_states:bnsd, cos,sin:bnsd, kv cache:bsnd
        q_pe = q_pe.unsqueeze(2)
        cos, sin = self.rotary_emb.get_cos_sin(positions)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
        q_pe = q_pe.squeeze(2) #BSH
        q[..., self.qk_nope_head_dim:] = q_pe
        if isinstance(kv_cache, Dict):
            kv_cache = kv_cache.get("kv_cache")
        if kv_cache is not None and isinstance(kv_cache, Tuple) and kv_cache[0].numel() > 0:
            # k_pe:BNS,64 kv_a:BNS, 512, kv_states:bnsd, cos,sin:bnsd,kv cache:bsnd
            _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache.view(-1, 1, 1, 576), # bnsd
                self.kv_a_layernorm.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                attn_metadata.slot_mapping,
                kv_cache[1],
                kv_cache[0],
                k_rope_scale=None,
                c_kv_scale=torch.reciprocal(self.kv_scale).repeat(self.kv_lora_rank).view(-1) if self.use_faquant else None,
                k_rope_offset=None, c_kv_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ",
                is_output_kv=True) # adapter NZ
        else:
            latent_cache = latent_cache.view(-1, latent_cache.size(-1))
            # adapt end
            kv_a, _ = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            latent_cache = latent_cache.unsqueeze(1)
            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = latent_cache[:, :, self.kv_lora_rank:]
            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)
        attn_output = torch.empty(
            q.shape[0],
            self.num_local_heads,
            self.v_head_dim,
            device=q_nope.device,
            dtype=q_nope.dtype)

        if attn_metadata is not None:
            prefill_metadata = attn_metadata.prefill
            computed_tokens = 0

            for iter, (actual_seq_qlen, actual_seq_kvlen) in enumerate(zip(
                prefill_metadata.seq_qlen_group,
                prefill_metadata.seq_kvlen_group)
            ):
                if prefill_metadata.kv_index_list and kv_cache is not None and isinstance(kv_cache, Tuple) and\
                        kv_cache[0].numel() > 0:
                    # adapt nz
                    block_num, block_size, head_size, _ = kv_cache[0].shape
                    kv_cache_a = (kv_cache[0]
                                .view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                    kv_cache_pe = (kv_cache[1]
                                .view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                    kv_cache_a = kv_cache_a.transpose(1, 3)
                    kv_cache_pe = kv_cache_pe.transpose(1, 3)
                    # adapt end
                    kv_a = kv_cache_a.reshape(-1, kv_cache[0].shape[-1]) \
                        .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                    k_pe = kv_cache_pe.reshape(-1, kv_cache[1].shape[-1]) \
                        .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                prefill_kv_a = kv_a[:actual_seq_kvlen[-1]]
                prefill_k_pe = k_pe[:actual_seq_kvlen[-1]]

                kv = self.kv_b_proj.forward(prefill_kv_a)[0]
                kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                if prefill_metadata.max_query_len > 1:
                    attn_mask = self.attn_mask
                    sparse_mode = 3
                else:
                    attn_mask = None
                    sparse_mode = 0  # must be 0 if attn_mask is None
                prefill_k_rope = prefill_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)
                attn_output[computed_tokens:computed_tokens+actual_seq_qlen[-1]] = \
                    torch.ops.npu.npu_fused_infer_attention_score(
                        q_nope[computed_tokens:computed_tokens+actual_seq_qlen[-1]],
                        k_nope,
                        v,
                        query_rope=q_pe[computed_tokens:computed_tokens+actual_seq_qlen[-1]],
                        key_rope=prefill_k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=self.num_local_heads,
                        input_layout="TND",
                        atten_mask=attn_mask,
                        sparse_mode=sparse_mode,
                        actual_seq_lengths=actual_seq_qlen,
                        actual_seq_lengths_kv=actual_seq_kvlen,
                        scale=self.scale,
                        next_tokens=0)[0]
                computed_tokens += actual_seq_qlen[-1]
        else:
            attn_output.fill_(0)

        attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
        if model_extra_config.parall_config.o_proj_tp_size > 1:
            output, _ = self.o_proj.forward(attn_output, q.shape[0], 1, self.num_local_heads, self.v_head_dim)
        else:
            output = self.o_proj.forward(attn_output, comm_group=comm_group)[0]
        return output

    def _forward_decode(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        use_rmsnorm_rope_cache: bool = True
    ) -> torch.Tensor:
        if use_rmsnorm_rope_cache:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            key_cache, value_cache = kv_cache

            q_len = 1
            if model_extra_config.operator_opt_config.use_mlaprolog:
                block_num, block_size, head_size, _ = key_cache.shape
                bsz, _ = hidden_states.view(-1, 7168).shape
                if self.quant_symbol:
                    hidden_states_mla_prolog, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                else:
                    hidden_states_mla_prolog = hidden_states
                cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
                cache_index = attn_metadata.slot_mapping.view(bsz, -1)

                q_nope, q_pe, k_nope, k_rope, dequant_scale_q_nope = torch.ops.npu.npu_mla_prolog_v2(token_x = hidden_states_mla_prolog.view(bsz, 1, -1),
                    weight_dq=self.q_a_proj.weight, weight_uq_qr=self.q_b_proj.weight,
                    weight_uk=self.W_UK, weight_dkv_kr=self.kv_a_proj_with_mqa.weight,
                    rmsnorm_gamma_cq=self.q_a_layernorm.weight, rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
                    rope_sin=sin.squeeze(1), rope_cos=cos.squeeze(1), cache_index=cache_index,
                    kv_cache=key_cache.view(-1, 128, 1, 512), kr_cache=value_cache.view(-1, 128, 1, 64),
                    dequant_scale_x=pertoken_scale.view(-1, 1) if self.quant_symbol else None, # pertoken quant
                    dequant_scale_w_dq=self.q_a_proj.weight_scale.view(1, -1) if self.quant_symbol else None,
                    dequant_scale_w_uq_qr=self.q_b_proj.weight_scale.view(1, -1) if self.quant_symbol else None,
                    dequant_scale_w_dkv_kr=self.kv_a_proj_with_mqa.weight_scale.view(1, -1) if self.quant_symbol else None,
                    quant_scale_ckv=torch.reciprocal(self.kv_scale).repeat(self.kv_lora_rank).view(1, -1) if self.quant_symbol and self.use_faquant else None,
                    quant_scale_ckr=None,
                    smooth_scales_cq=None,
                    rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
                    rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
                    cache_mode = "PA_NZ")

                k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // (32 if self.use_faquant else 16), block_size, (32 if self.use_faquant else 16))
                k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim_nz, block_size, 16)
                q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
                q_pe = q_pe.view(bsz, self.num_local_heads, -1)
            else:
                if self.q_lora_rank is not None:
                    q_lowrank = self.q_a_proj(hidden_states)[0]
                else:
                    q_lowrank = self.q_proj(hidden_states)[0]

                if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                    with tng.scope.npu_stream_switch('11'):
                        kv = self.kv_a_proj_with_mqa(hidden_states)[0]

                    tng.scope.npu_wait_tensor(q_lowrank, q_lowrank)
                else:
                    kv = self.kv_a_proj_with_mqa(hidden_states)[0]

                if self.q_lora_rank is not None:
                    q, _ = self.q_a_layernorm(q_lowrank, self.norm_res[q_lowrank.shape[0]])
                    q = self.q_b_proj(q)[0]
                else:
                    q = q_lowrank
                bsz, _ = q.shape
                q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
                q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,n,s,d

                q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1) # n, bs, d
                q_nope = (
                    torch.matmul(q_nope, self.W_UK)
                    .transpose(1, 0)
                    .view(bsz, q_len, self.num_local_heads, -1)
                )

                if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                    with tng.scope.npu_stream_switch('11'):
                        kv = kv.unsqueeze(1).unsqueeze(1)
                        cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
                        # cos, sin = self.rotary_emb.get_cos_sin(positions)
                        tmp_slot_mapping = attn_metadata.slot_mapping
                        block_num, block_size, head_size, _ = key_cache.shape
                        k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                            kv, self.kv_a_layernorm.weight,
                            cos, sin, tmp_slot_mapping,
                            value_cache, key_cache,
                            epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ") # adapter NZ

                        # adapter nz
                        k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                        k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)

                        tng.scope.npu_wait_tensor(q_pe, k_nope)

                        # cos, sin = self.rotary_emb.get_cos_sin(positions)
                        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
                        q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
                        q_pe = q_pe.view(bsz, self.num_local_heads, -1)
                else:
                    kv = kv.unsqueeze(1).unsqueeze(1)
                    cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
                    # cos, sin = self.rotary_emb.get_cos_sin(positions)
                    tmp_slot_mapping = attn_metadata.slot_mapping
                    block_num, block_size, head_size, _ = key_cache.shape
                    k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                        kv, self.kv_a_layernorm.weight,
                        cos, sin, tmp_slot_mapping,
                        value_cache, key_cache,
                        epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ") # adapter NZ

                    # adapter nz
                    k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                    k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)

                    # cos, sin = self.rotary_emb.get_cos_sin(positions)
                    q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
                    q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
                    q_pe = q_pe.view(bsz, self.num_local_heads, -1)

            bsz, _, q_dim = q_nope.size()
            input_layout = "TND_NTD" if model_extra_config.operator_opt_config.use_a3_high_performance_cann else "TND"
            if self.enable_graph_mode:
                attn_output, _ = tng.ops.npu_fused_infer_attention_score(
                        q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=1, input_layout=input_layout,
                        scale=self.scale,
                        antiquant_mode=0, antiquant_scale=None,
                        block_table=attn_metadata.decode.block_table,
                        block_size=128,
                        actual_seq_lengths=self.actual_seq_lengths[bsz],
                        actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                        )
            else:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                        q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=1, input_layout=input_layout,
                        scale=self.scale,
                        antiquant_mode=0, antiquant_scale=None,
                        block_table=attn_metadata.decode.block_table,
                        block_size=128,
                        actual_seq_lengths=self.actual_seq_lengths[bsz],
                        actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                        )

            # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
            if model_extra_config.operator_opt_config.use_a3_high_performance_cann:
                attn_output = attn_output.view(self.num_local_heads, bsz*q_len, self.kv_lora_rank) # adapter BSND_NBSD
            else:
                attn_output = attn_output.squeeze(1).transpose(0, 1)
            # attn_output = pp_matmul(attn_output, self.W_UV, mm_type=4)
            attn_output = (
                torch.matmul(attn_output, self.W_UV)
                .transpose(1, 0)
                .reshape(bsz, q_len, -1)
            )
            attn_output = attn_output.view(
                -1, self.num_local_heads * self.v_head_dim)
            if model_extra_config.parall_config.o_proj_tp_size > 1:
                output, _ = self.o_proj.forward(attn_output, bsz, q_len, self.num_local_heads, self.v_head_dim)
            else:
                output, _ = self.o_proj.forward(attn_output)
        else:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            key_cache, value_cache = kv_cache

            if self.q_lora_rank is not None:
                q_lowrank = self.q_a_proj(hidden_states)[0]
            else:
                q_lowrank = self.q_proj(hidden_states)[0]

            kv = hidden_states
            kv = self.kv_a_proj_with_mqa(kv)[0]

            if self.q_lora_rank is not None:
                q = self.q_a_layernorm(q_lowrank)
                q = self.q_b_proj(q)[0]
            else:
                q = q_lowrank
            bsz, _ = q.shape
            q_len = 1
            q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,n,s,d

            q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1) # n, bs, d
            q_nope = (
                torch.matmul(q_nope, self.W_UK)
                .transpose(1, 0)
                .view(bsz, q_len, self.num_local_heads, -1)
            )

            kv = kv.unsqueeze(1).unsqueeze(1)
            cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
            # cos, sin = self.rotary_emb.get_cos_sin(positions)
            tmp_slot_mapping = attn_metadata.slot_mapping
            block_num, block_size, head_size, _ = key_cache.shape
            k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv, self.kv_a_layernorm.weight,
                cos, sin, tmp_slot_mapping,
                value_cache, key_cache,
                epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ") # adapter NZ

            # adapter nz
            k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
            k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)

            # cos, sin = self.rotary_emb.get_cos_sin(positions)
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
            q_nope = q_nope.view(bsz, 1, self.num_local_heads, self.kv_lora_rank)
            q_pe = q_pe.view(bsz, 1, self.num_local_heads, -1)

            bsz, q_len, _, q_dim = q_nope.size()
            if self.enable_graph_mode:
                attn_output, _ = tng.ops.npu_fused_infer_attention_score(
                        q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=1, input_layout="BSND",
                        scale=self.scale,
                        antiquant_mode=0, antiquant_scale=None,
                        block_table=attn_metadata.decode.block_table,
                        block_size=128,
                        actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                        )
            else:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                        q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=1, input_layout="BSND",
                        scale=self.scale,
                        antiquant_mode=0, antiquant_scale=None,
                        block_table=attn_metadata.decode.block_table,
                        block_size=128,
                        actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                        )

            # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
            attn_output = attn_output.squeeze(1).transpose(0, 1)
            # attn_output = attn_output.view(self.num_local_heads, bsz*q_len, self.kv_lora_rank) # adapter BSND_NBSD
            # attn_output = pp_matmul(attn_output, self.W_UV, mm_type=4)
            attn_output = (
                torch.matmul(attn_output, self.W_UV)
                .transpose(1, 0)
                .reshape(bsz, q_len, -1)
            )
            attn_output = attn_output.view(
                -1, self.num_local_heads * self.v_head_dim)
            if model_extra_config.parall_config.o_proj_tp_size > 1:
                output, _ = self.o_proj.forward(attn_output, bsz, q_len, self.num_local_heads, self.v_head_dim)
            else:
                output, _ = self.o_proj.forward(attn_output)
        return output