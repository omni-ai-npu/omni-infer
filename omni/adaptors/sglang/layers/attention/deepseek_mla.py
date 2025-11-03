from __future__ import annotations

from enum import IntEnum, auto
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.communicator import LayerScatterModes, ScatterMode
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import (
    is_fp8_fnuz,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope, get_rope_wrapper
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    bind_or_assign,
    get_bool_env_var,
    get_device_sm,
    get_int_env_var,
    is_flashinfer_available,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    use_intel_amx_backend,
)
from torch import nn
from transformers import PretrainedConfig

from omni.adaptors.sglang.layers.layernorm import RMSNorm
from omni.adaptors.sglang.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)


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

        self.mask_length = 2048
        self.attn_mask = ~torch.tril(
            torch.ones(
                (self.mask_length, self.mask_length),
                dtype=torch.bool,
                device=global_server_args_dict["device"],
            )
        )

        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
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
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        self.rotary_emb = get_rope_wrapper(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
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
        self.use_deep_gemm_bmm = False

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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        if (self.layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED) and (
            self.layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            attn_tp_all_gather_into_tensor(
                hidden_states,
                local_hidden_states,
            )

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
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )
        if forward_batch.is_prefill_idle:
            attn_output = q.new_empty(
                q.shape[0], self.num_local_heads * self.v_head_dim
            )
        else:
            k = k.view(-1, self.num_local_heads, self.qk_head_dim)
            v = v.view(-1, self.num_local_heads, self.v_head_dim)
            if q.ndim == 2:
                q = q.view(q.shape[0], self.num_local_heads, -1)
            bs_qlen, q_heads, q_dim = q.size()
            q_nope, q_rope = q.split([self.v_head_dim, self.qk_rope_head_dim], dim=-1)
            k_nope, k_rope = k.split([self.v_head_dim, self.qk_rope_head_dim], dim=-1)
            metadata = forward_batch.attn_backend.forward_metadata
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                v,
                query_rope=q_rope,
                key_rope=k_rope,
                num_heads=q_heads,
                input_layout="TND",
                atten_mask=self.attn_mask,
                sparse_mode=3,
                actual_seq_lengths=metadata.seq_lens_list_cumsum,
                actual_seq_lengths_kv=metadata.seq_lens_list_cumsum,
                scale=self.scaling,
                next_tokens=0,
            )
            attn_output = attn_output[..., : self.v_head_dim]
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
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states
        if self.q_lora_rank is not None:
            fused_qkv_a_proj_out = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, latent_cache = fused_qkv_a_proj_out.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            q = self.q_a_layernorm(q)
            k_nope = self.kv_a_layernorm(k_nope)

            k_nope = k_nope.unsqueeze(1)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        k_nope = k_nope.view(-1, 1, self.kv_lora_rank)
        forward_batch.token_to_kv_pool.set_mla_kv_buffer(
            self.attn_mqa,
            forward_batch.out_cache_loc,
            k_nope,
            k_pe,
        )
        padding_bs = forward_batch.input_ids.size(0)
        q_nope = q_nope_out.view(
            padding_bs, -1, self.num_local_heads, self.kv_lora_rank
        )
        q_pe = q_pe.view(
            padding_bs,
            -1,
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim - self.kv_lora_rank,
        )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(self.layer_id).to(
            q_nope_out.dtype
        )
        PAGE_SIZE = 128
        b, s, n, _ = q_nope.size()
        q_nope_dim = q_nope.shape[-1]
        _, k_heads, k_dim = k_cache.size()
        k_cache = k_cache.view(-1, PAGE_SIZE, k_dim)
        k_nope = k_cache[..., :q_nope_dim]
        k_rope = k_cache[..., q_nope_dim:]

        metadata = forward_batch.attn_backend.forward_metadata
        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_nope,
            k_nope,
            k_nope,
            query_rope=q_pe,
            key_rope=k_rope,
            num_heads=n,
            num_key_value_heads=1,
            input_layout="BSND",
            atten_mask=None,
            sparse_mode=0,
            scale=self.scaling,
            antiquant_mode=0,
            antiquant_scale=None,
            block_table=metadata.block_kv_indices,
            block_size=PAGE_SIZE,
            actual_seq_lengths_kv=metadata.seq_lens_list,
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        attn_bmm_output = torch.empty(
            (self.num_local_heads, attn_output.shape[0], self.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=attn_bmm_output,
        )
        attn_bmm_output = attn_bmm_output.transpose(0, 1).reshape(
            -1, self.num_local_heads * self.v_head_dim
        )

        output, _ = self.o_proj(attn_bmm_output)

        return output
