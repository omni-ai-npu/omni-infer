# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
NPU Attention backend vendored into omni_npu.attention.backends.

This is a minimized, self-contained version derived from omniinfer sources,
with the following adjustments:
- Replaced omni.* imports with local shims or safe defaults.
- Kept the core torch_npu kernels usage for prefill and decode.
- Removed optional features (best_ep, omni_cache, DSA, SP) for an MVP.
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Optional, Tuple
import math
import os
import torch
import torch_npu

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
)
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder as V1AttentionMetadataBuilder,
    CommonAttentionMetadata,
    AttentionCGSupport,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec
from omni_npu.compilation.utils import (
    capture_graph_task,
    OP_FIA_V1,
    OP_FIA_V2,
    OP_FIA_SINK,
)

from omni_npu.attention.backends.utils import register_attention_backend
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.plugin_decorators import attn_decorator

NZ_DIM_16 = 16
NZ_DIM_32 = 32
VLLM_NPU_ATTN = "VLLM_NPU_ATTN"


def get_nz_dim(
    cache_dtype: Optional[torch.dtype] = None,
    cache_dtype_str: Optional[str] = None,
) -> int:
    if cache_dtype == torch.int8:
        return NZ_DIM_32
    if cache_dtype_str is not None and "int8" in cache_dtype_str.lower():
        return NZ_DIM_32
    return NZ_DIM_16


@dataclass
class NPUMetadata:
    num_actual_tokens: int
    block_tables: torch.Tensor
    query_start_loc: list[int]
    seq_lens: list[int]
    max_query_len: Optional[int] = None
    slot_mapping: torch.Tensor = None
    num_prefills: int = 0
    num_decodes: int = 0
    num_decode_tokens: int = 0
    decode_threshold: int = 1
    use_kvnz_pure_prefill: bool = False


class NPUAttentionMetadataBuilder(V1AttentionMetadataBuilder[NPUMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    supports_uniform_spec_as_decode: ClassVar[bool] = True
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self._init_reorder_batch_threshold(
            self.reorder_batch_threshold,
            self.supports_uniform_spec_as_decode,
        )
        self.compilation_config = vllm_config.compilation_config
        # npu_fused_infer_attention_score_v2 does not support TND with dim 256
        # so UNIFORM_BATCH for BSND
        if self.kv_cache_spec.head_size == 256:
            self._cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> NPUMetadata:

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        max_query_len = common_attn_metadata.max_query_len
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
            )
        )
        use_kvnz_pure_prefill = False
        no_decodes = num_decodes == 0 and num_decode_tokens == 0
        if model_extra_config.operator_opt_config.kv_nz and num_prefills > 0 and no_decodes:
            context_lens = common_attn_metadata.compute_num_computed_tokens()[:num_prefills]
            use_kvnz_pure_prefill = (
                context_lens.numel() == num_prefills
                and torch.all(context_lens == 0).item()
            )
        attn_metadata = NPUMetadata(num_actual_tokens=num_actual_tokens,
                                       block_tables=block_table,
                                       query_start_loc=query_start_loc.tolist(),
                                       seq_lens=seq_lens.tolist(),
                                       max_query_len=max_query_len,
                                       slot_mapping=slot_mapping,
                                       num_prefills=num_prefills,
                                       num_decodes=num_decodes,
                                       num_decode_tokens=num_decode_tokens,
                                       decode_threshold=self.reorder_batch_threshold,
                                       use_kvnz_pure_prefill=use_kvnz_pure_prefill
                                    )
        return attn_metadata


@register_attention_backend(VLLM_NPU_ATTN)
class NPUAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_dtypes() -> list[torch.dtype]:  # type: ignore[override]
        return [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def get_name() -> str:
        return VLLM_NPU_ATTN

    @staticmethod
    def get_impl_cls() -> type["NPUAttentionBackendImpl"]:
        return NPUAttentionBackendImpl

    @staticmethod
    def get_metadata_cls():
        return NPUMetadata

    @staticmethod
    def get_builder_cls():
        return NPUAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: Optional[str] = None,
    ) -> tuple[int, ...]:
        if model_extra_config.operator_opt_config.kv_nz:
            nz_dim = get_nz_dim(cache_dtype_str=cache_dtype_str)
            return (num_blocks, num_kv_heads * head_size // nz_dim, block_size, nz_dim)
        return (num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: AttentionSpec,
    ) -> Tuple[torch.Tensor, ...]:
        block_size = kv_cache_spec.block_size
        num_kv_heads = kv_cache_spec.num_kv_heads
        head_size = kv_cache_spec.head_size
        dtype = kv_cache_spec.dtype
        if (hasattr(kv_cache_spec, "head_size_v")
            and kv_cache_spec.head_size_v is not None
            and kv_cache_spec.head_size_v != kv_cache_spec.head_size
        ):
            head_size_v = kv_cache_spec.head_size_v
        else:
            head_size_v = None

        raw_tensor = raw_tensor.view(dtype=dtype)
        shape = NPUAttentionBackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
        if head_size_v is None or head_size == head_size_v:
            kv_shapes = [shape, shape]
        else:
            val_shape = NPUAttentionBackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size_v)
            kv_shapes = [shape, val_shape]
        sizes = [math.prod(tensor_shape) for tensor_shape in kv_shapes]
        if raw_tensor.numel() != sum(sizes):
            raise RuntimeError(f"Raw tensor has {raw_tensor.numel()} elements, while"
                               f" the expected sizes for KV cache are {sizes}.")
        tensors = torch.split(raw_tensor, sizes)
        return tuple(t.view(tensor_shape) for t, tensor_shape in zip(tensors, kv_shapes))


class NPUAttentionBackendImpl(AttentionImpl[NPUMetadata]):
    SHARE_MASK_TRIL_SPARSE = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        sinks: torch.Tensor | None = None,
        head_size_v: int | None = None,
        sink_len: int | None = None,
    ) -> None:
        # setting sink_len = 0 enables the swa custom op, to improve performance on long context.
        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(
                alibi_slopes, dtype=torch.float32, device=current_platform.device_type
            )
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type
        self.is_cross_attention = attn_type == AttentionType.ENCODER_DECODER
        self.attn_sinks = sinks
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if self.num_heads % self.num_kv_heads != 0:
            raise RuntimeError("self.num_heads must be divisible by self.num_kv_heads")

        if attn_type not in (AttentionType.DECODER, AttentionType.ENCODER_DECODER):
            raise NotImplementedError(f"Only support decoder and encoder-decoder models, but {attn_type=}.")

        if NPUAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE is None:
            NPUAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device="npu")
            )
        self.sparse_mode = 4 if sliding_window else 3
        self.sink_len = sink_len

    @attn_decorator(type='gqa')
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: Tuple,
        attn_metadata: NPUMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output

        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError("layer._k_scale_float and layer._v_scale_float must both be 1.0")
        if self.is_cross_attention:
            # This path is for Whisper-like encoder-decoder models, which are
            # likely the only encoder-decoder models this backend will support.
            if (key is None) != (value is None):
                raise NotImplementedError(
                    "NPU encoder-decoder attention requires key and value to be both present or both absent."
                )
            has_key_value = key is not None and value is not None
            has_slot_mapping = (
                attn_metadata.slot_mapping is not None
                and attn_metadata.slot_mapping.numel() > 0
            )
            if has_key_value and not has_slot_mapping:
                raise NotImplementedError(
                    "NPU encoder-decoder attention requires slot_mapping when populating KV cache."
                )

        # View q to TND, kv to TH.
        use_kv_nz = model_extra_config.operator_opt_config.kv_nz
        query = query.view(-1, self.num_heads, self.head_size).contiguous()
        if use_kv_nz:
            key = (key.view(-1, self.num_kv_heads, key.shape[-1]).contiguous()
                   if key is not None else None)
            value = (value.view(-1, self.num_kv_heads, value.shape[-1]).contiguous()
                     if value is not None else None)
        else:
            key = (key.view(-1, self.num_kv_heads * key.shape[-1]).contiguous()
                   if key is not None else None)
            value = (value.reshape(-1, self.num_kv_heads * value.shape[-1])
                     if value is not None else None)

        # npu kv rmsnorm not enable, use the default methods
        has_custom_models = "omni_custom_models" in os.environ.get("VLLM_PLUGINS", "")
        kv_rmsnorm_enabled = getattr(model_extra_config.operator_opt_config, 'enable_kv_rmsnorm_rope_cache', False)
        should_update_kv_cache = (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
            and attn_metadata.slot_mapping is not None
            and not (kv_rmsnorm_enabled and has_custom_models)
        )
        if use_kv_nz:
            defer_kv_nz_prefill_cache_update = (
                should_update_kv_cache
                and not self.is_cross_attention
                and self.sink_len == 0
                and attn_metadata.use_kvnz_pure_prefill
            )
            if should_update_kv_cache:
                torch_npu.npu_scatter_pa_kv_cache(
                    key,
                    value,
                    kv_cache[0],
                    kv_cache[1],
                    attn_metadata.slot_mapping.view(-1),
                )
        else:
            defer_kv_nz_prefill_cache_update = False
            if should_update_kv_cache:
                # update kv cache
                slots = attn_metadata.slot_mapping.view(-1, 1)
                torch_npu.npu_scatter_nd_update_(kv_cache[0].view(-1, key.shape[-1]), slots, key)
                torch_npu.npu_scatter_nd_update_(kv_cache[1].view(-1, value.shape[-1]), slots, value)

        actual_seq_qlen = attn_metadata.query_start_loc[1:]
        forward_context = get_forward_context()
        if self.sink_len is not None:
            input_layout = "TND"
            next_tokens = 0
            kwargs = {
                "query": query[:actual_seq_qlen[-1]],
                "key": kv_cache[0],
                "value": kv_cache[1],
                "num_query_heads": self.num_heads,
                "num_key_value_heads": self.num_kv_heads,
                "input_layout": input_layout,
                "softmax_scale": self.scale,
                "block_table": attn_metadata.block_tables,
                "block_size": kv_cache[0].shape[-2] if use_kv_nz else kv_cache[0].shape[1],
                "sparse_mode": self.sparse_mode,
                "pre_tokens": self.sliding_window if self.sliding_window is not None else 0,
                "next_tokens": next_tokens,
                "sink_number": self.sink_len,
                "atten_mask": None if self.is_cross_attention else NPUAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                "actual_seq_qlen": actual_seq_qlen,
                "actual_seq_kvlen": attn_metadata.seq_lens,
            }
            attn_output_shape = (num_tokens, self.num_heads, self.head_size)
            ref_slice = query[:actual_seq_qlen[-1]]
            attn_output = torch.empty(
                attn_output_shape, dtype=ref_slice.dtype, device=ref_slice.device,
            )
            softmax_lse = torch.empty(
                num_tokens, dtype=ref_slice.dtype, device=ref_slice.device,
            )
            if forward_context.capturing:
                capture_graph_task(
                    op_desc=OP_FIA_SINK,
                    op_kwargs=kwargs,
                    out_tensors=[attn_output, softmax_lse],
                    num_tokens=num_tokens,
                    layer_name=layer.layer_name,
                )
            else:
                attn_output = torch.ops.custom.npu_fused_infer_attention_sink(
                    **kwargs
                )[0]
            output[:actual_seq_qlen[-1]].copy_(attn_output)
            return output

        # npu_fused_infer_attention_score_v2 does not support TND with dim 256
        # TODO: delete the branch when the operator supports it.
        use_bsnd = self.head_size == 256
        if use_bsnd:
            forward_context = get_forward_context()

            # 1: Pure Decode Phase with no prefills (Graph capture is active here)
            if attn_metadata.num_prefills == 0:
                query_decode = query.unsqueeze(1)
                key_cache = kv_cache[0]
                value_cache = kv_cache[1]
                block_table = attn_metadata.block_tables
                block_size = kv_cache[0].shape[1]

                kwargs = {
                    "query": query_decode,
                    "key": key_cache,
                    "value": value_cache,
                    "num_heads": self.num_heads,
                    "num_key_value_heads": self.num_kv_heads,
                    "input_layout": "BSND",
                    "scale": self.scale,
                    "block_table": block_table,
                    "block_size": block_size,
                    "sparse_mode": 0,
                    "atten_mask": None,
                    "actual_seq_lengths_kv": attn_metadata.seq_lens,
                }

                attn_output_shape = query_decode.shape[:-1] + (self.head_size, )
                attn_output = torch.empty(attn_output_shape, dtype=query.dtype, device=query.device)

                if forward_context.capturing:
                    softmax_lse = torch.empty(num_tokens, dtype=query.dtype, device=query.device)
                    capture_graph_task(
                        op_desc=OP_FIA_V1,
                        op_kwargs=kwargs,
                        out_tensors=[attn_output, softmax_lse],
                        num_tokens=num_tokens,
                        layer_name=layer.layer_name,
                    )
                else:
                    attn_output = torch_npu.npu_fused_infer_attention_score(**kwargs)[0]

                output.copy_(attn_output.view(-1, self.num_heads, self.head_size))
                return output

            # 2 & 3: Hybrid / Prefill Phase (Graph capture is not active here)

            # 2: Decode portion of a hybrid batch
            if attn_metadata.num_decode_tokens > 0:
                query_decode = query[:attn_metadata.num_decode_tokens].unsqueeze(1)
                kwargs = {
                    "query": query_decode,
                    "key": kv_cache[0],
                    "value": kv_cache[1],
                    "num_heads": self.num_heads,
                    "num_key_value_heads": self.num_kv_heads,
                    "input_layout": "BSND",
                    "scale": self.scale,
                    "block_table": attn_metadata.block_tables[:attn_metadata.num_decodes],
                    "block_size": kv_cache[0].shape[1],
                    "sparse_mode": 0,
                    "atten_mask": None,
                    "actual_seq_lengths_kv": attn_metadata.seq_lens[:attn_metadata.num_decodes],
                }
                attn_output = torch_npu.npu_fused_infer_attention_score(**kwargs)[0]
                output[:attn_metadata.num_decode_tokens].copy_(attn_output.view(-1, self.num_heads, self.head_size))

            # 3: Prefill portion loop
            cu_q_lens = attn_metadata.query_start_loc
            for i in range(attn_metadata.num_decodes, attn_metadata.num_decodes + attn_metadata.num_prefills):
                if attn_metadata.seq_lens[i] == 0:
                    continue
                assert key is not None and value is not None
                one_query = query[cu_q_lens[i] : cu_q_lens[i + 1]].unsqueeze(0)
                one_key = key[cu_q_lens[i] : cu_q_lens[i + 1]].view(1, -1, self.num_kv_heads, self.head_size)
                one_value = value[cu_q_lens[i] : cu_q_lens[i + 1]].view(1, -1, self.num_kv_heads, self.head_size)
                is_chunked_prefill = attn_metadata.seq_lens[i] > one_key.shape[1]
                kwargs = {
                    "query": one_query,
                    "key": kv_cache[0] if is_chunked_prefill else one_key,
                    "value": kv_cache[1] if is_chunked_prefill else one_value,
                    "num_heads": self.num_heads,
                    "num_key_value_heads": self.num_kv_heads,
                    "input_layout": "BSND",
                    "scale": self.scale,
                    "block_table": attn_metadata.block_tables[i : i + 1] if is_chunked_prefill else None,
                    "block_size": kv_cache[0].shape[1] if is_chunked_prefill else 0,
                    "sparse_mode": 0 if self.is_cross_attention else 3,
                    "atten_mask": None if self.is_cross_attention else NPUAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                    "actual_seq_lengths_kv": attn_metadata.seq_lens[i : i + 1],
                }
                attn_out = torch_npu.npu_fused_infer_attention_score(**kwargs)[0]
                attn_output = attn_out.view(-1, self.num_heads, self.head_size)
                output[cu_q_lens[i] : cu_q_lens[i + 1]].copy_(attn_output)

            return output

        input_layout = "TND"
        sparse_mode = 0 if self.is_cross_attention else 3
        atten_mask = None if self.is_cross_attention else NPUAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE
        next_tokens = 0
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
        block_table = attn_metadata.block_tables
        if use_kv_nz:
            if defer_kv_nz_prefill_cache_update:
                key_cache = value[:actual_seq_qlen[-1]]
                value_cache = key[:actual_seq_qlen[-1]]
                block_table = None
                block_size = 0
            else:
                block_num = kv_cache[0].shape[0]
                block_size = kv_cache[0].shape[-2]
                nz_dim = get_nz_dim(cache_dtype=kv_cache[0].dtype)
                key_cache = key_cache.view(
                    block_num,
                    self.num_kv_heads,
                    self.head_size // nz_dim,
                    block_size,
                    nz_dim,
                )
                value_cache = value_cache.view(
                    block_num,
                    self.num_kv_heads,
                    self.head_size_v // nz_dim,
                    block_size,
                    nz_dim,
                )
        else:
            block_size = kv_cache[0].shape[1]
        kwargs = {
            "query": query[:actual_seq_qlen[-1]],
            "value": value_cache,
            "num_query_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "input_layout": input_layout,
            "block_table": block_table,
        }
        attn_output, softmax_lse = torch_npu._npu_fused_infer_attention_score_v2_infer_output(**kwargs)
        
        kwargs.update({
            "key": key_cache,
            "softmax_scale": self.scale,
            "block_size": block_size,
            "sparse_mode": sparse_mode,
            "atten_mask": atten_mask,
            "actual_seq_qlen": actual_seq_qlen,
            "actual_seq_kvlen": attn_metadata.seq_lens,
        })
        if self.sliding_window and not self.is_cross_attention:
            sparse_mode = 4
            kwargs.update({
                "sparse_mode": sparse_mode,
                "pre_tokens": self.sliding_window,
                "next_tokens": next_tokens,
            })
        if getattr(self, "attn_sinks", None) is not None and not self.is_cross_attention:
            kwargs.update({
                "learnable_sink": self.attn_sinks.view(self.num_heads),
            })

        forward_context = get_forward_context()
        if forward_context.capturing:
            capture_graph_task(
                op_desc=OP_FIA_V2,
                op_kwargs=kwargs,
                out_tensors=[attn_output, softmax_lse],
                num_tokens=num_tokens,
                layer_name=layer.layer_name,
            )
        else:
            attn_output = torch_npu.npu_fused_infer_attention_score_v2(
                **kwargs,
            )[0]

        output[:actual_seq_qlen[-1]].copy_(attn_output)
        return output
    
    
class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
