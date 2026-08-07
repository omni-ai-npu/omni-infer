# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Minimal, self-contained NPU MLA attention with indexer backend for omni_npu.

This implementation currently delegates to the standard NPU attention
backend to remain fully self-contained and avoid external dependencies.
It satisfies vLLM's backend interface so the platform selector can
import and use it. We can iterate later with true MLA specialization.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple, TypeVar, TYPE_CHECKING

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.model_executor.models.utils import extract_layer_index
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionCGSupport, AttentionLayer, AttentionType, CommonAttentionMetadata
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonPrefillMetadata,
    QueryLenSupport,
)
from vllm.v1.attention.backend import (
    AttentionBackend,  # type: ignore
    SparseMLAAttentionImpl,
    AttentionCGSupport,
)

from vllm.v1.kv_cache_interface import AttentionSpec

from omni_npu.attention.backends.utils import (
    _maybe_padded_raw_tensor_to_strided_caches,
    paged_cache,
    register_attention_backend,
    SPManager,
    KVSPMaganer,
)
from omni_npu.model_config.config_loader.loader import model_extra_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.utils import KVCacheLayoutType
    from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

logger = init_logger(__name__)
NPUDSA = "NPUDSA"


@register_attention_backend(NPUDSA)
class NPUDSABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return NPUDSA

    @staticmethod
    def get_builder_cls() -> type["NPUDSAMetadataBuilder"]:
        return NPUDSAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["NPUDSAImpl"]:
        return NPUDSAImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return True

    @staticmethod
    def _reshape_kv_cache_noncontiguous(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: "DSAAttentionSpec",
    ) -> tuple[torch.Tensor, ...]:
        if kv_cache_spec.cache_dtype_str == "fp8_ds_mla":
            shapes = ((656,), (128,), (1,))
            dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float32)
        elif kv_cache_spec.cache_dtype_str == "int8_ds_mla":
            shapes = ((656,), (128,), (1,))
            dtypes = (torch.int8, torch.int8, torch.float16)
        elif kv_cache_spec.cache_dtype_str == "hif8_ds_mla":
            shapes = ((656,), (128,), (1,))
            dtypes = (torch.uint8, torch.uint8, torch.float32)
        elif kv_cache_spec.cache_dtype_str == "li_int8_ds_mla":
            shapes = ((576,), (128,), (1,))
            dtypes = (torch.bfloat16, torch.int8, torch.float16)
        else:
            shapes = ((576,), (128,))
            dtypes = (model_extra_config.dtype, model_extra_config.dtype)

        return _maybe_padded_raw_tensor_to_strided_caches(
            raw_tensor,
            num_blocks=num_blocks,
            block_size=kv_cache_spec.block_size,
            shapes=shapes,
            dtypes=dtypes,
            page_size_bytes=kv_cache_spec.page_size_bytes,
        )

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: AttentionSpec,
    ) -> Tuple[torch.Tensor, ...]:
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            return NPUDSABackend._reshape_kv_cache_noncontiguous(
                raw_tensor,
                num_blocks,
                kv_cache_spec,
            )
        vllm_config = get_current_vllm_config()
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        block_size = kv_cache_spec.block_size
        dtype = kv_cache_spec.dtype
        raw_tensor = raw_tensor.view(dtype=dtype)
        base_shape = (num_blocks, block_size, 1)
        if cache_dtype_str == 'hif8_ds_mla':
            shapes = [(*base_shape, 656), (*base_shape, 128), (*base_shape, 4)]
            dtypes = [dtype, dtype, torch.float32]
        else:
            shapes = [(*base_shape, 512), (*base_shape, 64), (*base_shape, 128)]
            dtypes = [dtype, dtype, dtype]
        sizes = [math.prod(shape) for shape in shapes]
        if raw_tensor.numel() != sum(sizes):
            raise RuntimeError(f"Raw tensor has {raw_tensor.numel()} elements, while"
                               f" the expected sizes for KV cache are {sizes}.")
        tensors = torch.split(raw_tensor, sizes)
        return tuple(t.view(shape).view(dt) for t, shape, dt in zip(tensors, shapes, dtypes))


@dataclass
class NPUDSAPrefillMetadata(MLACommonPrefillMetadata):
    query_cumlens: torch.Tensor = None
    seq_lens: torch.Tensor = None

    prefix_meta: Optional[Any] = None
    slot_mapping: torch.Tensor = None
    slot_mapping_2d: torch.Tensor = None


@dataclass
class NPUDSADecodeMetadata(MLACommonDecodeMetadata):
    query_cumlens: torch.Tensor
    mc2_mask: torch.Tensor = None
    slot_mapping: torch.Tensor = None
    slot_mapping_2d: torch.Tensor = None
    num_actual_tokens: int = None


@dataclass
class NPUDSAMetadata(MLACommonMetadata[NPUDSADecodeMetadata]):
    get_slot_mapping_2d = lambda: None
    slot_mapping_cache = None
    first_layer_idx = -1


M = TypeVar("M", bound=NPUDSAMetadata)


class NPUDSAMetadataBuilder(MLACommonMetadataBuilder[NPUDSAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, NPUDSAMetadata)
        # First DSA layer idx within this group. With multi-group hybrid attention,
        # builder is instantiated per group and `layer_names` contains only the
        # DSA layers assigned to this group. Take the min of the parsed idxs so
        # we don't depend on upstream ordering. Used by _lazy_slot_mapping_2d
        # to gate the recompute on a layer-idx comparison the graph trace can
        # fixate per group.
        self.first_layer_idx = (
            min(extract_layer_index(n) for n in layer_names) if layer_names else -1
        )
        self.prefill_metadata_cls = NPUDSAPrefillMetadata

        if self.aot_schedule:
            raise ValueError("AOT schedule should be enabled.")
        self.uniform_decode_query_len = (
            1
            if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )
        max_decode_tokens = self.vllm_config.scheduler_config.max_num_seqs * self.uniform_decode_query_len
        self.mc2_mask = torch.zeros(max_decode_tokens, dtype=torch.bool, device=current_platform.device_type)
        self.sink_len = getattr(self.vllm_config.model_config.hf_config, "param_sink_number", 0)

        self.decode_cudagraph_max_bs = max_decode_tokens
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.ena_kvsp = bool(self.dcp_world_size > 1)
        if self.ena_kvsp:
            assert self.dcp_world_size == get_tp_group().world_size
            assert self.cp_kv_cache_interleave_size == self.vllm_config.cache_config.block_size

        self.force_first_chunk_context = (
            self.vllm_config.cache_config.enable_prefix_caching or
            (self.vllm_config.scheduler_config.enable_chunked_prefill and
             not model_extra_config.operator_opt_config.optimize_first_chunk)
        )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> NPUDSADecodeMetadata:
        num_actual_tokens = query_start_loc_cpu[-1]
        return NPUDSADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            query_cumlens=query_start_loc_device[1:],
            num_actual_tokens=num_actual_tokens,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> NPUDSAMetadata:

        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if metadata.decode is not None and self.vllm_config.kv_transfer_config is not None:
            # for pd-mixed, TP is used, no need to use mc2_mask
            metadata.decode.mc2_mask = self._generate_activate_mask(metadata.decode.num_actual_tokens)
            metadata.decode.seq_lens = common_attn_metadata.seq_lens[:metadata.num_decodes]

            # DSA-KVSP is designed for prefill only on pd-disagg
            # TODO: decode-kvsp is temporarily added before kv-trans finished kvsp
            if self.ena_kvsp:
                decode = metadata.decode
                decode.kvsp_manager = KVSPMaganer(
                    q_cumlens=torch.cat([decode.query_cumlens.new_zeros(1), decode.query_cumlens]),  # [B + 1]
                    kv_lens=metadata.decode.seq_lens,
                    blk_table=decode.block_table,
                )

        if metadata.prefill is not None:
            metadata.prefill.query_cumlens = metadata.prefill.query_start_loc[1:]
            metadata.prefill.seq_lens = common_attn_metadata.seq_lens[
                metadata.num_decodes : metadata.num_decodes + metadata.num_prefills
            ]

            if model_extra_config.parall_config.ena_seq_parallel:
                prefill = metadata.prefill
                mome_kernel_width = getattr(self.vllm_config.model_config.hf_config, "router_sliding_window", 0)
                computed_lens = prefill.seq_lens - (prefill.query_start_loc[1:] - prefill.query_start_loc[:-1])
                has_chunked_context = (
                    getattr(prefill, "chunked_context", None) is not None
                    or self.force_first_chunk_context
                )
                prefill.sp_manager = SPManager.init_cp(
                    cumlens=prefill.query_start_loc,  # [B + 1]
                    computed_lens=computed_lens,
                    block_table_ref=prefill.block_table,
                    table_size=prefill.block_table.size(1),
                    mome_kernel_width=mome_kernel_width,
                    has_chunked_context=has_chunked_context,
                )
                if mome_kernel_width == 0:
                    prefill.cache_fn = paged_cache(
                        metadata.slot_mapping,
                        prefill.query_start_loc,  # [B + 1]
                    )

                if self.ena_kvsp:
                    prefill.kvsp_manager = KVSPMaganer(
                        q_cumlens=prefill.query_start_loc,  # [B + 1]
                        kv_lens=metadata.prefill.seq_lens,
                        blk_table=prefill.block_table,
                    )

        # slot_mapping_2d for op "npu_ai_infra_scatter_block_update_"
        # for graph capture, only called inside model
        metadata.first_layer_idx = self.first_layer_idx
        metadata.get_slot_mapping_2d = self._lazy_slot_mapping_2d(metadata)

        return metadata

    def _lazy_slot_mapping_2d(self, metadata):
        slots = metadata.slot_mapping
        pg = self.kv_cache_spec.block_size
        def inner_get_slot_mapping_2d(layer_idx=-1):
            if layer_idx == -1 or layer_idx == metadata.first_layer_idx:
                metadata.slot_mapping_cache = torch.stack([slots // pg, slots % pg], dim=-1)
            return metadata.slot_mapping_cache
        return inner_get_slot_mapping_2d

    def _generate_activate_mask(self, num_actual_tokens):
        self.mc2_mask.fill_(False)
        self.mc2_mask[:num_actual_tokens].fill_(True)
        return self.mc2_mask


class NPUDSAImpl(SparseMLAAttentionImpl[NPUDSAMetadata]):
    # not real can-return-lse, just enable vllm-DCP for KVSP
    can_return_lse_for_decode: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: object | None = None,
        **mla_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.qk_head_dim = mla_args["qk_head_dim"]
        self.v_head_dim = mla_args["v_head_dim"]
        self.kv_b_proj = mla_args["kv_b_proj"]
        self.q_pad_num_heads = mla_args.get("q_pad_num_heads")

        self.indexer = indexer
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None and hasattr(indexer, "topk_indices_buffer") else topk_indices_buffer
        )

        self.supports_quant_query_input = False
        self.dcp_world_size = -1

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "NPUDSAImpl does not support one of the following: alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention are not implemented for NPUDSAImpl"
            )

    def update_sink_kv(self, sink_k_pe: torch.Tensor, sink_compressed_kv: torch.Tensor) -> None:
        self.sink_len = sink_compressed_kv.shape[0]
        self.sink_k_nope = sink_compressed_kv.unsqueeze(1).contiguous()
        self.sink_kv = torch.cat([sink_compressed_kv.unsqueeze(1), sink_k_pe.unsqueeze(1)], dim=-1).contiguous()

    def _absorb_prolog(
        self,
        q: torch.Tensor,
    ):
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        N, B, P = q_nope.shape
        _, _, L = self.W_UK_T.shape
        ql_nope = q_nope.new_empty((N, B, L))

        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        torch.bmm(q_nope, self.W_UK_T, out=ql_nope)
        # Convert from (N, B, L) to (B, N, L)
        ql_nope = ql_nope.transpose(0, 1)
        return ql_nope, q_pe

    def _apply_sparse_attention(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attn_metadata: NPUDSAMetadata,
    ):
        block_table, actual_seq_lens_key, actual_seq_lens_query = self.get_args_from_attn_metadata(attn_metadata)
        bs = q_nope.shape[0]
        if self.indexer is not None:
            topk_buf = self.indexer.topk_indices_buffer
            topk_tokens = self.indexer.topk_tokens
        else:
            topk_buf = self.topk_indices_buffer
            topk_tokens = topk_buf.shape[-1]
        sparse_indices = topk_buf[:bs].view(bs, 1, topk_tokens)

        if getattr(self, "sink_len", 0):
            sink_indices = torch.arange(self.sink_len, device=sparse_indices.device, dtype=sparse_indices.dtype).expand(
                bs, 1, self.sink_len
            )
            mask = (sparse_indices != -1).to(sparse_indices.dtype)
            sparse_indices = torch.concat((sink_indices, sparse_indices + mask * self.sink_len), dim=2)

        return torch.ops.custom.npu_sparse_flash_attention_enhance(
            query=q_nope,
            key=kv_cache[0],
            value=kv_cache[0],
            sparse_indices=sparse_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lens_query,
            actual_seq_lengths_kv=actual_seq_lens_key,
            query_rope=q_pe,
            key_rope=kv_cache[1],
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            attention_mode=2,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )[0]

    def _parse_mqa_q(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(q, tuple):
            return q

        if q.shape[-1] == self.kv_lora_rank + self.qk_rope_head_dim:
            return q.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        return self._absorb_prolog(q)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: NPUDSAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q_nope, q_pe = self._parse_mqa_q(q)

        num_actual_toks = attn_metadata.num_actual_tokens
        q_nope = q_nope[:num_actual_toks, ...]
        q_pe = q_pe[:num_actual_toks, ...]

        assert isinstance(kv_c_and_k_pe_cache, (list, tuple)), (
            f"NPUDSAImpl expects tuple KV cache, got {type(kv_c_and_k_pe_cache)}"
        )

        attn_out = self._apply_sparse_attention(
            q_nope, q_pe, kv_c_and_k_pe_cache, attn_metadata
        )
        return attn_out, None

    @staticmethod
    def get_args_from_attn_metadata(attn_metadata: NPUDSAMetadata):
        if attn_metadata.num_decodes > 0:
            decode_block_table = attn_metadata.decode.block_table
            decode_seq_lens = attn_metadata.decode.seq_lens
        else:
            decode_block_table = torch.empty(
                (0, attn_metadata.prefill.block_table.shape[1]),
                device=attn_metadata.prefill.block_table.device,
                dtype=attn_metadata.prefill.block_table.dtype,
            )
            decode_seq_lens = torch.empty(
                (0,), device=attn_metadata.prefill.seq_lens.device, dtype=attn_metadata.prefill.seq_lens.dtype
            )
        if attn_metadata.num_prefills > 0:
            prefill_block_table = attn_metadata.prefill.block_table
            prefill_seq_lens = attn_metadata.prefill.seq_lens
        else:
            prefill_block_table = torch.empty(
                (0, attn_metadata.decode.block_table.shape[1]),
                device=attn_metadata.decode.block_table.device,
                dtype=attn_metadata.decode.block_table.dtype,
            )
            prefill_seq_lens = torch.empty(
                (0,), device=attn_metadata.decode.seq_lens.device, dtype=attn_metadata.decode.seq_lens.dtype
            )

        block_table = torch.concat((decode_block_table, prefill_block_table))
        actual_seq_lens_key = torch.concat((decode_seq_lens, prefill_seq_lens)).to(torch.int32)
        actual_seq_lens_query = attn_metadata.query_start_loc.to(torch.int32)[1:]
        return (block_table, actual_seq_lens_key, actual_seq_lens_query)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)

        self.W_UV = self.W_UV.contiguous() # 避免 post_attn_absorb 前的连续化
        self.W_UK_T = self.W_UK_T.contiguous() # 避免 mlaprolog 前的连续化
