# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Minimal, self-contained NPU MLA attention backend for omni_npu.

This implementation currently delegates to the standard NPU attention
backend to remain fully self-contained and avoid external dependencies.
It satisfies vLLM's backend interface so the platform selector can
import and use it. We can iterate later with true MLA specialization.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple, TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    AttentionCGSupport,
    CommonAttentionMetadata,
    MLAAttentionImpl,
)
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonPrefillMetadata,
    QueryLenSupport,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.kv_cache_interface import AttentionSpec

from omni_npu.connector.utils import TP_Convertor
from omni_npu.attention.backends.utils import (
    SPManager,
    register_attention_backend,
    _maybe_padded_raw_tensor_to_strided_caches,
)
from omni_npu.model_config.config_loader.loader import model_extra_config

logger = init_logger(__name__)

try:
    import omni_training_custom_ops
except ImportError:
    logger.warning_once("Failed to import omni_training_custom_ops")
try:
    import omni_custom_ops
except ImportError:
    logger.warning_once("Failed to import omni_custom_ops")

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.utils import KVCacheLayoutType
    from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode


NPUMLA = "NPUMLA"


@register_attention_backend(NPUMLA)
class NPUMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return NPUMLA

    @staticmethod
    def get_builder_cls() -> type["NPUMLAMetadataBuilder"]:
        return NPUMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["NPUMLAImpl"]:
        return NPUMLAImpl

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return True

    @staticmethod
    def _reshape_kv_cache_noncontiguous(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: AttentionSpec,
    ) -> Tuple[torch.Tensor, ...]:
        return _maybe_padded_raw_tensor_to_strided_caches(
            raw_tensor,
            num_blocks=num_blocks,
            block_size=kv_cache_spec.block_size,
            shapes=((512,), (64,)),
            dtypes=(kv_cache_spec.dtype,) * 2,
            page_size_bytes=kv_cache_spec.page_size_bytes,
        )

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: AttentionSpec,
    ) -> Tuple[torch.Tensor, ...]:
        if model_extra_config.operator_opt_config.use_noncontiguous_kv:
            return NPUMLABackend._reshape_kv_cache_noncontiguous(
                raw_tensor,
                num_blocks,
                kv_cache_spec,
            )
        block_size = kv_cache_spec.block_size
        dtype = kv_cache_spec.dtype
        raw_tensor = raw_tensor.view(dtype=dtype)
        shapes = [(num_blocks, block_size, 512), (num_blocks, block_size, 64)]
        sizes = [math.prod(shape) for shape in shapes]
        if raw_tensor.numel() != sum(sizes):
            raise RuntimeError(
                f"Raw tensor has {raw_tensor.numel()} elements, while the expected sizes for KV cache are {sizes}."
            )
        tensors = torch.split(raw_tensor, sizes)
        return tuple(t.view(shape) for t, shape in zip(tensors, shapes))


@dataclass
class NPUMLAPrefillMetadata(MLACommonPrefillMetadata):
    query_cumlens: list[int] = None
    seq_lens: list[int] = None
    slot_mapping: torch.Tensor = None
    slot_mapping_2d: torch.Tensor = None
    num_tokens: int | None = None
    sp_manager: Optional[SPManager] = None


@dataclass
class NPUMLADecodeMetadata(MLACommonDecodeMetadata):
    query_cumlens: torch.Tensor
    mc2_mask: torch.Tensor = None
    slot_mapping: torch.Tensor = None
    slot_mapping_2d: torch.Tensor = None
    num_tokens: int | None = None
    num_actual_tokens: int = None


@dataclass
class NPUMLAMetadata(MLACommonMetadata[NPUMLADecodeMetadata]):
    decode_threshold: int = 1
    slot_mapping_2d: torch.Tensor = None

    def get_slot_mapping_2d(self):
        return None


class NPUMLAMetadataBuilder(MLACommonMetadataBuilder[NPUMLAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    supports_uniform_spec_as_decode: ClassVar[bool] = True
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, NPUMLAMetadata)
        self.prefill_metadata_cls = NPUMLAPrefillMetadata
        if self.dcp_local_block_size != 1:
            raise ValueError("DCP only support cp_kv_cache_interleave_size == 1.")
        if self.aot_schedule:
            raise ValueError("AOT schedule should be enabled.")

        # FIXME (zhao): since current the max length of input of mc2_mask only support 256, so we clamp it to 256
        max_decode_tokens = min(256, self.vllm_config.scheduler_config.max_num_seqs * self.reorder_batch_threshold)
        self.mc2_mask = torch.zeros(max_decode_tokens, dtype=torch.bool, device=current_platform.device_type)

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> NPUMLADecodeMetadata:
        if model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
            seq_lens = seq_lens_device
            query_cumlens = query_start_loc_device[1:]
            num_tokens = num_decode_tokens
        else:
            seq_lens = seq_lens_device.cpu().tolist()
            query_cumlens = query_start_loc_cpu[1:].tolist()
            num_tokens = query_start_loc_cpu[-1]

        num_actual_tokens = query_start_loc_cpu[-1]
        return NPUMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens,
            query_cumlens=query_cumlens,
            num_tokens=num_tokens,
            num_actual_tokens=num_actual_tokens,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> NPUMLAMetadata:
        # NOTE: this overrides MLACommonMetadataBuilder.build to inline the
        # MLA_SYNC_FIX (previously applied via inject_mla_sync_fix.sh) so we
        # avoid a GPU->CPU sync and use pinned H2D copies inside build().
        # The body below mirrors upstream common.py:build with the 4 fixes
        # applied; the NPU-specific post-processing runs afterwards.
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        # Upper bound is exact for prefill rows and needs no D2H sync.
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=(self.query_len_support != QueryLenSupport.VARLEN),
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            # MLA_SYNC_FIX[1/4]: reuse existing CPU data instead of
            # compute_num_computed_tokens().cpu(), which triggers a GPU->CPU sync.
            query_lens_cpu = (
                common_attn_metadata.query_start_loc_cpu[1:]
                - common_attn_metadata.query_start_loc_cpu[:-1]
            )
            num_computed_tokens_cpu = (
                seq_lens_cpu - query_lens_cpu
            )

            reqs_start = num_decodes  # prefill_start

            context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            prefill_query_start_loc = (
                query_start_loc[reqs_start:] - query_start_loc[reqs_start]
            )

            chunked_context_metadata = None
            if max_context_len_cpu > 0:
                # NOTE: it is recommend you read the `Chunked Prefill` section
                # in the comment at the top of the file before trying to
                # understand the following code

                # currently we allocate an equal amount of workspace for each
                # prefill in the batch, we could probably use a more advanced
                # algorithm here and allocate more workspace to prefills with
                # longer context lengths
                max_context_chunk = (
                    self.chunked_prefill_workspace_size // num_prefills_with_context_cpu
                )

                if self.aot_schedule:
                    # align max_context_chunk to page_size by rounding down,
                    # currently the `gather_and_maybe_dequant_cache` kernel
                    # cannot handle `context_chunk_starts` that are not aligned
                    # to page_size
                    max_context_chunk = round_down(max_context_chunk, self.page_size)

                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)

                # if `max_context_chunk = 256`, `num_chunks = 3`, and
                #   `num_prefills_with_context = 4`, create a tensor that looks
                # like [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
                # Note(simon): this is done in CPU because of downstream's
                # of `to_list`.
                chunk_starts = (
                    torch.arange(num_chunks, dtype=torch.int32)
                    .unsqueeze(1)
                    .expand(-1, num_prefills)
                    * max_context_chunk
                )
                chunk_ends = torch.min(
                    context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk
                )
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

                cu_seq_lens_cpu = torch.zeros(
                    num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=True
                )
                torch.cumsum(
                    chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
                )
                chunk_total_token = cu_seq_lens_cpu[:, -1]

                max_token_num_over_chunk = chunk_total_token.max().item()
                # MLA_SYNC_FIX[2/4]: pin_memory=True so the later non_blocking
                # H2D copy is truly asynchronous.
                token_to_seq_tensor_cpu = torch.zeros(
                    [num_chunks, max_token_num_over_chunk], dtype=torch.int32, pin_memory=True
                )
                range_idx = torch.arange(num_prefills, dtype=torch.int32)
                for i in range(num_chunks):
                    chunk_token_to_seq_tensor = torch.repeat_interleave(
                        range_idx, chunk_seq_lens[i]
                    )
                    chunk_len = chunk_token_to_seq_tensor.shape[0]
                    token_to_seq_tensor_cpu[i, :chunk_len] = chunk_token_to_seq_tensor

                if self.dcp_world_size > 1:
                    local_context_lens_allranks = get_dcp_local_seq_lens(
                        context_lens_cpu,
                        self.dcp_world_size,
                        None,
                        self.dcp_local_block_size,
                    )
                    # Note(qcs): The max local context lengths
                    # padded to `dcp_local_block_size`.
                    padded_local_context_lens_cpu: torch.Tensor = (
                        cdiv(
                            context_lens_cpu,
                            self.dcp_virtual_block_size,
                        )
                        * self.dcp_local_block_size
                    )
                    # Note(hc): The above max_context_chunk already enforces
                    # block_size alignment, DCP just need the block_size can
                    # be divisible by dcp_world_size, because DCP use
                    # cp_gather_cache which not require `cp_chunk_starts`
                    # aligned to page_size.
                    assert max_context_chunk % self.dcp_world_size == 0
                    padded_local_max_context_chunk_across_ranks = (
                        cdiv(
                            max_context_chunk,
                            self.dcp_virtual_block_size,
                        )
                        * self.dcp_local_block_size
                    )
                    local_chunk_starts = (
                        torch.arange(num_chunks, dtype=torch.int32)
                        .unsqueeze(1)
                        .expand(-1, num_prefills)
                        * padded_local_max_context_chunk_across_ranks
                    )
                    local_chunk_ends = torch.min(
                        padded_local_context_lens_cpu.unsqueeze(0),
                        local_chunk_starts
                        + padded_local_max_context_chunk_across_ranks,
                    )
                    padded_local_chunk_seq_lens = (
                        local_chunk_ends - local_chunk_starts
                    ).clamp(min=0)

                    padded_local_cu_chunk_seq_lens_cpu = torch.zeros(
                        num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=True
                    )
                    torch.cumsum(
                        padded_local_chunk_seq_lens,
                        dim=1,
                        out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
                        dtype=torch.int32,
                    )

                chunked_context_metadata_cls = MLACommonPrefillMetadata.ChunkedContextMetadata

                if self.dcp_world_size > 1:
                    chunked_context_metadata = chunked_context_metadata_cls(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        # MLA_SYNC_FIX[4/4]: contiguous().pin_memory() before
                        # non_blocking H2D (DCP path).
                        starts=local_chunk_starts.contiguous().pin_memory().to(device, non_blocking=True),
                        seq_tot=padded_local_chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        seq_lens=chunk_seq_lens,
                        token_to_seq=token_to_seq_tensor_cpu.to(
                            device, non_blocking=True
                        ),
                        chunk_total_token=chunk_total_token.tolist(),
                        workspace=self.chunked_prefill_workspace,
                        padded_local_chunk_seq_lens=padded_local_chunk_seq_lens.tolist(),
                        local_context_lens_allranks=local_context_lens_allranks.tolist(),
                        padded_local_cu_seq_lens=padded_local_cu_chunk_seq_lens_cpu.to(
                            device, non_blocking=True
                        ),
                        cu_seq_lens_lst=cu_seq_lens_cpu.tolist(),
                        chunk_size=padded_local_max_context_chunk_across_ranks,
                    )
                else:
                    chunked_context_metadata = chunked_context_metadata_cls(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        # MLA_SYNC_FIX[3/4]: contiguous().pin_memory() before
                        # non_blocking H2D (normal path).
                        starts=chunk_starts.contiguous().pin_memory().to(device, non_blocking=True),
                        seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        seq_lens=chunk_seq_lens,
                        token_to_seq=token_to_seq_tensor_cpu.to(
                            device, non_blocking=True
                        ),
                        chunk_total_token=chunk_total_token,
                        workspace=self.chunked_prefill_workspace,
                    )

                assert (
                    max(chunked_context_metadata.max_seq_lens)
                    <= self.chunked_prefill_workspace_size
                )

            prefill_metadata = self.prefill_metadata_cls(
                block_table=block_table_tensor[reqs_start:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=max_query_len,
                chunked_context=chunked_context_metadata,
            )

        decode_metadata = None
        if num_decodes > 0:
            dcp_tot_seq_lens_device = None
            if self.dcp_world_size > 1:
                dcp_tot_seq_lens_device = seq_lens[:num_decodes]
                seq_lens = dcp_local_seq_lens

                # After DCP distribution, the maximum number of tokens for any rank is
                # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
                # and I is cp_kv_cache_interleave_size.
                # This eliminates GPU->CPU sync while minimizing workspace
                # over-allocation.
                num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
                max_seq_len = (
                    (max_seq_len + num_partitions - 1) // num_partitions
                ) * self.cp_kv_cache_interleave_size

            decode_metadata = self._build_decode(
                block_table_tensor=block_table_tensor[:num_decodes, ...],
                seq_lens_device=seq_lens[:num_decodes],
                max_seq_len=max_seq_len,
                query_start_loc_cpu=query_start_loc_cpu[: num_decodes + 1],
                query_start_loc_device=query_start_loc[: num_decodes + 1],
                num_decode_tokens=num_decode_tokens,
                dcp_tot_seq_lens_device=dcp_tot_seq_lens_device,
            )

        metadata = self.metadata_cls(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            # MLACommonMetadata Chunk prefill specific
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        # === NPU-specific post-processing (was previously applied on top of
        #     super().build() in the old override) ===
        metadata.decode_threshold = self.reorder_batch_threshold

        # update decode_metadata
        if metadata.decode is not None:
            if self.vllm_config.kv_transfer_config is not None:
                # for pd-mixed, TP is used, no need to use mc2_mask
                if self.vllm_config.kv_transfer_config.kv_role == "kv_consumer":
                    metadata.decode.mc2_mask = self.generate_activate_mask(
                        metadata.decode.num_actual_tokens,
                    )

            if hasattr(self, "sink_len") and self.sink_len > 0:
                # for static sink attention, we need to add the sink length to the seq_lens
                metadata.decode.sink_len = self.sink_len
                if model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
                    metadata.decode.seq_lens.masked_fill_(metadata.decode.seq_lens == 0, self.sink_len)
                else:
                    metadata.decode.seq_lens = [
                        self.sink_len if seq == 0 else seq 
                        for seq in metadata.decode.seq_lens
                    ]

        # update prefill metadata
        if metadata.prefill is not None:
            query_cumlens = metadata.prefill.query_start_loc[1:]
            seq_lens = common_attn_metadata.seq_lens[
                metadata.num_decodes:metadata.num_decodes + metadata.num_prefills
            ]
            if not model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
                query_cumlens = query_cumlens.cpu().tolist()
                seq_lens = seq_lens.cpu().tolist()

            metadata.prefill.query_cumlens = query_cumlens
            metadata.prefill.seq_lens = seq_lens
            metadata.prefill.num_tokens = query_cumlens[-1]

            if model_extra_config.parall_config.ena_swa_attn_seq_parallel:
                assert model_extra_config.parall_config.ena_seq_parallel, (
                    "ena_swa_attn_seq_parallel requires ena_seq_parallel"
                )
                # Reuse CPU query_start_loc (already synced) — same [B+1] layout
                qsl_cpu = common_attn_metadata.query_start_loc_cpu[reqs_start:]
                query_cumlens = qsl_cpu - qsl_cpu[0]
                metadata.prefill.sp_manager = SPManager.init_sp(tok=int(query_cumlens[-1]))
                metadata.prefill.sp_manager.init_sp_attn(
                    query_cumlens=query_cumlens,
                    computed_lens=context_lens_cpu,
                    block_table_ref=metadata.prefill.block_table,
                )

            if hasattr(self, "sink_len") and self.sink_len > 0:
                metadata.prefill.sink_len = self.sink_len
                if model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
                    metadata.prefill.seq_lens[metadata.prefill.seq_lens == 0] = self.sink_len
                else:
                    metadata.prefill.seq_lens = [
                        self.sink_len if seq == 0 else seq 
                        for seq in metadata.prefill.seq_lens
                    ]

        if self.dcp_world_size > 1:
            self.prepare_dcp_slots(metadata)
            self.prepare_dcp_ag_reorg(metadata)
            # for D node in PD-seperate(D-TP >= P-TP) mode
            TP_Convertor.do_scheduled_kv_reorg()

        # slot_mapping_2d for op "npu_ai_infra_scatter_block_update_"
        # for graph capture, only called inside model
        metadata.get_slot_mapping_2d = self._lazy_slot_mapping_2d(metadata)

        return metadata

    def _lazy_slot_mapping_2d(self, metadata):
        def get_slot_mapping_2d():
            if metadata.slot_mapping_2d is None:
                block_size = self.kv_cache_spec.block_size
                metadata.slot_mapping_2d = torch.stack(
                    [
                        metadata.slot_mapping // block_size,
                        metadata.slot_mapping % block_size,
                    ],
                    dim=-1,
                )
            return metadata.slot_mapping_2d

        return get_slot_mapping_2d

    def generate_activate_mask(self, num_actual_tokens):
        self.mc2_mask.fill_(False)
        self.mc2_mask[:num_actual_tokens].fill_(True)
        return self.mc2_mask

    def prepare_dcp_slots(self, metadata):
        slots = metadata.slot_mapping
        cfg = {"dtype": torch.int32, "device": slots.device}
        kv_idx = torch.arange(slots.size(0), **cfg)[slots != -1]

        metadata.dcp_local_kv_idx = kv_idx
        metadata.dcp_local_slots = slots[kv_idx].to(**cfg)

    def prepare_dcp_ag_reorg(self, metadata, pg=128):
        prefill_meta = metadata.prefill
        if prefill_meta is None:
            return
        chunk_ctx = prefill_meta.chunked_context
        if chunk_ctx is None:
            return

        starts = chunk_ctx.starts  # npu.int32[chk, seq]   start_pos of dcp chunks
        cu_lens = chunk_ctx.padded_local_cu_seq_lens  # npu.int32[chk, seq+1] token_num of dcp chunks
        g_cu_lens = chunk_ctx.cu_seq_lens  # npu.int32[chk, seq+1] token_num of chunks
        blk_table = prefill_meta.block_table  # npu.int32[seq, *]     kv_cache slot mapping

        chunk_ctx.dcp_local_idx = [  # kv gather idx for dcp_chunk[i]
            self.paged_index(cu_lens_i, starts_i, blk_table, pg) for cu_lens_i, starts_i in zip(cu_lens, starts)
        ]
        chunk_ctx.dcp_reorg_order = [  # reorg after all-gather
            self.reorg_index(cu_lens_i, g_cu_lens_i, self.dcp_world_size)
            for cu_lens_i, g_cu_lens_i in zip(cu_lens, g_cu_lens)
        ]

    @staticmethod
    def paged_index(
        cu_lens: torch.Tensor,  # int32[seq+1]
        starts: torch.Tensor,  # int32[seq]
        table: torch.Tensor,  # int32[seq, *]
        pg: int = 128,
    ):
        assert cu_lens.dim() == 1
        assert starts.dim() == 1
        assert table.dim() == 2
        assert starts.size(0) + 1 == cu_lens.size(0)
        assert starts.size(0) == table.size(0)
        seq_lens = cu_lens.diff()
        cfg = {"dtype": torch.int32, "device": starts.device}
        serial = torch.arange(cu_lens[-1], **cfg)
        token_seq = torch.arange(seq_lens.size(0), **cfg).repeat_interleave(seq_lens, dim=0)
        token_pos = serial + (starts - cu_lens[:-1]).repeat_interleave(seq_lens, dim=0)
        return table[token_seq, token_pos // pg] * pg + token_pos % pg

    @staticmethod
    def reorg_index(
        local_cu_lens: torch.Tensor,  # int32[seq+1]
        global_cu_lens: torch.Tensor,  # int32[seq+1]
        dcp: int = 16,
    ):
        assert local_cu_lens.dim() == 1
        assert global_cu_lens.dim() == 1
        assert local_cu_lens.size(0) == global_cu_lens.size(0)
        global_lens = global_cu_lens.diff()
        serial = torch.arange(global_cu_lens[-1], dtype=torch.int32, device=global_lens.device)
        token_pos = serial - global_cu_lens[:-1].repeat_interleave(global_lens, dim=0)
        offset = local_cu_lens[:-1].repeat_interleave(global_lens, dim=0)
        return offset + (token_pos // dcp) + (token_pos % dcp * local_cu_lens[-1])


class NPUMLAImpl(MLAAttentionImpl[NPUMLAMetadata]):
    can_return_lse_for_decode: bool = True
    SHARE_MASK_TRIL_SPARSE = None
    MAX_WINDOW_SIZE = 2**20 - 1

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
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = kv_lora_rank
        self.qk_rope_head_dim: int = qk_rope_head_dim

        self.supports_quant_query_input = False
        self.dcp_world_size = -1
        self.q_pad_num_heads = None

        self.sink_k_pe = None
        self.sink_compressed_kv = None
        self.sink_len = 0
        self.sliding_window = sliding_window

        unsupported_features = [alibi_slopes, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError("NPUMLAImpl does not support one of the following: alibi_slopes, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention are not implemented for NPUMLAImpl"
            )

        self.ensure_decode_attn_mask()

    @classmethod
    def ensure_decode_attn_mask(cls) -> None:
        if cls.SHARE_MASK_TRIL_SPARSE is None:
            cls.SHARE_MASK_TRIL_SPARSE = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device="npu"))

    def update_sink_kv(self, sink_k_pe: torch.Tensor, sink_compressed_kv: torch.Tensor) -> None:
        self.sink_k_pe = sink_k_pe.unsqueeze(1)
        self.sink_compressed_kv = sink_compressed_kv.unsqueeze(1)
        self.sink_len = sink_compressed_kv.shape[0]

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        super().process_weights_after_loading(act_dtype)

        self.W_UK_T = self.W_UK_T.contiguous()
        self.W_UV = self.W_UV.contiguous()

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        raise NotImplementedError(
            "NPUMLAImpl does not support vLLM's generic MLA KV-cache update. "
            "PanguV2 updates its split KV cache in npu_pangu_forward."
        )

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: NPUMLAMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        raise NotImplementedError(
            "NPUMLAImpl.forward_mha is intentionally unsupported. PanguV2 "
            "prefill uses the npu_pangu_forward direct path."
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: NPUMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError(
            "NPUMLAImpl.forward_mqa is intentionally unsupported. PanguV2 "
            "decode uses the npu_pangu_forward direct path."
        )


class NPUMLAPrefillBackend(MLAPrefillBackend):
    """Stub MLA prefill backend; actual NPU prefill uses custom ops."""

    @staticmethod
    def get_name() -> str:
        return "NPU_MLA_PREFILL"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "NPUMLAPrefillBackend is a placeholder; NPU MLA prefill uses custom ops."
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "NPUMLAPrefillBackend is a placeholder; NPU MLA prefill uses custom ops."
        )
