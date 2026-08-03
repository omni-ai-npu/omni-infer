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
import torch_npu
from typing_extensions import deprecated

from vllm.platforms import current_platform
from vllm.forward_context import get_forward_context
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    AttentionCGSupport,
    CommonAttentionMetadata
)
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonPrefillMetadata,
    QueryLenSupport,
)
from vllm.v1.attention.backend import (
    AttentionBackend,  # type: ignore
    AttentionCGSupport,
    MLAAttentionImpl

)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.kv_cache_interface import AttentionSpec

from omni_npu.connector.utils import TP_Convertor
from omni_npu.attention import ops
from omni_npu.compilation.utils import (
    capture_graph_task,
    OP_FIA_V1,
    OP_FIA_SINK,
)
from omni_npu.attention.backends.utils import register_attention_backend, _maybe_padded_raw_tensor_to_strided_caches
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
class NPUMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return NPUMLA

    @staticmethod
    def get_builder_cls() -> type["NPUMLAMetadataBuilder"]:
        return NPUMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["NPUMLAImpl"]:
        return NPUMLAImpl

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
    get_slot_mapping_2d = lambda: None


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
            reqs_start = num_decodes  # prefill_start

            # vLLM 0.25.1 provides a CPU upper bound that is exact for
            # prefill rows. Prefer it over the deprecated ``seq_lens_cpu``
            # property, which may introduce a device-to-host synchronization.
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            if seq_lens_cpu is None:
                # Keep compatibility with unit tests and callers that do not
                # populate the 0.25.1 upper-bound buffer.
                seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            prefill_query_lens_cpu = (
                query_start_loc_cpu[reqs_start + 1 : num_reqs + 1]
                - query_start_loc_cpu[reqs_start:num_reqs]
            )
            context_lens_cpu = (
                seq_lens_cpu[reqs_start:num_reqs] - prefill_query_lens_cpu
            )
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
                metadata.decode.mc2_mask = self.generate_activate_mask(
                    metadata.decode.num_actual_tokens,
                )

            if hasattr(self, "sink_len") and self.sink_len > 0:
                # for static sink attention, we need to add the sink length to the seq_lens
                metadata.decode.sink_len = self.sink_len
                if model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
                    metadata.decode.seq_lens[metadata.decode.seq_lens == 0] = self.sink_len
                else:
                    metadata.decode.seq_lens = [
                        self.sink_len if seq == 0 else seq 
                        for seq in metadata.decode.seq_lens
                    ]

        # update prefill metadata
        if metadata.prefill is not None:
            query_cumlens = metadata.prefill.query_start_loc[1:]
            seq_lens = common_attn_metadata.seq_lens[
                metadata.num_decodes : metadata.num_decodes + metadata.num_prefills
            ]
            if not model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
                query_cumlens = query_cumlens.cpu().tolist()
                seq_lens = seq_lens.cpu().tolist()

            metadata.prefill.query_cumlens = query_cumlens
            metadata.prefill.seq_lens = seq_lens
            metadata.prefill.num_tokens = query_cumlens[-1]

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

        self.chunked_prefill_workspace_size = MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
            get_current_vllm_config()
        )

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

    def _v_up_proj(self, x: torch.Tensor):
        x = x.view(self.num_heads, -1, self.kv_lora_rank)

        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        out2 = torch.bmm(x, self.W_UV)
        out_new = out2.transpose(0, 1).contiguous().view(-1, self.num_heads * self.v_head_dim)
        return out_new

    def _compute_prefill_context_dcp(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_meta: MLACommonMetadata,
    ):
        prefill_meta = attn_meta.prefill
        assert prefill_meta is not None
        chunk_ctx = prefill_meta.chunked_context
        assert chunk_ctx is not None

        for i, toks in enumerate(chunk_ctx.seq_tot):  # for each chunk

            def kv_ag_reorg(cache):
                cache = cache.flatten(end_dim=-2)  # [*, 128, D] -> [*, D]
                cache = cache[chunk_ctx.dcp_local_idx[i]]  # prepare local
                cache = get_tp_group().all_gather(cache, dim=0)  # all_gather
                return cache[chunk_ctx.dcp_reorg_order[i]]  # reorg

            kv_c_normed, k_pe = (kv_ag_reorg(it) for it in kv_cache)

            kv_nope = self.kv_b_proj(kv_c_normed)[0]
            kv_nope = kv_nope.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_heads, 1)
            cu_q_lens = prefill_meta.query_cumlens
            cu_kv_lens = chunk_ctx.cu_seq_lens[i, 1:]

            suffix_out, suffix_lse = torch_npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                v,
                query_rope=q_pe,
                key_rope=k_pe,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_heads,
                input_layout="TND",
                actual_seq_lengths=cu_q_lens,
                actual_seq_lengths_kv=cu_kv_lens,
                scale=self.scale,
                softmax_lse_flag=True,
            )

            if i == 0:
                out = suffix_out
                lse = suffix_lse
            else:
                prefix_out = out
                prefix_lse = lse
                out = torch.empty_like(prefix_out, dtype=torch.float32)
                lse = torch.empty_like(prefix_lse, dtype=torch.float32)
                ops.merge_attn_states(
                    output=out,
                    output_lse=lse,
                    prefix_output=prefix_out,
                    prefix_lse=prefix_lse,
                    suffix_output=suffix_out,
                    suffix_lse=suffix_lse,
                )

        return out, lse

    def _compute_prefill_context(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            ops.gather_and_maybe_dequant_cache(
                src_cache=(kv_c_cache, k_pe_cache),
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                batch_size=attn_metadata.num_prefills,
                kv_cache_dtype=self.kv_cache_dtype,
                scale=k_scale,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )

            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            attn_output, attn_softmax_lse = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                v,
                query_rope=q_pe,
                key_rope=k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_heads, 1),
                num_heads=self.num_heads,
                num_key_value_heads=self.num_heads,
                input_layout="TND",
                atten_mask=None,
                sparse_mode=0,  # for prefix, no mask on attention matrix
                actual_seq_lengths=prefill_metadata.query_cumlens,
                actual_seq_lengths_kv=prefill_metadata.chunked_context.cu_seq_lens[i, 1:],
                scale=self.scale,
                next_tokens=0,
                softmax_lse_flag=True,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                ops.merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: NPUMLAMetadata,
        k_scale: torch.Tensor,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None

        has_context = attn_metadata.prefill.chunked_context is not None
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        tnd_cumlens = attn_metadata.prefill.query_cumlens

        if self.sink_len > 0:
            q_nope = q_nope.transpose(0, 1)
            ql_nope = torch.bmm(q_nope, self.W_UK_T).transpose(0, 1)
            query_heads = 1 << (self.num_heads - 1).bit_length()
            pad_len = query_heads - self.num_heads
            if pad_len > 0:
                ql_nope_pad = ql_nope.new_empty((ql_nope.shape[0], pad_len, ql_nope.shape[-1]))
                ql_nope = torch.cat([ql_nope, ql_nope_pad], dim=1)
                q_pe_pad = q_pe.new_empty((q_pe.shape[0], pad_len, q_pe.shape[-1]))
                q_pe = torch.cat([q_pe, q_pe_pad], dim=1)

            if self.sliding_window is not None:
                window_size = self.sliding_window - 1
            else:
                window_size = NPUMLAImpl.MAX_WINDOW_SIZE
            kwargs = {
                "query": ql_nope,
                "key": kv_cache[0],
                "value": kv_cache[0],
                "query_rope": q_pe,
                "key_rope": kv_cache[1],
                "num_query_heads": query_heads,
                "num_key_value_heads": 1,
                "input_layout": "TND",
                "softmax_scale": self.scale,
                "block_table": attn_metadata.prefill.block_table,
                "block_size": 128,
                "actual_seq_qlen": attn_metadata.prefill.query_cumlens,
                "actual_seq_kvlen": attn_metadata.prefill.seq_lens,
                "atten_mask": NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                "sparse_mode": 4,
                "sink_number": self.sink_len,
                "pre_tokens": window_size,
                "next_tokens": 0,
            }
            if model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
                q_cumlens = attn_metadata.prefill.query_cumlens.to(torch.int64)
                kv_lens = attn_metadata.prefill.seq_lens.to(torch.int64)
                meta_data_args = {
                    "num_heads_q": query_heads,
                    "num_heads_kv": 1,
                    "head_dim_qk": ql_nope.shape[-1],
                    "head_dim_v": kv_cache[0].shape[-1],
                    "actual_seq_lengths": q_cumlens,
                    "actual_seq_lengths_kv": kv_lens,
                    "sparse_mode": 4,
                    "pre_tokens": window_size,
                    "next_tokens": 0,
                    "input_layout": "TND",
                    "input_layout_kv": "BnBsH",
                    "rope_head_dim": q_pe.shape[-1],
                    "sink_num": self.sink_len,
                    "block_size": 128,
                }
                meta_data = torch.ops.custom._npu_fused_infer_attention_sink_metadata(**meta_data_args)
                kwargs.update({
                    "actual_seq_qlen": q_cumlens,
                    "actual_seq_kvlen": kv_lens,
                    "meta_data": meta_data,
                })
            output = (
                torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0].transpose(0, 1).contiguous()
            )
            return self._v_up_proj(output[: self.num_heads])

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        if model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
            q_cumlens = tnd_cumlens.to(torch.int64)
            kv_lens = tnd_cumlens.to(torch.int64)
            pre_tokens = (1 << 31) - 1
            next_tokens = 0
            current_stream = torch.npu.current_stream()
            stream_limit = torch.npu.get_stream_limit(current_stream)
            query = q_nope.contiguous()
            query_rope = q_pe.contiguous()
            key = k_nope.contiguous()
            value = v.contiguous()
            key_rope = k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_heads, 1).contiguous()
            meta_data = torch.ops.custom._npu_fused_infer_attention_sink_metadata(
                self.num_heads,
                self.num_heads,
                query.shape[-1],
                value.shape[-1],
                actual_seq_lengths=q_cumlens,
                actual_seq_lengths_kv=kv_lens,
                batch_size=q_cumlens.shape[0],
                sparse_mode=3,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                input_layout="TND",
                input_layout_kv="TND",
                sink_num=0,
                k_sink_num=0,
                rope_head_dim=q_pe.shape[-1],
                aic_core_num=stream_limit["cube_core_num"],
                aiv_core_num=stream_limit["vector_core_num"],
            )
            output, output_lse = torch.ops.custom.npu_fused_infer_attention_sink(
                query=query,
                key=key,
                value=value,
                query_rope=query_rope,
                key_rope=key_rope,
                atten_mask=NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=q_cumlens,
                actual_seq_kvlen=kv_lens,
                meta_data=meta_data,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_heads,
                softmax_scale=self.scale,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                input_layout="TND",
                sparse_mode=3,
                sink_number=0,
                return_softmax_lse=has_context,
            )
        else:
            output, output_lse = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                v,
                query_rope=q_pe,
                key_rope=k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_heads, 1),
                num_heads=self.num_heads,
                num_key_value_heads=self.num_heads,
                input_layout="TND",
                atten_mask=NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                sparse_mode=3,
                actual_seq_lengths=tnd_cumlens,
                actual_seq_lengths_kv=tnd_cumlens,
                scale=self.scale,
                next_tokens=0,
                softmax_lse_flag=has_context,
            )

        if has_context:
            if self.dcp_world_size > 1:
                context_output, context_lse = self._compute_prefill_context_dcp(
                    q_nope,
                    q_pe,
                    kv_cache,
                    attn_metadata,
                )  # DCP not support scaled kvcache now
            else:
                context_output, context_lse = self._compute_prefill_context(
                    q_nope=q_nope,
                    q_pe=q_pe,
                    kv_c_cache=kv_cache[0],
                    k_pe_cache=kv_cache[1],
                    attn_metadata=attn_metadata,
                    k_scale=k_scale,
                )
            merged_output = torch.empty_like(output, dtype=torch.float32)
            ops.merge_attn_states(
                output=merged_output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=output,
                suffix_lse=output_lse,
            )
            output = merged_output

        return output.to(model_extra_config.dtype).flatten(start_dim=-2)

    def _forward_decode_dcp(
        self,
        ql_nope: torch.Tensor,  # [bs, 8, 512]
        q_pe: torch.Tensor,  # [bs, 8, 512]
        kv_cache: Tuple[torch.Tensor, torch.Tensor],  # [*, pg, 512], [*, pg, 64]
        attn_meta: NPUMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        decode_meta = attn_meta.decode
        assert decode_meta is not None

        blk_table = decode_meta.block_table
        cu_q_lens = decode_meta.query_cumlens
        cu_kv_lens = decode_meta.seq_lens

        tp_group = get_tp_group().device_group
        tp_size = get_tp_group().world_size
        D = self.kv_lora_rank  # only support D=512
        N = self.num_heads * tp_size  # only support head=128

        def gather_head(q, N, D):
            assert q.dim() == 3
            q = q.transpose(0, 1).flatten()  # TND -> NTD
            q = get_tp_group().all_gather(q)
            return q.view(N, -1, D).transpose(0, 1)  # NTD -> TND

        # DCP does not yet support graph mode or sink attn
        num_tokens = decode_meta.query_cumlens[-1]
        ql_nope = ql_nope[:num_tokens]
        q_pe = q_pe[:num_tokens]

        full_q_nope = gather_head(ql_nope, N, D)  # TND
        full_q_rope = gather_head(q_pe, N, 64)  # TND

        out, lse = torch.ops.npu.npu_fused_infer_attention_score(
            full_q_nope,  # [T, N, D]
            kv_cache[0],  # [*, pg, D]
            kv_cache[0],  # [*, pg, D]
            query_rope=full_q_rope,  # [T, N, 64]
            key_rope=kv_cache[1],  # [*, pg, 64]
            num_heads=N,
            num_key_value_heads=1,
            input_layout="TND_NTD",
            scale=self.scale,
            sparse_mode=3,
            atten_mask=NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
            block_size=128,
            block_table=blk_table,
            actual_seq_lengths=cu_q_lens,
            actual_seq_lengths_kv=cu_kv_lens,
            softmax_lse_flag=True,
        )  # -> out[N, T, D], lse[T, N, 1]

        cp_out = out.view(N, -1)  # bf16[N, TD]
        cp_lse = lse.view(-1, N).transpose(0, 1)  # fp32[N, T]
        tp_out = torch.empty_like(cp_out)  # bf16[N, TD]
        tp_lse = torch.empty_like(cp_lse)  # fp32[N, T]

        torch.distributed.all_to_all_single(tp_out.flatten(), cp_out.flatten(), group=tp_group)
        torch.distributed.all_to_all_single(tp_lse.flatten(), cp_lse.flatten(), group=tp_group)

        sect = [self.num_heads] * tp_size  # head split pattern

        # TODO: "npu_attention_update" does not yet support tp > 16
        if tp_size <= 16:
            merged, _ = torch_npu.npu_attention_update(
                lse=[it.flatten() for it in tp_lse.split(sect, dim=0)],
                local_out=[it.view(-1, D) for it in tp_out.float().split(sect, dim=0)],
                update_type=0,
            )
        else:
            merged, _ = ops.attention_update_torch(
                outs=[it.view(-1, D) for it in tp_out.float().split(sect, dim=0)],
                lses=[it.flatten() for it in tp_lse.split(sect, dim=0)],
            )
        return merged.to(model_extra_config.dtype).view(self.num_heads, -1, D)  # NTD

    def _forward_decode(
        self,
        decode_ql_nope: torch.Tensor,
        decode_q_pe: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: NPUMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert attn_metadata.decode is not None

        if self.sink_len > 0:
            query_heads = 1 << (self.num_heads - 1).bit_length()
            pad_len = query_heads - self.num_heads
            ql_nope_pad = decode_ql_nope.new_empty((decode_ql_nope.shape[0], pad_len, decode_ql_nope.shape[-1]))
            decode_ql_nope = torch.cat([decode_ql_nope, ql_nope_pad], dim=1)
            q_pe_pad = decode_q_pe.new_empty((decode_q_pe.shape[0], pad_len, decode_q_pe.shape[-1]))
            decode_q_pe = torch.cat([decode_q_pe, q_pe_pad], dim=1)
        else:
            query_heads = self.num_heads

        # In graph/dummy-run mode, decode queries may be padded while
        # actual_seq_lengths still records real token count.
        # Keep query T aligned with actual_seq_lengths[-1] for TND kernels.
        if model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
            num_tokens = attn_metadata.decode.num_tokens
        else:
            num_tokens = attn_metadata.decode.query_cumlens[-1]
        decode_ql_nope = decode_ql_nope[:num_tokens]
        decode_q_pe = decode_q_pe[:num_tokens]
        forward_context = get_forward_context()
        if self.sink_len > 0:
            if self.sliding_window is not None:
                window_size = self.sliding_window - 1
            else:
                window_size = NPUMLAImpl.MAX_WINDOW_SIZE
            kwargs = {
                "query": decode_ql_nope,
                "key": kv_cache[0],
                "value": kv_cache[0],
                "query_rope": decode_q_pe,
                "key_rope": kv_cache[1],
                "num_query_heads": query_heads,
                "num_key_value_heads": 1,
                "input_layout": "TND",
                "softmax_scale": self.scale,
                "block_table": attn_metadata.decode.block_table,
                "block_size": 128,
                "actual_seq_qlen": attn_metadata.decode.query_cumlens,
                "actual_seq_kvlen": attn_metadata.decode.seq_lens,
                "atten_mask": NPUMLAImpl.SHARE_MASK_TRIL_SPARSE,
                "sparse_mode": 4,
                "sink_number": self.sink_len,
                "pre_tokens": window_size,
                "next_tokens": 0,
            }
            attn_output_shape = (num_tokens, query_heads, self.kv_lora_rank)
            attn_output = torch.empty(attn_output_shape, dtype=decode_ql_nope.dtype, device=decode_ql_nope.device)
            softmax_lse = torch.empty(num_tokens, dtype=decode_ql_nope.dtype, device=decode_ql_nope.device)
            if forward_context.capturing and not model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
                capture_graph_task(
                    op_desc=OP_FIA_SINK,
                    op_kwargs=kwargs,
                    out_tensors=[attn_output, softmax_lse],
                    num_tokens=num_tokens,
                    layer_name=layer.layer_name,
                )
                output = attn_output.transpose(0, 1).contiguous()
            elif model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
                q_cumlens = attn_metadata.decode.query_cumlens.to(torch.int64)
                kv_lens = attn_metadata.decode.seq_lens.to(torch.int64)
                current_stream = torch.npu.current_stream()
                stream_limit = torch.npu.get_stream_limit(current_stream)
                meta_data_args = {
                    "num_heads_q": query_heads,
                    "num_heads_kv": 1,
                    "head_dim_qk": decode_ql_nope.shape[-1],
                    "head_dim_v": kv_cache[0].shape[-1],
                    "actual_seq_lengths": q_cumlens,
                    "actual_seq_lengths_kv": kv_lens,
                    "sparse_mode": 4,
                    "pre_tokens": window_size,
                    "next_tokens": 0,
                    "input_layout": "TND",
                    "input_layout_kv": "BnBsH",
                    "rope_head_dim": decode_q_pe.shape[-1],
                    "sink_num": self.sink_len,
                    "block_size": 128,
                    "aic_core_num": stream_limit["cube_core_num"],
                    "aiv_core_num": stream_limit["vector_core_num"],
                }
                meta_data = torch.ops.custom._npu_fused_infer_attention_sink_metadata(**meta_data_args)
                kwargs.update({
                    "actual_seq_qlen": q_cumlens,
                    "actual_seq_kvlen": kv_lens,
                    "meta_data": meta_data,
                })
                output = torch.ops.custom.npu_fused_infer_attention_sink(
                    **kwargs
                )[0].transpose(0, 1).contiguous()
            else:
                output = (
                    torch.ops.custom.npu_fused_infer_attention_sink(**kwargs)[0].transpose(0, 1).contiguous()
                )  # TND -> NTD
        else:
            num_kv_heads = 1
            input_layout = "TND_NTD"
            attn_mask = NPUMLAImpl.SHARE_MASK_TRIL_SPARSE
            sparse_mode = 3
            block_size = 128
            attn_output_shape = (query_heads, num_tokens, self.kv_lora_rank)
            attn_output = torch.empty(attn_output_shape, dtype=decode_ql_nope.dtype, device=decode_ql_nope.device)
            softmax_lse = torch.empty(num_tokens, dtype=decode_ql_nope.dtype, device=decode_ql_nope.device)
            kwargs = {
                "query": decode_ql_nope,
                "key": kv_cache[0],
                "value": kv_cache[0],
                "query_rope": decode_q_pe,
                "key_rope": kv_cache[1],
                "num_heads": query_heads,
                "num_key_value_heads": num_kv_heads,
                "input_layout": input_layout,
                "scale": self.scale,
                "antiquant_mode": 0,
                "antiquant_scale": None,
                "block_table": attn_metadata.decode.block_table,
                "block_size": block_size,
                "actual_seq_lengths": attn_metadata.decode.query_cumlens,
                "actual_seq_lengths_kv": attn_metadata.decode.seq_lens,
                "atten_mask": attn_mask,
                "sparse_mode": sparse_mode,
            }
            use_aicpu_fa_tiling = model_extra_config.operator_opt_config.use_aicpu_fa_tiling
            if forward_context.capturing and not use_aicpu_fa_tiling:
                capture_graph_task(
                    op_desc=OP_FIA_V1,
                    op_kwargs=kwargs,
                    out_tensors=[attn_output, softmax_lse],
                    num_tokens=num_tokens,
                    layer_name=layer.layer_name,
                )
                output = attn_output
            elif use_aicpu_fa_tiling:
                q_cumlens = attn_metadata.decode.query_cumlens.to(torch.int64)
                kv_lens = attn_metadata.decode.seq_lens.to(torch.int64)
                pre_tokens = (1 << 31) - 1
                next_tokens = (1 << 31) - 1
                current_stream = torch.npu.current_stream()
                stream_limit = torch.npu.get_stream_limit(current_stream)
                meta_data = torch.ops.custom._npu_fused_infer_attention_sink_metadata(
                    query_heads,
                    num_kv_heads,
                    decode_ql_nope.shape[-1],
                    kv_cache[0].shape[-1],
                    actual_seq_lengths=q_cumlens,
                    actual_seq_lengths_kv=kv_lens,
                    batch_size=attn_metadata.decode.block_table.shape[0],
                    sparse_mode=sparse_mode,
                    pre_tokens=pre_tokens,
                    next_tokens=next_tokens,
                    input_layout="TND",
                    input_layout_kv="BnBsH",
                    sink_num=0,
                    k_sink_num=0,
                    rope_head_dim=decode_q_pe.shape[-1],
                    block_size=block_size,
                    aic_core_num=stream_limit["cube_core_num"],
                    aiv_core_num=stream_limit["vector_core_num"],
                )
                output = torch.ops.custom.npu_fused_infer_attention_sink(
                    query=decode_ql_nope,
                    key=kv_cache[0],
                    value=kv_cache[0],
                    query_rope=decode_q_pe,
                    key_rope=kv_cache[1],
                    atten_mask=attn_mask,
                    actual_seq_qlen=q_cumlens,
                    actual_seq_kvlen=kv_lens,
                    block_table=attn_metadata.decode.block_table,
                    meta_data=meta_data,
                    num_query_heads=query_heads,
                    num_key_value_heads=num_kv_heads,
                    softmax_scale=self.scale,
                    pre_tokens=pre_tokens,
                    next_tokens=next_tokens,
                    input_layout=input_layout,
                    sparse_mode=sparse_mode,
                    block_size=block_size,
                    sink_number=0,
                )[0]
            else:
                # output shape: (N, T, D)
                output = torch.ops.npu.npu_fused_infer_attention_score(**kwargs)[0]

        if self.sink_len > 0:
            output = output[: self.num_heads]

        return output

    @deprecated(
        "Legacy vLLM 0.14 MLA entry point. vLLM 0.25.1 dispatches through "
        "do_kv_cache_update(), forward_mha(), and forward_mqa(). "
        "PanguV2 uses npu_pangu_forward()."
    )
    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: NPUMLAMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deprecated vLLM 0.14 monolithic MLA implementation.

        This method is not called by the vLLM 0.25.1 MLA pipeline. It is
        temporarily retained as a migration reference and must not be treated
        as a fallback for the generic MLA path.
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for NPUMLAImpl")

        if attn_metadata is None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens
        assert isinstance(kv_cache, (list, tuple)), f"{type(kv_cache)=}."
        assert len(kv_cache) == 2, f"{len(kv_cache)=}."
        k_nope, k_rope = kv_cache
        assert isinstance(k_nope, torch.Tensor) and isinstance(k_rope, torch.Tensor), (
            f"{type(k_nope)=}, {type(k_rope)=}."
        )
        assert k_nope.numel() > 0 and k_rope.numel() > 0, f"{k_nope.shape=}, {k_rope.shape=}"

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]

        def store_kv(cache, kv):
            if self.dcp_world_size == 1:
                slots = attn_metadata.slot_mapping
            else:
                # TODO: DCP not yet support graph mode
                slots = attn_metadata.dcp_local_slots
                kv = kv[attn_metadata.dcp_local_kv_idx]
            cache = cache.flatten(end_dim=-2)
            slots = slots.view(-1, 1)
            torch_npu.npu_scatter_nd_update_(cache, slots, kv)

        # write the latent and rope to kv cache
        store_kv(kv_cache[0], k_c_normed)
        store_kv(kv_cache[1], k_pe.squeeze(1))

        if has_prefill:
            output = output[:num_actual_toks, ...]
            q = q[:num_actual_toks, ...]
            k_c_normed = k_c_normed[:num_actual_toks, ...]
            k_pe = k_pe[:num_actual_toks, ...]
            prefill_q = q[num_decode_tokens:]
            prefill_k_pe = k_pe[num_decode_tokens:]
            prefill_k_c_normed = k_c_normed[num_decode_tokens:]
            output[num_decode_tokens:] = self._forward_prefill(
                prefill_q,
                prefill_k_c_normed,
                prefill_k_pe,
                kv_cache,
                attn_metadata,
                layer._k_scale,
            )
        if has_decode:
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)
            N, B, P = decode_q_nope.shape
            _, _, L = self.W_UK_T.shape
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

            # call decode attn
            if self.dcp_world_size == 1:
                attn_out = self._forward_decode(decode_ql_nope, decode_q_pe, kv_cache, attn_metadata, layer)
            else:
                attn_out = self._forward_decode_dcp(decode_ql_nope, decode_q_pe, kv_cache, attn_metadata, layer)

            # v_up projection
            out_proj = self._v_up_proj(attn_out)
            output[: out_proj.shape[0]].copy_(out_proj)
        return output_padded

    @staticmethod
    def _insert_tensor_by_start_loc(
        raw_tensor: torch.Tensor, insert_segment: torch.Tensor, start_loc: list[int]
    ) -> torch.Tensor:
        segment_len = insert_segment.shape[0]
        num_inserts = len(start_loc) - 1
        total_len = segment_len * num_inserts + raw_tensor.shape[0]
        offset = 0
        # allocate result tensor
        result = torch.empty(total_len, *raw_tensor.shape[1:], device=raw_tensor.device, dtype=raw_tensor.dtype)

        for i in range(num_inserts):
            # write insert segment to result
            result[offset : offset + segment_len] = insert_segment
            offset += segment_len
            # write raw tensor to result
            seg_len = start_loc[i + 1] - start_loc[i]
            result[offset : offset + seg_len] = raw_tensor[start_loc[i] : start_loc[i + 1]]
            offset += seg_len

        return result

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
