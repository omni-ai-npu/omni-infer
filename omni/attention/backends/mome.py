# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import copy
from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from omni_npu import envs
from omni_npu.attention.backends.attention import NPUAttentionBackendImpl
from omni_npu.attention.backends.utils import _maybe_padded_raw_tensor_to_strided_caches, register_attention_backend
from omni_npu.model_config.config_loader.loader import model_extra_config

logger = init_logger(__name__)
NPUPanguMome = "NPUPanguMome"


@register_attention_backend(NPUPanguMome)
class NPUPanguMomeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return NPUPanguMome

    @staticmethod
    def get_builder_cls() -> type["NPUMomeAttentionMetadataBuilder"]:
        return NPUMomeAttentionMetadataBuilder

    @staticmethod
    def reshape_kv_cache(
        raw_tensor: torch.Tensor,
        num_blocks: int,
        kv_cache_spec: "MomeSpec",
    ) -> tuple[torch.Tensor, ...]:
        return _maybe_padded_raw_tensor_to_strided_caches(
            raw_tensor,
            num_blocks=num_blocks,
            block_size=kv_cache_spec.num_total_tokens,
            shapes=kv_cache_spec.shapes,
            dtypes=kv_cache_spec.dtypes,
            page_size_bytes=kv_cache_spec.page_size_bytes,
        )

    @staticmethod
    def get_impl_cls() -> type["NPUAttentionBackendImpl"]:
        return NPUAttentionBackendImpl

    @classmethod
    def is_ssm(cls) -> bool:
        return True

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return True


@dataclass
class FlashComm2Metadata:
    padded_local_size: int  # the padded token number
    token_range: list[list[int]]  # [t][0,1] is the token range on rank t
    local_size: list[int]  # [t] is the number of actual tokens on rank t

    req_idx_start: list[int]  # req_idx_start[t]:req_idx_end[t] is the requests owned by rank t
    req_idx_end: list[int]
    num_valid_cache: list[int]  # the number of requests whose caches are owned by rank t

    max_local_reqs: int

    qsl_local: torch.Tensor  # the query_start_loc on this rank
    num_computed_tokens_local: torch.Tensor  # num_computed_tokens on this rank

    # the following fields are to be used with npu_ai_infra_scatter_block_update_,
    # which limits the hiddem dim <= 2048 to scatter
    rearrange_ratio: int
    cache_indices_rearranged: torch.Tensor  # （bsz * state_len * rearrange_ratio, 2)


@dataclass
class NPUMomeAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int

    # Query and cache management fields
    query_start_loc: torch.Tensor  # shape: [batch + 1,]
    cache_indices: torch.Tensor  # shape: [batch,] or [batch, max_num_blocks]
    max_query_len: int = 1
    pad_slot_id: int = PAD_SLOT_ID
    B_size: int = 1  # Mome block size (kv_cache_spec.block_size)

    # Speculative decoding support
    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # State and token computed tracking
    num_computed_tokens: torch.Tensor | None = None  # shape: [batch,]
    block_idx_last_computed_token: torch.Tensor | None = None  # shape: [batch,]
    block_idx_first_scheduled_token: torch.Tensor | None = None  # shape: [batch,]
    block_idx_last_scheduled_token: torch.Tensor | None = None  # shape: [batch,]


class NPUMomeAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.ALWAYS
    reorder_batch_threshold: int = 1
    supports_update_block_table: bool = False

    # Bound each step by NPUModelRunner; None during dummy runs and capture.
    runner_num_prompt_tokens: torch.Tensor | None = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        from vllm.v1.kv_cache_interface import MomeSpec

        assert isinstance(kv_cache_spec, MomeSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.mome_block_size = kv_cache_spec.block_size
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec

        self.is_pd_disagg = vllm_config.kv_transfer_config is not None
        self.is_decode_node = self.is_pd_disagg and vllm_config.kv_transfer_config.kv_role == "kv_consumer"

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.fake_num_spec = max(self.num_spec, 1) if self.is_pd_disagg else self.num_spec
        self.kernel_width = getattr(vllm_config.model_config.hf_config, "router_sliding_window", 3)
        self.state_len = self.kernel_width - 1 + self.fake_num_spec  # NOTE: need to be consistent with the model script
        self.enable_flashcomm2 = (
            model_extra_config.parall_config.enable_flashcomm2
            and not vllm_config.cache_config.enable_prefix_caching
        )

        self.reuse_prefilled_tokens = envs.OMNI_REUSE_PREFILLED_TOKENS
        self.use_full_cuda_graph = self.compilation_config.cudagraph_mode.has_full_cudagraphs()

        self.decode_cudagraph_max_bs = self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        # Allocate buffers for prefix caching support
        if self.vllm_config.cache_config.enable_prefix_caching:
            max_num_blocks = cdiv(self.vllm_config.model_config.max_model_len, self.mome_block_size)
            self.cache_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs, max_num_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_computed_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_first_scheduled_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_last_scheduled_token = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
        else:
            self.cache_indices_tensor = torch.empty(
                (self.decode_cudagraph_max_bs,),
                dtype=torch.int32,
                device=device,
            )
            self.block_idx_first_scheduled_token = None
            self.block_idx_last_scheduled_token = None

        # Buffers for cudagraph
        self.num_computed_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _compute_prefix_caching_block_indices(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        mome_block_size: int,
        num_accepted_tokens: torch.Tensor,
        num_prompt_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        # compute the block index of the running cache
        if num_prompt_tokens is None or self.num_spec == 0:
            block_idx_last_computed_token = cdiv(num_computed_tokens, mome_block_size) - 1
        else:
            # With speculative decoding and num_computed_tokens > num_prompt_tokens,
            # the last scheduling must be MTP and num_accepted_tokens is meaningful,
            # and we need to get the correct block index for the running cache
            block_idx_last_computed_token = torch.where(
                num_computed_tokens > num_prompt_tokens,
                cdiv(num_computed_tokens - num_accepted_tokens + self.num_spec + 1, mome_block_size) - 1,
                cdiv(num_computed_tokens, mome_block_size) - 1,
            )
        if not self.reuse_prefilled_tokens and self.is_decode_node and num_prompt_tokens is not None:
            # Since there is no chunked prefill or prefix cache hit on the decode node,
            # the only case where num_computed_tokens < num_prompt_tokens is decode recompute,
            # and we need to locate the block of running cache (instead of prefix cache)
            # and read the [-3:-1] tokens from the cache (see below)
            block_idx_last_computed_token = torch.where(
                num_computed_tokens < num_prompt_tokens,
                cdiv(num_computed_tokens + 1, mome_block_size) - 1,
                block_idx_last_computed_token,
            )

        # which is <= block index for the first scheduled token
        block_idx_first_scheduled_token = cdiv(num_computed_tokens + 1, mome_block_size) - 1
        # which is <= block index of the last scheduled token
        block_idx_last_scheduled_token = cdiv(common_attn_metadata.seq_lens, mome_block_size) - 1
        # -1 in case it's non-computed and causes later issues with indexing
        block_idx_last_computed_token = torch.clamp(block_idx_last_computed_token, min=0)
        # -1 in the case we have a padded request (0 seq-len)
        block_idx_last_scheduled_token = torch.clamp(block_idx_last_scheduled_token, min=0)

        return (
            block_idx_last_computed_token,
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
        )

    def _build_for_flashcomm2(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        cache_indices: torch.Tensor,
    ) -> FlashComm2Metadata:

        assert not self.vllm_config.cache_config.enable_prefix_caching, (
            "Currently FlashComm2 is not compatible with prefix caching."
        )

        tp_group = get_tp_group()
        tp_size = tp_group.world_size
        tp_rank = tp_group.rank_in_group
        device = common_attn_metadata.query_start_loc.device

        # determine the padded size -
        # this has to be consistent with how it is done in the model script
        num_actual_tokens = common_attn_metadata.query_start_loc_cpu[-1].item()
        pad_size = (tp_size - (num_actual_tokens % tp_size)) % tp_size
        assert (num_actual_tokens + pad_size) % tp_size == 0
        padded_local_size = (num_actual_tokens + pad_size) // tp_size

        token_range = [
            [min(t * padded_local_size, num_actual_tokens), min((t + 1) * padded_local_size, num_actual_tokens)]
            for t in range(tp_size)
        ]
        local_size = [r[1] - r[0] for r in token_range]

        # determine req_idx_start[t], which is the last index j
        # such that qsl[j] <= token_range[t][0], and
        # req_idx_end[t], which is the first index j
        # such that qsl[j] >= token_range[t][1]
        bsz = common_attn_metadata.num_reqs
        req_idx_start = [0 for _ in range(tp_size)]
        req_idx_end = [0 for _ in range(tp_size)]
        cur_t = 1
        j = 0
        qsl = common_attn_metadata.query_start_loc_cpu.tolist()
        while j <= common_attn_metadata.num_reqs and cur_t < tp_size:
            if qsl[j] > token_range[cur_t][0]:
                req_idx_end[cur_t - 1] = j
                req_idx_start[cur_t] = j - 1
                cur_t += 1
            elif qsl[j] == token_range[cur_t][0]:
                req_idx_end[cur_t - 1] = j
                req_idx_start[cur_t] = j
                cur_t += 1
            else:  # qsl[j] < token_range[cur_t][0]
                j += 1
        # NOTE: there may be trailing ranks that hold no tokens at all
        req_idx_start[cur_t:] = [bsz] * (tp_size - cur_t)
        req_idx_end[cur_t - 1 :] = [bsz] * (tp_size - cur_t + 1)

        temp = [j if ran[1] == qsl[j] else j - 1 for j, ran in zip(req_idx_end, token_range)]
        num_valid_cache = [max(k - j, 0) for j, k in zip(req_idx_start, temp)]
        max_local_reqs = max([k - j for j, k in zip(req_idx_start, req_idx_end)])
        assert max_local_reqs >= 1

        # prepare qsl_local and num_computed_tokens_local,
        # if there are actual tokens on this rank
        if local_size[tp_rank] > 0:
            qsl_local = qsl[req_idx_start[tp_rank] : req_idx_end[tp_rank] + 1]
            qsl_local = [x - token_range[tp_rank][0] for x in qsl_local]
            qsl_local[0] = 0
            qsl_local[-1] = local_size[tp_rank]
            qsl_local = torch.tensor(qsl_local, dtype=torch.int32, device=device)

            num_computed_tokens_local = common_attn_metadata.num_computed_tokens_cpu[
                req_idx_start[tp_rank] : req_idx_end[tp_rank]
            ].clone()
            num_computed_tokens_local[0] += token_range[tp_rank][0] - qsl[req_idx_start[tp_rank]]
            num_computed_tokens_local = num_computed_tokens_local.to(dtype=torch.int32, device=device)
        else:
            qsl_local = None
            num_computed_tokens_local = None

        metadata = FlashComm2Metadata(
            padded_local_size=padded_local_size,
            token_range=token_range,
            local_size=local_size,
            req_idx_start=req_idx_start,
            req_idx_end=req_idx_end,
            num_valid_cache=num_valid_cache,
            max_local_reqs=max_local_reqs,
            qsl_local=qsl_local,
            num_computed_tokens_local=num_computed_tokens_local,
            rearrange_ratio=None,
            cache_indices_rearranged=None,
        )

        self._update_cache_indices_for_flashcomm2(metadata, cache_indices)

        return metadata

    def _update_cache_indices_for_flashcomm2(self, fc2_metadata: FlashComm2Metadata, cache_indices: torch.Tensor):
        device = cache_indices.device
        bsz = cache_indices.shape[0]
        # calculate the rearranged cache indices
        rearrange_ratio = 4  # 8192 (max mome hidden dim) / 2048 (max allowed by scatter) = 4
        token_idx_per_req = torch.arange(
            self.state_len * rearrange_ratio,
            dtype=torch.int32,
            device=device,
        )
        cache_indices_rearranged = torch.empty(
            (bsz, self.state_len * rearrange_ratio, 2),
            dtype=torch.int32,
            device=device,
        )
        cache_indices_rearranged[:, :, 0] = cache_indices.unsqueeze(1)  # (bsz, 1)
        cache_indices_rearranged[:, :, 1] = token_idx_per_req.unsqueeze(0)  # (1, state_len * rearrange_ratio)
        cache_indices_rearranged = cache_indices_rearranged.view(-1, 2)  # (bsz * state_len * rearrange_ratio, 2)
        fc2_metadata.rearrange_ratio = rearrange_ratio
        fc2_metadata.cache_indices_rearranged = cache_indices_rearranged

    # vLLM uses isinstance(builder, GDNAttentionMetadataBuilder) to decide
    # whether to pass speculative-decoding metadata. MoME participates in that
    # builder protocol but intentionally produces its own metadata type.
    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_prompt_tokens: torch.Tensor | None = None,
        allow_runner_num_prompt_tokens: bool = True,
    ) -> NPUMomeAttentionMetadata:
        """
        Build Mome attention metadata from common attention metadata.
        """
        num_reqs = common_attn_metadata.num_reqs

        # num_reqs is the padded count, matching compute_num_computed_tokens().
        if (
            num_prompt_tokens is None
            and allow_runner_num_prompt_tokens
            and self.runner_num_prompt_tokens is not None
        ):
            num_prompt_tokens = self.runner_num_prompt_tokens[:num_reqs]

        # Split decodes and prefills
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.reorder_batch_threshold
        )

        num_computed_tokens = common_attn_metadata.compute_num_computed_tokens()
        block_idx_last_computed_token = None
        block_idx_first_scheduled_token = None
        block_idx_last_scheduled_token = None

        if num_accepted_tokens is None:
            num_accepted_tokens = torch.full_like(num_computed_tokens, self.fake_num_spec + 1)
        elif num_prompt_tokens is not None:  # For graph capture, num_prompt_tokens is None
            num_accepted_tokens = num_accepted_tokens.clone()
            # if the previous schedule is prefill (computed <= prompt), we reset num_accepted_tokens = num_spec + 1
            # so the MoME kernel reads the last few tokens from the cache
            num_accepted_tokens.masked_fill_(num_computed_tokens <= num_prompt_tokens, self.fake_num_spec + 1)

        # if the current schedule is the recompute of the last prompt,
        # we reset num_accepted_tokens = fake_num_spec to let the kernel read the [-3:-1] tokens from the cache
        if self.is_decode_node and num_prompt_tokens is not None:
            num_accepted_tokens.masked_fill_(num_computed_tokens < num_prompt_tokens, self.fake_num_spec)

        # Get cache indices
        apc_enabled = self.vllm_config.cache_config.enable_prefix_caching
        padded_reqs_mask = common_attn_metadata.seq_lens == 0
        if apc_enabled:
            cache_indices = common_attn_metadata.block_table_tensor
            cache_indices = torch.where(
                padded_reqs_mask[:, None],
                PAD_SLOT_ID,
                cache_indices,
            )
            (
                block_idx_last_computed_token,
                block_idx_first_scheduled_token,
                block_idx_last_scheduled_token,
            ) = self._compute_prefix_caching_block_indices(
                common_attn_metadata,
                self.mome_block_size,
                num_accepted_tokens,
                num_prompt_tokens,
            )
        else:
            cache_indices = common_attn_metadata.block_table_tensor[:, 0]
            cache_indices = torch.where(
                padded_reqs_mask,
                PAD_SLOT_ID,
                cache_indices,
            )

        # For cudagraph: copy to persistent buffer if applicable
        if (
            num_prefills == 0
            and num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            self.cache_indices_tensor[:num_decodes].copy_(cache_indices, non_blocking=True)
            cache_indices = self.cache_indices_tensor[
                :num_decodes
            ]  # NOTE: slice to num_decodes instead of num_decode_tokens
            self.cache_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.num_computed_tokens[:num_decodes].copy_(num_computed_tokens, non_blocking=True)
            num_computed_tokens = self.num_computed_tokens[
                :num_decodes  # NOTE: slice to num_decodes instead of num_decode_tokens
            ]

            if num_accepted_tokens is not None:
                self.num_accepted_tokens[:num_decodes].copy_(num_accepted_tokens, non_blocking=True)
                num_accepted_tokens = self.num_accepted_tokens[:num_decodes]

            if self.vllm_config.cache_config.enable_prefix_caching:
                self.block_idx_last_computed_token[:num_decodes].copy_(block_idx_last_computed_token, non_blocking=True)
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :num_decodes  # NOTE: slice to num_decodes instead of num_decode_tokens
                ]

                self.block_idx_first_scheduled_token[:num_decodes].copy_(
                    block_idx_first_scheduled_token, non_blocking=True
                )
                block_idx_first_scheduled_token = self.block_idx_first_scheduled_token[
                    :num_decodes  # NOTE: slice to num_decodes instead of num_decode_tokens
                ]

                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=True
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :num_decodes  # NOTE: slice to num_decodes instead of num_decode_tokens
                ]

        max_query_len = common_attn_metadata.max_query_len

        attn_metadata = NPUMomeAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            cache_indices=cache_indices,
            max_query_len=max_query_len,
            pad_slot_id=PAD_SLOT_ID,
            B_size=self.mome_block_size,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=num_computed_tokens,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            num_reqs=num_reqs,
        )

        if num_prefills == 0:
            attn_metadata.prefill = None
        else:
            prefill_query_start_loc = (
                common_attn_metadata.query_start_loc[num_decodes:] - common_attn_metadata.query_start_loc[num_decodes]
            )
            attn_metadata.prefill = NPUMomeAttentionMetadata(
                num_prefills=num_prefills,
                num_prefill_tokens=num_prefill_tokens,
                num_decodes=0,
                num_decode_tokens=0,
                num_reqs=num_prefills,
                query_start_loc=prefill_query_start_loc,
                cache_indices=cache_indices[num_decodes:],
                max_query_len=max_query_len,
                pad_slot_id=PAD_SLOT_ID,
                B_size=self.mome_block_size,
                num_accepted_tokens=None,
                num_computed_tokens=num_computed_tokens[num_decodes:],
                block_idx_last_computed_token=block_idx_last_computed_token[num_decodes:] if apc_enabled else None,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token[num_decodes:] if apc_enabled else None,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token[num_decodes:] if apc_enabled else None,
            )
        if num_decodes == 0:
            attn_metadata.decode = None
            if self.enable_flashcomm2:
                attn_metadata.fc2_metadata = self._build_for_flashcomm2(
                    common_attn_metadata,
                    cache_indices,
                )
        else:
            attn_metadata.decode = NPUMomeAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                num_reqs=num_decodes,
                query_start_loc=common_attn_metadata.query_start_loc[: num_decodes + 1],
                cache_indices=cache_indices[:num_decodes],
                max_query_len=max_query_len,
                pad_slot_id=PAD_SLOT_ID,
                B_size=self.mome_block_size,
                num_accepted_tokens=num_accepted_tokens[:num_decodes] if num_accepted_tokens is not None else None,
                num_computed_tokens=num_computed_tokens[:num_decodes],
                block_idx_last_computed_token=block_idx_last_computed_token[:num_decodes] if apc_enabled else None,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token[:num_decodes] if apc_enabled else None,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token[:num_decodes] if apc_enabled else None,
            )

        return attn_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
        num_accepted_tokens: torch.Tensor | None = None,
        num_prompt_tokens: torch.Tensor | None = None,
    ) -> NPUMomeAttentionMetadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            fast_build=True,
            num_prompt_tokens=num_prompt_tokens,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert m.num_reqs <= self.decode_cudagraph_max_bs and m.num_actual_tokens <= self.decode_cudagraph_max_bs, (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(
            m.query_start_loc
        )  # always use not-None num_accepted_tokens for graph capturing
        # m carries fabricated seq_lens; build() must not use the bound buffer.
        return self.build(
            0,
            m,
            num_accepted_tokens=num_accepted_tokens,
            allow_runner_num_prompt_tokens=False,
        )

    def update_block_table(
        self,
        metadata: NPUMomeAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> NPUMomeAttentionMetadata:
        new_metadata = copy.copy(metadata)
        prefix_caching = self.vllm_config.cache_config.enable_prefix_caching
        cache_indices = blk_table if prefix_caching else blk_table[:, 0]
        num_accepted_tokens = metadata.num_accepted_tokens
        num_computed_tokens = metadata.num_computed_tokens
        block_idx_last_computed_token = metadata.block_idx_last_computed_token
        block_idx_first_scheduled_token = metadata.block_idx_first_scheduled_token
        block_idx_last_scheduled_token = metadata.block_idx_last_scheduled_token
        num_reqs = blk_table.shape[0]

        # For CUDA graphs, copy to persistent buffer
        if (
            metadata.num_prefills == 0
            and num_reqs <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            persistent_cache_indices = self.cache_indices_tensor[:num_reqs]
            persistent_cache_indices.copy_(cache_indices, non_blocking=True)
            cache_indices = persistent_cache_indices

            if num_computed_tokens is not None:
                persistent_num_computed_tokens = self.num_computed_tokens[:num_reqs]
                persistent_num_computed_tokens.copy_(num_computed_tokens, non_blocking=True)
                num_computed_tokens = persistent_num_computed_tokens

            if num_accepted_tokens is not None:
                persistent_num_accepted_tokens = self.num_accepted_tokens[:num_reqs]
                persistent_num_accepted_tokens.copy_(num_accepted_tokens, non_blocking=True)
                num_accepted_tokens = persistent_num_accepted_tokens

            if prefix_caching:
                persistent_blcct = self.block_idx_last_computed_token[:num_reqs]
                persistent_blcct.copy_(block_idx_last_computed_token, non_blocking=True)
                block_idx_last_computed_token = persistent_blcct

                persistent_bfst = self.block_idx_first_scheduled_token[:num_reqs]
                persistent_bfst.copy_(block_idx_first_scheduled_token, non_blocking=True)
                block_idx_first_scheduled_token = persistent_bfst

                persistent_blst = self.block_idx_last_scheduled_token[:num_reqs]
                persistent_blst.copy_(block_idx_last_scheduled_token, non_blocking=True)
                block_idx_last_scheduled_token = persistent_blst

        new_metadata.cache_indices = cache_indices
        new_metadata.num_accepted_tokens = num_accepted_tokens
        new_metadata.num_computed_tokens = num_computed_tokens
        new_metadata.block_idx_last_computed_token = block_idx_last_computed_token
        new_metadata.block_idx_first_scheduled_token = block_idx_first_scheduled_token
        new_metadata.block_idx_last_scheduled_token = block_idx_last_scheduled_token

        if hasattr(new_metadata, "fc2_metadata"):
            self._update_cache_indices_for_flashcomm2(new_metadata.fc2_metadata, cache_indices)

        if new_metadata.prefill is not None:
            new_metadata.prefill = copy.copy(metadata.prefill)
            new_metadata.prefill.cache_indices = cache_indices[metadata.num_decodes :]
            if num_computed_tokens is not None:
                new_metadata.prefill.num_computed_tokens = num_computed_tokens[metadata.num_decodes :]
            if num_accepted_tokens is not None:
                new_metadata.prefill.num_accepted_tokens = num_accepted_tokens[metadata.num_decodes :]
            if prefix_caching:
                new_metadata.prefill.block_idx_last_computed_token = block_idx_last_computed_token[
                    metadata.num_decodes :
                ]
                new_metadata.prefill.block_idx_first_scheduled_token = block_idx_first_scheduled_token[
                    metadata.num_decodes :
                ]
                new_metadata.prefill.block_idx_last_scheduled_token = block_idx_last_scheduled_token[
                    metadata.num_decodes :
                ]
        if new_metadata.decode is not None:
            new_metadata.decode = copy.copy(metadata.decode)
            new_metadata.decode.cache_indices = cache_indices[: metadata.num_decodes]
            if num_computed_tokens is not None:
                new_metadata.decode.num_computed_tokens = num_computed_tokens[: metadata.num_decodes]
            if num_accepted_tokens is not None:
                new_metadata.decode.num_accepted_tokens = num_accepted_tokens[: metadata.num_decodes]
            if prefix_caching:
                new_metadata.decode.block_idx_last_computed_token = block_idx_last_computed_token[
                    : metadata.num_decodes
                ]
                new_metadata.decode.block_idx_first_scheduled_token = block_idx_first_scheduled_token[
                    : metadata.num_decodes
                ]
                new_metadata.decode.block_idx_last_scheduled_token = block_idx_last_scheduled_token[
                    : metadata.num_decodes
                ]

        return new_metadata


def bind_num_prompt_tokens(attn_groups, num_prompt_tokens: torch.Tensor | None) -> None:
    """Bind the prompt-length buffer on every MoME builder; None to unbind.

    attn_groups nests three levels: kv-cache group, attention group, ubatch.
    """
    for groups in attn_groups:
        for group in groups:
            for builder in group.metadata_builders:
                if isinstance(builder, NPUMomeAttentionMetadataBuilder):
                    builder.runner_num_prompt_tokens = num_prompt_tokens
