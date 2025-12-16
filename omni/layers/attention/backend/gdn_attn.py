# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""
from dataclasses import dataclass
from typing import ClassVar, Optional, List

import torch
import numpy as np

from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from omni.layers.attention.backend.attention import AscendMetadata, AscendAttentionMetadataBuilder, AscendAttentionState


class GDNAttentionBackend(AttentionBackend):

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder


@dataclass
class GDNAttentionMetadata:
    block_tables: torch.Tensor  # not used
    query_lens: torch.Tensor  # not used
    query_lens_list: List  # not used
    seq_lens: torch.Tensor  # not used
    seq_lens_list: List  # not used
    max_query_len: Optional[int]  # not used
    slot_mapping: torch.Tensor  # not used
    slot_indices: torch.Tensor  # not used
    is_only_prefill: bool  # not used
    attn_state: AscendAttentionState  # not used
    cos: Optional[torch.Tensor]  # not used
    sin: Optional[torch.Tensor]  # not used
    is_pd_seperate_d: bool  # not used
    kv_index: Optional[torch.Tensor]  # not used

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    have_decode: bool
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: Optional[torch.Tensor] = None

    spec_query_start_loc: Optional[
        torch.Tensor] = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: Optional[
        torch.Tensor] = None  # shape: [batch - num_spec_decodes + 1,]

    spec_state_indices_tensor: Optional[
        torch.Tensor] = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: Optional[
        torch.Tensor] = None  # shape: [batch - num_spec_decodes,]
    spec_sequence_masks: Optional[torch.Tensor] = None  # shape: [batch,]
    spec_token_masks: Optional[
        torch.
        Tensor] = None  # shape: [num_prefill_tokens + num_decode_tokens,]
    num_accepted_tokens: Optional[torch.Tensor] = None  # shape: [batch,]
    num_spec_tokens: int = 0

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: Optional[dict] = None
    cu_seqlen: Optional[int] = None
    batch_ptr: Optional[torch.Tensor] = None
    token_chunk_offset_ptr: Optional[torch.Tensor] = None


class GDNAttentionMetadataBuilder(AscendAttentionMetadataBuilder):

    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(self, runner, kv_cache_spec = None,
                 block_table = None):
        assert isinstance(kv_cache_spec, MambaSpec)
        super().__init__(runner, kv_cache_spec, block_table)
        self.vllm_config = runner.vllm_config
        self.compilation_config = self.vllm_config.compilation_config
        self.speculative_config = self.vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        if self.speculative_config:
            self.num_spec = self.speculative_config.num_speculative_tokens  # noqa: E501
        else:
            self.num_spec = 0
        self.use_spec_decode = self.num_spec > 0
        self.reorder_batch_threshold = self.num_spec + 1  # type: ignore[misc]

        self.use_full_cuda_graph = False
        self.decode_cudagraph_max_bs = min(
            self.vllm_config.scheduler_config.max_num_seqs *
            (self.num_spec + 1), self.compilation_config.max_capture_size)

        self.spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=runner.device,
        )
        self.non_spec_state_indices_tensor = torch.empty(
            (self.decode_cudagraph_max_bs, ),
            dtype=torch.int32,
            device=runner.device,
        )
        self.spec_sequence_masks = torch.empty(
            (self.decode_cudagraph_max_bs, ),
            dtype=torch.bool,
            device=runner.device,
        )
        self.spec_token_masks = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1), ),
            dtype=torch.bool,
            device=runner.device,
        )
        self.spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1, ),
            dtype=torch.int32,
            device=runner.device,
        )
        self.non_spec_query_start_loc = torch.empty(
            (self.decode_cudagraph_max_bs + 1, ),
            dtype=torch.int32,
            device=runner.device,
        )
        self.num_accepted_tokens = torch.empty(
            (self.decode_cudagraph_max_bs, ),
            dtype=torch.int32,
            device=runner.device,
        )

    def build(  # type: ignore[override]
        self,
        num_reqs,
        num_actual_tokens,
        max_query_len,
        common_attn_metadata: CommonAttentionMetadata,
        num_computed_tokens_cpu_tensor = None,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        num_draft_tokens: Optional[torch.Tensor] = None,
        fast_build: bool = False,
        graph_pad_size: int = -1,
        num_spec_tokens: int = 0,
        **kwargs,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata
        query_start_loc = m.query_start_loc.to(self.runner.device)
        context_lens = num_computed_tokens_cpu_tensor
        context_lens_tensor = context_lens.to(query_start_loc.device)
        seq_lens_tensor = m.seq_lens
        max_num_reqs = seq_lens_tensor.shape[0]        
        if (not self.use_spec_decode or num_draft_tokens is None
                or num_draft_tokens.sum().item() == 0):
            spec_sequence_masks = None
        else:
            spec_sequence_masks = torch.full((max_num_reqs,), True, dtype=torch.bool, device=query_start_loc.device)
            if spec_sequence_masks.sum().item() == 0:
                spec_sequence_masks = None
        if graph_pad_size > 0 and self.runner.attn_state == AscendAttentionState.DecodeOnly:
            padding = torch.full((graph_pad_size // (num_spec_tokens + 1), ),
                                    query_start_loc[-1],
                                    dtype=query_start_loc.dtype,
                                    device=self.runner.device)
            query_start_loc = torch.cat([query_start_loc.to(padding.device), padding], dim=0)
        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(num_reqs, num_actual_tokens, max_query_len, m, decode_threshold=1))
            num_spec_decodes = 0
            num_spec_decode_tokens = 0
            spec_token_masks = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = self.block_table.block_table[:, 0]
            if num_decodes == 0:
                non_spec_state_indices_tensor[num_prefills:] = 0
            if num_prefills == 0:
                non_spec_state_indices_tensor[num_decodes:] = 0
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            num_accepted_tokens = None
        else:
            num_spec_decodes = spec_sequence_masks.sum().item()
            query_lens = query_start_loc[1:] - query_start_loc[:-1]

            non_spec_query_lens = query_lens[~spec_sequence_masks]
            num_decodes = (non_spec_query_lens == 1).sum().item()
            num_prefills = non_spec_query_lens.size(0) - num_decodes
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens.sum().item(
            ) - num_decode_tokens

            if num_prefills == 0 and num_decodes == 0:
                spec_token_masks = torch.ones(
                    (num_spec_decodes * (self.num_spec + 1)),
                    dtype=torch.bool,
                    device=query_start_loc.device)
                spec_state_indices_tensor = self.block_table.block_table[:, :self.
                                                                 num_spec + 1]
                # need to clean kv cache indices for padding slots
                num_spec_reqs = num_spec_decodes - graph_pad_size // (num_spec_tokens + 1)
                spec_state_indices_tensor[num_spec_reqs:,...] = 0

                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc
                non_spec_query_start_loc = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens)
                spec_state_indices_tensor = self.block_table.block_table[
                    spec_sequence_masks, :self.num_spec + 1]
                non_spec_state_indices_tensor = \
                    self.block_table.block_table[~spec_sequence_masks, 0]

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device)
                torch.cumsum(query_lens[spec_sequence_masks],
                             dim=0,
                             out=spec_query_start_loc[1:])
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device)
                torch.cumsum(query_lens[~spec_sequence_masks],
                             dim=0,
                             out=non_spec_query_start_loc[1:])

            num_spec_decode_tokens = (query_lens.sum().item() -
                                      num_prefill_tokens - num_decode_tokens)
            assert num_accepted_tokens is not None
            # num_accepted_tokens: pad to max_num_reqs as tensor, fill missing with 0
            assert(num_accepted_tokens.shape[0] <= max_num_reqs)
            pad_size = max_num_reqs - num_accepted_tokens.shape[0]
            num_accepted_tokens = torch.cat(
                [num_accepted_tokens, torch.zeros((pad_size,), dtype=num_accepted_tokens.dtype, device=num_accepted_tokens.device)],
                dim=0
            )

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks]
        else:
            has_initial_state = None
        num_actual_tokens = num_prefill_tokens + num_decode_tokens + \
            num_spec_decode_tokens

        # prepare tensors for cudagraph
        #
        # With speculative decoding, the xgrammar backend may rollback tokens
        # and causing some sequences has less draft tokens than self.num_spec.
        #
        # In above cases, the max possible batch size for n tokens, can be
        # min(n, cudagraph_max_bs).
        if (self.use_full_cuda_graph and num_prefills == 0 and num_decodes == 0
                and num_spec_decodes <= self.decode_cudagraph_max_bs
                and num_spec_decode_tokens <= self.decode_cudagraph_max_bs):
            num_actual_tokens = self.vllm_config.pad_for_cudagraph(
                num_actual_tokens)
            batch_size = min(self.decode_cudagraph_max_bs, num_actual_tokens)

            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True)
            spec_state_indices_tensor = self.spec_state_indices_tensor[:
                                                                       batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks, non_blocking=True)
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert spec_token_masks is not None
            self.spec_token_masks[:spec_token_masks.size(0)].copy_(
                spec_token_masks, non_blocking=True)
            spec_token_masks = self.spec_token_masks[:num_actual_tokens]
            spec_token_masks[spec_token_masks.size(0):].fill_(False)

            self.spec_query_start_loc[:num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True)
            spec_num_query_tokens = spec_query_start_loc[
                -1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[:batch_size + 1]
            spec_query_start_loc[num_spec_decodes +
                                 1:].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True)
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (self.use_full_cuda_graph and num_prefills == 0
                and num_spec_decodes == 0
                and num_decodes <= self.decode_cudagraph_max_bs):
            num_actual_tokens = self.vllm_config.pad_for_cudagraph(
                num_actual_tokens)
            batch_size = num_actual_tokens

            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True)
            non_spec_state_indices_tensor = \
                self.non_spec_state_indices_tensor[:batch_size]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            self.non_spec_query_start_loc[:num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True)
            non_spec_num_query_tokens = non_spec_query_start_loc[
                -1]  # type: ignore[index]
            non_spec_query_start_loc = \
                self.non_spec_query_start_loc[:batch_size + 1]
            non_spec_query_start_loc[num_decodes +
                                     1:].fill_(non_spec_num_query_tokens)

        ascend_attn_metadata = super().build(
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            common_prefix_len=None,
            graph_pad_size=graph_pad_size,
            **kwargs,
        )

        attn_metadata = GDNAttentionMetadata(
            block_tables=ascend_attn_metadata.block_tables,
            query_lens=ascend_attn_metadata.query_lens,
            query_lens_list=ascend_attn_metadata.query_lens_list,
            seq_lens=ascend_attn_metadata.seq_lens,
            seq_lens_list=ascend_attn_metadata.seq_lens_list,
            max_query_len=ascend_attn_metadata.max_query_len,
            slot_mapping=ascend_attn_metadata.slot_mapping,
            slot_indices=ascend_attn_metadata.slot_indices,
            is_only_prefill=ascend_attn_metadata.is_only_prefill,
            attn_state=ascend_attn_metadata.attn_state,
            cos=ascend_attn_metadata.cos,
            sin=ascend_attn_metadata.sin,
            is_pd_seperate_d=ascend_attn_metadata.is_pd_seperate_d,
            kv_index=ascend_attn_metadata.kv_index,
            # up to here, just copy AscendMetadata and not used.
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            have_decode=(num_decodes > 0),
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_masks=spec_token_masks,
            num_accepted_tokens=num_accepted_tokens,
            num_spec_tokens=num_spec_tokens,
        )
        return attn_metadata

    def mark_static_for_attn_metadata(self, attn_metadata):
        if attn_metadata is not None:
            if attn_metadata.non_spec_query_start_loc is not None:
                torch._dynamo.mark_static(attn_metadata.non_spec_query_start_loc)
            if attn_metadata.spec_query_start_loc is not None:
                torch._dynamo.mark_static(attn_metadata.spec_query_start_loc)
            # torch._dynamo.mark_static(attn_metadata.query_lens)
            # torch._dynamo.mark_static(attn_metadata.seq_lens)

    def build_dummy(  # type: ignore[override]
        self,
        num_tokens: int, 
        max_pad_size: int = -1,
    ) -> GDNAttentionMetadata:

        if max_pad_size == -1:
            max_pad_size = self.runner.max_batch_size
        slot_mapping = torch.zeros(max_pad_size,
                                   dtype=self.runner.slot_mapping_cpu.dtype,
                                   device=self.runner.device)
        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_pad_size, self.runner.graph_block_tables.shape[1]))
        block_table = graph_block_tables.to(
            device=self.runner.device,
            dtype=self.runner.input_batch.block_table[0].get_device_tensor(max_pad_size).dtype
        )

        query_lens = torch.ones(max_pad_size, dtype=torch.long, device=self.runner.device, pin_memory=True)
        seq_lens = query_lens * 2

        slot_indices = torch.stack([slot_mapping // self.block_size, slot_mapping % self.block_size], dim=1)

        fake_positions = torch.zeros(max_pad_size, dtype=torch.int64, device=self.device)

        cos, sin = None, None

        is_pd_seperate_d = self.runner.vllm_config.kv_transfer_config is not None and \
                           self.runner.vllm_config.kv_transfer_config.kv_role == 'kv_consumer'

        non_spec_query_start_loc = torch.tensor([0, 1], device=self.runner.device)
        non_spec_state_indices_tensor = self.block_table.block_table[:, 0]

        ascend_attn_metadata = AscendMetadata(
            num_actual_tokens=num_tokens,
            block_tables=block_table,
            query_lens=query_lens,
            query_lens_list=query_lens.tolist(),
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            slot_mapping=slot_mapping,
            slot_indices=slot_indices,
            is_only_prefill=False,
            attn_state=self.runner.attn_state,
            cos=cos,
            sin=sin,
            is_pd_seperate_d=is_pd_seperate_d
        )

          
        attn_metadata = GDNAttentionMetadata(
            block_tables=ascend_attn_metadata.block_tables,
            query_lens=ascend_attn_metadata.query_lens,
            query_lens_list=ascend_attn_metadata.query_lens_list,
            seq_lens=ascend_attn_metadata.seq_lens,
            seq_lens_list=ascend_attn_metadata.seq_lens_list,
            max_query_len=ascend_attn_metadata.max_query_len,
            slot_mapping=ascend_attn_metadata.slot_mapping,
            slot_indices=ascend_attn_metadata.slot_indices,
            is_only_prefill=ascend_attn_metadata.is_only_prefill,
            attn_state=ascend_attn_metadata.attn_state,
            cos=ascend_attn_metadata.cos,
            sin=ascend_attn_metadata.sin,
            is_pd_seperate_d=ascend_attn_metadata.is_pd_seperate_d,
            kv_index=ascend_attn_metadata.kv_index,
            # up to here, just copy AscendMetadata
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=num_tokens,
            have_decode=(num_tokens > 0),
            num_decode_tokens=num_tokens,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=num_tokens,
            has_initial_state=None,
            spec_query_start_loc=None,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=None,
            spec_token_masks=None,
            num_accepted_tokens=None,
            num_spec_tokens=0,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
            self, common_attn_metadata: CommonAttentionMetadata):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (m.num_reqs * (self.num_spec + 1) <= m.num_actual_tokens
                and ((m.num_reqs + 1) * (self.num_spec + 1)
                     >= m.num_actual_tokens)), \
            "GDN only supports decode-only full CUDAGraph capture. " \
            "Make sure all cudagraph capture sizes <= max_num_seq."

        num_accepted_tokens = torch.full((m.num_reqs, ),
                                         m.max_query_len,
                                         dtype=torch.int32,
                                         device=m.query_start_loc.device)
        num_drafted_tokens = torch.full((m.num_reqs, ),
                                        self.num_spec,
                                        dtype=torch.int32,
                                        device=m.query_start_loc.device)

        # Fixes query-start loc for spec-sequence-indices.
        m.query_start_loc = torch.arange(0,
                                         m.num_actual_tokens + 1,
                                         step=m.max_query_len,
                                         device=m.query_start_loc.device,
                                         dtype=torch.int32)
        m.num_computed_tokens_cpu = (m.seq_lens_cpu - torch.full(
            (m.num_reqs, ), m.max_query_len, dtype=torch.int32, device='cpu'))

        return self.build(0, m, num_accepted_tokens, num_drafted_tokens)
