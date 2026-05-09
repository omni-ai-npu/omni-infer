# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Optional, List, Type, TypeVar, Dict, Tuple
import itertools
import numpy as np
import torch
import torch_npu

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionMetadata,
    MLAAttentionImpl,
    AttentionType
)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import ModelConfig
from vllm.distributed import get_world_group
from vllm.platforms import current_platform
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.utils import is_pin_memory_available
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

from omni.models.config_loader.loader import model_extra_config
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.adaptors.vllm.worker.npu_model_runner import NPUModelRunner
from omni.layers.attention.backend.attention_dummy_builder import DummyAttentionMetadataBuilder
from omni.layers.attention.backend.utils import create_aligend_tensor
from omni.accelerators.cache import OmniAttentionSpec, compute_omni_attn_metadata
from omni.accelerators.cache.omni_cache import BaseOmniCache, PrefixCopyMeta
from omni.adaptors.vllm.utils import get_attr_by_names
from vllm.logger import logger
import os, time, numpy as np, torch

def group_request_list(seq_lens, query_lens, block_tables, threshold):
    s_lens_result = []
    q_lens_result = []
    blocks_result = []
    s_lens_current_group = []
    q_lens_current_group = []
    blocks_current_group = []
    current_sum = 0
    for seq_len, query_len, block_table in zip(seq_lens, query_lens, block_tables):
        if current_sum + seq_len > threshold and len(s_lens_current_group) > 0:
            s_lens_result.append(s_lens_current_group)
            q_lens_result.append(q_lens_current_group)
            blocks_result.append(blocks_current_group)
            s_lens_current_group = []
            q_lens_current_group = []
            blocks_current_group = []
            current_sum = 0
        s_lens_current_group.append(seq_len)
        q_lens_current_group.append(query_len)
        blocks_current_group.append(block_table)
        current_sum += seq_len
    if q_lens_current_group:
        s_lens_result.append(s_lens_current_group)
        q_lens_result.append(q_lens_current_group)
        blocks_result.append(blocks_current_group)
    return s_lens_result, q_lens_result, blocks_result

class AscendMLABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "VLLM_ASCEND_MLA"

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AscendMLAMetadata

    @staticmethod
    def get_builder_cls():
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        head_size = 512 + 64 + 128 + 1 if model_extra_config.operator_opt_config.enable_dsa else 512 + 64
        return (num_blocks, block_size, 1, head_size)

    @staticmethod
    def get_impl_cls() -> Type["MLAAttentionImpl"]:
        return AscendMLAImpl

    @staticmethod
    def init_kv_cache_each_layer(
                            kv_cache_shape,
                            dtype,
                            device,
                            model_config: "ModelConfig",
                            enable_graph_mode,
                            is_prefill: bool = True) -> tuple[torch.Tensor, ...]:
        # KVCache needs to store the shape of the reduced dimension as [num_blocks, block_size, 1, kv_lora_rank] [num_blocks, block_size, 1, rope_dim]
        # The shape of the augmented dimension is [num_blocks, block_size, head_num, head_dim]
        kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
        qk_rope_dim_names = ['attention_qk_rope_dim', 'qk_rope_head_dim']
        kv_lora_dim = get_attr_by_names(model_config.hf_text_config, kv_lora_dim_names, 0)
        qk_rope_dim = get_attr_by_names(model_config.hf_text_config, qk_rope_dim_names, 0)
        layer_kv_cache_nope = create_aligend_tensor(
                        kv_cache_shape[:-2] +
                        (1, kv_lora_dim, ),
                        dtype=dtype if not model_extra_config.operator_opt_config.fa_quant else torch.int8,
                        pin_memory=True,
                        device=device)
        layer_kv_cache_pe = create_aligend_tensor(
                            kv_cache_shape[:-2] +
                            (1, qk_rope_dim, ),
                            dtype=dtype,
                            pin_memory=True,
                            device=device)
        if device != 'cpu':
            # force tensor format to ND
            layer_kv_cache_nope = torch_npu.npu_format_cast(layer_kv_cache_nope, 2)
            layer_kv_cache_pe = torch_npu.npu_format_cast(layer_kv_cache_pe, 2)
        if model_extra_config.operator_opt_config.enable_dsa:
            layer_indexer_k_nope = create_aligend_tensor(
                            kv_cache_shape[:-2] +
                            (1, 128, ),
                            dtype= dtype if not model_extra_config.operator_opt_config.enable_indexer_quant else torch.int8,
                            pin_memory=True,
                            device=device)
            layer_indexer_k_nope_scale = create_aligend_tensor(
                            kv_cache_shape[:-2] +
                            (1, 1, ),
                            dtype=torch.float16,
                            pin_memory=True,
                            device=device)
            # if use omni_cache, only keep k_indexer in decoder side
            if model_extra_config.operator_opt_config.use_omni_cache and not is_prefill:
                return (layer_indexer_k_nope, layer_indexer_k_nope_scale)
            return (layer_kv_cache_nope, layer_kv_cache_pe, layer_indexer_k_nope, layer_indexer_k_nope_scale)
        else:
            return (layer_kv_cache_nope, layer_kv_cache_pe)

    @staticmethod
    def swap_blocks(
            src_kv_cache: List[torch.Tensor],
            dst_kv_cache: List[torch.Tensor],
            src_to_dst: torch.Tensor,
    ) -> None:
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        for src_cache, dst_cache in zip(src_kv_cache, dst_kv_cache):
            if src_cache is None or dst_cache is None:
                continue
            dst_cache[dst_indices] = src_cache[src_indices].to(dst_cache.device)

@dataclass
class AscendMLAPrefillMetadata:
    """ Prefill Specific Metadata for Ascend"""
    attn_mask: torch.Tensor
    query_lens: list[int]
    seq_lens: list[int]
    input_positions: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    slot_mapping: Optional[torch.Tensor] = None
    num_prefill_tokens: int = 0

    # adaptor for chunk-prefill & prefix-caching use
    seq_qlen_group: Optional[list] = None
    seq_kvlen_group: Optional[list] = None
    kv_index_list: Optional[list] = None

    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    cos_q: Optional[torch.Tensor] = None
    sin_q: Optional[torch.Tensor] = None

    sp_split_list: Optional[list[int]] = None
    sp_zigzag_index: Optional[list[int]] = None
    sp_reverse_index: Optional[list[int]] = None
    sp_reverse_split_list: Optional[list[int]] = None
    sp_input_indices: Optional[torch.Tensor] = None
    sp_reverse_indices: Optional[torch.Tensor] = None
    sp_input_actual_tokens: int = 0
    actual_query_lens: Optional[torch.Tensor] = None
    computed_seq_lens: Optional[torch.Tensor] = None

    prefix_meta: Optional[list[PrefixCopyMeta]] = None


@dataclass
class AscendMLADecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: Optional[torch.Tensor] = None
    num_decode_tokens: int = 0
    num_padded_tokens: int = 0
    mc2_mask: Optional[torch.Tensor] = None
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    best_topk: Optional[torch.Tensor] = None

@dataclass
class AscendMLAMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # The dimension of the attention heads
    head_dim: Optional[int] = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: Optional[AscendMLADecodeMetadata] = None
    prefill: Optional[AscendMLAPrefillMetadata] = None

    omni_cache: Optional[BaseOmniCache] = None

    def __post_init__(self):
        pass

    @staticmethod
    def advance_step(metadata, positions, block_size, pad_mask, model_layer):
        block_table = metadata.decode.block_table
        block_indices = block_table.gather(dim=1, index=(positions // block_size).reshape(-1, 1)).view(-1)
        block_offsets = positions % block_size
        metadata.slot_mapping[:] = torch.where(
            pad_mask,
            metadata.slot_mapping,
            block_indices * block_size + block_offsets)
        metadata.decode.seq_lens[:] = (positions + 1).to(metadata.decode.seq_lens.dtype)
        cos, sin = model_layer.self_attn.rotary_emb.get_cos_sin(metadata.decode.input_positions)
        metadata.decode.cos = cos
        metadata.decode.sin = sin

M = TypeVar("M", bound=AscendMLAMetadata)


class AscendMLAMetadataBuilder(DummyAttentionMetadataBuilder):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(self,
                 runner: "NPUModelRunner",
                 kv_cache_spec: AttentionSpec = None,
                 block_table: BlockTable = None,
                 metadata_cls: Optional[AscendMLAMetadata] = None):
        self.metadata_cls: Optional[AscendMLAMetadata] = metadata_cls \
            if metadata_cls is not None else AscendMLAMetadata  # type: ignore
        self.runner = runner
        scheduler_config = runner.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled
        self.block_size = self.runner.block_size
        self.base_index = np.array(list(range(0, self.block_size)))
        self.base_block = self.block_size * np.ones([1, self.block_size])
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table
        self.decode_gear_list = model_extra_config.task_config.decode_gear_list
        if self.decode_gear_list:
            self.mc2_mask = torch.zeros(self.decode_gear_list[-1], dtype=torch.bool, device=current_platform.device_type)
        self.already_mark_static = False
        self._sp_split_indices_cache = {}
        self._sp_input_indices_cache = {}
        self._sp_reverse_indices_cache = {}
        self._sp_split_indices_cache_limit = 256

        self.decode_num_tokens = torch.zeros(
            runner.max_num_reqs, dtype=torch.int32, device=runner.device
        )
        self.decode_num_tokens_cpu = torch.zeros(
            runner.max_num_reqs, dtype=torch.int32, pin_memory=is_pin_memory_available(),
        )
        self.is_hybrid = self.runner.vllm_config.additional_config.get("enable_hybrid_graph_mode", False)

    def generate_activate_mask(self, actual_seqs_num, batch_size):
        if len(self.decode_gear_list) > 1:
            gear = next((g for g in self.decode_gear_list if g >= batch_size), self.decode_gear_list[-1])
            self.mc2_mask = torch.zeros(gear, dtype=torch.bool, device=current_platform.device_type)
        else:
            self.mc2_mask.zero_()
        self.mc2_mask[:actual_seqs_num].fill_(True)

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # full attention layers reorder first, then other layers agree
        if isinstance(self.kv_cache_spec, OmniAttentionSpec):
            return False

        # In the case of prefill nodes in PD disaggregation, there is no need to
        # reorder the batch, since it's IMPOSSIBLE to have decode requests.
        # Besides, when APC is on, if a request has exactly one more token
        # than a previous one whose cache is hit, this request will be considered
        # as 'decode' mistakenly.
        kv_transfer_config = self.runner.vllm_config.kv_transfer_config
        if kv_transfer_config is not None and kv_transfer_config.kv_role in ["kv_producer", "kv_both"]:
            self._num_decodes = 0
            self._num_prefills = len(input_batch.req_ids)
            self._num_decode_tokens = 0
            self._num_prefill_tokens = scheduler_output.total_num_scheduled_tokens
            return False

        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound
        decodes = []
        prefills = []
        decode_num_tokens = []
        num_decode_tokens = 0
        num_prefill_tokens = 0
        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            num_spec_decode_tokens = len(spec_decode_tokens) if spec_decode_tokens is not None else 0
            # Spec decode requests are decode-like only when the scheduler
            # provides exactly one target token plus the speculative tokens.
            # Resumed/chunked prefill may still carry stale spec tokens, but
            # must remain in the prefill side until the chunk is complete.
            is_decode_req = (num_tokens == 1 or
                             (num_spec_decode_tokens > 0
                              and num_tokens == num_spec_decode_tokens + 1))
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the TritonMLA._forward_decode only supports
            # num_tokens = 1
            # Only in decode the spec tokens are scheduled
            if is_decode_req:
                decodes.append(i)
                decode_num_tokens.append(num_tokens)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill],
                                        decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens
        self.decode_num_tokens_cpu[:num_decodes] = torch.tensor(decode_num_tokens)
        self.decode_num_tokens.copy_(self.decode_num_tokens_cpu, non_blocking=True)

        return modified_batch

    def cal_best_topk(self, cur_batch_size):
        world_size = get_world_group().world_size
        batch_size = cur_batch_size * world_size
        top_k = self.runner.model.config.num_experts_per_tok
        step = batch_size // world_size * top_k
        global_rank = get_world_group().rank_in_group
        experts_tp_size = 1
        try:
            num_routed_experts = self.runner.model.config.n_routed_experts
        except:
            num_routed_experts = self.runner.model.config.num_routed_experts
        cur_topk_list = [
            i % num_routed_experts for i in range(
            global_rank // experts_tp_size * step, (global_rank // experts_tp_size + 1) * step)]
        return torch.Tensor(cur_topk_list).to(dtype=torch.int32, device=current_platform.device_type, non_blocking=True).view(batch_size // world_size, -1)

    def _get_graph_runner_block_tables(
            self, num_decode_tokens: int, block_tables: torch.Tensor, total_tokens:int) -> torch.Tensor:

        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        if max_batch_size < num_decode_tokens:
            is_mixed_batch = self._num_decodes > 0 and self._num_prefills > 0
            runs_compiled_model = (
                hasattr(self.runner, "_should_run_compiled_model")
                and self.runner._should_run_compiled_model(self.runner.attn_state))
            if (self.runner.attn_state == AscendAttentionState.DecodeOnly
                    or runs_compiled_model
                    or not is_mixed_batch):
                raise RuntimeError("max_batch_size must be greater than or equal to num_decode_tokens")
            graph_block_tables = torch.zeros((total_tokens, max_blocks),
                                             dtype=block_tables.dtype,
                                             device=block_tables.device)
        elif isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((total_tokens, max_blocks),
                                             dtype=block_tables.dtype,
                                             device=block_tables.device)
        else:
            graph_block_tables = self.runner.graph_block_tables.to(
                device=block_tables.device, dtype=block_tables.dtype, non_blocking=True)

        num_blocks = block_tables.size(1)
        if num_blocks <= max_blocks:
            graph_block_tables[:num_decode_tokens, :
                               num_blocks] = block_tables[:num_decode_tokens, :
                                                          num_blocks]
        else:
            graph_block_tables[:num_decode_tokens, :
                               max_blocks] = block_tables[:num_decode_tokens, :
                                                          max_blocks]

        return graph_block_tables

    def _get_decode_padded_tokens(self, num_decode_tokens: int, graph_pad_size: int) -> int:
        if self.runner.attn_state == AscendAttentionState.DecodeOnly:
            return num_decode_tokens + graph_pad_size

        padded_tokens = num_decode_tokens
        sp_size = model_extra_config.parall_config.attn_sp_size
        if self.decode_gear_list:
            padded_tokens = next((
                gear for gear in self.decode_gear_list
                if gear >= num_decode_tokens and (sp_size <= 1 or gear % sp_size == 0)
            ), padded_tokens)

        if sp_size > 1 and padded_tokens % sp_size != 0:
            padded_tokens = ((padded_tokens + sp_size - 1) // sp_size) * sp_size

        return padded_tokens

    def get_kv_index(self, seq_lens, block_tables):
        kv_index = []
        for seq_len, block_table in zip(seq_lens, block_tables):
            if torch.is_tensor(block_table):
                if block_table.device.type != "cpu":
                    raise RuntimeError(
                        "get_kv_index expects CPU/numpy block tables to avoid NPU->CPU sync.")
                block_table = block_table.numpy()
            else:
                block_table = np.asarray(block_table)
            index = self.base_index + np.expand_dims(block_table, axis=-1) * self.base_block.repeat(block_table.shape[0], axis=0)
            kv_index.append(index.reshape(-1)[:seq_len])
        return torch.tensor(np.concatenate(kv_index, axis=0), dtype=torch.long, device="cpu").npu()

    def prepare_sp_split_indices(self, query_lens_list):
        sp_size = get_tensor_model_parallel_world_size()
        sp_rank = get_tensor_model_parallel_rank()
        cache_key = (
            tuple(int(query_len) for query_len in query_lens_list),
            model_extra_config.parall_config.attn_sp_size,
            sp_rank,
        )
        cached_indices = self._sp_split_indices_cache.get(cache_key)
        if cached_indices is not None:
            return cached_indices

        query_lens = torch.tensor(query_lens_list)
        bsz = query_lens.shape[-1]

        # get zigzag index
        cp_piece_num = sp_size * 2
        seq_per_batch = torch.ceil(query_lens / (cp_piece_num))   # seq_len for each batch and piece
        split_list = seq_per_batch.repeat_interleave(cp_piece_num).int().tolist()
        zigzag_index = []
        for batch_id in range(bsz):
            zigzag_index.extend([batch_id * cp_piece_num + sp_rank,
                (batch_id + 1) * cp_piece_num - sp_rank - 1])
        zigzag_index = zigzag_index[::2] + zigzag_index[1::2]
        # get zigzag reverse index
        cp_reverse_index = []
        for batch_id in range(bsz):
            cp_reverse_index.extend(
                list(range(batch_id, cp_piece_num * bsz, 2 * bsz)) +\
                list(range((cp_piece_num - 1) * bsz + batch_id, 0, -2 * bsz))
            )
        reverse_split_list = seq_per_batch.repeat_interleave(2).repeat(sp_size).view(-1).int().tolist()
        reverse_split_list = reverse_split_list[::2] + reverse_split_list[1::2]
        indices = (split_list, zigzag_index, cp_reverse_index, reverse_split_list)
        if len(self._sp_split_indices_cache) >= self._sp_split_indices_cache_limit:
            self._sp_split_indices_cache.clear()
        self._sp_split_indices_cache[cache_key] = indices
        return indices

    def pad_inputs(self, input, query_lens, sp_size, pad_value):
        count = 0
        res = []
        for len in query_lens:
            pad_size = (sp_size - len % sp_size) % sp_size
            tmp_tensor = input[count:count + len]
            padded_tensor = self.runner.model.model.pad_tensor(tmp_tensor, pad_size, pad_value)
            res.append(padded_tensor)
            count += len
        return torch.cat(res, dim=0)

    def build_sp_input_indices(self, query_lens_list, sp_split_list, sp_zigzag_index, device):
        cache_key = (
            tuple(int(query_len) for query_len in query_lens_list),
            model_extra_config.parall_config.attn_sp_size,
            get_tensor_model_parallel_rank(),
        )
        indices = self._sp_input_indices_cache.get(cache_key)
        if indices is None:
            pad_idx = sum(query_lens_list)
            padded_indices = []
            offset = 0
            split_size = model_extra_config.parall_config.attn_sp_size * 2
            for query_len in query_lens_list:
                pad_size = (split_size - query_len % split_size) % split_size
                padded_indices.extend(range(offset, offset + query_len))
                padded_indices.extend([pad_idx] * pad_size)
                offset += query_len

            split_indices = []
            offset = 0
            for split_len in sp_split_list:
                split_indices.append(padded_indices[offset:offset + split_len])
                offset += split_len

            sp_input_indices = []
            for split_idx in sp_zigzag_index:
                sp_input_indices.extend(split_indices[split_idx])
            indices = (padded_indices, sp_input_indices)
            if len(self._sp_input_indices_cache) >= self._sp_split_indices_cache_limit:
                self._sp_input_indices_cache.clear()
            self._sp_input_indices_cache[cache_key] = indices
        padded_indices, sp_input_indices = indices
        return (
            torch.tensor(padded_indices, dtype=torch.long, device=device),
            torch.tensor(sp_input_indices, dtype=torch.long, device=device),
        )

    def build_sp_reverse_indices(self, query_lens_list, sp_reverse_split_list, sp_reverse_index, device):
        cache_key = (
            tuple(int(query_len) for query_len in query_lens_list),
            model_extra_config.parall_config.attn_sp_size,
            get_tensor_model_parallel_rank(),
        )
        sp_reverse_indices = self._sp_reverse_indices_cache.get(cache_key)
        if sp_reverse_indices is None:
            split_offsets = []
            offset = 0
            for split_len in sp_reverse_split_list:
                split_offsets.append(offset)
                offset += split_len
            sp_reverse_indices = []
            for split_idx in sp_reverse_index:
                start = split_offsets[split_idx]
                split_len = sp_reverse_split_list[split_idx]
                sp_reverse_indices.extend(range(start, start + split_len))
            if len(self._sp_reverse_indices_cache) >= self._sp_split_indices_cache_limit:
                self._sp_reverse_indices_cache.clear()
            self._sp_reverse_indices_cache[cache_key] = sp_reverse_indices
        return torch.tensor(sp_reverse_indices, dtype=torch.long, device=device)

    def pad_inputs_by_index(self, input, indices, pad_value, actual_tokens=None):
        if actual_tokens is not None:
            input = input[:actual_tokens]
        pad_shape = (1, ) + tuple(input.shape[1:])
        padding = input.new_full(pad_shape, pad_value)
        return torch.cat([input, padding], dim=0).index_select(0, indices)

    def prepare_sp_inputs(self,
                          positions,
                          query_lens_list,
                          seq_lens_list,
                          slot_mapping,
                          query_lens):
        computed_seq_lens_list = [seq_len - query_len for seq_len, query_len in zip(seq_lens_list, query_lens_list)]
        sp_seq_lens_list = [computed_seq_len + math.ceil(query_len / model_extra_config.parall_config.attn_sp_size / 2) for computed_seq_len, query_len in zip(computed_seq_lens_list, query_lens_list)]
        sp_split_list, sp_zigzag_index, sp_reverse_index, sp_reverse_split_list = self.prepare_sp_split_indices(query_lens_list)

        # prepare sp positions
        sp_size = model_extra_config.parall_config.attn_sp_size
        sp_padded_input_indices, sp_input_indices = self.build_sp_input_indices(
            query_lens_list, sp_split_list, sp_zigzag_index, positions.device)
        sp_reverse_indices = self.build_sp_reverse_indices(
            query_lens_list, sp_reverse_split_list, sp_reverse_index, positions.device)
        actual_tokens = sum(query_lens_list)
        padded_positions = self.pad_inputs_by_index(
            positions, sp_padded_input_indices, 0, actual_tokens)
        cos, sin = self.runner.model.model.layers[0].self_attn.rotary_emb.get_cos_sin(padded_positions)
        positions = self.pad_inputs_by_index(positions, sp_input_indices, 0, actual_tokens)

        sp_seq_lens = torch.tensor(sp_seq_lens_list, dtype=torch.int64).npu()
        computed_seq_lens = torch.tensor(computed_seq_lens_list, dtype=torch.int64).npu()
        query_lens = torch.cumsum(torch.ceil(query_lens / sp_size / 2).to(torch.int64), dim=0)
        # prepare sp slotmapping
        slot_mapping = self.pad_inputs_by_index(
            slot_mapping, sp_padded_input_indices, PAD_SLOT_ID, actual_tokens)

        return sp_split_list, sp_zigzag_index, sp_reverse_index, sp_reverse_split_list, sp_input_indices, sp_reverse_indices, positions, sp_seq_lens, computed_seq_lens, cos, sin, slot_mapping, query_lens

    def build(self,
              num_reqs: int,
              num_actual_tokens: int,
              max_query_len: int,
              common_prefix_len: Optional[int] = None,
              graph_pad_size: int = -1) -> AscendMLAMetadata:
        if isinstance(self.kv_cache_spec, OmniAttentionSpec):
            return self.build_omni_attn_metadata(num_reqs, num_actual_tokens, graph_pad_size)

        if self._num_decodes + self._num_prefills != num_reqs:
            raise RuntimeError("self._num_decodes + self._num_prefills must be equal to num_reqs")

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.runner.device
        block_table = self.block_table.get_device_tensor()[:num_reqs]

        slot_mapping = self.block_table.slot_mapping_cpu[:num_actual_tokens].to(
            device, non_blocking=True)
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            device, non_blocking=True)
        actual_slot_mapping = slot_mapping
        actual_input_positions = input_positions
        is_mixed_batch = self._num_decodes > 0 and self._num_prefills > 0

        if self.runner.omni_cache is not None:
            assert isinstance(self.runner.omni_cache, BaseOmniCache), \
                f"Omni cache type is {type(self.runner.omni_cache)}"
        omni_cache = self.runner.omni_cache

        # pad prefill to avoid error of operator's shape assert
        if graph_pad_size > 0:
            padding = torch.full((graph_pad_size, ),
                                    PAD_SLOT_ID,
                                    dtype=slot_mapping.dtype,
                                    device=device)
            padding_0 = torch.zeros(graph_pad_size,
                                    dtype=input_positions.dtype,
                                    device=device)
            slot_mapping = torch.cat([slot_mapping, padding])
            input_positions = torch.cat([input_positions, padding_0])

        prefill_metadata = None
        if self._num_prefills > 0:
            seq_lens_list = self.runner.seq_lens_cpu[:num_reqs].tolist()
            query_lens_list = (self.runner.seq_lens_np[:num_reqs] - self.runner.input_batch.num_computed_tokens_cpu[:num_reqs]).tolist()

            reqs_start = self._num_decodes  # prefill_start
            tokens_start = self._num_decode_tokens
            prefill_seq_lens_list = seq_lens_list[reqs_start:]
            prefill_query_lens_list = query_lens_list[reqs_start:]
            prefill_block_table = block_table[reqs_start:, ...]
            if is_mixed_batch:
                prefill_slot_mapping = actual_slot_mapping[tokens_start:num_actual_tokens]
                tmp_input_position = actual_input_positions[tokens_start:num_actual_tokens]
            else:
                prefill_slot_mapping = slot_mapping[tokens_start:]
                tmp_input_position = input_positions[tokens_start:]
            prefix_meta = None

            if not model_extra_config.operator_opt_config.use_omni_cache:
                prefill_block_table_np = self.block_table.get_numpy_array()[reqs_start:num_reqs]
                # Group request for Chunk-Prefill
                seq_kvlen_group, seq_qlen_group, block_groups = group_request_list(
                    prefill_seq_lens_list,
                    prefill_query_lens_list,
                    prefill_block_table_np,
                    self.runner.max_num_tokens)

                # Prepare kv index for prefill get kv_latent from kv_cache
                if self.runner.attn_state == AscendAttentionState.ChunkedPrefill:
                    kv_index_list = []
                    if block_table is not None and block_table.numel() > 0:
                        for seq_lens, block_tables in zip(seq_kvlen_group, block_groups):
                            kv_index = self.get_kv_index(seq_lens, block_tables)
                            kv_index_list.append(kv_index)
                else:
                    kv_index_list = None

                seq_qlen_group = [list(itertools.accumulate(sub_list)) for sub_list in seq_qlen_group]
                seq_kvlen_group = [list(itertools.accumulate(sub_list)) for sub_list in seq_kvlen_group]
            else:
                seq_qlen_group = [list(itertools.accumulate(prefill_query_lens_list))]
                seq_kvlen_group = [list(itertools.accumulate(prefill_seq_lens_list))]
                kv_index_list = None

                prefix_meta = omni_cache.get_prefill_prefix_copy_meta(
                    block_size=self.block_size,
                    kv_lens=self.runner.input_batch.num_computed_tokens_cpu[reqs_start:num_reqs],
                    query_lens_list=prefill_query_lens_list,
                    block_tables=self.block_table.get_numpy_array()[reqs_start:num_reqs],
                    attn_state=self.runner.attn_state,
                )

                omni_cache.synchronize_h2d(
                    prefix_meta=prefix_meta,
                    layer_idx=0,
                )

                volatile_block_table, volatile_slot_mapping = omni_cache.get_volatile_metadata(
                    prefill_query_lens_list,
                    prefill_seq_lens_list,
                    graph_pad_size,
                    PAD_SLOT_ID,
                    prefill_slot_mapping,
                )
                if volatile_block_table is not None:
                    prefill_block_table = volatile_block_table[:self._num_prefills]
                    prefill_slot_mapping = volatile_slot_mapping

            first_layer_ind = self.runner.model.model.start_layer
            if not model_extra_config.operator_opt_config.enable_dsa:
                query_lens = prefill_query_lens_list
                seq_lens = prefill_seq_lens_list
            else:
                actual_query_lens = torch.tensor(prefill_query_lens_list, dtype=torch.int64).npu()
                if model_extra_config.parall_config.attn_sp_size == 1:
                    query_lens = torch.cumsum(actual_query_lens, dim=0)
                seq_lens = torch.tensor(prefill_seq_lens_list, dtype=torch.int64).npu()
            if model_extra_config.parall_config.attn_sp_size > 1:
                sp_split_list, sp_zigzag_index, sp_reverse_index, sp_reverse_split_list, sp_input_indices, sp_reverse_indices, tmp_input_position, seq_lens, computed_seq_lens, cos, sin, prefill_slot_mapping, query_lens  = \
                    self.prepare_sp_inputs(positions=tmp_input_position,
                                        query_lens_list=prefill_query_lens_list,
                                        seq_lens_list=prefill_seq_lens_list,
                                        slot_mapping=prefill_slot_mapping,
                                        query_lens=actual_query_lens,
                                        )
                # 在sp场景下，只有切分后长度的位置信息
                cos_q, sin_q = self.runner.model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(tmp_input_position)
            else:
                if isinstance(self.runner.model.model.layers[first_layer_ind].self_attn, torch.nn.ModuleList):
                    cos, sin = self.runner.model.model.layers[first_layer_ind].self_attn[0].rotary_emb.get_cos_sin(tmp_input_position)
                else:
                    cos, sin = self.runner.model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(tmp_input_position)

            prefill_metadata = AscendMLAPrefillMetadata(
                attn_mask=self.runner.attn_mask,
                query_lens=query_lens,
                seq_lens=seq_lens,
                input_positions=tmp_input_position,
                block_table=prefill_block_table,
                max_query_len=max_query_len,
                slot_mapping=prefill_slot_mapping,
                num_prefill_tokens=prefill_slot_mapping.shape[0],
                seq_qlen_group=seq_qlen_group,
                seq_kvlen_group=seq_kvlen_group,
                kv_index_list=kv_index_list,
                sin=sin,
                cos=cos,
                sin_q=sin_q if model_extra_config.parall_config.attn_sp_size > 1 else None,
                cos_q=cos_q if model_extra_config.parall_config.attn_sp_size > 1 else None,
                sp_split_list=sp_split_list if model_extra_config.parall_config.attn_sp_size > 1 else None,
                sp_zigzag_index=sp_zigzag_index if model_extra_config.parall_config.attn_sp_size > 1 else None,
                sp_reverse_index=sp_reverse_index if model_extra_config.parall_config.attn_sp_size > 1 else None,
                sp_reverse_split_list=sp_reverse_split_list if model_extra_config.parall_config.attn_sp_size > 1 else None,
                sp_input_indices=sp_input_indices if model_extra_config.parall_config.attn_sp_size > 1 else None,
                sp_reverse_indices=sp_reverse_indices if model_extra_config.parall_config.attn_sp_size > 1 else None,
                sp_input_actual_tokens=sum(prefill_query_lens_list) if model_extra_config.parall_config.attn_sp_size > 1 else 0,
                actual_query_lens=actual_query_lens if model_extra_config.parall_config.attn_sp_size > 1 else None,
                computed_seq_lens=computed_seq_lens if model_extra_config.parall_config.attn_sp_size > 1 else None,
                prefix_meta=prefix_meta if model_extra_config.operator_opt_config.use_omni_cache else None,
            )

        decode_metadata = None

        if self._num_decodes > 0:
            if self.runner.attn_state == AscendAttentionState.DecodeOnly or is_mixed_batch:
                uniform_decode_tokens = bool(torch.all(
                    self.decode_num_tokens_cpu[:self._num_decodes] == self.decode_num_tokens_cpu[0]))
                num_tokens_per_req = int(self.decode_num_tokens_cpu[0].item()) if uniform_decode_tokens else None

                if model_extra_config.operator_opt_config.enable_dsa and model_extra_config.operator_opt_config.use_omni_cache:
                    time0 = time.perf_counter()
                    req_ids_update_mapping = self.runner.input_batch.req_id_to_index
                    req_ids_update = [req_id for req_id, _ in sorted(req_ids_update_mapping.items(), key=lambda item: item[1])]
                    req_ids_record = getattr(omni_cache, 'req_ids_record', None)
                    if req_ids_record is None:
                        req_ids_record = req_ids_update.copy()

                    if req_ids_record != req_ids_update and num_tokens_per_req is not None:
                        # update selection_kv_block_status
                        original_shape = omni_cache.selection_kv_block_status.shape
                        num_layers = omni_cache.selection_kv_block_status.shape[0]
                        head_num = omni_cache.selection_kv_block_status.shape[-2]
                        topk_plus_1 = omni_cache.selection_kv_block_status.shape[-1]

                        reshaped_view = omni_cache.selection_kv_block_status.view(
                            num_layers, -1, num_tokens_per_req, head_num, topk_plus_1
                        )
                        self._update_status_buffered(
                            omni_cache, reshaped_view, req_ids_record, req_ids_update, fill_value=-1
                        )

                        # update selection_kv_block_table
                        s_max_block_num = omni_cache.selection_kv_block_table.shape[-1]
                        reshaped_table_view = omni_cache.selection_kv_block_table.view(
                            -1, num_tokens_per_req, s_max_block_num
                        )
                        self._reorder_block_table_only(
                            omni_cache, reshaped_table_view, req_ids_record, req_ids_update
                        )

                    omni_cache.req_ids_record = req_ids_update.copy()
                    time1 = time.perf_counter()
                    if req_ids_record != req_ids_update:
                        logger.warning(f"======== Time cost for one-step block_status update is {(time1-time0)*1000}ms ========")
                    else:
                        logger.warning(f"++++++++ Time cost for one-step block_status update is {(time1-time0)*1000}ms ++++++++")

                decode_padded_tokens = self._get_decode_padded_tokens(
                    self._num_decode_tokens, graph_pad_size)
                decode_pad_size = decode_padded_tokens - self._num_decode_tokens
                decode_input_positions = actual_input_positions[:self._num_decode_tokens]
                decode_slot_mapping = actual_slot_mapping[:self._num_decode_tokens]
                if decode_pad_size > 0:
                    decode_input_positions = torch.cat([
                        decode_input_positions,
                        torch.zeros(decode_pad_size,
                                    dtype=decode_input_positions.dtype,
                                    device=decode_input_positions.device)
                    ])
                    decode_slot_mapping = torch.cat([
                        decode_slot_mapping,
                        torch.full((decode_pad_size, ),
                                   PAD_SLOT_ID,
                                   dtype=decode_slot_mapping.dtype,
                                   device=decode_slot_mapping.device)
                    ])

                seq_lens = (decode_input_positions + 1).to(self.runner.seq_lens.dtype)
                decode_block_table = block_table[:self._num_decodes, ...]
                # has speculative tokens
                if self._num_decode_tokens > self._num_decodes:
                    decode_block_table = decode_block_table.repeat_interleave(
                        self.decode_num_tokens[:self._num_decodes], dim=0, output_size=self._num_decode_tokens
                    )
                if decode_pad_size > 0:
                    decode_block_table = torch.cat([
                        decode_block_table,
                        torch.zeros((decode_pad_size, ) + decode_block_table.shape[1:],
                                    dtype=decode_block_table.dtype,
                                    device=decode_block_table.device)
                    ], dim=0)
                decode_block_table = self._get_graph_runner_block_tables(
                    self._num_decode_tokens, decode_block_table, decode_padded_tokens)

                self.generate_activate_mask(self._num_decode_tokens, decode_padded_tokens)
                first_layer_ind = self.runner.model.model.start_layer
                if isinstance(self.runner.model.model.layers[first_layer_ind].self_attn, torch.nn.ModuleList):
                    cos, sin = self.runner.model.model.layers[first_layer_ind].self_attn[0].rotary_emb.get_cos_sin(decode_input_positions)
                else:
                    cos, sin = self.runner.model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(decode_input_positions)
                best_topk = None
                if model_extra_config.operator_opt_config.best_ep:
                    best_topk = self.cal_best_topk(decode_padded_tokens)
            else:
                # Supported decode metadata paths are pure DecodeOnly and
                # mixed ChunkedPrefill with both decode and prefill requests.
                # A non-mixed ChunkedPrefill state that still contains decode
                # tokens does not have a verified MLA decode metadata contract.
                raise NotImplementedError(
                    "Non-mixed ChunkedPrefill with decode tokens is not supported currently.")

            decode_metadata = AscendMLADecodeMetadata(
                input_positions=decode_input_positions,
                block_table=decode_block_table,
                seq_lens=seq_lens,
                slot_mapping=decode_slot_mapping,
                num_decode_tokens=self._num_decode_tokens,
                num_padded_tokens=decode_padded_tokens,
                mc2_mask=self.mc2_mask,
                cos=cos,
                sin=sin,
                best_topk=best_topk)

        metadata_slot_mapping = slot_mapping
        if is_mixed_batch:
            metadata_slot_mapping = torch.cat(
                [decode_metadata.slot_mapping, prefill_metadata.slot_mapping],
                dim=0)
        elif decode_metadata is not None:
            metadata_slot_mapping = decode_metadata.slot_mapping
        elif prefill_metadata is not None:
            metadata_slot_mapping = prefill_metadata.slot_mapping

        return self.metadata_cls(  # type: ignore
            num_actual_tokens=num_actual_tokens,
            slot_mapping=metadata_slot_mapping,
            num_decodes=self._num_decodes,
            num_decode_tokens=self._num_decode_tokens,
            num_prefills=self._num_prefills,
            attn_mask=self.runner.attn_mask,
            attn_state=self.runner.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            omni_cache=omni_cache,
        )

    def _update_status_buffered(
        self,
        omni_cache,
        reshaped_status: torch.Tensor,
        old_req_ids: List[Optional[str]],
        new_req_ids: List[Optional[str]],
        *,
        fill_value: Any = -1,
    ) -> None:
        if old_req_ids is None:
            old_req_ids = []
        if not new_req_ids:
            return

        L, B, T, H, K = reshaped_status.shape
        work_buffer = omni_cache.selection_kv_block_status_buffer
        if work_buffer.shape != reshaped_status.shape:
            work_buffer = torch.empty_like(reshaped_status)

        work_buffer.fill_(fill_value)

        M = min(len(old_req_ids), B)
        N = min(len(new_req_ids), B)

        old_id2idx = {rid: i for i, rid in enumerate(old_req_ids[:M]) if rid is not None}

        mapped_src = []
        mapped_tgt = []
        for tgt in range(N):
            rid = new_req_ids[tgt]
            src = old_id2idx.get(rid, None)
            if src is not None and 0 <= src < M:
                mapped_src.append(src)
                mapped_tgt.append(tgt)

        if mapped_src:
            device = reshaped_status.device
            src_idx = torch.tensor(mapped_src, dtype=torch.long, device=device)
            tgt_idx = torch.tensor(mapped_tgt, dtype=torch.long, device=device)
            selected = reshaped_status.index_select(1, src_idx)
            work_buffer.index_copy_(1, tgt_idx, selected)

        reshaped_status.copy_(work_buffer)

    def _build_table_perm(self,
        old_req_ids: List[str],
        new_req_ids: List[str],
        B: int) -> Tuple[List[int], int]:
        M = min(len(old_req_ids), B)
        old_eff = old_req_ids[:M]
        old_id2idx = {rid: i for i, rid in enumerate(old_eff)}
        kept = [old_id2idx[r] for r in new_req_ids if r in old_id2idx]
        remaining = [i for i, r in enumerate(old_eff) if r not in set(new_req_ids)]
        perm_eff = kept + remaining
        assert len(perm_eff) == M
        assert sorted(perm_eff) == list(range(M))
        return perm_eff, M

    def _reorder_block_table_only(
        self,
        omni_cache,
        reshaped_table: torch.Tensor,
        old_req_ids: List[str],
        new_req_ids: List[str]
    ) -> None:
        B, T, S = reshaped_table.shape
        if not old_req_ids or B == 0:
            return

        perm_eff, M = self._build_table_perm(old_req_ids, new_req_ids, B)

        work_buffer = omni_cache.selection_kv_block_table_buffer
        if work_buffer.shape != reshaped_table.shape:
            work_buffer = torch.empty_like(reshaped_table)

        device = reshaped_table.device

        work_buffer.copy_(reshaped_table)

        if M > 0:
            perm_tensor = torch.tensor(perm_eff, dtype=torch.long, device=device)
            new_order_first = reshaped_table[:M].index_select(0, perm_tensor)
            work_buffer[:M].copy_(new_order_first)

        reshaped_table.copy_(work_buffer)

    def build_omni_attn_metadata(
        self,
        num_reqs: int,
        num_actual_tokens: int,
        graph_pad_size: int,
    ) -> AscendMLAMetadata:
        decode_metadata = None
        ref: AscendMLAMetadata = self.runner.full_attn_metadata
        num_decodes, num_decode_tokens, num_prefills, prefill_metadata, ref_d = \
            ref.num_decodes, ref.num_decode_tokens, ref.num_prefills, ref.prefill, ref.decode

        if ref_d is not None:
            omni_block_table, omni_slot_mapping, omni_seq_lens = compute_omni_attn_metadata(
                self.kv_cache_spec,
                self.block_table,
                num_decodes,
                num_decode_tokens,
                num_prefills,
                self.runner.input_batch.num_prompt_tokens[:num_reqs],
                self.runner.seq_lens_np[:num_reqs],
                self.runner.device,
                ref_d.block_table.shape,
            )

            if omni_seq_lens is None:
                omni_seq_lens = torch.clamp(ref_d.seq_lens, min=0, max=self.kv_cache_spec.max_compressed_len)

            input_positions, cos, sin, best_topk = \
                ref_d.input_positions, ref_d.cos, ref_d.sin, ref_d.best_topk

            self.generate_activate_mask(num_actual_tokens, num_actual_tokens + graph_pad_size)
            decode_metadata = AscendMLADecodeMetadata(
                input_positions=input_positions,
                block_table=omni_block_table,
                seq_lens=omni_seq_lens,
                mc2_mask=self.mc2_mask,
                cos=cos.clone(),
                sin=sin.clone(),
                best_topk=best_topk)
        else:
            raise RuntimeError(f"Full attention metadata should not be None!")

        return self.metadata_cls(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=omni_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            attn_mask=self.runner.attn_mask,
            attn_state=self.runner.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

    def build_dummy(self, num_tokens: int, max_pad_size:int = -1) -> AscendMLAMetadata:
        if max_pad_size == -1:
            max_pad_size = self.runner.max_batch_size
        input_positions = torch.zeros(max_pad_size,
                                  dtype=self.runner.positions_cpu.dtype,
                                  device=self.runner.device)
        slot_mapping = torch.zeros(max_pad_size,
                                dtype=self.runner.slot_mapping_cpu.dtype,
                                device=self.runner.device)
        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_pad_size, self.runner.graph_block_tables.shape[1]))
        block_table = graph_block_tables.to(
            device=self.runner.device,
            dtype=self.runner.input_batch.block_table[0].get_device_tensor().dtype
        )

        seq_lens = torch.ones(max_pad_size, dtype=torch.long, device=self.runner.device, pin_memory=True) * 2
        first_layer_ind = self.runner.model.model.start_layer
        if isinstance(self.runner.model.model.layers[first_layer_ind].self_attn, torch.nn.ModuleList):
            cos, sin = self.runner.model.model.layers[first_layer_ind].self_attn[0].rotary_emb.get_cos_sin(input_positions)
        else:
            cos, sin = self.runner.model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(input_positions)
        best_topk = None
        self.generate_activate_mask(0, max_pad_size)
        if model_extra_config.operator_opt_config.best_ep:
            best_topk = self.cal_best_topk(max_pad_size)
        decode_metadata = AscendMLADecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                mc2_mask=self.mc2_mask,
                cos=cos,
                sin=sin,
                best_topk=best_topk)
        return self.metadata_cls(  # type: ignore
            num_actual_tokens=num_tokens,
            slot_mapping=slot_mapping,
            num_decodes=num_tokens,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            attn_mask=self.runner.attn_mask,
            attn_state=self.runner.attn_state,
            prefill=None,
            decode=decode_metadata,
            omni_cache=self.runner.omni_cache
        )

    def mark_static_for_attn_metadata(self, attn_metadata):
        if self.already_mark_static:
            return
        if attn_metadata.decode.cos is not None:
            torch._dynamo.mark_static(attn_metadata.decode.cos)
        if attn_metadata.decode.sin is not None:
            torch._dynamo.mark_static(attn_metadata.decode.sin)
        if attn_metadata.decode.mc2_mask is not None:
            torch._dynamo.mark_static(attn_metadata.decode.mc2_mask)
        if attn_metadata.decode.best_topk is not None:
            torch._dynamo.mark_static(attn_metadata.decode.best_topk)
        if attn_metadata.decode.block_table is not None:
            torch._dynamo.mark_static(attn_metadata.decode.block_table)
        # if attn_metadata.decode.seq_lens is not None:
        #     torch._dynamo.mark_static(attn_metadata.decode.seq_lens)
        if attn_metadata.slot_mapping is not None:
            torch._dynamo.mark_static(attn_metadata.slot_mapping)
        self.already_mark_static = True


class AscendMLAImpl(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.blocksparse_params = blocksparse_params
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type

    def forward(
        self,
        *args,
        **kwargs
    ) -> torch.Tensor:
        # This method should be implemented in the subclass
        raise NotImplementedError("AscendMLAImpl.forward is not implemented.")
