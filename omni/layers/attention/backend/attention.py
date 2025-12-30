# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import os
import numpy as np
import torch
import torch_npu
import torchair as tng
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type
import torch.nn.functional as F
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import PAD_SLOT_ID, CommonAttentionState
from vllm.model_executor.layers.rotary_embedding import DynamicNTKScalingRotaryEmbedding
from vllm.model_executor.models.utils import extract_layer_index
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import (direct_register_custom_op, supports_dynamo, is_pin_memory_available)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable
from vllm.platforms import current_platform
from vllm.config import (get_current_vllm_config, CompilationLevel)
from omni.layers.rotary_embedding import QwenMRotaryEmbedding, MRotaryEmbeddingInterleaved
from omni.layers.attention.backend.attention_dummy_builder import DummyAttentionMetadataBuilder
from omni.models.config_loader.loader import model_extra_config

# 延迟导入避免循环依赖
BaseOmniCache = None
PrefixCopyMeta = None
def _load_cache_classes():
    global BaseOmniCache, PrefixCopyMeta
    if BaseOmniCache is None:
        from omni.accelerators.cache.omni_cache import BaseOmniCache as _BC, PrefixCopyMeta as _PCM
        BaseOmniCache = _BC
        PrefixCopyMeta = _PCM


def get_nz_dim():
    enable_c8 = getattr(model_extra_config.operator_opt_config, "enable_c8", False)
    return 32 if enable_c8 else 16

class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3


def unified_ascend_attention_with_output(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
        sink_query: Optional[torch.Tensor] = None,
        sink_key: Optional[torch.Tensor] = None,
        sink_value: Optional[torch.Tensor] = None,
        v_head_size: Optional[int] = None,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]

    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      output,
                      trace_flag=False,
                      sink_query=sink_query,
                      sink_key=sink_key,
                      sink_value=sink_value,
                      v_head_size=v_head_size)
    return


def unified_attention_with_output_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)


@dataclass
class AscendMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    block_tables: torch.Tensor
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    query_lens: torch.Tensor
    query_lens_list: List
    seq_lens: torch.Tensor
    seq_lens_list: List
    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor = None
    slot_indices: torch.Tensor = None
    is_only_prefill: bool = False
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    is_pd_seperate_d: bool = False
    kv_index: Optional[torch.Tensor] = None
    mc2_mask: Optional[torch.Tensor] = None
    prefix_meta: Optional[PrefixCopyMeta] = None
    omni_cache: Optional[BaseOmniCache] = None
    additional_metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def advance_step(metadata, positions, block_size, pad_mask, model_layer):
        block_table = metadata.block_tables
        block_indices = block_table.gather(dim=1, index=(positions // block_size).reshape(-1, 1)).view(-1)
        block_offsets = positions % block_size
        metadata.slot_mapping[:] = torch.where(
            pad_mask,
            metadata.slot_mapping,
            block_indices * block_size + block_offsets)
        metadata.seq_lens[:] = (positions + 1).to(metadata.seq_lens.dtype)

class AscendAttentionMetadataBuilder(DummyAttentionMetadataBuilder):

    def __init__(self, runner, kv_cache_spec: AttentionSpec = None,
                 block_table: BlockTable = None):
        self.runner = runner
        self.dtype = runner.dtype
        self.device = runner.device
        self.kv_cache_spec = kv_cache_spec
        self.block_size = runner.block_size
        self.block_table = block_table
        self.decode_gear_list = model_extra_config.task_config.decode_gear_list
        self.mc2_mask = None
        if self.decode_gear_list:
            self.mc2_mask = torch.zeros(self.decode_gear_list[-1], dtype=torch.bool, device=current_platform.device_type)

        self.decode_num_tokens = torch.zeros(
            runner.max_num_reqs, dtype=torch.int32, device=runner.device
        )
        self.decode_num_tokens_cpu = torch.zeros(
            runner.max_num_reqs, dtype=torch.int32, pin_memory=is_pin_memory_available(),
        )

        current_layers_names = []
        for i, kv_cache_group in enumerate(self.runner.kv_cache_config.kv_cache_groups):
            if kv_cache_spec == kv_cache_group.kv_cache_spec:
                current_layers_names = kv_cache_group.layer_names
        self.is_spec_metadata = hasattr(self.runner, "drafter") \
            and sorted(self.runner.drafter.attn_layer_names) == sorted(current_layers_names)
        self.omni_cache = getattr(self.runner, "omni_cache", None)

    def generate_activate_mask(self, actual_seqs_num, batch_size):
        if len(self.decode_gear_list) > 1:
            gear = next((g for g in self.decode_gear_list if g >= batch_size), self.decode_gear_list[-1])
            self.mc2_mask = torch.zeros(gear, dtype=torch.bool, device=current_platform.device_type)
        else:
            self.mc2_mask.zero_()
        self.mc2_mask[:actual_seqs_num].fill_(True)

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
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
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the TritonMLA._forward_decode only supports
            # num_tokens = 1
            # Only in decode the spec tokens are scheduled
            if req_id in scheduler_output.scheduled_spec_decode_tokens or num_tokens == 1:
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

        # Save for next `build` call
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens
        self.decode_num_tokens_cpu[:num_decodes] = torch.tensor(decode_num_tokens)
        self.decode_num_tokens.copy_(self.decode_num_tokens_cpu, non_blocking=True)

        return modified_batch

    def _get_graph_runner_block_tables(
            self, num_decode_tokens: int, block_tables: torch.Tensor) -> torch.Tensor:

        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        if max_batch_size < num_decode_tokens:
            raise RuntimeError("max_batch_size must be greater than or equal to num_decode_tokens")

        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_batch_size, max_blocks),
                                             dtype=block_tables.dtype,
                                             device=block_tables.device)
        else:
            graph_block_tables = self.runner.graph_block_tables.to(
                device=block_tables.device, dtype=block_tables.dtype, non_blocking=True)

        graph_block_tables = graph_block_tables[:block_tables.shape[0]]

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
    
    def get_kv_index(self, seq_lens: torch.Tensor, block_tables: torch.Tensor, block_size: int):
        num_reqs = seq_lens.shape[0]
        if block_tables.shape[0] != num_reqs:
            raise RuntimeError(f"block_tables and seq_lens do not match. seq_lens.shape={seq_lens.shape}, block_tables.shape={block_tables.shape}")
        slots = block_tables[..., None] * block_size + torch.arange(block_size)
        slots = slots.reshape(num_reqs, -1)
        kv_index = torch.empty(seq_lens.sum(), dtype=slots.dtype)
        start = 0
        for i, n in enumerate(seq_lens):
            kv_index[start:start+n] = slots[i, :n]
            start += n
        return kv_index
    
    def build(self,
              num_reqs,
              num_actual_tokens,
              max_query_len,
              common_prefix_len,
              graph_pad_size=-1):

        block_table = self.block_table.get_device_tensor()[:num_reqs]

        seq_lens = self.runner.seq_lens_cpu[:num_reqs]
        query_lens = seq_lens - self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]

        slot_mapping = self.block_table.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True)
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True)

        attn_state = self.runner.attn_state

        prefix_meta: Optional[PrefixCopyMeta] = None
        kv_transfer_cfg = self.runner.vllm_config.kv_transfer_config
        if (self.omni_cache is not None
            and model_extra_config.operator_opt_config.use_omni_cache
            and kv_transfer_cfg is not None
            and kv_transfer_cfg.kv_role != "kv_consumer"
            and attn_state != AscendAttentionState.DecodeOnly):
            kv_lens = self.runner.input_batch.num_computed_tokens_cpu[:num_reqs]
            query_lens_list = query_lens.tolist()
            block_tables_np = self.block_table.get_numpy_array()[:num_reqs]
            prefix_meta = self.omni_cache.get_prefill_prefix_copy_meta(
                block_size=self.block_size,
                kv_lens=kv_lens,
                query_lens_list=query_lens_list,
                block_tables=block_tables_np,
                attn_state=attn_state,
            )

        if graph_pad_size > 0:
            padding = torch.full((graph_pad_size, ),
                                    PAD_SLOT_ID, # sink value stored in block 0, so pad values should not be 0
                                    dtype=slot_mapping.dtype,
                                    device=self.runner.device)
            slot_mapping = torch.cat([slot_mapping, padding])
            padding_0 = torch.zeros(graph_pad_size,
                                    dtype=input_positions.dtype,
                                    device=self.runner.device)
            input_positions = torch.cat([input_positions, padding_0])

        if self.runner.attn_state == AscendAttentionState.DecodeOnly:
            self.generate_activate_mask(num_actual_tokens, num_actual_tokens + graph_pad_size)
            seq_lens = (input_positions + 1).to(seq_lens.dtype)
            query_lens = torch.ones_like(seq_lens)
            block_table = block_table[:self._num_decodes, ...]
            # has speculative tokens
            if self._num_decode_tokens > self._num_decodes:
                block_table = block_table.repeat_interleave(
                    self.decode_num_tokens[:self._num_decodes], dim=0, output_size=self._num_decode_tokens
                )
            block_table_padding = torch.zeros(
                (graph_pad_size, ) + block_table.shape[1:],
                dtype=block_table.dtype,
                device=block_table.device)
            block_table = torch.cat([block_table, block_table_padding],
                                    dim=0)
            block_table = self._get_graph_runner_block_tables(
                self._num_decode_tokens, block_table)
            kv_index = None
        elif self.runner.is_hybrid_chunked_prefill_graph_mode and self.runner.attn_state == AscendAttentionState.ChunkedPrefill:
            kv_index = None
            seq_lens = seq_lens.to(self.runner.device)
            query_lens = query_lens.to(self.runner.device)
            padding_token_num = (graph_pad_size + num_actual_tokens) - torch.sum(query_lens)
            seq_padding_size = (self.runner.max_num_reqs + 1) - num_reqs
            seq_padding = torch.zeros((seq_padding_size,), dtype=seq_lens.dtype, device=self.runner.device)
            seq_lens = torch.cat([seq_lens, seq_padding], dim=0)
            query_padding = torch.zeros((seq_padding_size,), dtype=query_lens.dtype, device=self.runner.device)
            query_lens = torch.cat([query_lens, query_padding], dim=0)
            query_lens[-1] = padding_token_num
            seq_lens[-1] = padding_token_num
            
            block_table = self.block_table.get_device_tensor()
            block_table_padding = torch.zeros(
                (1,) + block_table.shape[1:],
                dtype=block_table.dtype,
                device=block_table.device)
            block_table = torch.cat([block_table, block_table_padding], dim=0)
        elif model_extra_config.operator_opt_config.use_tnd_pa:
            kv_index = None
            if graph_pad_size > 0:
                padding = torch.tensor([graph_pad_size], dtype=query_lens.dtype, device=query_lens.device)
                query_lens = torch.cat([query_lens, padding], dim=0).to(self.runner.device)
                seq_lens = torch.cat([seq_lens, padding], dim=0).to(self.runner.device)
                block_table_padding = torch.zeros(
                    (1, ) + block_table.shape[1:],
                    dtype=block_table.dtype,
                    device=block_table.device,
                )
                block_table = torch.cat([block_table, block_table_padding], dim=0)
        else:
            kv_index = self.get_kv_index(
                seq_lens=seq_lens,
                block_tables=self.block_table.get_cpu_tensor()[:num_reqs],
                block_size=self.block_size,
            )
            kv_index = kv_index.to(self.runner.device)
            
        slot_indices = torch.stack([slot_mapping // self.block_size, slot_mapping % self.block_size], dim=1)

        if hasattr(self.runner.model, 'language_model') and hasattr(self.runner.model.language_model, 'model'):
            first_layer_ind = self.runner.model.language_model.model.start_layer
            Rotary_List = [QwenMRotaryEmbedding, DynamicNTKScalingRotaryEmbedding, MRotaryEmbeddingInterleaved]
            if type(self.runner.model.language_model.model.layers[first_layer_ind].self_attn.rotary_emb) in Rotary_List:
                cos, sin = None, None
            else:
                cos, sin = self.runner.model.language_model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(input_positions)
        else:
            first_layer_ind = self.runner.model.model.start_layer
            if self.is_spec_metadata:
                cos, sin = self.runner.drafter.model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(input_positions)
            else:
                cos, sin = self.runner.model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(input_positions)

        is_pd_seperate_d = self.runner.vllm_config.kv_transfer_config is not None and \
                           self.runner.vllm_config.kv_transfer_config.kv_role == 'kv_consumer'

        attn_metadata = AscendMetadata(num_actual_tokens=num_actual_tokens,
                                       block_tables=block_table,
                                       query_lens=query_lens,
                                       query_lens_list=query_lens.tolist(),
                                       seq_lens=seq_lens,
                                       seq_lens_list=seq_lens.tolist(),
                                       max_query_len=max_query_len,
                                       slot_mapping=slot_mapping,
                                       slot_indices=slot_indices,
                                       attn_state=attn_state,
                                       cos=cos,
                                       sin=sin,
                                       is_pd_seperate_d=is_pd_seperate_d,
                                       kv_index=kv_index,
                                       mc2_mask=self.mc2_mask,
                                       prefix_meta=prefix_meta,
                                       omni_cache=self.omni_cache)
        return attn_metadata

    def build_dummy(self, num_tokens: int, max_pad_size: int = -1) -> AscendMetadata:
        if max_pad_size == -1:
            max_pad_size = self.runner.max_batch_size
        self.generate_activate_mask(0, max_pad_size)
        slot_mapping = torch.zeros(max_pad_size,
                                   dtype=self.runner.slot_mapping_cpu.dtype,
                                   device=self.runner.device)
        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_pad_size, self.runner.graph_block_tables.shape[1]))
            if self.runner.is_hybrid_chunked_prefill_graph_mode:
                graph_block_tables = torch.zeros((self.runner.max_num_reqs + 1, self.runner.graph_block_tables.shape[1]))
        block_table = graph_block_tables.to(
            device=self.runner.device,
            dtype=self.runner.input_batch.block_table[0].get_device_tensor().dtype
        )

        query_lens = torch.ones(max_pad_size, dtype=torch.long, device=self.runner.device, pin_memory=True)
        seq_lens = query_lens * 2

        if self.runner.is_hybrid_chunked_prefill_graph_mode:
            query_lens = torch.zeros(self.runner.max_num_reqs + 1, dtype=torch.long, device=self.runner.device, pin_memory=True)
            query_lens[0] = max_pad_size
            seq_lens = query_lens + 1

        slot_indices = torch.stack([slot_mapping // self.block_size, slot_mapping % self.block_size], dim=1)

        fake_positions = torch.zeros(max_pad_size, dtype=torch.int64, device=self.device)

        if hasattr(self.runner.model, 'language_model') and hasattr(self.runner.model.language_model, 'model'):
            first_layer_ind = self.runner.model.language_model.model.start_layer
            Rotary_List = [QwenMRotaryEmbedding, DynamicNTKScalingRotaryEmbedding, MRotaryEmbeddingInterleaved]
            if type(self.runner.model.language_model.model.layers[first_layer_ind].self_attn.rotary_emb) in Rotary_List:
                cos, sin = None, None
            else:
                cos, sin = self.runner.model.language_model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(fake_positions)
        else:
            first_layer_ind = self.runner.model.model.start_layer
            if self.is_spec_metadata:
                cos, sin = self.runner.drafter.model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(fake_positions)
            else:
                cos, sin = self.runner.model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(fake_positions)

        is_pd_seperate_d = self.runner.vllm_config.kv_transfer_config is not None and \
                           self.runner.vllm_config.kv_transfer_config.kv_role == 'kv_consumer'

        return AscendMetadata(
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
            is_pd_seperate_d=is_pd_seperate_d,
            kv_index=None,
            mc2_mask=self.mc2_mask,
            prefix_meta=None,
            omni_cache=self.omni_cache,
        )

    def mark_static_for_attn_metadata(self, attn_metadata):
        if attn_metadata is not None:
            torch._dynamo.mark_static(attn_metadata.block_tables)
            torch._dynamo.mark_static(attn_metadata.query_lens)
            torch._dynamo.mark_static(attn_metadata.seq_lens)
            if attn_metadata.slot_mapping is not None:
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
            if attn_metadata.slot_indices is not None:
                torch._dynamo.mark_static(attn_metadata.slot_indices)
            if attn_metadata.cos is not None:
                torch._dynamo.mark_static(attn_metadata.cos)
            if attn_metadata.sin is not None:
                torch._dynamo.mark_static(attn_metadata.sin)
            if attn_metadata.mc2_mask is not None:
                torch._dynamo.mark_static(attn_metadata.mc2_mask)


class AscendAttentionBackendImpl(AttentionImpl):

    SHARE_MASK_TRIL_SPARSE = None

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]] = None,
            logits_soft_cap: Optional[float] = None,
            attn_type: str = AttentionType.DECODER,
            use_irope: bool = False,
            kv_stream = None,
            attn_sinks: torch.Tensor = None
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device=current_platform.device_type)
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type
        self.attn_sinks = attn_sinks
        if self.attn_sinks is not None:
            torch._dynamo.mark_static(self.attn_sinks)

        if self.num_heads % self.num_kv_heads != 0:
            raise RuntimeError("self.num_heads must be divisible by self.num_kv_heads")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None

        cur_vllm_config = get_current_vllm_config()
        self.enable_graph_mode = (cur_vllm_config.npu_compilation_config.level >  \
                                  CompilationLevel.NO_COMPILATION and supports_dynamo())
        
        self.is_pd_seperate_d = cur_vllm_config.kv_transfer_config is not None and cur_vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        self.is_hybrid_chunked_prefill_graph_mode = self.enable_graph_mode and not self.is_pd_seperate_d and \
            not cur_vllm_config.additional_config.get("enable_hybrid_graph_mode", False) and cur_vllm_config.scheduler_config.enable_chunked_prefill

        if AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE is None:
            AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device="npu")
            )
        self.use_tnd_pa = model_extra_config.operator_opt_config.use_tnd_pa
        self.kv_stream = kv_stream

    def forward(self, *args, **kwargs):
        # adapted for Gpt-oss
        if self.attn_sinks is not None:
            if self.use_tnd_pa:
                return self.forward_sink_pa(*args, **kwargs)
            else:
                return self.forward_sink(*args, **kwargs)
        # adapted for Pangu 72Bv2 TODO: Refactor the function
        elif "sink_key" in kwargs:
            return self.forward_sink_attention(*args, **kwargs)
        # other
        else:
            if self.use_tnd_pa:
                return self.forward_pa(*args, **kwargs)
            else:
                return self.forward_vanilla(*args, **kwargs)

    def forward_sink(
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: Tuple,
            attn_metadata: AscendMetadata,
            output: Optional[torch.Tensor] = None,
            trace_flag: bool = True,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        attn_type = self.attn_type
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        if attn_metadata is None:
            return output.view(num_tokens, self.hidden_size)
        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError("layer._k_scale_float and layer._v_scale_float must both be 1.0")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                        "encoder/decoder cross-attention "
                                        "are not implemented for "
                                        "PallasAttentionBackendImpl")

        query = query.view(-1, self.num_heads, self.head_size).contiguous()
        key = key.view(-1, self.num_kv_heads, self.head_size).contiguous()
        value = value.view(-1, self.num_kv_heads, self.head_size).contiguous()

                # update kv cache
        if kv_cache[0].numel() > 0 or kv_cache[1].numel():
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            block_size = self.key_cache.shape[1]

            cast_key = key.reshape(-1, 1, self.num_kv_heads * self.head_size)
            cast_value = value.reshape(-1, 1, self.num_kv_heads * self.head_size)

            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                # if prefill does not use paged attention,
                # (1) saving keys and values into kv_cache, and
                # (2) GQA
                # can run simultaneously in two streams
                if self.kv_stream is not None and not hasattr(layer, 'quant_method') and attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                    stream_for_reshape_and_cache = self.kv_stream
                    self.kv_stream.wait_stream(torch.npu.current_stream())
                else:
                    stream_for_reshape_and_cache = torch.npu.current_stream()
                with torch.npu.stream(stream_for_reshape_and_cache):
                    torch_npu._npu_reshape_and_cache(
                        key,
                        value,
                        self.key_cache.view(self.key_cache.shape[0], block_size, self.num_kv_heads, self.head_size),
                        self.value_cache.view(self.value_cache.shape[0], block_size, self.num_kv_heads, self.head_size),
                        attn_metadata.slot_mapping.int()
                    )
            else:
                torch_npu.scatter_update_(self.key_cache, attn_metadata.slot_indices, cast_key, -2)
                torch_npu.scatter_update_(self.value_cache, attn_metadata.slot_indices, cast_value, -2)

        if self.attn_sinks is not None:
            sinks = self.attn_sinks.view(self.num_heads)
        else:
            sinks = None

        if self.sliding_window is None:
            sparse_mode = 3
            pre_tokens = torch.iinfo(torch.int32).max
        else:
            sparse_mode = 4
            pre_tokens = self.sliding_window

        if hasattr(layer, 'quant_method'):
            pass
        # V0-Style scheduler situation.
        elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            actual_seq_qlen = attn_metadata.query_lens.cumsum(dim=0)
            actual_seq_kvlen = attn_metadata.seq_lens.cumsum(dim=0)
            attn_output = torch_npu.npu_fused_infer_attention_score_v2(
                query[:actual_seq_qlen[-1], :],
                key[:actual_seq_qlen[-1], :],
                value[:actual_seq_qlen[-1], :],
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=0,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                learnable_sink=sinks
            )[0]
            output[:actual_seq_qlen[-1], :].copy_(attn_output)
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_batch = attn_metadata.seq_lens.shape[0]
            query = query.view(num_batch, -1, self.num_heads * self.head_size)
            block_tables = attn_metadata.block_tables
            attn_output = None

            query = query.view(-1, self.num_heads, self.head_size)
            block_num, block_size = self.key_cache.shape[0], self.key_cache.shape[1]
            self.key_cache = self.key_cache.view(block_num, block_size, self.num_kv_heads * self.head_size)
            self.value_cache = self.value_cache.view(block_num, block_size, self.num_kv_heads * self.head_size)
            if self.enable_graph_mode:
                attn_output, _ = tng.ops.npu_fused_infer_attention_score_v2(
                    torch.transpose(query.view(num_batch, -1, self.num_heads, self.head_size), 1, 2),
                    self.key_cache,              # block_num, 128, 8 * 64
                    self.value_cache,            # block_num, 128, 8 * 64
                    num_query_heads=self.num_heads, # 64
                    num_key_value_heads=self.num_kv_heads, # 8
                    input_layout="BNSD",
                    softmax_scale=self.scale,  # 1/8
                    actual_seq_qlen=attn_metadata.query_lens,
                    actual_seq_kvlen=attn_metadata.seq_lens,  # []
                    block_table=block_tables,  # 64,32
                    block_size=block_size,     # = 128
                    pre_tokens=pre_tokens,
                    next_tokens=0,
                    sparse_mode=sparse_mode,
                    atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                    learnable_sink=sinks                # (64,)
                )
            else:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                    query,                  # 64, 64, 64
                    self.key_cache,              # block_num, 128, 8 * 64
                    self.value_cache,            # block_num, 128, 8 * 64
                    num_query_heads=self.num_heads, # 64
                    num_key_value_heads=self.num_kv_heads, # 8
                    input_layout="TND",
                    softmax_scale=self.scale,  # 1/8
                    actual_seq_qlen=attn_metadata.query_lens.cumsum(dim=0), # [1,2,....,64] = [1,1,...,1].cumsum()
                    actual_seq_kvlen=attn_metadata.seq_lens,  # []
                    block_table=block_tables,  # 64,32
                    block_size=block_size,     # = 128
                    pre_tokens=pre_tokens,
                    next_tokens=0,
                    sparse_mode=sparse_mode,
                    atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                    learnable_sink=sinks                # (64,)
                )

            output = output.view_as(attn_output)
            output.copy_(attn_output)
        else:
            # attn with sink are not implemented chunked prefill.
            raise NotImplementedError("Attention with sink are not implemented chunked prefill.")

        return output.view(num_tokens, self.hidden_size)

    def forward_sink_pa(
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: Tuple,
            attn_metadata: AscendMetadata,
            output: Optional[torch.Tensor] = None,
            trace_flag: bool = True,
    ) -> torch.Tensor:
        # unsupport enable_c8
        NZ_DIM = 16
        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)

        if attn_metadata is None:
            return output.view(num_tokens, self.hidden_size)

        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError("layer._k_scale_float and layer._v_scale_float must both be 1.0")
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                        "encoder/decoder cross-attention "
                                        "are not implemented for "
                                        "PallasAttentionBackendImpl")
        # View q k v to TND.
        query = query.view(-1, self.num_heads, self.head_size).contiguous()
        key = key.view(-1, self.num_kv_heads, self.head_size).contiguous()
        value = value.view(-1, self.num_kv_heads, self.head_size).contiguous()
        num_batch = attn_metadata.seq_lens.shape[0]
        # value = value.contiguous()

        # update kv cache
        if kv_cache[0].numel() > 0 or kv_cache[1].numel():
            block_size = kv_cache[0].shape[-2]
            assert block_size == 128, f"{block_size}"
            torch_npu.npu_scatter_pa_kv_cache(
                key,
                value,
                kv_cache[0],
                kv_cache[1],
                attn_metadata.slot_mapping
            )

        if self.attn_sinks is not None:
            sinks = self.attn_sinks.view(self.num_heads)
        else:
            sinks = None

        if self.sliding_window is None:
            sparse_mode = 3
            pre_tokens = torch.iinfo(torch.int32).max
        else:
            sparse_mode = 4
            pre_tokens = self.sliding_window

        if (self.enable_graph_mode and attn_metadata.attn_state == AscendAttentionState.DecodeOnly) or (self.is_hybrid_chunked_prefill_graph_mode and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill):
            attn_output = tng.ops.npu_fused_infer_attention_score_v2(
                query,
                kv_cache[0].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                kv_cache[1].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                block_table=attn_metadata.block_tables,
                block_size=block_size,
                sparse_mode=sparse_mode,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=attn_metadata.query_lens.cumsum(dim=0),
                actual_seq_kvlen=attn_metadata.seq_lens,
                inner_precise=1,
                pre_tokens=pre_tokens,
                next_tokens=0,
                learnable_sink=sinks
            )[0]
        else:
            attn_output = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                kv_cache[0].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                kv_cache[1].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                block_table=attn_metadata.block_tables,
                block_size=block_size,
                sparse_mode=sparse_mode,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=attn_metadata.query_lens.cumsum(dim=0),
                actual_seq_kvlen=attn_metadata.seq_lens,
                pre_tokens=pre_tokens,
                next_tokens=0,
                learnable_sink=sinks
            )[0]

        output = output.view_as(attn_output)
        output.copy_(attn_output)

        return output.view(num_tokens, self.hidden_size)

    def forward_pa(
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: Tuple,
            attn_metadata: AscendMetadata,
            output: Optional[torch.Tensor] = None,
            trace_flag: bool = True,
    ) -> torch.Tensor:
        NZ_DIM = get_nz_dim()
        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)

        if attn_metadata is None:
            return output.view(num_tokens, -1)

        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError("layer._k_scale_float and layer._v_scale_float must both be 1.0")
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                        "encoder/decoder cross-attention "
                                        "are not implemented for "
                                        "PallasAttentionBackendImpl")
        # View q k v to TND.
        query = query.view(-1, self.num_heads, self.head_size).contiguous()
        key = key.view(-1, self.num_kv_heads, self.head_size).contiguous()
        value = value.view(-1, self.num_kv_heads, self.head_size).contiguous()
        num_batch = attn_metadata.seq_lens.shape[0]

        if model_extra_config.operator_opt_config.enable_c8:
            # Scale-format constraints under GQA quantization with BNSD layout.
            k_scale = layer.k_scale.view(self.num_kv_heads, 1, -1) 
            v_scale = layer.v_scale.view(self.num_kv_heads, 1, -1)
        else:
            k_scale = None 
            v_scale = None
        use_omni_cache = model_extra_config.operator_opt_config.use_omni_cache
        omni_cache = getattr(attn_metadata, "omni_cache", None)
        block_size = kv_cache[0].shape[-2] if kv_cache[0].numel() > 0 else 128

        # update kv cache
        if kv_cache[0].numel() > 0 or kv_cache[1].numel():
            assert block_size == 128, f"{block_size}"
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly and use_omni_cache and omni_cache is not None:
                kv_states = [
                    key.view(-1, 1, self.num_kv_heads * self.head_size),
                    value.view(-1, 1, self.num_kv_heads * self.head_size),
                ]
                stream_for_copy = torch.npu.current_stream()
                kv_event = torch.npu.Event(blocking=False, enable_timing=False)
                with torch.npu.stream(stream_for_copy):
                    kv_event.record(stream_for_copy)
                try:
                    layer_idx = extract_layer_index(layer.layer_name)
                except Exception:
                    layer_idx = 0
                omni_cache.synchronize_d2h(kv_states, layer_idx, kv_event)
            else:
                if attn_metadata.attn_state == AscendAttentionState.DecodeOnly and use_omni_cache and omni_cache is not None and getattr(attn_metadata, "prefix_meta", None) is not None:
                    try:
                        layer_idx = extract_layer_index(layer.layer_name)
                    except Exception:
                        layer_idx = 0
                    omni_cache.synchronize_h2d(
                        prefix_meta=attn_metadata.prefix_meta,
                        layer_idx=layer_idx,
                    )
                if model_extra_config.operator_opt_config.enable_c8:
                    quant_key = torch_npu.npu_quantize(key.view(-1, self.num_kv_heads * self.head_size), k_scale.view(-1), None, torch.qint8, -1, True)
                    quant_value = torch_npu.npu_quantize(value.view(-1, self.num_kv_heads * self.head_size), v_scale.view(-1), None, torch.qint8, -1, True)
                    quant_key = quant_key.view(-1, self.num_kv_heads, self.head_size).contiguous()
                    quant_value = quant_value.view(-1, self.num_kv_heads, self.head_size).contiguous()
                    torch_npu.npu_scatter_pa_kv_cache(
                        quant_key,
                        quant_value,
                        kv_cache[0],
                        kv_cache[1],
                        attn_metadata.slot_mapping.int()
                    )
                else:
                    torch_npu.npu_scatter_pa_kv_cache(
                        key,
                        value,
                        kv_cache[0],
                        kv_cache[1],
                        attn_metadata.slot_mapping
                    )
    
        if self.enable_graph_mode and attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            attn_output = tng.ops.npu_fused_infer_attention_score_v2(
                torch.transpose(query.view(num_batch, -1, self.num_heads, self.head_size), 1, 2),
                kv_cache[0].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                kv_cache[1].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                softmax_scale=self.scale,
                dequant_scale_key=k_scale,
                dequant_scale_value=v_scale,
                key_quant_mode=0,
                value_quant_mode=0,
                block_table=attn_metadata.block_tables,
                block_size=block_size,
                actual_seq_kvlen=attn_metadata.seq_lens,
                inner_precise=1
            )[0]  
        elif (attn_metadata.attn_state == AscendAttentionState.PrefillNoCache or not attn_metadata.is_pd_seperate_d) and model_extra_config.operator_opt_config.enable_c8:
            attn_output = torch_npu.npu_fused_infer_attention_score_v2(
                    query.unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    num_query_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="BSND",
                    softmax_scale=self.scale,
                    sparse_mode=3,
                    actual_seq_qlen=attn_metadata.query_lens.cumsum(dim=0),
                    actual_seq_kvlen=attn_metadata.seq_lens,
                    atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                )[0].view(-1, self.num_heads, self.head_size)
        elif self.is_hybrid_chunked_prefill_graph_mode and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            attn_output =tng.ops.npu_fused_infer_attention_score_v2(
                query,
                kv_cache[0].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                kv_cache[1].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                block_table=attn_metadata.block_tables,
                block_size=block_size,
                sparse_mode=3,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=attn_metadata.query_lens.cumsum(dim=0),
                actual_seq_kvlen=attn_metadata.seq_lens,
            )[0]
        else:
            attn_output = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                kv_cache[0].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                kv_cache[1].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                block_table=attn_metadata.block_tables,
                block_size=block_size,
                sparse_mode=3,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=attn_metadata.query_lens.cumsum(dim=0),
                actual_seq_kvlen=attn_metadata.seq_lens,
            )[0]

        output = output.view_as(attn_output)
        output.copy_(attn_output)
    
        return output.view(num_tokens, self.hidden_size)

    def forward_vanilla(
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: Tuple,
            attn_metadata: AscendMetadata,
            output: Optional[torch.Tensor] = None,
            trace_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [2, num_blocks, block_size,
                               num_kv_heads * head_size]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads * head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)

        if attn_metadata is None:
            return output.view(num_tokens, -1)

        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError("layer._k_scale_float and layer._v_scale_float must both be 1.0")
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                        "encoder/decoder cross-attention "
                                        "are not implemented for "
                                        "PallasAttentionBackendImpl")
        if model_extra_config.operator_opt_config.enable_c8:
            k_scale = layer.k_scale
            v_scale = layer.v_scale

        # View q k v to BSH.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        value = value.contiguous()
        use_omni_cache = model_extra_config.operator_opt_config.use_omni_cache
        # update kv cache
        omni_cache = getattr(attn_metadata, "omni_cache", None)
        stream_for_reshape_and_cache = torch.npu.current_stream()
        if kv_cache[0].numel() > 0 or kv_cache[1].numel():
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            block_size = self.key_cache.shape[1]

            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                if use_omni_cache and omni_cache is not None:
                    kv_states = [
                        key.view(-1, 1, self.num_kv_heads * self.head_size),
                        value.view(-1, 1, self.num_kv_heads * self.head_size),
                    ]
                    stream_for_copy = self.kv_stream if (
                        self.kv_stream is not None
                        and not hasattr(layer, 'quant_method')
                        and attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
                    ) else torch.npu.current_stream()
                    kv_event = torch.npu.Event(blocking=False, enable_timing=False)
                    with torch.npu.stream(stream_for_copy):
                        kv_event.record(stream_for_copy)
                    try:
                        layer_idx = extract_layer_index(layer.layer_name)
                    except Exception:
                        layer_idx = 0
                    omni_cache.synchronize_d2h(kv_states, layer_idx, kv_event)
                else:
                    if model_extra_config.operator_opt_config.enable_c8:
                        quant_key = torch_npu.npu_quantize(key.view(-1, self.num_kv_heads * self.head_size), k_scale.view(-1), None, torch.qint8, -1, True)
                        quant_value = torch_npu.npu_quantize(value.view(-1, self.num_kv_heads * self.head_size), v_scale.view(-1), None, torch.qint8, -1, True)
                        quant_key = quant_key.view(-1, self.num_kv_heads, self.head_size).contiguous()
                        quant_value = quant_value.view(-1, self.num_kv_heads, self.head_size).contiguous()

                    if self.kv_stream is not None and not hasattr(layer, 'quant_method') and attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                        stream_for_reshape_and_cache = self.kv_stream
                        self.kv_stream.wait_stream(torch.npu.current_stream())
                    else:
                        stream_for_reshape_and_cache = torch.npu.current_stream()
                    with torch.npu.stream(stream_for_reshape_and_cache):
                        torch_npu._npu_reshape_and_cache(
                            key if not model_extra_config.operator_opt_config.enable_c8 else quant_key,
                            value if not model_extra_config.operator_opt_config.enable_c8 else quant_value,
                            self.key_cache.view(self.key_cache.shape[0], block_size, self.num_kv_heads, self.head_size),
                            self.value_cache.view(self.value_cache.shape[0], block_size, self.num_kv_heads, self.head_size),
                            attn_metadata.slot_mapping.int()
                        )
            else:
                if use_omni_cache and omni_cache is not None and getattr(attn_metadata, "prefix_meta", None) is not None:
                    try:
                        layer_idx = extract_layer_index(layer.layer_name)
                    except Exception:
                        layer_idx = 0
                    omni_cache.synchronize_h2d(
                        prefix_meta=attn_metadata.prefix_meta,
                        layer_idx=layer_idx,
                    )
                if model_extra_config.operator_opt_config.enable_c8:
                    torch_npu.npu_quant_scatter_(self.key_cache, attn_metadata.slot_indices, key.view(-1, 1, self.num_kv_heads * self.head_size), k_scale.view(-1), axis=-2, quant_axis=-1)
                    torch_npu.npu_quant_scatter_(self.value_cache, attn_metadata.slot_indices, value.view(-1, 1, self.num_kv_heads * self.head_size), v_scale.view(-1), axis=-2, quant_axis=-1)
                else:
                    cast_key = key.reshape(-1, 1, self.num_kv_heads * self.head_size)
                    cast_value = value.reshape(-1, 1, self.num_kv_heads * self.head_size)
                    torch_npu.scatter_update_(self.key_cache, attn_metadata.slot_indices, cast_key, -2)
                    torch_npu.scatter_update_(self.value_cache, attn_metadata.slot_indices, cast_value, -2)

        # V0-Style scheduler situation.
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:             
            if not (os.getenv("ENABLE_PREFILL_TND", "0") == "1"):
                if attn_metadata is None:
                    raise RuntimeError("attn_metadata must not be None")

                if len(attn_metadata.query_lens_list) == 1:
                    if attn_metadata.additional_metadata != None and attn_metadata.additional_metadata["model_type"] == "hstu_inference_ranking":
                        attn_output = torch.ops.mxrec.hstu_jagged(
                            q=query,
                            k=key,
                            v=value,
                            mask=None,
                            attn_bias=None,
                            mask_type=0,
                            max_seq_len=attn_metadata.max_query_len,
                            silu_scale=1.0 / (self.head_size ** 0.5),
                            seq_offset=attn_metadata.additional_metadata["query_start"],
                            num_context=None,
                            num_target=attn_metadata.additional_metadata["num_candidates"],
                            target_group_size=1,
                        ).view(-1, self.num_heads, self.head_size)
                    else:
                        attn_output = torch_npu.npu_fused_infer_attention_score(
                            query.unsqueeze(0),
                            key.unsqueeze(0),
                            value.unsqueeze(0),
                            num_heads=self.num_heads,
                            num_key_value_heads=self.num_kv_heads,
                            input_layout="BSND",
                            scale=self.scale,
                            sparse_mode=3,
                            actual_seq_lengths=attn_metadata.query_lens_list,
                            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                            atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                        )[0].view(-1, self.num_heads, self.head_size)

                    output = output.view_as(attn_output)
                    output.copy_(attn_output)
                else:
                    actual_seq_qlen = np.array(attn_metadata.query_lens).cumsum().tolist()
                    actual_seq_kvlen = np.array(attn_metadata.seq_lens).cumsum().tolist()

                    attn_output = torch_npu.npu_fusion_attention(
                        query[:actual_seq_qlen[-1], :],
                        key[:actual_seq_qlen[-1], :],
                        value[:actual_seq_qlen[-1], :],
                        head_num=self.num_heads,
                        input_layout="TND",
                        scale=self.scale,
                        atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                        sparse_mode=3,
                        actual_seq_qlen=actual_seq_qlen,
                        actual_seq_kvlen=actual_seq_kvlen)[0]

                    output[:actual_seq_qlen[-1], :].copy_(attn_output)
            else:
                if attn_metadata is None:
                    raise RuntimeError("attn_metadata must not be None")
                actual_seq_qlen = np.array(attn_metadata.query_lens).cumsum().tolist()
                actual_seq_kvlen = np.array(attn_metadata.seq_lens).cumsum().tolist()
                attn_output = torch_npu.npu_fused_infer_attention_score(
                    query[:actual_seq_qlen[-1],:,:], 
                    key[:actual_seq_qlen[-1],:,:],
                    value[:actual_seq_qlen[-1],:,:],
                    num_heads = self.num_heads,
                    num_key_value_heads =  self.num_kv_heads,
                    input_layout = "TND",
                    scale = self.scale,
                    sparse_mode = 3,
                    actual_seq_lengths = actual_seq_qlen,
                    actual_seq_lengths_kv =  actual_seq_kvlen,
                    atten_mask = AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                )[0].view(-1, self.num_heads, self.head_size)
                
                output[:actual_seq_qlen[-1], :].copy_(attn_output)
            if stream_for_reshape_and_cache != torch.npu.current_stream():
                torch.npu.current_stream().wait_stream(stream_for_reshape_and_cache)
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            if attn_metadata.additional_metadata != None and attn_metadata.additional_metadata["model_type"] == "hstu_inference_ranking":
                attn_output = torch.ops.mxrec.hstu_paged(
                    q=query,
                    k=key,
                    v=value,
                    k_cache=self.key_cache,
                    v_cache=self.value_cache,
                    mask=None,
                    attn_bias=None,
                    mask_type=2,
                    max_seq_len=attn_metadata.max_query_len,
                    max_seq_len_k=attn_metadata.additional_metadata["max_seq_len_k"],
                    silu_scale=1.0 / attn_metadata.max_query_len,
                    seq_offset=attn_metadata.additional_metadata["query_start"],   
                    seq_offset_k=attn_metadata.additional_metadata["seq_offset_k"],
                    seq_offset_t=attn_metadata.additional_metadata["seq_offset_t"],
                    page_offsets=attn_metadata.additional_metadata["page_offsets"], 
                    page_ids=attn_metadata.additional_metadata["page_ids"], 
                    last_page_len=attn_metadata.additional_metadata["last_page_len"],
                    num_target=attn_metadata.query_lens
                )
            else:
                if model_extra_config.operator_opt_config.use_omni_cache and omni_cache is not None and attn_metadata.prefix_meta is not None:
                    try:
                        layer_idx = extract_layer_index(layer.layer_name)
                    except Exception:
                        layer_idx = 0
                    omni_cache.synchronize_h2d(
                        prefix_meta=attn_metadata.prefix_meta,
                        layer_idx=layer_idx,
                    )

                block_num, block_size = self.key_cache.shape[0], self.key_cache.shape[1]

                num_batch = attn_metadata.seq_lens.shape[0]
                query = query.view(num_batch, -1, self.num_heads * self.head_size)
                block_tables = attn_metadata.block_tables
                attn_output = None
                if self.enable_graph_mode:
                    attn_output, _ = tng.ops.npu_fused_infer_attention_score(
                        torch.transpose(query.view(num_batch, -1, self.num_heads, self.head_size), 1, 2),
                        self.key_cache,
                        self.value_cache,
                        num_heads=self.num_heads,
                        num_key_value_heads=self.num_kv_heads,
                        input_layout="BNSD",
                        scale=self.scale,
                        key_antiquant_scale = k_scale.view(1, -1) if model_extra_config.operator_opt_config.enable_c8 else None,
                        value_antiquant_scale = v_scale.view(1, -1) if model_extra_config.operator_opt_config.enable_c8 else None,
                        key_antiquant_mode=0,
                        value_antiquant_mode=0,
                        actual_seq_lengths_kv=attn_metadata.seq_lens,
                        block_table=block_tables,
                        block_size=block_size,
                        inner_precise=1
                    )
                else:
                    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                        query,
                        self.key_cache,
                        self.value_cache,
                        num_heads=self.num_heads,
                        num_key_value_heads=self.num_kv_heads,
                        input_layout="BSH",
                        scale=self.scale,
                        key_antiquant_scale = k_scale.view(1, -1) if model_extra_config.operator_opt_config.enable_c8 else None,
                        value_antiquant_scale = v_scale.view(1, -1) if model_extra_config.operator_opt_config.enable_c8 else None,
                        key_antiquant_mode=0,
                        value_antiquant_mode=0,
                        actual_seq_lengths_kv=attn_metadata.seq_lens,
                        block_table=block_tables,
                        block_size=block_size,
                    )

            output = output.view_as(attn_output)
            output.copy_(attn_output)

        # Normal V1 situation.
        else:
            # use chunked prefill for head size 192 scenario, like deepseek
            # paged_attention_splitfuse maybe crash at such scenario

            all_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)[attn_metadata.kv_index].contiguous()
            all_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)[attn_metadata.kv_index].contiguous()
            actual_seq_qlen = np.array(attn_metadata.query_lens).cumsum().tolist()
            actual_seq_kvlen = np.array(attn_metadata.seq_lens).cumsum().tolist()
            attn_output = torch_npu.npu_fusion_attention(
                query,
                all_key,
                all_value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                sparse_mode=3,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )[0]

            output = output.view_as(attn_output)
            output.copy_(attn_output)
            
        return output.view(num_tokens, self.hidden_size)
    
    def forward_sink_attention(
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: Tuple,
            attn_metadata: AscendMetadata,
            output: Optional[torch.Tensor] = None,
            trace_flag: bool = True,
            sink_pad_params: Optional[dict] = None,
            sink_query: Optional[torch.Tensor] = None,
            sink_key: Optional[torch.Tensor] = None,
            sink_value: Optional[torch.Tensor] = None,
            v_head_size: Optional[int] = None,
    ) -> torch.Tensor:
        assert sink_key is not None, "sink_key must be provided"
        NZ_DIM = get_nz_dim()
        num_tokens = query.shape[0]
        if v_head_size == None:
            v_head_size = self.head_size
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 v_head_size,
                                 dtype=query.dtype,
                                 device=query.device)

        if attn_metadata is None:
            return output.view(num_tokens, -1)

        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError("layer._k_scale_float and layer._v_scale_float must both be 1.0")
        
        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                        "encoder/decoder cross-attention "
                                        "are not implemented for "
                                        "PallasAttentionBackendImpl")
        
        # View q k v to TND.
        query = query.view(-1, self.num_heads, self.head_size).contiguous()
        key = key.view(-1, self.num_kv_heads, self.head_size).contiguous()
        value = value.view(-1, self.num_kv_heads, v_head_size).contiguous()

        # Update kv cache
        if kv_cache[0].numel() > 0 or kv_cache[1].numel() > 0:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            block_size = kv_cache[0].shape[-2]
            assert block_size == 128, f"Expected block_size=128, got {block_size}"
            torch_npu.npu_scatter_pa_kv_cache(
                key,
                value,
                kv_cache[0],
                kv_cache[1],
                attn_metadata.slot_mapping
            )         

        # Store sink kv in block 0 (kv cache starts from block 1 and slots 128)
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache or (not self.is_hybrid_chunked_prefill_graph_mode and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill):
            slots = torch.arange(0, 128, device=sink_key.device, dtype=torch.int32)
            torch_npu.npu_scatter_pa_kv_cache(
                sink_key,
                sink_value,
                kv_cache[0],
                kv_cache[1],
                slots
            )        

        block_size = self.value_cache.shape[-2]
        num_batch = attn_metadata.query_lens.shape[0]
        
        # Sink stored in block 0, so pad block_tables with 0 at the beginning
        if sink_pad_params is None:
            block_tables = F.pad(attn_metadata.block_tables, (1, 0, 0, 0), value=0)
            actual_seq_lengths_kv = attn_metadata.seq_lens + 128
        else:
            block_tables = sink_pad_params['sink_block_tables']
            actual_seq_lengths_kv = sink_pad_params['sink_actual_seq_lengths_kv']

        if self.enable_graph_mode and attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            attn_output = tng.ops.npu_fused_infer_attention_score_v2(
                torch.transpose(query.view(num_batch, -1, self.num_heads, self.head_size), 1, 2),
                kv_cache[0].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                kv_cache[1].view(-1, self.num_kv_heads, v_head_size // NZ_DIM, block_size, NZ_DIM),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                softmax_scale=self.scale,
                block_table=block_tables,
                block_size=block_size,
                sparse_mode=3,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_kvlen=actual_seq_lengths_kv,
                inner_precise=1
            )[0]
            attn_output = attn_output[:, :, :, :v_head_size]
        # TODO: ChunkedPrefill
        elif self.is_hybrid_chunked_prefill_graph_mode and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            attn_output =tng.ops.npu_fused_infer_attention_score_v2(
                query,
                kv_cache[0].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                kv_cache[1].view(-1, self.num_kv_heads, v_head_size // NZ_DIM, block_size, NZ_DIM),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                block_table=block_tables,
                block_size=block_size,
                sparse_mode=3,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                actual_seq_qlen=attn_metadata.query_lens.cumsum(dim=0),
                actual_seq_kvlen=actual_seq_lengths_kv,
            )[0]
        else:
            atten_mask = ~torch.tril(torch.ones((2048, 2048), device='npu', dtype=torch.bool))
            attn_output = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                kv_cache[0].view(-1, self.num_kv_heads, self.head_size // NZ_DIM, block_size, NZ_DIM),
                kv_cache[1].view(-1, self.num_kv_heads, v_head_size // NZ_DIM, block_size, NZ_DIM),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                softmax_scale=self.scale,
                block_table=block_tables,
                block_size=block_size,
                sparse_mode=3,
                atten_mask=atten_mask,
                actual_seq_qlen=attn_metadata.query_lens.cumsum(dim=0),
                actual_seq_kvlen=actual_seq_lengths_kv,
            )[0]

        output = output.view_as(attn_output)
        output.copy_(attn_output)
    
        return output.view(num_tokens, -1)
        
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            v_head_size: int = None
    ) -> Tuple[int, ...]:
        if v_head_size is None:
            if model_extra_config.operator_opt_config.use_tnd_pa:
                NZ_DIM = get_nz_dim()
                return (2, num_blocks, num_kv_heads * head_size // NZ_DIM, block_size, NZ_DIM)
            else:
                return (2, num_blocks, block_size, num_kv_heads * head_size)
        # adapted for Pangu 72Bv2, for different K and V head sizes
        else:
            if model_extra_config.operator_opt_config.use_tnd_pa:
                NZ_DIM = get_nz_dim()
                k_cache_shape = (num_blocks, num_kv_heads * head_size // NZ_DIM, block_size, NZ_DIM)
                v_cache_shape = (num_blocks, num_kv_heads * v_head_size // NZ_DIM, block_size, NZ_DIM)
                return (k_cache_shape, v_cache_shape)
            else:
                k_cache_shape = (num_blocks, block_size, num_kv_heads * head_size)
                v_cache_shape = (num_blocks, block_size, num_kv_heads * v_head_size)
                return (k_cache_shape, v_cache_shape)

    @staticmethod
    def swap_blocks(
            src_kv_cache: List[torch.Tensor],
            dst_kv_cache: List[torch.Tensor],
            src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
            kv_caches: List[torch.Tensor],
            src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def init_kv_cache_each_layer(kv_cache_shape, dtype, device, model_config: "ModelConfig", enable_graph_mode) -> \
    tuple[torch.Tensor, ...]:
        # KVCache needs to store the shape of the reduced dimension [num_blocks, block_size, 1, kv_lora_rank] [num_blocks, block_size, 1, rope_dim]
        # The shape of the augmented dimension is [num_blocks, block_size, head_num, head_dim]
        
        # adapted for Pangu 72Bv2,  for different K and V head sizes
        if isinstance(kv_cache_shape, tuple) and len(kv_cache_shape) == 2 and isinstance(kv_cache_shape[0], tuple):
            k_cache_shape, v_cache_shape = kv_cache_shape
            layer_k_cache = torch.zeros(k_cache_shape,
                                        dtype=dtype,
                                        device=device)
            layer_v_cache = torch.zeros(v_cache_shape,
                                        dtype=dtype,
                                        device=device)
            if not int(os.getenv("NO_NPU_MOCK", "0")) and device != "cpu":
                layer_k_cache = torch_npu.npu_format_cast(layer_k_cache, 2)
                layer_v_cache = torch_npu.npu_format_cast(layer_v_cache, 2)
            
            return (layer_k_cache, layer_v_cache)
        else:
            # Original implementation for same K and V head sizes
            layer_kv_caches = torch.zeros(
                kv_cache_shape,
                dtype=dtype if not getattr(model_extra_config.operator_opt_config, "enable_c8", False) else torch.int8,
                device=device,
            )
            if not int(os.getenv("NO_NPU_MOCK", "0")) and device != "cpu":
                torch_npu.npu_format_cast(layer_kv_caches, 2)
            return (layer_kv_caches[0], layer_kv_caches[1])
