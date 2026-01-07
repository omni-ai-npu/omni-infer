#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import copy
import gc
import os
import time
from typing import TYPE_CHECKING, Dict, Optional, Union, Any, List
from contextlib import nullcontext

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from vllm.config import CompilationLevel, VllmConfig, get_layers_from_vllm_config
from vllm.attention.layer import Attention
from vllm.attention import AttentionType
from vllm.distributed.parallel_state import get_pp_group, get_tensor_model_parallel_world_size, get_dp_group, get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils import (DeviceMemoryProfiler, is_pin_memory_available,
                        LayerBlockType, LazyLoader, cdiv)
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from omni.models.config_loader.loader import model_extra_config, call_config_updater
from omni.adaptors.vllm.forward_context import set_forward_context
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.attention.backend.attention_dummy_builder import DummyAttentionMetadataBuilder
from omni.layers.sampler import SimpleSampler, AscendSamplerV1
from omni.layers.npu_sampler_cache import PenaltyCache, ProbCache
from omni.adaptors.vllm.platform import NPUPlatform
from omni.adaptors.vllm.spec_decode.post_drafter import PostDrafter
from omni.adaptors.vllm.worker.cache_engine import CacheEngine
from omni.adaptors.vllm.utils import get_attr_by_names
from omni.accelerators.reasoning_compression.apply import init_reasoner_compression_configs
from omni.accelerators.reasoning_compression.config import ThinkCompressDict

import json
profiling_is_set = os.getenv("PROFILING_NAMELIST", None) is not None
if profiling_is_set:
    from omni.tools.profiler.apply_profiler_patches import patch_ModelRunnerOutput
    patch_ModelRunnerOutput()

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    
logger = init_logger("vllm.npu_model_runner")

_GLOBAL_STEP = 0
MAX_GEAR_NUM = 6
NPU_GENERATOR_OFFSET_STEP = 12 # ascend npu, move 12 every one generation, which is 4 on cuda.

PRE_NUM_REQS = 0
PRE_NUM_INPUT_TOKENS = 0
COUNTER = 0
PRE_COST = 0
COST_THRESHOLD: float = float(os.environ.get("NPU_MODEL_RUNNER_COST_THRESHOLD", "0"))

def _get_pad_size(num_seqs):
    tp_size = get_tensor_model_parallel_world_size()
    if model_extra_config.parall_config.attn_sp_size > 1:
        tp_size = tp_size * 2
    return (tp_size - num_seqs % tp_size) % tp_size

class GraphCompileConfiguration:
    """
    When the graph mode is turned on
    you can set the gear or clarify the static shape by inheriting this class to speed up the model running
    """

    def set_dynamic_gears(self, *args, **kwargs):
        pass


    def mark_static_for_graph(self, *args, **kwargs):
        torch._dynamo.mark_static(args[0])
        torch._dynamo.mark_static(args[1])

def mark_static_for_graph_default(
        input_ids,
        inputs_embeds: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[torch.Tensor]] = None,
        hidden_states: Optional[torch.Tensor] = None
    ):
    if input_ids is not None:
        torch._dynamo.mark_static(input_ids)
    if inputs_embeds is not None:
        torch._dynamo.mark_static(inputs_embeds)
    if positions is not None:
        torch._dynamo.mark_static(positions)

    if kv_caches is not None:
        for kv_cache in kv_caches:
            if kv_cache is not None:
                torch._dynamo.mark_static(kv_cache[0]) # k_cache
                torch._dynamo.mark_static(kv_cache[1]) # v_cache

    if hidden_states is not None:
        torch._dynamo.mark_static(hidden_states)

class NPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        init_reasoner_compression_configs(vllm_config)
        super().__init__(vllm_config, device)
        self.cache_engine = None
        self.head_size = self.model_config.get_head_size()
        self.block_size = vllm_config.cache_config.block_size

        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            vllm_config.parallel_config, LayerBlockType.attention)
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        if vllm_config.additional_config is not None:
            self.use_rejection_sampler = vllm_config.additional_config.get("use_rejection_sampler", False)
            self.use_penalty = vllm_config.additional_config.get("use_penalty", False)
            self.total_step = vllm_config.additional_config.get("multi_step", 1)
            self.combine_block = vllm_config.additional_config.get("combine_block", 1)
            self.use_process_before_sample = vllm_config.additional_config.get("use_process_before_sample", False)
        else:
            self.use_rejection_sampler = False
            self.use_penalty = False
            self.total_step = 1
            self.combine_block = 1
            self.use_process_before_sample = False
        self.curr_step = 0
        if self.total_step > 1:
            self.prepare_inputs_event = torch.Event()
        else:
            self.prepare_inputs_event = None
        num_tokens_per_reqs_decode = 1 if not self.use_spec_decode else (1 + self.speculative_config.num_speculative_tokens)
        self.num_tokens_per_reqs_decode = num_tokens_per_reqs_decode
        self.decode_max_num_tokens = self.max_num_reqs * self.num_tokens_per_reqs_decode
        if get_pp_group().is_last_rank:
            from omni.adaptors.vllm.sample.sampler import AscendSamplerV1 as NewAscendSamplerV1
            self.sampler = NewAscendSamplerV1(self)
            if self.use_spec_decode:
                from omni.adaptors.vllm.sample.validator import SimpleValidator, SparseRejectionSamplerValidator
                if not self.use_rejection_sampler:
                    self.rejection_sampler = SimpleValidator(vllm_config, device, self)
                else:
                    self.rejection_sampler = SparseRejectionSamplerValidator(vllm_config, device, self)
                self.drafter = PostDrafter(vllm_config, device, self)


        self._init_graph_options()

        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=is_pin_memory_available())
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=is_pin_memory_available())
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int64,
                                    device=self.device)
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int64,
                                        device="cpu",
                                        pin_memory=is_pin_memory_available())
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        self.chunk_next_tokens = torch.zeros(
            self.max_num_reqs * num_tokens_per_reqs_decode, dtype= torch.int64, device=self.device
        )
        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size*self.combine_block)*self.combine_block
        self.graph_block_tables = np.zeros(
            (self.max_num_reqs * num_tokens_per_reqs_decode,
             self.max_num_blocks_per_req),
            dtype=np.int32)

        self.cu_num_draft_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device,
        )
        self.logits_indices = torch.zeros(
            self.decode_max_num_tokens, dtype=torch.int32, device=self.device,
        )
        self.target_logits_indices = torch.zeros(
            self.decode_max_num_tokens, dtype=torch.int32, device=self.device,
        )
        self.bonus_logits_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device,
        )

        self.cu_num_draft_tokens_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, pin_memory=is_pin_memory_available(),
        )
        self.logits_indices_cpu = torch.zeros(
            self.decode_max_num_tokens, dtype=torch.int32, pin_memory=is_pin_memory_available(),
        )
        self.target_logits_indices_cpu = torch.zeros(
            self.decode_max_num_tokens, dtype=torch.int32, pin_memory=is_pin_memory_available(),
        )
        self.bonus_logits_indices_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, pin_memory=is_pin_memory_available(),
        )

        self.attn_mask = None
        self.attn_state = None

        self.model_mark_static = False
        self.dummy_model_mark_static = False
        self.drafter_mark_static = False
        self.dummy_drafter_mark_static = False

        self.arange_npu = torch.arange(max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
                                       dtype=torch.int64,
                                       device=self.device)
        self.arange_npu_int32 = torch.arange(max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
                                             dtype=torch.int32,
                                             device=self.device)
        
        self.omni_cache = None
        rank = get_tensor_model_parallel_rank()
        self.training_data_save_path = os.environ.get('TRAINING_DATA_SAVE_PATH', "")
        self.token_threshold = int(os.environ.get("TRAINING_DATA_TOKEN_THRESHOLD", 1024))
        prepare_for_training = self.training_data_save_path != "" and rank == 0
        self.save_hidden_states = False
        self.save_token_ids = False
        if prepare_for_training:
            if self.vllm_config.kv_transfer_config.kv_role == "kv_consumer":
                self.save_token_ids = True
            else:
                self.save_hidden_states = True

        if model_extra_config.operator_opt_config.enable_c8 and not self.vllm_config.model_config.use_mla:
            self.kv_cache_dtype = torch.int8

        self.finished_sending = set()
        self.finished_recving = set()
        self.loading_kv_failure = set()

    def _init_graph_options(self):
        from vllm.utils import supports_dynamo

        self.enable_torchair_graph_mode = (
                    self.vllm_config.npu_compilation_config.level > CompilationLevel.NO_COMPILATION and supports_dynamo())
        self.use_cached_npu_graph = self.vllm_config.npu_compilation_config.use_ge_graph_cached
        self.decode_gear_list = self.vllm_config.npu_compilation_config.decode_gear_list
        if not self.use_spec_decode:
            self.max_batch_size = self.max_num_reqs
        elif not self.speculative_config.enable_adaptive:
            self.max_batch_size = self.max_num_reqs * (1 + self.speculative_config.num_speculative_tokens)
        else:
            if self.decode_gear_list is None or len(self.decode_gear_list) == 0:
                raise RuntimeError("When enable adaptive speculative decoding, decode_gear_list must be set.")
            self.max_batch_size = self.decode_gear_list[0]
        self.is_pd_seperate_d = self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        self.is_hybrid_chunked_prefill_graph_mode = self.enable_torchair_graph_mode and not self.is_pd_seperate_d and \
            not self.vllm_config.additional_config.get("enable_hybrid_graph_mode", False) and self.vllm_config.scheduler_config.enable_chunked_prefill
        if self.is_hybrid_chunked_prefill_graph_mode:    
            self.max_batch_size = self.max_num_tokens

        if self.decode_gear_list is None:
            self.decode_gear_list = []
            self.decode_gear_list.append(self.max_batch_size)

    def _calc_spec_decode_metadata_same_num(
        self,
        num_draft_tokens,
        cu_num_scheduled_tokens,
    ) -> SpecDecodeMetadata:
        if num_draft_tokens[0] == 0:
            logits_indices = cu_num_scheduled_tokens - 1
            self.logits_indices_cpu[:logits_indices.size] = torch.from_numpy(logits_indices)
            self.logits_indices.copy_(self.logits_indices_cpu, non_blocking=True)
            logits_indices = self.logits_indices[:logits_indices.size]
            metadata = SpecDecodeMetadata(
                draft_token_ids=torch.zeros((0,), dtype=self.input_ids.dtype, device=self.input_ids.device),
                num_draft_tokens=num_draft_tokens.tolist(),
                cu_num_draft_tokens=torch.zeros((cu_num_scheduled_tokens.size,), dtype=self.cu_num_draft_tokens.dtype, device=self.cu_num_draft_tokens.device),
                target_logits_indices=logits_indices,
                bonus_logits_indices=logits_indices,
                logits_indices=logits_indices,
            )
        else:
            # Decode Only
            num_tokens = cu_num_scheduled_tokens[-1]
            batch_size = cu_num_scheduled_tokens.size
            input_ids = self.input_ids[:num_tokens]
            target_range = self.arange_npu_int32[:num_draft_tokens[0]]
            token_start_indices = self.arange_npu_int32[:batch_size] * (num_draft_tokens[0] + 1)
            metadata = SpecDecodeMetadata(
                draft_token_ids=input_ids.view(batch_size, -1)[:, 1:].reshape(-1),
                num_draft_tokens=num_draft_tokens.tolist(),
                cu_num_draft_tokens=self.arange_npu_int32[:batch_size] * num_draft_tokens[0] + num_draft_tokens[0],
                target_logits_indices=(token_start_indices[:, None] + target_range[None, :]).reshape(-1),
                bonus_logits_indices=token_start_indices + num_draft_tokens[0],
                logits_indices=self.arange_npu_int32[:num_tokens],
            )

        return metadata

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        if (num_draft_tokens[0] == num_draft_tokens).all():
            return self._calc_spec_decode_metadata_same_num(
                num_draft_tokens, cu_num_scheduled_tokens,
            )
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens,
                                    num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens,
                                    num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # Adapt: Optimize the CPU -> NPU copy.
        self.cu_num_draft_tokens_cpu[:cu_num_draft_tokens.size] = torch.from_numpy(cu_num_draft_tokens)
        self.logits_indices_cpu[:logits_indices.size] = torch.from_numpy(logits_indices)
        self.target_logits_indices_cpu[:target_logits_indices.size] = torch.from_numpy(target_logits_indices)
        self.bonus_logits_indices_cpu[:bonus_logits_indices.size] = torch.from_numpy(bonus_logits_indices)

        self.cu_num_draft_tokens.copy_(self.cu_num_draft_tokens_cpu, non_blocking=True)
        self.logits_indices.copy_(self.logits_indices_cpu, non_blocking=True)
        self.target_logits_indices.copy_(self.target_logits_indices_cpu, non_blocking=True)
        self.bonus_logits_indices.copy_(self.bonus_logits_indices_cpu, non_blocking=True)

        cu_num_draft_tokens = self.cu_num_draft_tokens[:cu_num_draft_tokens.size]
        logits_indices = self.logits_indices[:logits_indices.size]
        target_logits_indices = self.target_logits_indices[:target_logits_indices.size]
        bonus_logits_indices = self.bonus_logits_indices[:bonus_logits_indices.size]

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[dict[str, Any], int, torch.Tensor, torch.Tensor, bool]:
        # Check input valid
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if total_num_scheduled_tokens <= 0:
            raise RuntimeError("total_num_scheduled_tokens must be greater than 0")
        num_reqs = self.input_batch.num_reqs
        if num_reqs <= 0:
            raise RuntimeError("num_reqs must be greater than 0")
        num_input_tokens = total_num_scheduled_tokens
        tp_rank = get_tensor_model_parallel_rank()
        if tp_rank == 0:
            if COST_THRESHOLD == 0:
                logger.info(f"current num reqs = {num_reqs}, num_input_tokens = {num_input_tokens}")
            else:
                global PRE_NUM_REQS, PRE_NUM_INPUT_TOKENS, COUNTER
                if num_reqs != PRE_NUM_REQS or num_input_tokens != PRE_NUM_INPUT_TOKENS:
                    logger.info(f"current num reqs = {num_reqs}, num_input_tokens = {num_input_tokens}, last_counter = {COUNTER}")
                    PRE_NUM_REQS = num_reqs
                    PRE_NUM_INPUT_TOKENS = num_input_tokens
                    COUNTER = 0
                else:
                    COUNTER += 1

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        num_scheduled_tokens = np.array([
            scheduler_output.num_scheduled_tokens[req_id]
            for req_id in self.input_batch.req_ids
        ], dtype=np.int32)
        max_num_scheduled_tokens = num_scheduled_tokens.max()
        num_scheduled_spec_decode_reqs = len(scheduler_output.scheduled_spec_decode_tokens)

        # Prepare positions
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens)
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)

        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens], non_blocking=True)
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            # Common case (1D positions)
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
            positions = self.positions[:num_input_tokens]

        self.seq_lens_np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens

        # Calculate the slot mapping for each KV cache group.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[kv_cache_group_id]
            # NOTE(runze): since each request has at most M blocks, the offset is at most M-1
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                np.minimum(positions_np // block_size, block_table.max_num_blocks_per_req - 1))
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
            block_offsets = positions_np % block_size
            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:total_num_scheduled_tokens])

        # check and set attention state
        can_decode = self.vllm_config.kv_transfer_config is None or self.vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        num_no_spec_reqs = np.sum(num_scheduled_tokens == 1)
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        if can_decode and (num_scheduled_spec_decode_reqs + num_no_spec_reqs == num_reqs):
            attn_state = AscendAttentionState.DecodeOnly
        elif np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # splitfuse
        else:
            attn_state = AscendAttentionState.ChunkedPrefill

        if self.is_hybrid_chunked_prefill_graph_mode and attn_state == AscendAttentionState.DecodeOnly:
            attn_state = AscendAttentionState.ChunkedPrefill

        self.attn_state = attn_state

        # calculate max_batch_size and padding size
        graph_pad_size = 0
        if self.enable_torchair_graph_mode and len(self.decode_gear_list) > 1:
            if attn_state == AscendAttentionState.DecodeOnly or self.is_hybrid_chunked_prefill_graph_mode:
                self.max_batch_size = self._get_max_token_num(self.vllm_config.parallel_config.data_parallel_size > 1, total_num_scheduled_tokens)

        if attn_state == AscendAttentionState.DecodeOnly:
            if total_num_scheduled_tokens > self.max_batch_size:
                raise RuntimeError("num_reqs is bigger than max_batch_size")
            graph_pad_size = self.max_batch_size - total_num_scheduled_tokens
        elif self.is_hybrid_chunked_prefill_graph_mode:
            graph_pad_size = self.max_batch_size - total_num_scheduled_tokens    
        else:
            # The reduce_scatter in the TP communication domain after embedding, P goes through this
            graph_pad_size = _get_pad_size(num_input_tokens)

        if graph_pad_size >= 0:
            if self.uses_mrope:
                padding_positions = torch.zeros(positions.size(0), graph_pad_size, dtype=positions.dtype, device=positions.device)
                positions = torch.cat([positions, padding_positions], dim=1)
            else:
                padding_positions = torch.zeros(graph_pad_size, dtype=positions.dtype, device=positions.device)
                positions = torch.cat([positions, padding_positions])

        extra_builder_kwargs = {'graph_pad_size': graph_pad_size}

        # build attention metadata
        attn_metadata = {}
        self.full_attn_metadata = None
        if not int(os.getenv("NO_NPU_MOCK", "0")):
            for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
                # Prepare for cascade attention if enabled & beneficial.
                common_prefix_len = 0
                if self.cascade_attn_enabled:
                    common_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        scheduler_output.
                        num_common_prefix_blocks[kv_cache_group_id],
                        kv_cache_group_spec.kv_cache_spec,
                        self.attn_metadata_builders[kv_cache_group_id],
                    )
                attn_metadata_i = self.attn_metadata_builders[kv_cache_group_id].build(
                    num_reqs=num_reqs,
                    num_actual_tokens=total_num_scheduled_tokens,
                    max_query_len=max_num_scheduled_tokens,
                    common_prefix_len=None,
                    **extra_builder_kwargs,
                )
                if kv_cache_group_id == 0:
                    self.full_attn_metadata = attn_metadata_i

                if not isinstance(self.attn_metadata_builders[kv_cache_group_id], DummyAttentionMetadataBuilder):
                    raise ValueError(f"{self.attn_metadata_builders[kv_cache_group_id]} does not implement DummyAttentionMetadataBuilder")
                if attn_state == AscendAttentionState.DecodeOnly and model_extra_config.operator_opt_config.mtp_remove_redundant_kv:
                    num_speculative_tokens = 0 if not self.speculative_config else self.speculative_config.num_speculative_tokens
                    mtp_idx = torch.arange(1, self.max_batch_size, 1 + num_speculative_tokens, dtype=torch.int64).npu()
                    new_block_table = torch.index_select(attn_metadata_i.decode.block_table, dim=0, index=mtp_idx)
                    new_seq_lens = torch.index_select(attn_metadata_i.decode.seq_lens, dim=0, index=mtp_idx)
                    attn_metadata_i.decode.block_table = new_block_table
                    attn_metadata_i.decode.seq_lens = new_seq_lens
                if self.enable_torchair_graph_mode and attn_state == AscendAttentionState.DecodeOnly:
                    self.attn_metadata_builders[kv_cache_group_id].mark_static_for_attn_metadata(attn_metadata_i)
                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        # Prepare input_ids
        token_indices = (positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Copy the tensors to the NPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)

        if model_extra_config.parall_config.attn_sp_size > 1:
            sp_size = model_extra_config.parall_config.attn_sp_size * 2
            cu_num_tokens = np.empty_like(num_scheduled_tokens)
            cu_num_tokens[0] = num_scheduled_tokens[0]
            for i in range(1, num_scheduled_tokens.size):
                prev_aligned = ((cu_num_tokens[i - 1] + sp_size - 1) // sp_size) * sp_size
                cu_num_tokens[i] = prev_aligned + num_scheduled_tokens[i]

        if self.use_spec_decode:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            sample_indices = spec_decode_metadata.logits_indices
        else:
            sample_indices = cu_num_tokens - 1
            sample_indices = torch.from_numpy(sample_indices).to(self.device, non_blocking=True)
            spec_decode_metadata = None

        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return attn_metadata, graph_pad_size, sample_indices, positions, spec_decode_metadata

    def advance_step_spec(self,
            input_tokens, sampled_tokens, spec_tokens, input_positions,
            seq_lens, slot_mapping, block_table, accepted_num,
            num_reqs, num_queries, block_size):
        if spec_tokens is None:
            token_each_reqs = 1
        else:
            token_each_reqs = 1 + len(spec_tokens[0])
        total_num_scheduled_tokens = token_each_reqs * num_reqs
        if accepted_num is None:
            input_positions += 1
        else:
            input_positions += torch.repeat_interleave(accepted_num, token_each_reqs) + 1
        seq_lens.copy_((input_positions + 1).to(seq_lens.dtype))
        index = torch.argmin(torch.cat([sampled_tokens,
            torch.full((num_reqs, 1), -1, device=sampled_tokens.device)],
            dim = 1), dim = 1) - 1
        last_tokens = sampled_tokens[torch.arange(num_reqs, device='npu'), index]
        if token_each_reqs == 1:
            input_tokens[:num_reqs] = last_tokens.to(dtype=input_tokens.dtype)
        else:
            input_tokens_2d = input_tokens.view(-1, token_each_reqs)
            input_tokens_2d[:num_reqs, 0] = last_tokens
            input_tokens_2d[:num_reqs, 1:] = spec_tokens

        req_indices = torch.repeat_interleave(torch.arange(num_reqs, device='npu'),
            token_each_reqs, dim=0)
        max_num_blocks_per_req = block_table.shape[1]
        block_table_indices = (
            req_indices * max_num_blocks_per_req +
            input_positions // block_size
        )
        block_numbers = block_table.flatten()[block_table_indices]
        block_offsets = input_positions % block_size
        slot_mapping.copy_(block_numbers * block_size + block_offsets)

    def deal_metadata(
        self,
        attn_metadata,
        input_ids,
        positions,
        is_dummy = False,
    ):
        def deal_slots(slot_mapping, batch_lens, seq_lens):
            start_idx = 0
            dealed_slot_mapping = []
            for idx in range(seq_lens.shape[0]):
                dealed_slot_mapping.append(slot_mapping[start_idx : start_idx + batch_lens[idx]])
                if idx < seq_lens.shape[0] - 1:
                    start_idx = seq_lens[idx + 1]

            return torch.cat(dealed_slot_mapping, dim=0)

        if attn_metadata is not None:
            logger.info("Start to deal_metadata")
            first_metadata = next(iter(attn_metadata.values()))
            is_prefill = input_ids[-1] == -1 if not is_dummy else False
            uid = input_ids[0]
            input_ids_list = []
            query_start_loc = torch.zeros(first_metadata.query_lens.shape[0] + 1, dtype=torch.int64, device=first_metadata.query_lens.device)
            torch.cumsum(first_metadata.query_lens, dim=0, out=query_start_loc[1:])
            query_start_loc = query_start_loc.npu()

            if not is_dummy:
                for idx in range(query_start_loc.shape[0] - 1):
                    start_idx = query_start_loc[idx]
                    end_idx = query_start_loc[idx + 1]
                    batch_input_ids = input_ids[start_idx + 1 : end_idx]
                    batch_input_ids = batch_input_ids[batch_input_ids != -1]
                    input_ids_list.append(batch_input_ids)
                input_ids = torch.cat(input_ids_list, dim=0)

            positions = positions[:input_ids.shape[0]]
            batch_lens = (first_metadata.query_lens - 2).npu() if not is_dummy else first_metadata.query_lens.npu()
            max_query_len = batch_lens.max().item()
            sum_tokens = batch_lens.sum().item()
            batch_lens_list = batch_lens.tolist()
            
            first_metadata = next(iter(attn_metadata.values()))
            seq_lens = first_metadata.seq_lens.npu()        # 如果是decode的话，seq_lens应当是batch_lens + prec_his_len
            batch_size = seq_lens.shape[0]
            query_start = torch.zeros(batch_size + 1, dtype=torch.int64, device=seq_lens.device)
            torch.cumsum(batch_lens, dim=0, out=query_start[1:])
            
            if not is_prefill:
                block_table = first_metadata.block_tables
                deal_seq_lens = seq_lens - batch_lens
                num_blocks_per_seq = torch.ceil(deal_seq_lens / self.block_size).to(torch.int64) # 向上取整

                # page_offsets 就是 num_blocks_per_seq 的前缀和
                page_offsets = torch.zeros(batch_size + 1, dtype=torch.int64, device=block_table.device)
                torch.cumsum(num_blocks_per_seq, dim=0, out=page_offsets[1:])

                last_page_len_tensor = deal_seq_lens % self.block_size
                last_page_len_tensor[last_page_len_tensor == 0] = self.block_size # 这里是如果最后一个 block 的长度等于 0，我就给他设置成 block_size
                last_page_len = last_page_len_tensor.to(torch.int64)

                seq_offset_k = torch.zeros(seq_lens.shape[0] + 1, dtype=torch.int64, device=block_table.device)
                torch.cumsum(seq_lens, dim=0, out=seq_offset_k[1:]) # 计算出每个请求的前缀和

                seq_offset_t = torch.zeros(batch_lens.shape[0] + 1, dtype=torch.int64, device=block_table.device)
                torch.cumsum(batch_lens, dim=0, out=seq_offset_t[1:]) # 这里是根据测试的脚本得到：seq_offset_t 是 num_candidates 的前缀和

                max_seq_len_k = torch.max(seq_lens).item()

                current_block_tables = block_table[:batch_size]

                page_ids_list = []
                for i in range(batch_size):
                    num_blocks_for_req = page_offsets[i + 1] - page_offsets[i]
                    valid_blocks = current_block_tables[i, :num_blocks_for_req]
                    page_ids_list.append(valid_blocks)

                page_ids = torch.cat(page_ids_list).to(torch.int64)
            else:
                num_candidates = torch.zeros_like(batch_lens)
            # breakpoint()
            for metadata in attn_metadata.values():
                metadata.query_lens = batch_lens
                metadata.query_lens_list = batch_lens_list
                metadata.seq_lens = batch_lens
                metadata.seq_lens_list = batch_lens_list
                metadata.max_query_len = max_query_len
                metadata.num_actual_tokens = sum_tokens
                metadata.additional_metadata = {}
                metadata.additional_metadata["uid"] = uid
                metadata.additional_metadata["query_start"] = query_start
                if not is_prefill:
                    metadata.attn_state = AscendAttentionState.DecodeOnly
                    metadata.slot_indices = deal_slots(metadata.slot_indices, batch_lens, metadata.seq_lens)
                    metadata.additional_metadata["max_seq_len_k"] = max_seq_len_k
                    metadata.additional_metadata["seq_offset_k"] = seq_offset_k
                    metadata.additional_metadata["seq_offset_t"] = seq_offset_t
                    metadata.additional_metadata["page_offsets"] = page_offsets
                    metadata.additional_metadata["page_ids"] = page_ids
                    metadata.additional_metadata["last_page_len"] = last_page_len
                else:
                    metadata.slot_mapping = deal_slots(metadata.slot_mapping, batch_lens, metadata.seq_lens)
                    metadata.additional_metadata["num_candidates"] = num_candidates
                
        return attn_metadata, input_ids, positions

    def _simple_prepare_inputs(
        self,
        attn_metadata,
        positions,
        cached_token,
        cached_spec,
        accepted_num = 0
    ) -> torch.Tensor:
        if isinstance(accepted_num, int):
            assert accepted_num == 0
            accepted_num = None

        token_each_reqs = 1
        if cached_spec is not None:
            token_each_reqs = 1 + len(cached_spec[0])
        num_reqs = self.input_batch.num_reqs
        total_num_scheduled_tokens = token_each_reqs*num_reqs
        first_kv_group = True
        if len(self.kv_cache_config.kv_cache_groups) > 1:
            backup_positions = positions.clone()
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[kv_cache_group_id]
            first_layer_in_group = kv_cache_group_spec.layer_names[0]
            attn_metadata_i = attn_metadata[first_layer_in_group]
            slot_mapping = block_table.slot_mapping
            if kv_cache_group_spec.kv_cache_spec.use_mla:
                seq_lens = attn_metadata_i.decode.seq_lens
            else:
                seq_lens = attn_metadata_i.seq_lens
            if first_kv_group:
                first_kv_group = False
            else:
                positions.copy_(backup_positions)

            if accepted_num is None:
                self.advance_step_spec(self.input_ids[:total_num_scheduled_tokens], cached_token,
                    cached_spec, positions[:total_num_scheduled_tokens],
                    seq_lens[:total_num_scheduled_tokens],
                    slot_mapping[:total_num_scheduled_tokens],
                    block_table.get_device_tensor(), accepted_num,
                    num_reqs, num_reqs, block_size)
            else:
                torch_npu.npu_advance_step_flashattn(
                    input_tokens=self.input_ids[:total_num_scheduled_tokens],
                    sampled_token_ids=cached_token.to(dtype=torch.int64),
                    spec_token=cached_spec.contiguous().to(dtype=torch.int64),
                    input_positions=positions[:total_num_scheduled_tokens],
                    seq_lens=seq_lens[:total_num_scheduled_tokens],
                    slot_mapping=slot_mapping[:total_num_scheduled_tokens],
                    block_tables=block_table.get_device_tensor()[:num_reqs].to(dtype=torch.int64),
                    accepted_num=accepted_num.to(dtype=torch.int64),
                    num_seqs=num_reqs,
                    num_queries=num_reqs,
                    block_size=block_size)

            if kv_cache_group_spec.kv_cache_spec.use_mla:
                attn_metadata_i.slot_mapping[:total_num_scheduled_tokens] = block_table.slot_mapping[:total_num_scheduled_tokens]
                attn_metadata_i.decode.input_positions[:total_num_scheduled_tokens] = positions[:total_num_scheduled_tokens]
                first_layer_ind = self.model.model.start_layer
                cos, sin = self.model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(attn_metadata_i.decode.input_positions)
                attn_metadata_i.decode.cos = cos
                attn_metadata_i.decode.sin = sin
            else:
                attn_metadata_i.slot_mapping[:total_num_scheduled_tokens] = block_table.slot_mapping[:total_num_scheduled_tokens]
                attn_metadata_i.slot_indices = torch.stack([attn_metadata_i.slot_mapping // block_size,
                    attn_metadata_i.slot_mapping % block_size], dim=1)
                if attn_metadata_i.attn_state == AscendAttentionState.PrefillNoCache:
                    attn_metadata_i.seq_lens_list = attn_metadata_i.seq_lens.tolist()
                else:
                    attn_metadata_i.seq_lens_list = []

                if getattr(self, 'drafter', None) is not None and first_layer_in_group in self.drafter.attn_layer_names:
                    cos, sin = next(self.drafter.model.model.layers.children()).self_attn.rotary_emb.get_cos_sin(positions)
                else:
                    cos, sin = self.model.model.layers[0].self_attn.rotary_emb.get_cos_sin(positions)
                attn_metadata_i.cos = cos
                attn_metadata_i.sin = sin
            if kv_cache_group_id == 0:
                self.full_attn_metadata = attn_metadata_i

            if (self.enable_torchair_graph_mode and self.attn_state == AscendAttentionState.DecodeOnly) or self.is_hybrid_chunked_prefill_graph_mode:
                self.attn_metadata_builders[kv_cache_group_id].mark_static_for_attn_metadata(attn_metadata_i)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata, positions

    def _execute_model(
        self,
        scheduler_output,
        attn_metadata,
        graph_pad_size,
        sample_indices,
        positions,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        start_before_f = time.time()
        num_input_tokens = scheduler_output.total_num_scheduled_tokens
        input_ids = self.input_ids[:num_input_tokens]

        if self.model_config.hf_config.model_type == "hstu_inference_ranking":
            attn_metadata, input_ids, positions = self.deal_metadata(attn_metadata, input_ids, positions)

        model_kwargs = {}
        raw_hidden_states = None
        if not int(os.getenv("NO_NPU_MOCK", "0")):
            attn_state = next(iter(attn_metadata.values())).attn_state
        else:
            attn_state = 0

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model and get_pp_group().is_first_rank:
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)

            self.inputs_embeds[:num_input_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]

            if graph_pad_size >= 0:
                if attn_state == AscendAttentionState.DecodeOnly:
                    padding_embeds = torch.zeros(graph_pad_size, inputs_embeds.size(-1), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                else:
                    vocab_size = self.model_config.get_vocab_size()
                    padding_embeds = torch.randint(1, vocab_size, (graph_pad_size, inputs_embeds.size(-1)), dtype=input_ids.dtype, device=input_ids.device)

                inputs_embeds = torch.cat([inputs_embeds, padding_embeds])

            input_ids = None
        else:
            if graph_pad_size >= 0:
                if attn_state == AscendAttentionState.DecodeOnly or (self.is_hybrid_chunked_prefill_graph_mode and attn_state == AscendAttentionState.ChunkedPrefill):
                    padding = torch.zeros(graph_pad_size, dtype=input_ids.dtype, device=input_ids.device)
                else:
                    vocab_size = self.model_config.get_vocab_size()
                    padding = torch.randint(1, vocab_size, (graph_pad_size,), dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([input_ids, padding])

            inputs_embeds = None

        model_kwargs["selected_indices"] = sample_indices if attn_state != AscendAttentionState.DecodeOnly else None

        start_fc = time.time()
        start_fc_exit = 0
        # Run forward pass
        with set_forward_context(attn_metadata, self.vllm_config):
            start_setup_connector = time.time()
            self.maybe_setup_kv_connector(scheduler_output)
            model_kwargs["kv_caches"] = self.kv_caches
            model_kwargs["attn_metadata"] = attn_metadata
            start_f = time.time()

            if model_extra_config.task_config.enable_omni_placement:
                is_prompt = attn_state != AscendAttentionState.DecodeOnly
                global _GLOBAL_STEP
                self.planner.place_experts()
                _GLOBAL_STEP = _GLOBAL_STEP + 1 if not is_prompt else 0

            if self.enable_torchair_graph_mode and attn_state == AscendAttentionState.DecodeOnly or \
                (self.is_hybrid_chunked_prefill_graph_mode and attn_state == AscendAttentionState.ChunkedPrefill):
                start_debug = time.time()
                logger.debug("Start running compiled model.")
                if not self.model_mark_static:
                    if isinstance(self.model, GraphCompileConfiguration):
                        self.model.mark_static_for_graph(input_ids, positions, attn_metadata, self.kv_caches)
                    else:
                        mark_static_for_graph_default(input_ids, inputs_embeds, positions, self.kv_caches)
                    self.model_mark_static = True
                start_os_env = time.time()
                start_time = time.time()
                forward_results = self.model(
                            input_ids=input_ids,
                            positions=positions,
                            intermediate_tensors=intermediate_tensors,
                            inputs_embeds=inputs_embeds,
                            **model_kwargs,
                        )
                end_model = time.time()
                cost_model = end_model - start_time
                cost_os_env = start_time - start_os_env
                cost_debug = start_debug - start_os_env
                logger.debug(f" ***** model forward: {cost_model:.6f}, os env: {cost_os_env:.6f}, debug: {cost_debug:.6f}")
            else:
                logger.debug("Start running eager model.")
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = self.get_finished_kv_transfers(scheduler_output)
            start_fc_exit = time.time()
        if isinstance(forward_results, tuple):
            raw_hidden_states, hidden_states = forward_results
        else:
            hidden_states = forward_results
            raw_hidden_states = forward_results
        start_ret = time.time()
        cost_before_fc = start_fc - start_before_f
        cost_fc = start_ret - start_fc
        cost_setup_connector = start_f - start_setup_connector
        cost_fc_exit = start_ret - start_fc_exit
        logger.debug(f" ***** before fc {cost_before_fc:.6f}, fc {cost_fc:.6f}={cost_setup_connector:.6f}+{cost_fc_exit:.6f}")

        if self.save_hidden_states and attn_state == AscendAttentionState.PrefillNoCache:
            num_scheduled_tokens = np.array([
                scheduler_output.num_scheduled_tokens[req_id]
                for req_id in self.input_batch.req_ids
            ], dtype=np.int32)

            req_slice = np.zeros(num_scheduled_tokens.size + 1, dtype=num_scheduled_tokens.dtype)
            np.cumsum(num_scheduled_tokens, out=req_slice[1:])

            input_ids_cpu = input_ids.cpu()
            hidden_states_cpu = raw_hidden_states.cpu()
            for i, req_id in enumerate(self.input_batch.req_ids):
                req = self.requests.get(req_id, None)
                if len(req.prompt_token_ids) < self.token_threshold:
                    continue
                data = {
                    'req_id': req_id,
                    'input_ids': input_ids_cpu[req_slice[i]:req_slice[i + 1]].clone(),
                    'hidden_states': hidden_states_cpu[req_slice[i]:req_slice[i + 1]].clone(),
                }
                filename = os.path.join(self.training_data_save_path, f"hidden-states-{time.time_ns()}.pt")
                torch.save(data, filename)

        return hidden_states, raw_hidden_states, input_ids, finished_sending, finished_recving

    def kv_connector_no_forward(
            self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with set_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))
            loading_kv_failure = self.get_loading_kv_failure_req_ids()
        if not finished_sending and not finished_recving and not loading_kv_failure:
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.finished_sending = finished_sending
        output.finished_recving = finished_recving
        output.loading_kv_failure = loading_kv_failure
        if profiling_is_set:
            finished_sending_raw = set()
            finished_recving_raw = set()
            for item in finished_sending:
                req_id, headers_str = item.split('|', 1)
                finished_sending_raw.add(req_id)
                output.finished_sending_headers[req_id] = json.loads(headers_str)
            for item in finished_recving:
                req_id, headers_str = item.split('|', 1)
                finished_recving_raw.add(req_id)
                output.finished_recving_headers[req_id] = json.loads(headers_str)
            output.finished_sending = finished_sending_raw
            output.finished_recving = finished_recving_raw
        return output

    @staticmethod
    def get_loading_kv_failure_req_ids() -> Optional[set[str]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_load_kv_failure_reqs()
        return None

    def _prepare_kv_cache(self, scheduler_output):
        if scheduler_output.blocks_to_swap_in is not None and any(scheduler_output.blocks_to_swap_in):
            self.cache_engine.swap_in(scheduler_output.blocks_to_swap_in)

        if scheduler_output.blocks_to_swap_out is not None and any(scheduler_output.blocks_to_swap_out):
            self.cache_engine.swap_out(scheduler_output.blocks_to_swap_out)

    def save_tokens(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        for req_id in scheduler_output.finished_req_ids:
            req = self.requests.get(req_id, None)
            if req is None or len(req.output_token_ids) < self.token_threshold:
                continue
            to_save = {
                'req_id' : req_id,
                'prompt_token_ids' : req.prompt_token_ids,
                'output_token_ids' : req.output_token_ids,
            }
            filename = os.path.join(self.training_data_save_path, f"token-ids-{time.time_ns()}.pt")
            torch.save(to_save, filename)


    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        start_0 = time.time()

        if self.save_token_ids:
            self.save_tokens(scheduler_output)

        # Update KVConnector with the KVConnector metadata forward().
        self._update_states(scheduler_output)

        self._prepare_kv_cache(scheduler_output)

        self.total_step = scheduler_output.num_step
        # cached values
        attn_metadata = None
        positions = None
        graph_pad_size = None
        sampled_tokens = None
        sample_indices = None
        spec_tokens_tensor = None

        # cached return values
        cached_sampled_token_ids = None
        accepted_num = 0
        sampled_token_ids_list = []

        cost_upd_states = time.time() - start_0
        cost_proc_reqs = 0
        cost_logits = 0
        cost_bitmask = 0
        cost_disc = 0
        cost_sampler = 0
        cost_drafter = 0
        cost_device_output = 0

        self.enable_sleep_mode = bool(int(os.getenv("VLLM_ENABLE_SLEEP_MODE", '0')))
        for self.curr_step in range(self.total_step):
            start_1 = time.time()
            if not scheduler_output.total_num_scheduled_tokens:
                if get_dp_group().world_size > 1 and (not self.enable_sleep_mode):
                   self._dummy_run(1)
                else:
                    time.sleep(0.001) # release GIL
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOuptut if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output)
            if self.curr_step == 0:
                attn_metadata, graph_pad_size, sample_indices, positions, spec_decode_metadata = self._prepare_inputs(scheduler_output)
            else:
                attn_metadata, positions = self._simple_prepare_inputs(attn_metadata, positions,
                        sampled_tokens, spec_tokens_tensor, accepted_num)
            # put MemCpy's sync here to overlap compute and data transfer
            if int(os.environ.get("BATCH_COPY_ASYNC", "0")) == 1:
                if self.omni_cache is not None:
                    self.omni_cache.ascend_cl_stream.sync()
                else:
                    logger.warning("==== omni_cache is None, cannot sync ascend_cl_stream ====")
            hidden_states, raw_hidden_states, input_ids, temp_finished_sending, temp_finished_recving = self._execute_model(scheduler_output,
                                                attn_metadata, graph_pad_size, sample_indices, positions, intermediate_tensors)
            if self.prepare_inputs_event is not None:
                self.prepare_inputs_event.synchronize()
            if temp_finished_sending is not None:
                self.finished_sending.update(temp_finished_sending)
            if temp_finished_recving is not None:
                self.finished_recving.update(temp_finished_recving)
            tmp_loading_kv_failure = self.get_loading_kv_failure_req_ids()
            if tmp_loading_kv_failure is not None:
                self.loading_kv_failure.update(tmp_loading_kv_failure)

            if not get_pp_group().is_last_rank:
                return hidden_states

            sampling_metadata = self.input_batch.sampling_metadata
            if self.curr_step == 0:
                self.sampler.prepare_cache(scheduler_output.scheduled_new_reqs, self.input_batch.req_ids, sampling_metadata, self.input_batch)
            if temp_finished_sending is not None:
                self.finished_sending.update(temp_finished_sending)
            if temp_finished_recving is not None:
                self.finished_recving.update(temp_finished_recving)
            tmp_loading_kv_failure = self.get_loading_kv_failure_req_ids()
            if tmp_loading_kv_failure is not None:
                self.loading_kv_failure.update(tmp_loading_kv_failure)
            start_2 = time.time()

            if isinstance(sample_indices, int):
                logits = self.model.compute_logits(hidden_states[:sample_indices], None)
                if self.use_spec_decode:
                    sample_indices = torch.arange(sample_indices, dtype=torch.int32, device=self.device)
            else:
                if hidden_states.shape[0] == sample_indices.shape[0]:
                    # assume indices=[x1,x2,...,xn], if xn >= n, we cannot slice,
                    # if xn < n, then indices=[0,1,...,n-1], no need to slice.
                    logits = self.model.compute_logits(hidden_states, None)
                else:
                    logits = self.model.compute_logits(hidden_states[sample_indices], None)
            start_3 = time.time()
            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)
            start_4 = time.time()

            # TODO move into scheduler or prepare_inputs
            # find the requests that are doing chunk prefill
            discard_sampled_tokens_req_indices = []
            chunk_next_tokens = [] if self.use_spec_decode else None
            chunk_next_indices = [] if self.use_spec_decode else None

            num_decodes = self.attn_metadata_builders[0]._num_decodes
            num_prefills = self.attn_metadata_builders[0]._num_prefills
            for i, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                if seq_len < req_state.num_tokens:
                    # Ignore the sampled token.
                    # Rewind the generator state as if the token was not sampled.
                    generator = self.input_batch.generators.get(i)
                    if generator is not None:
                        generator.set_offset(generator.get_offset() - NPU_GENERATOR_OFFSET_STEP)
                    # Record the index of the request that should not be sampled,
                    # so that we could clear the sampled tokens before returning.
                    discard_sampled_tokens_req_indices.append(i)
                    if self.use_spec_decode:
                        chunk_next_tokens.append(req_state.get_token_id(seq_len))
                        chunk_next_indices.append(sample_indices[-num_prefills + i])
            if self.use_spec_decode and len(chunk_next_tokens) > 0:
                chunk_next_tokens = torch.tensor(chunk_next_tokens) # CPU
                chunk_next_tokens_buffer = self.chunk_next_tokens[:chunk_next_tokens.numel()]
                chunk_next_tokens_buffer.copy_(chunk_next_tokens, non_blocking=True)
                chunk_next_tokens = chunk_next_tokens_buffer
                chunk_next_indices = torch.stack(chunk_next_indices)
            else:
                chunk_next_tokens = None
                chunk_next_indices = None
            start_5 = time.time()

            if self.use_process_before_sample:
                sampler_output = self.model.process_before_sample(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                    req_ids = self.input_batch.req_ids)

            # Sample the next token and get logprobs if needed.
            if not self.use_spec_decode:
                sampler_output = self.sampler(logits=logits, sampling_metadata=sampling_metadata)
            else:
                sampler_output, last_accepted_index, accepted_num = self.drafter.verify_and_prepare_inputs(
                    input_ids=input_ids,
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                    spec_decode_metadata=spec_decode_metadata,
                    num_prefills=num_prefills,
                    num_decodes=num_decodes,
                    chunk_next_tokens=chunk_next_tokens,
                    chunk_next_indices=chunk_next_indices,
                )
            start_6 = time.time()

            if not self.use_spec_decode:
                # Speculative decoding is not enabled.
                spec_tokens_tensor = None
            else:
                spec_tokens_tensor = self.drafter.propose(
                    num_tokens=input_ids.numel(),
                    positions=positions.clone(),
                    kv_caches=self.kv_caches,
                    attn_metadata=attn_metadata,
                    previous_hidden_states=raw_hidden_states,
                    last_accepted_index=last_accepted_index,
                    sample_indices=sample_indices,
                    sampling_metadata=sampling_metadata,
                )
            start_7 = time.time()

            # NOTE: NPU -> CPU Sync happens here.
            # Move as many CPU operations as possible before this sync point.

            # Get the valid generated tokens.
            sampled_token_ids = sampler_output.sampled_token_ids
            if self.use_spec_decode:
                sampled_token_ids = self.sampler.update_sampled_token_ids(sampled_token_ids, sampling_metadata)
            sampled_token_ids_list.append(sampled_token_ids)
            sampled_tokens = sampled_token_ids
            if self.prepare_inputs_event is not None:
                self.prepare_inputs_event.record()

            # Clear KVConnector state after all KVs are generated.
            if has_kv_transfer_group():
                get_kv_transfer_group().clear_connector_metadata()

            start_8 = time.time()
            cost_proc_reqs += start_2 - start_1
            cost_logits += start_3 - start_2
            cost_bitmask += start_4 - start_3
            cost_disc += start_5 - start_4
            cost_sampler += start_6 - start_5
            cost_drafter += start_7 - start_6
            cost_device_output += start_8 - start_7

        start_9 = time.time()
        if self.prepare_inputs_event is not None:
            self.prepare_inputs_event.wait()

        spec_token_ids = None if spec_tokens_tensor is None else self.rejection_sampler.parse_output(
            spec_tokens_tensor,
            self.input_batch.vocab_size,
        )

        sampled_token_ids_tensor = torch.cat(sampled_token_ids_list, dim=1)
        if spec_tokens_tensor is not None:
            cached_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids_tensor,
                self.input_batch.vocab_size,
            )
        else:
            cached_sampled_token_ids = sampled_token_ids_tensor.tolist()

        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() if logprobs_tensors is not None else None

        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            cached_sampled_token_ids[i].clear()
            if spec_token_ids is not None:
                spec_token_ids[i].clear()

        cost_output = time.time() - start_9
        cost = cost_upd_states + cost_proc_reqs + cost_logits + cost_bitmask + cost_sampler + cost_disc + cost_drafter + cost_device_output + cost_output
        
        global PRE_COST, COST_THRESHOLD
        if abs(cost - PRE_COST) >= COST_THRESHOLD:
            logger.info(f" ***** execute model cost:{cost:.6f}="
                        f"{cost_upd_states:.6f}+{cost_proc_reqs:.6f}+{cost_logits:.6f}+{cost_bitmask:.6f}"
                        f"+{cost_disc:.6f}+{cost_sampler:.6f}+{cost_drafter:.6f}+{cost_device_output:.6f}+{cost_output:.6f}")
        PRE_COST = cost
        finished_sending = self.finished_sending
        finished_recving = self.finished_recving
        loading_kv_failure = self.loading_kv_failure
        self.finished_sending = set()
        self.finished_recving = set()
        self.loading_kv_failure = set()
        
        prompt_logprobs_dict = {}
        if len(scheduler_output.scheduled_new_reqs) > 0 and scheduler_output.scheduled_new_reqs[0].sampling_params.prompt_logprobs:
            from vllm.v1.outputs import LogprobsTensors
            num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
            for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
                prompt_logprobs_dict[req_id] = LogprobsTensors(
                    logprob_token_ids=hidden_states.cpu(),
                    logprobs=hidden_states.cpu(),
                    selected_token_ranks=positions[:hidden_states.shape[0]].cpu()
                )
    
        # Support thinking compression by updating the current sampled token ids and length status
        if ThinkCompressDict.reasoner_early_think_stopping_enabled == 1:
            for index, cur_sampled_token_ids in enumerate(cached_sampled_token_ids):
                sampling_metadata.seq_data[index]._output_token_ids.extend(cur_sampled_token_ids)
                sampling_metadata.seq_data[index].cur_seq_length += len(cur_sampled_token_ids)
                

        output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=cached_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            loading_kv_failure=loading_kv_failure,
        )
        if profiling_is_set:
            finished_sending_raw = set()
            finished_recving_raw = set()
            for item in finished_sending:
                req_id, headers_str = item.split('|', 1)
                finished_sending_raw.add(req_id)
                output.finished_sending_headers[req_id] = json.loads(headers_str)
            for item in finished_recving:
                req_id, headers_str = item.split('|', 1)
                finished_recving_raw.add(req_id)
                output.finished_recving_headers[req_id] = json.loads(headers_str)
            output.finished_sending = finished_sending_raw
            output.finished_recving = finished_recving_raw
        return output

    def recompute_fallback(self, scheduler_output: "SchedulerOutput"):
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            if not req_data.resumed_from_preemption:
                for i in range(len(self.kv_cache_config.kv_cache_groups)):
                    if i < len(req_state.block_ids):
                        for _ in range(len(req_data.new_block_ids[i])):
                            req_state.block_ids[i].pop()
            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                continue
            for i, block_table in enumerate(self.input_batch.block_table.block_tables):
                num_blocks = len(req_data.new_block_ids[i])
                block_table.num_blocks_per_row[req_index] -= num_blocks

        remove_req_indices = []
        for new_req in scheduler_output.scheduled_new_reqs:
            req_index = self.input_batch.remove_request(new_req.req_id)
            if req_index is None:
                continue
            remove_req_indices.append(req_index)
        if len(remove_req_indices) > 0:
            self.input_batch.condense(remove_req_indices)

    @torch.inference_mode()
    def _dummy_run(self, num_tokens: int, is_capture_model: bool = False) -> torch.Tensor:
        if self.is_multimodal_model:
            input_ids, inputs_embeds = None, self.inputs_embeds[:num_tokens]
        else:
            input_ids, inputs_embeds = self.input_ids[:num_tokens], None

        # Prepare intermediate_tensors
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_num_tokens = num_tokens // get_tensor_model_parallel_world_size() if get_pp_group().world_size > 1 else num_tokens
            if self.intermediate_tensors is None:
                self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                    batch_size=intermediate_num_tokens, dtype=self.dtype, device=self.device)
            intermediate_tensors = IntermediateTensors({
                k: v[:intermediate_num_tokens] for k, v in self.intermediate_tensors.items()
            })

        positions = self.mrope_positions[:, :num_tokens] if self.uses_mrope else self.positions[:num_tokens]
        raw_hidden_states = None

        # No kv_caches: profile run
        if not self.kv_caches:
            with set_forward_context(None, self.vllm_config):
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )
                if isinstance(forward_results, tuple):
                    raw_hidden_states, hidden_states = forward_results
                else:
                    hidden_states = forward_results
                    raw_hidden_states = forward_results
                if self.use_spec_decode and get_pp_group().is_last_rank:
                    self.drafter.propose(
                        num_tokens=num_tokens,
                        positions=positions,
                        kv_caches=None,
                        attn_metadata=None,
                        previous_hidden_states=raw_hidden_states,
                        last_accepted_index=None,
                        sample_indices=None,
                    )
            return hidden_states

        # With kv_caches: dummy run for graph capture/placement
        if self.enable_torchair_graph_mode and len(self.decode_gear_list) > 1:
            self.max_batch_size = self._get_max_token_num(
                self.vllm_config.parallel_config.data_parallel_size > 1, num_tokens)
        if self.is_multimodal_model:
            fake_input = torch.zeros((self.max_batch_size, inputs_embeds.shape[1]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            if self.uses_mrope:
                fake_positions = torch.zeros((self.mrope_positions.shape[0], self.max_batch_size), dtype=torch.int64, device=self.device)
            else:
                fake_positions = torch.zeros(self.max_batch_size, dtype=torch.int64, device=self.device)
            inputs_embeds, positions = fake_input, fake_positions
        else:
            fake_input = torch.zeros(self.max_batch_size, dtype=input_ids.dtype, device=input_ids.device)
            fake_positions = torch.zeros(self.max_batch_size, dtype=input_ids.dtype, device=input_ids.device)
            input_ids, positions = fake_input, fake_positions
        self.attn_state = AscendAttentionState.DecodeOnly

        if self.is_hybrid_chunked_prefill_graph_mode:
            self.attn_state = AscendAttentionState.ChunkedPrefill

        # Build dummy attn_metadata
        attn_metadata = {}
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            builder = self.attn_metadata_builders[kv_cache_group_id]
            if not isinstance(builder, DummyAttentionMetadataBuilder):
                raise ValueError(f"{builder} does not implement DummyAttentionMetadataBuilder")
            attn_metadata_i = builder.build_dummy(num_tokens, self.max_batch_size)
            if model_extra_config.operator_opt_config.mtp_remove_redundant_kv:
                num_speculative_tokens = 0 if not self.speculative_config else self.speculative_config.num_speculative_tokens
                mtp_idx = torch.arange(1, self.max_batch_size, 1 + num_speculative_tokens, dtype=torch.int64).npu()
                new_block_table = torch.index_select(attn_metadata_i.decode.block_table, dim=0, index=mtp_idx)
                new_seq_lens = torch.index_select(attn_metadata_i.decode.seq_lens, dim=0, index=mtp_idx)
                attn_metadata_i.decode.block_table = new_block_table
                attn_metadata_i.decode.seq_lens = new_seq_lens
            if self.enable_torchair_graph_mode:
                builder.mark_static_for_attn_metadata(attn_metadata_i)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        model_kwargs = {
            "kv_caches": self.kv_caches,
            "attn_metadata": attn_metadata,
            "selected_indices": None
        }

        if self.model_config.hf_config.model_type == "hstu_inference_ranking":
            attn_metadata, input_ids, positions = self.deal_metadata(attn_metadata, input_ids, positions, True)

        with set_forward_context(attn_metadata, self.vllm_config):
            use_compile = self.enable_torchair_graph_mode
            for _ in range(self.total_step):
                if use_compile:
                    if not self.dummy_model_mark_static:
                        if isinstance(self.model, GraphCompileConfiguration):
                            self.model.mark_static_for_graph(input_ids, positions, attn_metadata, self.kv_caches)
                        else:
                            mark_static_for_graph_default(input_ids, inputs_embeds, positions, self.kv_caches)
                        self.dummy_model_mark_static = True
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs
                )
                if isinstance(forward_results, tuple):
                    raw_hidden_states, hidden_states = forward_results
                else:
                    hidden_states = forward_results
                    raw_hidden_states = forward_results
                if self.use_spec_decode and get_pp_group().is_last_rank:
                    self.drafter.prepare_dummy_input(input_ids)
                    self.drafter.propose(
                        num_tokens=input_ids.numel(),
                        positions=positions,
                        kv_caches=self.kv_caches,
                        attn_metadata=attn_metadata,
                        previous_hidden_states=raw_hidden_states,
                        last_accepted_index=None,
                        sample_indices=None,
                    )
        return hidden_states

    def profile_run(self) -> None:
        if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_role == "kv_consumer" \
            and not model_extra_config.task_config.enable_attn_ffn_disaggregation:
            hidden_states = self._dummy_run(self.max_batch_size * get_dp_group().world_size)
        elif model_extra_config.task_config.enable_attn_ffn_disaggregation:
            hidden_states = self._dummy_run(self.max_num_reqs)
        else:
            hidden_states = self._dummy_run(self.max_num_tokens)

        NPUPlatform.synchronize()
        del hidden_states
        self.encoder_cache.clear()
        gc.collect()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() if not int(os.getenv("NO_NPU_MOCK", "0")) else nullcontext() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if hasattr(self.model, "process_weights_after_loading") and callable(getattr(self.model, "process_weights_after_loading")):
                self.model.process_weights_after_loading()
            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.model_config, self.scheduler_config,
                                                  self.lora_config, self.device)
            self.drafter_list = []
            if hasattr(self, "drafter"):
                self.drafter.load_model(self.model)

            if not hasattr(self.model, "process_before_sample") or not callable(getattr(self.model, "process_before_sample")):
                self.use_process_before_sample = False

        if not int(os.getenv("NO_NPU_MOCK", "0")):
            logger.info("Loading model weights took %.4f GB", m.consumed_memory / float(2**30))

        if model_extra_config.task_config.enable_omni_placement:
            from omni.accelerators.placement.omni_placement.omni_planner import OmniPlanner
            first_k_dense_replace_names = ['num_dense_layers', 'first_k_dense_replace']
            first_k_dense_replace = get_attr_by_names(self.model.config, first_k_dense_replace_names, 0)
            param_dict = dict(self.model.named_parameters())
            self.planner = OmniPlanner()
            self.planner.init_dram_weights(param_dict, first_k_dense_replace=first_k_dense_replace)

    def omni_placement_pause(self) -> None:
        if model_extra_config.task_config.enable_omni_placement:
            self.planner.placement_pause()

    def omni_placement_resume(self) -> None:
        if model_extra_config.task_config.enable_omni_placement:
            self.planner.placement_resume()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_caches: Dict[str, torch.Tensor] = {}
        cpu_caches: Dict[str, torch.Tensor] = {}
        self.kv_cache_config = kv_cache_config
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_num_blocks_per_req*self.block_size,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=is_pin_memory_available(),
            vocab_size=self.model_config.get_vocab_size(),
            block_size=self.cache_config.block_size
        )
        self.input_batch.token_ids_cpu_tensor = torch.zeros(
            (self.max_num_reqs, self.model_config.max_model_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=False,
        )
        self.input_batch.token_ids_cpu = self.input_batch.token_ids_cpu_tensor.numpy()
        self.initialize_attn_backend(kv_cache_config)
        preemption_mode = self.vllm_config.scheduler_config.preemption_mode

        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                if tensor_config.size % kv_cache_spec.page_size_bytes != 0:
                    raise RuntimeError("tensor_config.size must be divisible by kv_cache_spec.page_size_bytes")
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    # adapted for Pangu 72Bv2
                    hf_config = self.vllm_config.model_config.hf_config
                    v_channels = getattr(hf_config, "v_channels", None)
                    if v_channels is None:
                        kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                            num_blocks, 
                            kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads, 
                            kv_cache_spec.head_size)
                    else:
                        kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                            num_blocks, 
                            kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads, 
                            kv_cache_spec.head_size,
                            v_channels)
                
                    kv_caches[layer_name] = self.attn_backends[i].init_kv_cache_each_layer(
                        kv_cache_shape, 
                        self.dtype,
                        self.device,
                        self.model_config,
                        self.enable_torchair_graph_mode)
                        
                    if preemption_mode and preemption_mode == "swap":
                        cpu_num_blocks = int(self.vllm_config.cache_config.swap_space_bytes //
                                          kv_cache_spec.page_size_bytes // len(kv_cache_config.tensors))
                        cpu_kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                            cpu_num_blocks, kv_cache_spec.block_size,
                            kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                        cpu_caches[layer_name] = self.attn_backends[i].init_kv_cache_each_layer(cpu_kv_cache_shape, self.dtype,
                                                                                           "cpu",
                                                                                           self.model_config,
                                                                                           self.enable_torchair_graph_mode)
                else:
                    raise ValueError("Unknown KV cache spec type.")

        if preemption_mode and preemption_mode == "swap":
            self.cache_engine = CacheEngine(self.attn_backends, self.kv_cache_config, gpu_cache=kv_caches, cpu_cache=cpu_caches)

        if not int(os.getenv("NO_NPU_MOCK", "0")):
            bind_kv_cache(
                kv_caches,
                self.vllm_config.compilation_config.static_forward_context,
                self.kv_caches)

        if has_kv_transfer_group():
            self.kv_caches_dict = kv_caches
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def initialize_omni_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        from omni.accelerators.cache.omni_cache import create_omni_cache
        if self.vllm_config.kv_transfer_config is None:
            raise NotImplementedError("Currently only support PD disaggregation, but KV transfer config is None.")
        if len(kv_cache_config.kv_cache_groups) > 1 and self.vllm_config.kv_transfer_config.kv_role != "kv_consumer":
            raise RuntimeError(f"Only support single KV cache group in Prefill nodes, but got {len(kv_cache_config.kv_cache_groups)}.")

        self.kv_cache_config = kv_cache_config
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_num_blocks_per_req*self.block_size,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=is_pin_memory_available(),
            vocab_size=self.model_config.get_vocab_size(),
            block_size=self.cache_config.block_size
        )
        self.input_batch.token_ids_cpu_tensor = torch.zeros(
            (self.max_num_reqs, self.model_config.max_model_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=False,
        )
        self.input_batch.token_ids_cpu = self.input_batch.token_ids_cpu_tensor.numpy()
        self.initialize_attn_backend(kv_cache_config)

        omni_cache = create_omni_cache(
            kv_cache_config=self.kv_cache_config,
            vllm_config=self.vllm_config,
            runner=self,
        )

        get_kv_transfer_group().register_kv_caches(
            omni_cache.MEMMAP_PATH,
            omni_cache.dtype,
            block_len_dtype=omni_cache.block_len_dtype,
            omni_cache=omni_cache
        )

        self.omni_cache = omni_cache

    def unregister_kv_caches(self):
        if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_connector == "AscendHcclConnectorV1":
            if has_kv_transfer_group():
                logger.info(f"unregister_kv_caches")
                get_kv_transfer_group().unregister_kv_caches()

    def reregister_kv_caches(self):
        if self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_connector == "AscendHcclConnectorV1":
            if has_kv_transfer_group():
                logger.info(f"reregister_kv_caches")
                get_kv_transfer_group().register_kv_caches(self.kv_caches_dict)

    def capture_model(self) -> None:
        if self.enable_torchair_graph_mode:
            decode_gear_list = self.decode_gear_list
            graph_num = len(decode_gear_list)
            use_spec_decode = self.vllm_config.speculative_config is not None
            base_time = 4
            min_time = base_time * graph_num
            max_time = 2 * base_time * graph_num
            mtp_time_rate = 1.5
            if use_spec_decode:
                min_time *= mtp_time_rate
                max_time *= mtp_time_rate

            logger.info(f"The current directory is {os.getcwd()}")
            logger.info(
                "Capturing torchair graph, this usually takes %.1f~%.1f mins.",
                min_time, max_time)
            # Trigger torchair graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            for idx, num_tokens in enumerate(
                    reversed(decode_gear_list)):
                self._dummy_run(num_tokens, True)
                logger.info("Batchsize %d is compiled successfully: %d/%d.",
                            num_tokens, idx + 1, graph_num)
        else:
            logger.warning(
                "Skipping NPU graph capture. Please add "
                "-O %s to use NPU graphs.", CompilationLevel.PIECEWISE)

        if model_extra_config.task_config.enable_omni_placement:
            self.planner.start_dynamic_optimize_expert_load_balance()

    def _get_closest_gear(self, max_num_token):
        for gear in self.decode_gear_list:
            if gear >= max_num_token:
                return gear
        raise ValueError(f"decode input batch size {max_num_token} exceeds maximum gear {max(self.decode_gear_list)}.")

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        from omni.accelerators.cache import get_omni_hybrid_kv_cache_spec, check_omni_attn_cmd_arg
        kv_transfer_config = self.vllm_config.kv_transfer_config
        is_kv_consumer = kv_transfer_config is None or kv_transfer_config.kv_role == "kv_consumer"
        enable_omni_attn = check_omni_attn_cmd_arg(self.vllm_config.additional_config)
        if int(os.getenv("NO_NPU_MOCK", "0")):
            kv_cache_spec: dict[str, KVCacheSpec] = {}
            block_size = self.vllm_config.cache_config.block_size
            use_mla = self.vllm_config.model_config.use_mla
            kv_cache_spec["mock.0"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=16,
                dtype=torch.bfloat16,
                use_mla=use_mla
            )
            return kv_cache_spec
        elif enable_omni_attn and is_kv_consumer:
            return get_omni_hybrid_kv_cache_spec(self)
        return self._get_kv_cache_spec_dsa()

    def _get_kv_cache_spec_dsa(self):
        from omni.accelerators.cache.omni_cache import DecodeOmniCache
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            # if use omni_cache, set head_size being k_indexer size, to get more blocks
            if (model_extra_config.operator_opt_config.enable_dsa and
                    model_extra_config.operator_opt_config.use_omni_cache and
                    self.vllm_config.kv_transfer_config.kv_role == "kv_consumer"):
                head_size = sum([512, 64, 128])
                self.vllm_config.cache_config.num_gpu_blocks_override = \
                    DecodeOmniCache.calc_cache_shape_for_decode(
                        num_layers=len(layers),
                        block_size=block_size,
                        head_size=head_size,
                        dtype=self.kv_cache_dtype)[1]
                attn_module.head_size = 128 + 2 # indexer and scale
            # TODO: Support other attention modules, e.g., cross-attention
            if attn_module.attn_type == AttentionType.DECODER:
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=use_mla)
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def _get_max_token_num(self, is_enable_dp, num_tokens):
        if is_enable_dp:
            local_batch_tensor = torch.tensor([num_tokens], dtype=torch.int64, device='cpu')
            dist.all_reduce(local_batch_tensor, group=get_dp_group().cpu_group, op=dist.ReduceOp.MAX)
            global_batch_size = local_batch_tensor.item()
            return self._get_closest_gear(global_batch_size)
        return self._get_closest_gear(num_tokens)
