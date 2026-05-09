#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/v1/spec_decode/eagle.py
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
#

import torch
import torch_npu
import torch.nn as nn
import copy
from typing import Optional, List, Dict

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.distributed import get_tp_group

from omni.adaptors.vllm.forward_context import set_forward_context
from omni.layers.sampler import random_choice
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.models.config_loader.loader import model_extra_config

def mark_static_for_graph_default(
        input_ids,
        previous_hidden_states: Optional[torch.Tensor] = None,
    ):
    torch._dynamo.mark_static(input_ids)
    if isinstance(previous_hidden_states, List):
        # for eagle3
        for item in previous_hidden_states:
            torch._dynamo.mark_static(item)
    elif previous_hidden_states is not None:
        # for eagle/mtp
        torch._dynamo.mark_static(previous_hidden_states)

class PostDrafter(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)
        self.drafter_list = []
        self.method = self.vllm_config.speculative_config.method
        self.enable_adaptive = self.vllm_config.speculative_config.enable_adaptive
        self.mark_static = False
        self.rejection_sampler = runner.rejection_sampler
        self.use_rejection_sampler = runner.use_rejection_sampler
        self.topk = runner.topk

        # eagle proposer set dtype as int32, while we need int64
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=device)
        self.positions = None
        self.hidden_states = None
        self.arange = torch.arange(runner.decode_max_num_tokens, device=device)
        self.dsa_stream = torch_npu.npu.Stream()
        self.main_sampler = runner.rejection_sampler.main_sampler
        # TODO check model type
        if self.method not in ('deepseek_mtp', 'eagle', 'eagle3', 'pangu_ultra_moe_mtp', 'qwen3_mtp', 'pangu_moe_v2_mtp', 'glm4_moe_mtp'):
            raise ValueError(f"Speculative method should be one of ('deepseek_mtp', 'eagle', 'eagle3', 'pangu_ultra_moe_mtp', 'qwen3_mtp', 'pangu_moe_v2_mtp', 'glm4_moe_mtp'), while get {self.method}.")

        self.n_predictor = self.vllm_config.model_config.hf_config.num_nextn_predict_layers if self.method == 'deepseek_mtp' else 1
        self.is_autogressive = self.speculative_config.num_speculative_tokens > self.n_predictor

        self.minus_one = -torch.ones(1, device=device)
        self.device = device

        self.is_hybrid = self.vllm_config.kv_transfer_config is None

    def _get_compact_sample_indices(
            self,
            sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        compact_indices = getattr(self.runner, "spec_input_token_indices", None)
        if compact_indices is None:
            return sample_indices
        if compact_indices.numel() != sample_indices.numel():
            return sample_indices
        return compact_indices


    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        self.model = get_model(vllm_config=self.vllm_config, model_config=draft_model_config)
        self.model.set_share_weight(target_model)

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)

        self.attn_layer_names = sorted(draft_attn_layer_names)

    def verify_and_prepare_inputs(self,
                                  input_ids,
                                  logits,
                                  sampling_metadata,
                                  spec_decode_metadata,
                                  num_prefills,
                                  num_decodes,
                                  chunk_next_tokens: Optional[torch.Tensor] = None,
                                  chunk_next_indices: Optional[torch.Tensor] = None,
    ):
        sampler_output, forward_tokens, last_accepted_index, accepted_num = self.rejection_sampler(
            metadata=spec_decode_metadata,
            draft_probs=None,
            logits=logits,
            input_ids=input_ids,
            sampling_metadata=sampling_metadata,
        )

        self.input_ids[:input_ids.numel() - 1] = input_ids[1:]
        compact_logits_indices = self._get_compact_sample_indices(
            spec_decode_metadata.logits_indices)
        self.input_ids[compact_logits_indices[last_accepted_index]] = forward_tokens

        if chunk_next_indices is not None:
            self.input_ids[chunk_next_indices] = chunk_next_tokens

        return sampler_output, last_accepted_index, accepted_num

    def prepare_dummy_input(self, input_ids):
        self.input_ids[:input_ids.numel() - 1] = input_ids[1:]

    def _simple_advance_step(
            self,
            positions,
            attn_metadata,
            block_size,
            model_layer,
    ):
        if isinstance(attn_metadata, Dict):
            # suppose that types of attn in layers of drafter is same, and share one attn_metadata
            attn_metadata = attn_metadata[self.attn_layer_names[0]]

        pad_mask = attn_metadata.slot_mapping == self.minus_one
        positions[:] = torch.where(pad_mask, positions, positions + 1)

        attn_metadata.advance_step(attn_metadata, positions, block_size, pad_mask, model_layer)

    def _should_convert_hybrid_prefill_to_decode(self, attn_state, is_dummy) -> bool:
        if is_dummy:
            return False
        if not self.is_hybrid:
            return False
        if self.speculative_config.num_speculative_tokens <= 1:
            return False

        if attn_state == AscendAttentionState.PrefillNoCache or attn_state == AscendAttentionState.ChunkedPrefill:
            return True
        
        return False
    
    def _is_mla_metadata(self, metadata) -> bool:
        if metadata is None:
            return False
        # avoid circular import by importing AscendMLAMetadata here
        from omni.layers.attention.backend.mla import AscendMLAMetadata

        return isinstance(metadata, AscendMLAMetadata)

    def _is_mixed_mla_metadata(self, metadata) -> bool:
        return (self._is_mla_metadata(metadata)
                and getattr(metadata, "decode", None) is not None
                and getattr(metadata, "prefill", None) is not None
                and getattr(metadata, "num_decodes", 0) > 0
                and getattr(metadata, "num_prefills", 0) > 0)

    def _build_decode_metadata_from_mixed_metadata(self, attn_metadata):
        if isinstance(attn_metadata, Dict):
            converted_metadata = {}
            memo = {}
            for layer_name, attn_metadata_i in attn_metadata.items():
                metadata_key = id(attn_metadata_i)
                if metadata_key not in memo:
                    memo[metadata_key] = self._build_decode_metadata_from_mixed_metadata(attn_metadata_i)
                converted_metadata[layer_name] = memo[metadata_key]
            return converted_metadata

        decode_attn_metadata = copy.copy(attn_metadata)
        decode_attn_metadata.prefill = None
        decode_attn_metadata.slot_mapping = attn_metadata.decode.slot_mapping
        decode_attn_metadata.num_prefills = 0
        decode_attn_metadata.num_actual_tokens = attn_metadata.decode.num_padded_tokens
        decode_attn_metadata.num_input_tokens = attn_metadata.decode.num_padded_tokens
        decode_attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        return decode_attn_metadata
    
    def _get_num_prefills_and_decodes(self, first_attn_metadata):
        num_prefills = getattr(first_attn_metadata, "num_prefills", None)
        num_decodes = getattr(first_attn_metadata, "num_decodes", None)

        # Non mla metadata may not have num_prefills/num_decodes attributes
        if num_prefills is None or num_decodes is None:
            if getattr(self.runner, "attn_metadata_builders", None):
                builder = self.runner.attn_metadata_builders[0]
                num_prefills = getattr(builder, "_num_prefills")
                num_decodes = getattr(builder, "_num_decodes")

        if num_prefills is None:
            logger.warning("Cannot get num_prefills from metadata or builder, default to 0.")
            num_prefills = 0

        if num_decodes is None:
            logger.warning("Cannot get num_decodes from metadata or builder, default to 0.")
            num_decodes = 0

        return num_prefills, num_decodes

    def _get_prefill_req_indices_from_metadata(
            self,
            attn_metadata,
            sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        first_attn_metadata = attn_metadata
        if isinstance(attn_metadata, Dict):
            first_attn_metadata = attn_metadata[self.attn_layer_names[0]]

        num_prefills, num_decodes = self._get_num_prefills_and_decodes(first_attn_metadata)

        # if counts are unavailable, assume all sampled requests are prefill
        if num_prefills <= 0:
            return torch.arange(
                sample_indices.numel(),
                dtype=torch.long,
                device=sample_indices.device,
            )
        
        num_total_reqs = num_prefills + num_decodes
        req_start = num_total_reqs - num_prefills

        return torch.arange(
            req_start,
            num_total_reqs,
            dtype=torch.long,
            device=sample_indices.device,
        )
    
    def _get_hybrid_decode_graph_batch_size(self, actual_batch_size: int) -> int:
        if not getattr(self.runner, "enable_torchair_graph_mode", False):
            return self.runner.max_batch_size
        
        return self.runner._get_max_token_num(
            self.vllm_config.parallel_config.data_parallel_size > 1,
            actual_batch_size,
        )
    
    def _pad_first_dim(self, tensor: torch.Tensor, target_size: int, pad_value=0) -> torch.Tensor:
        padding_size = target_size - tensor.shape[0]
        if padding_size <= 0:
            return tensor

        padding = torch.full(
            (padding_size,) + tensor.shape[1:],
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device
        )
        return torch.cat([tensor, padding], dim=0)

    def _build_decode_metadata_from_prefill_metadata(
            self,
            attn_metadata,
            continuation_positions: torch.Tensor,
            prefill_req_indices: torch.Tensor,
            sample_indices: torch.Tensor,
    ):
        """
        Build a decode-style attention metadata object for the prefill requests
        that should continue speculative decoding in hybrid mode.
        """

        if isinstance(attn_metadata, Dict):
            converted_metadata = {}
            memo = {}
            for layer_name, attn_metadata_i in attn_metadata.items():
                metadata_key = id(attn_metadata_i)
                if metadata_key not in memo:
                    memo[metadata_key] = self._build_decode_metadata_from_prefill_metadata(
                        attn_metadata_i,
                        continuation_positions,
                        prefill_req_indices,
                        sample_indices
                    )
                converted_metadata[layer_name] = memo[metadata_key]
            return converted_metadata
        
        num_prefills = prefill_req_indices.numel()
        graph_batch_size = self._get_hybrid_decode_graph_batch_size(num_prefills)
        prefill_sample_indices = sample_indices[prefill_req_indices].contiguous()

        # shallow copy
        decode_attn_metadata = copy.copy(attn_metadata)
        decode_attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        decode_attn_metadata.num_actual_tokens = num_prefills
        
        first_layer = next(self.model.model.layers.children())
        first_self_attn = (
            first_layer.self_attn[0]
            if isinstance(first_layer.self_attn, torch.nn.ModuleList)
            else first_layer.self_attn
        )
        rotary_emb = first_self_attn.rotary_emb

        # MLA metadata carries separate prefill/decode sub-structures.
        # For the hybrid continuation path, we must explicitly build a decode
        if self._is_mla_metadata(decode_attn_metadata):
            prefill_metadata = decode_attn_metadata.prefill
            if prefill_metadata is None:
                raise RuntimeError(
                    "MLA hybrid prefill-to-decode conversion requires prefill metadata but got None."
                )
            
            from omni.layers.attention.backend.mla import AscendMLADecodeMetadata

            prefill_block_table = prefill_metadata.block_table
            if prefill_block_table is None:
                decode_block_table = None
            elif prefill_block_table.shape[0] == num_prefills:
                decode_block_table = prefill_block_table.clone()
            else:
                decode_block_table = prefill_block_table[
                    prefill_req_indices
                ].contiguous()

            if decode_block_table is not None:
                if getattr(self.runner, "enable_torchair_graph_mode", False):
                    decode_block_table = (
                        self.runner.attn_metadata_builders[0]
                        ._get_graph_runner_block_tables(
                            num_prefills,
                            decode_block_table,
                            graph_batch_size,
                        )
                    )
                else:
                    # Match normal DecodeOnly padding in single-op mode.
                    decode_block_table = self._pad_first_dim(
                        decode_block_table,
                        graph_batch_size,
                        pad_value=0,
                    ) 

            mc2_mask = None
            if getattr(self.runner, "enable_torchair_graph_mode", False):
                builder = self.runner.attn_metadata_builders[0]
                builder.generate_activate_mask(num_prefills, graph_batch_size)
                mc2_mask = builder.mc2_mask
            
            best_topk = None
            if model_extra_config.operator_opt_config.best_ep:
                best_topk = self.runner.attn_metadata_builders[0].cal_best_topk(
                    graph_batch_size
                )

            decode_metadata = AscendMLADecodeMetadata(
                input_positions=continuation_positions,
                block_table=decode_block_table,
                seq_lens=(continuation_positions + 1),
                mc2_mask=mc2_mask,
                cos=None,
                sin=None,
                best_topk=best_topk,
            )

            # Rotary embeddings must be recomputed for the new decode positions.
            cos, sin = rotary_emb.get_cos_sin(decode_metadata.input_positions)
            decode_metadata.cos = cos
            decode_metadata.sin = sin

            decode_attn_metadata.decode = decode_metadata
            decode_attn_metadata.prefill = None
            decode_attn_metadata.num_input_tokens = graph_batch_size
            decode_attn_metadata.num_decodes = num_prefills
            decode_attn_metadata.num_decode_tokens = num_prefills
            decode_attn_metadata.num_prefills = 0

            if decode_attn_metadata.slot_mapping is not None:
                decode_attn_metadata.slot_mapping = decode_attn_metadata.slot_mapping[
                    prefill_sample_indices
                ].contiguous()
                decode_attn_metadata.slot_mapping = self._pad_first_dim(
                    decode_attn_metadata.slot_mapping,
                    graph_batch_size,
                    pad_value=-1,
                )

            return decode_attn_metadata
        
        # Non-MLA metadata
        decode_attn_metadata.num_actual_tokens = num_prefills

        decode_attn_metadata.block_tables = decode_attn_metadata.block_tables[
            prefill_req_indices
        ].contiguous()
        decode_attn_metadata.block_tables = self._pad_first_dim(
            decode_attn_metadata.block_tables,
            graph_batch_size,
            pad_value=-1,
        )

        if getattr(self.runner, "enable_torchair_graph_mode", False):
            decode_attn_metadata.block_tables = (
                self.runner.attn_metadata_builders[0]
                ._get_graph_runner_block_tables(
                    num_prefills,
                    decode_attn_metadata.block_tables,
                )
            )
        
        decode_attn_metadata.query_lens = torch.ones(
            graph_batch_size, 
            dtype=decode_attn_metadata.query_lens.dtype, 
            device=decode_attn_metadata.query_lens.device,
        )

        decode_attn_metadata.query_lens_list = [1] * graph_batch_size

        decode_attn_metadata.seq_lens = (continuation_positions + 1).to(
            decode_attn_metadata.seq_lens.dtype
        )

        decode_attn_metadata.seq_lens_list = (
            continuation_positions + 1
        ).tolist()

        decode_attn_metadata.max_query_len = 1
     
        if decode_attn_metadata.slot_mapping is not None:
            decode_attn_metadata.slot_mapping = decode_attn_metadata.slot_mapping[
                prefill_sample_indices
            ].contiguous()
            decode_attn_metadata.slot_mapping = self._pad_first_dim(
                decode_attn_metadata.slot_mapping,
                graph_batch_size,
                pad_value=0,
            )

        if decode_attn_metadata.slot_indices is not None:
            block_size = self.vllm_config.cache_config.block_size
            decode_attn_metadata.slot_indices = torch.stack(
                [
                    decode_attn_metadata.slot_mapping // block_size,
                    decode_attn_metadata.slot_mapping % block_size,
                ],
                dim=1,
            )

        decode_attn_metadata.is_only_prefill = False

        decode_attn_metadata.attn_state = AscendAttentionState.DecodeOnly
            
        cos, sin = rotary_emb.get_cos_sin(continuation_positions)
        decode_attn_metadata.cos = cos
        decode_attn_metadata.sin = sin

        decode_attn_metadata.kv_index = None

        if getattr(self.runner, "enable_torchair_graph_mode", False):
            decode_attn_metadata.mc2_mask = torch.zeros(
                graph_batch_size,
                dtype=torch.bool,
                device=continuation_positions.device,
            )
            decode_attn_metadata.mc2_mask[:num_prefills] = True
        else:
            decode_attn_metadata.mc2_mask = None

        return decode_attn_metadata

    def _convert_prefill_step_to_decode_step(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            attn_metadata,
            previous_hidden_states,
            last_accepted_index: torch.Tensor,
            sample_indices: torch.Tensor,
    ):
        """
        Convert the current hybrid prefill drafting state into a compact
        decode-style continuation state after the first prefill spec token
        has been sampled.
        """
        prefill_req_indices = self._get_prefill_req_indices_from_metadata(attn_metadata, sample_indices)
        prefill_batch_size = prefill_req_indices.numel()

        if prefill_batch_size == 0:
            logger.warning("No prefill requests found for hybrid continuation.")
            return (
                input_ids,
                positions,
                attn_metadata,
                previous_hidden_states,
                last_accepted_index,
                sample_indices,
            )
        
        graph_batch_size = self._get_hybrid_decode_graph_batch_size(prefill_batch_size)
        
        prefill_sample_indices = sample_indices[prefill_req_indices].contiguous()
        compact_sample_indices = self._get_compact_sample_indices(sample_indices)
        compact_prefill_sample_indices = compact_sample_indices[
            prefill_req_indices].contiguous()

        continuation_input_ids = self.input_ids[:graph_batch_size]
        continuation_input_ids[:prefill_batch_size].copy_(
            input_ids[compact_prefill_sample_indices])
        if graph_batch_size > prefill_batch_size:
            continuation_input_ids[prefill_batch_size:graph_batch_size].fill_(0)

        # Get continuation positions for the prefill requests
        continuation_positions = positions[prefill_sample_indices].contiguous()

        continuation_positions = self._pad_first_dim(
            continuation_positions, graph_batch_size, pad_value=0
        )

        # Get modified attention metadata for the continuation step
        continuation_attn_metadata = self._build_decode_metadata_from_prefill_metadata(
            attn_metadata=attn_metadata,
            continuation_positions=continuation_positions,
            prefill_req_indices=prefill_req_indices,
            sample_indices=sample_indices,
        )

        # Get hidden states for the continuation step 
        def compact_hidden_states(hidden_states_i):
            if hidden_states_i is None or not torch.is_tensor(hidden_states_i) or hidden_states_i.dim() == 0:
                return hidden_states_i

            hidden_tokens = hidden_states_i.shape[0]
            if hidden_tokens == input_ids.shape[0]:
                hidden_states_i = hidden_states_i[
                    compact_prefill_sample_indices].contiguous()
            elif hidden_states_i.shape[0] == sample_indices.numel():
                hidden_states_i = hidden_states_i[prefill_req_indices].contiguous()
            elif (prefill_sample_indices.numel() > 0
                  and hidden_tokens > int(prefill_sample_indices.max().item())):
                hidden_states_i = hidden_states_i[prefill_sample_indices].contiguous()
            else:
                logger.warning(
                    "Unexpected hidden state shape %s when converting hybrid prefill to decode; keep it unchanged.",
                    tuple(hidden_states_i.shape),
                )
                return hidden_states_i

            return self._pad_first_dim(
                hidden_states_i, graph_batch_size, pad_value=0
            )
        
        if isinstance(previous_hidden_states, tuple):
            continuation_hidden_states = tuple(
                compact_hidden_states(x)
                for x in previous_hidden_states
            )
        elif isinstance(previous_hidden_states, list):
            continuation_hidden_states = [
                compact_hidden_states(x)
                for x in previous_hidden_states
            ]
        else:
            continuation_hidden_states = compact_hidden_states(previous_hidden_states)

        continuation_last_accepted_index = torch.arange(
            prefill_batch_size,
            dtype=last_accepted_index.dtype,
            device=last_accepted_index.device,
        )

        continuation_sample_indices = torch.arange(
            prefill_batch_size,
            dtype=sample_indices.dtype,
            device=sample_indices.device,
        )

        from vllm import forward_context as vllm_forward_context
        if getattr(vllm_forward_context, "_forward_context", None) is not None:
            vllm_forward_context._forward_context.attn_metadata = continuation_attn_metadata

        return (
            continuation_input_ids,
            continuation_positions,
            continuation_attn_metadata,
            continuation_hidden_states,
            continuation_last_accepted_index,
            continuation_sample_indices,
        )

    @torch.inference_mode()
    def propose(self,
                num_tokens,
                positions,
                kv_caches,
                attn_metadata,
                previous_hidden_states,
                last_accepted_index,
                sample_indices,
                sampling_metadata=None,
                **kwargs,
    ):
        input_ids = self.input_ids[:num_tokens]
        if self.method == 'eagle3':
            previous_hidden_states = self.model.combine_hidden_states(previous_hidden_states)
            if previous_hidden_states.shape[0] < input_ids.shape[0]:
                previous_hidden_states = get_tp_group().all_gather(previous_hidden_states, dim=0)

        if model_extra_config.operator_opt_config.use_omni_cache:
            kv_cache_flag = attn_metadata
        else:
            kv_cache_flag = kv_caches
            
        if kv_cache_flag is None:
            with set_forward_context(None, self.vllm_config):
                for i in range(self.speculative_config.num_speculative_tokens):
                    input_ids = self.input_ids[:num_tokens]
                    self.model(
                        input_ids=input_ids,
                        positions=positions,
                        kv_caches=None,
                        attn_metadata=None,
                        previous_hidden_states=previous_hidden_states,
                        mtp_layer_idx=i,
                    )
                return None
        else:
            first_attn_metadate = attn_metadata
            if isinstance(attn_metadata, dict):
                 first_attn_metadate = attn_metadata[self.attn_layer_names[0]]
            attn_state = first_attn_metadate.attn_state
            mixed_original_num_reqs = None

            if self._is_mixed_mla_metadata(first_attn_metadate):
                mixed_original_num_reqs = first_attn_metadate.num_decodes + first_attn_metadate.num_prefills
                num_decodes = first_attn_metadate.num_decodes
                num_decode_tokens = first_attn_metadate.decode.num_decode_tokens
                decode_padded_tokens = first_attn_metadate.decode.num_padded_tokens
                compact_input_ids = self.input_ids[:decode_padded_tokens]
                compact_input_ids[:num_decode_tokens].copy_(input_ids[:num_decode_tokens])
                if decode_padded_tokens > num_decode_tokens:
                    compact_input_ids[num_decode_tokens:decode_padded_tokens].fill_(0)
                input_ids = compact_input_ids
                positions = self._pad_first_dim(positions[:num_decode_tokens], decode_padded_tokens, pad_value=0)

                def compact_decode_hidden_states(hidden_states_i):
                    if hidden_states_i is None or not torch.is_tensor(hidden_states_i) or hidden_states_i.dim() == 0:
                        return hidden_states_i
                    return self._pad_first_dim(hidden_states_i[:num_decode_tokens], decode_padded_tokens, pad_value=0)

                if isinstance(previous_hidden_states, tuple):
                    previous_hidden_states = tuple(compact_decode_hidden_states(x) for x in previous_hidden_states)
                elif isinstance(previous_hidden_states, list):
                    previous_hidden_states = [compact_decode_hidden_states(x) for x in previous_hidden_states]
                else:
                    previous_hidden_states = compact_decode_hidden_states(previous_hidden_states)

                attn_metadata = self._build_decode_metadata_from_mixed_metadata(attn_metadata)
                if last_accepted_index is not None:
                    last_accepted_index = last_accepted_index[:num_decodes].contiguous()
                if sample_indices is not None:
                    sample_indices = sample_indices[:num_decodes].contiguous()
                num_tokens = input_ids.numel()
                first_attn_metadate = attn_metadata
                if isinstance(attn_metadata, dict):
                    first_attn_metadate = attn_metadata[self.attn_layer_names[0]]
                attn_state = first_attn_metadate.attn_state

            draft_forward_tokens_list = []

            if self.runner.enable_torchair_graph_mode and attn_state == AscendAttentionState.DecodeOnly \
                and (not self.mark_static) and not getattr(first_attn_metadate, "force_eager", False):
                from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration
                if isinstance(self.model, GraphCompileConfiguration):
                    self.model.mark_static_for_graph()
                mark_static_for_graph_default(input_ids, previous_hidden_states)
                self.mark_static = True

            with set_forward_context(attn_metadata, self.vllm_config):
                is_dummy = (last_accepted_index is None) or (sample_indices is None)
                
                should_convert_prefill_to_decode = self._should_convert_hybrid_prefill_to_decode(
                    attn_state,
                    is_dummy
                )

                # avoid prefill entering the adaptive
                use_decode_adaptive = self.enable_adaptive and (
                    attn_state == AscendAttentionState.DecodeOnly
                )

                if not is_dummy:
                    batch_size = last_accepted_index.numel()
                    if use_decode_adaptive:
                        drafter_logits_range = torch.empty((
                            self.speculative_config.num_speculative_tokens, batch_size), device=self.device)
                        min_acc = self.speculative_config.min_num_speculative_tokens
                        spec_budget = self.runner.max_batch_size - self.runner.max_num_reqs - batch_size * min_acc
                for i in range(self.speculative_config.num_speculative_tokens):
                    if i >= self.n_predictor:
                        if attn_state == AscendAttentionState.DecodeOnly:
                            self._simple_advance_step(positions, attn_metadata, self.vllm_config.cache_config.block_size, next(self.model.model.layers.children()))
                        else:
                            break
                    input_ids = self.input_ids[:num_tokens]
                    drafter_logits, next_hidden_states = self.model(
                        input_ids=input_ids,
                        positions=positions,
                        kv_caches=kv_caches,
                        attn_metadata=attn_metadata,
                        previous_hidden_states=previous_hidden_states,
                        selected_indices=None if attn_state == AscendAttentionState.DecodeOnly else sample_indices,
                        mtp_layer_idx=i,
                    )
                    # TODO use one eagle/mtp as autoregressive to predict more than one token
                    if not is_dummy:
                        if drafter_logits is None:
                            # keep same with computation in model runner
                            if next_hidden_states.shape[0] == sample_indices.shape[0]:
                                drafter_logits = self.model.compute_logits(next_hidden_states, None)
                            else:
                                drafter_logits = self.model.compute_logits(next_hidden_states[sample_indices], None)

                        if self.use_rejection_sampler:
                            output = self.main_sampler.apply_sampling_params(
                                drafter_logits[last_accepted_index], sampling_metadata, None, input_ids[last_accepted_index])
                            if isinstance(output, tuple):
                                mtp_probs, mtp_ids = output
                            else:
                                all_sampled_tokens = output.argmax(dim=-1)
                                mtp_probs = torch.zeros_like(output)
                                mtp_probs[self.arange[:batch_size], all_sampled_tokens] = 1

                            if self.topk > 0:
                                mtp_topk_token_probs = mtp_probs[:, -self.topk:]
                                mtp_topk_token_ids = mtp_ids[:, -self.topk:]
                                mtp_selected_indices = random_choice(mtp_topk_token_probs, {}, self.dsa_stream)
                                self.rejection_sampler.main_sampler.prob_cache.update_sparse_rejection_sampler(mtp_topk_token_ids, mtp_topk_token_probs, mtp_selected_indices, i)
                                draft_forward_tokens = mtp_topk_token_ids[self.arange[:batch_size], mtp_selected_indices].view(-1)
                            else:
                                draft_forward_tokens = random_choice(mtp_probs, {}, self.dsa_stream)
                                self.rejection_sampler.main_sampler.prob_cache.update_sparse_rejection_sampler(None, mtp_probs, None, i)
                            draft_forward_tokens_list.append(draft_forward_tokens)
                        else:
                            draft_forward_tokens = drafter_logits[last_accepted_index].argmax(dim=-1)
                            draft_forward_tokens_list.append(draft_forward_tokens)

                        # apply adaptive speculative decoding
                        if use_decode_adaptive:
                            drafter_logits_range_i = (torch.max(drafter_logits[last_accepted_index], dim=-1).values -
                                torch.min(drafter_logits[last_accepted_index], dim=-1).values)
                            if i < min_acc:
                                pass
                            if i == min_acc:
                                drafter_logits_range[i] = drafter_logits_range_i
                            else:
                                drafter_logits_range[i] = torch.min(drafter_logits_range[i - 1], drafter_logits_range_i)
                    if i == self.speculative_config.num_speculative_tokens - 1:
                        break
                    self.input_ids[:num_tokens] = torch.roll(input_ids, -1, -1)
                    if not is_dummy:
                        if attn_state == AscendAttentionState.DecodeOnly:
                            input_ids[last_accepted_index] = draft_forward_tokens
                        else: # prefill
                            compact_sample_indices = self._get_compact_sample_indices(
                                sample_indices)
                            input_ids[compact_sample_indices] = draft_forward_tokens
                    if not model_extra_config.operator_opt_config.skip_mtp_hidden_states:
                        previous_hidden_states = next_hidden_states

                    if(
                        should_convert_prefill_to_decode
                        and i == self.n_predictor - 1
                    ):
                        input_ids, positions, attn_metadata, previous_hidden_states, last_accepted_index, sample_indices = \
                            self._convert_prefill_step_to_decode_step(
                                input_ids=input_ids,
                                positions=positions,
                                attn_metadata=attn_metadata,
                                previous_hidden_states=previous_hidden_states,
                                last_accepted_index=last_accepted_index,
                                sample_indices=sample_indices,
                            )
                        
                        attn_state = AscendAttentionState.DecodeOnly
                        batch_size = last_accepted_index.numel()
                        num_tokens = input_ids.numel()

            if is_dummy:
                return None
            else:
                draft_forward_tokens_list = torch.stack(draft_forward_tokens_list, dim=0)
                def pad_mixed_draft_tokens(draft_tokens):
                    if mixed_original_num_reqs is None:
                        return draft_tokens
                    pad_reqs = mixed_original_num_reqs - draft_tokens.shape[0]
                    if pad_reqs <= 0:
                        return draft_tokens
                    pad_tokens = torch.full(
                        (pad_reqs, draft_tokens.shape[1]),
                        self.runner.input_batch.vocab_size,
                        dtype=draft_tokens.dtype,
                        device=draft_tokens.device)
                    return torch.cat([draft_tokens, pad_tokens], dim=0)

                if use_decode_adaptive:
                    drafter_logits_range_flat = drafter_logits_range[min_acc:].view(-1)
                    _, probs_idx = torch.sort(drafter_logits_range_flat, descending=True)
                    probs_idx = probs_idx[:spec_budget]
                    masked_draft_forward_tokens_list = torch.full_like(draft_forward_tokens_list, -1, device=self.device)
                    masked_draft_forward_tokens_list[:min_acc] = draft_forward_tokens_list[:min_acc]
                    masked_draft_forward_tokens_list[min_acc:].view(-1)[probs_idx] = draft_forward_tokens_list[min_acc:].view(-1)[probs_idx]
                    return pad_mixed_draft_tokens(masked_draft_forward_tokens_list.t())
                else:
                    return pad_mixed_draft_tokens(draft_forward_tokens_list.t())
