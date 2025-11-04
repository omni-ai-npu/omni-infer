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
        if self.method not in ('deepseek_mtp', 'eagle', 'eagle3', 'pangu_ultra_moe_mtp'):
            raise ValueError(f"Speculative method should be one of ('deepseek_mtp', 'eagle', 'eagle3', 'pangu_ultra_moe_mtp'), while get {self.method}.")

        self.n_predictor = self.vllm_config.model_config.hf_config.num_nextn_predict_layers if self.method == 'deepseek_mtp' else 1
        self.is_autogressive = self.speculative_config.num_speculative_tokens > self.n_predictor

        self.minus_one = -torch.ones(1, device=device)
        self.device = device

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

        self.attn_layer_names = list(draft_attn_layer_names)

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
        self.input_ids[spec_decode_metadata.logits_indices[last_accepted_index]] = forward_tokens
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

        if kv_caches is None:
            with set_forward_context(None, self.vllm_config):
                for i in range(self.speculative_config.num_speculative_tokens):
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
            draft_forward_tokens_list = []

            if self.runner.enable_torchair_graph_mode and attn_state == AscendAttentionState.DecodeOnly \
                and (not self.mark_static):
                from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration
                if isinstance(self.model, GraphCompileConfiguration):
                    self.model.mark_static_for_graph()
                mark_static_for_graph_default(input_ids, previous_hidden_states)
                self.mark_static = True

            with set_forward_context(attn_metadata, self.vllm_config):
                is_dummy = (last_accepted_index is None) or (sample_indices is None)
                if not is_dummy:
                    batch_size = last_accepted_index.numel()
                    if self.enable_adaptive and attn_state == AscendAttentionState.DecodeOnly:
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
                        if self.enable_adaptive and attn_state == AscendAttentionState.DecodeOnly:
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
                            input_ids[sample_indices] = draft_forward_tokens
                    previous_hidden_states = next_hidden_states
            if is_dummy:
                return None
            else:
                draft_forward_tokens_list = torch.stack(draft_forward_tokens_list, dim=0)
                if self.enable_adaptive and attn_state == AscendAttentionState.DecodeOnly:
                    drafter_logits_range_flat = drafter_logits_range[min_acc:].view(-1)
                    _, probs_idx = torch.sort(drafter_logits_range_flat, descending=True)
                    probs_idx = probs_idx[:spec_budget]
                    masked_draft_forward_tokens_list = torch.full_like(draft_forward_tokens_list, -1, device=self.device)
                    masked_draft_forward_tokens_list[:min_acc] = draft_forward_tokens_list[:min_acc]
                    masked_draft_forward_tokens_list[min_acc:].view(-1)[probs_idx] = draft_forward_tokens_list[min_acc:].view(-1)[probs_idx]
                    return masked_draft_forward_tokens_list.t()
                else:
                    return draft_forward_tokens_list.t()
