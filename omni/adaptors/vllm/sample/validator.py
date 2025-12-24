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
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler as RejectionSamplerV1
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from omni.layers.sampler import random_choice
from omni.models.config_loader.loader import model_extra_config

class SimpleValidator(RejectionSamplerV1):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.previous_frequency_penalties = []
        self.previous_repetition_penalties = []
        self.previous_presence_penalties = []
        self.main_sampler = runner.sampler
        self.runner = runner
        self.minus_one = None
        self.minus_ones = None
        self.force_accept_rate = model_extra_config.operator_opt_config.control_accept_rate
        self.enable_force_accept = self.force_accept_rate >= 0 and self.force_accept_rate <= 1

    def forward(self,
                metadata: SpecDecodeMetadata,
                draft_probs: Any,
                logits: torch.Tensor,
                input_ids: torch.Tensor,
                sampling_metadata,
    ) -> SamplerOutput:
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            raise ("Logprobs gathered is not supported in current version")
        if self.minus_one is None:
            # prepare const on npu
            self.minus_one = -torch.ones(1, 1, device=input_ids.device, dtype=input_ids.dtype)
            self.minus_ones = -torch.ones(
                (self.runner.max_num_reqs, self.runner.num_tokens_per_reqs_decode),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

        batch_size = len(metadata.num_draft_tokens)

        key_tokens = input_ids[metadata.logits_indices]

        all_sampled_tokens = self.main_sampler.apply_sampling_params(
            logits, sampling_metadata, metadata, key_tokens, do_sample=True,
        )

        last_accepted_index = metadata.bonus_logits_indices - metadata.cu_num_draft_tokens
        last_accepted_index[1:] += metadata.cu_num_draft_tokens[:-1]
        
        valid_flag = torch.ones(batch_size, dtype=bool, device=input_ids.device)
        if self.enable_force_accept:
            with torch_npu.npu.stream(self.main_sampler.sampler_preparing_stream): 
                forced_accepted = torch.empty_like(key_tokens, dtype=torch.float32).uniform_() < self.force_accept_rate

        if (metadata.max_spec_len == torch.tensor(metadata.num_draft_tokens)).all():
            accepted = key_tokens.roll(-1, 0) == all_sampled_tokens if not self.enable_force_accept else forced_accepted
            accepted[metadata.bonus_logits_indices] = ~valid_flag
            _, accepted_num = accepted.view(batch_size, -1).min(-1)
            accepted_num = accepted_num.to(torch.int32)
            offset = self.runner.arange_npu_int32[:metadata.max_spec_len + 1]
            output_token_ids = torch.where(offset[None, :] <= accepted_num[:, None], all_sampled_tokens.view(batch_size, -1), self.minus_one[0])
        else:
            output_token_ids = self.minus_ones[:batch_size, :metadata.max_spec_len + 1].clone()
            indices = last_accepted_index.clone()
            accepted_num = torch.zeros_like(last_accepted_index)
            for i in range(metadata.max_spec_len + 1):
                now_indices = indices % key_tokens.numel()
                sampled_token_ids = all_sampled_tokens[now_indices]
                if i > 0:
                    valid_flag &= output_token_ids[:, i - 1] == key_tokens[now_indices] if not self.enable_force_accept else forced_accepted[now_indices]
                    valid_flag &= indices <= metadata.bonus_logits_indices
                    sampled_token_ids = torch.where(valid_flag, sampled_token_ids, self.minus_one[0])
                    accepted_num += valid_flag

                output_token_ids[:, i] = sampled_token_ids
                indices += 1
        last_accepted_index += accepted_num
        forward_tokens = output_token_ids.gather(1, accepted_num.view(-1, 1)).reshape(-1)

        self.main_sampler.revert_rejected_tokens(accepted_num, key_tokens, metadata)
        if self.main_sampler.penalty_cache is not None:
            self.main_sampler.save_token_ids(forward_tokens)

        sampler_output = SamplerOutput(
            sampled_token_ids = output_token_ids,
            logprobs_tensors = None
        )

        return sampler_output, forward_tokens, last_accepted_index, accepted_num

class SparseRejectionSamplerValidator(RejectionSamplerV1):

    #TODO(fanyuda): support the feature combo of sparse rejection sampler and mixture of spec decoding

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vllm_config = vllm_config
        self.device = device
        self.previous_frequency_penalties = []
        self.previous_repetition_penalties = []
        self.previous_presence_penalties = []
        self.main_sampler = runner.sampler
        self.minus_one = None
        self.max_num_tokens = runner.decode_max_num_tokens
        self.arange = None

    def forward(self,
                metadata: SpecDecodeMetadata,
                draft_probs: Any,
                logits: torch.Tensor,
                input_ids: torch.Tensor,
                sampling_metadata,
    ) -> SamplerOutput:
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            raise ("Logprobs gathered is not supported in current version")
        if self.arange is None:
            # prepare const on npu
            self.arange = torch.arange(self.max_num_tokens, device=input_ids.device)
        batch_size = len(metadata.num_draft_tokens)
        output_token_ids = torch.full(
            (batch_size, metadata.max_spec_len + 1),
            -1,
            dtype=input_ids.dtype,
            device=self.device,
        )

        key_tokens = input_ids[metadata.logits_indices]


        num_sampling_tokens_per_req = (metadata.logits_indices.numel() // batch_size)
        num_spec_tokens_per_req = num_sampling_tokens_per_req - 1
        output = self.main_sampler.apply_sampling_params(
            logits, sampling_metadata, metadata, key_tokens,
        )

        num_total_tokens = sum([i + 1 for i in metadata.num_draft_tokens])
        num_total_draft_tokens = num_total_tokens - batch_size
        num_draft_tokens = metadata.cu_num_draft_tokens.clone()
        num_draft_tokens[1:] -= metadata.cu_num_draft_tokens[:-1]

        begin_indices = metadata.bonus_logits_indices - num_draft_tokens
        
        vocab_size = logits.shape[1]
        num_tokens = logits.shape[0]

        main_probs = torch.zeros_like(logits, dtype=torch.float32)
        if isinstance(output, tuple):
            probs, idx = output
            if idx is not None:
                main_probs.scatter_(1, idx, probs)
            else:
                main_probs = probs
        else:
            all_sampled_tokens = output.argmax(dim=-1)
            main_probs[self.arange[:num_tokens], all_sampled_tokens] = 1.
        
        is_same_spec_len = all(spec_len == metadata.max_spec_len for spec_len in metadata.num_draft_tokens)
        if is_same_spec_len:
            mtp_probs = self.main_sampler.prob_cache.topk_spec_token_probs[:batch_size].flatten(0, 1)
        else:
            flat_valid_idx = torch.cat(
                [torch.arange(spec_len + 1) + i * self.main_sampler.prob_cache.topk_spec_token_probs.shape[1]
                    for i, spec_len in enumerate(metadata.num_draft_tokens)]
            )
            mtp_probs = self.main_sampler.prob_cache.topk_spec_token_probs[:batch_size].flatten(0, 1)[flat_valid_idx]
            
        accepted_num = torch.zeros_like(begin_indices)
        if metadata.max_spec_len > 0:
            # decode phase

            target_token_probs = torch.gather(main_probs, 1, key_tokens.roll(-1, -1).unsqueeze(1)).squeeze(1)
            # flag to check whether numbers of spec tokens are all equal
            # last prob of each request is 0.0
            draft_token_probs = torch.gather(mtp_probs, 1, key_tokens.roll(-1, -1).unsqueeze(1)).squeeze(1)
            accepted_probs = target_token_probs / draft_token_probs

            # TODO put uniform to dsa stream
            accepted = torch.empty_like(accepted_probs).uniform_() < accepted_probs # boolean mask
            computed_msk = self.main_sampler.prob_cache.computed[:batch_size].repeat_interleave(
                repeats=num_draft_tokens + 1, dim=0, output_size=num_total_tokens,
            )
            accepted &= computed_msk
            if is_same_spec_len:
                accepted = accepted.view(batch_size, -1)
                accepted[:, -1] = False
                accepted_num = accepted.min(-1).indices
                output_token_ids[:, :-1] = torch.where(
                    accepted_num.unsqueeze(1) > self.arange[:metadata.max_spec_len].unsqueeze(0),
                    key_tokens.view(batch_size, -1)[:, 1:],
                    -1
                )
            else:
                valid_flag = torch.ones(batch_size, dtype=bool, device=input_ids.device)
                indices = begin_indices.clone()
                end_indices = metadata.bonus_logits_indices - self.arange[:batch_size]
                for spec_i in range(metadata.max_spec_len):
                    valid_flag &= accepted[indices % accepted.numel()]
                    valid_flag &= indices <= end_indices
                    accepted_num += valid_flag
                    output_token_ids[:, spec_i] = torch.where(
                        valid_flag,
                        key_tokens[indices % accepted.numel()],
                        -1
                    )
                    indices += 1

        begin_indices.add_(accepted_num)
        last_accepted_index = begin_indices
        if self.vllm_config.speculative_config.enable_adaptive:
            mtp_probs[last_accepted_index] = 0.

        recover_probs = main_probs[last_accepted_index] - mtp_probs[last_accepted_index]
        forward_tokens = random_choice(recover_probs, sampling_metadata.generators, self.main_sampler.sampler_preparing_stream)
        output_token_ids[self.arange[:batch_size], accepted_num] = forward_tokens
        self.main_sampler.revert_rejected_tokens(accepted_num, key_tokens, metadata)
        if self.main_sampler.penalty_cache is not None:
            self.main_sampler.penalty_cache.save_token_ids(forward_tokens)

        sampler_output = SamplerOutput(
            sampled_token_ids = output_token_ids,
            logprobs_tensors = None
        )

        return sampler_output, forward_tokens, last_accepted_index, accepted_num