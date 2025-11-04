import torch
from typing import Any
import torch_npu

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.rejection_sampler import RejectionSampler as RejectionSamplerV1
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from omni.layers.sampler import random_choice
from omni.models.config_loader.loader import model_extra_config

class SimpleValidator(RejectionSamplerV1):

    def __init__(self, runner, *args, **kwargs) -> None:
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
        output_token_ids = self.minus_ones[:batch_size, :metadata.max_spec_len + 1].clone()

        key_tokens = input_ids[metadata.logits_indices]

        all_sampled_tokens = self.main_sampler.apply_sampling_params(
            logits, sampling_metadata, metadata, key_tokens, do_sample=True,
        )

        indices = metadata.bonus_logits_indices - metadata.cu_num_draft_tokens
        indices[1:] += metadata.cu_num_draft_tokens[:-1]
        last_accepted_index = indices.clone()
        valid_flag = torch.ones(batch_size, dtype=bool, device=input_ids.device)
        accepted_num = torch.zeros_like(last_accepted_index)

        if self.enable_force_accept:
            with torch_npu.npu.stream(self.main_sampler.sampler_preparing_stream): 
                accepted = torch.empty_like(key_tokens, dtype=torch.float32).uniform_() < self.force_accept_rate

        for i in range(metadata.max_spec_len + 1):
            now_indices = indices % key_tokens.numel()
            sampled_token_ids = all_sampled_tokens[now_indices]
            if i == 0:
                forward_tokens = sampled_token_ids
            else:
                valid_flag &= output_token_ids[:, i - 1] == key_tokens[now_indices] if not self.enable_force_accept else accepted[now_indices]
                valid_flag &= indices <= metadata.bonus_logits_indices
                sampled_token_ids = torch.where(valid_flag, sampled_token_ids, self.minus_one[0])
                forward_tokens = torch.where(valid_flag, sampled_token_ids, forward_tokens)
                accepted_num += valid_flag

            output_token_ids[:, i] = sampled_token_ids
            indices += 1
        last_accepted_index += accepted_num

        self.main_sampler.revert_rejected_tokens(accepted_num, key_tokens, metadata)


        sampler_output = SamplerOutput(
            sampled_token_ids = output_token_ids,
            logprobs_tensors = None
        )

        return sampler_output, forward_tokens, last_accepted_index, accepted_num

class SparseRejectionSamplerValidator(RejectionSamplerV1):

    #TODO(fanyuda): support the feature combo of sparse rejection sampler and mixture of spec decoding

    def __init__(self, main_sampler, topk, max_num_tokens, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.previous_frequency_penalties = []
        self.previous_repetition_penalties = []
        self.previous_presence_penalties = []
        self.main_sampler = main_sampler
        self.minus_one = None
        self.topk = topk
        self.max_num_tokens = max_num_tokens
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
        if self.minus_one is None:
            # prepare const on npu
            self.minus_one = -torch.ones(1, 1, device=input_ids.device, dtype=input_ids.dtype)
            self.arange = torch.arange(self.max_num_tokens, device=input_ids.device)

        batch_size = len(metadata.num_draft_tokens)
        output_token_ids = torch.ones(
            (batch_size, metadata.max_spec_len + 1),
            dtype=input_ids.dtype,
            device=input_ids.device,
        ) * self.minus_one[0]

        key_tokens = input_ids[metadata.logits_indices]


        num_sampling_tokens_per_req = (metadata.logits_indices.numel() // batch_size)
        num_spec_tokens_per_req = num_sampling_tokens_per_req - 1
        output = self.main_sampler.apply_sampling_params(
            logits, sampling_metadata, metadata, key_tokens,
        )

        if isinstance(output, tuple):
            probs, idx = output
            all_sampled_tokens = self.main_sampler.do_sample(probs.clone(), idx, sampling_metadata, metadata)

        else:
            # ALL GREEDY
            all_sampled_tokens = output.argmax(dim=-1)

        indices = metadata.bonus_logits_indices - metadata.cu_num_draft_tokens
        indices[1:] += metadata.cu_num_draft_tokens[:-1]
        last_accepted_index = indices.clone()
        accepted_num = torch.zeros_like(last_accepted_index)

        if metadata.max_spec_len > 0:
            # decode phase
            # Currently, we always assume that all the requests share the same number of speculated tokens

            main_probs = None
            vocab_size = logits.shape[1]
            num_tokens = num_sampling_tokens_per_req * batch_size

            if isinstance(output, tuple):
                main_probs = self.recover_prob_topk(probs, idx, vocab_size, self.topk) if idx is not None else probs
            else:
                main_probs = torch.zeros_like(output)
                main_probs[self.arange[:num_tokens], all_sampled_tokens] = 1

            target_probs = main_probs.view(batch_size, -1, vocab_size)[:, :-1, :].view(-1, vocab_size)
            num_tokens = num_spec_tokens_per_req * batch_size
            token_indices = self.arange[:num_tokens]

            if self.topk > 0:
                topk_spec_token_ids = self.main_sampler.prob_cache.topk_spec_token_ids[:batch_size].view(-1, self.topk)
                topk_spec_token_probs = self.main_sampler.prob_cache.topk_spec_token_probs[:batch_size].view(-1, self.topk)
                draft_token_indices = self.main_sampler.prob_cache.selected_indices[:batch_size].view(-1)

                draft_token_ids = topk_spec_token_ids[token_indices, draft_token_indices].view(-1)
                draft_token_probs = topk_spec_token_probs[token_indices, draft_token_indices].view(-1)
                target_token_probs = target_probs[token_indices, draft_token_ids].view(-1)
            else:
                topk_spec_token_probs = self.main_sampler.prob_cache.topk_spec_token_probs[:batch_size].view(-1, vocab_size)
                topk_spec_token_ids = torch.empty_like(topk_spec_token_probs)
                draft_token_ids = key_tokens.view(batch_size, -1)[:, 1:].view(-1)
                draft_token_probs = topk_spec_token_probs[token_indices, draft_token_ids]
                target_token_probs = target_probs[token_indices, draft_token_ids].view(-1)

            accepted_probs = target_token_probs / draft_token_probs
            accepted = torch.empty_like(accepted_probs).uniform_() < accepted_probs # boolean mask

            computed_msk = self.main_sampler.prob_cache.computed[:batch_size].unsqueeze(1).expand(-1, num_spec_tokens_per_req).view(-1)
            accepted &= computed_msk

            accepted = accepted.view(batch_size, -1)
            valid_flag = torch.ones(batch_size, dtype=bool, device=input_ids.device)
            for i in range(metadata.max_spec_len + 1):
                sampled_token_ids = all_sampled_tokens[indices % key_tokens.numel()]
                if i > 0:
                    valid_flag &= accepted[:, i - 1]
                    valid_flag &= indices <= metadata.bonus_logits_indices
                    sampled_token_ids = torch.where(valid_flag, sampled_token_ids, self.minus_one[0])
                    accepted_num += valid_flag

                output_token_ids[:, i] = sampled_token_ids
                indices += 1
        else:
            # prefill phase
            forward_tokens = all_sampled_tokens[indices]
            output_token_ids[:, 0] = forward_tokens


        last_accepted_index += accepted_num

        if metadata.max_spec_len > 0:
            accepted_mask = accepted_num == num_spec_tokens_per_req
            output_token_ids[:, :-1] = key_tokens.view(batch_size, -1)[:, 1:]
            output_token_ids = output_token_ids.view(-1)
            bias = self.arange[:batch_size]
            resample_indices = last_accepted_index
            drafter_resample_indices = resample_indices - bias - accepted_mask.int()
            resample_tokens = self._reject_sampling(sampling_metadata.generators, main_probs[resample_indices], topk_spec_token_ids[drafter_resample_indices], topk_spec_token_probs[drafter_resample_indices])
            output_token_ids[resample_indices] = torch.where(accepted_mask, output_token_ids[resample_indices], resample_tokens)
            forward_tokens = output_token_ids[last_accepted_index]
            output_token_ids = output_token_ids.view(batch_size, -1)

        self.main_sampler.revert_rejected_tokens(accepted_num, key_tokens, metadata)


        sampler_output = SamplerOutput(
            sampled_token_ids = output_token_ids,
            logprobs_tensors = None
        )

        return sampler_output, forward_tokens, last_accepted_index, accepted_num

    def _reject_sampling(self, generators, target_probs, topk_spec_token_ids, topk_spec_token_probs) -> torch.Tensor:
        if self.topk > 0:
            draft_probs = torch.zeros_like(target_probs)
            draft_probs = self.recover_sparse_prob(draft_probs, topk_spec_token_probs, topk_spec_token_ids)
        else:
            draft_probs = topk_spec_token_probs
        recovered_probs = target_probs - draft_probs

        sampled_token_ids = random_choice(recovered_probs, generators, self.main_sampler.sampler_preparing_stream)
        return sampled_token_ids

    def recover_sparse_prob(self, recovered_prob: torch.Tensor, prob: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        batch_size, idx_size = prob.shape
        i_indices = self.arange[:batch_size].unsqueeze(1).expand(-1, idx_size)
        j_indices = idx

        recovered_prob[i_indices.flatten(), j_indices.flatten()] = prob.flatten()
        return recovered_prob

    def recover_prob_topk(self, prob: torch.Tensor, idx: torch.Tensor, vocab_size: int, topk: int) -> torch.Tensor:
        prob = prob[:, -topk:]
        idx = idx[:, -topk:]

        recovered_prob = torch.zeros((prob.shape[0], vocab_size), device=prob.device)
        return self.recover_sparse_prob(recovered_prob, prob, idx)