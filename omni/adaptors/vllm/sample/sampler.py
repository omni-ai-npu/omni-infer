
from typing import Dict, List, Optional
import torch
import torch_npu

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.penalties import apply_min_token_penalties
from vllm.v1.sample.sampler import Sampler as SamplerV1
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from omni.models.config_loader.loader import model_extra_config
from omni.layers.npu_sampler_cache import PenaltyCache, ProbCache
from omni.layers.sampler import AscendTopKTopPSamplerV1

_SAMPLING_EPS = 1e-5

def apply_top_k_only(
    logits_or_prob: torch.Tensor,
    k: torch.Tensor,
    is_logits: bool,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits_or_prob.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits_or_prob.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits_or_prob.masked_fill_(logits_or_prob < top_k_mask, -float("inf") if is_logits else float(0))
    return logits_or_prob

def apply_top_k_top_p(
    logits_or_prob: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    is_logits: bool,
) -> torch.Tensor:
    if p is None:
        if k is not None:
            logits_or_prob = apply_top_k_only(logits_or_prob, k, is_logits)
        if is_logits:
            probs = logits_or_prob.softmax(dim=-1, dtype=torch.float32)
        else:
            probs = logits_or_prob / logits_or_prob.sum(dim=-1, keepdim=True)
        return probs, None

    logits_or_prob_sort, logits_or_prob_idx = logits_or_prob.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_or_prob_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_or_prob_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_or_prob_sort < top_k_mask
        logits_or_prob_sort.masked_fill_(top_k_mask, -float("inf") if is_logits else 0)

    # Apply top-p.
    if is_logits:
        probs_sort = logits_or_prob_sort.softmax(dim=-1)
    else:
        probs_sort = logits_or_prob_sort / logits_or_prob_sort.sum(dim=-1, keepdim=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    # at least one
    top_p_mask[:, -1] = False
    probs_sort.masked_fill_(top_p_mask, 0)
    probs = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    return probs, logits_or_prob_idx

def _apply_penalties_v1(logits: torch.Tensor, prompt_mask: torch.Tensor,
                    output_mask: torch.Tensor,
                    output_bin_counts: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor,
                    do_presence_penalties,
                    do_frequency_penalties,
                    do_repetition_penalties) -> torch.Tensor:
    num_seqs, vocab_size = logits.shape
    if do_repetition_penalties:
        repetition_penalties = (repetition_penalties - 1)[:, None].repeat(1, vocab_size)
        repetition_penalties = repetition_penalties * (prompt_mask[:num_seqs] | output_mask[:num_seqs]) + 1
        logits = torch.where(logits > 0, logits / repetition_penalties, logits * repetition_penalties)
    
    if do_frequency_penalties:
        logits = logits - frequency_penalties.unsqueeze(dim=1) * output_bin_counts[:num_seqs]

    if do_presence_penalties:
        logits = logits - presence_penalties.unsqueeze(dim=1) * output_mask[:num_seqs]
    
    return logits

class AscendSamplerV1(SamplerV1):
    def __init__(self, runner):
        super().__init__()
        self.sampling_eps = torch.ones((1,), device=runner.device, dtype=torch.float32) * _SAMPLING_EPS
        self.sampler_preparing_stream = torch.npu.Stream()
        self.penalty_cache = PenaltyCache(runner.max_num_reqs, runner.input_batch.vocab_size, runner.device) if runner.use_penalty else None
        self.prob_cache = ProbCache(runner.max_num_reqs, runner.num_tokens_per_reqs_decode - 1, runner.topk, runner.input_batch.vocab_size, runner.device) if runner.use_rejection_sampler else None
        self.topk_topp_sampler = AscendTopKTopPSamplerV1()

    def expand_sampling_metadata(
            self,
            sampling_metadata: SamplingMetadata,
            spec_metadata: SpecDecodeMetadata,
    ):
        num_total_tokens = sum([i + 1 for i in spec_metadata.num_draft_tokens])
        num_sample_tokens = spec_metadata.cu_num_draft_tokens.clone()
        num_sample_tokens[1:] -= spec_metadata.cu_num_draft_tokens[:-1]
        num_sample_tokens += 1
        repeat_parameters = {
            'repeats': num_sample_tokens,
            'dim': 0,
            'output_size': num_total_tokens,
        }
        if sampling_metadata.temperature is None:
            temperature = self.sampling_eps.repeat(num_total_tokens)
        else:
            temperature = torch.where(sampling_metadata.temperature < self.sampling_eps, self.sampling_eps, sampling_metadata.temperature)
            temperature = temperature.repeat_interleave(**repeat_parameters)
        
        top_p = None if sampling_metadata.top_p is None else sampling_metadata.top_p.repeat_interleave(**repeat_parameters)
        top_k = None if sampling_metadata.top_k is None else sampling_metadata.top_k.repeat_interleave(**repeat_parameters)
        min_p = None if sampling_metadata.min_p is None else sampling_metadata.min_p.repeat_interleave(**repeat_parameters)
        allowed_token_ids_mask = None if sampling_metadata.allowed_token_ids_mask is None \
            else sampling_metadata.allowed_token_ids_mask.repeat_interleave(**repeat_parameters)

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=sampling_metadata.all_greedy,
            all_random=sampling_metadata.all_random,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            generators=sampling_metadata.generators,
            max_num_logprobs=sampling_metadata.max_num_logprobs,
            prompt_token_ids=sampling_metadata.prompt_token_ids,
            frequency_penalties=sampling_metadata.frequency_penalties,
            presence_penalties=sampling_metadata.presence_penalties,
            repetition_penalties=sampling_metadata.repetition_penalties,
            output_token_ids=sampling_metadata.output_token_ids,
            min_tokens=sampling_metadata.min_tokens,
            no_penalties=sampling_metadata.no_penalties,
            logit_bias=sampling_metadata.logit_bias,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=sampling_metadata.bad_words_token_ids,
        )

    # TODO apply min p on logits directly
    def apply_min_p(
        self,
        logits: torch.Tensor,
        min_p: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Filters logits using adaptive probability thresholding.
        """
        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Reshape min_p for broadcasting
        adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
        # Identify valid tokens using threshold comparison
        valid_token_mask = probability_values >= adjusted_min_p
        if return_logits:
            logits[~valid_token_mask] = -float('inf')
            return logits
        else:
            # Apply mask using boolean indexing
            probability_values[~valid_token_mask] = 0
            return probability_values

    # only get probability and indices(if sort)
    def apply_sampling_params(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            spec_metadata: Optional[SpecDecodeMetadata] = None,
            input_ids: Optional[torch.Tensor] = None,
            do_sample: bool = False,
    ) -> torch.Tensor:
        if spec_metadata is not None:
            sampling_metadata = self.expand_sampling_metadata(sampling_metadata, spec_metadata)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        # Apply allowed token ids. 
        logits = self.apply_allowed_token_ids(logits, sampling_metadata)

        # TODO apply these sampling parameters if there are speculative tokens
        if spec_metadata is None:
            # Apply bad words exclusion.
            logits = self.apply_bad_words(logits, sampling_metadata)
            # Apply logits bias.
            logits = self.apply_logits_bias(logits, sampling_metadata)

        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata, spec_metadata, input_ids)

        if not sampling_metadata.all_greedy:
            use_npu_top_k_top_p_sample = model_extra_config.operator_opt_config.enable_topktoppsample_op
            # Apply temperature.
            logits = self.apply_temperature(logits, sampling_metadata.temperature)
            # Apply min_p.
            if sampling_metadata.min_p is not None:
                logits_or_prob = self.apply_min_p(logits, sampling_metadata.min_p, return_logits=use_npu_top_k_top_p_sample)
                is_logits = use_npu_top_k_top_p_sample
            else:
                logits_or_prob = logits
                is_logits = True

            if do_sample:
                p = sampling_metadata.top_p
                k = sampling_metadata.top_k
                if use_npu_top_k_top_p_sample == False or p is None or k is None:
                    probs, idx = apply_top_k_top_p(
                        logits_or_prob, sampling_metadata.top_k, sampling_metadata.top_p, is_logits,
                    )
                    return self.do_sample(
                        probs, idx, sampling_metadata, spec_metadata,
                    )
                else:
                    logits = logits_or_prob.type(torch.bfloat16)
                    p = p.type(torch.bfloat16)
                    k = k.type(torch.int32)
                    q = self.generate_random_sequence(
                        logits, sampling_metadata, spec_metadata,
                    )
                    res = torch_npu.npu_top_k_top_p_sample(logits, k, p, q)
                    return res[0]
            else:
                return apply_top_k_top_p(
                    logits_or_prob, sampling_metadata.top_k, sampling_metadata.top_p, is_logits
                )
        else:
            if do_sample:
                return logits.argmax(dim=-1)
            else:
                return logits

    def generate_random_sequence(
        self,
        probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_metadata: Optional[SpecDecodeMetadata] = None,
    ):
        generators = sampling_metadata.generators
        batchsize = probs.shape[0] if spec_metadata is None else len(spec_metadata.num_draft_tokens)
        req_arange = list(range(batchsize + 1))
        if spec_metadata is not None:
            for i in range(batchsize):
                req_arange[i + 1] = req_arange[i] + spec_metadata.num_draft_tokens[i] + 1

        with torch_npu.npu.stream(self.sampler_preparing_stream) :
            q = torch.empty_like(probs)
            # NOTE(woosuk): To batch-process the requests without their own seeds,
            # which is the common case, we first assume that every request does
            # not have its own seed. Then, we overwrite the values for the requests
            # that have their own seeds.
            if len(generators) != batchsize:
                q.exponential_()
            if generators:
                # TODO(woosuk): This can be slow because we handle each request
                # one by one. Optimize this.
                for i, generator in generators.items():
                    q[req_arange[i]:req_arange[i + 1]].exponential_(generator=generator)
        torch.npu.default_stream().wait_stream(self.sampler_preparing_stream)
        return q

    def do_sample(
            self,
            probs: torch.Tensor,
            idx: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
            spec_metadata: Optional[SpecDecodeMetadata] = None,
    ):
        q = self.generate_random_sequence(
            probs, sampling_metadata, spec_metadata,
        )
        res = probs.div_(q).argmax(dim=-1).view(-1)
        if idx == None:
            return res
        else:
            return torch.gather(idx, 1, res.unsqueeze(1)).view(-1)

    def apply_penalties(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            spec_metadata = None,
            input_ids = None,
    ) -> torch.Tensor:
        if self.penalty_cache is None:
            return logits
        if sampling_metadata.min_tokens:
            apply_min_token_penalties(logits,
                                      sampling_metadata.output_token_ids,
                                      sampling_metadata.min_tokens)
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            if spec_metadata is None:
                logits = _apply_penalties_v1(
                    logits,
                    self.penalty_cache.prompt_mask,
                    self.penalty_cache.output_mask,
                    self.penalty_cache.output_bin_counts,
                    sampling_metadata.presence_penalties,
                    sampling_metadata.frequency_penalties,
                    sampling_metadata.repetition_penalties,
                    self.penalty_cache.do_presence_penalties,
                    self.penalty_cache.do_frequency_penalties,
                    self.penalty_cache.do_repetition_penalties
                )
            else:
                batch_size = len(spec_metadata.num_draft_tokens)
                num_total_tokens = sum([i + 1 for i in spec_metadata.num_draft_tokens])
                num_sample_tokens = spec_metadata.cu_num_draft_tokens.clone()
                num_sample_tokens[1:] -= spec_metadata.cu_num_draft_tokens[:-1]
                num_sample_tokens += 1

                tmp_logits = torch.empty(
                    (batch_size * (spec_metadata.max_spec_len + 1), logits.shape[-1]),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                indices = torch.arange(batch_size, device=logits.device) * spec_metadata.max_spec_len
                indices[1:] -= spec_metadata.cu_num_draft_tokens[:-1]
                indices = indices.repeat_interleave(
                    repeats=num_sample_tokens,
                    dim=0,
                    output_size=num_total_tokens,
                ) + torch.arange(input_ids.numel(), device=input_ids.device)
                tmp_logits[indices] = logits

                token_indices = spec_metadata.bonus_logits_indices - spec_metadata.cu_num_draft_tokens
                token_indices[1:] += spec_metadata.cu_num_draft_tokens[:-1]

                select_indices = torch.arange(batch_size, device=logits.device) * (spec_metadata.max_spec_len + 1)
                for i in range(spec_metadata.max_spec_len + 1):
                    if self.penalty_cache is not None:
                        self.penalty_cache.save_token_ids(input_ids[token_indices % input_ids.numel()])
                    tmp_logits[select_indices] = _apply_penalties_v1(
                        tmp_logits[select_indices],
                        self.penalty_cache.prompt_mask,
                        self.penalty_cache.output_mask,
                        self.penalty_cache.output_bin_counts,
                        sampling_metadata.presence_penalties,
                        sampling_metadata.frequency_penalties,
                        sampling_metadata.repetition_penalties,
                        self.penalty_cache.do_presence_penalties,
                        self.penalty_cache.do_frequency_penalties,
                        self.penalty_cache.do_repetition_penalties
                    )
                    select_indices += 1
                    token_indices += 1

                logits = tmp_logits[indices].clone()
        return logits

    def revert_rejected_tokens(self, accepted_num, input_ids, spec_metadata):
        if not self.penalty_cache or not self.penalty_cache.do_penalties:
            return
        indices = spec_metadata.bonus_logits_indices - spec_metadata.cu_num_draft_tokens
        indices[1:] += spec_metadata.cu_num_draft_tokens[:-1]
        indices += spec_metadata.max_spec_len
        for i in range(spec_metadata.max_spec_len):
            select_indices = indices % input_ids.numel()
            accepted_mask = accepted_num >= spec_metadata.max_spec_len - i
            self.penalty_cache.revert_rejected_tokens(accepted_mask, input_ids[select_indices])
            indices -= 1

    def forward(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            update_penalty: Optional[bool] = True,
    ) -> SamplerOutput:
        # TODO construct forward by above functions
        result = super().forward(logits, sampling_metadata)
        if self.penalty_cache is not None and update_penalty:
            self.penalty_cache.save_token_ids(result.sampled_token_ids)
        return result

    def prepare_cache(self, *args, **kwargs):
        if self.penalty_cache:
            self.penalty_cache.prepare_cache(*args, **kwargs)
        if self.prob_cache:
            self.prob_cache.prepare_cache(*args, **kwargs)