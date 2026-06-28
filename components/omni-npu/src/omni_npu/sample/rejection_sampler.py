# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from dataclasses import replace
from typing import Optional

import torch
import torch_npu

from vllm.logger import init_logger

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.rejection_sampler import (
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    GREEDY_TEMPERATURE,
    RejectionSampler,
    generate_uniform_probs
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.sample.ops.topk_topp_sampler import apply_top_k_top_p_npu
from omni_npu.v1.utils import on_ascend950

logger = init_logger(__name__)

NEW_GREEDY_TEMPERATURE = 1e-6 # convert greedy cases into random cases


class NPURejectionSampler(RejectionSampler):
    def __init__(self, sampler: Sampler):
        super().__init__(sampler)
        self.dsa_stream = sampler.dsa_stream
        self.use_npu_sample = not (
            model_extra_config.operator_opt_config.disable_npu_top_k_top_p_sample
            or on_ascend950()
        )

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()

        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )

        if draft_probs is None:
            target_token_ids, target_logits = compute_probs_and_sample(
                target_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
                metadata,
                self.dsa_stream,
                self.use_npu_sample
            )

            output_token_ids = simple_verify(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                target_token_ids,
                bonus_token_ids,
                sampling_metadata,
            )

        else:
            # [num_tokens, vocab_size]
            # NOTE(woosuk): `target_logits` can be updated in place inside the
            # `compute_probs` function.
            target_logits = compute_probs(
                target_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
                self.use_npu_sample,
            )
            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

            output_token_ids = rejection_sample(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                draft_probs,
                target_probs,
                bonus_token_ids,
                sampling_metadata,
                self.dsa_stream,
            )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    stream: torch.npu.Stream,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1).to(torch.int32)
        rejection_greedy_sample_native(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    if model_extra_config.operator_opt_config.use_topk_topp_stream:
        with torch_npu.npu.stream(stream):
            uniform_probs = generate_uniform_probs(
                num_tokens,
                num_draft_tokens,
                sampling_metadata.generators,
                device,
            )
        torch.npu.default_stream().wait_stream(stream)
    else:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        stream
    )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_native(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return output_token_ids


# exactly same with compute_probs from vllm/vllm/v1/sample/rejection_sampler.py
# to use apply_top_k_top_p_npu
def compute_probs(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
    use_npu_sample: bool = True,
) -> torch.Tensor:
    """Compute probability distribution from logits based on sampling metadata.

    This function applies temperature scaling to the logits and converts
    them to probabilities using softmax. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be converted to probabilities.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Probability distribution (softmax of scaled logits)
            if non-greedy sampling is used, otherwise returns the
            original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    if not use_npu_sample:
        logits = apply_top_k_top_p(logits, top_k, top_p)
    else:
        logits = apply_top_k_top_p_npu(logits, top_k, top_p)
    return logits


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    replaced_x = torch.where(x == replace_from, replace_to, x)
    num_tokens_tensor = cu_num_tokens.clone()
    num_tokens_tensor[1:] -= cu_num_tokens[:-1]
    expanded_x = torch.repeat_interleave(
        replaced_x,
        num_tokens_tensor,
        output_size=num_tokens,
    )
    return expanded_x


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    stream: torch.npu.Stream,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    if model_extra_config.operator_opt_config.use_topk_topp_stream:
        with torch_npu.npu.stream(stream):
            q = torch.empty_like(
                target_probs,
                dtype=torch.float32,
            )
            q.exponential_()
            for i, generator in sampling_metadata.generators.items():
                # Do not generate random numbers for requests with no draft tokens.
                # This can be important for reproducibility.
                if num_draft_tokens[i] > 0:
                    q[i].exponential_(generator=generator)
        torch.npu.default_stream().wait_stream(stream)
    else:
        q = torch.empty_like(
            target_probs,
            dtype=torch.float32,
        )
        q.exponential_()
        for i, generator in sampling_metadata.generators.items():
            # Do not generate random numbers for requests with no draft tokens.
            # This can be important for reproducibility.
            if num_draft_tokens[i] > 0:
                q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_native(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        vocab_size, # next_power_of_2(vocab_size) is useless in native
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


def sample_recovered_tokens_native(
    recovered_token_ids, # [num_tokens]
    cu_num_draft_tokens, # [batch_size]
    draft_token_ids, # [num_tokens]
    draft_probs, # [num_tokens, vocab_size] or None
    target_probs, # [num_tokens, vocab_size]
    q, # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE, # next_power_of_2(vocab_size) is useless in native
    NO_DRAFT_PROBS,
):
    num_tokens = target_probs.shape[0]

    if draft_probs is None:
        sample_probs = target_probs.clone()
        offset = torch.arange(
            num_tokens, dtype=draft_token_ids.dtype, device=draft_token_ids.device,
        ) * sample_probs.shape[1]
        sample_probs.view(-1)[draft_token_ids.view(-1) + offset] -= 1
    else:
        sample_probs = target_probs - draft_probs

    num_draft_tokens = cu_num_draft_tokens.clone()
    num_draft_tokens[1:] -= cu_num_draft_tokens[:-1]
    q = torch.repeat_interleave(
        q, num_draft_tokens, dim=0, output_size=num_tokens,
    )
    recovered_token_ids[:] = (sample_probs / q).argmax(dim=-1).to(torch.int32)


def rejection_greedy_sample_native(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    is_greedy,  # [batch_size] or None
    max_spec_len,
):
    accepted = draft_token_ids == target_argmax

    select_tokens_by_accepted(
        output_token_ids,
        accepted,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
    )


def rejection_random_sample_native(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS,
):
    num_tokens = target_probs.shape[0]

    if draft_probs is None:
        draft_prob = torch.ones(
            num_tokens,
            dtype=target_probs.dtype,
            device=target_probs.device,
        )
    else:
        draft_prob = draft_probs.gather(1, draft_token_ids.view(-1, 1)).view(-1)

    target_prob = target_probs.gather(1, draft_token_ids.view(-1, 1)).view(-1)
    accepted = target_prob / draft_prob >= uniform_probs

    select_tokens_by_accepted(
        output_token_ids,
        accepted,
        cu_num_draft_tokens,
        draft_token_ids,
        recovered_token_ids,
        bonus_token_ids,
        ~is_greedy,
        max_spec_len,
    )


def select_tokens_by_accepted(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    accepted, # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    recovered_token_ids,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    fill_this_time, # [batch_size] or None
    max_spec_len, # int
):
    batch_size = cu_num_draft_tokens.shape[0]
    num_tokens = accepted.shape[0]
    device = accepted.device

    num_draft_tokens = cu_num_draft_tokens.clone()
    num_draft_tokens[1:] -= cu_num_draft_tokens[:-1]

    # get arange
    pre_cu_num_draft_tokens = torch.zeros_like(cu_num_draft_tokens)
    pre_cu_num_draft_tokens[1:] = cu_num_draft_tokens[:-1]
    index_offset = torch.arange(batch_size, dtype=cu_num_draft_tokens.dtype, device=cu_num_draft_tokens.device) * (max_spec_len + 1) - pre_cu_num_draft_tokens
    offsets = torch.repeat_interleave(
        index_offset,
        num_draft_tokens,
        dim=0,
        output_size=num_tokens,
    )
    arange = torch.arange(num_tokens, dtype=cu_num_draft_tokens.dtype, device=cu_num_draft_tokens.device) + offsets

    accepted_mat = torch.zeros(batch_size, (max_spec_len + 1), dtype=accepted.dtype, device=device)
    accepted_mat.view(-1)[arange] = accepted
    accepted_mat = accepted_mat.to(torch.int32).cumprod(dim=-1)
    accepted_num = accepted_mat.sum(dim=-1)
    accepted_mat = accepted_mat.to(torch.bool)
    accepted = accepted_mat.view(-1)[arange]

    recovered_mat = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    recovered_mat.view(-1)[arange] = recovered_token_ids
    recovered_mat.scatter_(1, num_draft_tokens.view(-1, 1), bonus_token_ids.view(-1, 1))

    draft_token_ids = torch.where(accepted, draft_token_ids, PLACEHOLDER_TOKEN_ID)
    # Create output buffer.
    output = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output.view(-1)[arange] = draft_token_ids
    output.scatter_(1, accepted_num.view(-1, 1), recovered_mat.gather(1, accepted_num.view(-1, 1)))

    if fill_this_time is not None:
        output_token_ids[:] = torch.where(fill_this_time[:, None], output, output_token_ids)
    else:
        output_token_ids[:] = output


def compute_probs_and_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
    metadata: SpecDecodeMetadata,
    stream: torch.npu.Stream,
    use_npu_sample,
) -> torch.Tensor:

    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1

    if sampling_metadata.all_greedy:
        return logits.argmax(dim=-1).to(torch.int32), logits

    num_tokens = logits.shape[0]
    if sampling_metadata.temperature is None:
        temperature = torch.ones((num_tokens,), device=logits.device, dtype=torch.float32)
    else:
        temperature = expand_batch_to_tokens(
            sampling_metadata.temperature,
            cu_num_draft_tokens,
            num_tokens,
            replace_from=GREEDY_TEMPERATURE,
            replace_to=NEW_GREEDY_TEMPERATURE,
        )
    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        ).type(torch.int32)
    elif use_npu_sample:
        top_k = torch.ones((logits.shape[0],), dtype=torch.int32, device=logits.device) * logits.shape[1]
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )
    elif use_npu_sample:
        top_p = torch.ones(logits.shape[0], dtype=logits.dtype, device=logits.device)

    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    if model_extra_config.operator_opt_config.use_topk_topp_stream:
        with torch_npu.npu.stream(stream):
            q = generate_random_sequence(
                logits, sampling_metadata, metadata,
            ).type(torch.float32)
        torch.npu.default_stream().wait_stream(stream)
    else:
        q = generate_random_sequence(
            logits, sampling_metadata, metadata,
        ).type(torch.float32)

    if not use_npu_sample:
        logits = apply_top_k_top_p(logits, top_k, top_p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        token_ids = probs.div_(q).argmax(dim=-1).view(-1)
        return token_ids.to(torch.int32), logits

    res = torch_npu.npu_top_k_top_p_sample(
        logits,
        top_k,
        top_p,
        q,
        is_need_logits=True,
    )
    return res[0].to(torch.int32), res[1]


def simple_verify(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, 1]
    target_token_ids: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert cu_num_draft_tokens.ndim == 1
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    device = target_token_ids.device
    assert draft_token_ids.is_contiguous()
    assert bonus_token_ids.is_contiguous()

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    rejection_greedy_sample_native(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        None,
        max_spec_len,
    )
    return output_token_ids


def generate_random_sequence(
    probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    spec_metadata: Optional[SpecDecodeMetadata] = None,
):
    generators = sampling_metadata.generators
    batchsize = probs.shape[0] if spec_metadata is None else len(spec_metadata.num_draft_tokens)
    req_arange = list(range(batchsize + 1))
    if spec_metadata is not None:
        for i in range(batchsize):
            req_arange[i + 1] = req_arange[i] + spec_metadata.num_draft_tokens[i]
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
            if req_arange[i + 1] > req_arange[i]:
                q[req_arange[i]:req_arange[i + 1]].exponential_(generator=generator)
    return q