# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""A layer that samples the next tokens from the model's outputs."""
from typing import Optional, Any
import os
import torch
import torch_npu

from vllm.config.model import LogprobsMode
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as SamplerV1
from vllm.v1.outputs import SamplerOutput as SamplerOutputV1
from vllm.forward_context import get_forward_context

from omni_npu.sample.ops.topk_topp_sampler import NPUTopKTopPSampler

FP32_EPS = 2 ** -24
ENABLE_NPU_PENALTY_CACHE = os.getenv("OMNI_NPU_PENALTY_CACHE", "0") == "1"

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
        repetition_penalties = repetition_penalties * (prompt_mask[:num_seqs, :vocab_size] | output_mask[:num_seqs, :vocab_size]) + 1
        logits = torch.where(logits > 0, logits / repetition_penalties, logits * repetition_penalties)

    if do_frequency_penalties:
        logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts[:num_seqs, :vocab_size]

    if do_presence_penalties:
        logits -= presence_penalties.unsqueeze(dim=1) * output_mask[:num_seqs, :vocab_size]

    return logits
     

class NPUSamplerV1(SamplerV1):
    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__(logprobs_mode)
        self.dsa_stream = torch_npu.npu.Stream()
        self.topk_topp_sampler = NPUTopKTopPSampler(logprobs_mode, self.dsa_stream)

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
        *args,
        **kwargs
    ) -> SamplerOutputV1:
        # --- THE SWITCH ---
        if not ENABLE_NPU_PENALTY_CACHE:
            return super().forward(logits, sampling_metadata, *args, **kwargs)
        
        if sampling_metadata is None:
            forward_context = get_forward_context()
            sampling_metadata = forward_context.sampling_metadata
            
        input_batch = getattr(self, "npu_input_batch", None)

        if logits is not None:
            if sampling_metadata.logitsprocs:
                for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
                    logits = processor.apply(logits)

            if not sampling_metadata.no_penalties and input_batch is not None:
                do_presence = len(input_batch.presence_penalties_reqs) > 0
                do_frequency = len(input_batch.frequency_penalties_reqs) > 0
                do_repetition = len(input_batch.repetition_penalties_reqs) > 0
                logits = _apply_penalties_v1(
                    logits,
                    input_batch.prompt_mask,
                    input_batch.output_mask,
                    input_batch.output_bin_counts,
                    sampling_metadata.presence_penalties,
                    sampling_metadata.frequency_penalties,
                    sampling_metadata.repetition_penalties,
                    do_presence_penalties=do_presence,
                    do_frequency_penalties=do_frequency,
                    do_repetition_penalties=do_repetition
                )

            if sampling_metadata.all_greedy:
                sampled_token_ids = logits.argmax(dim=-1).to(torch.int32)
            else:
                if getattr(sampling_metadata, "temperature", None) is not None:
                    t = sampling_metadata.temperature.unsqueeze(-1)
                    logits = torch.where(t > FP32_EPS, logits / t, logits)

                sampled_token_ids, _ = self.topk_topp_sampler(
                    logits,
                    sampling_metadata.generators,
                    sampling_metadata.top_k,
                    sampling_metadata.top_p
                )
                
                if getattr(sampling_metadata, "temperature", None) is not None:
                    greedy_mask = (sampling_metadata.temperature < FP32_EPS)
                    if greedy_mask.any():
                        greedy_tokens = logits.argmax(dim=-1)
                        sampled_token_ids = torch.where(
                            greedy_mask.unsqueeze(-1), 
                            greedy_tokens.unsqueeze(-1), 
                            sampled_token_ids
                        )
                        
                sampled_token_ids = sampled_token_ids.to(torch.int32)

            if input_batch is not None and not sampling_metadata.no_penalties:
                input_batch.update_sampled_tokens(sampled_token_ids)

            if sampled_token_ids.dim() == 1:
                sampled_token_ids = sampled_token_ids.unsqueeze(-1)

            return SamplerOutputV1(
                sampled_token_ids=sampled_token_ids,
                logprobs_tensors=None,
            )
        else:
            return SamplerOutputV1(
                sampled_token_ids=torch.tensor([[]], dtype=torch.int32, device="cpu"),
                logprobs_tensors=None,
            )