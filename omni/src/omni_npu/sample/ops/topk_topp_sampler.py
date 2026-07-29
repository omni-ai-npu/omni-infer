# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Optional

import torch
import torch_npu
import torch.nn as nn
from packaging import version

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler as V1TopKTopPSampler

from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.utils import on_ascend950
from omni_npu.layers.utils import named_stream

logger = init_logger(__name__)


def apply_top_k_top_p_npu(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    if k is None and p is None:
        return logits
    logits = logits.type(model_extra_config.dtype)
    if p is not None:
        p = p.type(model_extra_config.dtype)
    else:
        p = torch.ones(logits.shape[0], dtype=model_extra_config.dtype, device=logits.device)
    if k is not None:
        k = k.type(torch.int32)
    else:
        k = torch.ones((logits.shape[0],), dtype=torch.int32, device=logits.device) * logits.shape[1]
    _, logits = torch_npu.npu_top_k_top_p_sample(logits, k, p, q=None, is_need_logits=True)
    return logits


# edit from vllm.v1.sample.ops.topk_topp_sampler.random_sample
def generate_coins(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    sampler,
):
    # Under spec decode the rejection sampler pre-generates this step's bonus
    # noise once and stashes it here; consume and clear it. bonus_q is set and
    # consumed within the same forward, so it is always shape-matched.
    if sampler.bonus_q is not None:
        q = sampler.bonus_q
        sampler.bonus_q = None
        return q
    cur_stream = torch.npu.current_stream()
    with torch.npu.stream(sampler.dsa_stream):
        q = torch.empty_like(probs, dtype=torch.float32)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    cur_stream.wait_stream(sampler.dsa_stream)
    return q


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits, None
        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k), None

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits_sort, logits_idx


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def random_sample(
    probs: torch.Tensor,
    idx: Optional[torch.Tensor],
    generators: dict[int, torch.Generator],
    sampler,
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = generate_coins(probs, generators, sampler)
    res = probs.div_(q).argmax(dim=-1).view(-1)
    if idx is None:
        return res
    else:
        return torch.gather(idx, 1, res.unsqueeze(1)).view(-1)


class NPUTopKTopPSampler(V1TopKTopPSampler):
    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs", sampler = None) -> None:
        super().__init__(logprobs_mode)
        if on_ascend950():
            self.forward = self.forward_native
        else:
            self.apply_top_k_top_p = apply_top_k_top_p_npu
            self.forward = self.forward_npu
        # Owning sampler: carries the noise side stream and the spec-decode
        # bonus_q stash (see generate_coins).
        self.sampler = sampler
        self.dsa_stream = sampler.dsa_stream if sampler is not None else named_stream("dsa_stream")

    def forward_npu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if model_extra_config.operator_opt_config.disable_npu_top_k_top_p_sample:
            logits, idx = apply_top_k_top_p(logits, k, p)
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            token_ids = random_sample(probs, idx, generators, self.sampler)

            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
            return token_ids, logits_to_return

        logits = logits.type(model_extra_config.dtype)
        if p is not None:
            p = p.type(model_extra_config.dtype)
        else:
            p = torch.ones(logits.shape[0], dtype=model_extra_config.dtype, device=logits.device)
        if k is not None:
            k = k.type(torch.int32)
        else:
            k = torch.ones((logits.shape[0],), dtype=torch.int32, device=logits.device) * logits.shape[1]
        q = generate_coins(logits, generators, self.sampler)
        token_ids, logits = torch_npu.npu_top_k_top_p_sample(logits, k, p, q=q, is_need_logits=True)

        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        return token_ids, logits_to_return
