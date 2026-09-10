# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shared Triton-Ascend Gumbel helpers for MRv2 NPU sampling."""

from vllm.triton_utils import tl, triton


@triton.jit
def _npu_gumbel_block_argmax(  # pragma: no cover
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if APPLY_TEMPERATURE:
        if temp != 0.0:
            # Match the behavior of the upstream _temperature_kernel.
            logits = logits / temp

    if processed_logits_ptr is not None:
        if processed_logits_col_ptr is not None:
            col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr
            + req_state_idx * processed_logits_stride
            + col * vocab_size
            + block,
            logits,
            mask=mask,
        )

    logits = logits.to(tl.float32)
    if temp != 0.0:
        seed = tl.load(seeds_ptr + req_state_idx)
        # NPU: cast pos to int32. Triton-Ascend cannot lower the uint64 path
        # used by upstream Philox random generation on Ascend vector cores.
        pos = tl.load(pos_ptr + token_idx).to(tl.int32)
        gumbel_seed = tl.randint(seed, pos)
        # NPU: use float32 tl.rand; Triton-Ascend does not support upstream's
        # float64 random/Gumbel path. The epsilon form avoids log1p.
        r = tl.rand(gumbel_seed, block).to(tl.float32)
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx
