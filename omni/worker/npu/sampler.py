# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU sampling ops for the V2 runner.

Two symbols are injected by
``omni/vllm_patches/usefull_patch/patch_mrv2_sampler.py``:

1. ``gumbel_sample`` — ported from vllm-ascend's
   ``vllm_ascend/worker/v2/sample/gumbel.py`` (the Ascend-native adaptation of
   upstream ``vllm/v1/worker/gpu/sample/gumbel.py``). Uses ``tl.rand`` (Philox)
   for the uniform draw and the ``+1e-20`` epsilon form
   ``-log(-log(r + 1e-20) + 1e-20)`` for the Gumbel transform — this avoids
   ``tldevice.log1p`` which is a returning-None stub on triton-ascend 3.2.2.

2. ``apply_top_k_top_p`` — NPU override that unconditionally takes the PyTorch
   sort path (``apply_top_k_top_p_pytorch``). The upstream Qrita Triton kernel
   (``_topk_topp_kernel``, ~960 lines) fails to compile on triton-ascend 3.2.2
   (``scf.while`` + dynamic ``memref.reinterpret_cast`` → BiShengHIR pipeline
   crash, verified 2026-08-13). The sort path uses only mature CANN ops
   (sort/gather/cumsum/masked_fill) and matches the numpy sort oracle 100%
   on mask layout. It is a pure filter (returns masked logits; sampling stays
   in ``gumbel_sample``), matching the MRv2 dispatch contract.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=[
        "local_argmax_stride",
        "local_max_stride",
        "processed_logits_stride",
        "logits_stride",
        "vocab_size",
    ]
)
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    if temp != 0.0 and APPLY_TEMPERATURE:
        # NOTE: Match the behavior of the upstream _temperature_kernel.
        logits = logits / temp

    if processed_logits_ptr is not None:
        # Store the temperature-applied logits.
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

    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        # NOTE: change pos's dtype to tl.int32, because triton-ascend's
        # compiler doesn't support uint64 of pos arg.
        pos = tl.load(pos_ptr + token_idx).to(tl.int32)
        gumbel_seed = tl.randint(seed, pos)

        # NOTE: r is tl.float64 in upstream vllm, change it to tl.float32,
        # because triton-ascend's compiler does not support float64.
        r = tl.rand(gumbel_seed, block).to(tl.float32)
        # epsilon form (not log1p): tldevice.log1p is a returning-None stub on
        # triton-ascend 3.2.2; +1e-20 keeps the argument of both logs positive.
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    idx = tl.argmax(logits, axis=0)
    token_id = block_idx * BLOCK_SIZE + idx
    value = tl.max(logits, axis=0)
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    """Gumbel-max sample per token row.

    ``temp == 0`` rows are greedy (argmax, no noise); ``temp != 0`` rows get
    Gumbel noise scaled by the row's temperature. Mixed batches are handled
    correctly — each row is decided independently by its own ``temp``.
    """
    if use_fp64:
        raise NotImplementedError(
            "[omni-npu/mrv2] FP64 Gumbel sampling is not supported on NPU."
        )
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.int64,
        device=logits.device,
    )
    local_max = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.float32,
        device=logits.device,
    )
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,
        output_processed_logits_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )
    # NOTE: Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """NPU top-k / top-p filter — pure mask, returns logits (no sampling).

    Unconditionally delegates to the PyTorch sort implementation. The upstream
    Qrita Triton kernel (``apply_top_k_top_p_triton``) fails to compile on
    triton-ascend 3.2.2 (BiShengHIR ``scf.while`` + dynamic index crash,
    verified 2026-08-13), so the upstream dispatch's ``num_tokens >= 8 →
    Triton`` branch must be bypassed. The sort path uses only mature CANN ops
    and matches the numpy sort oracle 100% on mask layout.

    This is a pure filter: masked positions become ``-inf``, kept positions
    keep their value; sampling stays in ``gumbel_sample``. Signature mirrors
    upstream ``apply_top_k_top_p`` so it replaces it via ``setattr``.
    """
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    if p is None and k is None:
        return logits
    # allow_cpu_sync=True routes top-k-only through apply_top_k_only
    # (torch.topk, no full-vocab sort); same semantics as upstream's CPU path.
    return apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
