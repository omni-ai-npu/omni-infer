# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Tuple, Optional
import torch


def gather_and_maybe_dequant_cache(
    src_cache: Tuple[torch.Tensor, torch.Tensor],
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: Optional[torch.Tensor] = None,
) -> None:
    """
    An equivalent impl of `vllm._custom_ops.gather_and_maybe_dequant_cache`
    using torch. It uses `starts[i]` and `block_table` to index the starts
    of KVs of the current batch, and extract the KV chunks into `workspace`
    contiguously where the cumulative lengths are `cu_seq_lens[i]`.
    """
    for i in range(batch_size):
        dst_start = cu_seq_lens[i]
        dst_end = cu_seq_lens[i + 1]
        seq_len = dst_end - dst_start
        if seq_len <= 0:
            continue

        block_size = src_cache[0].size(1)
        seq_start = 0 if seq_starts is None else seq_starts[i]
        seq_end = seq_start + seq_len
        positions = torch.arange(seq_start, seq_end, device=block_table.device)
        block_idx, slots = positions // block_size, positions % block_size
        block_ids = block_table[i, block_idx]

        dst[dst_start:dst_end].copy_(torch.cat(
            [
                src_cache[0][block_ids, slots],
                src_cache[1][block_ids, slots]
            ],
            dim=-1,
        ))


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: Optional[torch.Tensor] = None,
) -> None:
    """
    A torch implementation of `vllm._custom_ops.merge_attn_states`. It merges two chunks
    of attention outputs and the corresponding LSE into one.
    The shapes of `output`, `prefix_output` & `suffix_output` should be the same and in
    (T, N, D) format. And the shapes of `output_lse` (if given), `prefix_lse` and `suffix_lse`
    should be (T, N, 1).
    """
    assert output.dim() == 3, f"output should have 3 dimensions, but got shape {output.shape}."
    assert prefix_lse.dim() == 3 and prefix_lse.size(-1) == 1, \
        f"LSE should have 3 dimensions and the last should be 1, but got shape {prefix_lse.shape}."

    prefix_output = prefix_output.float()
    suffix_output = suffix_output.float()

    # map inf to -inf
    prefix_lse = torch.where(prefix_lse == float('inf'),
                             torch.tensor(float('-inf'), device=prefix_lse.device),
                             prefix_lse)
    suffix_lse = torch.where(suffix_lse == float('inf'),
                             torch.tensor(float('-inf'), device=suffix_lse.device),
                             suffix_lse)

    # for numerical stability
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    prefix_lse_adj = prefix_lse - max_lse
    suffix_lse_adj = suffix_lse - max_lse

    prefix_se = torch.exp(prefix_lse_adj)
    suffix_se = torch.exp(suffix_lse_adj)
    out_se = prefix_se + suffix_se

    # calculate merged LSE
    if output_lse is not None:
        out_lse = torch.where(
            out_se > 0,
            torch.log(out_se) + max_lse,
            torch.tensor(float('-inf'), device=out_se.device)
        )
        output_lse.copy_(out_lse)

    # rescale
    prefix_scale = torch.where(
        out_se > 0,
        prefix_se / out_se,
        torch.zeros_like(prefix_se)
    )
    suffix_scale = torch.where(
        out_se > 0,
        suffix_se / out_se,
        torch.zeros_like(suffix_se)
    )

    # merge attention outputs by scale
    merged_output = prefix_output * prefix_scale + suffix_output * suffix_scale
    output.copy_(merged_output)


def attention_update_torch(
    outs: torch.Tensor, # [N, T, D] or list
    lses: torch.Tensor, # [N, T] or list
):
    if isinstance(outs, list):
        assert isinstance(lses, list) and len(outs) == len(lses)
        outs = torch.stack(outs, dim=0)  # [N, T, D]
        lses = torch.stack(lses, dim=0)  # [N, T]

    assert outs.dim() == 3 and lses.dim() == 2
    assert outs.size(0) == lses.size(0)
    assert outs.size(1) == lses.size(1)
    outs = outs.transpose(0, 1).float() # fp32[T, N, D]
    lses = lses.transpose(0, 1).float() # fp32[T, N]

    # inf -> -inf
    _inf = torch.tensor(float("-inf"), device=lses.device)
    lses = torch.where(lses == float("inf"), _inf, lses)

    max_lses = lses.max(dim=-1)[0]               # [T]
    ses = torch.exp(lses - max_lses.view(-1, 1)) # [T, N]
    se = ses.sum(dim=-1)                         # [T]
    valid = se > 0                               # [T]

    lse = torch.where( # [T]
        valid, torch.log(se) + max_lses, _inf
    )
    sc = torch.where(
        valid.view(-1, 1), ses / se.view(-1, 1), torch.zeros_like(ses)
    ).unsqueeze(2) # [T, N, 1]
    out = (outs * sc).sum(1) # sum dim-N

    return out, lse # fp32[T, D], fp32[T]
