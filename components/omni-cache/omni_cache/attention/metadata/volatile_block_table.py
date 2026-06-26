# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Fake (volatile) block_table rewriting for Task #12.

The prefill side allocates a contiguous HBM buffer per attention type
sized to `max_num_reqs × max_blocks_per_req + 1` (see
`PrefillOmniCache.initialize_device_cache`). Real block_ids handed out
by the scheduler can point outside that buffer's addressable range, so
we swap in a *fake* block_table whose entries are sequential indices
into the buffer. `omni_cache.volatile_table` is that fake table —
`torch.arange(1, 1 + max_num_reqs * max_blocks_per_req).reshape(max_num_reqs, max_blocks_per_req)`.

Each attention metadata builder's prefill `build()` calls
`apply_volatile_block_table` after `super().build()`, swaps the result
into metadata, and preserves the original table as
`metadata._real_block_tables` for Task #13's CPU-side scatter to the
real host-pool block_ids.
"""

from __future__ import annotations

from typing import Tuple

import torch


def apply_volatile_block_table(
    block_tables: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    volatile_table: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace `block_tables` / `slot_mapping` with fake-table equivalents.

    Returns `(fake_block_tables, fake_slot_mapping, real_block_tables)`.
    The caller writes the first two into the attention metadata and
    preserves the third on the metadata object as `_real_block_tables`
    so the D2H rewrite step (Task #13) can scatter the HBM buffer back
    to the real host-pool block_ids.

    Args:
        block_tables: `[num_reqs, max_blocks_per_req]`, int32 or int64.
        slot_mapping: `[num_tokens]` flat slot indices (real).
        query_start_loc: `[num_reqs + 1]` cumulative query lengths.
        volatile_table: `[max_reqs, max_blocks_per_req]` the fake table.
        block_size: tokens per block.
    """
    num_reqs = int(query_start_loc.shape[0] - 1)
    if num_reqs == 0:
        return (
            volatile_table[:0].clone(),
            slot_mapping.clone(),
            block_tables.detach().clone(),
        )
    if num_reqs > volatile_table.shape[0]:
        raise ValueError(
            f"volatile_table has room for {volatile_table.shape[0]} reqs "
            f"but the batch has {num_reqs}; bump max_num_reqs or truncate."
        )

    real_block_tables = block_tables_copy(block_tables)
    fake_block_tables = volatile_table[:num_reqs].clone()
    fake_slot_mapping = _rebuild_slot_mapping(
        query_start_loc=query_start_loc,
        fake_block_tables=fake_block_tables,
        block_size=block_size_int(block_size),
        slot_mapping_dtype=slot_mapping.dtype,
    )
    return fake_block_tables, fake_slot_mapping, real_block_tables


def block_tables_copy(t: torch.Tensor) -> torch.Tensor:
    """Defensive clone so the caller can stash a reference that won't
    change if the framework mutates the original."""
    return t.detach().clone() if t is not None else t


def block_size_int(x) -> int:
    return int(x)


def _rebuild_slot_mapping(
    query_start_loc: torch.Tensor,
    fake_block_tables: torch.Tensor,
    block_size: int,
    slot_mapping_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute `fake_slot_mapping[tok] = fake_block_id * block_size + offset`.

    Each request's tokens at flat positions `[query_start_loc[r],
    query_start_loc[r+1])` get offsets `[0, seq_len)` within its own
    sequence; the fake slot is
    `fake_block_tables[r, offset // block_size] * block_size + offset % block_size`.
    """
    device = query_start_loc.device
    cumlens = query_start_loc.to(torch.int64).to(device)
    num_reqs = int(cumlens.numel() - 1)
    num_tokens = int(cumlens[-1].item())
    slot_mapping = torch.empty(num_tokens, dtype=slot_mapping_dtype, device=device)
    fake_bt = fake_block_tables.to(torch.int64).to(device)

    for req in range(num_reqs):
        start = int(cumlens[req].item())
        end = int(cumlens[req + 1].item())
        seq_len = end - start
        if seq_len == 0:
            continue
        offs = torch.arange(seq_len, dtype=torch.int64, device=device)
        block_idx = offs // block_size
        offset_in_block = offs % block_size
        # Clip against available blocks — shouldn't happen in practice.
        block_idx = block_idx.clamp(max=fake_bt.shape[1] - 1)
        fake_blocks = fake_bt[req, block_idx]
        slot_mapping[start:end] = (
            (fake_blocks * block_size + offset_in_block).to(slot_mapping_dtype)
        )
    return slot_mapping
