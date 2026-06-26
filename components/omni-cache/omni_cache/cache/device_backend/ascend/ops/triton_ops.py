# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Block-table and HBM-slot ops for the decode-side HBM buffer pool.

These used to be Triton kernels (hence the module name), but on Ascend NPU
the triton package isn't available, so we fall back to pure PyTorch
implementations. They carry the same semantics as the originals and are
correct; they're just not as fast as the JIT'd version on GPU.

The ops are wired from:
  - omni_cache.attention.metadata.compress.fake_build_compress
    (build_fake_block_table_kernel_compress)
  - omni_cache.attention.metadata.sliding_window_attention
    (update_block_table_kernel_swa / move_hbm_slots_kernel_swa)
  - omni_cache.cache.decode.gather_selection.*
    (move_slots_kernel[_batched_layers])

All kernels are invoked through the triton-style `[grid](args...)` call,
so we emulate that with a small wrapper class.
"""

import os

import torch


class _GridCallable:
    """Callable that accepts triton-style [grid](args) then (kwargs)."""
    __slots__ = ("_fn", "_grid")

    def __init__(self, fn):
        self._fn = fn
        self._grid = None

    def __getitem__(self, grid):
        self._grid = grid
        return self

    def __call__(self, *args, **kwargs):
        grid = self._grid
        self._grid = None
        # PyTorch impl doesn't need grid — we just run the loop.
        return self._fn(*args, grid=grid, **kwargs)


def _kernel(fn):
    """Decorator: turn `def kernel(args, *, BLOCK_SIZE, grid=None): ...` into
    a triton-style launchable via `kernel[grid](args, BLOCK_SIZE=...)`."""
    return _GridCallable(fn)


# ---------------------------------------------------------------------------
# build_fake_block_table_kernel_compress
# Write the fake (per-request) block_table: each request i gets
#   block_table[i, :select_len] = base_block_list[:select_len] + req_offset * real_indices[i]
# ---------------------------------------------------------------------------
@_kernel
def build_fake_block_table_kernel_compress(
    block_table,       # Tensor [num_reqs, max_blocks_per_req], int
    base_block_list,   # Tensor [num_base_blocks], int
    real_indices,      # Tensor [num_reqs], int. A negative value skips that row.
    stride_bt_req,     # ignored (kept for signature compatibility)
    stride_bt_blk,     # ignored
    req_offset_base,   # int — slots per request
    select_len,        # int — number of blocks to write per request
    *,
    BLOCK_SIZE=None,
    grid=None,
):
    del stride_bt_req, stride_bt_blk, BLOCK_SIZE, grid  # unused in PyTorch impl

    base_slice = base_block_list[:select_len].to(block_table.dtype)
    # valid: rows where real_indices >= 0
    valid = real_indices >= 0
    offsets = (req_offset_base * real_indices).to(block_table.dtype)
    # broadcast: [num_reqs, 1] + [select_len]
    values = base_slice.unsqueeze(0) + offsets_col(offsets)
    block_table[valid_rows(valid=valid), :select_len] = values[valid]


def offsets_col(offsets: torch.Tensor) -> torch.Tensor:
    return offsets.unsqueeze(1)


def valid_rows(valid: torch.Tensor) -> torch.Tensor:
    return valid


# ---------------------------------------------------------------------------
# move_slots_kernel
# For each request, shift its three adjacent blocks in a pool by one:
#   pool[base+1] = pool[base+2]; pool[base+2] = pool[base+3]; pool[base+3] = 0
# ---------------------------------------------------------------------------
@_kernel
def move_slots_kernel(
    pool,              # Tensor [num_blocks_total, block_size] (any inner shape)
    offsets,           # Tensor [num_reqs] int — per-request base slot index
    stride_p_batch,    # ignored
    stride_p_block,    # ignored
    *,
    BLOCK_SIZE=None,
    grid=None,
):
    del stride_p_batch, stride_p_block, BLOCK_SIZE, grid
    for offset in offsets.tolist():
        pool[offset + 1].copy_(pool[offset + 2])
        pool[offset + 2].copy_(pool[offset + 3])
        pool[offset + 3].zero_()


# ---------------------------------------------------------------------------
# move_slots_batched_layers_kernel
# Same as move_slots_kernel but run over (req, layer). layer_ptrs is a
# 1D int64 tensor of raw pointer values; for the Python fallback we pass the
# layer tensors themselves as a list in `layer_tensors`.
# ---------------------------------------------------------------------------
@_kernel
def move_slots_batched_layers_kernel(
    layer_ptrs_or_tensors,   # list[Tensor] — each is a pool for one layer
    offsets,                 # Tensor [num_reqs] int
    stride_block,            # ignored
    *,
    BLOCK_SIZE=None,
    grid=None,
):
    del stride_block, BLOCK_SIZE, grid

    layers = layer_ptrs_or_tensors
    offs_list = offsets.tolist()

    # Accept either an already-expanded list of tensors, or a 1-D pointer
    # tensor where indices are layer ids. We can't materialize raw pointers
    # back into tensors, so callers using the fallback must pass a list.
    if not isinstance(layers, (list, tuple)):
        raise RuntimeError(
            "move_slots_batched_layers_kernel fallback requires a list of "
            "per-layer tensors, not a pointer tensor."
        )

    for pool in layers:
        for offset in offs_list:
            pool[offset + 1].copy_(pool[offset + 2])
            pool[offset + 2].copy_(pool[offset + 3])
            pool[offset + 3].zero_()


# ---------------------------------------------------------------------------
# update_block_table_kernel_swa
# For each request, update the SWA block table in-place when the new token
# crosses a block boundary. See dsa.py for the exact semantics we preserved.
# ---------------------------------------------------------------------------
@_kernel
def update_block_table_kernel_swa(
    block_table_old,
    block_table_new,
    first_token_slots,
    real_indices,
    block_offsets_out,
    move_hbm_flag,
    stride_bt_req,
    stride_bt_blk,
    req_offset_base,
    max_blocks,
    *,
    BLOCK_SIZE=None,
    grid=None,
):
    del stride_bt_req, stride_bt_blk, BLOCK_SIZE, grid

    for pid in range(block_table_old.shape[0]):
        real_idx = int(real_indices[pid].item())
        if real_idx < 0:
            continue

        slot = int(first_token_slots[pid].item())
        target_block_id = slot // 128

        row = block_table_old[pid, :max_blocks]
        matches = (row == target_block_id).nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            continue
        end_idx = int(matches[-1].item())

        nonzero_mask = (row != 0) & (torch.arange(row.numel(), device=row.device) <= end_idx)
        nz_idx = nonzero_mask.nonzero(as_tuple=True)[0]
        if nz_idx.numel() == 0:
            continue
        matched_idx = int(nz_idx[0].item())

        num_non_zeros = end_idx - matched_idx + 1
        offset_blc = req_offset_base * real_idx

        new_row = block_table_new_for(block_table_new, pid)
        if num_non_zeros == 1:
            new_row[end_idx] = offset_blc + 2
            current_offset = offset_blc + 2
        else:
            new_row[end_idx - 1] = offset_blc + 1
            new_row[end_idx] = offset_blc + 2
            current_offset = offset_blc + 2

        block_offsets_out = block_offsets_out if hasattr(block_offsets_out, "__setitem__") else block_offsets_out
        block_offsets_out[pid] = int(target_block_id - current_offset)

        if (slot % 128 == 0) and (num_non_zeros > 2):
            move_hbm_flag[pid] = 1


def block_table_new_for(block_table_new, pid):
    return block_table_new[pid]


# ---------------------------------------------------------------------------
# move_hbm_slots_kernel_swa
# For each (layer, request-to-move), shift HBM slots within a layer's pool.
# ---------------------------------------------------------------------------
@_kernel
def move_hbm_slots_kernel_swa(
    pools,             # list of per-layer pool tensors (Python list or stacked tensor)
    indices_to_move,   # Tensor [num_moves] int — which request indices to shift
    real_indices,      # Tensor [num_reqs] int
    req_offset_base,
    num_layers,
    num_to_move,
    *,
    BLOCK_SIZE=None,
    grid=None,
):
    del BLOCK_SIZE, grid

    if not isinstance(pools, (list, tuple)):
        raise RuntimeError(
            "move_hbm_slots_kernel_swa fallback requires a list of per-layer tensors"
        )

    for layer_idx in range(num_layers):
        pool = pools[layer_idx]
        for move_idx in range(num_to_move):
            batch_idx = int(indices_to_move[move_idx].item())
            real_idx = int(real_indices[batch_idx].item())
            offset = req_offset_base * real_idx
            pool[offset + 1].copy_(pool[offset + 2])
            pool[offset + 2].zero_()


__all__ = [
    "update_block_table_kernel_swa",
    "move_hbm_slots_kernel_swa",
    "build_fake_block_table_kernel_compress",
    "move_slots_kernel",
    "move_slots_batched_layers_kernel",
]
