# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MoME state save / restore for PrefillOmniCache."""
import os
import copy
from typing import List

import torch
from vllm.logger import init_logger

logger = init_logger("vllm.v1.omni")


def _copy_mome_slots_to_host(
    state_tensor: torch.Tensor,
    host_layer: torch.Tensor,
    prefill_slots: torch.Tensor,
    dp_local_rank: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> None:
    """Copy per-request MoMe conv_state slots from device HBM to host pool.

    state_tensor: device HBM of shape (num_cache_lines, *inner) indexed by
        MoMe `cache_indices` (i.e. block_id).
    host_layer: mmap-backed host pool slice for (layer_idx, :) with shape
        (dp_world, num_blocks, *host_inner) OR (num_blocks, *host_inner).
        The host slot is page_size_padded bytes wide; MoMe's actual state
        fills a contiguous prefix.
    prefill_slots: 1D int tensor of block_ids to snapshot (one per prefill
        request in this forward pass).

    If shapes don't align (e.g. a non-Pangu host layout), we skip
    quietly; the caller wraps this in a debug-logged try/except.
    """
    if prefill_slots.numel() == 0:
        return

    # Normalise host_layer to (num_blocks, *inner). Pangu V2's shared
    # host pool exposes a leading dp_world dim from `_init_dp_sharding`;
    # squeeze dp=1 out or index the right rank otherwise.
    if host_layer.dim() >= 2 and host_layer.shape[0] == 1:
        hl = host_layer[0]
    elif host_layer.dim() >= 3 and host_layer.shape[0] > dp_local_rank:
        hl = host_layer[dp_local_rank]
    else:
        hl = host_layer

    per_block_flat = hl.reshape(hl.shape[0], -1)  # (num_blocks, inner_elems)
    state_flat = state_tensor.reshape(state_tensor.shape[0], -1)

    # Drop out-of-bounds indices before index_select: MoMe metadata may
    # carry PAD_SLOT_ID (=-1) or stale slot ids that don't map to either
    # the HBM state or the host-pool slot arrays; index_select raises on
    # them otherwise.
    num_state_rows = state_flat.shape[0]
    num_host_blocks = per_block_flat.shape[0]
    src_idx = prefill_slots.to(torch.int64)
    valid = (
        (src_idx >= 0)
        & (src_idx < num_state_rows)
        & (src_idx < num_host_blocks)
    )
    if not bool(valid.all()):
        src_idx = src_idx[valid]
    if src_idx.numel() == 0:
        return

    gathered = state_flat.index_select(0, src_idx).to(
        device="cpu", non_blocking=True
    ).contiguous()
    dst_idx = src_idx.detach().to("cpu")

    n_state_elems = gathered.shape[1]
    # Each TP rank's shard occupies `[rank * n, rank * (n+1))` within
    # the per-block host slot. Without this, every rank would race to
    # write its private shard at offset 0 and clobber the others.
    rank_start = tp_rank * n_state_elems
    rank_end = rank_start + n_state_elems
    if rank_end > per_block_flat.shape[1]:
        return

    per_block_flat[dst_idx, rank_start:rank_end] = gathered.to(per_block_flat.dtype)


def _restore_mome_slots_from_host(
    state_tensor: torch.Tensor,
    host_layer: torch.Tensor,
    request_slots: torch.Tensor,
    dp_local_rank: int,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> None:
    """Task #16 mirror of `_copy_mome_slots_to_host`.

    Copies the host-pool slot into the decode-side device `conv_states`
    tensor at `state_tensor[request_slots[i], ...]`. The host slot was
    laid out by the prefill snapshot as
    `host_layer[block_id][tp_rank * n : (tp_rank + 1) * n]`, so we read
    back this rank's window and write it to the matching cache slot.
    """
    if request_slots_empty(request_slots):
        return

    # Normalise host layout to (num_blocks, *inner). Mirror the shape logic
    # from `_copy_mome_slots_to_host`: a leading dp-world dim when dp=1 is
    # squeezed; otherwise index the caller's local rank.
    if host_layer.dim() >= 2 and host_layer.shape[0] == 1:
        hl = host_layer[0]
    elif host_layer.dim() >= 3 and host_layer.shape[0] > dp_local_rank:
        hl = host_layer[dp_local_rank]
    else:
        hl = host_layer

    per_block_flat = hl.reshape(hl.shape[0], -1)
    state_flat = state_tensor.reshape(state_tensor.shape[0], -1)

    num_state_rows = state_flat.shape[0]
    num_host_blocks = per_block_flat.shape[0]
    dst_idx = request_slots.to(torch.int64)
    valid = (
        (dst_idx >= 0)
        & (dst_idx < num_state_rows)
        & (dst_idx < num_host_blocks)
    )
    if not bool(valid.all()):
        dst_idx = dst_idx[valid]
    if dst_idx.numel() == 0:
        return

    # Read this rank's shard window.
    n_state_elems = state_flat.shape[1]
    rank_start = tp_rank * n_state_elems
    rank_end = rank_start + n_state_elems
    if rank_end > per_block_flat.shape[1]:
        return

    src_idx = dst_idx.to(per_block_flat.device)
    # Clone from host pool rows, move to device.
    gathered = per_block_flat[src_idx, rank_start:rank_end].to(
        device=state_tensor.device, non_blocking=True
    ).to(state_tensor.dtype)

    # Reshape back to the per-slot state shape.
    gathered = gathered.reshape(-1, *state_tensor.shape[1:])
    # Scatter into state_tensor's rows.
    state_tensor.index_copy_(0, dst_idx.to(state_tensor.device), gathered)


def request_slots_empty(t) -> bool:
    return t is None or t.numel() == 0


def _copy_mome_single_state_to_host(
    state_tensor: torch.Tensor,
    host_layer: torch.Tensor,
    prefill_slots: torch.Tensor,
    dp_local_rank: int,
) -> None:
    """Write a single MoMe state's prefill rows into its own group's
    host pool. Each MoMe group has its own `kvi_tensors` entry shaped
    `(num_layers, num_blocks, block_size, head_dim)`. The state
    tensor's leading dim is `num_cache_lines == num_blocks`, so
    `state_tensor[block_id]` and `host_layer[block_id]` line up 1:1;
    we just copy byte-identical content into the flattened row.
    """
    if prefill_slots.numel() == 0:
        return
    if host_layer.dim() == 4 and host_layer.shape[0] == 1:
        hl = host_layer[0]
    elif host_layer.dim() >= 3 and host_layer.shape[0] > dp_local_rank:
        hl = host_layer[dp_local_rank]
    else:
        hl = host_layer

    per_block_flat = hl.reshape(hl.shape[0], -1)
    state_flat = state_tensor.reshape(state_tensor.shape[0], -1)

    num_state_rows = state_flat.shape[0]
    num_host_blocks = per_block_flat.shape[0]
    src_idx = prefill_slots.to(torch.int64)
    valid = (
        (src_idx >= 0)
        & (src_idx < num_state_rows)
        & (src_idx < num_host_blocks)
    )
    if not bool(valid.all()):
        src_idx = src_idx[valid]
    if src_idx.numel() == 0:
        return

    gathered = state_flat.index_select(0, src_idx).to(
        device="cpu", non_blocking=True
    ).contiguous()
    dst_idx_cpu = src_idx.detach().to("cpu")

    n_state_elems = gathered.shape[1]
    if n_state_elems > per_block_flat.shape[1]:
        return

    # Byte-copy: reinterpret both sides as uint8 to avoid dtype mismatch.
    per_block_bytes = per_block_flat.view(torch.uint8).reshape(per_block_flat.shape[0], -1)
    gathered_bytes = gathered.view(torch.uint8).reshape(gathered.shape[0], -1)
    n_bytes = gathered_bytes.shape[1]
    per_block_bytes[dst_idx_cpu, :n_bytes] = gathered_bytes


def _restore_mome_single_state_from_host(
    state_tensor: torch.Tensor,
    host_layer: torch.Tensor,
    request_slots: torch.Tensor,
    dp_local_rank: int,
) -> None:
    """Inverse of `_copy_mome_single_state_to_host` — read bytes from the
    group's host-pool per-block slot and write into the per-group MoMe
    state tensor at matching block_id."""
    if request_slots.numel() == 0:
        return
    if host_layer.dim() == 4 and host_layer.shape[0] == 1:
        hl = host_layer[0]
    elif host_layer.dim() >= 3 and host_layer.shape[0] > dp_local_rank:
        hl = host_layer[dp_local_rank]
    else:
        hl = host_layer

    per_block_flat = hl.reshape(hl.shape[0], -1)
    per_block_bytes = per_block_flat.view(torch.uint8).reshape(per_block_flat.shape[0], -1)

    dst_idx = request_slots.to(torch.int64)
    valid = (
        (dst_idx >= 0)
        & (dst_idx < state_tensor.shape[0])
        & (dst_idx < per_block_bytes.shape[0])
    )
    if not bool(valid.all()):
        dst_idx = dst_idx[valid]
    if dst_idx.numel() == 0:
        return
    dst_cpu = dst_idx.detach().to("cpu")

    state_bytes = (state_tensor.numel() // state_tensor.shape[0]) * state_tensor.element_size()
    host_chunk = per_block_bytes[dst_cpu, :state_bytes].contiguous()
    on_dev = host_chunk.to(device=state_tensor.device, non_blocking=True)
    as_state = on_dev.view(state_tensor.dtype).reshape(
        dst_cpu.shape[0], *state_tensor.shape[1:]
    )
    state_tensor.index_copy_(0, dst_idx.to(state_tensor.device), as_state)


def _copy_mome_states_to_host_unified(
    state_tuple,
    host_layer: torch.Tensor,
    prefill_slots: torch.Tensor,
    dp_local_rank: int,
    max_states: int | None = None,
) -> None:
    """Pack a layer's MoMe state tuple into the unified host pool slot.

    vLLM lays out MambaSpec state within the shared raw_tensor as:
        [state0 bytes | state1 bytes | state2 bytes | padding ...]
    per block of `page_size_padded` bytes. The host pool mirrors the
    same physical bytes, so we write each state's bytes to the same
    offset within the block slot that vLLM uses on the device side.

    host_layer: `(num_blocks, block_size, head_dim)` (or with leading
    dp axis; we pick the correct rank).
    """
    if prefill_slots.numel() == 0:
        return
    state_tuple = tuple(state_tuple)
    if not state_tuple:
        return
    # All three MoMe states are `disable_tp=True` in
    # patch_mome_hybrid.py, so every TP rank holds the full state and
    # 8 ranks write identical bytes — idempotent. `max_states` is
    # retained as a debug knob; pass None to snapshot all states.
    if max_states is not None:
        state_tuple = state_tuple[:max_states]
        if not state_tuple:
            return
    first = state_tuple[0]
    if any(s.shape[0] != first.shape[0] for s in state_tuple):
        return

    # Normalise host_layer to (num_blocks, *inner).
    # Prefill-side pool is 4D `(dp, num_blocks, block_size, head_dim)`
    # (dp = 1 for Pangu V2); decode-side pool is 3D after the die-axis
    # collapse. For 4D pick the right rank; for 3D use as-is.
    if host_layer.dim() == 4:
        hl = host_layer[dp_local_rank] if host_layer.shape[0] > dp_local_rank else host_layer[0]
    else:
        hl = host_layer

    per_block = hl.reshape(hl.shape[0], -1)
    per_block_bytes = per_block.view(torch.uint8).reshape(per_block.shape[0], -1)
    host_bytes_per_block = per_block_bytes.shape[1]

    state_byte_widths = [
        (s.numel() // s.shape[0]) * s.element_size() for s in state_tuple
    ]
    if sum(state_byte_widths) > host_bytes_per_block:
        import os as _os_sz
        if int(_os_sz.environ.get("OMNI_CACHE_MOME_DEBUG", "0")):
            logger.warning(
                "[MOME-SNAP] state bytes %d > slot %d; "
                "widths=%s state_shapes=%s — skipping snapshot",
                sum(state_byte_widths), host_bytes_per_block,
                state_byte_widths, [tuple(s.shape) for s in state_tuple],
            )
        return

    src_idx = prefill_slots.to(torch.int64)
    valid = (
        (src_idx >= 0)
        & (src_idx < state_tuple[0].shape[0])
        & (src_idx < per_block_bytes.shape[0])
    )
    if not bool(valid.all()):
        src_idx = src_idx[valid]
    if src_idx.numel() == 0:
        return
    dst_idx_cpu = src_idx.detach().to("cpu")

    running_off = 0
    for state, width in zip(state_tuple, state_byte_widths):
        # Blocking D2H — the bytes need to be fully materialized on CPU
        # before we write them into the shared-memory host pool, otherwise
        # the OX transfer can race the in-flight copy and send stale bytes.
        gathered_cpu = state.index_select(0, src_idx).to(
            device="cpu", non_blocking=False
        ).contiguous()
        gathered_bytes = gathered_cpu.view(torch.uint8).reshape(
            gathered_cpu.shape[0], -1
        )
        per_block_bytes[dst_idx_cpu, running_off:running_off + width] = gathered_bytes
        running_off += width


def _restore_mome_states_from_host_unified(
    state_tuple,
    host_layer: torch.Tensor,
    request_slots: torch.Tensor,
    dp_local_rank: int,
    max_states: int | None = None,
) -> None:
    """Inverse of `_copy_mome_states_to_host_unified`: read packed MoMe
    bytes out of the per-block host-pool slot back into device state
    tensors. `max_states` caps how many leading states to restore; pass
    the same value used by the matching snapshot.
    """
    if request_slots.numel() == 0:
        return
    state_tuple = tuple(state_tuple)
    if not state_tuple:
        return
    if max_states is not None:
        state_tuple = state_tuple[:max_states]
        if not state_tuple:
            return

    # Normalise host_layer to (num_blocks, *inner). Only 4D host layers
    # carry a dp axis; 3D is already (num_blocks, block_size, head_dim).
    if host_layer.dim() == 4:
        hl = host_layer[dp_local_rank] if host_layer.shape[0] > dp_local_rank else host_layer[0]
    else:
        hl = host_layer

    per_block_flat = hl.reshape(hl.shape[0], -1)
    per_block_bytes = per_block_flat.view(torch.uint8).reshape(per_block_flat.shape[0], -1)

    state_byte_widths = [
        (s.numel() // s.shape[0]) * s.element_size() for s in state_tuple
    ]
    if sum(state_byte_widths) > per_block_bytes.shape[1]:
        import os as _os
        if int(_os.environ.get("OMNI_CACHE_MOME_DEBUG", "0")):
            logger.warning(
                "[MOME-REST] states %dB > slot %dB; widths=%s — skip restore",
                sum(state_byte_widths), per_block_bytes.shape[1],
                state_byte_widths,
            )
        return

    dst_idx = request_slots_to_int64(request_slots)
    valid = (
        (dst_idx >= 0)
        & (dst_idx < state_tuple[0].shape[0])
        & (dst_idx < per_block_bytes.shape[0])
    )
    if not bool(valid.all()):
        dst_idx = dst_idx[valid]
    if dst_idx.numel() == 0:
        return
    dst_cpu = dst_idx.detach().to("cpu")

    running_off = 0
    for state, state_width in zip(state_tuple, state_byte_widths):
        host_chunk = per_block_bytes[dst_cpu, running_off:running_off + state_width].contiguous()
        on_dev = host_chunk.to(device=state.device, non_blocking=False)
        as_state = on_dev.view(state.dtype).reshape(
            dst_cpu.shape[0], *state.shape[1:]
        )
        state.index_copy_(0, dst_idx.to(state.device), as_state)
        running_off += state_width


def request_slots_to_int64(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.int64)

