# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Sliding window attention utilities for hybrid attention.

This module provides sliding window attention functions for fake attention metadata
in hybrid attention mode, managing request window states and block table
mappings.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Set

import numpy as np
import torch

from omni_cache.cache.utils.ops import (
    _batch_fill_block_table_with_reorder,
    _batch_fill_slot_mapping,
    torch_to_numpy_zero_copy,
)

if TYPE_CHECKING:
    from omni_cache.cache.core.base import RequestWindowState


def _cleanup_departed_states(cache, cur_metadata_grp_id: int) -> None:
    """Remove states for requests that left the batch.

    Identifies and removes window states for requests no longer
    in the current batch for this metadata group.

    Args:
        cache: The DecodeOmniCache instance
        cur_metadata_grp_id: Current metadata group ID
    """
    req_ids_list = cache.req_ids_update_buffer if hasattr(cache, "req_ids_update_buffer") else []
    current_req_ids = set(req_ids_list)

    sks_to_remove = []
    for sk, state_idx in list(cache._sk_to_state_idx.items()):
        rid, grp_id = sk
        if grp_id == cur_metadata_grp_id and rid not in current_req_ids:
            sks_to_remove.append((sk, state_idx))

    # Free those states
    for sk, state_idx in sks_to_remove:
        rid, grp_id = sk
        if sk in cache.req_window_states:
            cache.req_window_states.pop(sk)
        cache.last_num_non_zeros.pop(sk, None)
        if rid in cache.record_req_ids_swa_block:
            cache.record_req_ids_swa_block.pop(rid)
        del cache._sk_to_state_idx[sk]
        cache._free_state_indices.add(state_idx)


def _compute_request_metadata(
    cache, num_reqs: int, query_start_loc: np.ndarray, seq_lens_cpu: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute metadata for all requests in the batch.

    Calculates sequence lengths, L_curr values, query positions,
    and other metadata needed for window state management.

    Args:
        cache: The DecodeOmniCache instance
        num_reqs: Number of requests in the batch
        query_start_loc: Query start locations per request
        seq_lens_cpu: Sequence lengths on CPU

    Returns:
        Tuple of (actual_num_reqs, req_ids, seq_lens_batch, L_currs, q_starts, q_lens)
    """
    actual_num_reqs = min(num_reqs, len(cache.req_ids_update_buffer))
    if actual_num_reqs == 0:
        return 0, None, None, None, None, None

    req_ids = cache.req_ids_update_buffer[:actual_num_reqs]
    seq_lens_batch = seq_lens_cpu[:actual_num_reqs]
    L_currs = (seq_lens_batch - 1) // 128

    q_starts = query_start_loc[:actual_num_reqs]
    q_ends = query_start_loc[1:actual_num_reqs + 1]
    q_lens = q_ends - q_starts

    return actual_num_reqs, req_ids, seq_lens_batch, L_currs, q_starts, q_lens


def _lookup_or_allocate_state_indices(
    cache, req_ids: np.ndarray, L_currs: np.ndarray, cur_metadata_grp_id: int
) -> Tuple[np.ndarray, List[Tuple]]:
    """Look up existing state indices or allocate new ones.

    For each request, either finds its existing state index or allocates
    a new one from the free pool.

    Args:
        cache: The DecodeOmniCache instance
        req_ids: Array of request IDs
        L_currs: Array of L_curr values
        cur_metadata_grp_id: Current metadata group ID

    Returns:
        Tuple of (batch_state_indices, new_state_indices)
        - batch_state_indices: Array of state indices for all requests
        - new_state_indices: List of tuples for newly allocated states
    """
    num_reqs = len(req_ids)
    batch_state_indices = cache._batch_state_indices_buffer[:num_reqs]
    new_state_indices = []

    for i in range(num_reqs):
        rid = req_ids[i]
        sk = (rid, cur_metadata_grp_id)
        L_curr = L_currs[i]

        if sk in cache._sk_to_state_idx:
            batch_state_indices[i] = cache._sk_to_state_idx[sk]
        else:
            if not cache._free_state_indices:
                raise RuntimeError("No free state buffers available")
            state_idx = cache._free_state_indices.pop()
            cache._sk_to_state_idx[sk] = state_idx
            batch_state_indices[i] = state_idx
            new_state_indices.append((i, rid, L_curr, sk, state_idx))

    return batch_state_indices, new_state_indices


def _check_state_switching(cache, rid: str, cur_metadata_grp_id: int, block_table: np.ndarray, batch_idx: int) -> bool:
    """Check if a request is switching window states.

    Determines if the physical block start has changed, indicating
    a window state transition.

    Args:
        cache: The DecodeOmniCache instance
        rid: Request ID
        cur_metadata_grp_id: Current metadata group ID
        block_table: Block table array
        batch_idx: Index in the batch for this request

    Returns:
        True if switching state, False otherwise
    """
    recorded_infos = cache.record_req_ids_swa_block.get(rid, None)
    if recorded_infos is None:
        return False

    recorded_info = recorded_infos[cur_metadata_grp_id]
    row_orig = block_table[batch_idx]
    active_physical_ids = row_orig[row_orig > 0]
    current_phys_start = active_physical_ids[0] if active_physical_ids.size > 0 else -1

    return current_phys_start != -1 and recorded_info[0] is not None and current_phys_start != recorded_info[0]


def _clone_switching_pool_buffers(cache, cur_metadata_grp_id: int, base_addr: int) -> None:
    """Clone pool buffers when switching with init_win=2.

    Copies pool[base_addr + 2] to pool[base_addr + 1] for all layers
    when a state transition occurs.

    Args:
        cache: The DecodeOmniCache instance
        cur_metadata_grp_id: Current metadata group ID
        base_addr: Base address in the pool
    """
    offset_blc = base_addr
    layers = cache.kv_cache_config.kv_cache_groups[cur_metadata_grp_id].layer_names
    for ln in layers:
        pool = cache.hbm_buffer_pool[cur_metadata_grp_id][ln]
        pool[offset_blc + 1] = pool[offset_blc + 2].clone()


def _initialize_window_state(
    cache,
    batch_idx: int,
    rid: str,
    L_curr: int,
    sk: tuple,
    state_idx: int,
    cur_metadata_grp_id: int,
    block_table: np.ndarray,
) -> "RequestWindowState":
    """Initialize a new window state for a request.

    Sets up buffer values, handles switching logic, and creates
    the RequestWindowState object.

    Args:
        cache: The DecodeOmniCache instance
        batch_idx: Index in the batch
        rid: Request ID
        L_curr: Current L value
        sk: State key (rid, grp_id)
        state_idx: Allocated state index
        cur_metadata_grp_id: Current metadata group ID
        block_table: Block table array

    Returns:
        Initialized RequestWindowState object
    """
    idx_in_pool = cache.req_id_to_idx.get(rid, -1)
    pool_unit = cache.hbm_buffer_block_table_pool[cur_metadata_grp_id]["req_offset"]
    init_win = 2 if L_curr >= 2 else 1
    base_addr = pool_unit * idx_in_pool

    # Check if switching now
    is_switching_now = _check_state_switching(cache, rid, cur_metadata_grp_id, block_table, batch_idx)

    # If switching and init_win == 2, clone pool buffers
    if init_win == 2 and is_switching_now:
        _clone_switching_pool_buffers(cache, cur_metadata_grp_id, base_addr)

    cache._base_addrs_buffer[state_idx] = base_addr
    cache._win_sizes_buffer[state_idx] = init_win
    cache._max_lbs_buffer[state_idx] = -1
    cache._num_assigneds_buffer[state_idx] = 0
    cache._is_switching_buffer[state_idx] = is_switching_now

    # Initialize logic buffers
    offset = state_idx * cache.MAX_LOGIC_BLOCKS
    cache._logic_to_phys_buffer[offset:offset + cache.MAX_LOGIC_BLOCKS].fill(0)
    cache._logic_valid_buffer[offset:offset + cache.MAX_LOGIC_BLOCKS].fill(False)

    # Create state object
    from omni_cache.cache.core.base import RequestWindowState

    state = RequestWindowState(
        base=base_addr,
        win_size=init_win,
        state_idx=state_idx,
        _omni_cache_ref=cache,
    )
    cache.req_window_states[sk] = state
    return state


def _prepare_slot_infos(
    cache,
    q_starts: np.ndarray,
    q_lens: np.ndarray,
    seq_lens_batch: np.ndarray,
    base_addrs: np.ndarray,
    batch_state_indices: np.ndarray,
    actual_num_reqs: int,
) -> List["_SlotInfo"]:
    """Prepare slot info objects for slot mapping computation.

    Creates _SlotInfo objects for requests with valid query lengths.

    Args:
        cache: The DecodeOmniCache instance
        q_starts: Query start positions
        q_lens: Query lengths
        seq_lens_batch: Sequence lengths
        base_addrs: Base addresses for each request
        batch_state_indices: State indices for each request
        actual_num_reqs: Actual number of requests

    Returns:
        List of _SlotInfo objects
    """
    slot_infos = []
    last_seq_lens = seq_lens_batch - q_lens
    valid_mask = q_lens > 0

    if not valid_mask.any():
        return slot_infos

    from omni_cache.cache.core.base import RequestWindowState

    valid_indices = np.where(valid_mask)[0]

    for i in valid_indices:
        temp_state = RequestWindowState(
            base=base_addrs[i],
            win_size=cache._win_sizes_buffer[batch_state_indices[i]],
            state_idx=batch_state_indices[i],
            _omni_cache_ref=cache,
        )
        slot_infos.append(_SlotInfo(int(q_starts[i]), int(q_lens[i]), int(last_seq_lens[i]), temp_state))

    return slot_infos


def post_process_fake_metadata(
    cache,
    num_reqs: int,
    slot_mapping: np.ndarray,
    block_table: np.ndarray,
    common_attn_metadata,
    cur_metadata_grp_id: int,
    query_start_loc: np.ndarray,
    seq_lens_cpu: np.ndarray,
    draft_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-process fake metadata for hybrid attention H2D transfer.

    This function manages request window states, block table mappings, and
    slot mappings for requests in hybrid attention mode. It handles state
    cleanup for departed requests and initializes new states for new requests.

    Args:
        cache: The DecodeOmniCache instance
        num_reqs: Number of requests in the batch
        slot_mapping: Slot mapping array to be filled
        block_table: Block table array to be filled
        common_attn_metadata: Common attention metadata
        cur_metadata_grp_id: Current metadata group ID
        query_start_loc: Query start locations per request
        seq_lens_cpu: Sequence lengths on CPU
        draft_index: Optional draft index for MTP

    Returns:
        Tuple of (block_table, slot_mapping) after processing
    """
    # Delay initialization of req_window_states until first use
    if not hasattr(cache, "req_window_states"):
        cache.req_window_states: Dict[Tuple, "RequestWindowState"] = {}

    # Step 1: Clean up states for requests that left the batch
    _cleanup_departed_states(cache, cur_metadata_grp_id)

    # Step 2: Compute request metadata
    actual_num_reqs, req_ids, seq_lens_batch, L_currs, q_starts, q_lens = _compute_request_metadata(
        cache, num_reqs, query_start_loc, seq_lens_cpu
    )

    if actual_num_reqs == 0:
        return block_table, slot_mapping

    # Step 3: Look up or allocate state indices
    batch_state_indices, new_state_indices = _lookup_or_allocate_state_indices(
        cache, req_ids, L_currs, cur_metadata_grp_id
    )

    # Step 4: Initialize new states
    for batch_idx, rid, L_curr, sk, state_idx in new_state_indices:
        _initialize_window_state(cache, batch_idx, rid, L_curr, sk, state_idx, cur_metadata_grp_id, block_table)

    # Step 5: Collect base addresses and prepare for block table fill
    base_addrs = cache._base_addrs_buffer[batch_state_indices]
    cache._L_currs_buffer[:actual_num_reqs] = L_currs

    # Step 6: Fill block tables
    _batch_fill_block_table_with_reorder(
        block_table,
        base_addrs,
        cache._logic_to_phys_buffer,
        cache._logic_valid_buffer,
        cache._L_currs_buffer[:actual_num_reqs],
        cache._win_sizes_buffer,
        cache._num_assigneds_buffer,
        cache._max_lbs_buffer,
        batch_state_indices,
        cache.MAX_LOGIC_BLOCKS,
    )

    # Step 7: Prepare and compute slot mapping
    slot_infos = _prepare_slot_infos(
        cache, q_starts, q_lens, seq_lens_batch, base_addrs, batch_state_indices, actual_num_reqs
    )
    _compute_slot_mapping_vectorized(cache, slot_mapping, slot_infos)

    return block_table, slot_mapping


class _SlotInfo:
    """Minimal dataclass for slot mapping information."""

    __slots__ = ["q_start", "q_len", "last_seq_len", "state"]

    def __init__(self, q_start: int, q_len: int, last_seq_len: int, state):
        self.q_start = q_start
        self.q_len = q_len
        self.last_seq_len = last_seq_len
        self.state = state


def _compute_slot_mapping_vectorized(
    cache,
    slot_mapping: np.ndarray,
    slot_infos: List["_SlotInfo"],
) -> None:
    """Vectorized slot mapping computation.

    Eliminates Python loop by collecting all data into arrays
    and using a single Numba JIT call.
    """
    if not slot_infos:
        return

    num_slot_infos = len(slot_infos)

    # Reuse pre-allocated buffers
    slot_q_starts = np.zeros(num_slot_infos, dtype=np.int64)
    slot_q_lens = np.zeros(num_slot_infos, dtype=np.int64)
    slot_last_seq_lens = np.zeros(num_slot_infos, dtype=np.int64)
    slot_base_addrs = np.zeros(num_slot_infos, dtype=np.int32)
    slot_state_indices = np.zeros(num_slot_infos, dtype=np.int32)

    for i, info in enumerate(slot_infos):
        slot_q_starts[i] = info.q_start
        slot_q_lens[i] = info.q_len
        slot_last_seq_lens[i] = info.last_seq_len
        slot_base_addrs[i] = info.state.base
        slot_state_indices[i] = info.state.state_idx

    _batch_fill_slot_mapping(
        slot_mapping,
        slot_q_starts,
        slot_q_lens,
        slot_last_seq_lens,
        slot_base_addrs,
        slot_state_indices,
        cache._logic_to_phys_buffer,
        cache.MAX_LOGIC_BLOCKS,
    )
