# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import ctypes
import math
from typing import Optional

import numpy as np
import torch
from numba import njit
from vllm.v1.kv_cache_interface import KVCacheConfig

_current_stream = None


def divide_or_raise(a: int, b: int):
    if a % b != 0:
        raise ValueError(f"Error! Number 'a' {a} is not divisible by number 'b' {b}.")
    return a // b


def _is_hybrid_attention_enabled(kv_cache_config: Optional[KVCacheConfig]) -> bool:
    if kv_cache_config is None:
        return False
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not groups:
        return False
    return len(groups) > 1


def calculate_copy_block_stats(consecutive_blocks: list[list[tuple[int, int]]]) -> tuple[int, int]:
    flatten = [pair[1] - pair[0] for segs in consecutive_blocks for pair in segs]
    return len(flatten), sum(flatten)


def generate_full_block_slot(slot_mapping, query_lens, block_size):
    blocks = slot_mapping // block_size
    device = slot_mapping.device
    index_per_block = torch.arange(block_size, dtype=slot_mapping.dtype, device=device)
    result = []
    start = 0
    for query_len in query_lens:
        end = start + query_len
        num_block = math.ceil((end - start) / block_size)
        query_blocks = blocks[start:end]
        block_index = torch.arange(num_block, device=device) * block_size
        query_blocks = query_blocks[block_index]
        query_slot = index_per_block.repeat(num_block, 1)
        query_slot = query_slot + (query_blocks * block_size).unsqueeze(1)
        result.append(query_slot)
        start = end
    return torch.concat(result, dim=0).view(-1)


def pad_inputs(input: torch.Tensor, query_lens: list[int], sp_size: int, pad_value: int):
    count = 0
    res = []
    for length in query_lens:
        pad_size = (sp_size - length % sp_size) % sp_size
        tmp_tensor = input[count:count + length]
        padded_tensor = pad_tensor(tmp_tensor, pad_size, pad_value)
        res.append(padded_tensor)
        count += length
    return torch.cat(res, dim=0)


def pad_tensor(tensor, pad_size, pad_value=0):
    padded_shape = (pad_size, tensor.shape[-1]) if tensor.dim() > 1 else (pad_size,)
    padding = torch.full(
        padded_shape,
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding])


def torch_to_numpy_zero_copy(tensor):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    ptr = tensor.data_ptr()
    dtype_map = {
        torch.int32: ctypes.c_int32,
        torch.int64: ctypes.c_int64,
        torch.float32: ctypes.c_float,
        torch.float64: ctypes.c_double,
    }
    ctype = dtype_map.get(tensor.dtype)
    if ctype is None:
        raise TypeError(f"Unsupported dtype: {tensor.dtype}")

    total_elements = tensor.numel()
    c_array = (ctype * total_elements).from_address(ptr)
    np_array = np.ctypeslib.as_array(c_array)
    return np_array.reshape(tensor.shape)


@njit(cache=True)
def _batch_fill_block_table_with_reorder(
    block_table: np.ndarray,
    base_addrs: np.ndarray,
    logic_to_phys_flat: np.ndarray,
    logic_valid_flat: np.ndarray,
    L_currs: np.ndarray,
    win_sizes: np.ndarray,
    num_assigneds: np.ndarray,
    max_lbs: np.ndarray,
    batch_state_indices: np.ndarray,
    max_logic_blocks: int,
) -> None:
    num_reqs = base_addrs.shape[0]

    for i in range(num_reqs):
        base = int(base_addrs[i])
        state_idx = int(batch_state_indices[i])
        num_assigned = int(num_assigneds[state_idx])
        max_lb = int(max_lbs[state_idx])
        win_size = int(win_sizes[state_idx])
        L_curr = int(L_currs[i])

        if L_curr > max_lb and max_lb != -1:
            win_size = min(3, win_size + 1)
            win_sizes[state_idx] = win_size

        start_lb_idx = L_curr - (win_size - 1)
        if start_lb_idx < 0:
            start_lb_idx = 0

        offset = state_idx * max_logic_blocks

        for lb_idx in range(start_lb_idx, L_curr + 1):
            if not logic_valid_flat[offset + lb_idx]:
                assigned_id = (num_assigned % 3) + 1
                logic_to_phys_flat[offset + lb_idx] = assigned_id
                logic_valid_flat[offset + lb_idx] = True
                num_assigned += 1

            block_table[i, lb_idx] = base + int(logic_to_phys_flat[offset + lb_idx])

            if lb_idx > max_lb:
                max_lb = lb_idx

        num_assigneds[state_idx] = num_assigned
        max_lbs[state_idx] = max_lb


@njit(cache=True)
def _batch_fill_slot_mapping(
    slot_mapping: np.ndarray,
    q_starts: np.ndarray,
    q_lens: np.ndarray,
    last_seq_lens: np.ndarray,
    base_addrs: np.ndarray,
    state_indices: np.ndarray,
    logic_to_phys_full_buffer: np.ndarray,
    max_logic_blocks: int,
) -> None:
    num_reqs = len(q_starts)

    for i in range(num_reqs):
        q_start = int(q_starts[i])
        q_len = int(q_lens[i])
        last_seq_len = int(last_seq_lens[i])
        base = int(base_addrs[i])
        state_idx = int(state_indices[i])
        offset = state_idx * max_logic_blocks

        start_lb = last_seq_len // 128
        start_offset = last_seq_len % 128

        pos = 0
        curr_lb = start_lb
        curr_offset = start_offset

        while pos < q_len:
            space = 128 - curr_offset
            chunk = min(q_len - pos, space)
            phys_id = int(logic_to_phys_full_buffer[offset + curr_lb])
            slot_base = (base + phys_id) * 128 + curr_offset

            for j in range(chunk):
                slot_mapping[q_start + pos + j] = slot_base + j

            pos += chunk
            curr_lb += 1
            curr_offset = 0


def current_stream() -> torch.npu.Stream:
    """Return the current NPU stream, avoiding repeated object creation.

    ``torch.npu.current_stream()`` constructs a new stream object per call,
    which is expensive.  We cache the stream reference so that callers can
    use this function instead.
    """
    global _current_stream
    if _current_stream is None:
        _current_stream = torch.npu.current_stream()
    return _current_stream


__all__ = [
    "divide_or_raise",
    "_is_hybrid_attention_enabled",
    "calculate_copy_block_stats",
    "generate_full_block_slot",
    "pad_inputs",
    "pad_tensor",
    "torch_to_numpy_zero_copy",
    "_batch_fill_block_table_with_reorder",
    "_batch_fill_slot_mapping",
]
