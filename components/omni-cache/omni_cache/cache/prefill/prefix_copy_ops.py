# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Prefix copy operations for chunked prefill (APC - Automatic Prefix Caching)."""

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from vllm.logger import init_logger

from omni_cache.cache.utils.debug import (
    apc_debug_enabled,
    should_log_rank,
    summarize_array,
)

logger = init_logger("vllm.v1.omni")

if TYPE_CHECKING:
    from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache


def _safe_int(value) -> int:
    try:
        return int(value)
    except TypeError:
        return int(value.item())


class PrefixCopyMeta:
    """Metadata for prefix copy operations."""

    def __init__(
        self,
        consecutive_blocks: List[List[List[int]]],
        query_lens: List[int],
        query_slots: torch.Tensor
    ):
        self.consecutive_blocks = consecutive_blocks
        self.query_lens = query_lens
        self.query_slots = query_slots


def compute_prefix_segments(
    kv_lens: np.ndarray,
    query_lens_list: List[int],
    block_tables: np.ndarray,
    block_size: int,
    arange_cpu: torch.Tensor,
    device: torch.device,
) -> Optional[Tuple[List, List[int], torch.Tensor]]:
    """Compute prefix segments for APC.

    Args:
        kv_lens: Array of KV lengths per request.
        query_lens_list: List of query lengths.
        block_tables: Block tables array.
        block_size: Size of each block.
        arange_cpu: Pre-allocated arange tensor on CPU.
        device: Target device.

    Returns:
        Tuple of (all_segs, query_lens_list, q_slots) or None if no remainder.
    """
    bsz = block_tables.shape[0]
    all_segs, q_slots, q_slot_start = [], [], 0
    last_block_idx, remainder = (kv_lens - 1) // block_size, kv_lens % block_size
    apc_dbg = apc_debug_enabled() and should_log_rank()

    for i in range(bsz):
        m = last_block_idx[i]
        m_int = _safe_int(m)
        kv_len_int = _safe_int(kv_lens[i])
        query_len_int = _safe_int(query_lens_list[i])
        remainder_int = _safe_int(remainder[i])

        if m < 0:
            segs = []
        elif m == 0:
            single_block = block_tables[i, 0].item()
            # Skip block 0 (null/padding block)
            if single_block == 0:
                segs = []
            else:
                segs = [[single_block, single_block + 1]]
        else:
            bt = block_tables[i, :m + 1]
            split_idx = np.where(np.diff(bt, n=1, axis=0) != 1)[0] + 1
            start_idx = np.r_[0, split_idx]
            end_idx = np.r_[split_idx - 1, m]

            start_blocks = bt[start_idx]
            end_blocks = bt[end_idx] + 1
            segs = np.stack([start_blocks, end_blocks], axis=1).tolist()

        if apc_dbg:
            bt_prefix = block_tables[i, :m_int + 1] if m_int >= 0 else []
            logger.warning(
                "[APCDBG/PREFIX_SEG] tp_rank=%s dp_rank=%s stage=%s "
                "group_idx=%s layer_name=%s req_idx=%d kv_len=%d query_len=%d "
                "last_block_idx=%d remainder=%d %s raw_segs=%s",
                None,
                None,
                None,
                None,
                None,
                i,
                kv_len_int,
                query_len_int,
                m_int,
                remainder_int,
                summarize_array("bt_prefix", bt_prefix),
                segs,
            )

        all_segs.append(segs)
        q_slot_start += kv_lens[i]
        q_slots.append(arange_cpu[:query_lens_list[i]] + q_slot_start)
        q_slot_start += query_lens_list[i]

    q_slots = torch.cat(q_slots).to(device=device, non_blocking=True)

    # Filter: skip block 0 and deduplicate across all requests
    seen = set()
    filtered_all_segs = []
    for req_segs in all_segs:
        filtered_req_segs = []
        for seg in req_segs:
            start, end = seg
            if start == 0:  # Skip block 0
                continue
            seg_tuple = (start, end)
            if seg_tuple not in seen:
                seen.add(seg_tuple)
                filtered_req_segs.append(seg)
        filtered_all_segs.append(filtered_req_segs)
        if apc_dbg:
            logger.warning(
                "[APCDBG/PREFIX_SEG] tp_rank=%s dp_rank=%s stage=%s "
                "group_idx=%s layer_name=%s req_idx=%d filtered_segs=%s",
                None,
                None,
                None,
                None,
                None,
                len(filtered_all_segs) - 1,
                filtered_req_segs,
            )

    return filtered_all_segs, query_lens_list, q_slots


def get_current_rank_host_data(
    cache: "PrefillOmniCache",
    layer_idx: int,
    prefix_meta: PrefixCopyMeta,
) -> torch.Tensor:
    """Get host data for current rank during prefix copy.

    Args:
        cache: The PrefillOmniCache instance.
        layer_idx: Current layer index.
        prefix_meta: Prefix copy metadata.

    Returns:
        Concatenated host data tensor.
    """
    from omni_cache.cache.core.constants import LOCAL_DP_SIZE

    block_ids = []
    for block_ranges in prefix_meta.consecutive_blocks:
        for start_block_id, end_block_id in block_ranges:
            block_ids.extend(range(start_block_id, end_block_id))

    num_heads_per_node = len(block_ids) * cache.node_block_size
    num_heads_per_rank = num_heads_per_node // LOCAL_DP_SIZE

    current_rank_block_ids_start_idx = (
        cache.tp_local_rank * num_heads_per_rank // cache.node_block_size
    )
    current_rank_block_ids_end_idx = (
        (cache.tp_local_rank + 1) * num_heads_per_rank // cache.node_block_size
    )
    block_ids_of_this_rank = block_ids[
        current_rank_block_ids_start_idx:current_rank_block_ids_end_idx + 1
    ]

    offset_in_block = cache.tp_local_rank * num_heads_per_rank % cache.node_block_size
    block_id = block_ids_of_this_rank[0]
    head_offset = block_id * cache.node_block_size + offset_in_block

    remain_heads = num_heads_per_rank
    num_copyed_heads = min(remain_heads, cache.node_block_size - offset_in_block)
    remain_heads -= num_copyed_heads

    kvi_tensors = []
    host_data = []
    for _i, tensor in enumerate(cache.host_cache.kvi_tensors):
        host_data.append([])
        kvi_tensors.append(tensor.view(cache.num_layers, -1, tensor.shape[-1]))

    for index, block_id in enumerate(block_ids_of_this_rank[:-1]):
        if block_ids_of_this_rank[index + 1] == block_id + 1:
            num_copying_heads = min(remain_heads, cache.node_block_size)
            num_copyed_heads += num_copying_heads
            remain_heads -= num_copying_heads
        else:
            for idx, tensor in enumerate(kvi_tensors):
                host_data[idx].append(
                    tensor[layer_idx, head_offset:head_offset + num_copyed_heads]
                )

            block_id = block_ids_of_this_rank[index + 1]
            head_offset = block_id * cache.node_block_size
            num_copyed_heads = min(remain_heads, cache.node_block_size)
            remain_heads -= num_copying_heads

    for idx, tensor in enumerate(kvi_tensors):
        host_data[idx].append(tensor[layer_idx, head_offset:head_offset + num_copyed_heads])
        host_data[idx] = torch.concat(host_data[idx])

    host_data = torch.concat(host_data, dim=-1)
    return host_data
