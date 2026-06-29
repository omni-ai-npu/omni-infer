# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Static utility methods for decode cache operations."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch

from omni_cache.cache.utils.ops import torch_to_numpy_zero_copy

if TYPE_CHECKING:
    from vllm.v1.worker.block_table import BlockTable
    from omni_npu.worker.npu_model_runner import NPUModelRunner
    from omni_cache.cache.decode import DecodeOmniCache


def record_current_batch_order(input_batch, omni_cache: "DecodeOmniCache") -> None:
    """Record the current batch order for request tracking.

    Updates ``req_ids_update_buffer`` and ``id_to_idx_table`` so that
    downstream consumers (e.g. ``resolve_batch_idx_reqs`` for HBM lane
    resolution, block-table / slot-mapping builders) see the correct
    request-to-lane mapping for the current step. In the legacy path this also
    rebuilds ``req_id_to_idx`` from ``record_batch_idx_to_req``; with
    OmniCacheInputBatch enabled, H2D owns ``req_id_to_idx``.

    This is the **core** batch-order tracking logic, decoupled from
    gather-selection-specific cleanup which lives in
    ``GatherSelectionUpdater.record_current_batch_order``.

    Args:
        input_batch: The input batch containing request information.
        omni_cache: The OmniCache instance to update.
    """
    if not omni_cache.is_hybrid_attn:
        return
    if getattr(omni_cache, "record_batch_idx_to_req", None) is None:
        return

    # Step 1: Update request ID buffers from input batch
    req_ids_update_mapping = input_batch.req_id_to_index
    req_ids_update = [req_id for req_id, _ in sorted(req_ids_update_mapping.items(), key=lambda item: item[1])]

    if getattr(omni_cache, "req_ids_update_buffer", None) is None:
        omni_cache.req_ids_update_buffer = None
        omni_cache.req_ids_record_buffer = None
    else:
        omni_cache.req_ids_record_buffer = omni_cache.req_ids_update_buffer

    omni_cache.req_ids_update_buffer = req_ids_update

    use_input_batch_lane_mapping = int(_o.getenv("USE_OMNI_INPUT_BATCH", "0")) == 1

    # Step 2+3: Clean up stale legacy lane entries and, for the legacy path,
    # rebuild the reverse mapping. With OmniCacheInputBatch enabled, H2D owns
    # req_id_to_idx, so this function must not rewrite it.
    with omni_cache._lane_lock:
        if omni_cache.req_ids_record_buffer is not None:
            for idx_req_buffer, req_id in omni_cache.record_batch_idx_to_req.items():
                if req_id not in req_ids_update and req_id in omni_cache.req_ids_record_buffer:
                    omni_cache.record_batch_idx_to_req[idx_req_buffer] = None

        if not use_input_batch_lane_mapping:
            # Step 3: Rebuild req_id_to_idx from record_batch_idx_to_req.
            omni_cache.req_id_to_idx = {
                req_id: idx for idx, req_id in omni_cache.record_batch_idx_to_req.items() if req_id is not None
            }

    # Step 4: Synchronize id_to_idx_table to device
    if omni_cache.req_ids_record_buffer != omni_cache.req_ids_update_buffer:
        for i, req_id in enumerate(omni_cache.req_ids_update_buffer):
            omni_cache.indices_cpu_buffer[i] = omni_cache.req_id_to_idx.get(req_id, -1)

        num_reqs = len(omni_cache.req_ids_update_buffer)
        # `non_blocking=False` is intentional. `indices_cpu_buffer` is mutated
        # in-place every step, and the attention forward that follows reads
        # `id_to_idx_table` on the device to route per-request HBM lanes. An
        # async copy can race with either the next step's buffer rewrite or
        # the forward's kernel dispatch, so under concurrent 2-in-flight
        # requests on the same DP engine we saw one request read the other's
        # (or a stale) HBM lane and emit garbage. Forcing the copy to be
        # synchronous here costs a few hundred bytes of host-to-device DMA
        # per step — negligible at our BSZ — and eliminates the race.
        omni_cache.id_to_idx_table[:num_reqs].copy_(
            omni_cache.indices_cpu_buffer[:num_reqs],
            non_blocking=False,
        )


def get_block_table_np(
    runner: "NPUModelRunner",
    omni_cache: "DecodeOmniCache",
    blk_table: "BlockTable",
    kv_cache_gid: int,
    blk_table_tensor: Any,
    num_reqs_padded: int,
    num_tokens_padded: int,
) -> tuple:
    """Get numpy arrays for block table and slot mapping.

    Args:
        runner: The NPU model runner.
        omni_cache: The OmniCache instance.
        blk_table: The block table.
        kv_cache_gid: KV cache group ID.
        blk_table_tensor: Block table tensor.
        num_reqs_padded: Number of padded requests.
        num_tokens_padded: Number of padded tokens.

    Returns:
        Tuple of (blk_table_np, slot_mapping_np) numpy arrays.
    """
    if not hasattr(omni_cache, "blk_table_buffers"):
        omni_cache.blk_table_buffers = {}
    if kv_cache_gid not in omni_cache.blk_table_buffers:
        omni_cache.blk_table_buffers[kv_cache_gid] = torch.empty(
            (runner.vllm_config.scheduler_config.max_num_seqs, blk_table_tensor.shape[1]),
            dtype=torch.int32,
            pin_memory=True,
        )

    buf = torch_to_numpy_zero_copy(omni_cache.blk_table_buffers[kv_cache_gid])
    buf[:num_reqs_padded] = blk_table.get_numpy_array()[:num_reqs_padded]
    blk_table_np = buf[:num_reqs_padded]

    if not hasattr(omni_cache, "slot_mapping_buffers"):
        omni_cache.slot_mapping_buffers = {}
    if kv_cache_gid not in omni_cache.slot_mapping_buffers:
        max_len = runner.vllm_config.scheduler_config.max_num_seqs * (omni_cache.num_spec_token + 1)
        omni_cache.slot_mapping_buffers[kv_cache_gid] = torch.empty(max_len, dtype=torch.int64, pin_memory=True)

    buf = torch_to_numpy_zero_copy(omni_cache.slot_mapping_buffers[kv_cache_gid])
    if num_tokens_padded > buf.shape[0]:
        raise ValueError(f"num_tokens_padded({num_tokens_padded}) > buffer size({buf.shape[0]})")
    buf.fill(-1)
    slot_mapping_np = buf[:num_tokens_padded]

    return blk_table_np, slot_mapping_np


def update_status_buffered(
    omni_cache: "DecodeOmniCache",
    reshaped_status: Any,
    old_req_ids: List[Optional[str]],
    new_req_ids: List[Optional[str]],
    *,
    fill_value: Any = -1,
) -> None:
    """Update status buffer with reordered request IDs.

    Args:
        omni_cache: The OmniCache instance.
        reshaped_status: Reshaped status tensor.
        old_req_ids: Old request IDs.
        new_req_ids: New request IDs.
        fill_value: Value to use for filling.
    """
    from omni_cache.gather_selection.status_updater.updater import GatherSelectionUpdater

    GatherSelectionUpdater._update_status_buffered(
        omni_cache,
        reshaped_status,
        old_req_ids,
        new_req_ids,
        fill_value=fill_value,
    )


def reorder_block_table_only(
    omni_cache: "DecodeOmniCache", reshaped_table: Any, old_req_ids: List[str], new_req_ids: List[str]
) -> None:
    """Reorder block table with new request IDs.

    Args:
        omni_cache: The OmniCache instance.
        reshaped_table: Reshaped block table.
        old_req_ids: Old request IDs.
        new_req_ids: New request IDs.
    """
    from omni_cache.gather_selection.status_updater.updater import GatherSelectionUpdater

    GatherSelectionUpdater._reorder_block_table_only(
        omni_cache,
        reshaped_table,
        old_req_ids,
        new_req_ids,
    )


def maybe_update_selection_kv_block_status(
    input_batch, omni_cache: "DecodeOmniCache", num_scheduled_tokens: int
) -> None:
    """Maybe update selection KV block status.

    Args:
        input_batch: The input batch.
        omni_cache: The OmniCache instance.
        num_scheduled_tokens: Number of scheduled tokens.
    """
    from omni_cache.gather_selection.status_updater.updater import GatherSelectionUpdater

    GatherSelectionUpdater.maybe_update_selection_kv_block_status(
        input_batch,
        omni_cache,
        num_scheduled_tokens,
    )
