# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Gather Selection Updater for managing KV cache selection updates.

This module contains the GatherSelectionUpdater class which is responsible
for updating block table mappings and managing selection KV cache status
for dynamic gather selection during decode.
"""

import time
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class GatherSelectionUpdater:
    """Updater for gather selection block table and status management.

    This class manages the updates to selection KV cache block tables
    and statuses based on request ID changes and block table mappings.
    """

    def __init__(self, decode_omni_cache):
        """Initialize the updater with a reference to the decode cache.

        Args:
            decode_omni_cache: The DecodeOmniCache instance to manage.
        """
        self.decode_omni_cache = decode_omni_cache

    def updater(self, block_table_tensor: torch.Tensor) -> None:
        """Update block tables based on the current and previous state.

        Compares the current block table tensor with the recorded state
        and updates the selection KV block status and block table accordingly.

        Args:
            block_table_tensor: The current block table tensor.
        """
        # Convert block_table_tensor to a list of block tables, excluding zeros.
        block_tables = [sub[sub != 0].tolist() for sub in block_table_tensor]
        block_tables_record = getattr(self.decode_omni_cache, "block_tables_record", None)
        if block_tables_record is None:
            block_tables_record = block_tables.copy()

        req_ids_record = list(range(len(block_tables_record) + 1))
        req_ids_update = list(range(len(block_tables) + 1))

        for req_idx, block_table in enumerate(block_tables):
            block_table_tuple = tuple(block_table)

            if len(block_table) == 1 and self.decode_omni_cache.block_status.get(block_table_tuple, False) is False:
                req_ids_update[req_idx] = -1
                self.decode_omni_cache.block_status[block_table_tuple] = True
                continue

            for req_idx_rec, block_table_rec in enumerate(block_tables_record):
                if len(block_table) < len(block_table_rec):
                    continue

                if block_table[:len(block_table_rec)] == block_table_rec:
                    req_ids_update[req_idx] = req_ids_record[req_idx_rec]
                    self.decode_omni_cache.block_status[block_table_tuple] = True
                    break
            else:
                req_ids_update[req_idx] = -1
                self.decode_omni_cache.block_status[block_table_tuple] = True

        if req_ids_record != req_ids_update:
            num_layers = self.decode_omni_cache.selection_kv_block_status.shape[0]
            head_num = self.decode_omni_cache.selection_kv_block_status.shape[-2]
            topk_plus_1 = self.decode_omni_cache.selection_kv_block_status.shape[-1]
            num_tokens_per_req = self.decode_omni_cache.seq_len

            reshaped_view = self.decode_omni_cache.selection_kv_block_status.view(
                num_layers, -1, num_tokens_per_req, head_num, topk_plus_1
            )
            self._update_status_buffered(
                self.decode_omni_cache, reshaped_view, req_ids_record, req_ids_update, fill_value=-1
            )

            s_max_block_num = self.decode_omni_cache.selection_kv_block_table.shape[-1]
            reshaped_table_view = self.decode_omni_cache.selection_kv_block_table.view(
                -1, num_tokens_per_req, s_max_block_num
            )
            self._reorder_block_table_only(
                self.decode_omni_cache, reshaped_table_view, req_ids_record, req_ids_update
            )

        self.decode_omni_cache.block_tables_record = block_tables.copy()

    @staticmethod
    def _update_request_id_buffers(input_batch, omni_cache) -> list:
        """Update request ID buffers from input batch.

        Extracts sorted request IDs from input batch and manages
        the record/update buffer rotation.

        Args:
            input_batch: The input batch containing request information.
            omni_cache: The OmniCache instance to update.

        Returns:
            Sorted list of request IDs for current batch.
        """
        req_ids_update_mapping = input_batch.req_id_to_index
        req_ids_update = [req_id for req_id, _ in sorted(req_ids_update_mapping.items(), key=lambda item: item[1])]

        if getattr(omni_cache, "req_ids_update_buffer", None) is None:
            omni_cache.req_ids_update_buffer = None
            omni_cache.req_ids_record_buffer = None
        else:
            omni_cache.req_ids_record_buffer = omni_cache.req_ids_update_buffer

        omni_cache.req_ids_update_buffer = req_ids_update
        return req_ids_update

    @staticmethod
    def _cleanup_stale_window_states(omni_cache, current_req_ids: set) -> None:
        """Remove stale window states for requests no longer in batch.

        Cleans up window states, SWA blocks, and associated tracking data
        for requests that are no longer in the current batch.

        Args:
            omni_cache: The OmniCache instance to clean up.
            current_req_ids: Set of request IDs currently in the batch.
        """
        if not hasattr(omni_cache, "_sk_to_state_idx"):
            return

        if omni_cache.req_ids_record_buffer is None:
            return

        sks_to_remove = []
        for sk, state_idx in omni_cache._sk_to_state_idx.items():
            rid, grp_id = sk
            if (grp_id not in ["current_group"]) and rid not in current_req_ids:
                sks_to_remove.append((sk, state_idx))

        for sk, state_idx in sks_to_remove:
            rid, _ = sk
            omni_cache.record_req_ids_swa_block.pop(rid, None)

            keys_to_del = [k for k in omni_cache.last_num_non_zeros.keys() if k[0] == rid]
            for key in keys_to_del:
                omni_cache.last_num_non_zeros.pop(key, None)
                omni_cache.record_req_sched_times.pop(key, None)

            if sk in omni_cache.req_window_states:
                omni_cache.req_window_states.pop(sk)

            del omni_cache._sk_to_state_idx[sk]
            omni_cache._free_state_indices.add(state_idx)

    @staticmethod
    def _cleanup_gather_selection_tracking(omni_cache, current_req_ids: set) -> None:
        """Clean up gather-selection tracking data for removed requests.

        Removes SWA block records, non-zero counts, and scheduling
        times for requests that were in the previous batch but are no
        longer present.  The core ``record_batch_idx_to_req`` cleanup
        is handled by ``static_utils.record_current_batch_order``.

        Args:
            omni_cache: The OmniCache instance to clean up.
            current_req_ids: Set of request IDs currently in the batch.
        """
        if omni_cache.req_ids_record_buffer is None:
            return

        removed_req_ids = set(omni_cache.req_ids_record_buffer) - current_req_ids
        for req_id in removed_req_ids:
            omni_cache.record_req_ids_swa_block.pop(req_id, None)

            keys_to_del = [k for k in omni_cache.last_num_non_zeros.keys() if k[0] == req_id]
            for key in keys_to_del:
                omni_cache.last_num_non_zeros.pop(key, None)
                omni_cache.record_req_sched_times.pop(key, None)

    @staticmethod
    def _update_id_index_mappings(omni_cache) -> None:
        """Update request ID to batch index mappings.

        Rebuilds the legacy req_id_to_idx mapping from the current
        record_batch_idx_to_req dictionary. With USE_OMNI_INPUT_BATCH=1,
        H2D owns req_id_to_idx and this helper must not overwrite it.

        Args:
            omni_cache: The OmniCache instance to update.
        """
        import os
        if int(os.getenv("USE_OMNI_INPUT_BATCH", "0")):
            return
        omni_cache.req_id_to_idx = {
            req_id: idx
            for idx, req_id in omni_cache.record_batch_idx_to_req.items()
            if req_id is not None
        }

    @staticmethod
    def _sync_id_to_idx_table(omni_cache) -> None:
        """Synchronize ID to index table from CPU to device.

        Copies the updated indices from CPU buffer to device tensor
        when the request order has changed.

        Args:
            omni_cache: The OmniCache instance to update.
        """
        if omni_cache.req_ids_record_buffer == omni_cache.req_ids_update_buffer:
            return

        for i, req_id in enumerate(omni_cache.req_ids_update_buffer):
            omni_cache.indices_cpu_buffer[i] = omni_cache.req_id_to_idx.get(req_id, -1)

        num_reqs = len(omni_cache.req_ids_update_buffer)
        # Blocking copy — see the matching block in static_utils.py for the
        # rationale. The async copy races with the next buffer rewrite /
        # forward dispatch under concurrent requests.
        omni_cache.id_to_idx_table[:num_reqs].copy_(
            omni_cache.indices_cpu_buffer[:num_reqs],
            non_blocking=False,
        )

    @staticmethod
    def record_current_batch_order(input_batch, omni_cache) -> None:
        """Record the current batch order with gather-selection cleanup.

        Delegates core batch-order tracking (buffer rotation, lane
        mapping rebuild, device sync) to the standalone
        ``record_current_batch_order`` in ``static_utils``, then
        performs gather-selection-specific cleanup for stale requests.

        Args:
            input_batch: The input batch containing request information.
            omni_cache: The OmniCache instance to update.
        """
        from omni_cache.cache.decode.static_utils import (
            record_current_batch_order as _record_core,
        )
        _record_core(input_batch, omni_cache)

        if not omni_cache.is_hybrid_attn:
            return

        # Gather-selection-specific cleanup for requests removed from batch
        current_req_ids = set(omni_cache.req_ids_update_buffer)
        GatherSelectionUpdater._cleanup_stale_window_states(omni_cache, current_req_ids)
        GatherSelectionUpdater._cleanup_gather_selection_tracking(omni_cache, current_req_ids)

    @staticmethod
    def _update_status_buffered(
        omni_cache,
        reshaped_status: torch.Tensor,
        old_req_ids: List[Optional[int]],
        new_req_ids: List[Optional[int]],
        *,
        fill_value: Any = -1,
    ) -> None:
        """Update the selection KV block status buffer.

        Performs buffered updates to the status tensor based on request ID
        mappings from old to new order.

        Args:
            omni_cache: The OmniCache instance.
            reshaped_status: The reshaped status tensor to update.
            old_req_ids: The previous request IDs order.
            new_req_ids: The new request IDs order.
            fill_value: The value to fill for new entries (default: -1).
        """
        if old_req_ids is None:
            old_req_ids = []
        if not new_req_ids:
            return

        _, batch_size, _ = reshaped_status.shape
        work_buffer = omni_cache.selection_kv_block_status_buffer
        if work_buffer.shape != reshaped_status.shape:
            work_buffer = torch.empty_like(reshaped_status)

        work_buffer.fill_(fill_value)

        max_old = min(len(old_req_ids), batch_size)
        max_new = min(len(new_req_ids), batch_size)

        old_id2idx = {rid: i for i, rid in enumerate(old_req_ids[:max_old]) if rid is not None}

        mapped_src = []
        mapped_tgt = []
        for tgt in range(max_new):
            rid = new_req_ids[tgt]
            src = old_id2idx.get(rid, None)
            if src is not None and 0 <= src < max_old:
                mapped_src.append(src)
                mapped_tgt.append(tgt)

        if mapped_src:
            device = reshaped_status.device
            src_idx = torch.tensor(mapped_src, dtype=torch.long, device=device)
            tgt_idx = torch.tensor(mapped_tgt, dtype=torch.long, device=device)
            selected = reshaped_status.index_select(1, src_idx)
            work_buffer.index_copy_(1, tgt_idx, selected)

        reshaped_status.copy_(work_buffer)

    @staticmethod
    def _build_table_perm(
        old_req_ids: List[int],
        new_req_ids: List[int],
        batch_size: int,
    ) -> Tuple[List[int], int]:
        """Build a permutation for reordering the block table.

        Creates a permutation that places kept entries first, followed by
        remaining entries that are no longer needed.

        Args:
            old_req_ids: The previous request IDs order.
            new_req_ids: The new request IDs order.
            batch_size: The batch size limit.

        Returns:
            Tuple of (permutation list, max_old count).
        """
        max_old = min(len(old_req_ids), batch_size)
        if max_old == 0:
            return np.array([], dtype=np.int64), 0

        old_arr = np.array(old_req_ids[:max_old])
        new_arr = np.array(new_req_ids)

        sorter = np.argsort(old_arr)
        search_pos = np.searchsorted(old_arr, new_arr, sorter=sorter)
        search_pos = np.minimum(search_pos, max_old - 1)
        old_indices = sorter[search_pos]
        is_shared = old_arr[old_indices] == new_arr

        perm_eff = np.empty(max_old, dtype=np.int64)

        new_pos = np.arange(len(new_arr))
        shared_within = is_shared & (new_pos < max_old)
        perm_eff[new_pos[shared_within]] = old_indices[shared_within]

        used_mask = np.zeros(max_old, dtype=bool)
        used_mask[old_indices[is_shared]] = True
        filled_mask = np.zeros(max_old, dtype=bool)
        filled_mask[new_pos[shared_within]] = True

        perm_eff[~filled_mask] = np.where(~used_mask)[0]

        assert len(perm_eff) == max_old
        assert np.array_equal(np.sort(perm_eff), np.arange(max_old))
        return perm_eff, max_old

    @staticmethod
    def _reorder_block_table_only(
        omni_cache,
        reshaped_table: torch.Tensor,
        old_req_ids: List[int],
        new_req_ids: List[int],
    ) -> None:
        """Reorder the block table based on request ID changes.

        Reorders the block table tensor according to the permutation derived
        from old and new request ID orders.

        Args:
            omni_cache: The OmniCache instance.
            reshaped_table: The reshaped block table tensor.
            old_req_ids: The previous request IDs order.
            new_req_ids: The new request IDs order.
        """
        batch_size, _ = reshaped_table.shape
        if not old_req_ids or batch_size == 0:
            return

        perm_eff, max_old = GatherSelectionUpdater._build_table_perm(old_req_ids, new_req_ids, batch_size)

        work_buffer = omni_cache.selection_kv_block_table_buffer
        if work_buffer.shape != reshaped_table.shape:
            work_buffer = torch.empty_like(reshaped_table)

        device = reshaped_table.device

        work_buffer.copy_(reshaped_table)

        if max_old > 0:
            perm_tensor = torch.tensor(perm_eff, dtype=torch.long, device=device)
            new_order_first = reshaped_table[:max_old].index_select(0, perm_tensor)
            work_buffer[:max_old].copy_(new_order_first)

        reshaped_table.copy_(work_buffer)

    @staticmethod
    def maybe_update_selection_kv_block_status(input_batch, omni_cache, num_scheduled_tokens) -> None:
        """Conditionally update the selection KV block status.

        Checks if the request order has changed and updates the block status
        and block table accordingly. Logs timing for performance monitoring.

        Args:
            input_batch: The input batch containing request information.
            omni_cache: The OmniCache instance.
            num_scheduled_tokens: Number of tokens scheduled per request.
        """
        time0 = time.perf_counter()
        num_decode_tokens = 0
        num_decodes = len(num_scheduled_tokens)
        for num_tokens in num_scheduled_tokens:
            num_decode_tokens += num_tokens
        req_ids_update_mapping = input_batch.req_id_to_index
        req_ids_update = tuple(req_id for req_id, _ in sorted(req_ids_update_mapping.items(), key=lambda item: item[1]))

        req_ids_record = getattr(omni_cache, "req_ids_record", None)

        if req_ids_record != req_ids_update or req_ids_record is None:
            num_layers = omni_cache.selection_kv_block_status.shape[0]
            topk_plus_1 = omni_cache.selection_kv_block_status.shape[-1]

            reshaped_view = omni_cache.selection_kv_block_status.view(
                num_layers, -1, topk_plus_1
            )
            GatherSelectionUpdater._update_status_buffered(
                omni_cache, reshaped_view, req_ids_record, req_ids_update, fill_value=-1
            )
            s_max_block_num = omni_cache.selection_kv_block_table.shape[-1]
            reshaped_table_view = omni_cache.selection_kv_block_table.view(
                -1, s_max_block_num
            )
            GatherSelectionUpdater._reorder_block_table_only(
                omni_cache, reshaped_table_view, req_ids_record, req_ids_update
            )

            omni_cache.req_ids_record = req_ids_update
            time1 = time.perf_counter()
            logger.debug(f"======== Time cost for one-step block_status update is {(time1-time0)*1000}ms ========")
        else:
            time1 = time.perf_counter()
            logger.debug(f"++++++++ Time cost for one-step block_status update is {(time1-time0)*1000}ms ++++++++")


__all__ = ["GatherSelectionUpdater"]
