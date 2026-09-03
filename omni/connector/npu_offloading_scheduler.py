# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Scheduler-side overrides for the NPU offloading connector."""

from __future__ import annotations

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
    RequestOffloadState,
    TransferJobStatus,
    logger,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import GPULoadStoreSpec, OffloadKey


def storeable_num_blocks(
    group_config: GroupOffloadConfig, num_tokens: int
) -> int:
    """Offloaded blocks of *group_config* that may be stored at *num_tokens*.

    The trailing block of an EAGLE/MTP draft group is volatile and has no
    stable hash yet, so it must not be offered for store, and the store cursor
    must not move past it either.
    """
    num_blocks = num_tokens // group_config.offloaded_block_size
    if group_config.is_eagle_group:
        num_blocks = max(0, num_blocks - 1)
    return num_blocks


class NPUOffloadingConnectorScheduler(OffloadingConnectorScheduler):
    """Keeps the store cursor aligned with the keys actually offered for store.

    Upstream computes the store bound twice per step. The first pass of
    ``_build_store_jobs`` excludes the volatile trailing block of an EAGLE/MTP
    group, but the two paths that move ``next_stored_block_idx`` afterwards --
    ``RequestOffloadState.advance_stored_idx`` and the tail of the second pass
    -- do not. The cursor then sits one block past the last key that was ever
    offered to ``prepare_store``, and the next step's
    ``num_blocks <= start_block_idx`` check skips that block for good, leaving
    a hole in the offloaded key sequence. ``_maximal_prefix_lookup`` stops at
    the first miss, so a single hole voids every block behind it.

    ``_build_store_jobs`` below is forked from vLLM v0.25.1 (752a3a5044),
    ``offloading/scheduler.py``. Every deviation from upstream is bracketed by
    ``omni-npu diff start`` / ``omni-npu diff end``; when bumping vLLM, re-diff
    the method against upstream and keep only those brackets.
    """

    def _advance_stored_idx(
        self, req_status: RequestOffloadState, num_offloadable_tokens: int
    ) -> None:
        """``RequestOffloadState.advance_stored_idx`` with the EAGLE exclusion."""
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            group_state.next_stored_block_idx = storeable_num_blocks(
                group_config, num_offloadable_tokens
            )

    @override
    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        block_size_factor = self.config.block_size_factor
        store_jobs: dict[int, TransferJob] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req = req_status.req

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
            # with async scheduling, some tokens may be missing
            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)
            max_offload_tokens = req_status.max_offload_tokens
            if max_offload_tokens is not None:
                num_offloadable_tokens = min(num_offloadable_tokens, max_offload_tokens)

            # Skip decode-phase blocks: clamp to the prompt length so only
            # prefill (prompt) blocks become eligible for store. next_stored_idx
            # never advances past this boundary, so decode blocks are never
            # queued in this or any later step.
            if self.config.offload_prompt_only:
                num_offloadable_tokens = min(
                    num_offloadable_tokens, req.num_prompt_tokens
                )

            # Filter out blocks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                # --- omni-npu diff start: equivalent, routed through the
                # single storeable-bound helper so both passes agree ---
                num_blocks = storeable_num_blocks(
                    group_config, num_offloadable_tokens
                )
                # --- omni-npu diff end ---

                start_block_idx = group_state.next_stored_block_idx
                if num_blocks <= start_block_idx:
                    continue
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
                # For each block to offload, take the last corresponding GPU block.
                # e.g. if block size factor is 3 and GPU block IDs are
                # 1 5 6 7 2 4 9 3 8 then we'll take blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                offload_block_ids = group_state.block_ids[
                    start_block_idx * block_size_factor
                    + block_size_factor
                    - 1 : num_blocks * block_size_factor : block_size_factor
                ]
                assert len(offload_keys) == len(offload_block_ids)

                alignment_block_count = group_config.alignment_block_count
                tail = group_config.sliding_window_size_in_blocks

                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        continue
                    # Skip SWA blocks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing `tail` blocks are reachable by
                    # _sliding_window_lookup. For DeepSeek V4 with 100K
                    # tokens this reduces SWA stores by ~78%.
                    if alignment_block_count is not None:
                        assert tail is not None
                        abs_block_idx = start_block_idx + key_idx
                        pos_in_segment = abs_block_idx % alignment_block_count
                        if pos_in_segment < alignment_block_count - tail:
                            continue
                    new_offload_keys.append(offload_key)

            if not new_offload_keys:
                # --- omni-npu diff start: RequestOffloadState.advance_stored_idx
                # misses the same exclusion; advance through the helper instead ---
                self._advance_stored_idx(req_status, num_offloadable_tokens)
                # --- omni-npu diff end ---
                continue

            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            if not store_output.keys_to_store:
                # --- omni-npu diff start: RequestOffloadState.advance_stored_idx
                # misses the same exclusion; advance through the helper instead ---
                self._advance_stored_idx(req_status, num_offloadable_tokens)
                # --- omni-npu diff end ---
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            sliding_window_block_ids: list[int] = []
            non_sliding_window_block_ids: list[int] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_blocks is not None
                )
                # --- omni-npu diff start: upstream omits the EAGLE/MTP
                # exclusion here, so the cursor below lands one block past the
                # last key that was offered to prepare_store ---
                num_blocks = storeable_num_blocks(
                    group_config, num_offloadable_tokens
                )
                # --- omni-npu diff end ---
                start_block_idx = group_state.next_stored_block_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_block_idx:num_blocks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    offloaded_block_idx = start_block_idx + idx

                    self._events_tracker.record_store(
                        req, group_config, offloaded_block_idx, offload_key
                    )

                    gpu_block_idx = offloaded_block_idx * block_size_factor
                    for i in range(block_size_factor):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            sliding_window_block_ids.append(block_id)
                        else:
                            non_sliding_window_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_block_idx = num_blocks

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )

            logger.debug(
                "Request %s offloading %s blocks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

        return store_jobs
