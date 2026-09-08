# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Scheduler-side overrides for the NPU offloading connector."""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
    RequestGroupState,
    RequestOffloadState,
    TransferJobStatus,
    logger,
)
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadKey,
    OffloadPolicy,
    OffloadingSpec,
)
from vllm.v1.request import Request


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
    """NPU overrides on top of ``OffloadingConnectorScheduler``.

    Store cursor: upstream computes the store bound twice per step. The first
    pass of ``_build_store_jobs`` excludes the volatile trailing EAGLE/MTP
    block, but the two paths that move ``next_stored_block_idx`` afterwards --
    ``RequestOffloadState.advance_stored_idx`` and the tail of the second pass
    -- do not. The cursor then sits one block past the last key that was ever
    offered to ``prepare_store``, and the next step's
    ``num_blocks <= start_block_idx`` check skips that block for good.

    Hybrid lookup: upstream DDR lookup starts every group at the scalar HBM
    min. When one group is longer in HBM than in DDR, that miss aborts the
    whole external hit. Here each group continues from its own HBM length,
    then the assembled prefix is the min across groups.

    Forked methods are from vLLM v0.25.1 (752a3a5044) ``offloading/scheduler.py``.
    Deviations are bracketed by ``omni-npu diff start`` / ``omni-npu diff end``.
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

    def _in_flight_store_gpu_ids(self) -> set[int]:
        """GPU block ids already claimed by an in-flight store job.

        Same role as SimpleCPUOffload ``_in_flight_store_gpu_blocks``
        (vLLM #42903): skip SWA reuse and prefix-share duplicates so the
        same GPU block is not stored (or touched) twice.
        """
        in_flight = set(self._block_id_to_pending_jobs)
        for job_status in self._jobs.values():
            if not job_status.is_store:
                continue
            in_flight.update(job_status.non_sliding_window_block_ids or ())
            in_flight.update(job_status.sliding_window_block_ids or ())
        return in_flight

    def __init__(self, spec: OffloadingSpec):
        super().__init__(spec)
        # RequestOffloadState is slots=True and cannot take this field.
        self._per_group_local_tokens: dict[str, tuple[int, ...]] = {}

    def _per_group_map(self) -> dict[str, tuple[int, ...]]:
        mapping = getattr(self, "_per_group_local_tokens", None)
        if mapping is None:
            mapping = {}
            self._per_group_local_tokens = mapping
        return mapping

    def _capture_per_group_local_tokens(self, request: Request) -> None:
        per_group = getattr(request, "local_computed_tokens_per_group", None)
        mapping = self._per_group_map()
        if per_group is not None:
            mapping[request.request_id] = tuple(per_group)
            request.local_computed_tokens_per_group = None
        else:
            mapping.pop(request.request_id, None)

    def _group_local_tokens(
        self, req_status: RequestOffloadState, group_idx: int
    ) -> int:
        """GPU prefix length to start DDR lookup from for one KV-cache group.

        Prefers per-group HBM hits (hybrid joint lookup). Falls back to the
        scalar ``num_locally_computed_tokens`` used by the scheduler API.
        """
        req_id = req_status.req.request_id
        per_group = self._per_group_map().get(req_id)
        if per_group is None:
            per_group = getattr(
                req_status.req, "local_computed_tokens_per_group", None
            )
        if per_group is not None:
            return per_group[group_idx]
        return req_status.num_locally_computed_tokens

    @override
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        self._capture_per_group_local_tokens(request)
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    @override
    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        self._per_group_map().pop(request.request_id, None)
        return super().request_finished(request)

    @override
    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Forked from upstream ``OffloadingConnectorScheduler._lookup``. Each
        group continues from its own HBM hit; ``max_hit_size_tokens`` is still
        the min assembled prefix across groups.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing block
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                offloaded_block_size = group_config.offloaded_block_size
                offload_keys = group_state.offload_keys

                required_keys = req_status.req.num_tokens // offloaded_block_size
                if len(offload_keys) < required_keys:
                    raise RuntimeError(
                        f"Request {req_status.req.request_id}: group "
                        f"{group_idx} has {len(offload_keys)} offload keys, "
                        f"need at least {required_keys} for "
                        f"{req_status.req.num_tokens} tokens"
                    )

                is_eagle_unverified = (
                    group_config.is_eagle_group and group_idx not in eagle_verified
                )

                # Constrain to block-aligned boundary for this group
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * offloaded_block_size
                )
                if max_hit_size_tokens - num_computed_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )

                # For eagle groups, query one extra block that will be popped.
                # We only need to increase the query size for sliding window groups.
                query_max = max_hit_size_tokens
                if is_eagle_unverified and sliding_window_size_in_blocks is not None:
                    query_max = min(
                        max_hit_size_tokens + offloaded_block_size,
                        len(offload_keys) * offloaded_block_size,
                    )

                num_blocks = min(
                    cdiv(query_max, offloaded_block_size), len(offload_keys)
                )
                # --- omni-npu diff start: per-group HBM start ---
                group_start_tokens = self._group_local_tokens(req_status, group_idx)
                start_block_idx = group_start_tokens // offloaded_block_size
                # --- omni-npu diff end ---
                offload_keys = offload_keys[start_block_idx:num_blocks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_blocks: int | None
                # --- omni-npu diff start: HBM-covered group is not a miss ---
                if not offload_keys:
                    num_hit_blocks = 0
                elif sliding_window_size_in_blocks is None:
                    # --- omni-npu diff end ---
                    num_hit_blocks = self._maximal_prefix_lookup(
                        offload_keys, req_status.req_context
                    )
                else:
                    required_window = sliding_window_size_in_blocks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_blocks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )

                if num_hit_blocks is None:
                    defer_lookup = True
                else:
                    # --- omni-npu diff start: keep HBM prefix on DDR miss ---
                    if num_hit_blocks == 0:
                        assembled = group_start_tokens
                    else:
                        if is_eagle_unverified:
                            num_hit_blocks -= 1
                            eagle_verified.add(group_idx)
                        assembled = offloaded_block_size * (
                            start_block_idx + num_hit_blocks
                        )
                    max_hit_size_tokens = min(max_hit_size_tokens, assembled)
                    # --- omni-npu diff end ---
                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_blocks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # possibly delay request if any of the hit blocks is already being loaded
        if self._blocks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                offloaded_block_size = group_config.offloaded_block_size
                offload_keys = group_state.offload_keys
                num_blocks = cdiv(
                    num_computed_tokens + num_hit_tokens, offloaded_block_size
                )
                # --- omni-npu diff start: per-group HBM start ---
                start_block_idx = (
                    self._group_local_tokens(req_status, group_config.group_idx)
                    // offloaded_block_size
                )
                # --- omni-npu diff end ---
                offload_keys = offload_keys[start_block_idx:num_blocks]
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )
                if sliding_window_size_in_blocks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_blocks:]
                if any(key in self._blocks_being_loaded for key in offload_keys):
                    # hit blocks are being loaded, delay request
                    logger.debug(
                        "Delaying request %s since some of its"
                        " blocks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    @override
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            gpu_block_size = group_config.gpu_block_size
            offloaded_block_size = group_config.offloaded_block_size
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, gpu_block_size)

            if len(group_blocks) < num_gpu_blocks:
                raise RuntimeError(
                    f"Request {request.request_id}: group "
                    f"{group_config.group_idx} has {len(group_blocks)} GPU "
                    f"blocks, need at least {num_gpu_blocks} for "
                    f"{num_cached_tokens} cached tokens"
                )
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            # --- omni-npu diff start: check this group's HBM hit ---
            # Hashed prefix in the *used* range [:num_gpu_blocks] must cover
            # this group's HBM hit, capped by num_cached_tokens. A group can
            # have extra HBM hits past the assembled min; those blocks sit
            # past num_gpu_blocks and are not loaded. A lagging group may
            # start loading before scalar local.
            group_local_tokens = self._group_local_tokens(
                req_status, group_config.group_idx
            )
            hashed_prefix_tokens = (
                num_locally_computed_gpu_blocks * gpu_block_size
            )
            used_local_tokens = min(group_local_tokens, num_cached_tokens)
            if hashed_prefix_tokens < used_local_tokens:
                raise RuntimeError(
                    f"Request {request.request_id}: group "
                    f"{group_config.group_idx} hashed prefix "
                    f"{hashed_prefix_tokens} is shorter than used local "
                    f"{used_local_tokens} (group_local={group_local_tokens}, "
                    f"num_cached={num_cached_tokens}, "
                    f"scalar_local={num_locally_computed_tokens})"
                )
            # --- omni-npu diff end ---
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_blocks is not None:
                max_pending = (
                    group_config.sliding_window_size_in_blocks
                    * self.config.block_size_factor
                )
                if num_pending_gpu_blocks > max_pending:
                    raise RuntimeError(
                        f"Request {request.request_id}: group "
                        f"{group_config.group_idx} has "
                        f"{num_pending_gpu_blocks} pending GPU blocks, "
                        f"exceeds sliding-window limit {max_pending}"
                    )

            num_blocks = cdiv(num_cached_tokens, offloaded_block_size)
            if len(offload_keys) < num_blocks:
                raise RuntimeError(
                    f"Request {request.request_id}: group "
                    f"{group_config.group_idx} has {len(offload_keys)} "
                    f"offload keys, need at least {num_blocks} for "
                    f"{num_cached_tokens} cached tokens"
                )
            if num_pending_gpu_blocks:
                start_block_idx = (
                    num_locally_computed_gpu_blocks // self.config.block_size_factor
                )
                keys_to_load.extend(offload_keys[start_block_idx:num_blocks])

            pending_blocks = group_blocks[
                num_locally_computed_gpu_blocks:num_gpu_blocks
            ]
            for block in pending_blocks:
                dst_block_ids.append(block.block_id)
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit blocks for block-level policy; for
            # request-level, next_stored_block_idx stays at 0 so all
            # blocks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_block_idx = num_blocks

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # a load can only be issued when no other jobs are pending.
        if req_status.transfer_jobs:
            raise RuntimeError(
                f"Request {request.request_id}: cannot issue load job "
                f"{load_job_id} while jobs {sorted(req_status.transfer_jobs)} "
                "are still pending"
            )
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(keys_to_load)

    @override
    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        block_size_factor = self.config.block_size_factor
        store_jobs: dict[int, TransferJob] = {}
        # --- omni-npu diff start: vLLM #42903 eager in_flight GPU ids ---
        in_flight = self._in_flight_store_gpu_ids()
        # --- omni-npu diff end ---
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
                    - 1: num_blocks * block_size_factor: block_size_factor
                ]
                if len(offload_keys) != len(offload_block_ids):
                    raise RuntimeError(
                        f"Request {req_id}: group {group_config.group_idx} "
                        f"offload key count {len(offload_keys)} != "
                        f"GPU block id count {len(offload_block_ids)}"
                    )

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
                        if tail is None:
                            raise RuntimeError(
                                f"Request {req_id}: group "
                                f"{group_config.group_idx} has "
                                "alignment_block_count but no "
                                "sliding_window_size_in_blocks"
                            )
                        abs_block_idx = start_block_idx + key_idx
                        pos_in_segment = abs_block_idx % alignment_block_count
                        if pos_in_segment < alignment_block_count - tail:
                            continue
                    # --- omni-npu diff start: vLLM #42903 ---
                    if block_id in in_flight:
                        continue
                    in_flight.add(block_id)
                    # --- omni-npu diff end ---
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
            # --- omni-npu diff start: vLLM #42903 intra-spec dups ---
            seen_src: set[int] = set()
            # --- omni-npu diff end ---
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
                        # --- omni-npu diff start: vLLM #42903 intra-spec dups ---
                        if block_id in seen_src:
                            continue
                        seen_src.add(block_id)
                        # --- omni-npu diff end ---
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
                if not self._jobs[any_jid].is_store:
                    raise RuntimeError(
                        f"Request {req_id}: cannot issue store job {job_id} "
                        f"while load job {any_jid} is still pending"
                    )
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
