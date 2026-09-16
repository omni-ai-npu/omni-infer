# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    SpecGroup,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import Request, RequestStatus
import vllm.v1.core.sched.scheduler as scheduler_mod

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def offload_remaining_blocks(
    scheduler,
    request: Request,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_external_computed_tokens: int = 0,
    num_encoder_tokens: int = 0,
    newly_shared_blocks: set[int] | None = None,
) -> int:
    """Physical headroom to finish prefill, including speculative lookahead.

    Applying a rolling-cache cap to logical indices before subtracting the
    block-table length counts null placeholders as occupied HBM. Estimate the
    uncapped demand first, then cap it using recyclable physical pages.
    """
    mgr = scheduler.kv_cache_manager
    groups = (
        new_computed_blocks.blocks
        if new_computed_blocks is not None
        else mgr.empty_kv_cache_blocks.blocks
    )
    full_tokens = min(request.num_tokens, scheduler.max_model_len)
    slot_tokens = min(
        full_tokens + scheduler.num_lookahead_tokens, scheduler.max_model_len
    )
    computed = min(
        request.num_computed_tokens
        + num_new_computed_tokens
        + num_external_computed_tokens,
        scheduler.max_model_len,
    )
    remaining = 0
    for manager, local_blocks in zip(mgr.coordinator.single_type_managers, groups):
        if isinstance(manager, CrossAttentionManager):
            remaining += manager.get_num_blocks_to_allocate(
                request.request_id, num_encoder_tokens, [], 0, num_encoder_tokens
            )
            continue
        needed = manager.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=slot_tokens,
            new_computed_blocks=local_blocks,
            total_computed_tokens=computed,
            num_tokens_main_model=full_tokens,
            apply_admission_cap=False,
        )
        cap = manager._max_admission_blocks_per_request
        if cap is not None:
            # Shared pages are not guaranteed to become free when
            # this request recycles its window. Nulls are never physical pages.
            recyclable_block_ids = set()
            for block in manager.req_to_blocks.get(request.request_id, ()):
                if block.is_null or block.ref_cnt != 1:
                    continue
                if newly_shared_blocks and block.block_id in newly_shared_blocks:
                    continue
                recyclable_block_ids.add(block.block_id)
            recyclable = len(recyclable_block_ids)
            lookahead_blocks = (
                scheduler.num_lookahead_tokens + manager.block_size - 1
            ) // manager.block_size
            needed = min(needed, max(0, cap + lookahead_blocks - recyclable))
        remaining += max(0, needed)
    return remaining


def full_attention_group_id(attention_groups: list[SpecGroup]) -> int | None:
    """Dense FA group used as the per-group lookup reference, or None."""
    first = attention_groups[0]
    if isinstance(first.spec, FullAttentionSpec):
        return first.group_ids[0]
    return None


def is_offloading_connector(conn: Any) -> bool:
    if isinstance(conn, OffloadingConnector):
        return True
    if isinstance(conn, MultiConnector):
        return any(isinstance(c, OffloadingConnector) for c in conn._connectors)
    return False


def local_hits_from_blocks(
    kv_cache_groups, blocks: KVCacheBlocks
) -> tuple[int, ...]:
    return tuple(
        len(group) * kv_cache_groups[i].kv_cache_spec.block_size
        for i, group in enumerate(blocks.blocks)
    )


class _SkipHybridConnectorLookup:
    """Not a coordinator. ``isinstance(coord, this)`` is always false."""


class _ReadyOffloadQueueView:
    """The peek/pop interface used by Scheduler's waiting loop.

    Keep ownership in the original queue until pop: an allocation failure or
    exception must not lose the selected request. Do not mutate priorities.
    """

    def __init__(self, queue, request):
        self.queue = queue
        self.request = request

    def peek_request(self):
        return self.request

    def pop_request(self):
        self.queue.remove_request(self.request)
        return self.request


@register_patch(
    "HybridKVCacheCoordinatorOffloadPatch", HybridKVCacheCoordinator
)
class HybridKVCacheCoordinatorOffloadPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__"]

    def __init__(self, *args, **kwargs):
        HybridKVCacheCoordinatorOffloadPatch._upstream__init__(
            self, *args, **kwargs
        )
        self.full_attention_group_id = full_attention_group_id(
            self.attention_groups
        )

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("__init__")


@register_patch("KVCacheManagerOffloadJointLookupPatch", KVCacheManager)
class KVCacheManagerOffloadJointLookupPatch(VLLMPatch):
    _attr_names_to_apply = [
        "prefix_cache_lookup_enabled",
        "record_prefix_cache_stats",
        "get_computed_blocks_for_connector",
        "get_computed_blocks",
        "allocate_slots",
    ]

    def prefix_cache_lookup_enabled(self, request: Request) -> bool:
        return self.enable_caching and not request.skip_reading_prefix_cache

    def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:
        if not self.log_stats or not self.prefix_cache_lookup_enabled(request):
            return
        if self.prefix_cache_stats is None:
            raise RuntimeError("prefix_cache_stats is None while log_stats is set")
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_hits,
            preempted=request.num_preemptions > 0,
        )

    def get_computed_blocks_for_connector(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, bool]:
        coordinator = self.coordinator
        if not (
            self.kv_cache_config.has_mamba_layers
            and isinstance(coordinator, HybridKVCacheCoordinator)
            and getattr(coordinator, "full_attention_group_id", None) is not None
        ):
            blocks, num_local = (
                KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks(
                    self, request
                )
            )
            return blocks, num_local, False

        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, False

        fa_group_id = coordinator.full_attention_group_id
        computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
            request.block_hashes, request.num_tokens - 1
        )
        if any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):
            blocks, num_local = (
                KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks(
                    self, request
                )
            )
            return blocks, num_local, False

        num_local = per_group_hits[fa_group_id]
        blocks = self.create_kv_cache_blocks(computed)
        return blocks, num_local, min(per_group_hits) < num_local

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        if not getattr(self, "_omni_is_offloading_connector", False):
            return KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks(
                self, request
            )

        saved_log = self.log_stats
        self.log_stats = False
        try:
            blocks, num_local, hit_diverged = (
                self.get_computed_blocks_for_connector(request)
            )
            if (
                hit_diverged
                and num_local % self.coordinator.scheduler_block_size
            ):
                blocks, num_local = (
                    KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks(
                        self, request
                    )
                )
        finally:
            self.log_stats = saved_log

        per_group = local_hits_from_blocks(
            self.kv_cache_config.kv_cache_groups, blocks
        )
        if per_group:
            request.local_computed_tokens_per_group = per_group
            num_local = min(per_group)

        self.record_prefix_cache_stats(request, num_local)
        return blocks, num_local

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        scheduler = getattr(self, "_omni_offload_scheduler", None)
        if scheduler is not None:
            # Adopting local hits can remove another request's exclusive
            # recycling credit. Account for that before refcounts are touched.
            newly_shared_blocks = None
            if new_computed_blocks is not None:
                newly_shared_blocks = set()
                for group in new_computed_blocks.blocks:
                    for block in group:
                        if block.is_null or block.ref_cnt != 1:
                            continue
                        newly_shared_blocks.add(block.block_id)
            # Protect reservations from local misses and running requests too.
            # Use vLLM's lifecycle set: DMA completion alone does not release a
            # reservation; prefill completion, preemption or cancellation does.
            other_reserved = sum(
                offload_remaining_blocks(
                    scheduler, req, newly_shared_blocks=newly_shared_blocks
                )
                for req in scheduler._inflight_prefills
                if req is not request
            )
            if delay_cache_blocks and any(
                req is request for req in scheduler._inflight_prefills
            ):
                # A failed load can retry while still in vLLM's in-flight set.
                # The caller's aggregate then includes this request itself.
                reserved_blocks = max(
                    0, reserved_blocks - offload_remaining_blocks(scheduler, request)
                )
            reserved_blocks = max(reserved_blocks, other_reserved)
            if delay_cache_blocks or (
                request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
                and request.num_computed_tokens == 0
            ):
                required = offload_remaining_blocks(
                    scheduler,
                    request,
                    num_new_computed_tokens,
                    new_computed_blocks,
                    num_external_computed_tokens,
                    num_encoder_tokens,
                )
                watermark = self.watermark_blocks if has_scheduled_reqs else 0
                free = self.block_pool.get_num_free_blocks()
                if required + watermark + reserved_blocks > free:
                    return None
        return KVCacheManagerOffloadJointLookupPatch._upstream_allocate_slots(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=has_scheduled_reqs,
        )

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("get_computed_blocks", "allocate_slots")


@register_patch("SchedulerOffloadJointLookupPatch", Scheduler)
class SchedulerOffloadJointLookupPatch(VLLMPatch):
    _attr_names_to_apply = [
        "_is_offloading_connector",
        "__init__",
        "_request_remaining_blocks",
        "_select_waiting_queue_for_scheduling",
    ]

    def _select_waiting_queue_for_scheduling(self):
        upstream = (
            SchedulerOffloadJointLookupPatch
            ._upstream_select_waiting_queue_for_scheduling(self)
        )
        if (
            upstream is None
            or getattr(self.kv_cache_manager, "_omni_offload_scheduler", None)
            is not self
        ):
            return upstream

        admitted = {req.request_id for req in self._inflight_prefills}

        def ready(req):
            if req.request_id not in admitted:
                return False
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                return req.request_id in self.finished_recving_kv_req_ids
            return req.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)

        if ready(upstream.peek_request()):
            return upstream
        selected = None
        source = None
        for queue in (self.skipped_waiting, self.waiting):
            for req in queue:
                if ready(req):
                    if selected is None or (
                        self.policy == SchedulingPolicy.PRIORITY and req < selected
                    ):
                        selected, source = req, queue
                    # Each queue already iterates in policy order.
                    break
            if selected is not None and self.policy == SchedulingPolicy.FCFS:
                break
        if selected is None:
            return upstream
        # A new request may need the very reservation held by a loaded request
        # behind it. Upstream breaks on allocation failure, so let the holder
        # spend its reservation first instead of blocking both indefinitely.
        return _ReadyOffloadQueueView(source, selected)

    def _request_remaining_blocks(self, request: Request) -> int:
        if getattr(self.kv_cache_manager, "_omni_offload_scheduler", None) is self:
            return offload_remaining_blocks(self, request)
        return SchedulerOffloadJointLookupPatch._upstream_request_remaining_blocks(
            self, request
        )

    def _is_offloading_connector(self) -> bool:
        return is_offloading_connector(self.connector)

    def __init__(self, *args, **kwargs):
        SchedulerOffloadJointLookupPatch._upstream__init__(self, *args, **kwargs)
        is_offload = self._is_offloading_connector()
        mgr = self.kv_cache_manager
        mgr._omni_has_connector = self.connector is not None
        mgr._omni_is_offloading_connector = is_offload
        if is_offload:
            mgr._omni_offload_scheduler = self
            scheduler_mod.HybridKVCacheCoordinator = _SkipHybridConnectorLookup

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict(
            "__init__", "_request_remaining_blocks",
            "_select_waiting_queue_for_scheduling",
        )
