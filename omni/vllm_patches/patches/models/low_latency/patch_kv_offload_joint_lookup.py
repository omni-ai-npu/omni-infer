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
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import Request
import vllm.v1.core.sched.scheduler as scheduler_mod

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


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

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("get_computed_blocks")


@register_patch("SchedulerOffloadJointLookupPatch", Scheduler)
class SchedulerOffloadJointLookupPatch(VLLMPatch):
    _attr_names_to_apply = ["_is_offloading_connector", "__init__"]

    def _is_offloading_connector(self) -> bool:
        return is_offloading_connector(self.connector)

    def __init__(self, *args, **kwargs):
        SchedulerOffloadJointLookupPatch._upstream__init__(self, *args, **kwargs)
        is_offload = self._is_offloading_connector()
        mgr = self.kv_cache_manager
        mgr._omni_has_connector = self.connector is not None
        mgr._omni_is_offloading_connector = is_offload
        if is_offload:
            scheduler_mod.HybridKVCacheCoordinator = _SkipHybridConnectorLookup

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("__init__")
