# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Recover failed remote KV loads for hybrid KV-cache models.

vLLM's invalid-block recovery currently assumes that ``get_block_ids``
returns exactly one KV-cache group. Hybrid models return multiple groups,
which makes the single-item tuple unpack raise ``ValueError``. LLMDataDist
transfers the attention KV cache, so recovery must inspect the attention
group while preserving the rest of vLLM's recovery semantics.
"""

from __future__ import annotations

from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.request import Request

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


def _get_kv_load_recovery_block_ids(
    scheduler: Scheduler, request_id: str
) -> list[int]:
    """Return block IDs for the KV-cache group transferred by LLMDataDist."""
    block_id_groups = scheduler.kv_cache_manager.get_block_ids(request_id)
    if len(block_id_groups) == 1:
        return block_id_groups[0]

    kv_cache_groups = getattr(
        getattr(scheduler, "kv_cache_config", None), "kv_cache_groups", ()
    )
    for group_idx, group in enumerate(kv_cache_groups):
        if group_idx >= len(block_id_groups):
            continue
        kv_cache_spec = getattr(group, "kv_cache_spec", None)
        if isinstance(kv_cache_spec, AttentionSpec):
            return block_id_groups[group_idx]

    logger.warning(
        "Request %s has %d KV cache groups but no attention group; "
        "using group 0 for KV load recovery.",
        request_id,
        len(block_id_groups),
    )
    return block_id_groups[0]


@register_patch("HybridKVLoadFailureSchedulerPatch", Scheduler)
class HybridKVLoadFailureSchedulerPatch(VLLMPatch):
    """Make vLLM invalid-block recovery work with hybrid KV-cache groups."""

    _attr_names_to_apply = ["_update_requests_with_invalid_blocks"]

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        marked_invalid_block_ids: set[int] = set()

        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            block_id_groups = self.kv_cache_manager.get_block_ids(req_id)
            is_hybrid = len(block_id_groups) > 1
            req_block_ids = _get_kv_load_recovery_block_ids(self, req_id)
            req_num_computed_tokens = (
                request.num_computed_tokens
                - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(
                range(req_num_computed_blocks), req_block_ids
            ):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True
                if block_id in marked_invalid_block_ids:
                    continue

                marked_invalid_block_ids.add(block_id)
                if marked_invalid_block:
                    continue

                marked_invalid_block = True
                if is_hybrid:
                    continue
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if is_hybrid:
                    total_affected_tokens += req_num_computed_tokens
                    request.num_computed_tokens = 0
                    if evict_blocks:
                        for group in block_id_groups:
                            blocks_to_evict.update(group)
                elif not marked_invalid_block:
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(req_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict
