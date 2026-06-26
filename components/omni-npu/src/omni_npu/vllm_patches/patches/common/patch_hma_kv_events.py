# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import vllm.distributed.kv_events as kv_events
import vllm.v1.core.block_pool as block_pool_module
from vllm.config import VllmConfig
from vllm.distributed.kv_events import MEDIUM_GPU, KVCacheEvent
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    ExternalBlockHash,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_ORIGINAL_VLLM_CONFIG_POST_INIT = VllmConfig.__post_init__


class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    group_idx: int | None = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.block_hashes),
                self.parent_block_hash,
                tuple(self.token_ids),
                self.block_size,
                self.lora_id,
                self.medium,
                self.group_idx,
            )
        )


class BlockRemoved(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None = None

    def __hash__(self) -> int:
        return hash((tuple(self.block_hashes), self.medium, self.group_idx))


class KVEventBatch(kv_events.EventBatch):
    events: list[BlockStored | BlockRemoved | kv_events.AllBlocksCleared]


@register_patch("HMAKVEventsModulePatch", kv_events)
class HMAKVEventsModulePatch(VLLMPatch):
    _attr_names_to_apply = ["BlockStored", "BlockRemoved", "KVEventBatch"]

    BlockStored = BlockStored
    BlockRemoved = BlockRemoved
    KVEventBatch = KVEventBatch


@register_patch("HMAKVEventsBlockPoolGlobalsPatch", block_pool_module)
class HMAKVEventsBlockPoolGlobalsPatch(VLLMPatch):
    _attr_names_to_apply = ["BlockStored", "BlockRemoved"]

    BlockStored = BlockStored
    BlockRemoved = BlockRemoved


@register_patch("HMAKVEventsBlockPoolPatch", BlockPool)
class HMAKVEventsBlockPoolPatch(VLLMPatch):
    _attr_names_to_apply = ["cache_full_blocks", "_maybe_evict_cached_block"]

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[Any],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        if len(request.block_hashes) < num_full_blocks:
            raise ValueError(
                "The request does not contain enough block hashes for the "
                "number of full blocks."
            )
        if block_size == self.hash_block_size:
            block_hashes = request.block_hashes
        else:
            if block_size % self.hash_block_size != 0:
                raise ValueError(
                    "The block size must be divisible by the hash block size."
                )
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes: BlockHashList | BlockHashListWithBlockSize = block_hashes[
            num_cached_blocks:
        ]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            if blk.is_null:
                continue
            if blk.block_hash is not None:
                raise RuntimeError("Cannot cache a block that already has a hash.")
            block_hash = new_block_hashes[i]

            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks * block_size:num_full_blocks * block_size
                    ],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    group_idx=kv_cache_group_id,
                )
            )

    def _maybe_evict_cached_block(self, block: Any) -> bool:
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                    group_idx=get_group_id(block_hash),
                )
            )
        return True


@register_patch("HMAKVEventsVllmConfigPatch", VllmConfig)
class HMAKVEventsVllmConfigPatch(VLLMPatch):
    _attr_names_to_apply = ["__post_init__"]

    def __post_init__(self) -> None:
        kv_events_config = self.kv_events_config
        if kv_events_config is None:
            _ORIGINAL_VLLM_CONFIG_POST_INIT(self)
            return

        self.kv_events_config = None
        try:
            _ORIGINAL_VLLM_CONFIG_POST_INIT(self)
        finally:
            self.kv_events_config = kv_events_config
