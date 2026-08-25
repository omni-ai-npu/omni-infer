# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU CPUOffloadingSpec: replicated_layout (scheme A) with shared mmap.

Pangu DSA / SWA / Mome count as replicated. Each TP rank maps to slot 0 and
rotates D2H by CPU block id so store traffic is spread across ranks.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingWorker
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from omni_npu.v1.kv_offload.cpu.npu_shared_offload_region import (
    NPUSharedOffloadRegion,
)
from omni_npu.v1.kv_offload.cpu.npu_worker import NPUCPUOffloadingWorker

logger = init_logger(__name__)

_REPLICATED_SPEC_NAMES = frozenset(
    {
        "MLAAttentionSpec",
        "DSAAttentionSpec",
        "ShareKVSlidingWindowSpec",
        "MomeSpec",
    }
)


def _iter_group_specs(kv_cache_config: Any):
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if type(spec).__name__ == "UniformTypeKVCacheSpecs":
            yield from spec.kv_cache_specs.values()
        else:
            yield spec


def _spec_is_replicated(spec: Any) -> bool:
    name = type(spec).__name__
    if name == "UniformTypeKVCacheSpecs":
        return all(_spec_is_replicated(inner) for inner in spec.kv_cache_specs.values())
    return name in _REPLICATED_SPEC_NAMES


def infer_replicated_layout(vllm_config: Any, kv_cache_config: Any) -> bool:
    """True when every KV group is a full replica and only TP>1 is in play."""
    pc = vllm_config.parallel_config
    if pc.tensor_parallel_size <= 1:
        return False
    if getattr(pc, "pipeline_parallel_size", 1) != 1:
        return False
    if getattr(pc, "decode_context_parallel_size", 1) != 1:
        return False
    if getattr(pc, "prefill_context_parallel_size", 1) != 1:
        return False
    if pc.world_size != pc.tensor_parallel_size:
        return False
    groups = kv_cache_config.kv_cache_groups
    if not groups:
        return False
    return all(_spec_is_replicated(spec) for spec in _iter_group_specs(kv_cache_config))


class NPUCPUOffloadingSpec(CPUOffloadingSpec):
    """CPUOffloadingSpec with NPU transfers and upstream layout flags."""

    BLOCK_SIZE_ALIGNMENT = NPUSharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        if extra_config.get("canonical_layout"):
            logger.warning(
                "canonical_layout is ignored; NPU CPU offload uses "
                "replicated_layout (scheme A) with per-rank store rotation."
            )
        inferred = infer_replicated_layout(vllm_config, kv_cache_config)
        if "replicated_layout" in extra_config:
            self.replicated_layout = bool(extra_config["replicated_layout"])
        else:
            self.replicated_layout = inferred

        super().__init__(vllm_config, kv_cache_config)

        if self.replicated_layout:
            self._apply_single_copy_capacity(kv_cache_config)

        logger.info(
            "NPUCPUOffloadingSpec: replicated_layout=%s rotate_store=%s "
            "num_blocks=%d row_bytes=%d cpu_page=%d",
            self.replicated_layout,
            self.replicated_layout,
            self.num_blocks,
            self.kv_bytes_per_offloaded_block,
            self.cpu_page_size_per_worker,
        )

    def _apply_single_copy_capacity(self, kv_cache_config: KVCacheConfig) -> None:
        """Size the mmap row as one replica (community num_copies=1)."""
        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use or kv_cache_config.num_blocks <= 0:
            return

        is_packed = any(t.block_stride for t in kv_cache_config.kv_cache_tensors)
        if is_packed and not all(
            t.block_stride for t in kv_cache_config.kv_cache_tensors
        ):
            raise ValueError(
                "packed KV cache requires every tensor to have block_stride"
            )
        total_gpu_kv_bytes = (
            kv_cache_config.kv_cache_tensors[0].size
            if is_packed
            else sum(t.size for t in kv_cache_config.kv_cache_tensors)
        )
        kv_bytes_per_block = total_gpu_kv_bytes // kv_cache_config.num_blocks
        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        aligned_kv_bytes = round_up(
            kv_bytes_per_offloaded_block, self.BLOCK_SIZE_ALIGNMENT
        )
        self.kv_bytes_per_offloaded_block = aligned_kv_bytes
        self.cpu_page_size_per_worker = aligned_kv_bytes
        self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes

    def _create_mmap_region(self) -> NPUSharedOffloadRegion:
        mmap_rank = 0 if self.replicated_layout else get_tensor_model_parallel_rank()
        return NPUSharedOffloadRegion(
            instance_id=self.vllm_config.instance_id,
            num_blocks=self.num_blocks,
            rank=mmap_rank,
            kv_bytes_per_block=self.kv_bytes_per_offloaded_block,
            cpu_page_size=self.cpu_page_size_per_worker,
        )

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> NPUCPUOffloadingWorker:
        mmap_region = self._create_mmap_region()
        return NPUCPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            mmap_region=mmap_region,
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            rotate_store_writers=self.replicated_layout,
        )

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            self._worker = self.create_worker(kv_caches)
        if self._worker is None:
            raise RuntimeError("Failed to create NPU CPU offloading worker")
        return self._worker
