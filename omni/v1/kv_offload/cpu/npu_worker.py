# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU worker for OffloadingConnector (CPUOffloadingSpec)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from omni_npu.v1.kv_offload.cpu.npu_shared_offload_region import (
    NPUSharedOffloadRegion,
)

logger = init_logger(__name__)

# Matches vllm-ascend csrc/torch_binding.cpp::swap_blocks_batch.
_DIRECTION_H2D = 0
_DIRECTION_D2H = 1
_DIRECTION_D2D = 2
_SWAP_BLOCKS_BATCH = None


def _get_swap_blocks_batch():
    """Return vendored ``_swap_blocks_batch.swap_blocks_batch`` or None."""
    global _SWAP_BLOCKS_BATCH
    if _SWAP_BLOCKS_BATCH is False:
        return None
    if _SWAP_BLOCKS_BATCH is not None:
        return _SWAP_BLOCKS_BATCH
    try:
        from omni_npu.v1.kv_offload.cpu import _swap_blocks_batch as _ext

        fn = _ext.swap_blocks_batch
        host_location = getattr(_ext, "host_location", "MISSING")
        cann_batch = getattr(_ext, "cann_memcpy_batch", "MISSING")
    except Exception:
        logger.warning(
            "omni_npu.v1.kv_offload.cpu._swap_blocks_batch unavailable; "
            "KV offload uses per-block copy_"
        )
        _SWAP_BLOCKS_BATCH = False
        return None
    _SWAP_BLOCKS_BATCH = fn
    logger.info(
        "KV offload indexed copy via omni_npu.v1.kv_offload.cpu._swap_blocks_batch "
        "host_location=%s cann_batch=%s",
        host_location,
        cann_batch,
    )
    return fn


def _block_byte_ptr(tensor: torch.Tensor, block: int, byte_offset: int) -> int:
    return int(tensor.data_ptr()) + int(block) * int(tensor.stride(0)) + int(byte_offset)


def _descriptors_from_ops(
    ops: list[tuple[torch.Tensor, int, int, torch.Tensor, int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build CPU int64 src/dst/size arrays for ``swap_blocks_batch``."""
    n = len(ops)
    src_np = np.empty(n, dtype=np.int64)
    dst_np = np.empty(n, dtype=np.int64)
    sz_np = np.empty(n, dtype=np.int64)
    for i, (src_t, src_b, src_off, dst_t, dst_b, dst_off, nbytes) in enumerate(ops):
        src_np[i] = _block_byte_ptr(src_t, src_b, src_off)
        dst_np[i] = _block_byte_ptr(dst_t, dst_b, dst_off)
        sz_np[i] = nbytes
    return (
        torch.from_numpy(src_np),
        torch.from_numpy(dst_np),
        torch.from_numpy(sz_np),
    )


def pin_memory_region(region: NPUSharedOffloadRegion) -> None:
    """HostRegister PINNED on the shared mmap so batch H2D/D2H can DMA.

    Does not MAPPED / as_npu_view: that aliases a device VA and
    aclrtMemcpyBatchAsync becomes illegal D2D.
    """
    if region.is_pinned:
        return
    try:
        from omni_npu.v1.kv_offload.cpu.host_register import pin_host_mmap
    except Exception:
        logger.info("host-register extension unavailable; mmap stays unpinned")
        return

    ptr = region._base.data_ptr()
    nbytes = int(region._base.nbytes)
    aligned_4k = ptr % 4096 == 0
    logger.info(
        "pin mmap PINNED: ptr=0x%x size=%d hugepage=%s 4K-align=%s",
        ptr,
        nbytes,
        region._hugepage,
        aligned_4k,
    )
    if not aligned_4k:
        logger.warning("mmap ptr is not 4K aligned; skip host register")
        return

    try:
        device_id = int(torch.npu.current_device())
    except Exception:
        device_id = 0
    t0 = time.monotonic()
    try:
        pin_host_mmap(region._base, device_id=device_id)
    except Exception as e:
        logger.warning(
            "HostRegister PINNED failed (%s); MemcpyBatchAsync stays on unpinned mmap",
            e,
        )
        return

    region.is_pinned = True
    region._register_ptr = ptr
    region.npu_base = None
    logger.info(
        "Pinned SharedOffloadRegion PINNED-only device=%s size=%d in %.3f s",
        device_id,
        nbytes,
        time.monotonic() - t0,
    )


@dataclass
class Transfer:
    job_id: int
    stream: torch.npu.Stream
    start_event: torch.npu.Event
    end_event: torch.npu.Event
    num_bytes: int
    batch_src: torch.Tensor | None = None
    batch_dst: torch.Tensor | None = None
    batch_sizes: torch.Tensor | None = None


def _block_slice(
    tensor: torch.Tensor, block: int, byte_offset: int, num_bytes: int
) -> torch.Tensor:
    if byte_offset == 0 and num_bytes == tensor.shape[1]:
        return tensor[block]
    return tensor[block, byte_offset:byte_offset + num_bytes]


class NpuSingleDirectionOffloadingHandler:
    """NPU<->CPU block transfers on one dedicated stream.

    Preferred path: ``swap_blocks_batch`` H2D/D2H directly against mmap.
    MemcpyBatchAttr host side is HOST (runtime validator), not UNREGISTERED.
    """

    def __init__(
        self,
        npu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        block_size_factor: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        npu_to_cpu: bool,
        mmap_region: NPUSharedOffloadRegion | None = None,
        rotate_store_writers: bool = False,
        tp_rank: int = 0,
        tp_size: int = 1,
    ):
        if not npu_tensors or len(npu_tensors) != len(cpu_tensors):
            raise ValueError(
                "npu_tensors and cpu_tensors must be non-empty and same length, "
                f"got {len(npu_tensors)} and {len(cpu_tensors)}"
            )

        for npu_tensor, cpu_tensor in zip(npu_tensors, cpu_tensors):
            if npu_tensor.dtype != torch.int8 or npu_tensor.ndim != 2:
                raise ValueError(
                    "npu_tensor must be 2-D int8, "
                    f"got dtype={npu_tensor.dtype} ndim={npu_tensor.ndim}"
                )
            if npu_tensor.device.type != "npu":
                raise ValueError(
                    f"npu_tensor device must be npu, got {npu_tensor.device.type}"
                )
            if cpu_tensor.dtype != torch.int8 or cpu_tensor.ndim != 2:
                raise ValueError(
                    "cpu_tensor must be 2-D int8, "
                    f"got dtype={cpu_tensor.dtype} ndim={cpu_tensor.ndim}"
                )
            if cpu_tensor.device.type not in ("cpu", "npu"):
                raise ValueError(
                    "cpu_tensor device must be cpu or npu, "
                    f"got {cpu_tensor.device.type}"
                )
            _, npu_page_size = npu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            if cpu_page_size != npu_page_size * block_size_factor:
                raise ValueError(
                    f"cpu_page_size ({cpu_page_size}) must equal "
                    f"npu_page_size ({npu_page_size}) * "
                    f"block_size_factor ({block_size_factor})"
                )

        self.src_tensors = npu_tensors if npu_to_cpu else cpu_tensors
        self.dst_tensors = cpu_tensors if npu_to_cpu else npu_tensors
        self.npu_to_cpu = npu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs
        self.src_block_size_factor = 1 if npu_to_cpu else block_size_factor
        self.dst_block_size_factor = block_size_factor if npu_to_cpu else 1
        self._mmap_region = mmap_region
        self._rotate_store_writers = rotate_store_writers
        self._tp_rank = tp_rank
        self._tp_size = max(tp_size, 1)

        self._transfer_events: dict[int, torch.npu.Event] = {}
        self._transfers: deque[Transfer] = deque()
        self._transfers_by_id: dict[int, Transfer] = {}
        self._stream = torch.npu.Stream()
        self._event_pool: list[torch.npu.Event] = []
        self._swap_blocks_batch = _get_swap_blocks_batch()
        # Direct batch DMA is only legal as H2D/D2H. Mapped NPU aliases
        # would force D2D, which aclrtMemcpyBatchAsync rejects.
        if cpu_tensors[0].device.type != "cpu":
            raise ValueError(
                "cpu_tensors must stay on CPU for H2D/D2H; as_npu_view makes "
                "MemcpyBatchAsync D2D and fails with ACL_ERROR_RT_PARAM_INVALID"
            )
        self._batch_direction = (
            _DIRECTION_D2H if npu_to_cpu else _DIRECTION_H2D
        )

    def _owns_store_block(self, dst_block: int) -> bool:
        """Replicated store rotation: each rank writes CPU blocks it owns."""
        if not (self.npu_to_cpu and self._rotate_store_writers and self._tp_size > 1):
            return True
        return int(dst_block) % self._tp_size == self._tp_rank

    def _plan_group_copies(
        self,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        group_data_refs: list[CanonicalKVCacheRef],
        src_logical_blocks_to_skip: int,
        dst_logical_blocks_to_skip: int,
    ) -> list[tuple[torch.Tensor, int, int, torch.Tensor, int, int, int]]:
        """Return list of (src, src_block, src_byte, dst, dst_block, dst_byte, n)."""
        ops: list[tuple[torch.Tensor, int, int, torch.Tensor, int, int, int]] = []
        for data_ref in group_data_refs:
            t_idx = data_ref.tensor_idx
            src_tensor = self.src_tensors[t_idx]
            dst_tensor = self.dst_tensors[t_idx]
            page_size = data_ref.page_size_bytes
            for i in range(group_size):
                src_sub = src_logical_blocks_to_skip + i
                dst_sub = dst_logical_blocks_to_skip + i
                src_block = int(group_src[src_sub // self.src_block_size_factor])
                dst_block = int(group_dst[dst_sub // self.dst_block_size_factor])
                if not self._owns_store_block(dst_block):
                    continue
                src_page = (src_sub % self.src_block_size_factor) * page_size
                dst_page = (dst_sub % self.dst_block_size_factor) * page_size
                ops.append(
                    (
                        src_tensor,
                        src_block,
                        src_page,
                        dst_tensor,
                        dst_block,
                        dst_page,
                        page_size,
                    )
                )
        return ops

    def _recycle_transfer(self, transfer: Transfer) -> TransferResult:
        transfer_time = 0.0
        self._event_pool.append(transfer.end_event)
        self._event_pool.append(transfer.start_event)
        self._transfer_events.pop(transfer.job_id, None)
        self._transfers_by_id.pop(transfer.job_id, None)
        return TransferResult(
            job_id=transfer.job_id,
            success=True,
            transfer_size=transfer.num_bytes,
            transfer_time=transfer_time,
        )

    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        if not isinstance(src_spec, BlockIDsLoadStoreSpec):
            raise TypeError(
                f"src_spec must be BlockIDsLoadStoreSpec, got {type(src_spec)}"
            )
        if not isinstance(dst_spec, BlockIDsLoadStoreSpec):
            raise TypeError(
                f"dst_spec must be BlockIDsLoadStoreSpec, got {type(dst_spec)}"
            )

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        npu_spec = src_spec if self.npu_to_cpu else dst_spec
        if not isinstance(npu_spec, GPULoadStoreSpec):
            raise TypeError(
                f"npu_spec must be GPULoadStoreSpec, got {type(npu_spec)}"
            )
        group_sizes = npu_spec.group_sizes
        block_indices = npu_spec.block_indices
        if len(group_sizes) != len(self.kv_cache_groups_data_refs):
            raise ValueError(
                "group_sizes length must match kv_cache_groups_data_refs, "
                f"got {len(group_sizes)} and "
                f"{len(self.kv_cache_groups_data_refs)}"
            )

        src_offset = 0
        dst_offset = 0
        planned_ops: list[
            tuple[torch.Tensor, int, int, torch.Tensor, int, int, int]
        ] = []

        for group_size, block_idx, group_data_refs in zip(
            group_sizes, block_indices, self.kv_cache_groups_data_refs
        ):
            if group_size == 0:
                continue

            src_skip = block_idx % self.src_block_size_factor
            dst_skip = block_idx % self.dst_block_size_factor
            src_count = group_size + src_skip
            dst_count = group_size + dst_skip

            dst_blocks_count = cdiv(dst_count, self.dst_block_size_factor)
            src_blocks_count = cdiv(src_count, self.src_block_size_factor)
            src_end = src_offset + src_blocks_count
            dst_end = dst_offset + dst_blocks_count
            if src_end > num_src_blocks or dst_end > num_dst_blocks:
                raise ValueError(
                    f"block range overflow: src_end={src_end} "
                    f"num_src_blocks={num_src_blocks} dst_end={dst_end} "
                    f"num_dst_blocks={num_dst_blocks}"
                )

            planned_ops.extend(
                self._plan_group_copies(
                    src_blocks[src_offset:src_end],
                    dst_blocks[dst_offset:dst_end],
                    group_size,
                    group_data_refs,
                    src_skip,
                    dst_skip,
                )
            )
            src_offset = src_end
            dst_offset = dst_end

        if src_offset != num_src_blocks or dst_offset != num_dst_blocks:
            raise ValueError(
                f"block coverage mismatch: src_offset={src_offset} "
                f"num_src_blocks={num_src_blocks} dst_offset={dst_offset} "
                f"num_dst_blocks={num_dst_blocks}"
            )
        num_transfer_bytes = sum(op[-1] for op in planned_ops)

        stream = self._stream
        start_event = (
            self._event_pool.pop() if self._event_pool else torch.npu.Event()
        )
        end_event = (
            self._event_pool.pop() if self._event_pool else torch.npu.Event()
        )

        if self.npu_to_cpu:
            stream.wait_stream(torch.npu.current_stream())

        batch_src: torch.Tensor | None = None
        batch_dst: torch.Tensor | None = None
        batch_sizes: torch.Tensor | None = None
        swap_fn = self._swap_blocks_batch

        def _copy_ops_python(
            ops: list[tuple[torch.Tensor, int, int, torch.Tensor, int, int, int]],
        ) -> None:
            for (
                src_tensor,
                src_block,
                src_byte,
                dst_tensor,
                dst_block,
                dst_byte,
                page_size,
            ) in ops:
                src = _block_slice(src_tensor, src_block, src_byte, page_size)
                dst = _block_slice(dst_tensor, dst_block, dst_byte, page_size)
                dst.copy_(src, non_blocking=True)

        with torch.npu.stream(stream):
            start_event.record(stream)
            if planned_ops:
                if swap_fn is not None:
                    batch_src, batch_dst, batch_sizes = _descriptors_from_ops(
                        planned_ops
                    )
                    swap_fn(
                        batch_src, batch_dst, batch_sizes, self._batch_direction
                    )
                else:
                    _copy_ops_python(planned_ops)
            end_event.record(stream)

        transfer = Transfer(
            job_id=job_id,
            stream=stream,
            start_event=start_event,
            end_event=end_event,
            num_bytes=num_transfer_bytes,
            batch_src=batch_src,
            batch_dst=batch_dst,
            batch_sizes=batch_sizes,
        )
        self._transfer_events[job_id] = end_event
        self._transfers.append(transfer)
        self._transfers_by_id[job_id] = transfer
        logger.debug(
            "KV offload submit: direction=%s job_id=%s bytes=%s "
            "planned_ops=%s pending=%s stream=%s",
            "store" if self.npu_to_cpu else "load",
            job_id,
            num_transfer_bytes,
            len(planned_ops),
            len(self._transfers),
            stream,
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        pending_before = len(self._transfers)
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            results.append(self._recycle_transfer(transfer))
        if results:
            logger.debug(
                "KV offload complete: direction=%s completed=%s "
                "pending_before=%s pending_after=%s job_ids=%s",
                "store" if self.npu_to_cpu else "load",
                len(results),
                pending_before,
                len(self._transfers),
                [result.job_id for result in results],
            )
        return results

    def wait(self, job_ids: set[int]) -> None:
        logger.debug(
            "KV offload wait enter: direction=%s jobs=%s pending=%s",
            "store" if self.npu_to_cpu else "load",
            sorted(job_ids),
            len(self._transfers),
        )
        for job_id in job_ids:
            transfer = self._transfers_by_id.get(job_id)
            if transfer is None:
                continue
            # Fence before CPU block reuse: mmap must see store data.
            transfer.end_event.synchronize()
        logger.debug(
            "KV offload wait exit: direction=%s jobs=%s pending=%s",
            "store" if self.npu_to_cpu else "load",
            sorted(job_ids),
            len(self._transfers),
        )

    def shutdown(self) -> None:
        while self._transfers:
            transfer = self._transfers.popleft()
            transfer.end_event.synchronize()
            self._recycle_transfer(transfer)
        self._transfer_events.clear()
        self._transfers_by_id.clear()
        self._event_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None


class NPUCPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for NPU CPU offloading.

    Composes two handlers and exposes submit_store / submit_load for
    OffloadingConnectorWorker. Under replicated_layout every rank maps the
    same mmap slot; store copies are rotated by CPU block id.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        mmap_region: NPUSharedOffloadRegion | None = None,
        tp_rank: int = 0,
        tp_size: int = 1,
        rotate_store_writers: bool = False,
    ):
        pin_memory = PIN_MEMORY
        self._tp_rank = tp_rank
        self._tp_size = max(tp_size, 1)
        self._rotate_store_writers = rotate_store_writers
        logger.info(
            "Allocating %d CPU tensors "
            "(mmap=%s, pin_memory=%s, rotate_store=%s, tp_rank=%d/%d)...",
            len(kv_caches.tensors),
            mmap_region is not None,
            pin_memory,
            rotate_store_writers,
            tp_rank,
            self._tp_size,
        )
        # Shared mmap is the CPU KV pool. MemcpyBatchAsync host attr is HOST.
        # PINNED (no MAPPED) makes that path async DMA without a device alias.
        if mmap_region is not None:
            pin_memory_region(mmap_region)
        logger.info(
            "NPU CPU offload path: indexed_copy=%s hugepage=%s "
            "pinned=%s npu_mapped=%s",
            _get_swap_blocks_batch() is not None,
            bool(getattr(mmap_region, "_hugepage", False)) if mmap_region else False,
            bool(getattr(mmap_region, "is_pinned", False)) if mmap_region else False,
            mmap_region.npu_base is not None if mmap_region else False,
        )

        npu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            npu_page_size_bytes = kv_cache_tensor.page_size_bytes
            npu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, npu_page_size_bytes)
            )
            cpu_page_size_bytes = npu_page_size_bytes * block_size_factor

            if mmap_region is not None:
                cpu_tensor = mmap_region.create_next_worker_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_blocks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_blocks,
                    cpu_page_size_bytes,
                    num_cpu_blocks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            npu_tensors.append(npu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = NpuSingleDirectionOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=True,
            mmap_region=mmap_region,
            rotate_store_writers=rotate_store_writers,
            tp_rank=tp_rank,
            tp_size=self._tp_size,
        )
        self._load_handler = NpuSingleDirectionOffloadingHandler(
            npu_tensors=npu_tensors,
            cpu_tensors=cpu_tensors,
            block_size_factor=block_size_factor,
            kv_cache_groups_data_refs=kv_caches.group_data_refs,
            npu_to_cpu=False,
        )

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async NPU -> CPU."""
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Async CPU -> NPU."""
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        # Store then load; order is not sorted by job_id. wait() keys by id.
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        self._store_handler.shutdown()
        self._load_handler.shutdown()
