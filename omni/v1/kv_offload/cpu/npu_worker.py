# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU worker for OffloadingConnector (CPUOffloadingSpec)."""

from __future__ import annotations

import queue
import threading
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
    end_event: torch.npu.Event
    batch_src: torch.Tensor | None = None
    batch_dst: torch.Tensor | None = None
    batch_sizes: torch.Tensor | None = None


@dataclass
class QueuedJob:
    job_id: int
    src_spec: LoadStoreSpec
    dst_spec: LoadStoreSpec
    wait_stream: object | None = None


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
        self._device_index = npu_tensors[0].device.index or 0
        self._init_async_submit_state()

    def _init_async_submit_state(self) -> None:
        """Host-side queue so plan/enqueue leave the execute_model thread."""
        if getattr(self, "_lock", None) is not None:
            return
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queued: dict[int, QueuedJob] = {}
        self._submit_errors: dict[int, BaseException] = {}
        self._submit_queue: queue.Queue[QueuedJob | None] = queue.Queue()
        self._submit_thread: threading.Thread | None = None

    def _owns_store_block(self, dst_block: int) -> bool:
        """Replicated store rotation: each rank writes CPU blocks it owns."""
        if not (self.npu_to_cpu and self._rotate_store_writers and self._tp_size > 1):
            return True
        return int(dst_block) % self._tp_size == self._tp_rank

    def _group_block_indices(
        self,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_logical_blocks_to_skip: int,
        dst_logical_blocks_to_skip: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized src/dst subscripts and physical block ids for one group."""
        src_subs = np.arange(
            src_logical_blocks_to_skip,
            src_logical_blocks_to_skip + group_size,
            dtype=np.int64,
        )
        dst_subs = np.arange(
            dst_logical_blocks_to_skip,
            dst_logical_blocks_to_skip + group_size,
            dtype=np.int64,
        )
        src_blocks = np.asarray(group_src, dtype=np.int64)[
            src_subs // self.src_block_size_factor
        ]
        dst_blocks = np.asarray(group_dst, dtype=np.int64)[
            dst_subs // self.dst_block_size_factor
        ]
        if self.npu_to_cpu and self._rotate_store_writers and self._tp_size > 1:
            mask = np.mod(dst_blocks, self._tp_size) == self._tp_rank
            src_subs = src_subs[mask]
            dst_subs = dst_subs[mask]
            src_blocks = src_blocks[mask]
            dst_blocks = dst_blocks[mask]
        return src_subs, dst_subs, src_blocks, dst_blocks

    def _fill_group_descriptors(
        self,
        src_ptrs: np.ndarray,
        dst_ptrs: np.ndarray,
        sizes: np.ndarray,
        offset: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        group_data_refs: list[CanonicalKVCacheRef],
        src_logical_blocks_to_skip: int,
        dst_logical_blocks_to_skip: int,
    ) -> int:
        """Write one group's copy descriptors into preallocated int64 buffers."""
        if group_size <= 0 or not group_data_refs:
            return offset
        src_subs, dst_subs, src_blocks, dst_blocks = self._group_block_indices(
            group_src,
            group_dst,
            group_size,
            src_logical_blocks_to_skip,
            dst_logical_blocks_to_skip,
        )
        n = int(src_blocks.size)
        if n == 0:
            return offset
        src_factor = self.src_block_size_factor
        dst_factor = self.dst_block_size_factor
        for data_ref in group_data_refs:
            src_tensor = self.src_tensors[data_ref.tensor_idx]
            dst_tensor = self.dst_tensors[data_ref.tensor_idx]
            page_size = int(data_ref.page_size_bytes)
            sl = slice(offset, offset + n)
            src_ptrs[sl] = (
                int(src_tensor.data_ptr())
                + src_blocks * int(src_tensor.stride(0))
                + (src_subs % src_factor) * page_size
            )
            dst_ptrs[sl] = (
                int(dst_tensor.data_ptr())
                + dst_blocks * int(dst_tensor.stride(0))
                + (dst_subs % dst_factor) * page_size
            )
            sizes[sl] = page_size
            offset += n
        return offset

    def _iter_planned_groups(
        self,
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
        group_sizes,
        block_indices,
        num_src_blocks: int,
        num_dst_blocks: int,
    ):
        """Yield per-group slices after the same coverage checks as transfer."""
        src_offset = 0
        dst_offset = 0
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
            yield (
                group_size,
                group_data_refs,
                src_blocks[src_offset:src_end],
                dst_blocks[dst_offset:dst_end],
                src_skip,
                dst_skip,
            )
            src_offset = src_end
            dst_offset = dst_end

        if src_offset != num_src_blocks or dst_offset != num_dst_blocks:
            raise ValueError(
                f"block coverage mismatch: src_offset={src_offset} "
                f"num_src_blocks={num_src_blocks} dst_offset={dst_offset} "
                f"num_dst_blocks={num_dst_blocks}"
            )

    def _for_each_planned_group(
        self,
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
        group_sizes,
        block_indices,
        num_src_blocks: int,
        num_dst_blocks: int,
        visitor=None,
    ) -> None:
        """Walk planned groups once; visitor receives the unpacked group tuple."""
        for planned in self._iter_planned_groups(
            src_blocks, dst_blocks, group_sizes, block_indices,
            num_src_blocks, num_dst_blocks,
        ):
            if visitor is not None:
                visitor(*planned)

    def _plan_transfer_descriptors(
        self,
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
        group_sizes,
        block_indices,
        num_src_blocks: int,
        num_dst_blocks: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build all-group src/dst/size descriptor arrays without Python ops."""
        max_ops = 0
        for group_size, group_data_refs in zip(
            group_sizes, self.kv_cache_groups_data_refs
        ):
            if group_size:
                max_ops += int(group_size) * len(group_data_refs)
        src_ptrs = np.empty(max_ops, dtype=np.int64)
        dst_ptrs = np.empty(max_ops, dtype=np.int64)
        sizes = np.empty(max_ops, dtype=np.int64)
        offset = 0

        def _fill(
            group_size,
            group_data_refs,
            group_src,
            group_dst,
            src_skip,
            dst_skip,
        ):
            nonlocal offset
            offset = self._fill_group_descriptors(
                src_ptrs,
                dst_ptrs,
                sizes,
                offset,
                group_src,
                group_dst,
                group_size,
                group_data_refs,
                src_skip,
                dst_skip,
            )

        self._for_each_planned_group(
            src_blocks, dst_blocks, group_sizes, block_indices,
            num_src_blocks, num_dst_blocks,
            None if max_ops <= 0 else _fill,
        )
        return src_ptrs[:offset], dst_ptrs[:offset], sizes[:offset]

    def _recycle_transfer(self, transfer: Transfer) -> TransferResult:
        self._event_pool.append(transfer.end_event)
        self._transfer_events.pop(transfer.job_id, None)
        self._transfers_by_id.pop(transfer.job_id, None)
        return TransferResult(job_id=transfer.job_id, success=True)

    def submit(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Queue a transfer. Plan/enqueue run on the submit thread."""
        return self._queue_transfer(job_id, src_spec, dst_spec)

    def _queue_transfer(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        wait_stream = None
        if self.npu_to_cpu:
            # Capture the compute stream here. transfer_async runs on the
            # submit thread, where current_stream() is not the writer stream.
            try:
                wait_stream = torch.npu.current_stream()
            except Exception:
                wait_stream = None
        job = QueuedJob(
            job_id=job_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
            wait_stream=wait_stream,
        )
        self._ensure_submit_thread()
        with self._lock:
            if job_id in self._queued or job_id in self._transfers_by_id:
                raise ValueError(f"duplicate offload job_id {job_id}")
            self._queued[job_id] = job
        self._submit_queue.put(job)
        return True

    def _ensure_submit_thread(self) -> None:
        with self._lock:
            if self._submit_thread is not None and self._submit_thread.is_alive():
                return
            self._submit_thread = threading.Thread(
                target=self._submit_loop,
                name=(
                    "kv-offload-store-submit"
                    if self.npu_to_cpu
                    else "kv-offload-load-submit"
                ),
                daemon=True,
            )
            self._submit_thread.start()

    def _submit_loop(self) -> None:
        torch.npu.set_device(self._device_index)
        while True:
            job = self._submit_queue.get()
            try:
                if job is None:
                    break
                self.transfer_async(
                    job.job_id,
                    job.src_spec,
                    job.dst_spec,
                    wait_stream=job.wait_stream,
                )
            except Exception as exc:
                if job is None:
                    break
                logger.exception(
                    "KV offload submit failed job=%s",
                    job.job_id,
                )
                with self._lock:
                    self._queued.pop(job.job_id, None)
                    self._submit_errors[job.job_id] = exc
                    self._cv.notify_all()
            finally:
                self._submit_queue.task_done()

    def transfer_async(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: LoadStoreSpec,
        wait_stream: object | None = None,
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

        src_ptrs, dst_ptrs, op_sizes = self._plan_transfer_descriptors(
            src_blocks,
            dst_blocks,
            group_sizes,
            block_indices,
            num_src_blocks,
            num_dst_blocks,
        )
        num_ops = int(op_sizes.size)

        stream = self._stream
        with self._lock:
            end_event = self._event_pool.pop() if self._event_pool else None
        if end_event is None:
            end_event = torch.npu.Event()

        wait_compute = int(self.npu_to_cpu)
        if wait_compute:
            compute_stream = wait_stream
            if compute_stream is None:
                compute_stream = torch.npu.current_stream()
            stream.wait_stream(compute_stream)

        batch_src: torch.Tensor | None = None
        batch_dst: torch.Tensor | None = None
        batch_sizes: torch.Tensor | None = None
        swap_fn = self._swap_blocks_batch

        def _copy_ops_python() -> None:
            src_factor = self.src_block_size_factor
            dst_factor = self.dst_block_size_factor

            def _copy_group(
                group_size,
                group_data_refs,
                group_src,
                group_dst,
                src_skip,
                dst_skip,
            ):
                src_subs, dst_subs, src_blk, dst_blk = self._group_block_indices(
                    group_src,
                    group_dst,
                    group_size,
                    src_skip,
                    dst_skip,
                )
                for data_ref in group_data_refs:
                    src_tensor = self.src_tensors[data_ref.tensor_idx]
                    dst_tensor = self.dst_tensors[data_ref.tensor_idx]
                    page_size = int(data_ref.page_size_bytes)
                    src_pages = (src_subs % src_factor) * page_size
                    dst_pages = (dst_subs % dst_factor) * page_size
                    for i in range(src_blk.size):
                        src = _block_slice(
                            src_tensor, int(src_blk[i]), int(src_pages[i]), page_size
                        )
                        dst = _block_slice(
                            dst_tensor, int(dst_blk[i]), int(dst_pages[i]), page_size
                        )
                        dst.copy_(src, non_blocking=True)

            self._for_each_planned_group(
                src_blocks, dst_blocks, group_sizes, block_indices,
                num_src_blocks, num_dst_blocks, _copy_group,
            )

        with torch.npu.stream(stream):
            if num_ops:
                if swap_fn is not None:
                    batch_src = torch.from_numpy(np.ascontiguousarray(src_ptrs))
                    batch_dst = torch.from_numpy(np.ascontiguousarray(dst_ptrs))
                    batch_sizes = torch.from_numpy(np.ascontiguousarray(op_sizes))
                    swap_fn(
                        batch_src, batch_dst, batch_sizes, self._batch_direction
                    )
                else:
                    _copy_ops_python()
            end_event.record(stream)

        transfer = Transfer(
            job_id=job_id,
            stream=stream,
            end_event=end_event,
            batch_src=batch_src,
            batch_dst=batch_dst,
            batch_sizes=batch_sizes,
        )
        with self._lock:
            self._transfer_events[job_id] = end_event
            self._transfers.append(transfer)
            self._transfers_by_id[job_id] = transfer
            self._queued.pop(job_id, None)
            self._cv.notify_all()
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        with self._lock:
            while self._transfers and self._transfers[0].end_event.query():
                transfer = self._transfers.popleft()
                results.append(self._recycle_transfer(transfer))
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            transfer = None
            with self._cv:
                while True:
                    err = self._submit_errors.get(job_id)
                    if err is not None:
                        raise err
                    transfer = self._transfers_by_id.get(job_id)
                    if transfer is not None or job_id not in self._queued:
                        break
                    self._cv.wait(timeout=1.0)
            if transfer is None:
                continue
            # Fence before CPU block reuse: mmap must see store data.
            transfer.end_event.synchronize()

    def shutdown(self) -> None:
        thread = self._submit_thread
        if thread is not None and thread.is_alive():
            self._submit_queue.put(None)
            thread.join(timeout=60.0)
        self._submit_thread = None
        while True:
            try:
                leftover = self._submit_queue.get_nowait()
            except queue.Empty:
                break
            if leftover is not None:
                logger.warning(
                    "KV offload shutdown drops queued job=%s",
                    leftover.job_id,
                )
        with self._lock:
            self._queued.clear()
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
        """Async NPU -> CPU. Plan/enqueue run on the store submit thread."""
        return self._store_handler.submit(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Async CPU -> NPU. Plan/enqueue run on the load submit thread."""
        return self._load_handler.submit(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        # Store then load; order is not sorted by job_id. wait() keys by id.
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        self._store_handler.shutdown()
        self._load_handler.shutdown()
