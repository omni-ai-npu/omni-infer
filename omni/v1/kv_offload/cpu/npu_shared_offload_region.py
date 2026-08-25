# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Shared mmap region vendored from vLLM main, with Ascend-safe prefault.
"""
import errno
import mmap
import os
import time
from collections.abc import Callable

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
_HUGEPAGE_DIR = "/dev/hugepages"
_ENV_FLAG_TRUE = frozenset({"1", "true", "yes", "on"})


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in _ENV_FLAG_TRUE


def _align_up(num_bytes: int, alignment: int) -> int:
    return ((num_bytes + alignment - 1) // alignment) * alignment


def _hugetlbfs_mounted(path: str) -> bool:
    try:
        with open("/proc/mounts", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == path and parts[2] == "hugetlbfs":
                    return True
    except OSError:
        return False
    return False


def _check_shm_free_space(needed_bytes: int, path: str = "/dev/shm") -> None:
    st = os.statvfs(path)
    free_bytes = st.f_bavail * st.f_frsize
    if free_bytes < needed_bytes:
        raise RuntimeError(
            f"Not enough space in {path}: need {needed_bytes} bytes, "
            f"have {free_bytes} bytes"
        )


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


def _madvise_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    mmap_obj.madvise(_MADV_POPULATE_WRITE, offset, length)


def _fallback_populate_write(mmap_obj: mmap.mmap, offset: int, length: int) -> None:
    # Touch one byte per page via a read-modify-write so existing bytes are
    # preserved — a peer worker may have already written KV data into this
    # shared mmap by the time we run on a kernel without MADV_POPULATE_WRITE.
    arr = np.frombuffer(mmap_obj, dtype=np.uint8)
    arr[offset:offset + length:mmap.PAGESIZE] |= 0


def _mlock_host_tensor(tensor: torch.Tensor) -> None:
    """Lock a /dev/shm mapping in RAM so HostRegister / DMA pages stay resident.

    Hugepages are already non-swappable; callers should skip this path when
    ``OMNI_KV_OFFLOAD_HUGEPAGE`` is enabled.
    """
    try:
        from omni_npu.v1.kv_offload.cpu import _zero_copy_npu

        if _zero_copy_npu.mlock_host(tensor.data_ptr(), tensor.nbytes):
            logger.info("mmap mlock succeeded for %s bytes", tensor.nbytes)
        else:
            logger.warning("mmap mlock failed")
    except Exception as e:
        logger.warning("mmap mlock failed: %s", e)


def _get_populate_write_fn(
    mmap_obj: mmap.mmap,
) -> Callable[[mmap.mmap, int, int], None]:
    """Select the pre-faulting method once for this mmap."""
    try:
        _madvise_populate_write(mmap_obj, 0, mmap.PAGESIZE)
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        logger.warning(
            "MADV_POPULATE_WRITE is not supported; falling back to per-page "
            "writes for mmap pre-population. Startup may be slower."
        )
        return _fallback_populate_write
    return _madvise_populate_write


class NPUSharedOffloadRegion:
    """
    Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/hugepages/vllm_offload_{engine_id}.mmap when
    ``OMNI_KV_OFFLOAD_HUGEPAGE`` is 1/true/yes/on; otherwise /dev/shm.
    Hugepage size is ``OMNI_KV_OFFLOAD_HUGEPAGE_SIZE`` in bytes (default 2 MiB).
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        instance_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        if kv_bytes_per_block % self.page_size != 0:
            raise ValueError(
                f"kv_bytes_per_block ({kv_bytes_per_block}) must be a "
                f"multiple of page size ({self.page_size})"
            )

        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self.total_size_bytes = self.num_blocks * self._row_stride
        self.hugepage_size = int(
            os.getenv("OMNI_KV_OFFLOAD_HUGEPAGE_SIZE", str(2 * 1024 * 1024))
        )
        self._hugepage = _env_flag_enabled("OMNI_KV_OFFLOAD_HUGEPAGE")
        if self._hugepage:
            if not _hugetlbfs_mounted(_HUGEPAGE_DIR):
                raise RuntimeError(
                    f"OMNI_KV_OFFLOAD_HUGEPAGE is enabled but {_HUGEPAGE_DIR} "
                    "is not a hugetlbfs mount. Run setup_hugetlbfs.sh first."
                )
            mmap_dir = _HUGEPAGE_DIR
            self.mmap_size = _align_up(self.total_size_bytes, self.hugepage_size)
        else:
            mmap_dir = "/dev/shm"
            self.mmap_size = self.total_size_bytes
        self.mmap_path = f"{mmap_dir}/vllm_offload_{instance_id}.mmap"
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            self._worker_offset = rank * cpu_page_size
            self._worker_area_end = (rank + 1) * cpu_page_size
        else:
            self._worker_offset = 0
            self._worker_area_end = kv_bytes_per_block

        try:
            self.fd: int | None = os.open(
                self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
        except FileExistsError:
            # Joiner path — another worker won O_EXCL. Reopen and wait
            # for the file to reach expected size.
            self.fd = os.open(self.mmap_path, os.O_RDWR, 0o600)
            try:
                _wait_for_file_size(self.fd, self.mmap_size)
            except OSError:
                os.close(self.fd)
                raise
            logger.info("Opened existing mmap file %s", self.mmap_path)
        else:
            # Creator path. We won O_EXCL, so we own the file: any
            # failure here must clean up so concurrent joiners don't
            # land on a 0-byte stub and spin in _wait_for_file_size
            # for the full 30 s timeout.
            try:
                if not self._hugepage:
                    _check_shm_free_space(self.mmap_size)
                os.ftruncate(self.fd, self.mmap_size)
            except (RuntimeError, OSError):
                os.unlink(self.mmap_path)
                os.close(self.fd)
                raise
            self._creator = True
            logger.info(
                "Created mmap file %s (%.2f GB, hugepage=%s, page=%d, mmap_size=%d)",
                self.mmap_path,
                self.mmap_size / 1e9,
                self._hugepage,
                self.hugepage_size,
                self.mmap_size,
            )

        self.mmap_obj: mmap.mmap | None = mmap.mmap(
            self.fd,
            self.mmap_size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        populate_write_fn = _get_populate_write_fn(self.mmap_obj)

        if rank is not None:
            # Populate only this worker's pages (one slot per block row).
            worker_offset = rank * cpu_page_size
            _t0 = time.perf_counter()
            page_size = self.page_size
            for block in range(num_blocks):
                raw_offset = block * self._row_stride + worker_offset
                aligned_offset = (raw_offset // page_size) * page_size
                end = raw_offset + cpu_page_size
                aligned_length = end - aligned_offset
                populate_write_fn(self.mmap_obj, aligned_offset, aligned_length)
            logger.debug(
                "mmap prefault worker slots: %d blocks in %.3f s",
                num_blocks,
                time.perf_counter() - _t0,
            )
        else:
            # No rank — populate the entire shared region in one call.
            _t0 = time.perf_counter()
            populate_write_fn(self.mmap_obj, 0, self.mmap_size)
            logger.debug(
                "mmap prefault entire region: %.3f s", time.perf_counter() - _t0
            )

        self._base = torch.frombuffer(memoryview(self.mmap_obj), dtype=torch.int8)
        self._views: list[torch.Tensor] = []
        self._npu_views: list[torch.Tensor] = []
        self._canonical_offset = 0
        self.is_pinned: bool = False
        self._register_ptr: int | None = None
        self.npu_base: torch.Tensor | None = None
        if self._hugepage:
            logger.info(
                "skip mmap mlock: hugepages are already non-swappable (%s bytes)",
                self._base.nbytes,
            )
        else:
            _mlock_host_tensor(self._base)

    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        Each worker_block cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and block concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_blocks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per block for this  tensor.
        """
        if self.rank is None:
            raise RuntimeError("create_next_worker_view requires rank to be set")
        new_offset = self._worker_offset + tensor_page_size
        if new_offset > self._worker_area_end:
            raise ValueError(
                f"Worker offset {new_offset} exceeds worker area end "
                f"{self._worker_area_end} (overflowed by "
                f"{new_offset - self._worker_area_end} bytes)"
            )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def as_npu_view(self, cpu_view: torch.Tensor) -> torch.Tensor:
        """Return an NPU tensor aliasing the same mmap bytes as ``cpu_view``."""
        if self.npu_base is None:
            raise RuntimeError("as_npu_view requires npu_base to be initialized")
        view = torch.as_strided(
            self.npu_base,
            size=tuple(cpu_view.size()),
            stride=tuple(cpu_view.stride()),
            storage_offset=cpu_view.storage_offset(),
        )
        self._npu_views.append(view)
        return view

    def create_next_canonical_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view shared by all workers for one
        canonical tensor (canonical layout).

        Must be called once per canonical tensor, instead of
        create_next_worker_view. The full mmap layout is:

            |<-------- canonical area ------->|<-------- unused ------->|
            |  all workers share this area    |                         |
            |                                 |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            | [ canonical_t0 | canonical_t1 ] |                         |
            ^                ^
            _canonical_offset=0, then advances by each tensor's size

        Each canonical_t{i} cell is that tensor's canonical page for the
        block. Canonical areas are carved consecutively from the start of
        each block row; consecutive rows are separated by row_stride. Every
        worker gets the identical byte ranges and writes only its disjoint
        bytes within them, as described by its canonical mappings — unlike
        create_next_worker_view, which gives each worker a private
        cpu_page_size slot per row.

        The trailing unused bytes exist only when the canonical pages sum to
        less than row_stride: page-alignment padding of the row, or
        deduplication of KV replicated across workers (e.g. the MLA latent),
        where one canonical copy replaces world_size worker copies.

        Args:
            tensor_page_size: Canonical bytes per block for this tensor.
        """
        new_offset = self._canonical_offset + tensor_page_size
        if new_offset > self._row_stride:
            raise ValueError(
                f"Canonical offset {new_offset} exceeds row stride "
                f"{self._row_stride}"
            )
        view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._canonical_offset,
        )
        self._canonical_offset = new_offset
        self._views.append(view)
        return view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_blocks, row_stride_bytes). Secondary tiers address
        block *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        if np_arr.ctypes.data != self._base.data_ptr():
            raise RuntimeError(
                "view()/numpy() created a copy instead of sharing the mmap "
                "buffer; secondary tiers require zero-copy access to primary "
                "KV data"
            )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        if self.is_pinned and self._register_ptr:
            try:
                from omni_npu.v1.kv_offload.cpu.host_register import (
                    unregister_host_tensor,
                )

                ret = unregister_host_tensor(self._register_ptr)
                if ret != 0:
                    logger.warning(
                        "unregister_tensor failed for rank=%s (code=%s)",
                        self.rank,
                        ret,
                    )
            except Exception:
                logger.warning("unregister_tensor failed", exc_info=True)
            self.is_pinned = False
            self._register_ptr = None
        if self._npu_views is not None:
            self._npu_views.clear()
        self.npu_base = None
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
        if self._creator and getattr(self, "mmap_path", None):
            try:
                os.unlink(self.mmap_path)
                logger.info("Removed mmap file %s", self.mmap_path)
            except Exception:
                logger.warning(
                    "Failed to unlink path %s", self.mmap_path, exc_info=True
                )
            self._creator = False


# Back-compat name used by older omniinfer imports.
SharedOffloadRegion = NPUSharedOffloadRegion
