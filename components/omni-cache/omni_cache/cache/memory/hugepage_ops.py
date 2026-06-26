# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Hugepage memory operations for KV cache."""

import ctypes
import os
import mmap
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from .constants import REGIST_SIZE

if TYPE_CHECKING:
    from .memory_pool import KVCacheMemoryPool

logger = init_logger("vllm.v1.omni")


def open_hugepage_file(hugepage_path: str) -> int:
    """Open hugepage file and return file descriptor.

    Args:
        hugepage_path: Path to hugepage file.

    Returns:
        File descriptor.

    Raises:
        FileNotFoundError: If hugepage file doesn't exist.
    """
    if not os.path.exists(hugepage_path):
        raise FileNotFoundError(f"Hugepage file not found: {hugepage_path}")
    return os.open(hugepage_path, os.O_RDWR)


def create_memory_mapping(fd: int, mmap_size: int) -> mmap.mmap:
    """Create memory mapping from file descriptor.

    For hugetlbfs, ftruncate the file to the required size (aligned to
    hugepage boundary) before mmap.

    Args:
        fd: File descriptor.
        mmap_size: Size of memory mapping.

    Returns:
        Memory-mapped object.
    """
    # Align to 2MB hugepage boundary
    HUGEPAGE_SIZE = 2 * 1024 * 1024
    aligned_size = ((mmap_size + HUGEPAGE_SIZE - 1) // HUGEPAGE_SIZE) * HUGEPAGE_SIZE
    os.ftruncate(fd, aligned_size)
    mmap_obj = mmap.mmap(fd, aligned_size, flags=mmap.MAP_SHARED,
                         prot=mmap.PROT_READ | mmap.PROT_WRITE)

    # mlock the memory using ctypes to prevent swapping
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        buf = (ctypes.c_char * aligned_size).from_buffer(mmap_obj)
        result = libc.mlock(ctypes.byref(buf), aligned_size)
        if result != 0:
            errno = ctypes.get_errno()
            logger.warning(f"OmniCache mlock failed with error: {errno}")
        else:
            logger.info(f"OmniCache mlock succeeded for {aligned_size} bytes")
    except Exception as e:
        logger.warning(f"OmniCache mlock failed: {e}")

    return mmap_obj


def create_shared_tensor(
    mmap_obj: mmap.mmap,
    dtype: torch.dtype,
    size_total: int,
    element_size: int,
    num_ranks: int,
    rank: int,
    rank_stride: int = None,
) -> torch.Tensor:
    """Create shared tensor from memory-mapped buffer.

    Args:
        mmap_obj: Memory-mapped object.
        dtype: Tensor data type.
        size_total: Total size in bytes for this rank's data.
        element_size: Element size in bytes.
        num_ranks: Number of ranks.
        rank: Current rank.
        rank_stride: Byte stride between consecutive ranks in the mmap.
            Defaults to size_total (no padding between ranks).

    Returns:
        Shared tensor.
    """
    if rank_stride is None:
        rank_stride = size_total
    if num_ranks == 1:
        return torch.frombuffer(
            mmap_obj,
            dtype=dtype,
            count=size_total // element_size
        )
    else:
        offset_bytes = rank * rank_stride
        buf_view = memoryview(mmap_obj)[offset_bytes:]
        return torch.frombuffer(
            buf_view,
            dtype=dtype,
            count=size_total // element_size
        )


def split_kvi_tensors(
    shared_tensor: torch.Tensor,
    shapes: list,
    head_size_ratio: list,
    head_total_slices: int,
    num_layers: int
) -> list:
    """Split shared tensor into KVI tensors per component.

    Args:
        shared_tensor: Shared tensor buffer.
        shapes: List of shapes for each component.
        head_size_ratio: Head size ratios.
        head_total_slices: Total head slices.
        num_layers: Number of layers.

    Returns:
        List of KVI tensors.
    """
    cnt = shared_tensor.numel()
    if cnt % head_total_slices != 0:
        raise ValueError(
            f"Expected total elements divisible by {head_total_slices}, got {cnt}."
        )

    split_sizes = [cnt * r // head_total_slices for r in head_size_ratio]
    kvi_tensors = list(torch.split(shared_tensor, split_sizes))

    for i, (tensor, shape) in enumerate(zip(kvi_tensors, shapes)):
        kvi_tensors[i] = tensor.view([num_layers] + list(shape))

    return kvi_tensors


def setup_device_mapping(
    pool: "KVCacheMemoryPool",
    shared_tensor: torch.Tensor
) -> torch.Tensor:
    """Set up device mapping for shared tensor.

    Args:
        pool: Memory pool instance.
        shared_tensor: Shared tensor to map.

    Returns:
        Device-mapped tensor.
    """
    enable_dsa = pool.enable_dsa
    enable_host_mapping = enable_dsa or int(os.getenv("ENABLE_HOST_MAPPING", "1"))

    if not enable_host_mapping:
        return shared_tensor

    logger.warning(f"<<< host_swap_device: enable_dsa={enable_dsa}")

    ptr = shared_tensor.data_ptr()
    assert ptr % REGIST_SIZE == 0, f"Address 0x{ptr:x} not 4K aligned!"

    device_id = pool.device.index if pool.device else 0
    _, tensor_swap = pool.npu_tensor_register.host_tensor_register(
        shared_tensor, device_id=device_id
    )
    return tensor_swap


def create_layer_tensor_tuples(
    kvi_tensors_swap: list,
    num_layers: int
) -> list:
    """Create list of tensor tuples per layer.

    Args:
        kvi_tensors_swap: List of KVI tensors (device-mapped).
        num_layers: Number of layers.

    Returns:
        List of tensor tuples, one per layer.
    """
    shared_tensor_list = []
    for kvi_tensor in kvi_tensors_swap:
        tensor_list_swap = list(kvi_tensor.unsqueeze(-2).unbind(dim=0))
        shared_tensor_list.append(tensor_list_swap)

    return [tuple(l[i] for l in shared_tensor_list) for i in range(num_layers)]
