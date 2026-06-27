# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Zero-copy NPU tensor test and utilities.

This module provides test utilities for zero-copy NPU tensor operations.
"""

import torch
import os
import mmap

HUGEPAGE_DIR = "/dev/hugepages"
MAP_HUGETLB = 0x40000


def test_zero_copy():
    """Test zero-copy tensor registration."""
    # Import here to avoid errors if not built
    from . import zero_copy_npu
    import torch_npu

    x = torch.zeros((100, 2), device="npu:0")

    SIZE = 1024 * 1024 * 1024
    device_id = 0
    FILE_PATH = os.path.join(HUGEPAGE_DIR, "zero_copy_hugepage.bin")

    fd = os.open(FILE_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, SIZE)
    buf = mmap.mmap(fd, SIZE,
                    flags=mmap.MAP_SHARED | MAP_HUGETLB,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)

    host_tensor = torch.frombuffer(buf, dtype=torch.float32, count=SIZE // 4).contiguous()

    print("Host ptr :", hex(host_tensor.data_ptr()))
    print("is_pinned:", host_tensor.is_pinned())

    _, npu_tensor = zero_copy_npu.register_hugepage_as_npu_tensor(host_tensor, device_id)

    print("NPU tensor device:", npu_tensor.device)
    print("NPU tensor ptr   :", hex(npu_tensor.data_ptr()))
    print("NPU tensor size  :", npu_tensor.storage().nbytes())
    print("Host tensor ptr  :", hex(host_tensor.data_ptr()))
    print("Same storage?    :", host_tensor.data_ptr() == npu_tensor.data_ptr())

    host_tensor[:10] = 9.0

    tmp = npu_tensor + 1
    print(tmp[:100])
    print(f"tmp:{hex(tmp.data_ptr())}")
    out = (npu_tensor + 1.0)[:10].sum()
    print("NPU result:", out.item())

    print("Before change:", host_tensor[:10])
    npu_tensor[:10] = 888.22587
    import time
    time.sleep(.001)
    print("Host sees change:", host_tensor[:10])

    zero_copy_npu.unregister()
    os.close(fd)
    os.unlink(FILE_PATH)
    buf.close()


if __name__ == "__main__":
    test_zero_copy()
