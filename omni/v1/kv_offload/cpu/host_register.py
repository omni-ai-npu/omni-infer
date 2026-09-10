# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""HostRegister wrappers for the shared mmap KV pool."""

from __future__ import annotations

import torch


def _host_register_ext():
    from omni_npu.v1.kv_offload.cpu import _host_register

    return _host_register


def _zero_copy_ext():
    from omni_npu.v1.kv_offload.cpu import _zero_copy_npu

    return _zero_copy_npu


def pin_host_mmap(host_tensor: torch.Tensor, device_id: int = 0) -> None:
    """aclrtHostRegisterV2(PINNED) on an existing mmap tensor.

    Keeps the CPU pointer for H2D/D2H. Does not create a mapped NPU VA
    (MAPPED would make MemcpyBatchAsync D2D).
    """
    if not isinstance(host_tensor, torch.Tensor):
        raise TypeError("host_tensor must be a torch.Tensor")
    if host_tensor.device.type != "cpu":
        raise ValueError("host_tensor must be a CPU tensor")

    # Force torch_npu to create the ACL runtime/context before HostRegister.
    # Bare aclInit() would race with torch_npu's own ACL lifetime.
    torch.zeros((1, 1), device="npu")
    _host_register_ext().pin_host(
        host_tensor.data_ptr(), host_tensor.nbytes, device_id
    )


def register_host_tensor(
    host_tensor: torch.Tensor, device_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Register a hugepage host tensor and wrap it as an NPU tensor.

    Same sequence as omni-cache ``NPUTensorRegister.host_tensor_register``:
    ``aclrtHostRegisterV2(PINNED | MAPPED)``, then
    ``register_hugepage_as_npu_tensor`` (MAPPED device alias).
    """
    if not isinstance(host_tensor, torch.Tensor):
        raise TypeError("host_tensor must be a torch.Tensor")
    if host_tensor.device.type != "cpu":
        raise ValueError("host_tensor must be a CPU tensor")

    # Same ACL warmup as pin_host_mmap; required before HostRegisterV2.
    torch.zeros((1, 1), device="npu")
    _host_register_ext().register_tensor(
        host_tensor.data_ptr(), host_tensor.nbytes, device_id
    )
    _, npu_tensor = _zero_copy_ext().register_hugepage_as_npu_tensor(
        host_tensor, device_id
    )
    return host_tensor, npu_tensor


def unregister_host_tensor(cpu_ptr: int) -> int:
    """Undo HostRegister. Returns 0 on success.

    Does not tear down ACL: process ACL lifetime is owned by torch_npu.
    """
    return int(_host_register_ext().unregister_tensor(cpu_ptr))
