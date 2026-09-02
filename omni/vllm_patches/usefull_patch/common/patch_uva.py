# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Patch vLLM releases/v0.25.1 buffer_utils UVA helpers for Ascend NPU.
# In this vLLM version, UvaBuffer uses buffer_utils.is_uva_available and
# buffer_utils.get_accelerator_view_from_cpu_tensor.

import torch
import vllm.v1.worker.gpu.buffer_utils as buffer_utils
from vllm.platforms import current_platform

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


_orig_is_uva_available = buffer_utils.is_uva_available
_orig_get_accelerator_view_from_cpu_tensor = (
    buffer_utils.get_accelerator_view_from_cpu_tensor
)


def _is_npu_platform() -> bool:
    return getattr(current_platform, "device_type", None) == "npu"


def _current_npu_device_index() -> int:
    try:
        return int(torch.npu.current_device())
    except Exception:
        return 0


def _check_npu_uva_prereqs(cpu_tensor: torch.Tensor) -> None:
    if cpu_tensor.device.type != "cpu":
        raise RuntimeError("UVA source tensor must be on CPU")
    if not cpu_tensor.is_pinned():
        raise RuntimeError("UVA source tensor must be pinned")
    if not is_uva_available():
        raise RuntimeError(
            "NPU UVA is not available. Ensure PYTORCH_NPU_ALLOC_CONF includes "
            "pinned_mem_register:True before process start, does not include "
            "pin_memory_expandable_segments:True, torch.npu is available, and "
            "omni_npu.allocator.npu_uva is built."
        )


def _load_npu_uva_module():
    try:
        from omni_npu.allocator import npu_uva
    except Exception as e:
        raise RuntimeError(
            "omni_npu.allocator.npu_uva is required for NPU UVA. Build "
            "omni-npu on an Ascend NPU machine with CANN installed, for "
            "example: cd omni && python setup.py build_ext --inplace"
        ) from e
    return npu_uva


def is_uva_available() -> bool:
    if not _is_npu_platform():
        return bool(_orig_is_uva_available())

    platform_is_uva_available = getattr(current_platform, "is_uva_available", None)
    if not callable(platform_is_uva_available):
        return False
    return bool(platform_is_uva_available())


def get_accelerator_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    if not _is_npu_platform():
        return _orig_get_accelerator_view_from_cpu_tensor(cpu_tensor)

    _check_npu_uva_prereqs(cpu_tensor)
    npu_uva = _load_npu_uva_module()
    return npu_uva.get_npu_view_from_cpu_tensor(
        cpu_tensor, _current_npu_device_index()
    )


@register_patch("NPUBufferUtilsUvaPatch", buffer_utils)
class BufferUtilsPatch(VLLMPatch):
    _attr_names_to_apply = [
        "is_uva_available",
        "get_accelerator_view_from_cpu_tensor",
    ]

    is_uva_available = is_uva_available
    get_accelerator_view_from_cpu_tensor = get_accelerator_view_from_cpu_tensor
