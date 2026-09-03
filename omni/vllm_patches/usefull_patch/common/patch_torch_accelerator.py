# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Redirect ``torch.accelerator`` memory APIs to their ``torch.npu`` equivalents.

Upstream vLLM migrated its memory bookkeeping from ``current_platform.*`` to
``torch.accelerator.*``. ``torch.accelerator`` does not delegate to NPU, so
those call sites either report wrong numbers or raise. In particular,
``torch.accelerator.get_memory_info()`` routes through c10's
CachingDeviceAllocator and asserts the backend allocator is a DeviceAllocator;
NPU's caching allocator is not, so it fails with "Allocator for npu is not a
DeviceAllocator".

vLLM calls ``load_general_plugins()`` from ``WorkerBase.init_worker()``, before
the worker class is constructed, so this lands well ahead of the first memory
snapshot in ``init_device()``.
"""

import torch

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("TorchAcceleratorMemory", torch.accelerator)
class TorchAcceleratorMemoryPatch(VLLMPatch):
    """
    Point torch.accelerator's memory APIs at the torch.npu implementations.
    """

    _attr_names_to_apply = [
        "empty_cache",
        "memory_stats",
        "memory_reserved",
        "memory_allocated",
        "reset_peak_memory_stats",
        "get_memory_info",
    ]

    # Bound as plain functions, not staticmethod(): apply() reads these back
    # out of cls.__dict__ and sets them on a module, so no descriptor binding
    # happens. A staticmethod object is only callable itself from Python 3.10.
    empty_cache = torch.npu.empty_cache
    memory_stats = torch.npu.memory_stats
    memory_reserved = torch.npu.memory_reserved
    memory_allocated = torch.npu.memory_allocated
    reset_peak_memory_stats = torch.npu.reset_peak_memory_stats
    get_memory_info = torch.npu.mem_get_info
