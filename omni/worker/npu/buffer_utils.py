# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU buffer implementation for the upstream V2 runner."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from omni_npu.worker.npu.mode import RuntimeMode, resolve_mode


def _pin_memory_enabled() -> bool:
    try:
        from vllm.utils.torch_utils import PIN_MEMORY

        return bool(PIN_MEMORY)
    except Exception:  # noqa: BLE001
        return True


def _current_npu_device() -> torch.device:
    return torch.device("npu", torch.npu.current_device())


def _accelerator_view(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """Alias pinned host memory as an NPU tensor."""
    # Resolved off the module at call time on purpose: NPUBufferUtilsUvaPatch
    # rebinds this name, and a from-import here would freeze the upstream
    # version that raises on NPU.
    import vllm.v1.worker.gpu.buffer_utils as up_buffer_utils

    return up_buffer_utils.get_accelerator_view_from_cpu_tensor(cpu_tensor)


class NPUUvaBuffer:
    """Expose CPU, NumPy, and device views of an upstream UVA buffer."""

    __slots__ = ("cpu", "np", "_dev", "_fast")

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype) -> None:
        self._fast = resolve_mode() is RuntimeMode.NPU_UVA_VIEW
        # The UVA view requires pinned memory; _check_npu_uva_prereqs rejects
        # anything else, so the fast path cannot honour PIN_MEMORY=0.
        self.cpu = torch.zeros(
            size,
            dtype=dtype,
            device="cpu",
            pin_memory=True if self._fast else _pin_memory_enabled(),
        )
        self.np = self.cpu.numpy()
        self._dev = (
            _accelerator_view(self.cpu)
            if self._fast
            else torch.zeros(size, dtype=dtype, device=_current_npu_device())
        )

    @property
    def uva(self) -> torch.Tensor:
        if not self._fast:
            self._dev.copy_(self.cpu, non_blocking=True)
        return self._dev

    @property
    def device_bytes(self) -> int:
        """Return additional device memory used by this buffer."""
        return 0 if self._fast else self._dev.numel() * self._dev.element_size()
