# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Runtime mode selection for V2 NPU buffers."""

from __future__ import annotations

import enum
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# When set, NPUUvaBuffer aliases pinned host memory through a real NPU view
# instead of staging an H2D copy into a separate device tensor.
UVA_ENV = "OMNI_NPU_V2_UVA"


class RuntimeMode(enum.Enum):
    TORCH_COPY_TO_NPU = "torch_copy_to_npu"
    NPU_UVA_VIEW = "npu_uva_view"


_MODE: RuntimeMode | None = None
_UVA_BLOCKERS: tuple[str, ...] = ()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _uva_blockers() -> list[str]:
    if not _env_flag(UVA_ENV):
        return [f"{UVA_ENV} not set"]
    # Ask the platform rather than patch_uva's wrapper: both patch_uva and the
    # MRv2 patches are applied by the same plugin pass with no order between
    # them, and this mode is resolved from NPUUvaBuffer's constructor.
    from vllm.platforms import current_platform

    is_available = getattr(current_platform, "is_uva_available", None)
    if not callable(is_available):
        return [f"{type(current_platform).__name__} has no is_uva_available()"]
    if not is_available():
        return [
            "current_platform.is_uva_available() is False (PYTORCH_NPU_ALLOC_CONF "
            "needs pinned_mem_register:True and must not set "
            "pin_memory_expandable_segments:True; omni_npu.allocator.npu_uva must "
            "be importable)"
        ]
    return []


def resolve_mode() -> RuntimeMode:
    """Resolve and cache the process-wide buffer mode."""
    global _MODE, _UVA_BLOCKERS
    if _MODE is not None:
        return _MODE
    blockers = _uva_blockers()
    _UVA_BLOCKERS = tuple(blockers)
    _MODE = RuntimeMode.TORCH_COPY_TO_NPU if blockers else RuntimeMode.NPU_UVA_VIEW
    logger.info(
        "[omni-npu/mrv2] runtime mode=%s%s",
        _MODE.value,
        "" if not blockers else "; UVA view disabled: " + "; ".join(blockers),
    )
    return _MODE


def uva_blockers() -> tuple[str, ...]:
    """Return the reasons the UVA view is unavailable."""
    resolve_mode()
    return _UVA_BLOCKERS


def reset_for_testing() -> None:
    """Clear cached mode state."""
    global _MODE, _UVA_BLOCKERS
    _MODE = None
    _UVA_BLOCKERS = ()
