# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 buffer injection: binding, and coexistence with patch_uva."""

from __future__ import annotations

import pytest

from omni_npu.vllm_patches.usefull_patch.common import patch_mrv2_buffer_utils as patch_mod
from omni_npu.worker.npu.buffer_utils import NPUUvaBuffer


@pytest.fixture
def applied():
    cls = patch_mod.MRv2UvaBufferPatch
    target = cls._target
    saved = getattr(target, "UvaBuffer")
    owners = dict(getattr(target, "_omni_npu_applied_patches", {}))
    target._omni_npu_applied_patches = {}
    try:
        cls.apply()
        yield target
    finally:
        target.UvaBuffer = saved
        target._omni_npu_applied_patches = owners


def test_uva_buffer_is_replaced(applied):
    assert applied.UvaBuffer is NPUUvaBuffer


def test_pool_builds_the_npu_buffer(applied):
    """UvaBufferPool looks the class up as a module global, so the name is enough."""
    import inspect

    src = inspect.getsource(applied.UvaBufferPool)
    assert "UvaBuffer(" in src, "upstream changed construction; revisit this patch"


def test_coexists_with_patch_uva(applied):
    """patch_uva replaces two functions in the same module; ownership is per attr."""
    owners = applied._omni_npu_applied_patches
    assert owners["UvaBuffer"] == "MRv2UvaBufferPatch"
    assert "is_uva_available" not in owners or owners["is_uva_available"] != (
        "MRv2UvaBufferPatch"
    )
