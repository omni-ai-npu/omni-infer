# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 DP injection: bindings. Behaviour lives in test_eager_dp_padding.py."""

from __future__ import annotations

import pytest

from omni_npu.vllm_patches.usefull_patch import patch_mrv2_dp_utils as patch_mod
from omni_npu.worker.npu import dp_utils as npu_dp_utils
from unit.vllm_patch.useful_patch.patch_test_utils import applied_patches

_NAMES = (
    "MRv2DpUtilsPatch",
    "MRv2ModelRunnerDispatchPatch",
    "MRv2AutoregressiveSpeculatorDispatchPatch",
    "MRv2DflashSpeculatorDispatchPatch",
)


@pytest.fixture
def applied():
    classes = [getattr(patch_mod, n) for n in _NAMES if hasattr(patch_mod, n)]
    with applied_patches(classes) as applied_classes:
        yield applied_classes


def test_dispatch_binding_covers_the_speculators(applied):
    """The pad target must be published every step, and speculators dispatch too."""
    import vllm.v1.worker.gpu.spec_decode.autoregressive.speculator as up_ar
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as up_dflash

    assert up_ar.dispatch_cg_and_sync_dp is npu_dp_utils.dispatch_cg_and_sync_dp
    assert up_dflash.dispatch_cg_and_sync_dp is npu_dp_utils.dispatch_cg_and_sync_dp


def test_mrv1_dp_utils_module_is_untouched(applied):
    """MRv1 uses vllm/v1/worker/dp_utils.py, not the V2 gpu/dp_utils.py."""
    import vllm.v1.worker.dp_utils as v1_dp_utils

    assert (
        getattr(v1_dp_utils, "dispatch_cg_and_sync_dp", None)
        is not npu_dp_utils.dispatch_cg_and_sync_dp
    )
