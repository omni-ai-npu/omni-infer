# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

"""
Patches for vLLM 0.14.0 to add Pangu V2 hybrid attention support.
"""

from .patch_kv_cache_interface import (
    DSAAttentionSpec,
    MomeSpec,
    ShareKVSlidingWindowSpec,
    PanguNewKVCacheSpecsPatch,
    UniformTypeKVCacheSpecsPatch,
)
from .patch_single_type_kv_cache_manager import (
    MomeManager,
    ShareKVSlidingWindowManager,
    SingleTypeKVCacheManagerPatch,
)
from .patch_kv_cache_utils import OverrideGroupSizePatch
from .patch_worker_utils import WorkerUtilsPatch

__all__ = [
    "PanguNewKVCacheSpecsPatch",
    "UniformTypeKVCacheSpecsPatch",
    "SingleTypeKVCacheManagerPatch",
    "OverrideGroupSizePatch",
    "WorkerUtilsPatch",
]
