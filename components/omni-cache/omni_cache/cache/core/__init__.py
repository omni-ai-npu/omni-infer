# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Core cache implementations for OmniCache.

This module contains the base and decode cache implementations.
"""

from .base import (
    BaseOmniCache,
    RequestWindowState,
    _SlotInfo,
    PrefixCopyMeta,
    create_omni_cache,
    bind_kv_cache,
)
from .constants import (
    SIZE_BYTES_PER_LAYER_PREFILL,
    SIZE_BYTES_PER_LAYER_DECODE,
    BLOCK_SIZE,
    HEAD_DIM,
    LOCAL_DP_SIZE,
    NZ_DIM,
    PRE_CALC_PREFILL_BLOCK_NUM,
    ENABLE_C8_INDEXER,
    ENABLE_HOST_MAPPING,
)

__all__ = [
    # Base classes
    "BaseOmniCache",
    "RequestWindowState",
    "_SlotInfo",
    "PrefixCopyMeta",
    # Factory functions
    "create_omni_cache",
    "bind_kv_cache",
    # Constants
    "SIZE_BYTES_PER_LAYER_PREFILL",
    "SIZE_BYTES_PER_LAYER_DECODE",
    "BLOCK_SIZE",
    "HEAD_DIM",
    "LOCAL_DP_SIZE",
    "NZ_DIM",
    "PRE_CALC_PREFILL_BLOCK_NUM",
    "ENABLE_C8_INDEXER",
    "ENABLE_HOST_MAPPING",
]
