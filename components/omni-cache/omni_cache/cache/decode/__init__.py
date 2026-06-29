# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Decode module for OmniCache.

This module contains the decode-specific cache implementations and utilities.
"""

from .decode_omni_cache import DecodeOmniCache
from omni_cache.cache.transfer_engine import (
    calc_cache_shape_for_decode,
    calculate_kv_xsfer_params,
)
from omni_cache.cache.transfer_engine import (
    build_npu_block_layers,
    build_npu_block_layers_hybrid,
    build_npu_block_layers_hbm,
)
from .static_utils import (
    record_current_batch_order,
    get_block_table_np,
    update_status_buffered,
    reorder_block_table_only,
    maybe_update_selection_kv_block_status,
)
# Re-export GatherSelectionUpdater from gather_selection for convenience
from omni_cache.gather_selection import GatherSelectionUpdater

__all__ = [
    "DecodeOmniCache",
    "GatherSelectionUpdater",
    "calc_cache_shape_for_decode",
    "calculate_kv_xsfer_params",
    "build_npu_block_layers",
    "build_npu_block_layers_hybrid",
    "build_npu_block_layers_hbm",
    "record_current_batch_order",
    "get_block_table_np",
    "update_status_buffered",
    "reorder_block_table_only",
    "maybe_update_selection_kv_block_status",
]