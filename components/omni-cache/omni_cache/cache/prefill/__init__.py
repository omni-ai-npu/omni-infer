# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Prefill module for OmniCache.

This module contains the prefill-specific cache implementations and utilities.
"""

from omni_cache.cache.transfer_engine import (
    D2HContext,
    calc_cache_shape_for_prefill,
    copy_kv_to_buffers,
    prepare_d2h_metadata,
    submit_async_updates,
)
from omni_cache.cache.prefill.prefix_copy_ops import (
    PrefixCopyMeta,
    compute_prefix_segments,
    get_current_rank_host_data,
)
from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache
from omni_cache.cache.prefill.tensor_utils import (
    nd_to_nz,
    padding_kv_cache,
)

__all__ = [
    "PrefillOmniCache",
    "calc_cache_shape_for_prefill",
    "D2HContext",
    "prepare_d2h_metadata",
    "copy_kv_to_buffers",
    "submit_async_updates",
    "PrefixCopyMeta",
    "compute_prefix_segments",
    "get_current_rank_host_data",
    "padding_kv_cache",
    "nd_to_nz",
]
