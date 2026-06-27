# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Core constants for OmniCache.

This module contains fundamental constants used across cache implementations.
Placed in a separate module to avoid circular imports.
"""

import os

from vllm.utils.math_utils import cdiv

SIZE_BYTES_PER_LAYER_PREFILL = int(os.getenv("OMNI_CACHE_LAYER_BYTES", 4 * 1024 * 1024 * 1024))
SIZE_BYTES_PER_LAYER_DECODE = int(os.getenv("OMNI_CACHE_LAYER_BYTES", 4 * 1024 * 1024 * 1024))

BLOCK_SIZE = 128
HEAD_DIM = 512

LOCAL_DP_SIZE = int(os.getenv("OMNI_CACHE_LOCAL_DP_SIZE", "8"))
NZ_DIM = 16

PRE_CALC_PREFILL_BLOCK_NUM = \
    cdiv(SIZE_BYTES_PER_LAYER_PREFILL, 2) // (BLOCK_SIZE * HEAD_DIM)

ENABLE_C8_INDEXER = os.getenv("ENABLE_LI_FUSION", "0") == "1"
ENABLE_HOST_MAPPING = os.getenv("ENABLE_HOST_MAPPING", "1") == "1"
