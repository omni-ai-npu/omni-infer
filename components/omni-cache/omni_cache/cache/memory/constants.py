# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Constants for KV cache memory pool."""

import os

HUGE_PAGE_SIZE = 2 * 1024 * 1024  # 2MB
REGIST_SIZE = 4096
MAP_HUGETLB = 0x40000

ENABLE_C8_INDEXER = os.getenv("ENABLE_LI_FUSION", "0") == "1"
ENABLE_HOST_MAPPING = os.getenv("ENABLE_HOST_MAPPING", "0") == "1"

# ---------------------------------------------------------------------------
# Shape-rank constants — replace magic-number comparisons like
# `len(shape) == 4` with `len(shape) == SHAPE_RANK_PREFILL` etc.
# ---------------------------------------------------------------------------
SHAPE_RANK_PREFILL = 4   # (num_layers,  num_blocks, block_size, head_dim)
SHAPE_RANK_DECODE  = 5   # (num_ranks,   num_layers, num_blocks, block_size, head_dim)

# Pangu V2 DSA head dimensions
INDEXER_HEAD_DIM = 128        # indexer dimension for DSA attention
