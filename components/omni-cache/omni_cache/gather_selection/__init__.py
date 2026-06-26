# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Gather Selection module for OmniCache.

This module provides gather selection functionality for the decode cache,
including buffer management, status updates, and core gather operations.

The gather selection mechanism is used in compressed attention to dynamically
select relevant KV cache blocks based on top-k indices, reducing memory
bandwidth and computation for long sequence decoding.

Example usage:
    from omni_cache.gather_selection import (
        gather_selection,
        GatherSelectionUpdater,
        initialize_selection_buffers,
    )

    # Initialize buffers
    initialize_selection_buffers(omni_cache)

    # Perform gather selection
    gather_selection(omni_cache, attn_kwargs, layer_idx=0)

    # Update batch order
    GatherSelectionUpdater.record_current_batch_order(input_batch, omni_cache)
"""

from omni_cache.gather_selection.core.gather_selection import gather_selection
from omni_cache.gather_selection.core.buffers import SelectionBuffers, initialize_selection_buffers
from omni_cache.gather_selection.status_updater.updater import GatherSelectionUpdater
from omni_cache.attention.metadata.block_table import get_block_table_np

__all__ = [
    # Core operations
    "gather_selection",
    # Buffer management
    "SelectionBuffers",
    "initialize_selection_buffers",
    # Status Updater
    "GatherSelectionUpdater",
    # Utilities
    "get_block_table_np",
]