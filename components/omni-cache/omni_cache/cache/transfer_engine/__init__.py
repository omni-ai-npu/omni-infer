# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""KV Swap operations for OmniCache (H2D: DRAM→HBM, D2H: HBM→DRAM).

This module provides unified H2D (Host-to-Device) and D2H (Device-to-Host)
transfer operations for both decode and prefill cache operations.
"""

# Core manager (NEW)
from .manager import TransferManager

# Buffer management (NEW)
from .buffers import TransferBuffers, StreamManager, ThreadPoolManager

# Synchronization operations (NEW)
from .synchronize import (
    synchronize_h2d_prefill,
    synchronize_d2h_prefill,
    synchronize_d2h_hybrid,
    synchronize_h2d_decode,
    synchronize_d2h_decode,
)

# Decode H2D operations
from .decode import (
    build_npu_block_layers,
    build_npu_block_layers_hybrid,
    build_npu_block_layers_hbm,
    # NEW:
    prepare_h2d_copy_args,
    prepare_h2d_copy_args_hybrid,
    prepare_h2d_copy_args_hbm_buffer,
)

# Prefill D2H operations
from .prefill import (
    D2HContext,
    prepare_d2h_metadata,
    copy_kv_to_buffers,
    submit_async_updates,
    # NEW:
    update_host_cache_thread,
    update_host_cache_thread_compress,
)

# Cache shape calculations
from .shapes import (
    calc_cache_shape_for_decode,
    calc_cache_shape_for_prefill,
    calculate_kv_xsfer_params,
)

__all__ = [
    # Core manager
    "TransferManager",
    # Buffer management
    "TransferBuffers",
    "StreamManager",
    "ThreadPoolManager",
    # Synchronization
    "synchronize_h2d_prefill",
    "synchronize_d2h_prefill",
    "synchronize_d2h_hybrid",
    "synchronize_h2d_decode",
    "synchronize_d2h_decode",
    # Decode H2D operations
    "build_npu_block_layers",
    "build_npu_block_layers_hybrid",
    "build_npu_block_layers_hbm",
    "prepare_h2d_copy_args",
    "prepare_h2d_copy_args_hybrid",
    "prepare_h2d_copy_args_hbm_buffer",
    # Prefill D2H operations
    "D2HContext",
    "prepare_d2h_metadata",
    "copy_kv_to_buffers",
    "submit_async_updates",
    "update_host_cache_thread",
    "update_host_cache_thread_compress",
    # Shape calculations
    "calc_cache_shape_for_decode",
    "calc_cache_shape_for_prefill",
    "calculate_kv_xsfer_params",
]
