# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Triton operations for Ascend NPU.

This module provides Triton-based operations for metadata processing
and other compute tasks on Ascend NPU.
"""

from .triton_ops import (
    update_block_table_kernel_swa,
    move_hbm_slots_kernel_swa,
    build_fake_block_table_kernel_compress,
    move_slots_kernel,
    move_slots_batched_layers_kernel,
)

__all__ = [
    'update_block_table_kernel_swa',
    'move_hbm_slots_kernel_swa',
    'build_fake_block_table_kernel_compress',
    'move_slots_kernel',
    'move_slots_batched_layers_kernel',
]
