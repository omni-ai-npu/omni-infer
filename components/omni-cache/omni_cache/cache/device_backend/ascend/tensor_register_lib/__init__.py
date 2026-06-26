# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tensor registration library for Ascend NPU.

This package provides C++ extensions for tensor registration and zero-copy
memory operations on Ascend NPU devices.
"""

# The zero_copy_npu module is built as a C++ extension
try:
    from . import zero_copy_npu
    __all__ = ['zero_copy_npu']
except ImportError:
    # Extension not built yet
    __all__ = []
