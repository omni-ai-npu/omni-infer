# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend NPU backend support for OmniCache.

This module provides Ascend NPU specific implementations including:
- ACL (Ascend Computing Language) bindings for memory operations
- Tensor registration for zero-copy host-device transfers
- Triton operations for metadata processing
- Stream management
"""

from .memcopy import (
    aclrtMemLocation,
    aclrtMemcpyBatchAttr,
    aclrtStream,
    ACL_MEM_LOCATION_TYPE_HOST,
    ACL_MEM_LOCATION_TYPE_DEVICE,
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
    ACL_MEMCPY_DEFAULT,
    ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE,
    ACL_MEMCPY_INNER_DEVICE_TO_DEVICE,
    ACL_MEMCPY_INTER_DEVICE_TO_DEVICE,
    aclrtMemcpyBatch,
    aclrtMemcpyBatchAsync,
    aclrtCreateStream,
    aclrtSynchronizeStream,
    aclrtDestroyStream,
    aclrtMemcpyAsync,
)
from .tensor_register import NPUTensorRegister
from .streams import AscendCLStream
from . import ops
from . import tensor_register_lib

__all__ = [
    # Classes
    "NPUTensorRegister",
    "AscendCLStream",
    # Types
    "aclrtMemLocation",
    "aclrtMemcpyBatchAttr",
    "aclrtStream",
    # Constants
    "ACL_MEM_LOCATION_TYPE_HOST",
    "ACL_MEM_LOCATION_TYPE_DEVICE",
    "ACL_MEMCPY_HOST_TO_HOST",
    "ACL_MEMCPY_HOST_TO_DEVICE",
    "ACL_MEMCPY_DEVICE_TO_HOST",
    "ACL_MEMCPY_DEVICE_TO_DEVICE",
    "ACL_MEMCPY_DEFAULT",
    "ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE",
    "ACL_MEMCPY_INNER_DEVICE_TO_DEVICE",
    "ACL_MEMCPY_INTER_DEVICE_TO_DEVICE",
    # Functions
    "aclrtMemcpyBatch",
    "aclrtMemcpyBatchAsync",
    "aclrtCreateStream",
    "aclrtSynchronizeStream",
    "aclrtDestroyStream",
    "aclrtMemcpyAsync",
    # Submodules
    "ops",
    "tensor_register_lib",
]
