# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Device backend support for OmniCache.

This module provides device-specific implementations for different hardware devices.
Currently supports Ascend NPU.
"""

import warnings
from omni_cache.cache.device_backend.ascend import (
    ACL_MEMCPY_DEFAULT,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_INNER_DEVICE_TO_DEVICE,
    ACL_MEMCPY_INTER_DEVICE_TO_DEVICE,
    ACL_MEM_LOCATION_TYPE_DEVICE,
    ACL_MEM_LOCATION_TYPE_HOST,
    AscendCLStream,
    NPUTensorRegister,
    aclrtCreateStream,
    aclrtDestroyStream,
    aclrtMemLocation,
    aclrtMemcpyAsync,
    aclrtMemcpyBatch,
    aclrtMemcpyBatchAsync,
    aclrtMemcpyBatchAttr,
    aclrtStream,
    aclrtSynchronizeStream,
    ops,
    tensor_register_lib,
)

warnings.warn(
    "omni_cache.cache.backend is deprecated. "
    "Use omni_cache.cache.device_backend instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "ACL_MEMCPY_DEFAULT",
    "ACL_MEMCPY_DEVICE_TO_DEVICE",
    "ACL_MEMCPY_DEVICE_TO_HOST",
    "ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE",
    "ACL_MEMCPY_HOST_TO_DEVICE",
    "ACL_MEMCPY_HOST_TO_HOST",
    "ACL_MEMCPY_INNER_DEVICE_TO_DEVICE",
    "ACL_MEMCPY_INTER_DEVICE_TO_DEVICE",
    "ACL_MEM_LOCATION_TYPE_DEVICE",
    "ACL_MEM_LOCATION_TYPE_HOST",
    "AscendCLStream",
    "NPUTensorRegister",
    "aclrtCreateStream",
    "aclrtDestroyStream",
    "aclrtMemLocation",
    "aclrtMemcpyAsync",
    "aclrtMemcpyBatch",
    "aclrtMemcpyBatchAsync",
    "aclrtMemcpyBatchAttr",
    "aclrtStream",
    "aclrtSynchronizeStream",
    "ops",
    "tensor_register_lib",
]