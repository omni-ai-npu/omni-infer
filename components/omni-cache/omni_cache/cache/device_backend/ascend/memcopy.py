# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""ACL memory copy bindings for Ascend NPU.

This module provides low-level ctypes bindings for AscendCL memory operations:
- aclrtMemcpyBatch: Synchronous batch memory copy
- aclrtMemcpyBatchAsync: Asynchronous batch memory copy
- aclrtMemcpyAsync: Asynchronous single memory copy
- Memory location types and memcpy kinds
"""

import ctypes
import logging

# Try to load the AscendCL library, but don't fail if it's not available
# (allows importing on non-NPU systems for type checking/development)
try:
    libascendcl = ctypes.CDLL('libascendcl.so')
    _ACL_AVAILABLE = True
except OSError:
    libascendcl = None
    _ACL_AVAILABLE = False
    logging.debug("libascendcl.so not found, Ascend NPU operations will be unavailable")

ACL_MEM_LOCATION_TYPE_HOST = 0
ACL_MEM_LOCATION_TYPE_DEVICE = 1


class aclrtMemLocation(ctypes.Structure):
    _fields_ = [
        ('id', ctypes.c_uint32),
        ('type', ctypes.c_uint32)
    ]


class aclrtMemcpyBatchAttr(ctypes.Structure):
    _fields_ = [
        ('dstLoc', aclrtMemLocation),
        ('srcLoc', aclrtMemLocation),
        ('rsv', ctypes.c_uint8 * 16)
    ]


if _ACL_AVAILABLE:
    aclrtMemcpyBatch = libascendcl.aclrtMemcpyBatch

    try:
        aclrtMemcpyBatchAsync = libascendcl.aclrtMemcpyBatchAsync
    except AttributeError:
        logging.warning("aclrtMemcpyBatchAsync not found in libascendcl.so, please check your AscendCL version.")
        aclrtMemcpyBatchAsync = None

    aclrtMemcpyBatch.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(aclrtMemcpyBatchAttr),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    aclrtMemcpyBatch.restype = ctypes.c_int
else:
    aclrtMemcpyBatch = None
    aclrtMemcpyBatchAsync = None

aclrtStream = ctypes.c_void_p

ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
ACL_MEMCPY_DEFAULT = 4
ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE = 5
ACL_MEMCPY_INNER_DEVICE_TO_DEVICE = 6
ACL_MEMCPY_INTER_DEVICE_TO_DEVICE = 7

if _ACL_AVAILABLE:
    aclrtCreateStream = libascendcl.aclrtCreateStream
    aclrtCreateStream.argtypes = [ctypes.POINTER(aclrtStream)]
    aclrtCreateStream.restype = ctypes.c_int

    aclrtSynchronizeStream = libascendcl.aclrtSynchronizeStream
    aclrtSynchronizeStream.argtypes = [aclrtStream]
    aclrtSynchronizeStream.restype = ctypes.c_int

    aclrtDestroyStream = libascendcl.aclrtDestroyStream
    aclrtDestroyStream.argtypes = [aclrtStream]
    aclrtDestroyStream.restype = ctypes.c_int

    aclrtMemcpy = libascendcl.aclrtMemcpy
    aclrtMemcpy.argtypes = [
        ctypes.c_void_p,           # void *dst
        ctypes.c_size_t,           # size_t destMax
        ctypes.c_void_p,           # const void *src
        ctypes.c_size_t,           # size_t count
        ctypes.c_int,              # aclrtMemcpyKind kind
    ]
    aclrtMemcpy.restype = ctypes.c_int

    aclrtMemcpyAsync = libascendcl.aclrtMemcpyAsync
    aclrtMemcpyAsync.argtypes = [
        ctypes.c_void_p,           # void *dst
        ctypes.c_size_t,           # size_t destMax
        ctypes.c_void_p,           # const void *src
        ctypes.c_size_t,           # size_t count
        ctypes.c_int,              # aclrtMemcpyKind kind
        aclrtStream                # aclrtStream stream
    ]
    aclrtMemcpyAsync.restype = ctypes.c_int
else:
    aclrtCreateStream = None
    aclrtSynchronizeStream = None
    aclrtDestroyStream = None
    aclrtMemcpyAsync = None


def _ensure_acl_available():
    """Raise an error if ACL is not available."""
    if not _ACL_AVAILABLE:
        raise RuntimeError(
            "AscendCL library (libascendcl.so) is not available. "
            "Please install the CANN toolkit and ensure the library is in your library path."
        )


__all__ = [
    # Types
    'aclrtMemLocation',
    'aclrtMemcpyBatchAttr',
    'aclrtStream',
    # Constants
    'ACL_MEM_LOCATION_TYPE_HOST',
    'ACL_MEM_LOCATION_TYPE_DEVICE',
    'ACL_MEMCPY_HOST_TO_HOST',
    'ACL_MEMCPY_HOST_TO_DEVICE',
    'ACL_MEMCPY_DEVICE_TO_HOST',
    'ACL_MEMCPY_DEVICE_TO_DEVICE',
    'ACL_MEMCPY_DEFAULT',
    'ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE',
    'ACL_MEMCPY_INNER_DEVICE_TO_DEVICE',
    'ACL_MEMCPY_INTER_DEVICE_TO_DEVICE',
    # Functions
    'aclrtMemcpyBatch',
    'aclrtMemcpyBatchAsync',
    'aclrtCreateStream',
    'aclrtSynchronizeStream',
    'aclrtDestroyStream',
    'aclrtMemcpyAsync',
    # Helpers
    '_ACL_AVAILABLE',
    '_ensure_acl_available',
]
