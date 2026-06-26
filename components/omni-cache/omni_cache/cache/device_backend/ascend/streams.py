# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Stream management for Ascend NPU.

This module provides stream management functionality for Ascend NPU operations,
including stream creation, synchronization, and memory copy operations.
"""

import ctypes
from .memcopy import (
    aclrtStream,
    aclrtCreateStream,
    aclrtSynchronizeStream,
    aclrtDestroyStream,
    aclrtMemcpyBatch,
    aclrtMemcpyBatchAsync,
    aclrtMemcpyAsync,
    aclrtMemcpy,
    _ensure_acl_available,
)


class AscendCLStream:
    """AscendCL stream wrapper for managing NPU operations.

    This class provides a Pythonic interface to AscendCL streams,
    including async memory copies and batch operations.
    """

    def __init__(self):
        self._stream = aclrtStream()

    def create(self):
        """Create the ACL stream."""
        _ensure_acl_available()
        rc = aclrtCreateStream(ctypes.byref(self._stream))
        if rc != 0:
            raise RuntimeError(f"aclrtCreateStream failed with error code: {rc}")
        return rc

    def sync(self):
        """Synchronize the stream, waiting for all operations to complete."""
        _ensure_acl_available()
        rc = aclrtSynchronizeStream(self._stream)
        if rc != 0:
            raise RuntimeError(f"aclrtSynchronizeStream failed with error code: {rc}")
        return rc

    @property
    def stream_ptr(self):
        """Get the underlying stream pointer."""
        return self._stream

    def memcpy_sync(self, dst, dst_max, src, count, kind):
        """Perform an asynchronous memory copy.

        Args:
            dst: Destination pointer
            dst_max: Maximum destination size
            src: Source pointer
            count: Number of bytes to copy
            kind: Memory copy kind (e.g., ACL_MEMCPY_HOST_TO_DEVICE)
        """
        _ensure_acl_available()
        rc = aclrtMemcpy(dst, dst_max, src, count, kind)
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpy failed with error code: {rc}")
        return rc

    def memcpy_async(self, dst, dst_max, src, count, kind):
        """Perform an asynchronous memory copy.

        Args:
            dst: Destination pointer
            dst_max: Maximum destination size
            src: Source pointer
            count: Number of bytes to copy
            kind: Memory copy kind (e.g., ACL_MEMCPY_HOST_TO_DEVICE)
        """
        _ensure_acl_available()
        rc = aclrtMemcpyAsync(dst, dst_max, src, count, kind, self._stream)
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpyAsync failed with error code: {rc}")
        return rc

    def memcpy_batch(self, dst, dst_max, src, count, batch_count, attrs, attrsIndex, kind, fails):
        """Perform a synchronous batch memory copy.

        Args:
            dst: Array of destination pointers
            dst_max: Array of destination sizes
            src: Array of source pointers
            count: Array of byte counts
            batch_count: Number of copies in the batch
            attrs: Batch attributes
            attrsIndex: Attribute indices
            kind: Memory copy kind
            fails: Array to store failure information
        """
        _ensure_acl_available()
        rc = aclrtMemcpyBatch(dst,
                            dst_max,
                            src,
                            count,
                            batch_count,
                            attrs,
                            attrsIndex,
                            kind,
                            fails)
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpyBatch failed with error code: {rc}")
        return rc

    def memcpy_batch_async(self, dst, dst_max, src, count, batch_count, attrs, attrsIndex, kind, fails):
        """Perform an asynchronous batch memory copy.

        Args:
            dst: Array of destination pointers
            dst_max: Array of destination sizes
            src: Array of source pointers
            count: Array of byte counts
            batch_count: Number of copies in the batch
            attrs: Batch attributes
            attrsIndex: Attribute indices
            kind: Memory copy kind
            fails: Array to store failure information
        """
        _ensure_acl_available()
        if aclrtMemcpyBatchAsync is None:
            raise RuntimeError("aclrtMemcpyBatchAsync is not available in this AscendCL version")
        rc = aclrtMemcpyBatchAsync(dst,
                                dst_max,
                                src,
                                count,
                                batch_count,
                                attrs,
                                attrsIndex,
                                kind, fails,
                                self._stream)
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpyBatch failed with error code: {rc}")
        return rc

    def __del__(self):
        """Destructor - destroy the stream when the object is garbage collected."""
        # Note: We can't call _ensure_acl_available in __del__ as it may raise
        # Just try to destroy and ignore errors
        try:
            aclrtDestroyStream(self._stream)
        except Exception:
            pass


__all__ = ['AscendCLStream']
