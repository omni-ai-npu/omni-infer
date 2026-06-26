# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_ascend_acl.py

import pytest
import ctypes
from unittest.mock import Mock, patch, MagicMock, PropertyMock

from omni_cache.cache.device_backend.ascend import (
    aclrtMemLocation,
    aclrtMemcpyBatchAttr,
    AscendCLStream,
    ACL_MEM_LOCATION_TYPE_HOST,
    ACL_MEM_LOCATION_TYPE_DEVICE,
    ACL_MEMCPY_HOST_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_DEVICE_TO_DEVICE,
    ACL_MEMCPY_DEFAULT,
)

# ==================== ctypes 结构体测试 ====================

class TestAclrtMemLocation:
    """测试 aclrtMemLocation ctypes 结构体"""

    def test_structure_fields(self):
        assert hasattr(aclrtMemLocation, 'id')
        assert hasattr(aclrtMemLocation, 'type')

    def test_field_types(self):
        loc = aclrtMemLocation()
        loc.id = 1
        loc.type = 2
        assert loc.id == 1
        assert loc.type == 2

    def test_default_values(self):
        loc = aclrtMemLocation()
        assert loc.id == 0
        assert loc.type == 0

    def test_size_calculation(self):
        # id: c_uint32 (4 bytes), type: c_uint32 (4 bytes) = 8 bytes total
        expected_size = ctypes.sizeof(ctypes.c_uint32) * 2
        assert ctypes.sizeof(aclrtMemLocation) == expected_size


class TestAclrtMemcpyBatchAttr:
    """测试 aclrtMemcpyBatchAttr ctypes 结构体"""

    def test_structure_fields(self):
        assert hasattr(aclrtMemcpyBatchAttr, 'dstLoc')
        assert hasattr(aclrtMemcpyBatchAttr, 'srcLoc')
        assert hasattr(aclrtMemcpyBatchAttr, 'rsv')

    def test_nested_structure(self):
        attr = aclrtMemcpyBatchAttr()
        # dstLoc 和 srcLoc 应该是 aclrtMemLocation 类型
        assert isinstance(attr.dstLoc, aclrtMemLocation)
        assert isinstance(attr.srcLoc, aclrtMemLocation)

    def test_field_assignment(self):
        attr = aclrtMemcpyBatchAttr()
        attr.dstLoc.id = 1
        attr.dstLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE
        attr.srcLoc.id = 0
        attr.srcLoc.type = ACL_MEM_LOCATION_TYPE_HOST

        assert attr.dstLoc.id == 1
        assert attr.dstLoc.type == ACL_MEM_LOCATION_TYPE_DEVICE
        assert attr.srcLoc.id == 0
        assert attr.srcLoc.type == ACL_MEM_LOCATION_TYPE_HOST


# ==================== 常量测试 ====================

class TestACLConstants:
    """测试 ACL 相关常量"""

    def test_memory_location_constants(self):
        """测试内存位置常量"""
        assert ACL_MEM_LOCATION_TYPE_HOST == 0
        assert ACL_MEM_LOCATION_TYPE_DEVICE == 1

    def test_memcpy_kind_constants(self):
        """测试内存拷贝类型常量"""
        assert ACL_MEMCPY_HOST_TO_HOST == 0
        assert ACL_MEMCPY_HOST_TO_DEVICE == 1
        assert ACL_MEMCPY_DEVICE_TO_HOST == 2
        assert ACL_MEMCPY_DEVICE_TO_DEVICE == 3
        assert ACL_MEMCPY_DEFAULT == 4


# ==================== AscendCLStream 类测试 ====================

class TestAscendCLStreamInit:
    """测试 AscendCLStream 初始化"""

    def test_init_creates_stream_object(self):
        """测试初始化创建流对象"""
        stream = AscendCLStream()
        # _stream 应该是 aclrtStream 类型 (c_void_p)
        assert hasattr(stream, '_stream')
        assert isinstance(stream._stream, ctypes.c_void_p)

    def test_stream_ptr_property(self):
        """测试 stream_ptr 属性"""
        stream = AscendCLStream()
        ptr = stream.stream_ptr
        assert ptr is stream._stream


class TestAscendCLStreamCreate:
    """测试 AscendCLStream.create 方法"""

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtCreateStream')
    def test_create_success(self, mock_create):
        """测试成功创建流"""
        mock_create.return_value = 0  # ACL_SUCCESS

        stream = AscendCLStream()
        result = stream.create()

        assert result == 0
        mock_create.assert_called_once()

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtCreateStream')
    def test_create_failure_raises_runtime_error(self, mock_create):
        """测试创建失败抛出 RuntimeError"""
        mock_create.return_value = 1  # ACL_ERROR

        stream = AscendCLStream()
        with pytest.raises(RuntimeError, match="aclrtCreateStream failed"):
            stream.create()

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtCreateStream')
    def test_create_various_error_codes(self, mock_create):
        """测试各种错误码"""
        for error_code in [1, 2, 100, 50000]:
            mock_create.return_value = error_code
            stream = AscendCLStream()
            with pytest.raises(RuntimeError, match=f"error code: {error_code}"):
                stream.create()


class TestAscendCLStreamSync:
    """测试 AscendCLStream.sync 方法"""

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtSynchronizeStream')
    def test_sync_success(self, mock_sync):
        """测试成功同步流"""
        mock_sync.return_value = 0

        stream = AscendCLStream()
        result = stream.sync()

        assert result == 0
        mock_sync.assert_called_once_with(stream._stream)

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtSynchronizeStream')
    def test_sync_failure_raises_runtime_error(self, mock_sync):
        """测试同步失败抛出 RuntimeError"""
        mock_sync.return_value = 1

        stream = AscendCLStream()
        with pytest.raises(RuntimeError, match="aclrtSynchronizeStream failed"):
            stream.sync()


class TestAscendCLStreamMemcpyAsync:
    """测试 AscendCLStream.memcpy_async 方法"""

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtMemcpyAsync')
    def test_memcpy_async_success(self, mock_memcpy):
        """测试成功异歩拷贝"""
        mock_memcpy.return_value = 0

        stream = AscendCLStream()
        dst = ctypes.c_void_p(0x1000)
        src = ctypes.c_void_p(0x2000)

        result = stream.memcpy_async(
            dst=dst,
            dst_max=1024,
            src=src,
            count=1024,
            kind=ACL_MEMCPY_HOST_TO_DEVICE
        )

        assert result == 0
        mock_memcpy.assert_called_once()

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtMemcpyAsync')
    def test_memcpy_async_failure(self, mock_memcpy):
        """测试异歩拷贝失败"""
        mock_memcpy.return_value = 1

        stream = AscendCLStream()
        with pytest.raises(RuntimeError, match="aclrtMemcpyAsync failed"):
            stream.memcpy_async(
                dst=ctypes.c_void_p(0x1000),
                dst_max=1024,
                src=ctypes.c_void_p(0x2000),
                count=1024,
                kind=ACL_MEMCPY_HOST_TO_DEVICE
            )


class TestAscendCLStreamMemcpyBatch:
    """测试 AscendCLStream.memcpy_batch 方法"""

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtMemcpyBatch')
    def test_memcpy_batch_success(self, mock_memcpy):
        """测试成功批量拷贝"""
        mock_memcpy.return_value = 0

        stream = AscendCLStream()
        dst = (ctypes.c_void_p * 2)(ctypes.c_void_p(0x1000), ctypes.c_void_p(0x2000))
        dst_max = (ctypes.c_size_t * 2)(1024, 1024)
        src = (ctypes.c_void_p * 2)(ctypes.c_void_p(0x3000), ctypes.c_void_p(0x4000))
        count = (ctypes.c_size_t * 2)(1024, 1024)
        attrs = (aclrtMemcpyBatchAttr * 2)()
        attrsIndex = (ctypes.c_size_t * 2)(0, 1)
        fails = (ctypes.c_size_t * 2)(0, 0)

        result = stream.memcpy_batch(
            dst=dst,
            dst_max=dst_max,
            src=src,
            count=count,
            batch_count=2,
            attrs=attrs,
            attrsIndex=attrsIndex,
            kind=ACL_MEMCPY_HOST_TO_DEVICE,
            fails=fails
        )

        assert result == 0

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtMemcpyBatch')
    def test_memcpy_batch_failure(self, mock_memcpy):
        """测试批量拷贝失败"""
        mock_memcpy.return_value = 1

        stream = AscendCLStream()
        with pytest.raises(RuntimeError, match="aclrtMemcpyBatch failed"):
            stream.memcpy_batch(
                dst=None,
                dst_max=None,
                src=None,
                count=None,
                batch_count=0,
                attrs=None,
                attrsIndex=None,
                kind=0,
                fails=None
            )


class TestAscendCLStreamMemcpyBatchAsync:
    """测试 AscendCLStream.memcpy_batch_async 方法"""

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtMemcpyBatchAsync')
    def test_memcpy_batch_async_success(self, mock_memcpy):
        """测试成功异歩批量拷贝"""
        mock_memcpy.return_value = 0

        stream = AscendCLStream()
        result = stream.memcpy_batch_async(
            dst=None,
            dst_max=None,
            src=None,
            count=None,
            batch_count=0,
            attrs=None,
            attrsIndex=None,
            kind=0,
            fails=None
        )

        assert result == 0

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtMemcpyBatchAsync')
    def test_memcpy_batch_async_failure(self, mock_memcpy):
        """测试异歩批量拷贝失败"""
        mock_memcpy.return_value = 1

        stream = AscendCLStream()
        with pytest.raises(RuntimeError, match="aclrtMemcpyBatch failed"):
            stream.memcpy_batch_async(
                dst=None,
                dst_max=None,
                src=None,
                count=None,
                batch_count=0,
                attrs=None,
                attrsIndex=None,
                kind=0,
                fails=None
            )


class TestAscendCLStreamDestructor:
    """测试 AscendCLStream 析构函数"""

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtDestroyStream')
    def test_destructor_calls_destroy(self, mock_destroy):
        """测试析构时调用 aclrtDestroyStream"""
        mock_destroy.return_value = 0

        stream = AscendCLStream()
        stream_ptr = stream._stream

        # 显式调用 __del__
        stream.__del__()

        mock_destroy.assert_called_once_with(stream_ptr)


# ==================== 集成测试 ====================

class TestAscendCLStreamIntegration:
    """AscendCLStream 集成测试"""

    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtCreateStream')
    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtSynchronizeStream')
    @patch('omni_cache.cache.device_backend.ascend.streams.aclrtDestroyStream')
    def test_full_lifecycle(self, mock_destroy, mock_sync, mock_create):
        """测试完整生命周期: 创建 -> 使用 -> 销毁"""
        mock_create.return_value = 0
        mock_sync.return_value = 0
        mock_destroy.return_value = 0

        stream = AscendCLStream()
        stream.create()
        stream.sync()
        stream.__del__()

        mock_create.assert_called_once()
        mock_sync.assert_called_once()
        mock_destroy.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])