# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_kv_mem_pool.py

import pytest
try:
    import torch
except ImportError:
    torch = None
import mmap
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Import module under test
from omni_cache.cache.memory import KVCacheMemoryPool

if torch is None:
    pytest.skip("torch is not available", allow_module_level=True)


# ==================== Mock configurations ====================

class MockTPGroup:
    """Mock tensor parallel group."""
    def __init__(self, world_size=16, rank=0):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = rank


# Note: MockModelExtraConfig is no longer needed after refactoring.
# enable_dsa is now passed directly as a parameter to KVCacheMemoryPool.


class MockAscendCLStream:
    """Mock Ascend CL stream for async operations."""
    def __init__(self):
        self.sync_called = False
        self.memcpy_calls = []

    def memcpy_async(self, device_mem, device_max, host_mem, host_sizes, direction):
        self.memcpy_calls.append({
            'device_mem': device_mem,
            'device_max': device_max,
            'host_mem': host_mem,
            'host_sizes': host_sizes,
        })
        return 0

    def sync(self):
        self.sync_called = True


class MockNPUTensorRegister:
    """Mock NPU tensor register."""
    def host_tensor_register(self, tensor, device_id=0):
        return tensor, tensor


class MockDevice:
    """Mock torch device."""
    def __init__(self, index=0):
        self.index = index


# ==================== Fixtures ====================

@pytest.fixture
def mock_hugepage_file(tmp_path):
    """Create a temporary hugepage file for testing."""
    hugepage_file = tmp_path / "hugepage_test"
    size = 10 * 1024 * 1024  # 10MB
    hugepage_file.write_bytes(b'\x00' * size)
    return str(hugepage_file)


@pytest.fixture
def prefill_shape():
    """Return 4D shape for prefill mode."""
    return (2, 4, 16, 640)  # (num_layers, num_blocks, block_size, head_dim)


@pytest.fixture
def decode_shape():
    """Return 5D shape for decode mode."""
    return (2, 2, 4, 16, 640)  # (die_num, num_layers, num_blocks, block_size, head_dim)


# ==================== KVCacheMemoryPool Init tests ====================

class TestKVCacheMemoryPoolInit:
    """Test suite for KVCacheMemoryPool initialization."""

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_init_prefill_shape(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test initialization with 4D prefill shape."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        assert pool.num_layers == 2
        assert pool.num_blocks == 4
        assert pool.block_size == 16
        assert pool.num_ranks == 1
        assert pool.is_prefill == True
        assert pool.dtype == torch.bfloat16
        assert pool.element_size == 2
        assert len(pool.shapes) == 2
        assert pool.tp_world_size == 16
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_init_decode_shape(self, mock_get_tp, mock_hugepage_file, decode_shape):
        """Test initialization with 5D decode shape."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=decode_shape,
            rank=0
        )

        assert pool.num_layers == 2
        assert pool.num_blocks == 4
        assert pool.block_size == 16
        assert pool.num_ranks == 2
        assert pool.is_prefill == False

        del pool.shared_tensor_npu
        del pool.kvi_tensors
        del pool.kvi_tensors_swap
        del pool.shared_tensor
        del pool.shared_tensor_swap
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_init_invalid_shape_3d(self, mock_get_tp, mock_hugepage_file):
        """Test initialization with invalid 3D shape raises ValueError."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        with pytest.raises(ValueError, match="Unsupported shape for ram cache"):
            KVCacheMemoryPool(
                hugepage_path=mock_hugepage_file,
                mmap_size=10 * 1024 * 1024,
                shape=(2, 4, 16),
                rank=0
            )

    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister")
    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    def test_init_with_dsa_enabled(self, mock_get_tp, mock_npu_reg, mock_hugepage_file, prefill_shape):
        """Test initialization with DSA enabled."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)
        mock_npu_reg.return_value = MockNPUTensorRegister()

        dsa_shape = (2, 4, 16, 704)  # 704 = 512 + 64 + 128 (matches DSA head_size_ratio before TP division)

        # Pass enable_dsa=True directly as parameter (after refactoring)
        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=dsa_shape,
            rank=0,
            enable_dsa=True
        )

        # With DSA enabled, shapes should have 3 components
        assert len(pool.shapes) == 3
        # head_size_ratio is [512, 64, 128] / 16 = [32, 4, 8]
        assert pool.head_size_ratio == [32, 4, 8]
        if hasattr(pool, 'shared_tensor_npu'):
            del pool.shared_tensor_npu
        if hasattr(pool, 'kvi_tensors_swap'):
            del pool.kvi_tensors_swap
        if hasattr(pool, 'kvi_tensors'):
            del pool.kvi_tensors
        if hasattr(pool, 'shared_tensor'):
            del pool.shared_tensor
        if hasattr(pool, 'shared_tensor_swap'):
            del pool.shared_tensor_swap
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_init_file_not_found(self, mock_get_tp, prefill_shape):
        """Test initialization with non-existent hugepage file raises FileNotFoundError."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        with pytest.raises(FileNotFoundError, match="Hugepage file not found"):
            KVCacheMemoryPool(
                hugepage_path="/nonexistent/path/hugepage",
                mmap_size=10 * 1024 * 1024,
                shape=prefill_shape,
                rank=0
            )


# ==================== get_block tests ====================

class TestKVCacheMemoryPoolGetBlock:
    """Test suite for get_block method."""

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_get_block_valid_index(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test getting a block with valid index."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        block_tensors = pool.get_block(0)
        assert len(block_tensors) == 2
        assert block_tensors[0].shape[0] == pool.num_layers
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_get_block_invalid_index(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test getting a block with invalid index raises IndexError."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        with pytest.raises(IndexError, match="Block index.*out of range"):
            pool.get_block(10)
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_getitem_operator(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test the __getitem__ operator."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        block_tensors = pool[1]
        assert len(block_tensors) == 2
        pool.close()


# ==================== set_block tests ====================

class TestKVCacheMemoryPoolSetBlock:
    """Test suite for set_block method."""

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_set_block_valid(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test setting a block with valid tensors."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        tensor1 = torch.randn(pool.num_layers, pool.block_size, pool.shapes[0][2], dtype=torch.bfloat16)
        tensor2 = torch.randn(pool.num_layers, pool.block_size, pool.shapes[1][2], dtype=torch.bfloat16)

        pool.set_block(0, [tensor1, tensor2])

        retrieved = pool.get_block(0)
        assert len(retrieved) == 2
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_set_block_wrong_count(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test setting a block with wrong number of tensors raises ValueError."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        tensor1 = torch.randn(pool.num_layers, pool.block_size, pool.shapes[0][2], dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="Expected 2 tensors"):
            pool.set_block(0, [tensor1])
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_set_block_wrong_shape(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test setting a block with wrong tensor shape raises ValueError."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        tensor1 = torch.randn(pool.block_size, pool.shapes[0][2], dtype=torch.bfloat16)
        tensor2 = torch.randn(pool.block_size, pool.shapes[1][2], dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="shape mismatch"):
            pool.set_block(0, [tensor1, tensor2])
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_setitem_operator(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test the __setitem__ operator."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        tensor1 = torch.randn(pool.num_layers, pool.block_size, pool.shapes[0][2], dtype=torch.bfloat16)
        tensor2 = torch.randn(pool.num_layers, pool.block_size, pool.shapes[1][2], dtype=torch.bfloat16)

        pool[1] = [tensor1, tensor2]
        pool.close()


# ==================== batch_layer_copy_to_npu tests ====================

class TestKVCacheMemoryPoolBatchLayerCopy:
    """Test suite for batch_layer_copy_to_npu method."""

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_batch_layer_copy_to_npu(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test batch copy to NPU with consecutive blocks."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        # Create mock NPU blocks: [block][layer][kvi_tensor]
        npu_blocks = []
        for block_idx in range(2):
            block_layers = []
            for layer_idx in range(pool.num_layers):
                layer_tensors = []
                for kvi_idx in range(len(pool.shapes)):
                    tensor = torch.randn(pool.block_size, pool.shapes[kvi_idx][2], dtype=torch.bfloat16)
                    layer_tensors.append(tensor)
                block_layers.append(layer_tensors)
            npu_blocks.append(block_layers)

        block_ids = [0, 1]
        result = pool.batch_layer_copy_to_npu(block_ids, npu_blocks)

        assert len(result) == 4
        batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes = result
        assert len(batch_device_mem) > 0
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_batch_layer_copy_with_layer_indices(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test batch copy with custom layer indices."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        npu_blocks = []
        for block_idx in range(2):
            block_layers = []
            for layer_idx in range(pool.num_layers):
                layer_tensors = []
                for kvi_idx in range(len(pool.shapes)):
                    tensor = torch.randn(pool.block_size, pool.shapes[kvi_idx][2], dtype=torch.bfloat16)
                    layer_tensors.append(tensor)
                block_layers.append(layer_tensors)
            npu_blocks.append(block_layers)

        block_ids = [0, 1]
        layer_indices = [1, 0]

        result = pool.batch_layer_copy_to_npu(block_ids, npu_blocks, layer_indices)

        assert len(result) == 4
        pool.close()


# ==================== memcpy_async tests ====================

class TestKVCacheMemoryPoolMemcpyAsync:
    """Test suite for memcpy_async method."""

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_memcpy_async(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test async memcpy operation."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        batch_device_mem = [0x1000, 0x2000]
        batch_device_max = [1024, 2048]
        batch_host_mem = [0x3000, 0x4000]
        batch_host_sizes = [1024, 2048]

        pool.memcpy_async(batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes)

        assert pool.ascend_cl_stream.sync_called == True
        assert len(pool.ascend_cl_stream.memcpy_calls) == 2
        pool.close()


# ==================== close and cleanup tests ====================

class TestKVCacheMemoryPoolClose:
    """Test suite for close and cleanup methods."""

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_close_method(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test that close method properly cleans up resources."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        assert hasattr(pool, 'mmap_obj')
        assert hasattr(pool, 'fd')
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_del_method(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test that __del__ method calls close."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        pool.__del__()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_close_idempotent(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test close can be called repeatedly without raising."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        pool.close()
        pool.close()
        pool.__del__()


# ==================== info method tests ====================

class TestKVCacheMemoryPoolInfo:
    """Test suite for info method."""

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_info_method(self, mock_get_tp, mock_hugepage_file, prefill_shape, capsys):
        """Test that info method prints correct information."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        pool.info()

        captured = capsys.readouterr()
        assert "KV Cache Memory Pool" in captured.out
        assert "Data type" in captured.out
        assert "Blocks:" in captured.out
        pool.close()


# ==================== Integration tests ====================

class TestKVCacheMemoryPoolIntegration:
    """Integration-style tests for KVCacheMemoryPool."""

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_get_set_cycle(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test getting and setting blocks in a cycle."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        tensor1 = torch.randn(pool.num_layers, pool.block_size, pool.shapes[0][2], dtype=torch.bfloat16)
        tensor2 = torch.randn(pool.num_layers, pool.block_size, pool.shapes[1][2], dtype=torch.bfloat16)
        pool[0] = [tensor1, tensor2]

        retrieved = pool[0]
        assert len(retrieved) == 2
        pool.close()

    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.NPUTensorRegister", MockNPUTensorRegister)
    @patch("omni_cache.cache.memory.memory_pool.AscendCLStream", MockAscendCLStream)
    def test_multiple_blocks(self, mock_get_tp, mock_hugepage_file, prefill_shape):
        """Test handling multiple blocks."""
        mock_get_tp.return_value = MockTPGroup(world_size=16, rank=0)

        pool = KVCacheMemoryPool(
            hugepage_path=mock_hugepage_file,
            mmap_size=10 * 1024 * 1024,
            shape=prefill_shape,
            rank=0
        )

        for i in range(pool.num_blocks):
            tensor1 = torch.randn(pool.num_layers, pool.block_size, pool.shapes[0][2], dtype=torch.bfloat16)
            tensor2 = torch.randn(pool.num_layers, pool.block_size, pool.shapes[1][2], dtype=torch.bfloat16)
            pool[i] = [tensor1, tensor2]

        for i in range(pool.num_blocks):
            block = pool[i]
            assert len(block) == 2
        pool.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
