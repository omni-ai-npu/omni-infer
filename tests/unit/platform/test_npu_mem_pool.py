# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open
import pytest
import torch

# Mock omni.adaptors... module
mock_omni = MagicMock()
sys.modules["omni.adaptors.vllm.npu_mem_allocator"] = mock_omni

# Mock torch.npu
if not hasattr(torch, "npu"):
    torch.npu = MagicMock()
    torch.npu.memory = MagicMock()
    torch.npu.memory.NPUPluggableAllocator = MagicMock()
    torch.npu.memory.MemPool = MagicMock()
    torch.npu.memory.use_mem_pool = MagicMock()

# Import module under test
import omni.worker.npu_mem_pool as npu_mem_pool
from omni.worker.npu_mem_pool import NpuMemAllocator, find_loaded_library


class TestFindLoadedLibrary:
    def test_find_loaded_library_success(self):
        # Simulate content of /proc/self/maps
        mock_maps_content = (
            "55a 7fb r-xp 00000 /usr/lib/libc.so\n"
            "7fb 7fc ---p 00000 /usr/lib/libc.so\n"
            "address /path/to/libnpu_mem_allocator.so\n"
        )

        with patch("builtins.open", mock_open(read_data=mock_maps_content)):
            path = find_loaded_library("libnpu_mem_allocator")
            assert path == "/path/to/libnpu_mem_allocator.so"

    def test_find_loaded_library_not_found(self):
        with patch("builtins.open", mock_open(read_data="")):
            path = find_loaded_library("non_existent_lib")
            assert path is None


class TestNpuMemAllocator:

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Reset singleton and Mock state before each test."""
        # Reset singleton
        NpuMemAllocator.instance = None
        # Force availability to True to test logic
        npu_mem_pool.npu_mem_available = True

        # Mock underlying C functions
        npu_mem_pool.memcpy = MagicMock()
        npu_mem_pool.python_create_and_map = MagicMock()
        npu_mem_pool.python_unmap_and_release = MagicMock()

        # Mock environment variables
        with patch.dict("os.environ", {}, clear=True):
            yield

        # Cleanup
        NpuMemAllocator.instance = None

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        instance1 = NpuMemAllocator.get_instance()
        instance2 = NpuMemAllocator.get_instance()
        assert instance1 is instance2
        assert isinstance(instance1, NpuMemAllocator)

    def test_init_raises_error_with_bad_env(self):
        """Test error raised if env var contains expandable_segments."""
        with patch.dict("os.environ", {"PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"}):
            # Must set instance to None to trigger __init__
            NpuMemAllocator.instance = None
            with pytest.raises(RuntimeError, match="Expandable segments are not compatible"):
                NpuMemAllocator.get_instance()

    def test_python_malloc_and_free_callback(self):
        """Test malloc and free callback logic."""
        allocator = NpuMemAllocator.get_instance()

        # Simulate allocation params: (device, size, ptr, handle)
        fake_handle = (0, 1024, 123456, 999)
        fake_ptr = 123456

        # 1. Test Malloc
        allocator.current_tag = "test_layer"
        allocator.python_malloc_callback(fake_handle)

        assert fake_ptr in allocator.pointer_to_data
        data = allocator.pointer_to_data[fake_ptr]
        assert data.handle == fake_handle
        assert data.tag == "test_layer"

        # 2. Test Free
        returned_handle = allocator.python_free_callback(fake_ptr)

        assert fake_ptr not in allocator.pointer_to_data
        assert returned_handle == fake_handle

    @patch("torch.empty")
    def test_sleep_offloads_memory(self, mock_torch_empty):
        """Test Sleep mode: should offload memory to CPU and unmap NPU memory."""
        allocator = NpuMemAllocator.get_instance()

        # Prepare data
        ptr_addr = 0x100
        size = 1024
        fake_handle = (0, size, ptr_addr, 0x999)
        allocator.pointer_to_data[ptr_addr] = npu_mem_pool.AllocationData(
            handle=fake_handle, tag="default"
        )

        # Mock CPU tensor
        mock_cpu_tensor = MagicMock()
        mock_cpu_tensor.data_ptr.return_value = 0x200  # CPU address
        mock_torch_empty.return_value = mock_cpu_tensor

        # Execute Sleep
        allocator.sleep(offload_tags=("default",))

        # Verify:
        # 1. CPU Tensor created
        mock_torch_empty.assert_called_once()
        # 2. memcpy called (Device -> Host, flag 2)
        npu_mem_pool.memcpy.assert_called_with(
            0x200,  # dest (cpu)
            size,  # dest_max (destination buffer size)
            ptr_addr,  # src (npu)
            size,  # size
            2  # ACL_MEMCPY_DEVICE_TO_HOST
        )
        # 3. Backup tensor saved in Data
        assert allocator.pointer_to_data[ptr_addr].cpu_backup_tensor is not None
        # 4. unmap called
        npu_mem_pool.python_unmap_and_release.assert_called_with(*fake_handle)

    @patch("torch.empty")  # Just to prevent error, empty is not actually called in wake_up
    def test_wake_up_restores_memory(self, mock_torch_empty):
        """Test Wake Up mode: should remap and copy memory back."""
        allocator = NpuMemAllocator.get_instance()

        # Prepare data that is already asleep
        ptr_addr = 0x100
        size = 100
        fake_handle = (0, size, ptr_addr, 0x999)

        mock_backup_tensor = MagicMock()
        mock_backup_tensor.data_ptr.return_value = 0x200
        mock_backup_tensor.numel.return_value = 100
        mock_backup_tensor.element_size.return_value = 1  # 100 bytes

        data_obj = npu_mem_pool.AllocationData(handle=fake_handle, tag="default")
        data_obj.cpu_backup_tensor = mock_backup_tensor
        allocator.pointer_to_data[ptr_addr] = data_obj

        # Execute Wake Up
        allocator.wake_up(tags=["default"])

        # Verify:
        # 1. Remap
        npu_mem_pool.python_create_and_map.assert_called_with(*fake_handle)
        # 2. memcpy (Host -> Device, flag 1)
        npu_mem_pool.memcpy.assert_called_with(
            ptr_addr,  # dest (npu)
            size,  # dest_max (destination buffer size)
            0x200,  # src (cpu)
            size,
            1  # ACL_MEMCPY_HOST_TO_DEVICE
        )
        # 3. backup tensor cleared
        assert allocator.pointer_to_data[ptr_addr].cpu_backup_tensor is None

    def test_use_memory_pool_context_manager(self):
        """Test context manager correctly switches Tag."""
        allocator = NpuMemAllocator.get_instance()
        original_tag = allocator.current_tag

        # Mock use_memory_pool_with_allocator (involves underlying PyTorch calls)
        with patch("omni.worker.npu_mem_pool.use_memory_pool_with_allocator") as mock_ctx:
            mock_ctx.return_value.__enter__.return_value = "mock_data"

            with allocator.use_memory_pool(tag="temp_tag"):
                assert allocator.current_tag == "temp_tag"
                assert allocator.allocator_and_pools["temp_tag"] == "mock_data"

            # Revert after exit
            assert allocator.current_tag == original_tag

    def test_use_memory_pool_reuses_allocator_and_pool_for_same_tag(self):
        """Test same tag reuses existing pool on second call."""
        allocator = NpuMemAllocator.get_instance()
        original_tag = allocator.current_tag
        temp_tag = "temp_tag"

        first_pool = MagicMock(name="first_pool")
        first_alloc = MagicMock(name="first_alloc")
        first_data = (first_pool, first_alloc)

        with patch("omni.worker.npu_mem_pool.use_memory_pool_with_allocator") as mock_ctx, \
             patch("torch.npu.memory.use_mem_pool") as mock_use_mem_pool:
            mock_ctx.return_value.__enter__.return_value = first_data
            mock_use_mem_pool.return_value.__enter__.return_value = None

            # First call: create allocator+pool via use_memory_pool_with_allocator.
            with allocator.use_memory_pool(tag=temp_tag):
                assert allocator.current_tag == temp_tag
                assert allocator.allocator_and_pools[temp_tag] == first_data

            # Second call: should reuse existing allocator+pool.
            with allocator.use_memory_pool(tag=temp_tag):
                assert allocator.current_tag == temp_tag

            assert mock_ctx.call_count == 1
            mock_use_mem_pool.assert_called_once_with(first_pool)
            assert allocator.allocator_and_pools[temp_tag] == first_data
            assert allocator.current_tag == original_tag

    def test_get_current_usage(self):
        """Test memory usage statistics."""
        allocator = NpuMemAllocator.get_instance()

        # Add two allocation records
        allocator.pointer_to_data[1] = npu_mem_pool.AllocationData(
            handle=(0, 100, 1, 0), tag="a"
        )
        allocator.pointer_to_data[2] = npu_mem_pool.AllocationData(
            handle=(0, 200, 2, 0), tag="b"
        )

        assert allocator.get_current_usage() == 300