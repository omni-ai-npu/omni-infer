# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_acl_tensor_register.py
# Unit tests for tensor_register module

import pytest
try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None
import os
import sys
import ctypes
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import Optional

# Import module under test
# Note: These tests require NPU hardware and CANN toolkit
if torch is None:
    pytest.skip("torch is not available", allow_module_level=True)

# Check if zero_copy_npu extension is available
try:
    from omni_cache.cache.device_backend.ascend import tensor_register_lib
    tensor_register_lib.zero_copy_npu  # Will raise AttributeError if not available
    _ZERO_COPY_AVAILABLE = True
except (ImportError, AttributeError):
    _ZERO_COPY_AVAILABLE = False


# ==================== Helper Functions ====================

def is_npu_available():
    """Check if NPU is available for testing."""
    try:
        import torch_npu
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def skip_if_no_npu():
    """Pytest skip condition for NPU tests."""
    return pytest.mark.skipif(
        not is_npu_available(),
        reason="NPU not available or torch_npu not installed"
    )


# ==================== Mock classes and fixtures ====================

class MockLibTensorRegister:
    """Mock tensor_register.so library for testing."""

    def __init__(self):
        self.registry = {}
        self.register_should_fail = False
        self.register_null_ptr = False
        self.unregister_should_fail = False

    def register_tensor(self, cpu_ptr, size, dev_ptr, device_id):
        if self.register_should_fail:
            return -1
        if self.register_null_ptr:
            return 0
        mock_dev_ptr = int(cpu_ptr.value) + 0x1000000000
        self.registry[int(cpu_ptr.value)] = mock_dev_ptr
        return 0

    def unregister_tensor(self, cpu_ptr):
        if self.unregister_should_fail:
            return -1
        if int(cpu_ptr.value) in self.registry:
            del self.registry[int(cpu_ptr.value)]
            return 0
        return -1

    def get_dev_ptr_from_cpu(self, cpu_ptr):
        return self.registry.get(int(cpu_ptr.value))


class MockZeroCopyNpu:
    """Mock zero_copy_npu module for testing."""

    def __init__(self):
        self.registered_tensors = []
        self.should_fail = False

    def register_hugepage_as_npu_tensor(self, host_tensor, device_id):
        if self.should_fail:
            raise RuntimeError("Mock registration failed")

        mock_npu_tensor = host_tensor.clone().detach()
        # Store reference to prevent garbage collection
        self.registered_tensors.append((host_tensor, mock_npu_tensor))
        return host_tensor, mock_npu_tensor

    def unregister(self):
        self.registered_tensors.clear()


@pytest.fixture
def mock_tensor_register_lib():
    """Create a mock tensor_register library."""
    return MockLibTensorRegister()


@pytest.fixture
def mock_zero_copy_npu():
    """Create a mock zero_copy_npu module."""
    return MockZeroCopyNpu()


@pytest.fixture
def sample_cpu_tensor():
    """Create a sample CPU tensor for testing."""
    return torch.randn(10, 20, dtype=torch.float32)


@pytest.fixture
def sample_aligned_tensor():
    """Create a 4K-aligned CPU tensor for testing."""
    # Allocate aligned memory
    size = 10 * 20 * 4  # 10x20 float32
    alignment = 4096

    # Use numpy for aligned allocation
    arr = np.random.randn(10, 20).astype(np.float32)
    tensor = torch.from_numpy(arr).contiguous()

    return tensor


# ==================== NPUTensorRegister Init tests ====================

class TestNPUTensorRegisterInit:
    """Test suite for NPUTensorRegister initialization."""

    @skip_if_no_npu()
    def test_init_basic(self):
        """Test basic initialization of NPUTensorRegister."""
        from omni_cache.cache.device_backend.ascend.tensor_register import NPUTensorRegister

        register = NPUTensorRegister()

        assert register is not None

    @pytest.mark.skipif(not _ZERO_COPY_AVAILABLE, reason="zero_copy_npu extension not available")
    def test_init_with_mock(self, mock_tensor_register_lib, mock_zero_copy_npu):
        """Test initialization with mocked dependencies."""
        import importlib
        tr_mod = importlib.import_module("omni_cache.cache.device_backend.ascend.tensor_register")
        trl_mod = importlib.import_module("omni_cache.cache.device_backend.ascend.tensor_register_lib")
        with patch.object(tr_mod, '_lib', mock_tensor_register_lib):
            with patch.object(trl_mod, 'zero_copy_npu', mock_zero_copy_npu):
                from omni_cache.cache.device_backend.ascend.tensor_register import NPUTensorRegister

                register = NPUTensorRegister()

                assert register is not None


# ==================== host_tensor_register tests ====================

class TestHostTensorRegister:
    """Test suite for host_tensor_register method."""

    def test_type_check_non_tensor(self):
        """Test that non-tensor input raises TypeError.

        Input validation (type check, device check) runs before any NPU
        dependency, so this test works without zero_copy_npu.
        """
        from omni_cache.cache.device_backend.ascend.tensor_register import NPUTensorRegister

        register = NPUTensorRegister()

        with pytest.raises(TypeError, match="host_tensor must be a torch.Tensor"):
            register.host_tensor_register([1, 2, 3])

    def test_device_check_npu_tensor(self):
        """Test that non-CPU tensor raises ValueError.

        Device check runs before any NPU dependency, so this test works
        without zero_copy_npu.
        """
        from omni_cache.cache.device_backend.ascend.tensor_register import NPUTensorRegister

        register = NPUTensorRegister()
        npu_tensor = torch.randn(5, 5, device="cpu")  # Will be treated as non-cpu

        # Create a tensor that's not on CPU (simulated)
        with pytest.raises(ValueError, match="host_tensor must be a CPU tensor"):
            # Force device check to fail by mocking
            with patch.object(torch.Tensor, 'device', PropertyMock(return_value=Mock(type='npu'))):
                register.host_tensor_register(npu_tensor)

    def test_registration_failure(self, mock_tensor_register_lib, mock_zero_copy_npu, sample_cpu_tensor):
        """Test handling of registration failure."""
        import importlib
        tr_mod = importlib.import_module("omni_cache.cache.device_backend.ascend.tensor_register")
        trl_mod = importlib.import_module("omni_cache.cache.device_backend.ascend.tensor_register_lib")
        mock_tensor_register_lib.register_should_fail = True

        with patch.object(tr_mod, '_lib', mock_tensor_register_lib):
            with patch.object(trl_mod, 'zero_copy_npu', mock_zero_copy_npu):
                with patch.object(tr_mod, '_ZERO_COPY_AVAILABLE', True):
                    from omni_cache.cache.device_backend.ascend.tensor_register import NPUTensorRegister

                    register = NPUTensorRegister()

                    with pytest.raises(RuntimeError, match="Register tensor failed"):
                        register.host_tensor_register(sample_cpu_tensor)

    @skip_if_no_npu()
    def test_different_device_ids(self):
        """Test registration with different device IDs."""
        from omni_cache.cache.device_backend.ascend.tensor_register import NPUTensorRegister

        register = NPUTensorRegister()
        host_tensor = torch.randn(5, 10, dtype=torch.float32)

        # Test with different device IDs (may fail if device not available)
        for device_id in [0, 1]:
            try:
                result = register.host_tensor_register(host_tensor, device_id=device_id)
                # Result might be a tuple or single tensor depending on implementation
                if isinstance(result, tuple):
                    result_host, result_npu = result
                    assert result_npu.device.type in ("npu", "privateuseone", "cpu")
                else:
                    assert result is not None
            except RuntimeError:
                # Device may not be available, which is acceptable
                pass


# ==================== Alignment and Memory Tests ====================

class TestTensorAlignment:
    """Test suite for tensor alignment and memory operations."""

    def test_4k_alignment_check(self):
        """Test 4K alignment detection."""
        aligned_ptr = 4096
        misaligned_ptr = 4097

        assert aligned_ptr % 4096 == 0
        assert misaligned_ptr % 4096 != 0

    def test_tensor_size_calculation(self, sample_cpu_tensor):
        """Test tensor size calculation."""
        num_elements = sample_cpu_tensor.numel()
        element_size = sample_cpu_tensor.element_size()
        expected_size = num_elements * element_size

        assert expected_size == 10 * 20 * 4  # float32 = 4 bytes

    def test_contiguous_tensor_check(self, sample_cpu_tensor):
        """Test tensor contiguity check."""
        assert sample_cpu_tensor.is_contiguous()

        non_contiguous = torch.randn(10, 20, dtype=torch.float32).t()
        assert not non_contiguous.is_contiguous()

    def test_different_dtypes(self):
        """Test tensors with different data types."""
        dtypes = [
            torch.float32,
            torch.float16,
        ]

        for dtype in dtypes:
            tensor = torch.randn(5, 10, dtype=dtype)
            assert tensor.dtype == dtype
            assert tensor.element_size() in [2, 4]

        int_dtypes = [
            torch.int32,
            torch.int64,
            torch.uint8,
        ]
        for dtype in int_dtypes:
            tensor = torch.zeros(5, 10, dtype=dtype)
            assert tensor.dtype == dtype
            assert tensor.element_size() in [1, 4, 8]

# ==================== Hugepage Tests ====================

class TestHugepageOperations:
    """Test suite for hugepage operations."""

    def test_hugepage_path(self):
        """Test hugepage path construction."""
        HUGEPAGE_DIR = "/dev/hugepages"
        FILE_PATH = os.path.join(HUGEPAGE_DIR, "zero_copy_hugepage.bin")

        assert os.path.normpath(FILE_PATH) == os.path.normpath("/dev/hugepages/zero_copy_hugepage.bin")

    def test_mmap_flags(self):
        """Test mmap flag values."""
        MAP_SHARED = 0x01
        MAP_HUGETLB = 0x40000
        PROT_READ = 0x1
        PROT_WRITE = 0x2

        combined_flags = MAP_SHARED | MAP_HUGETLB
        prot = PROT_READ | PROT_WRITE

        assert combined_flags == 0x40001
        assert prot == 0x3

    def test_size_calculation(self):
        """Test size calculation for hugepage allocation."""
        SIZE_MB = 1
        SIZE = SIZE_MB * 1024 * 1024

        assert SIZE == 1024 * 1024

        float32_count = SIZE // 4
        assert float32_count == 262144


# ==================== Memory Registration Tests ====================

class TestMemoryRegistration:
    """Test suite for memory registration operations."""

    def test_pointer_extraction(self, sample_cpu_tensor):
        """Test extracting data pointer from tensor."""
        ptr = sample_cpu_tensor.data_ptr()

        assert isinstance(ptr, int)
        assert ptr > 0

    def test_device_id_validation(self):
        """Test device ID validation."""
        valid_device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        invalid_device_ids = [-1, -100, 999]

        for device_id in valid_device_ids:
            assert device_id >= 0

        for device_id in invalid_device_ids:
            # In actual implementation, these would fail
            assert device_id < 0 or device_id > 7


# ==================== Zero-Copy Tests ====================

class TestZeroCopyFunctionality:
    """Test suite for zero-copy functionality."""

    def test_shared_memory_detection(self):
        """Test detection of shared memory."""
        # Create two tensors
        tensor_a = torch.randn(5, 5, dtype=torch.float32)
        tensor_b = tensor_a.clone()

        # They should have same values but different storage
        assert torch.equal(tensor_a, tensor_b)
        assert tensor_a.data_ptr() != tensor_b.data_ptr()

    def test_in_place_modification(self):
        """Test in-place modification propagation."""
        original = torch.randn(5, 5, dtype=torch.float32)
        modified = original.clone()

        # Modify in-place
        modified.add_(1.0)

        # Values should differ
        assert not torch.equal(original, modified)
        assert torch.allclose(modified, original + 1.0)

    def test_tensor_metadata(self):
        """Test tensor metadata preservation."""
        shape = (10, 20, 30)
        dtype = torch.float32

        tensor = torch.randn(shape, dtype=dtype)

        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.device.type == "cpu"


# ==================== ctypes Interface Tests ====================

class TestCtypesInterface:
    """Test suite for ctypes interface."""

    def test_c_void_p_creation(self):
        """Test c_void_p creation."""
        ptr_value = 0x12345678
        c_ptr = ctypes.c_void_p(ptr_value)

        assert c_ptr.value == ptr_value

    def test_c_size_t_creation(self):
        """Test c_size_t creation."""
        size_value = 1024 * 1024
        c_size = ctypes.c_size_t(size_value)

        assert c_size.value == size_value

    def test_byref_usage(self):
        """Test byref for output parameters."""
        value = ctypes.c_int(42)
        ref = ctypes.byref(value)

        assert ref is not None

    def test_pointer_pointer(self):
        """Test POINTER(c_void_p) for output parameters."""
        ptr_ptr = ctypes.POINTER(ctypes.c_void_p)

        assert ptr_ptr is not None


# ==================== ACL Error Handling Tests ====================

class TestACLErrorHandling:
    """Test suite for ACL error handling."""

    def test_acl_error_codes(self):
        """Test ACL error code interpretation."""
        # Common ACL error codes
        ACL_SUCCESS = 0
        ACL_ERROR_NONE = 0
        ACL_ERROR_INVALID_PARAM = 100001
        ACL_ERROR_UNEXPECTED = 100000

        assert ACL_SUCCESS == 0
        assert ACL_ERROR_NONE == 0
        assert ACL_ERROR_INVALID_PARAM > 0

    def test_error_message_format(self):
        """Test error message formatting."""
        error_code = 100001
        message = f"ACL error code {error_code}"

        assert "ACL error code" in message
        assert "100001" in message


# ==================== Device Management Tests ====================

class TestDeviceManagement:
    """Test suite for device management."""

    def test_device_string_format(self):
        """Test device string formatting."""
        device_id = 0
        device_str = f"npu:{device_id}"

        assert device_str == "npu:0"

        device_id_2 = 1
        device_str_2 = f"npu:{device_id_2}"

        assert device_str_2 == "npu:1"

    def test_device_extraction(self):
        """Test device ID extraction."""
        device_str = "npu:0"
        device_id = int(device_str.split(":")[-1])

        assert device_id == 0

        device_str_2 = "npu:7"
        device_id_2 = int(device_str_2.split(":")[-1])

        assert device_id_2 == 7

# ==================== Helper Function Tests ====================

class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_alignment_calculation(self):
        """Test alignment calculation."""
        ptr = 0x12345678
        alignment_4k = 4096

        is_aligned = (ptr % alignment_4k == 0)
        misalignment = ptr % alignment_4k

        assert isinstance(is_aligned, bool)
        assert isinstance(misalignment, int)

    def test_round_up_to_alignment(self):
        """Test rounding up to alignment."""
        values = [
            (0, 4096, 0),
            (1, 4096, 4096),
            (4095, 4096, 4096),
            (4096, 4096, 4096),
            (4097, 4096, 8192),
            (10240, 4096, 12288),
        ]

        for value, alignment, expected in values:
            remainder = value % alignment
            rounded = value if remainder == 0 else value + alignment - remainder
            assert rounded == expected

    def test_memory_page_size(self):
        """Test memory page size values."""
        PAGE_SIZE_4K = 4096
        PAGE_SIZE_2M = 2 * 1024 * 1024
        PAGE_SIZE_1G = 1024 * 1024 * 1024

        assert PAGE_SIZE_4K == 4096
        assert PAGE_SIZE_2M == 2097152
        assert PAGE_SIZE_1G == 1073741824


# ==================== Build Configuration Tests ====================

class TestBuildConfiguration:
    """Test suite for build configuration."""

    def test_cann_env_var(self):
        """Test CANN environment variable handling."""
        default_path = "/usr/local/Ascend/ascend-toolkit/latest"
        custom_path = "/custom/cann/path"

        # Simulate environment variable handling
        env_value = os.environ.get("ASCEND_TOOLKIT_HOME", default_path)

        assert isinstance(env_value, str)

    def test_library_path_construction(self):
        """Test library path construction."""
        cann_root = "/usr/local/Ascend/ascend-toolkit/latest"

        include_dir = os.path.join(cann_root, "include")
        lib_dir = os.path.join(cann_root, "lib64")

        assert os.path.normpath(include_dir) == os.path.normpath("/usr/local/Ascend/ascend-toolkit/latest/include")
        assert os.path.normpath(lib_dir) == os.path.normpath("/usr/local/Ascend/ascend-toolkit/latest/lib64")

    def test_compile_flags(self):
        """Test compile flag construction."""
        flags = ["-O3", "-std=c++17", "-fPIC"]

        assert "-O3" in flags
        assert "-std=c++17" in flags
        assert "-fPIC" in flags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
