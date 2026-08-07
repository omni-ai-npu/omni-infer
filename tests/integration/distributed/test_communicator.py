# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for NPUCommunicator that REQUIRE actual NPU hardware.
These tests verify end-to-end functionality with real NPU devices.

Run these tests only on systems with NPU hardware and torch_npu installed.
"""

import unittest
import os

import pytest
import torch

# Skip all tests if NPU is not available
from ..utils.common_utils import (
    skipif_no_npu,
    NPU_AVAILABLE
)


@pytest.mark.integration
@skipif_no_npu
class TestNPUCommunicatorIntegration(unittest.TestCase):
    """Integration tests for NPUCommunicator (requires NPU hardware)"""

    @classmethod
    def setUpClass(cls):
        """Set up test class - verify NPU availability"""
        if not NPU_AVAILABLE:
            raise unittest.SkipTest("NPU hardware not available")
        
        # Initialize distributed environment if not already initialized
        if not torch.distributed.is_initialized():
            # For single-device testing, we'll skip distributed init
            # Multi-device tests should be run with torchrun
            pass

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('npu:0')
        torch.npu.set_device(self.device)

    def test_npu_device_available(self):
        """Test that NPU device is accessible"""
        self.assertTrue(hasattr(torch, 'npu'))
        self.assertGreater(torch.npu.device_count(), 0)
        
    def test_tensor_creation_on_npu(self):
        """Test creating tensors on NPU device"""
        tensor = torch.randn(4, 4, device=self.device)
        self.assertEqual(tensor.device.type, 'npu')
        self.assertEqual(tensor.shape, torch.Size([4, 4]))

    def test_npu_communicator_initialization_with_real_npu(self):
        """Test NPUCommunicator can be initialized with real NPU"""
        from omni_npu.distributed.communicator import NPUCommunicator
        from torch.distributed import new_group
        
        # This test verifies the communicator can be instantiated
        # Full distributed tests require multi-process setup with torchrun
        
        # Create a dummy process group for testing
        # In real usage, this would come from vLLM's distributed setup
        if torch.distributed.is_initialized():
            cpu_group = torch.distributed.group.WORLD
            device_group = torch.distributed.group.WORLD
            
            communicator = NPUCommunicator(
                cpu_group=cpu_group,
                device=self.device,
                device_group=device_group,
                unique_name="test_npu"
            )
            
            self.assertIsNotNone(communicator)
            self.assertEqual(communicator.dist_module, torch.distributed)
        else:
            # Skip if distributed not initialized
            self.skipTest("Distributed environment not initialized")

    def test_npu_memory_allocation(self):
        """Test NPU memory allocation and deallocation"""
        initial_memory = torch.npu.memory_allocated(self.device)
        
        # Allocate a large tensor
        large_tensor = torch.randn(1000, 1000, device=self.device)
        allocated_memory = torch.npu.memory_allocated(self.device)
        
        self.assertGreater(allocated_memory, initial_memory)
        
        # Free the tensor
        del large_tensor
        torch.npu.empty_cache()

    def test_npu_operations(self):
        """Test basic NPU tensor operations"""
        a = torch.randn(4, 4, device=self.device)
        b = torch.randn(4, 4, device=self.device)
        
        # Test basic operations
        c = a + b
        d = torch.matmul(a, b)
        
        self.assertEqual(c.device.type, 'npu')
        self.assertEqual(d.device.type, 'npu')
        self.assertEqual(c.shape, torch.Size([4, 4]))
        self.assertEqual(d.shape, torch.Size([4, 4]))


@pytest.mark.integration
@pytest.mark.multi_device
@pytest.mark.slow
@skipif_no_npu
class TestNPUCommunicatorMultiDevice(unittest.TestCase):
    """
    Multi-device integration tests for NPUCommunicator.
    
    These tests should be run with torchrun:
    torchrun --nproc_per_node=2 -m pytest tests/integration/test_npu_communicator_integration.py
    """

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment for multi-device tests"""
        if not NPU_AVAILABLE:
            raise unittest.SkipTest("NPU hardware not available")
        
        # Check if we have multiple NPUs
        if torch.npu.device_count() < 2:
            raise unittest.SkipTest("Multiple NPU devices required")
        
        # Initialize distributed if not already done
        if not torch.distributed.is_initialized():
            # Check if running with torchrun (environment variables set)
            import os
            if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
                raise unittest.SkipTest("Distributed environment not initialized. Run with torchrun.")
            
            # Initialize process group with HCCL backend for NPU
            # This MUST use HCCL - if it fails, the test should fail, not skip
            torch.distributed.init_process_group(backend='hccl')

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment"""
        # When running with torchrun, let it handle process group cleanup
        # Calling destroy_process_group() can cause SIGABRT on exit
        if torch.distributed.is_initialized():
            try:
                # Just ensure all ranks finish their tests
                torch.distributed.barrier()
            except Exception:
                pass
        # Note: Process group will be destroyed when the process exits

    def setUp(self):
        """Set up test fixtures"""
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(f'npu:{self.rank}')
        torch.npu.set_device(self.device)

    def test_all_reduce_integration(self):
        """Test all_reduce with real NPU devices"""
        from omni_npu.distributed.communicator import NPUCommunicator
        
        cpu_group = torch.distributed.group.WORLD
        device_group = torch.distributed.group.WORLD
        
        communicator = NPUCommunicator(
            cpu_group=cpu_group,
            device=self.device,
            device_group=device_group,
            unique_name="test_all_reduce"
        )
        
        # Create a tensor with rank-specific values
        tensor = torch.ones(4, 4, device=self.device) * (self.rank + 1)
        original_value = tensor[0, 0].item()
        
        # Perform all_reduce (sum)
        result = communicator.all_reduce(tensor)
        
        # After all_reduce with sum, each element should be sum of all ranks
        expected_sum = sum(range(1, self.world_size + 1))
        self.assertAlmostEqual(result[0, 0].item(), expected_sum, places=5)

    def test_all_gather_integration(self):
        """Test all_gather with real NPU devices"""
        from omni_npu.distributed.communicator import NPUCommunicator
        
        cpu_group = torch.distributed.group.WORLD
        device_group = torch.distributed.group.WORLD
        
        communicator = NPUCommunicator(
            cpu_group=cpu_group,
            device=self.device,
            device_group=device_group,
            unique_name="test_all_gather"
        )
        
        # Create a tensor with rank-specific values
        tensor = torch.ones(2, 4, device=self.device) * (self.rank + 1)
        
        # Perform all_gather
        result = communicator.all_gather(tensor, dim=0)
        
        # Result should have shape [2 * world_size, 4]
        expected_shape = torch.Size([2 * self.world_size, 4])
        self.assertEqual(result.shape, expected_shape)

    def test_send_recv_integration(self):
        """Test send/recv with real NPU devices"""
        from omni_npu.distributed.communicator import NPUCommunicator
        
        cpu_group = torch.distributed.group.WORLD
        device_group = torch.distributed.group.WORLD
        
        communicator = NPUCommunicator(
            cpu_group=cpu_group,
            device=self.device,
            device_group=device_group,
            unique_name="test_send_recv"
        )
        
        if self.rank == 0:
            # Rank 0 sends
            tensor = torch.ones(4, 4, device=self.device) * 42.0
            communicator.send(tensor, dst=1)
        elif self.rank == 1:
            # Rank 1 receives
            tensor = communicator.recv(torch.Size([4, 4]), torch.float32, src=0)
            self.assertAlmostEqual(tensor[0, 0].item(), 42.0, places=5)


if __name__ == '__main__':
    # Print NPU availability info
    print(f"NPU Available: {NPU_AVAILABLE}")
    if NPU_AVAILABLE:
        print(f"NPU Device Count: {torch.npu.device_count()}")
        print(f"Distributed Initialized: {torch.distributed.is_initialized()}")
    
    unittest.main()
