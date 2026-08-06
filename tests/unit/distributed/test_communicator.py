# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for NPUCommunicator that do NOT require actual NPU hardware.
These tests use mocking to verify the logic and API contracts.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from contextlib import contextmanager
import pytest
import torch
from torch.distributed import ProcessGroup


@contextmanager
def mock_distributed_environment():
    """Context manager to mock torch.distributed environment"""
    with patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.get_world_size', return_value=1), \
         patch('omni.distributed.communicator.CudaCommunicator.__init__', return_value=None):
        yield


@pytest.mark.unit
class TestNPUCommunicatorUnit(unittest.TestCase):
    """Unit tests for NPUCommunicator (no NPU hardware required)"""
    def test_init_with_torch_npu_available(self):
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            device = torch.device('npu:0')
            
            communicator = NPUCommunicator(
                cpu_group=cpu_group,
                device=device,
                device_group=device_group,
                unique_name="test"
            )
            
            self.assertIsNotNone(communicator)
            self.assertEqual(communicator.dist_module, torch.distributed)

    def test_init_without_torch_npu_raises_error(self):
        """Test NPUCommunicator raises RuntimeError when torch.npu is not available"""
        @contextmanager
        def del_torch_npu():
            torch_npu = getattr(torch, 'npu', None)
            if torch_npu is not None:
                delattr(torch, 'npu')
            yield 
            if torch_npu is not None:
                setattr(torch, 'npu', torch_npu)
                
        with del_torch_npu():
            with mock_distributed_environment():
                from omni.distributed.communicator import NPUCommunicator
                
                cpu_group = Mock(spec=ProcessGroup)
                device_group = Mock(spec=ProcessGroup)
                
                with self.assertRaises(RuntimeError) as context:
                    NPUCommunicator(
                        cpu_group=cpu_group,
                        device_group=device_group,
                    )
                
                self.assertIn("torch.npu", str(context.exception))
                self.assertIn("torch_npu is properly installed", str(context.exception))

    def test_all_reduce_delegates_to_torch_distributed(self):
        """Test all_reduce delegates to torch.distributed.all_reduce"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 1
            
            with patch.object(torch.distributed, 'all_reduce') as mock_all_reduce:
                input_tensor = torch.randn(4, 4)
                result = communicator.all_reduce(input_tensor)
                
                mock_all_reduce.assert_called_once_with(input_tensor, group=device_group)
                self.assertIs(result, input_tensor)

    def test_all_gather_shape_transformation(self):
        """Test all_gather performs correct shape transformation"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            # Mock the parent class __init__ to avoid real distributed calls
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            
            with patch.object(torch.distributed, 'all_gather_into_tensor') as mock_all_gather:
                input_tensor = torch.randn(2, 4)
                result = communicator.all_gather(input_tensor, dim=-1)
                
                mock_all_gather.assert_called_once()
                self.assertIsInstance(result, torch.Tensor)

    def test_reduce_scatter_world_size_one_returns_input(self):
        """Test reduce_scatter returns input tensor when world_size is 1"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.world_size = 1
            
            input_tensor = torch.randn(4, 4)
            result = communicator.reduce_scatter(input_tensor, dim=0)
            
            self.assertIs(result, input_tensor)

    def test_reduce_scatter_delegates_to_torch_distributed(self):
        """Test reduce_scatter delegates to torch.distributed.reduce_scatter_tensor"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            
            with patch.object(torch.distributed, 'reduce_scatter_tensor') as mock_reduce_scatter:
                input_tensor = torch.randn(4, 4)
                result = communicator.reduce_scatter(input_tensor)
                
                mock_reduce_scatter.assert_called_once()
                self.assertIsInstance(result, torch.Tensor)

    def test_send_with_explicit_destination(self):
        """Test send with explicit destination rank"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.rank_in_group = 0
            communicator.world_size = 2
            communicator.ranks = [0, 1]
            
            with patch.object(torch.distributed, 'send') as mock_send:
                tensor = torch.randn(4, 4)
                communicator.send(tensor, dst=1)
                
                mock_send.assert_called_once_with(tensor, 1, device_group)

    def test_send_with_default_destination(self):
        """Test send with default destination (next rank)"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.rank_in_group = 0
            communicator.world_size = 4
            communicator.ranks = [0, 1, 2, 3]
            
            with patch.object(torch.distributed, 'send') as mock_send:
                tensor = torch.randn(4, 4)
                communicator.send(tensor, dst=None)
                
                mock_send.assert_called_once_with(tensor, 1, device_group)

    def test_recv_creates_tensor_with_correct_shape(self):
        """Test recv creates tensor with correct shape and dtype"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            device = torch.device('cpu')
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.rank_in_group = 1
            communicator.world_size = 2
            communicator.ranks = [0, 1]
            communicator.device = device
            
            with patch.object(torch.distributed, 'recv') as mock_recv:
                size = torch.Size([4, 4])
                dtype = torch.float32
                result = communicator.recv(size, dtype)
                
                mock_recv.assert_called_once()
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape, size)
                self.assertEqual(result.dtype, dtype)

    def test_gather_on_destination_rank(self):
        """Test gather returns concatenated tensor on destination rank"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.rank_in_group = 0
            communicator.world_size = 2
            communicator.ranks = [0, 1]
            
            with patch.object(torch.distributed, 'gather') as mock_gather:
                input_tensor = torch.randn(2, 4)
                result = communicator.gather(input_tensor, dst=0)
                
                mock_gather.assert_called_once()
                self.assertIsInstance(result, torch.Tensor)

    def test_gather_on_non_destination_rank(self):
        """Test gather returns None on non-destination rank"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.rank_in_group = 1
            communicator.world_size = 2
            communicator.ranks = [0, 1]
            
            with patch.object(torch.distributed, 'gather') as mock_gather:
                input_tensor = torch.randn(2, 4)
                result = communicator.gather(input_tensor, dst=0, dim=0)
                
                mock_gather.assert_called_once()
                self.assertIsNone(result)

    def test_destroy_returns_none(self):
        """Test destroy method returns None"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            
            result = communicator.destroy()
            self.assertIsNone(result)

    def test_all_gatherv_raises_for_non_zero_dim(self):
        """Test all_gatherv raises NotImplementedError for dim != 0"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            
            input_tensor = torch.randn(2, 4)
            with self.assertRaises(NotImplementedError) as context:
                communicator.all_gatherv(input_tensor, dim=1)
            
            self.assertIn("only dim 0", str(context.exception))

    def test_negative_dim_handling(self):
        """Test that negative dim values are correctly converted"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            
            with patch.object(torch.distributed, 'all_gather_into_tensor'):
                input_tensor = torch.randn(2, 4)
                result = communicator.all_gather(input_tensor, dim=-1)
                self.assertIsInstance(result, torch.Tensor)

    def test_reduce_scatterv_with_sizes(self):
        """Test reduce_scatterv with explicit sizes parameter"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            communicator.rank_in_group = 0
            
            with patch.object(torch.distributed, 'reduce_scatter_tensor') as mock_reduce_scatter:
                input_tensor = torch.randn(6, 4)  # Total size 6, rank 0 gets 3, rank 1 gets 3
                sizes = [3, 3]
                result = communicator.reduce_scatterv(input_tensor, dim=0, sizes=sizes)
                
                mock_reduce_scatter.assert_called_once()
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape[0], 3)  # Should get chunk of size 3

    def test_reduce_scatterv_without_sizes(self):
        """Test reduce_scatterv without sizes parameter (fallback to standard reduce_scatter)"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            communicator.rank_in_group = 0
            
            with patch.object(torch.distributed, 'reduce_scatter_tensor') as mock_reduce_scatter:
                input_tensor = torch.randn(4, 4)  # Evenly divisible by world_size
                result = communicator.reduce_scatterv(input_tensor, dim=0)
                
                mock_reduce_scatter.assert_called_once()
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape[0], 2)  # Should get chunk of size 2

    def test_reduce_scatterv_negative_dim(self):
        """Test reduce_scatterv with negative dimension"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            communicator.rank_in_group = 0
            
            with patch.object(torch.distributed, 'reduce_scatter_tensor'):
                input_tensor = torch.randn(4, 4)
                result = communicator.reduce_scatterv(input_tensor, dim=-1)
                self.assertIsInstance(result, torch.Tensor)

    def test_reduce_scatterv_world_size_one(self):
        """Test reduce_scatterv when world_size is 1 (should return input)"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 1
            
            input_tensor = torch.randn(4, 4)
            def reduce_scatter_tensor(output, input_tensor, group):
                output.copy_(input_tensor)
            communicator.dist_module = MagicMock()
            communicator.dist_module.reduce_scatter_tensor.side_effect = reduce_scatter_tensor
            result = communicator.reduce_scatterv(input_tensor, dim=0)
            assert torch.equal(result, input_tensor)

    def test_all_gatherv_single_tensor_without_sizes(self):
        """Test all_gatherv with single tensor and no sizes parameter"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            communicator.rank_in_group = 0
            communicator.ranks = [0, 1]
            
            with patch.object(torch.distributed, 'all_gather_into_tensor') as mock_all_gather:
                input_tensor = torch.randn(2, 4)
                result = communicator.all_gatherv(input_tensor, dim=0)
                
                mock_all_gather.assert_called_once()
                self.assertIsInstance(result, torch.Tensor)

    def test_all_gatherv_single_tensor_with_sizes(self):
        """Test all_gatherv with single tensor and sizes parameter (broadcast fallback)"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            communicator.rank_in_group = 0
            communicator.ranks = [0, 1]
            
            with patch.object(torch.distributed, 'broadcast') as mock_broadcast:
                input_tensor = torch.randn(2, 4)
                sizes = [2, 2]
                result = communicator.all_gatherv(input_tensor, dim=0, sizes=sizes)
                
                # Should call broadcast for each rank except current rank
                self.assertEqual(mock_broadcast.call_count, 1)  # Only for rank 1
                self.assertIsInstance(result, torch.Tensor)

    def test_all_gatherv_tensor_list_without_sizes(self):
        """Test all_gatherv with tensor list and no sizes parameter"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            communicator.rank_in_group = 0
            communicator.ranks = [0, 1]
            
            with patch.object(torch.distributed, 'all_gather_into_tensor') as mock_all_gather:
                input_tensors = [torch.randn(2, 4), torch.randn(2, 4)]
                result = communicator.all_gatherv(input_tensors, dim=0)
                
                # Should call all_gather_into_tensor for each tensor
                self.assertEqual(mock_all_gather.call_count, 2)
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 2)
                self.assertIsInstance(result[0], torch.Tensor)

    def test_all_gatherv_tensor_list_with_sizes(self):
        """Test all_gatherv with tensor list and sizes parameter"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            communicator.rank_in_group = 0
            communicator.ranks = [0, 1]
            
            with patch.object(torch.distributed, 'broadcast') as mock_broadcast:
                input_tensors = [torch.randn(2, 4), torch.randn(2, 4)]
                sizes = [2, 2]
                result = communicator.all_gatherv(input_tensors, dim=0, sizes=sizes)
                
                # Should call broadcast for each tensor and each rank except current rank
                self.assertEqual(mock_broadcast.call_count, 2)  # 2 tensors × 1 other rank
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 2)

    def test_all_gatherv_non_zero_dim_raises_error(self):
        """Test all_gatherv raises NotImplementedError for dim != 0"""
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator
            
            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)
            
            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            
            input_tensor = torch.randn(2, 4)
            with self.assertRaises(NotImplementedError) as context:
                communicator.all_gatherv(input_tensor, dim=1)
            
            self.assertIn("only dim 0", str(context.exception))

    def test_empty_method(self):
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator

            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)

            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            communicator.rank_in_group = 0
            communicator.ranks = [0, 1]

            communicator.dispatch(None, None, None)
            communicator.combine(None, None)

    def test_prepare_communication_buffer_for_model(self):
        with mock_distributed_environment():
            from omni.distributed.communicator import NPUCommunicator

            cpu_group = Mock(spec=ProcessGroup)
            device_group = Mock(spec=ProcessGroup)

            communicator = NPUCommunicator(cpu_group=cpu_group, device_group=device_group)
            communicator.device_group = device_group
            communicator.world_size = 2
            communicator.rank_in_group = 0
            communicator.ranks = [0, 1]

            communicator.is_ep_communicator = True
            class FusedMoE:
                def maybe_init_modular_kernel(self):
                    self.call = True
            fused = FusedMoE()
            model = MagicMock()
            model.modules.return_value = [fused]
            communicator.prepare_communication_buffer_for_model(model)
            assert hasattr(fused, 'call') and fused.call

            communicator.is_ep_communicator = False
            ret = communicator.prepare_communication_buffer_for_model(model)
            assert ret is None

if __name__ == '__main__':
    unittest.main()