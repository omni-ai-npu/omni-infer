# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import unittest
import torch
from unittest.mock import MagicMock, patch
from omni_npu.worker.npu_input_batch import NPUInputBatch

class TestNPUInputBatch(unittest.TestCase):
    
    @patch('torch.npu.current_device', return_value=0)
    @patch('torch.device', return_value=torch.device('cpu'))
    def test_all_lifecycle_methods(self, mock_device, mock_curr_dev):
        # We manually init to bypass vLLM heavy constructor
        batch = NPUInputBatch.__new__(NPUInputBatch)
        batch.max_num_reqs = 4
        batch.sampling_metadata = MagicMock()
        batch.batch_update_builder = MagicMock()
        
        # 1. init_npu_tensors
        batch.init_npu_tensors(100)
        self.assertEqual(batch.prompt_mask.shape, (4, 100))
        
        # 2. _make_sampling_metadata
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch._make_sampling_metadata', return_value=MagicMock()) as mock_meta:
            meta = batch._make_sampling_metadata()
            self.assertEqual(meta.npu_input_batch, batch)
            
        # 3. add_request
        req = MagicMock()
        req.prompt_token_ids = [1, 2]
        req.output_token_ids = [3, 4]
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.add_request', return_value=1):
            idx = batch.add_request(req)
            self.assertEqual(idx, 1)
            self.assertTrue(batch.prompt_mask[1, 1].item())
            self.assertEqual(batch.output_bin_counts[1, 3].item(), 1)
            
        # 4. update_sampled_tokens
        batch.update_sampled_tokens(None) # Coverage for early exits
        batch.update_sampled_tokens(torch.tensor([]))

        batch.update_sampled_tokens(torch.tensor([2, -1, 4], dtype=torch.int64))
        self.assertEqual(batch.output_bin_counts[0, 2].item(), 1)
        
        # 5. condense
        batch.batch_update_builder.moved = [(1, 0, None)]
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.condense'):
            batch.condense()
            
        # 6. swap_states
        batch.output_bin_counts[0, 1] = 99
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.swap_states'):
            batch.swap_states(0, 2)
            
        # 7. revert_rejected_tokens (For Speculative Decoding Coverage)
        if hasattr(batch, 'revert_rejected_tokens'):
            accepted_mask = torch.tensor([False, True], dtype=torch.bool)
            token_ids = torch.tensor([1, 2], dtype=torch.int64)
            batch.revert_rejected_tokens(accepted_mask, token_ids)
            
            accepted_mask_2d = torch.tensor([[False], [True]], dtype=torch.bool)
            token_ids_2d = torch.tensor([[1], [2]], dtype=torch.int64)
            batch.revert_rejected_tokens(accepted_mask_2d, token_ids_2d)