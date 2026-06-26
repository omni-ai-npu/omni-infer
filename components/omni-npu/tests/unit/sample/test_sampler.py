# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import unittest
import torch
from unittest.mock import MagicMock, patch
import omni_npu.sample.sampler as sampler_mod
from omni_npu.sample.sampler import NPUSamplerV1, _apply_penalties_v1

class TestNPUSamplerV1(unittest.TestCase):

    def test_apply_penalties_v1_math(self):
        num_seqs = 2
        vocab_size = 10
        logits = torch.ones((num_seqs, vocab_size), dtype=torch.float32)
        prompt_mask = torch.zeros((num_seqs, vocab_size), dtype=torch.bool)
        output_mask = torch.zeros((num_seqs, vocab_size), dtype=torch.bool)
        output_bin_counts = torch.zeros((num_seqs, vocab_size), dtype=torch.int32)
        
        output_mask[0, 5] = True
        output_bin_counts[0, 5] = 2
        
        presence_penalties = torch.tensor([1.0, 0.0], dtype=torch.float32)
        frequency_penalties = torch.tensor([1.0, 0.0], dtype=torch.float32)
        repetition_penalties = torch.tensor([2.0, 1.0], dtype=torch.float32)
        
        out_logits = _apply_penalties_v1(
            logits.clone(), prompt_mask, output_mask, output_bin_counts,
            presence_penalties, frequency_penalties, repetition_penalties,
            do_presence_penalties=True, do_frequency_penalties=True, do_repetition_penalties=True
        )
        self.assertAlmostEqual(out_logits[0, 5].item(), -2.5, places=4)
        self.assertAlmostEqual(out_logits[0, 0].item(), 1.0, places=4)

    def test_bypass_when_cache_disabled(self):
        original_flag = sampler_mod.ENABLE_NPU_PENALTY_CACHE
        sampler_mod.ENABLE_NPU_PENALTY_CACHE = False
        
        try:
            with patch('vllm.v1.sample.sampler.Sampler.__init__', return_value=None), \
                 patch('vllm.v1.sample.sampler.Sampler.forward', return_value="bypassed"), \
                 patch('omni_npu.sample.sampler.torch_npu', MagicMock()), \
                 patch('omni_npu.sample.sampler.NPUTopKTopPSampler', MagicMock()):
                 
                sampler = NPUSamplerV1()
                res = sampler.forward(logits=torch.tensor([[1.0]]), sampling_metadata=MagicMock())
                self.assertEqual(res, "bypassed")
        finally:
            sampler_mod.ENABLE_NPU_PENALTY_CACHE = original_flag

    def test_forward_complex_paths(self):
        original_flag = sampler_mod.ENABLE_NPU_PENALTY_CACHE
        sampler_mod.ENABLE_NPU_PENALTY_CACHE = True
        
        mock_ctx = MagicMock()
        mock_meta = MagicMock()
        mock_ctx.sampling_metadata = mock_meta
        
        try:
            # We use patch.object here so Python handles the exact cleanup automatically
            # preventing any missing function errors in subsequent smoke tests.
            with patch.object(sampler_mod, 'get_forward_context', return_value=mock_ctx), \
                 patch('vllm.v1.sample.sampler.Sampler.__init__', return_value=None), \
                 patch('omni_npu.sample.sampler.torch_npu', MagicMock()), \
                 patch('omni_npu.sample.sampler.NPUTopKTopPSampler', MagicMock()):
                
                sampler = NPUSamplerV1()
                sampler.topk_topp_sampler = MagicMock(return_value=(torch.tensor([[2]]), None))
                
                mock_meta.logitsprocs = None
                mock_meta.no_penalties = True
                mock_meta.all_greedy = True
                mock_meta.temperature = None 
                
                out1 = sampler.forward(logits=torch.tensor([[1.0]]), sampling_metadata=None)
                self.assertEqual(out1.sampled_token_ids[0, 0].item(), 0)
                
                mock_meta.logitsprocs = MagicMock()
                mock_proc = MagicMock()
                mock_proc.apply.return_value = torch.tensor([[99.0]])
                mock_meta.logitsprocs.non_argmax_invariant = [mock_proc]
                
                out2 = sampler.forward(logits=torch.tensor([[1.0]]), sampling_metadata=mock_meta)
                self.assertEqual(out2.sampled_token_ids[0, 0].item(), 0)
                
                mock_meta.logitsprocs = None
                
                mock_meta.no_penalties = False
                mock_meta.presence_penalties = torch.zeros(1)
                mock_meta.frequency_penalties = torch.zeros(1)
                mock_meta.repetition_penalties = torch.ones(1)
                
                sampler.npu_input_batch = MagicMock()
                sampler.npu_input_batch.presence_penalties_reqs = [1]
                sampler.npu_input_batch.frequency_penalties_reqs = [1]
                sampler.npu_input_batch.repetition_penalties_reqs = [1]
                
                sampler.npu_input_batch.prompt_mask = torch.zeros(1, 2, dtype=torch.bool)
                sampler.npu_input_batch.output_mask = torch.zeros(1, 2, dtype=torch.bool)
                sampler.npu_input_batch.output_bin_counts = torch.zeros(1, 2, dtype=torch.int32)
                
                out3 = sampler.forward(logits=torch.tensor([[1.0, 5.0]]), sampling_metadata=mock_meta)
                self.assertEqual(out3.sampled_token_ids.shape, (1, 1))
                
                mock_meta.all_greedy = False
                mock_meta.temperature = torch.tensor([0.0])
                
                out4 = sampler.forward(logits=torch.tensor([[1.0, 5.0]]), sampling_metadata=mock_meta)
                self.assertEqual(out4.sampled_token_ids[0, 0].item(), 1)
                
                out5 = sampler.forward(logits=None, sampling_metadata=mock_meta)
                self.assertEqual(out5.sampled_token_ids.shape, (1, 0))
                self.assertEqual(out5.sampled_token_ids.device.type, "cpu")
                
        finally:
            sampler_mod.ENABLE_NPU_PENALTY_CACHE = original_flag