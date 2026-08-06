# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import unittest
import torch
from unittest.mock import MagicMock, patch
from omni.worker.npu_input_batch import NPUInputBatch


class TestNPUInputBatch(unittest.TestCase):

    def setUp(self):
        """Create a minimal NPUInputBatch instance for testing."""
        self.batch = NPUInputBatch.__new__(NPUInputBatch)
        self.batch.max_num_reqs = 4
        self.batch.sampling_metadata = MagicMock()
        self.batch.batch_update_builder = MagicMock()
        self.batch.disable_penalty_cache = True
        self.batch.disable_multi_mtp_cache = True

    # ── init_penalty_cache ──────────────────────────────────────────────

    def test_init_penalty_cache(self):
        """init_penalty_cache creates correct tensor shapes and attaches to sampling_metadata."""
        self.batch.init_penalty_cache(vocab_size=100, device=torch.device('cpu'))

        self.assertFalse(self.batch.disable_penalty_cache)
        self.assertEqual(self.batch.vocab_size, 100)
        self.assertEqual(self.batch.npu_device, torch.device('cpu'))
        self.assertEqual(self.batch.prompt_mask.shape, (4, 100))
        self.assertEqual(self.batch.output_mask.shape, (4, 100))
        self.assertEqual(self.batch.output_bin_counts.shape, (4, 100))
        self.assertIs(self.batch.sampling_metadata.npu_input_batch, self.batch)

    # ── init_target_model_hidden_states_cache ──────────────────────────

    def test_init_target_model_hidden_states_cache(self):
        """init_target_model_hidden_states_cache creates correct tensor shapes."""
        self.batch.init_target_model_hidden_states_cache(
            n_predict=5, hidden_size=64, dtype=torch.float16, device=torch.device('cpu'),
        )

        self.assertFalse(self.batch.disable_multi_mtp_cache)
        # Shape: (max_num_reqs, n_predict + 1, hidden_size)
        self.assertEqual(
            self.batch.target_model_hidden_states_cache.shape, (4, 6, 64))
        self.assertEqual(self.batch.target_model_hidden_states_cache.dtype, torch.float16)
        # Shape: (max_num_reqs, n_predict + 1)
        self.assertEqual(self.batch.target_token_ids_cache.shape, (4, 6))
        self.assertEqual(self.batch.target_token_ids_cache.dtype, torch.int32)

    # ── _make_sampling_metadata ─────────────────────────────────────────

    def test_make_sampling_metadata(self):
        """_make_sampling_metadata attaches npu_input_batch reference."""
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch._make_sampling_metadata',
                   return_value=MagicMock()) as _mock_meta:
            meta = self.batch._make_sampling_metadata()
            self.assertEqual(meta.npu_input_batch, self.batch)

    # ── add_request with penalty cache ─────────────────────────────────

    def test_add_request_with_penalty_cache(self):
        """add_request populates prompt_mask and output_bin_counts via _add_request_for_penalty_cache."""
        self.batch.init_penalty_cache(vocab_size=100, device=torch.device('cpu'))

        req = MagicMock()
        req.prompt_token_ids = [1, 2]
        req.output_token_ids = [3, 4]

        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.add_request',
                   return_value=1):
            idx = self.batch.add_request(req)

        self.assertEqual(idx, 1)
        self.assertTrue(self.batch.prompt_mask[1, 1].item())
        self.assertTrue(self.batch.prompt_mask[1, 2].item())
        self.assertEqual(self.batch.output_bin_counts[1, 3].item(), 1)
        self.assertEqual(self.batch.output_bin_counts[1, 4].item(), 1)

    def test_add_request_with_penalty_cache_resuming(self):
        """add_request restores output-bucket state when a resumed request has output_token_ids."""
        self.batch.init_penalty_cache(vocab_size=100, device=torch.device('cpu'))

        req = MagicMock()
        req.prompt_token_ids = [5]
        req.output_token_ids = [10, 10, 20]  # duplicates to exercise bincount

        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.add_request',
                   return_value=0):
            self.batch.add_request(req)

        self.assertEqual(self.batch.output_bin_counts[0, 10].item(), 2)
        self.assertEqual(self.batch.output_bin_counts[0, 20].item(), 1)
        self.assertTrue(self.batch.output_mask[0, 10].item())
        self.assertTrue(self.batch.output_mask[0, 20].item())

    # ── add_request with multi-MTP cache ───────────────────────────────

    def test_add_request_with_multi_mtp_cache(self):
        """add_request zeroes target cache tensors for the new request slot."""
        self.batch.init_target_model_hidden_states_cache(
            n_predict=3, hidden_size=8, dtype=torch.float32, device=torch.device('cpu'),
        )
        # Fill with non-zero data to verify zeroing
        self.batch.target_model_hidden_states_cache[0] = torch.ones(4, 8)
        self.batch.target_token_ids_cache[0] = torch.full((4,), 42, dtype=torch.int32)

        req = MagicMock()

        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.add_request',
                   return_value=0):
            self.batch.add_request(req)

        self.assertTrue(
            torch.all(self.batch.target_model_hidden_states_cache[0] == 0))
        self.assertTrue(
            torch.all(self.batch.target_token_ids_cache[0] == 0))

    # ── add_request with both caches enabled ───────────────────────────

    def test_add_request_with_both_caches(self):
        """add_request handles both penalty and multi-MTP caches simultaneously."""
        self.batch.init_penalty_cache(vocab_size=50, device=torch.device('cpu'))
        self.batch.init_target_model_hidden_states_cache(
            n_predict=2, hidden_size=4, dtype=torch.float32, device=torch.device('cpu'),
        )

        self.batch.target_model_hidden_states_cache[2] = torch.ones(3, 4)
        self.batch.target_token_ids_cache[2] = torch.full((3,), 99, dtype=torch.int32)

        req = MagicMock()
        req.prompt_token_ids = [10]
        req.output_token_ids = []

        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.add_request',
                   return_value=2):
            self.batch.add_request(req)

        # Penalty cache side
        self.assertTrue(self.batch.prompt_mask[2, 10].item())
        # Multi-MTP cache side — zeroed
        self.assertTrue(
            torch.all(self.batch.target_model_hidden_states_cache[2] == 0))
        self.assertTrue(
            torch.all(self.batch.target_token_ids_cache[2] == 0))

    # ── update_sampled_tokens ──────────────────────────────────────────

    def test_update_sampled_tokens(self):
        """update_sampled_tokens increments output_bin_counts and sets output_mask."""
        self.batch.init_penalty_cache(vocab_size=100, device=torch.device('cpu'))

        # Edge cases: None / empty
        self.batch.update_sampled_tokens(None)
        self.batch.update_sampled_tokens(torch.tensor([]))

        self.batch.update_sampled_tokens(
            torch.tensor([2, -1, 4], dtype=torch.int64))
        self.assertEqual(self.batch.output_bin_counts[0, 2].item(), 1)
        self.assertEqual(self.batch.output_bin_counts[2, 4].item(), 1)
        self.assertTrue(self.batch.output_mask[0, 2].item())
        self.assertTrue(self.batch.output_mask[2, 4].item())

    # ── condense ───────────────────────────────────────────────────────

    def test_condense_with_penalty_cache(self):
        """condense moves penalty tensors when slots are compacted."""
        self.batch.init_penalty_cache(vocab_size=100, device=torch.device('cpu'))
        # Put data into slot 1, slot 0 is the "empty" target
        self.batch.prompt_mask[1, :] = True
        self.batch.output_mask[1, 5] = True
        self.batch.output_bin_counts[1, 5] = 3

        # Start with empty moved; super().condense() will append the move entry
        moved = []
        self.batch.batch_update_builder.moved = moved

        def fake_super_condense():
            moved.append((1, 0, None))

        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.condense',
                   side_effect=fake_super_condense):
            self.batch.condense()

        # Data should have moved from slot 1 → slot 0
        self.assertTrue(self.batch.prompt_mask[0, :].all())
        self.assertEqual(self.batch.output_bin_counts[0, 5].item(), 3)
        # Old slot cleared
        self.assertFalse(self.batch.prompt_mask[1, :].any())
        self.assertEqual(self.batch.output_bin_counts[1, 5].item(), 0)

    def test_condense_with_multi_mtp_cache(self):
        """condense moves multi-MTP cache tensors when slots are compacted."""
        self.batch.init_target_model_hidden_states_cache(
            n_predict=2, hidden_size=4, dtype=torch.float32, device=torch.device('cpu'),
        )
        # Put data into slots 2 → move to 1
        self.batch.target_model_hidden_states_cache[2] = torch.arange(12, dtype=torch.float32).view(3, 4)
        self.batch.target_token_ids_cache[2] = torch.tensor([10, 11, 12], dtype=torch.int32)

        # Start with empty moved; super().condense() will append the move entry
        moved = []
        self.batch.batch_update_builder.moved = moved

        def fake_super_condense():
            moved.append((2, 1, None))

        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.condense',
                   side_effect=fake_super_condense):
            self.batch.condense()
        cache_device = self.batch.target_model_hidden_states_cache.device
        # Data in slot 1 should match former slot 2
        self.assertTrue(torch.equal(
            self.batch.target_model_hidden_states_cache[1],
            torch.arange(12, dtype=torch.float32, device=cache_device).view(3, 4)))
        self.assertTrue(torch.equal(
            self.batch.target_token_ids_cache[1],
            torch.tensor([10, 11, 12], dtype=torch.int32, device=cache_device)))
        # Old slot zeroed
        self.assertTrue(torch.all(self.batch.target_model_hidden_states_cache[2] == 0))
        self.assertTrue(torch.all(self.batch.target_token_ids_cache[2] == 0))

    # ── swap_states ────────────────────────────────────────────────────

    def test_swap_states_with_penalty_cache(self):
        """swap_states exchanges penalty tensors between two slots."""
        self.batch.init_penalty_cache(vocab_size=100, device=torch.device('cpu'))
        self.batch.prompt_mask[0, 0] = True
        self.batch.output_mask[0, 1] = True
        self.batch.output_bin_counts[0, 1] = 5

        self.batch.prompt_mask[2, 2] = True
        self.batch.output_mask[2, 3] = True
        self.batch.output_bin_counts[2, 3] = 9

        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.swap_states'):
            self.batch.swap_states(0, 2)

        self.assertTrue(self.batch.prompt_mask[2, 0].item())
        self.assertTrue(self.batch.output_mask[2, 1].item())
        self.assertEqual(self.batch.output_bin_counts[2, 1].item(), 5)

        self.assertTrue(self.batch.prompt_mask[0, 2].item())
        self.assertTrue(self.batch.output_mask[0, 3].item())
        self.assertEqual(self.batch.output_bin_counts[0, 3].item(), 9)

    def test_swap_states_with_multi_mtp_cache(self):
        """swap_states exchanges multi-MTP cache tensors between two slots."""
        self.batch.init_target_model_hidden_states_cache(
            n_predict=1, hidden_size=4, dtype=torch.float32, device=torch.device('cpu'),
        )
        self.batch.target_model_hidden_states_cache[0] = torch.tensor([[1., 2., 3., 4.]])
        self.batch.target_token_ids_cache[0] = torch.tensor([100], dtype=torch.int32)

        self.batch.target_model_hidden_states_cache[3] = torch.tensor([[5., 6., 7., 8.]])
        self.batch.target_token_ids_cache[3] = torch.tensor([200], dtype=torch.int32)

        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.swap_states'):
            self.batch.swap_states(0, 3)
        cache_device = self.batch.target_model_hidden_states_cache.device
        self.assertTrue(torch.equal(
            self.batch.target_model_hidden_states_cache[0],
            torch.tensor([[5., 6., 7., 8.], [5., 6., 7., 8.]], device=cache_device)))
        self.assertEqual(self.batch.target_token_ids_cache[0, 0].item(), 200)

        self.assertTrue(torch.equal(
            self.batch.target_model_hidden_states_cache[3],
            torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.]], device=cache_device)))
        self.assertEqual(self.batch.target_token_ids_cache[3, 0].item(), 100)

    # ── revert_rejected_tokens (speculative decoding) ──────────────────

    def test_revert_rejected_tokens(self):
        """revert_rejected_tokens handles both 1-D and 2-D masks."""
        self.batch.init_penalty_cache(vocab_size=100, device=torch.device('cpu'))

        if hasattr(self.batch, 'revert_rejected_tokens'):
            accepted_mask = torch.tensor([False, True], dtype=torch.bool)
            token_ids = torch.tensor([1, 2], dtype=torch.int64)
            self.batch.revert_rejected_tokens(accepted_mask, token_ids)

            accepted_mask_2d = torch.tensor(
                [[False], [True]], dtype=torch.bool)
            token_ids_2d = torch.tensor([[1], [2]], dtype=torch.int64)
            self.batch.revert_rejected_tokens(accepted_mask_2d, token_ids_2d)

    # ── guard clauses (disabled caches) ────────────────────────────────

    def test_add_request_penalty_cache_disabled(self):
        """_add_request_for_penalty_cache is a no-op when disable_penalty_cache is True."""
        self.assertTrue(self.batch.disable_penalty_cache)
        req = MagicMock()
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.add_request',
                   return_value=0):
            # Should not raise even though penalty tensors don't exist
            self.batch.add_request(req)

    def test_add_request_multi_mtp_cache_disabled(self):
        """_add_request_for_multi_mtp is a no-op when disable_multi_mtp_cache is True."""
        self.assertTrue(self.batch.disable_multi_mtp_cache)
        req = MagicMock()
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.add_request',
                   return_value=0):
            self.batch.add_request(req)

    def test_condense_penalty_cache_disabled(self):
        """_condense_penalty_cache is a no-op when disable_penalty_cache is True."""
        self.batch.batch_update_builder.moved = [(1, 0, None)]
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.condense'):
            # Should not raise
            self.batch.condense()

    def test_condense_multi_mtp_cache_disabled(self):
        """_condense_target_model_hidden_states_cache is a no-op when disable_multi_mtp_cache is True."""
        self.batch.batch_update_builder.moved = [(1, 0, None)]
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.condense'):
            # Should not raise
            self.batch.condense()

    def test_swap_states_penalty_cache_disabled(self):
        """_swap_states_for_penalty_cache is a no-op when disable_penalty_cache is True."""
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.swap_states'):
            # Should not raise
            self.batch.swap_states(0, 1)

    def test_swap_states_multi_mtp_cache_disabled(self):
        """_swap_states_for_target_model_hidden_states_cache is a no-op when disable_multi_mtp_cache is True."""
        with patch('vllm.v1.worker.gpu_input_batch.InputBatch.swap_states'):
            # Should not raise
            self.batch.swap_states(0, 1)


if __name__ == '__main__':
    unittest.main()
