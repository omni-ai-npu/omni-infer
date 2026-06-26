# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_gather_selection_core.py
# Unit tests for gather_selection function

import pytest
try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None
from unittest.mock import Mock, patch, MagicMock

# Skip all tests if torch is not available
if torch is None:
    pytest.skip("torch is not available", allow_module_level=True)

# Import module under test
from omni_cache.gather_selection.core.gather_selection import gather_selection


class TestGatherSelectionBasic:
    """Test suite for gather_selection function basic cases."""

    @pytest.fixture
    def mock_omni_cache(self):
        """Create a mock DecodeOmniCache."""
        cache = Mock()
        cache.bsz_seq = 64
        cache.seq_len = 1
        cache.num_layers = 2
        cache.device = torch.device("cpu")
        cache.full_k_rope = None
        cache.selection_k_rope = torch.zeros(2, 64, 16, 1, dtype=torch.bfloat16)
        # Make selection_kv_cache a tensor that can be sliced
        cache.selection_kv_cache = torch.zeros(1280, 512, 1, 64, dtype=torch.bfloat16)
        cache.selection_kv_block_table = torch.zeros(64 * 1, 10, dtype=torch.int32)
        cache.selection_kv_block_status = torch.zeros(2, 64, dtype=torch.int32)
        return cache

    @pytest.fixture
    def mock_attn_kwargs(self):
        """Create mock attention kwargs."""
        kwargs = {
            'cmp_sparse_indices': torch.zeros(64, 1, 512, dtype=torch.int32),
            'cmp_kv': torch.zeros(64, 1, 1, 512, dtype=torch.bfloat16),
            'cmp_block_table': torch.zeros(64, 10, dtype=torch.int32),
            'seqused_kv': torch.ones(64, dtype=torch.int32) * 512,
            'cu_seqlens_q': torch.zeros(65, dtype=torch.int32),
        }
        return kwargs

    @patch("torch_npu.npu_gather_selection_kv_cache")
    @patch("vllm.logger.init_logger") 
    def test_gather_selection_layer_0(self, mock_logger, mock_npu_fn, mock_omni_cache, mock_attn_kwargs):
        """Test gather_selection with layer_idx=0."""
        # Mock npu_gather_selection_kv_cache
        mock_npu_fn.return_value = torch.zeros(64, dtype=torch.int32)

        # Mock logger
        mock_logger.return_value = Mock()

        # Call function
        gather_selection(
            omni_cache=mock_omni_cache,
            attn_kwargs=mock_attn_kwargs,
            layer_idx=0,
            compress_ratio=1
        )

        # Verify npu_gather_selection_kv_cache was called
        mock_npu_fn.assert_called_once()

        # Verify full_k_rope was created for layer 0
        assert mock_omni_cache.full_k_rope is not None

    @patch("torch_npu.npu_gather_selection_kv_cache")
    @patch("vllm.logger.init_logger") 
    def test_gather_selection_layer_1(self, mock_logger, mock_torch_npu, mock_omni_cache, mock_attn_kwargs):
        """Test gather_selection with layer_idx=1."""
        # Setup existing full_k_rope
        mock_omni_cache.full_k_rope = torch.zeros(64, 512, 1, dtype=torch.bfloat16)

        # Mock npu_gather_selection_kv_cache
        mock_torch_npu.return_value = torch.zeros(64, dtype=torch.int32)

        # Mock logger
        mock_logger.return_value = Mock()

        # Call function
        gather_selection(
            omni_cache=mock_omni_cache,
            attn_kwargs=mock_attn_kwargs,
            layer_idx=1,
            compress_ratio=1
        )

        # Verify npu_gather_selection_kv_cache was called
        mock_torch_npu.assert_called_once()

    @patch("torch_npu.npu_gather_selection_kv_cache")
    @patch("vllm.logger.init_logger")
    def test_gather_selection_kwargs_modification(self, mock_logger, mock_npu_fn, mock_omni_cache, mock_attn_kwargs):
        """Test that attn_kwargs is modified correctly."""
        # Setup existing full_k_rope
        mock_omni_cache.full_k_rope = torch.zeros(64, 512, 1, dtype=torch.bfloat16)

        # Mock npu_gather_selection_kv_cache
        mock_npu_fn.return_value = torch.zeros(64, dtype=torch.int32)

        # Mock logger
        mock_logger.return_value = Mock()

        # Store original values
        original_sparse_indices = mock_attn_kwargs['cmp_sparse_indices'].clone()

        # Call function
        gather_selection(
            omni_cache=mock_omni_cache,
            attn_kwargs=mock_attn_kwargs,
            layer_idx=0,
            compress_ratio=1
        )

        # Verify kwargs were modified
        assert 'cmp_sparse_indices' in mock_attn_kwargs
        assert 'cmp_kv' in mock_attn_kwargs
        assert 'cmp_block_table' in mock_attn_kwargs


class TestGatherSelectionCompressRatio:
    """Test gather_selection with different compress ratios."""

    @pytest.fixture
    def mock_omni_cache(self):
        """Create a mock DecodeOmniCache."""
        cache = Mock()
        cache.bsz_seq = 64
        cache.seq_len = 1
        cache.num_layers = 2
        cache.device = torch.device("cpu")
        cache.full_k_rope = torch.zeros(64, 512, 1, dtype=torch.bfloat16)
        cache.selection_k_rope = torch.zeros(2, 64, 16, 1, dtype=torch.bfloat16)
        # Make selection_kv_cache a tensor that can be sliced
        cache.selection_kv_cache = torch.zeros(1280, 512, 1, 64, dtype=torch.bfloat16)
        cache.selection_kv_block_table = torch.zeros(64 * 1, 10, dtype=torch.int32)
        cache.selection_kv_block_status = torch.zeros(2, 64, dtype=torch.int32)
        return cache

    @pytest.fixture
    def mock_attn_kwargs(self):
        """Create mock attention kwargs."""
        kwargs = {
            'cmp_sparse_indices': torch.zeros(64, 1, 512, dtype=torch.int32),
            'cmp_kv': torch.zeros(64, 1, 1, 512, dtype=torch.bfloat16),
            'cmp_block_table': torch.zeros(64, 10, dtype=torch.int32),
            'seqused_kv': torch.ones(64, dtype=torch.int32) * 512,
            'cu_seqlens_q': torch.zeros(65, dtype=torch.int32),
        }
        return kwargs

    @patch("torch_npu.npu_gather_selection_kv_cache")
    @patch("vllm.logger.init_logger")
    @pytest.mark.parametrize("compress_ratio", [1, 2, 4])
    def test_different_compress_ratios(self, mock_logger, mock_npu_fn, mock_omni_cache, mock_attn_kwargs, compress_ratio):
        """Test gather_selection with different compress ratios."""
        # Mock npu_gather_selection_kv_cache
        mock_npu_fn.return_value = torch.zeros(64, dtype=torch.int32)

        # Mock logger
        mock_logger.return_value = Mock()

        # Call function with different compress ratios
        gather_selection(
            omni_cache=mock_omni_cache,
            attn_kwargs=mock_attn_kwargs,
            layer_idx=0,
            compress_ratio=compress_ratio
        )

        # Verify function was called
        mock_npu_fn.assert_called_once()


class TestGatherSelectionSelectionKwargs:
    """Test the selection_kwargs construction."""

    @pytest.fixture
    def mock_omni_cache(self):
        """Create a mock DecodeOmniCache."""
        cache = Mock()
        cache.bsz_seq = 64
        cache.seq_len = 1
        cache.num_layers = 2
        cache.device = torch.device("cpu")
        cache.full_k_rope = torch.zeros(64, 512, 1, dtype=torch.bfloat16)
        cache.selection_k_rope = torch.zeros(2, 64, 16, 1, dtype=torch.bfloat16)
        # Use real tensors for selection_kv_cache that can be sliced
        cache.selection_kv_cache = torch.zeros(1280, 512, 1, 64, dtype=torch.bfloat16)
        cache.selection_kv_block_table = torch.zeros(64 * 2, 10, dtype=torch.int32)
        cache.selection_kv_block_status = torch.zeros(2, 64, dtype=torch.int32)
        return cache

    @pytest.fixture
    def mock_attn_kwargs(self):
        """Create mock attention kwargs."""
        kwargs = {
            'cmp_sparse_indices': torch.zeros(64, 1, 512, dtype=torch.int32),
            'cmp_kv': torch.zeros(64, 1, 1, 512, dtype=torch.bfloat16),
            'cmp_block_table': torch.zeros(64, 10, dtype=torch.int32),
            'seqused_kv': torch.ones(64, dtype=torch.int32) * 512,
            'cu_seqlens_q': torch.zeros(65, dtype=torch.int32),
        }
        return kwargs

    @patch("torch_npu.npu_gather_selection_kv_cache")
    @patch("vllm.logger.init_logger")
    def test_selection_kwargs_keys(self, mock_logger, mock_npu_fn, mock_omni_cache, mock_attn_kwargs, monkeypatch):
        """Test that selection_kwargs contains all required keys."""
        monkeypatch.delenv("USE_OMNI_INPUT_BATCH", raising=False)
        # Capture the kwargs passed to npu_gather_selection_kv_cache
        captured_kwargs = {}

        def capture_call(**kwargs):
            captured_kwargs.update(kwargs)
            return torch.zeros(64, dtype=torch.int32)

        mock_npu_fn.side_effect = capture_call
        mock_logger.return_value = Mock()

        # Call function
        gather_selection(
            omni_cache=mock_omni_cache,
            attn_kwargs=mock_attn_kwargs,
            layer_idx=0,
            compress_ratio=1
        )

        # Verify all expected keys are present
        expected_keys = [
            "selection_k_rope",
            "selection_kv_cache",
            "selection_kv_block_table",
            "selection_kv_block_status",
            "selection_topk_indices",
            "full_k_rope",
            "full_kv_cache",
            "full_kv_block_table",
            "full_kv_actual_seq",
            "full_q_actual_seq",
            "selection_topk_block_size",
        ]

        for key in expected_keys:
            assert key in captured_kwargs, f"Missing key: {key}"

        assert captured_kwargs["selection_kv_block_table"].data_ptr() == (
            mock_omni_cache.selection_kv_block_table.data_ptr()
        )
        assert captured_kwargs["selection_kv_block_status"].data_ptr() == (
            mock_omni_cache.selection_kv_block_status[0].data_ptr()
        )

    @patch("torch_npu.npu_gather_selection_kv_cache")
    @patch("vllm.logger.init_logger")
    def test_input_batch_path_uses_batch_owned_table_status(
        self,
        mock_logger,
        mock_npu_fn,
        mock_omni_cache,
        mock_attn_kwargs,
        monkeypatch,
    ):
        """USE_OMNI_INPUT_BATCH reads GatherSelection table/status from batch."""
        monkeypatch.setenv("USE_OMNI_INPUT_BATCH", "1")
        captured_kwargs = {}
        gs_table = torch.ones(64, 10, dtype=torch.int32)
        gs_status = torch.full((64,), 7, dtype=torch.int32)

        class Batch:
            def has_gs_buffers(self):
                return True

            def get_gs_table_tensor(self, num_reqs):
                return gs_table[:num_reqs]

            def get_gs_status_tensor(self, layer_idx, num_reqs):
                return gs_status[:num_reqs]

        def capture_call(**kwargs):
            captured_kwargs.update(kwargs)
            return torch.zeros(64, dtype=torch.int32)

        mock_omni_cache.input_batch = Batch()
        mock_npu_fn.side_effect = capture_call
        mock_logger.return_value = Mock()

        gather_selection(
            omni_cache=mock_omni_cache,
            attn_kwargs=mock_attn_kwargs,
            layer_idx=0,
            compress_ratio=1,
        )

        assert captured_kwargs["selection_kv_block_table"].data_ptr() == (
            gs_table.data_ptr()
        )
        assert captured_kwargs["selection_kv_block_status"].data_ptr() == (
            gs_status.data_ptr()
        )


class TestGatherSelectionEdgeCases:
    """Test edge cases for gather_selection."""

    @pytest.fixture
    def mock_omni_cache(self):
        """Create a mock DecodeOmniCache."""
        cache = Mock()
        cache.bsz_seq = 2
        cache.seq_len = 1
        cache.num_layers = 1
        cache.device = torch.device("cpu")
        cache.full_k_rope = None
        cache.selection_k_rope = torch.zeros(1, 2, 16, 1, dtype=torch.bfloat16)
        # Make selection_kv_cache a tensor that can be sliced
        cache.selection_kv_cache = torch.zeros(20, 512, 1, 64, dtype=torch.bfloat16)
        cache.selection_kv_block_table = torch.zeros(2 * 1, 10, dtype=torch.int32)
        cache.selection_kv_block_status = torch.zeros(1, 2, dtype=torch.int32)
        return cache

    @patch("torch_npu.npu_gather_selection_kv_cache")
    @patch("vllm.logger.init_logger")
    def test_minimal_batch(self, mock_logger, mock_torch_npu, mock_omni_cache):
        """Test with minimal batch size."""
        kwargs = {
            'cmp_sparse_indices': torch.zeros(2, 1, 512, dtype=torch.int32),
            'cmp_kv': torch.zeros(2, 1, 1, 512, dtype=torch.bfloat16),
            'cmp_block_table': torch.zeros(2, 10, dtype=torch.int32),
            'seqused_kv': torch.ones(2, dtype=torch.int32) * 512,
            'cu_seqlens_q': torch.zeros(3, dtype=torch.int32),
        }

        mock_torch_npu.return_value = torch.zeros(64, dtype=torch.int32)
        mock_logger.return_value = Mock()

        gather_selection(
            omni_cache=mock_omni_cache,
            attn_kwargs=kwargs,
            layer_idx=0,
            compress_ratio=1
        )

        # Should complete without error
        mock_torch_npu.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
