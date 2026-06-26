# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_gather_selection_utils.py
# Unit tests for gather_selection utility functions

import pytest
try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None
from unittest.mock import Mock, patch, MagicMock, PropertyMock

# Skip all tests if torch is not available
if torch is None:
    pytest.skip("torch is not available", allow_module_level=True)

# Import module under test
from omni_cache.attention.metadata.block_table import get_block_table_np


class TestGetBlockTableNp:
    """Test suite for get_block_table_np function."""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock runner with vllm_config."""
        runner = Mock()
        runner.vllm_config.scheduler_config.max_num_reqs = 32
        runner.vllm_config.scheduler_config.max_num_seqs = 32
        return runner

    @pytest.fixture
    def mock_omni_cache(self):
        """Create a mock omni_cache with device."""
        cache = Mock()
        cache.device = torch.device("cpu")
        cache.num_spec_token = 5
        # Add blk_table_buffers and slot_mapping_buffers as empty dicts
        cache.blk_table_buffers = {}
        cache.slot_mapping_buffers = {}
        return cache

    @pytest.fixture
    def mock_blk_table(self):
        """Create a mock block table."""
        blk_table = Mock()
        blk_table.get_numpy_array = Mock(return_value=np.zeros((32, 128), dtype=np.int32))
        return blk_table

    @pytest.fixture
    def mock_blk_table_tensor(self):
        """Create a mock block table tensor."""
        tensor = torch.zeros(32, 128, dtype=torch.int32)
        return tensor

    @patch("omni_cache.cache.decode.static_utils.torch_to_numpy_zero_copy")
    def test_get_block_table_np_basic(self, mock_torch_to_np, mock_runner, mock_omni_cache, mock_blk_table, mock_blk_table_tensor):
        """Test basic functionality of get_block_table_np."""
        # Setup mock for torch_to_numpy_zero_copy
        mock_buf = np.zeros((32, 128), dtype=np.int32)
        mock_torch_to_np.return_value = mock_buf

        blk_table_np, slot_mapping_np = get_block_table_np(
            runner=mock_runner,
            omni_cache=mock_omni_cache,
            blk_table=mock_blk_table,
            kv_cache_gid=0,
            blk_table_tensor=mock_blk_table_tensor,
            num_reqs_padded=16,
            num_tokens_padded=32
        )

        # Verify buffers were created
        assert hasattr(mock_omni_cache, 'blk_table_buffers')
        assert 0 in mock_omni_cache.blk_table_buffers

        # Verify slot mapping buffers were created
        assert hasattr(mock_omni_cache, 'slot_mapping_buffers')
        assert 0 in mock_omni_cache.slot_mapping_buffers

        # Verify function returns numpy arrays
        assert isinstance(blk_table_np, np.ndarray)
        assert isinstance(slot_mapping_np, np.ndarray)

    @patch("omni_cache.cache.decode.static_utils.torch_to_numpy_zero_copy")
    def test_get_block_table_np_different_gid(self, mock_torch_to_np, mock_runner, mock_omni_cache, mock_blk_table, mock_blk_table_tensor):
        """Test get_block_table_np with different kv_cache_gid."""
        mock_buf = np.zeros((32, 128), dtype=np.int32)
        mock_torch_to_np.return_value = mock_buf

        # Test with gid=1
        blk_table_np, slot_mapping_np = get_block_table_np(
            runner=mock_runner,
            omni_cache=mock_omni_cache,
            blk_table=mock_blk_table,
            kv_cache_gid=1,
            blk_table_tensor=mock_blk_table_tensor,
            num_reqs_padded=8,
            num_tokens_padded=32
        )

        # Verify buffers were created for gid=1
        assert 1 in mock_omni_cache.blk_table_buffers
        assert 1 in mock_omni_cache.slot_mapping_buffers

    @patch("omni_cache.cache.decode.static_utils.torch_to_numpy_zero_copy")
    def test_get_block_table_np_buffer_reuse(self, mock_torch_to_np, mock_runner, mock_omni_cache, mock_blk_table, mock_blk_table_tensor):
        """Test that buffers are reused when called multiple times with same gid."""
        mock_buf = np.zeros((32, 128), dtype=np.int32)
        mock_torch_to_np.return_value = mock_buf

        # First call
        get_block_table_np(
            runner=mock_runner,
            omni_cache=mock_omni_cache,
            blk_table=mock_blk_table,
            kv_cache_gid=0,
            blk_table_tensor=mock_blk_table_tensor,
            num_reqs_padded=16,
            num_tokens_padded=32
        )

        first_buffer = mock_omni_cache.blk_table_buffers[0]

        # Second call with same gid - should reuse buffer
        get_block_table_np(
            runner=mock_runner,
            omni_cache=mock_omni_cache,
            blk_table=mock_blk_table,
            kv_cache_gid=0,
            blk_table_tensor=mock_blk_table_tensor,
            num_reqs_padded=16,
            num_tokens_padded=32
        )

        second_buffer = mock_omni_cache.blk_table_buffers[0]

        # Buffer should be the same object (reused)
        assert first_buffer is second_buffer

    @patch("omni_cache.cache.decode.static_utils.torch_to_numpy_zero_copy")
    def test_get_block_table_np_value_error(self, mock_torch_to_np, mock_runner, mock_omni_cache, mock_blk_table, mock_blk_table_tensor):
        """Test ValueError when num_tokens_padded exceeds buffer size."""
        # First call for blk_table buffer, second for slot_mapping buffer
        blk_buf = np.zeros((32, 128), dtype=np.int32)
        small_slot_buf = np.zeros((32,), dtype=np.int64)  # Small slot mapping buffer
        mock_torch_to_np.side_effect = [blk_buf, small_slot_buf]

        # This should raise ValueError because num_tokens_padded > buffer size
        with pytest.raises(ValueError, match="num_tokens_padded"):
            get_block_table_np(
                runner=mock_runner,
                omni_cache=mock_omni_cache,
                blk_table=mock_blk_table,
                kv_cache_gid=0,
                blk_table_tensor=mock_blk_table_tensor,
                num_reqs_padded=16,
                num_tokens_padded=100  # Larger than buffer
            )

    @patch("omni_cache.cache.decode.static_utils.torch_to_numpy_zero_copy")
    def test_get_block_table_np_slot_mapping_fill(self, mock_torch_to_np, mock_runner, mock_omni_cache, mock_blk_table, mock_blk_table_tensor):
        """Test that slot_mapping buffer is filled with -1."""

        real_buf = np.zeros((100,128), dtype=np.int64)
        mock_torch_to_np.return_value = real_buf

        get_block_table_np(
            runner=mock_runner,
            omni_cache=mock_omni_cache,
            blk_table=mock_blk_table,
            kv_cache_gid=0,
            blk_table_tensor=mock_blk_table_tensor,
            num_reqs_padded=16,
            num_tokens_padded=32
        )

        # 验证数组被填充为 -1（检查实际结果，而不是 mock 调用）
        assert np.all(real_buf == -1), "Buffer should be filled with -1"


class TestBlockTableEdgeCases:
    """Test edge cases for block table operations."""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock runner."""
        runner = Mock()
        runner.vllm_config.scheduler_config.max_num_reqs = 16
        runner.vllm_config.scheduler_config.max_num_seqs = 16
        return runner

    @pytest.fixture
    def mock_omni_cache(self):
        """Create a mock omni_cache."""
        cache = Mock()
        cache.device = torch.device("cpu")
        cache.num_spec_token = 0
        # Initialize with empty dicts for buffer management
        cache.blk_table_buffers = {}
        cache.slot_mapping_buffers = {}
        return cache

    @patch("omni_cache.cache.decode.static_utils.torch_to_numpy_zero_copy")
    def test_zero_reqs(self, mock_torch_to_np, mock_runner, mock_omni_cache):
        """Test with zero requests."""
        # First call for blk_table buffer, second for slot_mapping buffer
        blk_buf = np.zeros((16, 128), dtype=np.int32)
        slot_buf = np.zeros((16,), dtype=np.int64)
        mock_torch_to_np.side_effect = [blk_buf, slot_buf]

        blk_table = Mock()
        blk_table.get_numpy_array = Mock(return_value=np.zeros((16, 128), dtype=np.int32))

        blk_table_tensor = torch.zeros(16, 128, dtype=torch.int32)

        blk_table_np, slot_mapping_np = get_block_table_np(
            runner=mock_runner,
            omni_cache=mock_omni_cache,
            blk_table=blk_table,
            kv_cache_gid=0,
            blk_table_tensor=blk_table_tensor,
            num_reqs_padded=0,
            num_tokens_padded=0
        )

        # Should return empty arrays
        assert len(blk_table_np) == 0
        assert len(slot_mapping_np) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])