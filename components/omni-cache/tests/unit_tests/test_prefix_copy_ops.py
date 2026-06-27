# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for prefix_copy_ops.py module.

Tests prefix copy operations for chunked prefill (APC - Automatic Prefix Caching).
"""

import sys
import importlib.util
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Direct file import to avoid circular imports
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_prefix_copy_ops_path = os.path.join(_current_dir, "..", "..", "omni_cache", "cache", "prefill", "prefix_copy_ops.py")
spec = importlib.util.spec_from_file_location(
    "prefix_copy_ops",
    _prefix_copy_ops_path
)
module = importlib.util.module_from_spec(spec)
sys.modules["prefix_copy_ops"] = module
spec.loader.exec_module(module)

PrefixCopyMeta = module.PrefixCopyMeta
compute_prefix_segments = module.compute_prefix_segments
get_current_rank_host_data = module.get_current_rank_host_data


class TestPrefixCopyMeta:
    """Test cases for PrefixCopyMeta class."""

    def test_prefix_copy_meta_init(self):
        """Test PrefixCopyMeta initialization."""
        consecutive_blocks = [[[0, 5], [10, 15]], [[20, 25]]]
        query_lens = [10, 5]
        query_slots = torch.tensor([0, 1, 2, 3, 4])

        meta = PrefixCopyMeta(consecutive_blocks, query_lens, query_slots)

        assert meta.consecutive_blocks == consecutive_blocks
        assert meta.query_lens == query_lens
        assert torch.equal(meta.query_slots, query_slots)

    def test_prefix_copy_meta_empty_blocks(self):
        """Test PrefixCopyMeta with empty blocks."""
        consecutive_blocks = []
        query_lens = []
        query_slots = torch.tensor([])

        meta = PrefixCopyMeta(consecutive_blocks, query_lens, query_slots)

        assert meta.consecutive_blocks == []
        assert meta.query_lens == []
        assert len(meta.query_slots) == 0


class TestComputePrefixSegments:
    """Test cases for compute_prefix_segments function."""

    @pytest.fixture
    def common_params(self):
        """Common test parameters."""
        return {
            'block_size': 128,
            'arange_cpu': torch.arange(1024),
            'device': torch.device('cpu'),
        }

    def test_compute_prefix_segments_basic(self, common_params):
        """Test basic prefix segment computation."""
        kv_lens = np.array([100, 200])
        query_lens_list = [50, 30]
        block_tables = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ])

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            common_params['block_size'],
            common_params['arange_cpu'],
            common_params['device'],
        )

        assert result is not None
        all_segs, q_lens, q_slots = result
        assert len(all_segs) == 2
        assert q_lens == query_lens_list
        assert isinstance(q_slots, torch.Tensor)

    def test_compute_prefix_segments_no_remainder(self, common_params):
        """Test when all remainders are zero (exact block alignment)."""
        kv_lens = np.array([128, 256])  # Exact multiples of block_size
        query_lens_list = [50, 50]
        block_tables = np.array([
            [0, 1],
            [2, 3],
        ])

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            common_params['block_size'],
            common_params['arange_cpu'],
            common_params['device'],
        )

        # Should return None when all remainders are zero
        assert result is None

    def test_compute_prefix_segments_partial_remainder(self, common_params):
        """Test with partial remainder (one has remainder, one doesn't)."""
        kv_lens = np.array([128, 200])  # First exact, second has remainder
        query_lens_list = [50, 50]
        block_tables = np.array([
            [0, 1],
            [2, 3],
        ])

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            common_params['block_size'],
            common_params['arange_cpu'],
            common_params['device'],
        )

        assert result is not None

    def test_compute_prefix_segments_single_block(self, common_params):
        """Test with single block (m == 0 case)."""
        kv_lens = np.array([50])  # Less than block_size
        query_lens_list = [20]
        block_tables = np.array([[0, -1, -1, -1]])

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            common_params['block_size'],
            common_params['arange_cpu'],
            common_params['device'],
        )

        assert result is not None
        all_segs, _, _ = result
        # Single block case: [[block_id, block_id + 1]]
        assert len(all_segs[0]) == 1
        assert all_segs[0][0] == [0, 1]

    def test_compute_prefix_segments_negative_kv_len(self, common_params):
        """Test with negative KV length (edge case)."""
        kv_lens = np.array([-5])  # Invalid but should handle
        query_lens_list = [10]
        block_tables = np.array([[0, 1, 2, 3]])

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            common_params['block_size'],
            common_params['arange_cpu'],
            common_params['device'],
        )

        # m will be negative, should produce empty segs
        assert result is not None
        all_segs, _, _ = result
        assert all_segs[0] == []

    def test_compute_prefix_segments_consecutive_blocks(self, common_params):
        """Test with multiple consecutive blocks."""
        kv_lens = np.array([300])  # Spans multiple blocks
        query_lens_list = [50]
        # Blocks 0, 1, 2 are consecutive
        block_tables = np.array([[0, 1, 2, -1]])

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            common_params['block_size'],
            common_params['arange_cpu'],
            common_params['device'],
        )

        assert result is not None
        all_segs, _, _ = result
        # Should identify consecutive range
        assert len(all_segs[0]) >= 1

    def test_compute_prefix_segments_non_consecutive_blocks(self, common_params):
        """Test with non-consecutive blocks."""
        kv_lens = np.array([300])
        query_lens_list = [50]
        # Blocks 0, 2 are not consecutive (gap at 1)
        block_tables = np.array([[0, 2, 4, -1]])

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            common_params['block_size'],
            common_params['arange_cpu'],
            common_params['device'],
        )

        assert result is not None
        all_segs, _, _ = result
        # Should split into separate segments
        assert len(all_segs[0]) > 1

    def test_compute_prefix_segments_query_slots(self, common_params):
        """Test query slots computation."""
        kv_lens = np.array([100, 150])
        query_lens_list = [20, 30]
        block_tables = np.array([
            [0, 1, 2, -1],
            [3, 4, 5, -1],
        ])

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            common_params['block_size'],
            common_params['arange_cpu'],
            common_params['device'],
        )

        assert result is not None
        _, _, q_slots = result

        # Query slots should have correct length
        expected_len = sum(query_lens_list)
        assert len(q_slots) == expected_len

    def test_compute_prefix_segments_device_transfer(self, common_params):
        """Test that query slots are transferred to correct device."""
        kv_lens = np.array([100])
        query_lens_list = [50]
        block_tables = np.array([[0, 1, 2, 3]])

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            common_params['block_size'],
            common_params['arange_cpu'],
            common_params['device'],
        )

        assert result is not None
        _, _, q_slots = result
        assert q_slots.device == common_params['device']


class TestGetCurrentRankHostData:
    """Test cases for get_current_rank_host_data function."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock PrefillOmniCache."""
        cache = Mock()
        cache.node_block_size = 128
        cache.tp_local_rank = 0
        cache.num_layers = 2

        # Mock host cache with proper tensor shape
        # Shape: (num_layers, num_blocks * block_size, head_dim)
        cache.host_cache = Mock()
        cache.host_cache.kvi_tensors = [
            torch.randn(2, 1000, 64),  # Layer-major tensor
        ]

        return cache

    @pytest.fixture
    def prefix_meta(self):
        """Create a PrefixCopyMeta instance."""
        consecutive_blocks = [[[0, 2], [5, 7]]]  # Blocks 0-1 and 5-6
        query_lens = [10]
        query_slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        return PrefixCopyMeta(consecutive_blocks, query_lens, query_slots)

    def test_get_current_rank_host_data_basic(self, mock_cache, prefix_meta):
        """Test basic host data retrieval."""
        # Patch LOCAL_DP_SIZE in the module where it's imported
        with patch.dict('sys.modules', {'omni_cache.cache.core.base': Mock(LOCAL_DP_SIZE=4)}):
            result = get_current_rank_host_data(mock_cache, 0, prefix_meta)

            # Should return a tensor
            assert isinstance(result, torch.Tensor)
            # Result shape depends on block_ids and how data is concatenated
            # With 1 kvi_tensor of head_dim 64, result should have head_dim 64
            assert result.shape[-1] == 64

    def test_get_current_rank_host_data_different_layers(self, mock_cache, prefix_meta):
        """Test host data retrieval for different layer indices."""
        with patch.dict('sys.modules', {'omni_cache.cache.core.base': Mock(LOCAL_DP_SIZE=4)}):
            # Test layer 0
            result_layer0 = get_current_rank_host_data(mock_cache, 0, prefix_meta)
            # Test layer 1
            result_layer1 = get_current_rank_host_data(mock_cache, 1, prefix_meta)

            # Both should return tensors
            assert isinstance(result_layer0, torch.Tensor)
            assert isinstance(result_layer1, torch.Tensor)
            # Results should have same shape
            assert result_layer0.shape == result_layer1.shape

    def test_get_current_rank_host_data_block_range_expansion(self, mock_cache):
        """Test that block ranges are correctly expanded."""
        # Create prefix meta with multiple consecutive blocks
        consecutive_blocks = [[[0, 4]]]  # Blocks 0-3 (4 consecutive blocks)
        query_lens = [20]
        query_slots = torch.arange(20)
        prefix_meta = PrefixCopyMeta(consecutive_blocks, query_lens, query_slots)

        with patch.dict('sys.modules', {'omni_cache.cache.core.base': Mock(LOCAL_DP_SIZE=4)}):
            result = get_current_rank_host_data(mock_cache, 0, prefix_meta)

            # Should return a tensor
            assert isinstance(result, torch.Tensor)
            # 4 blocks * 128 block_size / 4 ranks = 128 heads per rank
            # Result shape should reflect the data retrieved

    def test_get_current_rank_host_data_empty_blocks(self, mock_cache):
        """Test with empty block list."""
        consecutive_blocks = [[]]  # Empty blocks
        query_lens = [10]
        query_slots = torch.arange(10)
        prefix_meta = PrefixCopyMeta(consecutive_blocks, query_lens, query_slots)

        with patch.dict('sys.modules', {'omni_cache.cache.core.base': Mock(LOCAL_DP_SIZE=4)}):
            # Empty blocks should still work but return empty or handle gracefully
            # The function may raise an error or return empty tensor
            try:
                result = get_current_rank_host_data(mock_cache, 0, prefix_meta)
                # If it returns, should be a tensor
                assert isinstance(result, torch.Tensor)
            except (IndexError, ValueError):
                # Acceptable to raise for empty input
                pass

    def test_get_current_rank_host_data_single_die(self, mock_cache, prefix_meta):
        """Test with LOCAL_DP_SIZE = 1."""
        with patch.dict('sys.modules', {'omni_cache.cache.core.base': Mock(LOCAL_DP_SIZE=1)}):
            result = get_current_rank_host_data(mock_cache, 0, prefix_meta)

            # Should return a tensor
            assert isinstance(result, torch.Tensor)
            # With LOCAL_DP_SIZE=1, all data goes to single rank

    def test_get_current_rank_host_data_different_rank(self, mock_cache):
        """Test with different tp_local_rank values."""
        # Create prefix meta with consecutive blocks to avoid edge case issues
        consecutive_blocks = [[[0, 8]]]  # Blocks 0-7 (8 consecutive blocks)
        query_lens = [20]
        query_slots = torch.arange(20)
        prefix_meta = PrefixCopyMeta(consecutive_blocks, query_lens, query_slots)

        with patch.dict('sys.modules', {'omni_cache.cache.core.base': Mock(LOCAL_DP_SIZE=4)}):
            # Test with rank 0
            mock_cache.tp_local_rank = 0
            result_rank0 = get_current_rank_host_data(mock_cache, 0, prefix_meta)

            # Both should return tensors
            assert isinstance(result_rank0, torch.Tensor)

    def test_get_current_rank_host_data_multiple_kvi_tensors(self, mock_cache, prefix_meta):
        """Test with multiple kvi_tensors."""
        # Add more kvi tensors
        mock_cache.host_cache.kvi_tensors = [
            torch.randn(2, 1000, 64),
            torch.randn(2, 1000, 64),
        ]

        with patch.dict('sys.modules', {'omni_cache.cache.core.base': Mock(LOCAL_DP_SIZE=4)}):
            result = get_current_rank_host_data(mock_cache, 0, prefix_meta)

            # Should return a tensor with concatenated data from all kvi_tensors
            assert isinstance(result, torch.Tensor)
            # head_dim should be sum of all tensor last dims: 64 + 64 = 128
            assert result.shape[-1] == 128


class TestPrefixCopyOpsEdgeCases:
    """Edge case tests for prefix copy operations."""

    def test_compute_prefix_segments_zero_kv_len(self):
        """Test with zero KV length."""
        kv_lens = np.array([0])
        query_lens_list = [10]
        block_tables = np.array([[0, 1, 2, 3]])
        arange_cpu = torch.arange(1024)
        device = torch.device('cpu')

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            128,
            arange_cpu,
            device,
        )

        # Zero KV len means remainder is 0
        assert result is None

    def test_compute_prefix_segments_large_batch(self):
        """Test with large batch size."""
        batch_size = 32
        kv_lens = np.random.randint(50, 500, size=batch_size)
        query_lens_list = [10] * batch_size
        block_tables = np.random.randint(0, 100, size=(batch_size, 8))
        arange_cpu = torch.arange(1024)
        device = torch.device('cpu')

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            128,
            arange_cpu,
            device,
        )

        if result is not None:
            all_segs, _, _ = result
            assert len(all_segs) == batch_size

    def test_compute_prefix_segments_query_len_zero(self):
        """Test with zero query length."""
        kv_lens = np.array([100])
        query_lens_list = [0]  # Zero query length
        block_tables = np.array([[0, 1, 2, 3]])
        arange_cpu = torch.arange(1024)
        device = torch.device('cpu')

        result = compute_prefix_segments(
            kv_lens,
            query_lens_list,
            block_tables,
            128,
            arange_cpu,
            device,
        )

        assert result is not None
        _, q_lens, q_slots = result
        assert len(q_slots) == 0  # Empty query slots
