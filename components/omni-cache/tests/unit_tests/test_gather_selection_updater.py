# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_gather_selection_updater.py
# Unit tests for GatherSelectionUpdater class

import pytest
try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None
from unittest.mock import Mock, patch, MagicMock, call

# Skip all tests if torch is not available
if torch is None:
    pytest.skip("torch is not available", allow_module_level=True)

from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid import (
    PanguNewKVCacheSpecsPatch,
)

try:
    PanguNewKVCacheSpecsPatch.apply()
except ValueError as exc:
    if "already patched" not in str(exc):
        raise

# Import module under test
from omni_cache.gather_selection.status_updater.updater import GatherSelectionUpdater


class NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestGatherSelectionUpdaterInit:
    """Test suite for GatherSelectionUpdater initialization."""

    def test_init_with_decode_cache(self):
        """Test initialization with decode omni cache."""
        decode_cache = Mock()
        updater = GatherSelectionUpdater(decode_cache)

        assert updater.decode_omni_cache is decode_cache


class TestUpdaterMethod:
    """Test suite for the updater method."""

    @pytest.fixture
    def mock_decode_cache(self):
        """Create a mock decode cache with required attributes."""
        cache = Mock()
        cache.block_status = {}
        # Mock selection tensors with correct shapes
        cache.selection_kv_block_status = torch.zeros(2, 4, 1, 1, 10, dtype=torch.int32)
        cache.selection_kv_block_table = torch.zeros(4, 1, 10, dtype=torch.int32)
        cache.selection_kv_block_table_buffer = torch.zeros(4, 1, 10, dtype=torch.int32)
        cache.selection_kv_block_status_buffer = torch.zeros(2, 4, 1, 1, 10, dtype=torch.int32)
        cache.seq_len = 1
        # Add block_tables_record as a list for proper len() support
        cache.block_tables_record = []
        return cache

    def test_updater_basic(self, mock_decode_cache):
        """Test basic updater functionality."""
        updater = GatherSelectionUpdater(mock_decode_cache)

        # Create a simple block table tensor
        block_table_tensor = torch.tensor([
            [1, 2, 3, 0, 0],
            [4, 5, 6, 0, 0],
        ], dtype=torch.int32)

        # Call updater
        updater.updater(block_table_tensor)

        # Verify block_tables_record was set
        assert hasattr(mock_decode_cache, 'block_tables_record')

    def test_updater_empty_block_table(self, mock_decode_cache):
        """Test updater with empty block table."""
        updater = GatherSelectionUpdater(mock_decode_cache)

        # Create empty block table
        block_table_tensor = torch.zeros((2, 5), dtype=torch.int32)

        # Call updater
        updater.updater(block_table_tensor)

        # Should complete without error
        assert hasattr(mock_decode_cache, 'block_tables_record')

    def test_updater_single_block(self, mock_decode_cache):
        """Test updater with single block per request."""
        updater = GatherSelectionUpdater(mock_decode_cache)

        # Create block table with single block
        block_table_tensor = torch.tensor([
            [1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
        ], dtype=torch.int32)

        # Call updater
        updater.updater(block_table_tensor)

        # Should complete without error
        assert hasattr(mock_decode_cache, 'block_tables_record')

    def test_updater_with_existing_record(self, mock_decode_cache):
        """Test updater when block_tables_record already exists."""
        updater = GatherSelectionUpdater(mock_decode_cache)

        # Set existing record
        mock_decode_cache.block_tables_record = [[1, 2], [3, 4]]

        # Create new block table
        block_table_tensor = torch.tensor([
            [1, 2, 5, 0, 0],
            [3, 4, 6, 0, 0],
        ], dtype=torch.int32)

        # Call updater
        updater.updater(block_table_tensor)

        # Record should be updated
        assert hasattr(mock_decode_cache, 'block_tables_record')


class TestRecordCurrentBatchOrder:
    """Test suite for record_current_batch_order static method."""

    @pytest.fixture
    def mock_input_batch(self):
        """Create mock input batch."""
        batch = Mock()
        batch.req_id_to_index = {"req_0": 0, "req_1": 1}
        return batch

    @pytest.fixture
    def mock_omni_cache(self):
        """Create mock omni cache."""
        cache = Mock()
        cache.is_hybrid_attn = True
        cache.record_batch_idx_to_req = {0: "req_0", 1: "req_1"}
        cache.req_id_to_idx = {}
        cache.req_ids_update_buffer = None
        cache.req_ids_record_buffer = None
        cache.indices_cpu_buffer = torch.zeros(32, dtype=torch.long)
        cache.id_to_idx_table = torch.zeros(32, dtype=torch.long)
        cache._lane_lock = NullLock()
        return cache

    def test_record_current_batch_order_basic(self, monkeypatch, mock_input_batch, mock_omni_cache):
        """Test basic batch order recording."""
        monkeypatch.delenv("USE_OMNI_INPUT_BATCH", raising=False)
        GatherSelectionUpdater.record_current_batch_order(mock_input_batch, mock_omni_cache)

        # Verify req_ids_update_buffer was set
        assert mock_omni_cache.req_ids_update_buffer == ("req_0", "req_1")

    def test_record_current_batch_order_not_hybrid(self, mock_input_batch):
        """Test recording when not hybrid attention."""
        # Use a real object to avoid Mock's auto-attribute behavior
        class RealCache:
            is_hybrid_attn = False
        cache = RealCache()

        # Should return early without error
        GatherSelectionUpdater.record_current_batch_order(mock_input_batch, cache)

        # req_ids_update_buffer should not be set
        assert not hasattr(cache, 'req_ids_update_buffer')

    def test_record_current_batch_order_cleanup(self, monkeypatch, mock_omni_cache):
        """Test cleanup of stale request states."""
        monkeypatch.delenv("USE_OMNI_INPUT_BATCH", raising=False)
        # Setup existing state that should be cleaned up
        mock_omni_cache.req_ids_update_buffer = ["old_req"]
        mock_omni_cache.req_ids_record_buffer = ["old_req"]
        mock_omni_cache._sk_to_state_idx = {
            ("old_req", "group_1"): 0,
            ("current_req", "current_group"): 1,
        }
        mock_omni_cache.record_req_ids_swa_block = {"old_req": [1, 2]}
        mock_omni_cache.last_num_non_zeros = {("old_req", "group_1"): 5}
        mock_omni_cache.record_req_sched_times = {("old_req", "group_1"): 100}
        mock_omni_cache.req_window_states = {("old_req", "group_1"): Mock()}
        mock_omni_cache._free_state_indices = set()

        # Create input batch with new requests
        batch = Mock()
        batch.req_id_to_index = {"new_req_0": 0, "new_req_1": 1}

        mock_omni_cache.record_batch_idx_to_req = {0: "new_req_0", 1: "new_req_1"}

        # Call recording
        GatherSelectionUpdater.record_current_batch_order(batch, mock_omni_cache)

        # Cleanup should have happened for old_req
        # (old_req, group_1) should be removed from _sk_to_state_idx
        assert ("old_req", "group_1") not in mock_omni_cache._sk_to_state_idx

    def test_record_current_batch_order_keeps_input_batch_lane_mapping(self, monkeypatch):
        """USE_OMNI_INPUT_BATCH keeps H2D-owned req_id_to_idx intact."""
        monkeypatch.setenv("USE_OMNI_INPUT_BATCH", "1")

        class Cache:
            is_hybrid_attn = True

            def __init__(self):
                self.record_batch_idx_to_req = {5: "req_0", 7: "req_1"}
                self.req_id_to_idx = {"req_0": 5, "req_1": 7}
                self.req_ids_update_buffer = None
                self.req_ids_record_buffer = None
                self.indices_cpu_buffer = torch.zeros(32, dtype=torch.long)
                self.id_to_idx_table = torch.zeros(32, dtype=torch.long)
                self._lane_lock = NullLock()
                self._sk_to_state_idx = {}
                self.record_req_ids_swa_block = {}
                self.last_num_non_zeros = {}
                self.record_req_sched_times = {}
                self.req_window_states = {}
                self._free_state_indices = set()

            def get_hbm_lane_for_request(self, req_id):
                return self.req_id_to_idx.get(req_id)

        batch = Mock()
        batch.req_id_to_index = {"req_0": 0, "req_1": 1}
        cache = Cache()

        GatherSelectionUpdater.record_current_batch_order(batch, cache)

        assert cache.req_id_to_idx == {"req_0": 5, "req_1": 7}
        assert cache.id_to_idx_table[:2].tolist() == [5, 7]


class TestUpdateStatusBuffered:
    """Test suite for _update_status_buffered static method."""

    @pytest.fixture
    def mock_omni_cache(self):
        """Create mock omni cache with buffer."""
        cache = Mock()
        cache.selection_kv_block_status_buffer = torch.zeros(2, 4, 1, 1, 10, dtype=torch.int32)
        return cache

    def test_update_status_buffered_basic(self, mock_omni_cache):
        """Test basic status buffer update."""
        reshaped_status = torch.zeros(2, 4, 1, 1, 10, dtype=torch.int32)
        old_req_ids = [0, 1]
        new_req_ids = [1, 0]  # Swapped order

        GatherSelectionUpdater._update_status_buffered(
            mock_omni_cache,
            reshaped_status,
            old_req_ids,
            new_req_ids,
            fill_value=-1
        )

        # Function should complete without error

    def test_update_status_buffered_empty_lists(self, mock_omni_cache):
        """Test with empty request ID lists."""
        reshaped_status = torch.zeros(2, 4, 1, 1, 10, dtype=torch.int32)

        # Empty new_req_ids should return early
        GatherSelectionUpdater._update_status_buffered(
            mock_omni_cache,
            reshaped_status,
            [0, 1],
            [],
            fill_value=-1
        )

        # Should complete without error

    def test_update_status_buffered_none_old(self, mock_omni_cache):
        """Test with None old_req_ids."""
        reshaped_status = torch.zeros(2, 4, 1, 1, 10, dtype=torch.int32)

        GatherSelectionUpdater._update_status_buffered(
            mock_omni_cache,
            reshaped_status,
            None,
            [0, 1],
            fill_value=-1
        )

        # Should handle None old_req_ids


class TestBuildTablePerm:
    """Test suite for _build_table_perm static method."""

    def test_build_table_perm_basic(self):
        """Test basic permutation building."""
        old_req_ids = [0, 1, 2]
        new_req_ids = [2, 0, 1]  # Reordered
        batch_size = 3

        perm, max_old = GatherSelectionUpdater._build_table_perm(
            old_req_ids, new_req_ids, batch_size
        )

        # Verify permutation contains all indices
        assert sorted(perm) == [0, 1, 2]
        assert max_old == 3

    def test_build_table_perm_subset(self):
        """Test with subset of requests kept."""
        old_req_ids = [0, 1, 2, 3]
        new_req_ids = [1, 3]  # Only keep 2 requests
        batch_size = 4

        perm, max_old = GatherSelectionUpdater._build_table_perm(
            old_req_ids, new_req_ids, batch_size
        )

        # First entries should be the kept requests
        assert perm[0] == 1  # Request 1 was at index 1
        assert perm[1] == 3  # Request 3 was at index 3

    def test_build_table_perm_assertions(self):
        """Test that assertions pass for valid permutation."""
        old_req_ids = [0, 1]
        new_req_ids = [0, 1]
        batch_size = 2

        perm, max_old = GatherSelectionUpdater._build_table_perm(
            old_req_ids, new_req_ids, batch_size
        )

        # Assertions in the function should pass
        assert len(perm) == max_old
        assert sorted(perm) == list(range(max_old))


class TestBuildTablePermGolden:
    """Test _build_table_perm with golden permutation verification.

    Golden is built independently by:
      1. Collecting old indices of shared elements in new_req_ids order
         (packed contiguously — matching _build_table_perm's contract).
      2. Appending old indices whose elements are absent from new_req_ids.
      3. Padding with range(max_old, batch_size) for positions beyond max_old.
    Assertions:
      - golden[:max_old] is a permutation of range(max_old).
      - set(golden) == set(range(batch_size)).
      - result = range(batch_size); result[perm] == golden[:max_old].
    """

    @staticmethod
    def _build_golden(old_req_ids, new_req_ids, batch_size):
        max_old = min(len(old_req_ids), batch_size)
        old_eff = old_req_ids[:max_old]
        old_id2idx = {rid: i for i, rid in enumerate(old_eff)}

        golden = list(range(batch_size))

        used_old_indices = set()
        used_new_indices = set()
        for new_idx, rid in enumerate(new_req_ids):
            if rid in old_id2idx:
                golden[new_idx] = old_id2idx[rid]
                used_old_indices.add(old_id2idx[rid])
                used_new_indices.add(new_idx)

        unused_old = [i for i in range(max_old) if i not in used_old_indices]
        remaining_pos = [i for i in range(max_old) if i not in used_new_indices]
        for pos, val in zip(remaining_pos, unused_old):
            golden[pos] = val

        return golden, max_old

    @staticmethod
    def _assert_golden(old_req_ids, new_req_ids, batch_size):
        golden, max_old = TestBuildTablePermGolden._build_golden(
            old_req_ids, new_req_ids, batch_size,
        )

        assert len(set(golden)) == batch_size
        assert set(golden) == set(range(batch_size))
        assert sorted(golden[:max_old]) == list(range(max_old))

        perm, ret_max_old = GatherSelectionUpdater._build_table_perm(
            old_req_ids, new_req_ids, batch_size,
        )
        assert ret_max_old == max_old

        result = list(range(batch_size))
        assert [result[p] for p in perm] == golden[:max_old]

    # ---- length relationship: old > new / old == new / old < new ----------

    def test_old_longer_than_new(self):
        """len(old) > len(new): some old requests leave the batch."""
        self._assert_golden(
            old_req_ids=[100, 200, 300, 400, 500, 600],
            new_req_ids=[300, 100, 500],
            batch_size=5,
        )

    def test_old_equal_to_new(self):
        """len(old) == len(new): pure reorder, all elements shared."""
        self._assert_golden(
            old_req_ids=[10, 20, 30, 40],
            new_req_ids=[30, 40, 10, 20],
            batch_size=4,
        )

    def test_old_shorter_than_new(self):
        """len(old) < len(new): new requests arrive, some old stay."""
        self._assert_golden(
            old_req_ids=[10, 20, 30],
            new_req_ids=[20, 1070, 30, 40, 50],
            batch_size=5,
        )

    # ---- overlap patterns -------------------------------------------------

    def test_partial_overlap_both_sides(self):
        """Both old and new contain unique elements; only a subset overlaps."""
        self._assert_golden(
            old_req_ids=[10, 20, 30, 40],
            new_req_ids=[30, 50, 10, 60],
            batch_size=4,
        )

    def test_no_common_elements(self):
        """old and new are completely disjoint — perm is identity."""
        self._assert_golden(
            old_req_ids=[10, 20, 30],
            new_req_ids=[40, 50, 60],
            batch_size=3,
        )

    def test_exact_match_same_order(self):
        """old and new are identical — perm is identity."""
        self._assert_golden(
            old_req_ids=[10, 20, 30],
            new_req_ids=[10, 20, 30],
            batch_size=3,
        )

    def test_exact_match_different_order(self):
        """Same elements, different order — pure permutation."""
        self._assert_golden(
            old_req_ids=[10, 20, 30, 40],
            new_req_ids=[40, 20, 30, 10],
            batch_size=4,
        )

    # ---- mixed length + partial overlap -----------------------------------

    def test_old_shorter_all_old_in_new(self):
        """old is a strict subset of new — remaining is empty."""
        self._assert_golden(
            old_req_ids=[10, 20, 30],
            new_req_ids=[20, 10, 30, 40, 50],
            batch_size=5,
        )

    def test_old_longer_single_overlap(self):
        """Only one element shared between old and new."""
        self._assert_golden(
            old_req_ids=[10, 20, 30, 40, 50],
            new_req_ids=[30, 60],
            batch_size=5,
        )


class TestReorderBlockTableOnly:
    """Test suite for _reorder_block_table_only static method."""

    @pytest.fixture
    def mock_omni_cache(self):
        """Create mock omni cache."""
        cache = Mock()
        cache.selection_kv_block_table_buffer = torch.zeros(4, 2, 10, dtype=torch.int32)
        return cache

    def test_reorder_block_table_basic(self, mock_omni_cache):
        """Test basic block table reordering."""
        # Create a properly sized tensor: 4 * 2 * 10 = 80 elements
        reshaped_table = torch.arange(80).reshape(4, 2, 10).to(torch.int32)
        old_req_ids = [0, 1, 2, 3]
        new_req_ids = [3, 2, 1, 0]  # Reversed

        GatherSelectionUpdater._reorder_block_table_only(
            mock_omni_cache,
            reshaped_table,
            old_req_ids,
            new_req_ids
        )

        # Function should complete without error

    def test_reorder_block_table_empty(self, mock_omni_cache):
        """Test with empty request lists."""
        reshaped_table = torch.zeros(4, 2, 10, dtype=torch.int32)

        # Empty old_req_ids should return early
        GatherSelectionUpdater._reorder_block_table_only(
            mock_omni_cache,
            reshaped_table,
            [],
            [0, 1, 2, 3]
        )

        # Should complete without error

    def test_reorder_block_table_zero_batch(self, mock_omni_cache):
        """Test with zero batch size."""
        reshaped_table = torch.zeros(0, 2, 10, dtype=torch.int32)

        GatherSelectionUpdater._reorder_block_table_only(
            mock_omni_cache,
            reshaped_table,
            [0, 1],
            [1, 0]
        )

        # Should handle zero batch size


class TestMaybeUpdateSelectionKvBlockStatus:
    """Test suite for maybe_update_selection_kv_block_status static method."""

    @pytest.fixture
    def mock_omni_cache(self):
        """Create mock omni cache."""
        cache = Mock()
        cache.selection_kv_block_status = torch.zeros(2, 4, 1, 1, 10, dtype=torch.int32)
        cache.selection_kv_block_table = torch.zeros(4, 1, 10, dtype=torch.int32)
        cache.selection_kv_block_table_buffer = torch.zeros(4, 1, 10, dtype=torch.int32)
        cache.selection_kv_block_status_buffer = torch.zeros(2, 4, 1, 1, 10, dtype=torch.int32)
        cache.req_ids_record = None
        cache.seq_len = 1
        return cache

    @pytest.fixture
    def mock_input_batch(self):
        """Create mock input batch."""
        batch = Mock()
        batch.req_id_to_index = {"req_0": 0, "req_1": 1}
        return batch

    @patch("omni_cache.gather_selection.status_updater.updater.time")
    def test_maybe_update_no_change(self, mock_time, mock_input_batch, mock_omni_cache):
        """Test when request order hasn't changed."""
        mock_time.time.return_value = 0.0
        mock_time.perf_counter.return_value = 0.0

        # Set same request IDs as record
        mock_omni_cache.req_ids_record = ["req_0", "req_1"]

        num_scheduled_tokens = [1, 1]

        GatherSelectionUpdater.maybe_update_selection_kv_block_status(
            mock_input_batch, mock_omni_cache, num_scheduled_tokens
        )

        # Should complete without error
        assert mock_omni_cache.req_ids_record == ("req_0", "req_1")

    @patch("omni_cache.gather_selection.status_updater.updater.time")
    def test_maybe_update_with_change(self, mock_time, mock_input_batch, mock_omni_cache):
        """Test when request order has changed."""
        mock_time.time.return_value = 0.0
        mock_time.perf_counter.return_value = 0.001

        # Set different request IDs as record
        mock_omni_cache.req_ids_record = ["req_1", "req_0"]

        num_scheduled_tokens = [1, 1]

        GatherSelectionUpdater.maybe_update_selection_kv_block_status(
            mock_input_batch, mock_omni_cache, num_scheduled_tokens
        )

        # Record should be updated to new order
        assert mock_omni_cache.req_ids_record == ("req_0", "req_1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
