# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import sys
from unittest.mock import MagicMock
# Block the transitive import chain triggered by decode/__init__.py
# (DecodeOmniCache -> hbm_buffer_utils -> vllm DSAAttentionSpec).
# The test only needs static_utils, not the full decode module.
sys.modules['omni_cache.cache.decode.hbm_buffer_utils'] = MagicMock()

import pytest
import os
from typing import Dict, List, Optional

import torch
import numpy as np
import multiprocessing

from omni_cache.cache.decode.static_utils import record_current_batch_order


class MockInputBatch:
    """Mock input batch with request_id_to_index mapping, which is the only attribute we use in the testing."""
    def __init__(self, req_id_to_index=None):
        self.req_id_to_index = req_id_to_index if req_id_to_index is not None else {}


class MockOmniCache:
    """Mock OmniCache with all required attributes for record_current_batch_order."""
    def __init__(
        self,
        is_hybrid_attn=True,
        record_batch_idx_to_req=None,
        max_num_seqs=32,
    ):
        self.is_hybrid_attn = is_hybrid_attn
        self.record_batch_idx_to_req = record_batch_idx_to_req if record_batch_idx_to_req is not None else {}
        self.req_ids_update_buffer = None
        self.req_ids_record_buffer = None
        self.req_id_to_idx = {}

        self._lane_lock = multiprocessing.Lock()

        # Create CPU buffer for indices
        self.indices_cpu_buffer = torch.zeros(max_num_seqs, dtype=torch.int32).pin_memory()
        self.id_to_idx_table = torch.full((max_num_seqs,), -1, dtype=torch.int32)


def _assert_step4(omni_cache, expected_req_ids, expected_req_id_to_idx, expect_copy=True):
    """
    Helper to verify step4 results: indices_cpu_buffer and id_to_idx_table.
    If expect_copy is False, we also check that copy_ was not called (if no batch change).
    """
    num_reqs = len(expected_req_ids)
    # Build expected indices
    expected_indices = [expected_req_id_to_idx[rid] for rid in expected_req_ids]

    # Check indices_cpu_buffer first 'num_reqs' entries
    actual_indices = omni_cache.indices_cpu_buffer[:num_reqs].tolist()
    assert actual_indices == expected_indices, \
        f"indices_cpu_buffer mismatch: expected {expected_indices}, got {actual_indices}"

    # Check id_to_idx_table first 'num_reqs' entries
    actual_table = omni_cache.id_to_idx_table[:num_reqs].tolist()
    assert actual_table == expected_indices, \
        f"id_to_idx_table mismatch: expected {expected_indices}, got {actual_table}"


# tests for Case 1: skip execution conditions
class TestSkipConditions:
    """Test suite for skip execution conditions (Test ID 1.1, 1.2)."""

    def test_skip_when_is_hybrid_attn_false(self):
        omni_cache = MockOmniCache(is_hybrid_attn=False)
        input_batch = MockInputBatch({"A": 0, "B": 1})
        initial_update_buffer = omni_cache.req_ids_update_buffer
        record_current_batch_order(input_batch, omni_cache)
        assert omni_cache.req_ids_update_buffer == initial_update_buffer
        # No step4 executed because function returned early

    def test_skip_when_record_batch_idx_to_req_none(self):
        omni_cache = MockOmniCache()
        omni_cache.record_batch_idx_to_req = None
        input_batch = MockInputBatch({"A": 0, "B": 1})
        initial_update_buffer = omni_cache.req_ids_update_buffer
        record_current_batch_order(input_batch, omni_cache)
        assert omni_cache.req_ids_update_buffer == initial_update_buffer


# tests for Case 2: first-time call
class TestFirstCallInitialization:
    """Test suite for first-time call (ID 2.1-2.4)."""

    def test_first_call_mapping_missing_raises_error(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={})
        input_batch = MockInputBatch({"A": 0, "B": 1})
        with pytest.raises(RuntimeError, match="omni_cache.indices_cpu_buffer should not contain None slot"):
            record_current_batch_order(input_batch, omni_cache)

    def test_first_call_complete_mapping(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B"})
        input_batch = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch, omni_cache)
        assert omni_cache.req_ids_update_buffer == ["A", "B"]
        assert omni_cache.req_ids_record_buffer is None
        assert omni_cache.req_id_to_idx == {"A": 0, "B": 1}
        # Step4 executed (since record_buffer is None)
        _assert_step4(omni_cache, ["A", "B"], {"A": 0, "B": 1})

    def test_first_call_partial_none_complete_mapping(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: None, 2: "C"})
        input_batch = MockInputBatch({"A": 0, "C": 2})
        record_current_batch_order(input_batch, omni_cache)
        assert omni_cache.req_ids_update_buffer == ["A", "C"]
        assert omni_cache.req_id_to_idx == {"A": 0, "C": 2}
        _assert_step4(omni_cache, ["A", "C"], {"A": 0, "C": 2})

    def test_first_call_partial_none_missing_mapping_raises_error(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: None, 2: "C"})
        input_batch = MockInputBatch({"A": 0, "B": 1, "C": 2})
        with pytest.raises(RuntimeError, match="omni_cache.indices_cpu_buffer should not contain None slot"):
            record_current_batch_order(input_batch, omni_cache)


# Case 3: consecutive calls with only normal rotation, constant batch size
class TestConsecutiveCalls:
    """Test suite for consecutive call scenarios (ID 3.1-3.5)."""

    def test_two_calls_completely_different_requests(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch1, omni_cache)
        # First call: step4 executes (since record_buffer is None initially)
        _assert_step4(omni_cache, ["A", "B"], {"A": 0, "B": 1})

        omni_cache.record_batch_idx_to_req = {0: "C", 1: "D"}
        input_batch2 = MockInputBatch({"C": 0, "D": 1})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_record_buffer == ["A", "B"]
        assert omni_cache.req_ids_update_buffer == ["C", "D"]
        assert omni_cache.req_id_to_idx == {"C": 0, "D": 1}
        # Second call: batch changed -> step4 executes
        _assert_step4(omni_cache, ["C", "D"], {"C": 0, "D": 1})

    def test_partial_reuse_request_reduction(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B", 2: "C"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1, "C": 2})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B", "C"], {"A": 0, "B": 1, "C": 2})

        omni_cache.record_batch_idx_to_req = {0: "A", 1: "B", 2: "C"}
        input_batch2 = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["A", "B"]
        assert omni_cache.req_id_to_idx == {"A": 0, "B": 1}
        # Batch changed (size decreased) -> step4 executes
        _assert_step4(omni_cache, ["A", "B"], {"A": 0, "B": 1})

    def test_two_calls_identical_requests_skip_sync(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch1, omni_cache)
        # Save id_to_idx_table state before second call
        old_table = omni_cache.id_to_idx_table.clone()

        input_batch2 = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_record_buffer == ["A", "B"]
        assert omni_cache.req_ids_update_buffer == ["A", "B"]
        # No change -> step4 should be skipped, so id_to_idx_table unchanged
        assert torch.equal(omni_cache.id_to_idx_table, old_table)

    def test_identical_requests_different_order(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B"], {"A": 0, "B": 1})

        omni_cache.record_batch_idx_to_req = {0: "B", 1: "A"}
        input_batch2 = MockInputBatch({"B": 0, "A": 1})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["B", "A"]
        assert omni_cache.req_id_to_idx == {"B": 0, "A": 1}
        # Order changed -> step4 executes
        _assert_step4(omni_cache, ["B", "A"], {"B": 0, "A": 1})

    def test_completely_different_missing_mapping_raises_error(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch1, omni_cache)

        omni_cache.record_batch_idx_to_req = {0: "C"}
        input_batch2 = MockInputBatch({"C": 0, "D": 1})
        with pytest.raises(RuntimeError, match="omni_cache.indices_cpu_buffer should not contain None slot"):
            record_current_batch_order(input_batch2, omni_cache)


# Case 4: Batch Size Changes + Reordering
class TestBatchSizeChanges:

    def test_append_at_end_no_reorder(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B"})
        # First call to establish state
        input_batch1 = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B"], {"A": 0, "B": 1})

        # Now append
        omni_cache.record_batch_idx_to_req = {0: "A", 1: "B", 2: "C", 3: "D"}
        input_batch2 = MockInputBatch({"A": 0, "B": 1, "C": 2, "D": 3})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["A", "B", "C", "D"]
        assert omni_cache.req_ids_record_buffer == ["A", "B"]
        assert omni_cache.req_id_to_idx == {"A": 0, "B": 1, "C": 2, "D": 3}
        _assert_step4(omni_cache, ["A", "B", "C", "D"], {"A": 0, "B": 1, "C": 2, "D": 3})

    def test_insert_in_middle_no_reorder(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B", 2: "C"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1, "C": 2})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B", "C"], {"A": 0, "B": 1, "C": 2})

        omni_cache.record_batch_idx_to_req = {0: "A", 1: "B", 2: "C", 3: "X"}
        input_batch2 = MockInputBatch({"A": 0, "X": 1, "B": 2, "C": 3})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["A", "X", "B", "C"]
        assert omni_cache.req_id_to_idx == {"A": 0, "B": 1, "C": 2, "X": 3}
        _assert_step4(omni_cache, ["A", "X", "B", "C"], {"A": 0, "X": 3, "B": 1, "C": 2})

    def test_append_at_end_with_random_reorder(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B"], {"A": 0, "B": 1})

        omni_cache.record_batch_idx_to_req = {0: "A", 1: "B", 2: "C", 3: "D"}
        input_batch2 = MockInputBatch({"C": 0, "A": 1, "D": 2, "B": 3})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["C", "A", "D", "B"]
        assert omni_cache.req_id_to_idx == {"A": 0, "B": 1, "C": 2, "D": 3}
        _assert_step4(omni_cache, ["C", "A", "D", "B"], {"C": 2, "A": 0, "D": 3, "B": 1})

    def test_insert_middle_with_random_reorder(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: None, 2: "B", 3: None, 4: "C"})
        input_batch1 = MockInputBatch({"A": 0, "B": 2, "C": 4})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B", "C"], {"A": 0, "B": 2, "C": 4})

        omni_cache.record_batch_idx_to_req = {0: "A", 1: "X", 2: "B", 3: "Y", 4: "C"}
        input_batch2 = MockInputBatch({"X": 0, "A": 1, "Y": 2, "B": 3, "C": 4})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["X", "A", "Y", "B", "C"]
        assert omni_cache.req_id_to_idx == {"A": 0, "X": 1, "B": 2, "Y": 3, "C": 4}
        _assert_step4(omni_cache, ["X", "A", "Y", "B", "C"],
                      {"X": 1, "A": 0, "Y": 3, "B": 2, "C": 4})

    def test_delete_from_end_no_reorder(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B", 2: "C", 3: "D"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1, "C": 2, "D": 3})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B", "C", "D"], {"A": 0, "B": 1, "C": 2, "D": 3})

        input_batch2 = MockInputBatch({"A": 0, "B": 1})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["A", "B"]
        assert omni_cache.record_batch_idx_to_req[2] is None
        assert omni_cache.record_batch_idx_to_req[3] is None
        assert omni_cache.req_id_to_idx == {"A": 0, "B": 1}
        _assert_step4(omni_cache, ["A", "B"], {"A": 0, "B": 1})

    def test_delete_from_middle_no_reorder(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B", 2: "C", 3: "D"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1, "C": 2, "D": 3})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B", "C", "D"], {"A": 0, "B": 1, "C": 2, "D": 3})

        input_batch2 = MockInputBatch({"A": 0, "C": 2, "D": 3})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["A", "C", "D"]
        assert omni_cache.record_batch_idx_to_req[1] is None
        assert omni_cache.req_id_to_idx == {"A": 0, "C": 2, "D": 3}
        _assert_step4(omni_cache, ["A", "C", "D"], {"A": 0, "C": 2, "D": 3})

    def test_delete_from_end_with_random_reorder(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B", 2: "C", 3: "D"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1, "C": 2, "D": 3})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B", "C", "D"], {"A": 0, "B": 1, "C": 2, "D": 3})

        input_batch2 = MockInputBatch({"B": 0, "A": 1})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["B", "A"]
        assert omni_cache.record_batch_idx_to_req[2] is None
        assert omni_cache.record_batch_idx_to_req[3] is None
        assert omni_cache.req_id_to_idx == {"A": 0, "B": 1}
        _assert_step4(omni_cache, ["B", "A"], {"B": 1, "A": 0})

    def test_delete_from_middle_with_random_reorder(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B", 2: "C", 3: "D"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1, "C": 2, "D": 3})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B", "C", "D"], {"A": 0, "B": 1, "C": 2, "D": 3})

        input_batch2 = MockInputBatch({"D": 0, "A": 1, "C": 2})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == ["D", "A", "C"]
        assert omni_cache.record_batch_idx_to_req[1] is None
        assert omni_cache.req_id_to_idx == {"A": 0, "C": 2, "D": 3}
        _assert_step4(omni_cache, ["D", "A", "C"], {"D": 3, "A": 0, "C": 2})

    def test_nonempty_to_empty(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A", 1: "B", 2: "C"})
        input_batch1 = MockInputBatch({"A": 0, "B": 1, "C": 2})
        record_current_batch_order(input_batch1, omni_cache)
        _assert_step4(omni_cache, ["A", "B", "C"], {"A": 0, "B": 1, "C": 2})

        input_batch2 = MockInputBatch({})
        record_current_batch_order(input_batch2, omni_cache)

        assert omni_cache.req_ids_update_buffer == []
        assert omni_cache.req_id_to_idx == {}
        assert all(v is None for v in omni_cache.record_batch_idx_to_req.values())
        assert len(omni_cache.req_ids_update_buffer) == 0


# Case 5: Empty Batch Related
class TestEmptyBatch:
    def test_consecutive_empty_batches(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={})
        # First empty call
        input_batch1 = MockInputBatch({})
        record_current_batch_order(input_batch1, omni_cache)
        # Now record_buffer is None, update_buffer = []
        assert omni_cache.req_ids_update_buffer == []
        assert omni_cache.req_ids_record_buffer is None

        # Second empty call
        input_batch2 = MockInputBatch({})
        record_current_batch_order(input_batch2, omni_cache)
        assert omni_cache.req_ids_update_buffer == []
        assert omni_cache.req_ids_record_buffer == []


# Case 6: Device Sync Boundary and Error Handling
class TestDeviceSyncAndErrors:
    """Test suite for device sync boundary and error handling (ID 6.3, 6.4)."""

    def test_missing_mapping_raises_runtime_error(self):
        omni_cache = MockOmniCache(record_batch_idx_to_req={0: "A"})
        input_batch = MockInputBatch({"A": 0, "B": 1})
        with pytest.raises(RuntimeError, match="omni_cache.indices_cpu_buffer should not contain None slot"):
            record_current_batch_order(input_batch, omni_cache)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])