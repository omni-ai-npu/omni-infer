# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import pytest
from typing import List, Optional
from omni_cache.attention.metadata import resolve_batch_idx_reqs

PAD_SLOT_ID = -1


# Mock class for omni_cache
class MockOmniCache:
    """Mock OmniCache with necessary attributes for resolve_batch_idx_reqs."""
    def __init__(self, req_ids_update_buffer=None, req_id_to_idx=None):
        self.req_ids_update_buffer = req_ids_update_buffer
        self.req_id_to_idx = req_id_to_idx

# Branch A: omni_cache is None
class TestBranchANoneCache:
    def test_cache_is_none(self):
        """Branch A: omni_cache is None -> returns None."""
        assert resolve_batch_idx_reqs(None, 4) is None


# Branch B: missing or empty req_ids_update_buffer
class TestBranchBMissingUpdateBuffer:
    def test_no_update_buffer_attr(self):
        """Branch B: req_ids_update_buffer attribute missing."""
        cache = MockOmniCache()
        del cache.req_ids_update_buffer
        cache.req_id_to_idx = {"a": 0}
        assert resolve_batch_idx_reqs(cache, 4) is None

    def test_update_buffer_empty_list(self):
        """Branch B: req_ids_update_buffer is empty list."""
        cache = MockOmniCache(req_ids_update_buffer=[], req_id_to_idx={"a": 0})
        assert resolve_batch_idx_reqs(cache, 4) is None

    def test_update_buffer_empty_tuple(self):
        """Branch B: req_ids_update_buffer is empty tuple."""
        cache = MockOmniCache(req_id_to_idx={"a": 0})
        cache.req_ids_update_buffer = ()
        assert resolve_batch_idx_reqs(cache, 4) is None

    def test_update_buffer_is_none(self):
        """Branch B: req_ids_update_buffer is None."""
        cache = MockOmniCache(req_ids_update_buffer=None, req_id_to_idx={"a": 0})
        assert resolve_batch_idx_reqs(cache, 4) is None


# Branch C: missing or None req_id_to_idx
class TestBranchCMissingIdToIdx:
    def test_no_id_to_idx_attr(self):
        """Branch C: req_id_to_idx attribute missing."""
        cache = MockOmniCache(req_ids_update_buffer=["a"])
        del cache.req_id_to_idx
        assert resolve_batch_idx_reqs(cache, 1) is None

    def test_id_to_idx_is_none(self):
        """Branch C: req_id_to_idx is None."""
        cache = MockOmniCache(req_ids_update_buffer=["a"], req_id_to_idx=None)
        assert resolve_batch_idx_reqs(cache, 1) is None


# Branch D: normal path (len(buffer) >= num_reqs)
class TestBranchDNormalResolution:
    def test_single_request_exists(self):
        """Branch D: single request, mapping exists."""
        cache = MockOmniCache(req_ids_update_buffer=["r0"], req_id_to_idx={"r0": 5})
        assert resolve_batch_idx_reqs(cache, 1) == [5]

    def test_multiple_all_hit(self):
        """Branch D: multiple requests, all found."""
        cache = MockOmniCache(
            req_ids_update_buffer=["r0", "r1", "r2"],
            req_id_to_idx={"r0": 0, "r1": 1, "r2": 2},
        )
        assert resolve_batch_idx_reqs(cache, 3) == [0, 1, 2]

    def test_multiple_partial_hit(self):
        """Branch D: some requests missing -> pad with -1."""
        cache = MockOmniCache(
            req_ids_update_buffer=["r0", "r1", "r2"],
            req_id_to_idx={"r0": 0, "r2": 2},
        )
        assert resolve_batch_idx_reqs(cache, 3) == [0, PAD_SLOT_ID, 2]

    def test_all_miss(self):
        """Branch D: no mapping found."""
        cache = MockOmniCache(
            req_ids_update_buffer=["r0", "r1"],
            req_id_to_idx={},
        )
        assert resolve_batch_idx_reqs(cache, 2) == [PAD_SLOT_ID, PAD_SLOT_ID]

    def test_dict_subclass(self):
        """Branch D: req_id_to_idx is a dict subclass."""
        from collections import defaultdict
        mapping = defaultdict(lambda: None, {"r0": 10})
        cache = MockOmniCache(req_ids_update_buffer=["r0", "r1"], req_id_to_idx=mapping)
        assert resolve_batch_idx_reqs(cache, 2) == [10, PAD_SLOT_ID]

    def test_no_get_method(self):
        """Branch D: req_id_to_idx has no get method."""
        class NoGet:
            pass
        cache = MockOmniCache(req_ids_update_buffer=["r0"], req_id_to_idx=NoGet())
        assert resolve_batch_idx_reqs(cache, 1) == [PAD_SLOT_ID]

    def test_num_reqs_zero(self):
        """Branch D: num_reqs = 0 -> empty list."""
        cache = MockOmniCache(req_ids_update_buffer=["r0", "r1"], req_id_to_idx={"r0": 0})
        assert resolve_batch_idx_reqs(cache, 0) == []

    def test_num_reqs_equal_buffer_length(self):
        """Branch D: num_reqs exactly equals buffer length."""
        cache = MockOmniCache(
            req_ids_update_buffer=["r0", "r1"],
            req_id_to_idx={"r0": 5, "r1": 6},
        )
        assert resolve_batch_idx_reqs(cache, 2) == [5, 6]


# Branch E: graph mode padding (len(buffer) < num_reqs)
class TestBranchEGraphPadding:
    def test_pad_one_extra(self):
        """Branch E: one extra padding slot."""
        cache = MockOmniCache(
            req_ids_update_buffer=["r0", "r1"],
            req_id_to_idx={"r0": 0, "r1": 1},
        )
        assert resolve_batch_idx_reqs(cache, 3) == [0, 1, PAD_SLOT_ID]

    def test_empty_buffer_with_positive_num_reqs(self):
        """Branch E: buffer empty, num_reqs > 0 -> None."""
        cache = MockOmniCache(req_ids_update_buffer=[], req_id_to_idx={"a": 1})
        assert resolve_batch_idx_reqs(cache, 5) is None

    def test_num_reqs_much_larger(self):
        """Branch E: many padding slots."""
        cache = MockOmniCache(req_ids_update_buffer=["r0"], req_id_to_idx={"r0": 42})
        result = resolve_batch_idx_reqs(cache, 32)
        expected = [42] + [PAD_SLOT_ID] * 31
        assert result == expected
        assert len(result) == 32

    def test_num_reqs_one_with_empty_buffer(self):
        """Branch E: buffer empty, num_reqs=1 -> None."""
        cache = MockOmniCache(req_ids_update_buffer=[], req_id_to_idx={"a": 1})
        assert resolve_batch_idx_reqs(cache, 1) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])