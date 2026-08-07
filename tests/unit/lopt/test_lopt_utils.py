# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import numpy as np
import pytest
import torch

from omni_npu.lopt.lopt_utils import (
    chunks,
    flatten,
    pairs,
)

# ── chunks ────────────────────────────────────────────────────────────

class TestChunks:
    def test_string_short(self):
        result = list(chunks("hello", chunk_size=100, overlap_length=10))
        assert result == ["hello"]

    def test_string_exact_chunk(self):
        text = "a" * 100
        result = list(chunks(text, chunk_size=50, overlap_length=10))
        assert len(result) >= 1
        assert "".join(result[0][10:]) in text or result[0] == text

    def test_string_long(self):
        text = "a" * 200 + "b" * 10
        result = list(chunks(text, chunk_size=50, overlap_length=5))
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) > 0

    def test_string_empty(self):
        result = list(chunks("", chunk_size=50, overlap_length=10))
        assert result == [""]

    def test_sequence_of_strings_basic(self):
        result = list(chunks(["hello", "world"], chunk_size=3, overlap_length=1))
        assert len(result) >= 1


# ── pairs ─────────────────────────────────────────────────────────────

class TestPairs:
    def test_two_chunks(self):
        result = list(pairs([[1, 2], [3, 4]]))
        assert result == [([1, 2], [3, 4])]

    def test_three_chunks(self):
        result = list(pairs([[1], [2], [3]]))
        assert result == [([1], [2]), ([2], [3])]

    def test_single_chunk(self):
        result = list(pairs([[1]]))
        assert result == []

    def test_empty(self):
        result = list(pairs([]))
        assert result == []


# ── flatten ───────────────────────────────────────────────────────────

class TestFlatten:
    def test_tensor(self):
        t = torch.tensor([[1, 2], [3, 4]])
        result = flatten(t)
        assert torch.equal(result, torch.tensor([1, 2, 3, 4]))

    def test_ndarray(self):
        a = np.array([[1, 2], [3, 4]])
        result = flatten(a)
        assert np.array_equal(result, np.array([1, 2, 3, 4]))

    def test_list(self):
        assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_list_deeply_nested(self):
        assert flatten([[[1, 2], [3, 4]], [[5, 6]]]) == [1, 2, 3, 4, 5, 6]

    def test_already_flat(self):
        assert flatten([1, 2, 3]) == [1, 2, 3]


# ── error paths ───────────────────────────────────────────────────────

class TestErrorPaths:
    def test_chunks_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported type"):
            list(chunks(123, chunk_size=10))

    def test_flatten_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported type"):
            flatten(123)