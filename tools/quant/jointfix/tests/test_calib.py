# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for core.calib_data — the slice/pad logic (no transformers/parquet needed)."""
import pytest

from jointfix.core.calib_data import _texts_to_samples


class _FakeTok:
    """Duck-typed tokenizer: encode() returns a fixed short id list per text."""
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def test_texts_to_samples_shapes():
    samples = _texts_to_samples(["a", "b"], _FakeTok(), n_samples=2, seq_len=4)
    assert len(samples) == 2
    assert all(len(s) == 4 for s in samples)


def test_texts_to_samples_pads_when_short():
    # one tiny text, but we ask for more tokens than it has -> repeats to fill
    samples = _texts_to_samples(["x"], _FakeTok(), n_samples=3, seq_len=5)
    assert len(samples) == 3
    assert all(len(s) == 5 for s in samples)


def test_texts_to_samples_zero():
    assert _texts_to_samples(["a"], _FakeTok(), n_samples=0, seq_len=4) == []


def test_texts_to_samples_empty_raises():
    class _EmptyTok:
        def encode(self, text, add_special_tokens=False):
            return []
    with pytest.raises(ValueError):
        _texts_to_samples(["a"], _EmptyTok(), n_samples=1, seq_len=4)
