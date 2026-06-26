# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for _prepare_blocks in prefill.py."""

import sys
from typing import List
from unittest.mock import MagicMock, Mock

import pytest

# ---------------------------------------------------------------------------
# Mock torch before loading the source
# ---------------------------------------------------------------------------
_mock_torch = MagicMock(name="torch")
_mock_torch.int32 = "int32"
_mock_torch.int64 = "int64"

_omni_cache_mod = MagicMock(name="omni_cache.cache")
_omni_cache_mod.omni_cache = Mock()
_omni_cache_mod.omni_cache.num_computed_tokens = [0]

_globals = {"__builtins__": __builtins__}
with open("omni_cache/cache/transfer_engine/prefill.py") as f:
    code = f.read()

_extra_modules = {
    "torch": _mock_torch,
    "torch.npu": MagicMock(),
    "omni_cache": MagicMock(),
    "omni_cache.cache": _omni_cache_mod,
    "vllm": MagicMock(),
    "vllm.logger": MagicMock(),
    "vllm.v1": MagicMock(),
    "vllm.v1.kv_cache_interface": MagicMock(),
    "vllm.config": MagicMock(),
}

# Inject mocks permanently (not via context manager) so that lazy
# `from omni_cache.cache import ...` inside _prepare_blocks resolves
# at call time.
sys.modules.update(_extra_modules)
exec(compile(code, "prefill.py", "exec"), _globals)

_prepare_blocks = _globals["_prepare_blocks"]


# ---------------------------------------------------------------------------
# Lightweight fake tensor
# ---------------------------------------------------------------------------
class _FakeTensor:
    """List-backed object that behaves like a 1D/2D int tensor."""

    def __init__(self, data, device="npu:0"):
        if not isinstance(data, list):
            data = [data]
        self._data = data
        self.device = device
        if data and isinstance(data[0], list):
            self._shape = (len(data), len(data[0]))
        else:
            self._shape = (len(data),)

    shape = property(lambda self: self._shape)

    def dim(self):
        return len(self._shape)

    def detach(self):
        return self

    def cpu(self):
        return _FakeTensor(self._data, device="cpu")

    def tolist(self):
        return self._data

    def to(self, *a, **kw):
        return self

    def is_floating_point(self):
        return False

    def long(self):
        return self

    def unsqueeze(self, dim):
        return _FakeTensor([self._data], device=self.device)

    def view(self, *a):
        return self

    def expand(self, size):
        return self

    def clone(self):
        return _FakeTensor(self._data, device=self.device)

    def squeeze(self, dim=None):
        return self

    def copy_(self, src, non_blocking=None):
        pass

    def __getitem__(self, key):
        # Boolean mask indexing: keep elements where mask is True
        if isinstance(key, _FakeTensor) and key._data:
            result = []
            mask_d = key._data
            src_d = self._data
            if mask_d and isinstance(mask_d[0], list):
                # 2D mask indexing 2D source
                for ri, mr in enumerate(mask_d):
                    for ci, flag in enumerate(mr):
                        if flag:
                            result.append(src_d[ri][ci])
            else:
                # 1D mask indexing 1D source
                for i, flag in enumerate(mask_d):
                    if flag:
                        result.append(src_d[i])
            return _FakeTensor(result, self.device)
        if isinstance(key, tuple):
            row, col = key
            if isinstance(col, slice):
                sub = self._data[row][col]
                return _FakeTensor(sub if isinstance(sub, list) else [sub], self.device)
            return _make_tensor(self._data[row][col], self.device)
        return _FakeTensor(self._data[key], self.device)

    def __setitem__(self, key, value):
        pass

    def __int__(self):
        v = self._data[0]
        if isinstance(v, list):
            v = v[0]
        return int(v)

    def __len__(self):
        return self._shape[0]

    def __iter__(self):
        return iter(self._data)

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        """Return a boolean mask tensor with same shape as self."""
        if self._data and isinstance(self._data[0], list):
            # 2D
            mask_data = [[v != other for v in row] for row in self._data]
        else:
            # 1D
            mask_data = [v != other for v in self._data]
        return _FakeTensor(mask_data, self.device)

    def __hash__(self):
        return id(self)

    def __floordiv__(self, other):
        if isinstance(other, _FakeTensor):
            other = other.item()
        vals = [int(v) // other for v in self._data]
        return _FakeTensor(vals, self.device)

    def item(self):
        v = self._data[0]
        if isinstance(v, list):
            return v[0]
        return v

    def to(self, *a, **kw):
        return self

    def new_zeros(self, shape):
        return _FakeTensor([], self.device)


def _t(data, dtype=None, device="npu:0"):
    if isinstance(data, _FakeTensor):
        return data
    return _FakeTensor(list(data) if isinstance(data, (list, tuple)) else [data], device)

_mock_torch.tensor = _t
_mock_torch.Tensor = _FakeTensor
_mock_torch.empty = Mock(return_value=_t([], device="cpu"))
def _fake_cat(tensors, dim=0):
    """Concatenate FakeTensors along the given dimension."""
    result_data = []
    for t in tensors:
        d = t._data
        result_data.append(d if isinstance(d[0], list) else d)
    # Horizontal cat: merge row by row
    combined = []
    for row_idx in range(len(result_data[0])):
        combined_row = []
        for d in result_data:
            combined_row.extend(d[row_idx] if isinstance(d[row_idx], list) else [d[row_idx]])
        combined.append(combined_row)
    return _FakeTensor(combined)
_mock_torch.cat = _fake_cat


# ---------------------------------------------------------------------------
# Fixture: reset num_computed_tokens before each test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_num_comp():
    _omni_cache_mod.omni_cache.num_computed_tokens = [0]
    yield


# ---------------------------------------------------------------------------
# Test helpers — use spec=[] to prevent Mock auto-creating attrs
# ---------------------------------------------------------------------------
def _meta(**kwargs):
    """Build attn_meta mock with ONLY the given attrs."""
    m = Mock(spec=[])
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


def _spec(block_size=128):
    s = Mock(spec=[])
    s.block_size = block_size
    return s


# ---------------------------------------------------------------------------
# Attention path
# ---------------------------------------------------------------------------
class TestAttentionPath:
    def test_raises_when_seq_lens_missing(self):
        m = _meta()
        with pytest.raises(RuntimeError, match="seq_lens"):
            _prepare_blocks(m, "npu:0", _spec(128))

    def test_raises_when_block_size_missing(self):
        m = _meta(seq_lens=_t([10]))
        with pytest.raises(RuntimeError, match="block_size"):
            _prepare_blocks(m, "npu:0", None)

    def test_raises_when_num_comp_tokens_missing(self):
        _omni_cache_mod.omni_cache.num_computed_tokens = None
        m = _meta(seq_lens=_t([10]))
        with pytest.raises(ValueError, match="num_computed_tokens"):
            _prepare_blocks(m, "npu:0", _spec(128))

    def test_empty_block_table_returns_empty(self):
        m = _meta(seq_lens=_t([10]), block_table=None)
        cpu, npu = _prepare_blocks(m, "npu:0", _spec(128))
        assert cpu.tolist() == []

    def test_skip_when_start_ge_total(self):
        """Already offloaded blocks are excluded; num_comp=256 >= seq_len=128."""
        _omni_cache_mod.omni_cache.num_computed_tokens = _t([256])
        m = _meta(seq_lens=_t([128]), block_table=_t([[5, 0]]))
        cpu, npu = _prepare_blocks(m, "npu:0", _spec(128))
        assert cpu.tolist() == []  # start_b=2 >= total_blocks=1 → skip

    def test_scalar_num_comp_expanded(self):
        """Single int num_computed_tokens: starts from block 0."""
        _omni_cache_mod.omni_cache.num_computed_tokens = 0
        m = _meta(seq_lens=_t([128]), block_table=_t([[5, 0]]))
        cpu, npu = _prepare_blocks(m, "npu:0", _spec(128))
        assert 5 in cpu.tolist()

    def test_num_comp_as_list(self):
        """List[int] num_computed_tokens: per-request start blocks."""
        _omni_cache_mod.omni_cache.num_computed_tokens = [0, 0]
        m = _meta(seq_lens=_t([128, 128]), block_table=_t([[5, 0], [7, 0]]))
        cpu, npu = _prepare_blocks(m, "npu:0", _spec(128))
        assert 5 in cpu.tolist()
        assert 7 in cpu.tolist()

    def test_merges_dsa_tables(self):
        """DSA extra tables are concatenated; all three blocks are pulled."""
        _omni_cache_mod.omni_cache.num_computed_tokens = _t([0])
        m = _meta(
            seq_lens=_t([128 * 3]),
            block_table=_t([[5]]),
            score_states_block_table=_t([[6]]),
            kv_states_block_table=_t([[7]]),
        )
        cpu, npu = _prepare_blocks(m, "npu:0", _spec(128))
        result = cpu.tolist()
        assert 5 in result
        assert 6 in result and 7 in result

    def test_deduplicates_across_requests(self):
        """Same block ID in multiple requests is returned only once."""
        _omni_cache_mod.omni_cache.num_computed_tokens = _t([0, 0])
        m = _meta(seq_lens=_t([128, 128]), block_table=_t([[5, 0], [5, 0]]))
        cpu, npu = _prepare_blocks(m, "npu:0", _spec(128))
        assert cpu.tolist().count(5) == 1


# ---------------------------------------------------------------------------
# MoME / Mamba path
# ---------------------------------------------------------------------------
class TestMomePath:
    def test_none_cache_indices_returns_empty(self):
        m = _meta(cache_indices=None)
        cpu, npu = _prepare_blocks(m, "npu:0")
        assert cpu.tolist() == []

    def test_non_apc_flattens(self):
        """Non-APC: all non-zero entries preserved (no dedup in this path)."""
        m = _meta(cache_indices=_t([[1, 2, 0], [1, 3, 0]]))
        cpu, npu = _prepare_blocks(m, "npu:0")
        result = cpu.tolist()
        assert 1 in result and 2 in result and 3 in result
        assert 0 not in result  # zeros are filtered

    def test_apc_slices_per_request(self):
        """APC: each request uses [start[i]:end[i]+1] columns."""
        m = _meta(
            cache_indices=_t([[1, 2, 3, 4, 0], [5, 6, 7, 8, 0]]),
            block_idx_first_scheduled_token=_t([0, 0]),
            block_idx_last_scheduled_token=_t([1, 2]),
        )
        cpu, npu = _prepare_blocks(m, "npu:0")
        result = cpu.tolist()
        assert 1 in result and 2 in result
        assert 5 in result and 6 in result and 7 in result

    def test_apc_skips_invalid_range(self):
        """APC: req 1 has end < start, only req 0's blocks are included."""
        m = _meta(
            cache_indices=_t([[1, 2, 0], [3, 4, 0]]),
            block_idx_first_scheduled_token=_t([0, 2]),
            block_idx_last_scheduled_token=_t([1, 1]),
        )
        cpu, npu = _prepare_blocks(m, "npu:0")
        result = cpu.tolist()
        assert 1 in result and 2 in result
        assert 3 not in result
