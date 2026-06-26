# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for Task #16 MoMe restore helper.

Covers `omni_cache.cache.prefill.prefill_omni_cache._restore_mome_slots_from_host`,
which is the mirror of `_copy_mome_slots_to_host` used on the decode side to
rehydrate conv_state from the shared host pool.
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch


def _import_helpers():
    """Load prefill helpers without triggering NPU-only imports."""
    for name in ("omni_npu", "omni_npu.worker", "omni_npu.worker.npu_model_runner"):
        sys.modules.setdefault(name, MagicMock())
    from omni_cache.cache.prefill.prefill_omni_cache import (  # noqa: WPS433
        _copy_mome_slots_to_host,
        _restore_mome_slots_from_host,
    )
    return _copy_mome_slots_to_host, _restore_mome_slots_from_host


# Module-level fallback: the file may live under prefill.prefill_omni_cache
# or directly under prefill (older layouts). Resolve at import time.
try:
    _COPY, _RESTORE = _import_helpers()
except ModuleNotFoundError:
    for name in ("omni_npu", "omni_npu.worker", "omni_npu.worker.npu_model_runner"):
        sys.modules.setdefault(name, MagicMock())
    from omni_cache.cache.prefill.prefill_omni_cache import (
        _copy_mome_slots_to_host as _COPY,
        _restore_mome_slots_from_host as _RESTORE,
    )


class TestRestoreMomeSlotsFromHost:
    """Pure-CPU validation of the restore helper."""

    def _make_state(self, num_cache_lines=8, state_shape=(2, 16), dtype=torch.bfloat16):
        flat = torch.arange(
            num_cache_lines * state_shape[0] * state_shape[1],
            dtype=dtype,
        )
        return flat.reshape(num_cache_lines, *state_shape).clone()

    def test_round_trip_snapshot_then_restore(self):
        state = self._make_state()
        host = torch.zeros((1, 8, 256), dtype=state.dtype)
        slots = torch.tensor([0, 2, 5], dtype=torch.int64)

        _COPY(state, host, slots, dp_local_rank=0, tp_rank=0, tp_size=1)

        # Fresh destination, simulating decode-side newly-allocated cache.
        restored = torch.zeros_like(state)
        _RESTORE(restored, host, slots, dp_local_rank=0, tp_rank=0, tp_size=1)

        # Only the snapshotted slots should match the original state; the
        # others stay zero.
        for slot in slots.tolist():
            assert torch.equal(restored[slot], state[slot]), (
                f"slot {slot} did not round-trip"
            )
        untouched = set(range(state.shape[0])) - set(slots.tolist())
        for slot in untouched:
            assert torch.all(restored[slot] == 0)

    def test_empty_slots_empty_noop(self):
        state = self._make_state()
        host = torch.randn(1, 8, 256, dtype=torch.float32).to(state.dtype)
        restored = torch.zeros_like(state)
        _RESTORE(
            restored,
            host,
            torch.tensor([], dtype=torch.int64),
            dp_local_rank=0,
        )
        assert torch.all(restored == 0)

    def test_out_of_range_slots_dropped(self):
        state = self._make_state(num_cache_lines=4)
        host = torch.zeros((1, 4, 256), dtype=state.dtype)
        # Snapshot only valid slots.
        _COPY(
            state,
            host,
            torch.tensor([0, 2], dtype=torch.int64),
            dp_local_rank=0,
        )

        restored = torch.zeros_like(state)
        # Mixed: valid (2), negative pad (-1), out-of-range (9_999).
        _RESTORE(
            restored,
            host,
            torch.tensor([2, -1, 9_999], dtype=torch.int64),
            dp_local_rank=0,
        )

        assert torch.equal(restored[2], state[2])
        # Other rows unchanged; no crash.
        assert torch.all(restored[0] == 0)
        assert torch.all(restored[1] == 0)
        assert torch.all(restored[3] == 0)

    def test_dp_greater_than_one_reads_correct_rank(self):
        state = self._make_state(num_cache_lines=4)
        host = torch.zeros((2, 4, 256), dtype=state.dtype)
        slots = torch.tensor([1, 3], dtype=torch.int64)

        # Snapshot on dp_local_rank=1.
        _COPY(state, host, slots, dp_local_rank=1, tp_rank=0, tp_size=1)

        # Sanity: dp=0 should still be zero.
        assert host_slice_is_zero(host, dp=0)

        # Restore on dp=1 should reproduce state.
        restored = torch.zeros_like(state)
        _RESTORE(restored, host, slots, dp_local_rank=1, tp_rank=0, tp_size=1)
        for s in slots.tolist():
            assert torch.equal(restored[s], state[s])

        # Restore on dp=0 should fetch nothing (zeros) — this validates
        # the dp dimension isn't being flattened.
        restored_dp0 = torch.zeros_like(state)
        _RESTORE(restored_dp0, host, slots, dp_local_rank=0, tp_rank=0, tp_size=1)
        for s in slots.tolist():
            assert torch.all(restored_dp0[s] == 0), (
                f"dp=0 slot {s} unexpectedly populated from dp=1 snapshot"
            )

    def test_out_of_range_tp_skips(self):
        """If the rank window wouldn't fit in the host slot we bail cleanly."""
        # state: (4, 2, 16) → 32 elems per block. Host inner = 4 → can't fit.
        state = torch.arange(4 * 2 * 16, dtype=torch.bfloat16).reshape(4, 2, 16)
        host = torch.zeros((1, 4, 4), dtype=state.dtype)

        restored = torch.zeros_like(state)
        _RESTORE(
            restored,
            host,
            torch.tensor([0, 1], dtype=torch.int64),
            dp_local_rank=0,
            tp_rank=0,
            tp_size=1,
        )
        # Restore must leave restored untouched, not crash.
        assert torch.all(restored == 0)

    def test_different_tp_ranks_read_different_offsets(self):
        """Snapshot at tp_rank=0, attempt restore with tp_rank=1: should
        read a disjoint window of the host slot (and therefore not equal
        the original state since that region was never written)."""
        state = torch.arange(4 * 32, dtype=torch.bfloat16).reshape(4, 32) + 1.0
        # host slot wide enough for 2 ranks worth of state (2 * 32 = 64).
        host = torch.zeros((1, 4, 128), dtype=state.dtype)
        slots = torch.tensor([0, 1, 2], dtype=torch.int64)

        _COPY(state, host, slots, dp_local_rank=0, tp_rank=0, tp_size=2)

        restored_rank0 = torch.zeros_like(state)
        _RESTORE(restored_rank0, host, slots, dp_local_rank=0,
                 tp_rank=0, tp_size=2)
        for slot in slots.tolist():
            assert torch.equal(restored_rank0[slot], state[slot]), (
                f"tp=0 restore mismatch for slot {slot}"
            )

        restored_rank1 = torch.zeros_like(state)
        _RESTORE(restored_rank1, host, slots, dp_local_rank=0,
                 tp_rank=1, tp_size=2)
        for slot in slots.tolist():
            # tp_rank=1's window in the host slot was never written, so it
            # should come back as zeros — proving the offset isn't aliased
            # to rank 0's region.
            assert torch.all(restored_rank1[slot] == 0), (
                f"tp=1 restore slot {slot} unexpectedly nonzero — offset "
                "logic may be aliasing rank 0's window"
            )

    def test_round_trip_higher_dim_state(self):
        """Shape >2 inner dims (e.g., (conv_dim, channels, kernel)) must
        reshape correctly on both snapshot and restore."""
        state = torch.randn(6, 2, 4, 8, dtype=torch.float32).to(torch.bfloat16)
        inner = 2 * 4 * 8
        host = torch.zeros((1, 6, inner * 2), dtype=torch.bfloat16)
        slots = torch.tensor([1, 4], dtype=torch.int64)

        _COPY(state, host, slots, dp_local_rank=0, tp_rank=0, tp_size=1)

        restored = torch.zeros_like(state)
        _RESTORE(restored, host, slots, dp_local_rank=0, tp_rank=0, tp_size=1)

        for slot in slots.tolist():
            assert torch.equal(restored[slot], state[slot])
        # Shape preserved.
        assert restored.shape == state.shape


def host_slice_is_zero(host: torch.Tensor, dp: int) -> bool:
    return bool(host[dp].abs().sum() == 0)
