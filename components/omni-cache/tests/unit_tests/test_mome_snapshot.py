# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for the MoMe snapshot and PD-connector fixes.

Covers:
- `_copy_mome_slots_to_host` (shape handling, TP offset, bounds filter,
  empty-slots, oversized-state).
- `ModelForwardPlugin.post_model_forward` dispatch rules (disabled when
  env off, prefill-only, decode no-op, swallows snapshot errors).
- `KVLoader._read_blocks` rank-0 gate for the single-sender OX protocol.
- `OmniCacheConnector.get_finished_count` returning 1 on the decode side.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch


# ----------------------------------------------------------------------------
# _copy_mome_slots_to_host — pure tensor helper, importable without NPU deps
# ----------------------------------------------------------------------------


def _import_copy_helper():
    """Load the helper without triggering the NPU-only class imports.

    `prefill_omni_cache` imports `from omni_npu...`; stub those out.
    """
    # Stub out optional NPU imports so the module loads on a plain box.
    for missing in ("omni_npu", "omni_npu.worker", "omni_npu.worker.npu_model_runner"):
        sys.modules.setdefault(missing, MagicMock())

    from omni_cache.cache.prefill.prefill_omni_cache import (
        _copy_mome_slots_to_host,
    )
    return _copy_mome_slots_to_host


class TestCopyMomeSlotsToHost:
    """Pure-CPU validation of the shape/offset/bounds logic."""

    def _inputs(self, num_blocks=8, num_cache_lines=8, state_shape=(2, 16),
                host_inner=256, slots=(0, 2, 5), dp=1):
        state = torch.arange(
            num_cache_lines * state_shape[0] * state_shape[1],
            dtype=torch.bfloat16,
        ).reshape(num_cache_lines, *state_shape)
        host = torch.zeros((dp, num_blocks, host_inner), dtype=torch.bfloat16)
        return state, host, torch.tensor(slots, dtype=torch.int64)

    def test_single_rank_writes_at_offset_zero(self):
        copy = _import_copy_helper()
        state, host, slots = self._inputs()
        n_state_elems = state.shape[1] * state.shape[2]

        copy(state, host, slots, dp_local_rank=0, tp_rank=0, tp_size=1)

        hl = host[0]
        for slot in slots.tolist():
            actual = hl[slot, :n_state_elems].reshape(state.shape[1:])
            expected = state[slot]
            assert torch.equal(actual, expected)

    def test_ranks_write_to_non_overlapping_regions(self):
        copy = _import_copy_helper()
        n_state_elems = None
        shared_host = torch.zeros((1, 4, 256), dtype=torch.bfloat16)
        for tp_rank in range(4):
            state, _, slots = self._inputs(num_blocks=4, slots=(1,))
            # Tag this rank's state so we can tell it apart.
            state = state + 100 * tp_rank
            copy(state, shared_host, slots, dp_local_rank=0,
                 tp_rank=tp_rank, tp_size=4)
            if n_state_elems is None:
                n_state_elems = state.shape[1] * state.shape[2]

        for tp_rank in range(4):
            start = tp_rank * n_state_elems
            end = start + n_state_elems
            region = shared_host[0, 1, start:end]
            # Region should reflect that rank's tag (offset 100*tp_rank).
            head_value = int(region[0].item())
            # bf16 rounding — just check the rank signature is present.
            assert head_value >= 100 * tp_rank

    def test_pad_slot_id_and_out_of_range_dropped(self):
        copy = _import_copy_helper()
        state, host, _ = self._inputs()
        slots = torch.tensor([2, -1, 10_000], dtype=torch.int64)

        copy(state, host, slots, dp_local_rank=0)

        # Only valid slot 2 should have been written.
        n = state.shape[1] * state.shape[2]
        assert host[0, 2, :n].abs().sum() != 0
        # Slot 1 left untouched.
        assert host[0, 1, :n].abs().sum() == 0

    def test_empty_slots_is_noop(self):
        copy = _import_copy_helper()
        state, host, _ = self._inputs()
        copy(state, host, torch.tensor([], dtype=torch.int64), dp_local_rank=0)
        assert host.abs().sum() == 0

    def test_host_slot_too_small_skips(self):
        copy = _import_copy_helper()
        state = torch.ones((4, 2, 16), dtype=torch.bfloat16)
        # host slot has 4 elements but state needs 32.
        host = torch.zeros((1, 4, 4), dtype=torch.bfloat16)
        slots = torch.tensor([0, 1], dtype=torch.int64)

        copy(state, host, slots, dp_local_rank=0, tp_rank=0, tp_size=1)

        assert host.abs().sum() == 0  # nothing written

    def test_dp_greater_than_one_picks_correct_rank(self):
        copy = _import_copy_helper()
        # Host with dp=2 — 2 independent slabs stacked on the leading dim.
        state = torch.full((4, 2, 4), 7.0, dtype=torch.bfloat16)
        host = torch.zeros((2, 4, 128), dtype=torch.bfloat16)
        slots = torch.tensor([0, 1], dtype=torch.int64)

        copy(state, host, slots, dp_local_rank=1, tp_rank=0, tp_size=1)

        # dp=0 untouched, dp=1 has data.
        assert host[0].abs().sum() == 0
        assert host[1, 0].abs().sum() != 0



# ----------------------------------------------------------------------
# ModelForwardPlugin
# ----------------------------------------------------------------------


def _get_plugin_cls():
    from omni_cache.attn_plugins.implementations import ModelForwardPlugin
    return ModelForwardPlugin


class TestModelForwardPlugin:

    def test_noop_when_env_disabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_OMNI_CACHE", "0")
        # Even with a prefill-like oc around, nothing should happen.
        plugin = _get_plugin_cls()()
        plugin.post_model_forward()  # must not raise
        plugin.pre_model_forward()

    def test_dispatches_snapshot_on_prefill(self, monkeypatch):
        monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
        monkeypatch.setenv("OMNI_CACHE_MOME_SNAPSHOT", "1")
        prefill = MagicMock()
        prefill.__class__.__name__ = "PrefillOmniCache"
        prefill.snapshot_mome_states = MagicMock()

        import omni_cache.cache as _cache_mod
        prev = getattr(_cache_mod, "omni_cache", None)
        _cache_mod.omni_cache = prefill
        try:
            _get_plugin_cls()().post_model_forward()
        finally:
            _cache_mod.omni_cache = prev

        prefill.snapshot_mome_states.assert_called_once()

    def test_skips_on_decode_instance(self, monkeypatch):
        monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
        decode = MagicMock()
        decode.__class__.__name__ = "DecodeOmniCache"

        import omni_cache.cache as _cache_mod
        prev = getattr(_cache_mod, "omni_cache", None)
        _cache_mod.omni_cache = decode
        try:
            _import_plugin_and_call_post()
        finally:
            _cache_mod.omni_cache = prev

        decode.snapshot_mome_states.assert_not_called()

    def test_swallows_snapshot_error(self, monkeypatch):
        """A bug in snapshot_mome_states must not crash the forward pass."""
        monkeypatch.setenv("ENABLE_OMNI_CACHE", "1")
        monkeypatch.setenv("OMNI_CACHE_MOME_SNAPSHOT", "1")
        prefill = MagicMock()
        prefill.__class__.__name__ = "PrefillOmniCache"
        prefill.snapshot_mome_states.side_effect = RuntimeError("boom")

        import omni_cache.cache as _cache_mod
        prev = getattr(_cache_mod, "omni_cache", None)
        _cache_mod.omni_cache = prefill
        try:
            _import_plugin_and_call_post()  # must not raise
        finally:
            _cache_mod.omni_cache = prev


def _import_plugin_and_call_post():
    """Instantiate the plugin and call post_model_forward with no args.

    Helper used by tests that need to swap the `omni_cache` global at
    runtime before invoking.
    """
    from omni_cache.attn_plugins.implementations import ModelForwardPlugin
    ModelForwardPlugin().post_model_forward()


# ----------------------------------------------------------------------------
# OmniCacheConnector.get_finished_count
# ----------------------------------------------------------------------------


class TestGetFinishedCount:
    """The decode-side aggregator needs `1` so a single rank-0 notification
    is enough to release the request; prior to the Task #19 fix this was
    `tp_world_size` and deadlocked when only one rank received the OX
    response."""

    def test_returns_one(self):
        from omni_cache.connector.connector import OmniCacheConnector
        # Call as unbound — we don't need a constructed connector for this
        # deterministic function, and avoiding construction keeps the test
        # independent of vllm_config wiring.
        fake_self = types.SimpleNamespace()
        assert OmniCacheConnector.get_finished_count(fake_self) == 1


# ----------------------------------------------------------------------------
# KVLoader._read_blocks rank-0 gate
# ----------------------------------------------------------------------------


class TestKvLoaderReadBlocks:

    def _make_loader(self, tp_rank: int, tp_world_size: int = 1):
        from omni_cache.connector.decode.kv_loader import KVLoader
        loader = KVLoader.__new__(KVLoader)
        loader.worker = types.SimpleNamespace(
            pending={},
            zmq_client=MagicMock(),
        )
        loader.omni_cache = types.SimpleNamespace(
            num_blocks=1024,
            dp_local_rank=0,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        return loader

    def _args(self):
        return dict(
            local_block_ids=[[1, 2, 3]],
            remote_block_ids=[10, 11, 12],
            dst_cluster_id="cluster",
            request_id="req-id",
            remote_request_id="remote-req-id",
            remote_host_ip="tcp://x:1",
            remote_dp_rank=0,
        )

    def test_rank_zero_sends(self):
        loader = self._make_loader(tp_rank=0, tp_world_size=4)
        loader._read_blocks(**self._args())
        assert "req-id" in loader.worker.pending
        loader.worker.zmq_client.send_request.assert_called_once()

    def test_nonzero_ranks_skip_send(self):
        for tp_rank in (1, 2, 3, 7):
            loader = self._make_loader(tp_rank=tp_rank, tp_world_size=8)
            loader._read_blocks(**self._args())
            assert loader.worker.pending == {}, \
                f"rank {tp_rank} should not have touched pending"
            loader.worker.zmq_client.send_request.assert_not_called()
