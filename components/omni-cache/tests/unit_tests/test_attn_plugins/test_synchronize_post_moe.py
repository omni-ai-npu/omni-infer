# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for _drain_d2h_stage, _drain_h2d_stage, _sync_and_clear.

These functions are pure helpers that operate on plain attributes of the
omni_cache object (copy_futures_stages, d2h_event_stages, h2d_event_stages).
They do NOT depend on torch, numpy or any NPU runtime, so we test them by
importing the source text directly, bypassing the heavy omni_cache import
chain.
"""

import sys
from concurrent.futures import Future
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Load the three target functions without pulling in the full omni_cache tree.
# ---------------------------------------------------------------------------
_src = {}
with open("omni_cache/attn_plugins/base.py") as f:
    exec(compile(f.read(), "base.py", "exec"), _src)

_drain_d2h_stage = _src["_drain_d2h_stage"]
_drain_h2d_stage = _src["_drain_h2d_stage"]
_sync_and_clear = _src["_sync_and_clear"]

# Give them a fake logger so they don't crash on logging calls
class _FakeLogger:
    def warning(self, *args, **kwargs):
        pass

_src["logger"] = _FakeLogger()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make_mock_event(sync_side_effect=None):
    """Return a MagicMock that looks like a torch.npu.Event."""
    evt = MagicMock()
    evt.synchronize = MagicMock(side_effect=sync_side_effect)
    return evt


def _make_oc(d2h_fut_dict=None, d2h_evt_dict=None, h2d_evt_dict=None,
             num_stages=2):
    """Build a mock omni_cache with per-stage dicts.

    All three attributes are always set (to avoid Mock auto-creation).
    Pass None for an empty list of dicts for that attribute.
    """
    oc = Mock(spec=[])  # spec=[] prevents auto-creation of arbitrary attrs
    oc.copy_futures_stages = (
        [d2h_fut_dict for _ in range(num_stages)]
        if d2h_fut_dict is not None else []
    )
    oc.d2h_event_stages = (
        [d2h_evt_dict for _ in range(num_stages)]
        if d2h_evt_dict is not None else []
    )
    oc.h2d_event_stages = (
        [h2d_evt_dict for _ in range(num_stages)]
        if h2d_evt_dict is not None else []
    )
    return oc


# ---------------------------------------------------------------------------
# _drain_d2h_stage
# ---------------------------------------------------------------------------
class TestDrainD2hStage:
    def test_drains_pending_future(self):
        fut = Future()
        fut.set_result(None)
        oc = _make_oc(d2h_fut_dict={0: fut})
        _drain_d2h_stage(oc, 0)

    def test_skips_none_future(self):
        oc = _make_oc(d2h_fut_dict={0: None})
        _drain_d2h_stage(oc, 0)

    def test_logs_future_exception(self):
        fut = Future()
        fut.set_exception(RuntimeError("scatter failed"))
        oc = _make_oc(d2h_fut_dict={0: fut})
        # Should not raise — logs and continues
        _drain_d2h_stage(oc, 0)

    def test_clears_dict_after_drain(self):
        fut = Future()
        fut.set_result(None)
        bucket = {0: fut, 1: Future()}
        bucket[1].set_result(None)
        oc = _make_oc(d2h_fut_dict=bucket)
        _drain_d2h_stage(oc, 0)
        assert len(oc.copy_futures_stages[0]) == 0

    def test_clears_list_after_drain(self):
        fut = Future()
        fut.set_result(None)
        oc = _make_oc(d2h_fut_dict=[fut])
        _drain_d2h_stage(oc, 0)
        assert len(oc.copy_futures_stages[0]) == 0

    def test_noop_when_missing_attr(self):
        oc = Mock(spec=[])
        _drain_d2h_stage(oc, 0)

    def test_noop_when_stage_out_of_bounds(self):
        oc = _make_oc(d2h_fut_dict={0: Future()}, num_stages=1)
        _drain_d2h_stage(oc, 1)

    def test_drains_d2h_events(self):
        evt = _make_mock_event()
        oc = _make_oc(d2h_evt_dict={0: evt})
        _drain_d2h_stage(oc, 0)
        evt.synchronize.assert_called_once()
        assert len(oc.d2h_event_stages[0]) == 0

    def test_logs_event_exception(self):
        evt = _make_mock_event(sync_side_effect=OSError("sync fail"))
        oc = _make_oc(d2h_evt_dict={0: evt})
        _drain_d2h_stage(oc, 0)  # should not raise

    def test_drains_both_futures_and_events(self):
        fut = Future()
        fut.set_result(None)
        evt = _make_mock_event()
        oc = _make_oc(d2h_fut_dict={0: fut}, d2h_evt_dict={0: evt})
        _drain_d2h_stage(oc, 0)
        evt.synchronize.assert_called_once()
        assert len(oc.copy_futures_stages[0]) == 0
        assert len(oc.d2h_event_stages[0]) == 0

    def test_calls_result_on_pending_future(self):
        """Every non-None future's result() is called to wait on completion."""
        fut = Mock(spec=Future)
        fut.result.return_value = None
        oc = _make_oc(d2h_fut_dict={0: fut})
        _drain_d2h_stage(oc, 0)
        fut.result.assert_called_once()

    def test_result_exception_is_caught(self):
        """Exception from result() is logged, not re-raised."""
        fut = Mock(spec=Future)
        fut.result.side_effect = RuntimeError("boom")
        oc = _make_oc(d2h_fut_dict={0: fut})
        _drain_d2h_stage(oc, 0)  # should not raise
        fut.result.assert_called_once()

    def test_handles_mixed_dict_items(self):
        """Dict with mix of real/None items all get processed."""
        real = Mock(spec=Future)
        real.result.return_value = None
        bucket = {0: real, 1: None}
        oc = _make_oc(d2h_fut_dict=bucket)
        _drain_d2h_stage(oc, 0)
        real.result.assert_called_once()
        assert len(oc.copy_futures_stages[0]) == 0


# ---------------------------------------------------------------------------
# _drain_h2d_stage
# ---------------------------------------------------------------------------
class TestDrainH2dStage:
    def test_drains_h2d_events(self):
        evt = _make_mock_event()
        oc = _make_oc(h2d_evt_dict={0: evt})
        _drain_h2d_stage(oc, 0)
        evt.synchronize.assert_called_once()
        assert len(oc.h2d_event_stages[0]) == 0

    def test_noop_when_missing_attr(self):
        oc = Mock(spec=[])
        _drain_h2d_stage(oc, 0)

    def test_noop_when_stage_out_of_bounds(self):
        oc = _make_oc(h2d_evt_dict={0: _make_mock_event()}, num_stages=1)
        _drain_h2d_stage(oc, 1)

    def test_handles_empty_dict(self):
        oc = _make_oc(h2d_evt_dict={})
        _drain_h2d_stage(oc, 0)


# ---------------------------------------------------------------------------
# _sync_and_clear
# ---------------------------------------------------------------------------
class TestSyncAndClear:
    def test_syncs_dict_bucket(self):
        evt = _make_mock_event()
        stages = [{"g0": evt}]
        _sync_and_clear(None, stages, 0, "test")
        evt.synchronize.assert_called_once()
        assert len(stages[0]) == 0

    def test_syncs_list_bucket(self):
        evt = _make_mock_event()
        stages = [[evt]]
        _sync_and_clear(None, stages, 0, "test")
        evt.synchronize.assert_called_once()
        assert len(stages[0]) == 0

    def test_syncs_single_item_bucket(self):
        evt = _make_mock_event()
        stages = [evt]
        _sync_and_clear(None, stages, 0, "test")
        evt.synchronize.assert_called_once()

    def test_skips_none_in_bucket(self):
        evt = _make_mock_event()
        stages = [{"g0": None, "g1": evt}]
        _sync_and_clear(None, stages, 0, "test")
        evt.synchronize.assert_called_once()

    def test_handles_none_bucket(self):
        stages = [None]
        _sync_and_clear(None, stages, 0, "test")  # should not raise

    def test_logs_event_exception(self):
        evt = _make_mock_event(sync_side_effect=RuntimeError("boom"))
        stages = [[evt]]
        _sync_and_clear(None, stages, 0, "test")  # should not raise

    def test_clears_dict_bucket(self):
        stages = [{"g0": _make_mock_event(), "g1": _make_mock_event()}]
        _sync_and_clear(None, stages, 0, "test")
        assert len(stages[0]) == 0

    def test_clears_list_bucket(self):
        stages = [[_make_mock_event(), _make_mock_event()]]
        _sync_and_clear(None, stages, 0, "test")
        assert len(stages[0]) == 0
