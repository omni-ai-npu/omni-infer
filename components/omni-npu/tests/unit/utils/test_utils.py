# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch

from omni_npu.v1.layers.utils import get_npu_execution_type


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="NPU required")
@pytest.mark.npu
@pytest.mark.parametrize("stream_input", [
    None,
    "stream_1",
])
def test_get_npu_execution_type_basic(stream_input):
    ctx = get_npu_execution_type(stream_input)
    assert ctx is not None

    if stream_input is None:
        assert isinstance(ctx, nullcontext)
    else:
        assert ctx is not None


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="NPU required")
@pytest.mark.npu
def test_get_npu_execution_type_with_npu_stream():
    stream_obj = torch.npu.Stream()
    ctx = get_npu_execution_type(stream_obj)
    assert ctx is not None

    with ctx:
        t = torch.ones(2, 2, device="npu")
        assert t.device.type == "npu"


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="NPU required")
@pytest.mark.npu
def test_get_npu_execution_type_other_types():
    ctx = get_npu_execution_type(12345)
    assert isinstance(ctx, nullcontext)


# ---------------------------------------------------------------------------
# Tests for omni_npu.v1.utils  (multi-stream & core-limit utilities)
# ---------------------------------------------------------------------------
import omni_npu.v1.utils as v1_utils
from omni_npu.v1.utils import (
    get_nth_last_sep_pos,
    get_last_two_parts,
)


@pytest.fixture(autouse=True)
def _reset_v1_utils_globals():
    """Reset module-level global state before each test."""
    v1_utils._OMNI_STREAM.clear()
    v1_utils._OMNI_EVENT.clear()
    v1_utils._current_stream = None
    yield
    v1_utils._OMNI_STREAM.clear()
    v1_utils._OMNI_EVENT.clear()
    v1_utils._current_stream = None


# -- get_nth_last_sep_pos -----------------------------------------------------

class TestGetNthLastSepPos:
    def test_basic(self):
        assert get_nth_last_sep_pos("a.b.c.d", '.', 2) == 3

    def test_n_equals_1(self):
        assert get_nth_last_sep_pos("a.b.c", '.', 1) == 3

    def test_n_equals_total_seps(self):
        assert get_nth_last_sep_pos("a.b.c", '.', 2) == 1

    def test_n_exceeds_seps(self):
        assert get_nth_last_sep_pos("a.b", '.', 3) == -1

    def test_no_sep_in_string(self):
        assert get_nth_last_sep_pos("abcd", '.', 1) == -1

    def test_n_less_than_1(self):
        assert get_nth_last_sep_pos("a.b.c", '.', 0) == -1

    def test_empty_sep(self):
        assert get_nth_last_sep_pos("a.b.c", '', 1) == -1

    def test_custom_sep(self):
        assert get_nth_last_sep_pos("a/b/c/d", '/', 2) == 3

    def test_empty_string(self):
        assert get_nth_last_sep_pos("", '.', 1) == -1


# -- get_last_two_parts -------------------------------------------------------

class TestGetLastTwoParts:
    def test_basic(self):
        assert get_last_two_parts("a.b.c.d") == "c.d"

    def test_exactly_two_parts(self):
        assert get_last_two_parts("x.y") == "x.y"

    def test_single_part(self):
        assert get_last_two_parts("abc") == "abc"

    def test_three_parts(self):
        assert get_last_two_parts("model.layer.weight") == "layer.weight"

    def test_custom_sep(self):
        assert get_last_two_parts("a/b/c", sep='/') == "b/c"

    def test_empty_string(self):
        assert get_last_two_parts("") == ""


# -- current_stream ------------------------------------------------------------

class TestCurrentStream:
    def test_returns_cached_stream(self):
        mock_stream = MagicMock()
        with patch("torch.npu.current_stream", return_value=mock_stream):
            result = v1_utils.current_stream()
            assert result is mock_stream

    def test_caches_on_second_call(self):
        mock_stream = MagicMock()
        with patch("torch.npu.current_stream", return_value=mock_stream) as mock_fn:
            v1_utils.current_stream()
            v1_utils.current_stream()
            mock_fn.assert_called_once()


# -- get_stream ----------------------------------------------------------------

class TestGetStream:
    def test_creates_and_returns_stream(self):
        mock_stream = MagicMock()
        with patch("torch.npu.Stream", return_value=mock_stream):
            result = v1_utils.get_stream("test_stream")
            assert result is mock_stream
            assert v1_utils._OMNI_STREAM["test_stream"] is mock_stream

    def test_returns_same_stream_on_second_call(self):
        mock_stream = MagicMock()
        with patch("torch.npu.Stream", return_value=mock_stream) as mock_cls:
            v1_utils.get_stream("s1")
            v1_utils.get_stream("s1")
            mock_cls.assert_called_once()

    def test_different_names_create_different_streams(self):
        streams = [MagicMock(), MagicMock()]
        with patch("torch.npu.Stream", side_effect=streams):
            s1 = v1_utils.get_stream("a")
            s2 = v1_utils.get_stream("b")
            assert s1 is not s2


# -- get_event -----------------------------------------------------------------

class TestGetEvent:
    def test_creates_and_returns_event(self):
        mock_event = MagicMock()
        with patch("torch.npu.Event", return_value=mock_event):
            result = v1_utils.get_event("evt1")
            assert result is mock_event
            assert v1_utils._OMNI_EVENT["evt1"] is mock_event

    def test_returns_same_event_on_second_call(self):
        mock_event = MagicMock()
        with patch("torch.npu.Event", return_value=mock_event) as mock_cls:
            v1_utils.get_event("e")
            v1_utils.get_event("e")
            mock_cls.assert_called_once()


# -- switch_npu_stream ---------------------------------------------------------

class TestSwitchNpuStream:
    def test_disabled_returns_nullcontext(self):
        ctx = v1_utils.switch_npu_stream(False, "any")
        assert isinstance(ctx, nullcontext)

    def test_enabled_returns_stream_context(self):
        mock_stream = MagicMock()
        mock_ctx = MagicMock()
        with patch("torch.npu.Stream", return_value=mock_stream), \
             patch("torch.npu.stream", return_value=mock_ctx) as mock_stream_fn:
            result = v1_utils.switch_npu_stream(True, "my_stream")
            mock_stream_fn.assert_called_once_with(mock_stream)
            assert result is mock_ctx


# -- limit_core_num ------------------------------------------------------------

class TestLimitCoreNum:
    def test_disabled_does_nothing(self):
        with v1_utils.limit_core_num(False, "s", 10, 5):
            pass  # should not raise

    def test_enabled_with_string_stream(self):
        mock_stream = MagicMock()
        with patch("torch.npu.Stream", return_value=mock_stream), \
             patch("torch.npu.set_stream_limit") as mock_set, \
             patch("torch.npu.reset_stream_limit") as mock_reset:
            with v1_utils.limit_core_num(True, "test", 8, 4):
                mock_set.assert_called_once_with(mock_stream, 8, 4)
            mock_reset.assert_called_once_with(mock_stream)

    def test_enabled_with_stream_object(self):
        mock_stream = MagicMock(spec=torch.npu.Stream)
        with patch("torch.npu.set_stream_limit") as mock_set, \
             patch("torch.npu.reset_stream_limit") as mock_reset:
            with v1_utils.limit_core_num(True, mock_stream, 6, 3):
                mock_set.assert_called_once_with(mock_stream, 6, 3)
            mock_reset.assert_called_once_with(mock_stream)

    def test_enabled_with_invalid_stream_type(self):
        with patch("torch.npu.set_stream_limit") as mock_set, \
             patch("torch.npu.reset_stream_limit") as mock_reset:
            with v1_utils.limit_core_num(True, 12345, 6, 3):
                mock_set.assert_not_called()
            mock_reset.assert_not_called()


# -- record_stream -------------------------------------------------------------

class TestRecordStream:
    def test_disabled_does_nothing(self):
        tensor = MagicMock()
        v1_utils.record_stream(False, tensor, "s")
        tensor.record_stream.assert_not_called()

    def test_enabled_with_string_stream(self):
        mock_stream = MagicMock()
        tensor = MagicMock()
        with patch("torch.npu.Stream", return_value=mock_stream):
            v1_utils.record_stream(True, tensor, "rs")
            tensor.record_stream.assert_called_once_with(mock_stream)

    def test_enabled_with_stream_object(self):
        mock_stream = MagicMock(spec=torch.npu.Stream)
        tensor = MagicMock()
        v1_utils.record_stream(True, tensor, mock_stream)
        tensor.record_stream.assert_called_once_with(mock_stream)

    def test_enabled_with_invalid_type(self):
        tensor = MagicMock()
        v1_utils.record_stream(True, tensor, 999)
        tensor.record_stream.assert_not_called()


# -- wait_event ----------------------------------------------------------------

class TestWaitEvent:
    def test_disabled_does_nothing(self):
        v1_utils.wait_event(False, "s", "e")  # should not raise

    def test_enabled_with_string_stream(self):
        mock_stream = MagicMock()
        mock_event = MagicMock()
        with patch("torch.npu.Stream", return_value=mock_stream), \
             patch("torch.npu.Event", return_value=mock_event):
            v1_utils.wait_event(True, "s1", "e1")
            mock_stream.wait_event.assert_called_once_with(mock_event)

    def test_enabled_with_stream_object(self):
        mock_stream = MagicMock(spec=torch.npu.Stream)
        mock_event = MagicMock()
        with patch("torch.npu.Event", return_value=mock_event):
            v1_utils.wait_event(True, mock_stream, "e1")
            mock_stream.wait_event.assert_called_once_with(mock_event)

    def test_enabled_with_invalid_stream(self):
        with patch("torch.npu.Event", return_value=MagicMock()):
            v1_utils.wait_event(True, 123, "e1")  # should not raise


# -- record_event --------------------------------------------------------------

class TestRecordEvent:
    def test_disabled_does_nothing(self):
        v1_utils.record_event(False, "s", "e")  # should not raise

    def test_enabled_with_string_stream(self):
        mock_stream = MagicMock()
        mock_event = MagicMock()
        with patch("torch.npu.Stream", return_value=mock_stream), \
             patch("torch.npu.Event", return_value=mock_event):
            v1_utils.record_event(True, "s1", "e1")
            mock_stream.record_event.assert_called_once_with(mock_event)

    def test_enabled_with_stream_object(self):
        mock_stream = MagicMock(spec=torch.npu.Stream)
        mock_event = MagicMock()
        with patch("torch.npu.Event", return_value=mock_event):
            v1_utils.record_event(True, mock_stream, "e1")
            mock_stream.record_event.assert_called_once_with(mock_event)

    def test_enabled_with_invalid_stream(self):
        with patch("torch.npu.Event", return_value=MagicMock()):
            v1_utils.record_event(True, 999, "e1")  # should not raise


# -- wait_stream ---------------------------------------------------------------

class TestWaitStream:
    def test_disabled_does_nothing(self):
        v1_utils.wait_stream(False, "s1", "s2")  # should not raise

    def test_enabled_with_both_string_streams(self):
        streams = [MagicMock(), MagicMock()]
        with patch("torch.npu.Stream", side_effect=streams):
            v1_utils.wait_stream(True, "a", "b")
            streams[0].wait_stream.assert_called_once_with(streams[1])

    def test_enabled_with_both_stream_objects(self):
        s1 = MagicMock(spec=torch.npu.Stream)
        s2 = MagicMock(spec=torch.npu.Stream)
        v1_utils.wait_stream(True, s1, s2)
        s1.wait_stream.assert_called_once_with(s2)

    def test_enabled_with_mixed_types(self):
        # Pre-populate _OMNI_STREAM directly to avoid patching torch.npu.Stream
        # (patching it would break the isinstance(..., torch.npu.Stream) check).
        named_stream = MagicMock()
        v1_utils._OMNI_STREAM["named"] = named_stream
        s2 = MagicMock(spec=torch.npu.Stream)
        v1_utils.wait_stream(True, "named", s2)
        named_stream.wait_stream.assert_called_once_with(s2)

    def test_enabled_with_invalid_first_stream(self):
        v1_utils.wait_stream(True, 123, "s2")  # should not raise


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-m", "npu"]))
