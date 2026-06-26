# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for the streaming reasoning→tool-parser relay.

Covers the helper functions in ``omni_npu/v1/parsers/_streaming_relay.py``
and exercises the wrapped methods on :class:`PanguReasoningParser` /
:class:`PanguToolParser` to confirm the boundary-chunk reasoning text is
preserved across the upstream-vLLM-v0.14.0 ``serving_chat`` overwrite
(see module docstring of ``_streaming_relay.py``).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from vllm.entrypoints.openai.protocol import DeltaMessage

from omni_npu.v1.parsers._streaming_relay import (
    _pending_reasoning,
    reattach_reasoning_to,
    reset_for_tests,
    stash_reasoning_from,
)


class _RelayTestBase(unittest.TestCase):
    """Shared setup/teardown for relay tests.

    ``conftest.py`` already runs ``reset_for_tests`` around every test in
    this directory via an autouse fixture, but ``unittest`` ``setUp``
    runs *after* the fixture's enter step, so tests that mutate the
    ContextVar inline (e.g. ``_pending_reasoning.set("stale")`` before
    invoking a helper) need the explicit reset here too — both to undo
    the test's own pre-condition setup and to leave the ContextVar
    clean for the next ``setUp``/test pair within the same class."""

    def setUp(self) -> None:
        reset_for_tests()

    def tearDown(self) -> None:
        reset_for_tests()


class TestStashReasoningFrom(_RelayTestBase):
    """Reasoning-parser side: ``stash_reasoning_from(delta)``."""

    def test_none_delta_clears_stash(self) -> None:
        _pending_reasoning.set("stale")
        stash_reasoning_from(None)
        self.assertIsNone(_pending_reasoning.get())

    def test_delta_without_reasoning_clears_stash(self) -> None:
        _pending_reasoning.set("stale")
        stash_reasoning_from(DeltaMessage(content="hello"))
        self.assertIsNone(_pending_reasoning.get())

    def test_delta_with_empty_reasoning_string_clears_stash(self) -> None:
        _pending_reasoning.set("stale")
        stash_reasoning_from(DeltaMessage(reasoning=""))
        self.assertIsNone(_pending_reasoning.get())

    def test_delta_with_reasoning_stores_it(self) -> None:
        stash_reasoning_from(DeltaMessage(reasoning="thinking…"))
        self.assertEqual(_pending_reasoning.get(), "thinking…")

    def test_delta_with_reasoning_and_content_stores_only_reasoning(self) -> None:
        # The classic boundary delta produced by basic_parsers when
        # </think> is mid-delta and there is content after it.
        stash_reasoning_from(
            DeltaMessage(reasoning="last thoughts", content="first content")
        )
        self.assertEqual(_pending_reasoning.get(), "last thoughts")

    def test_subsequent_call_overwrites_stash(self) -> None:
        stash_reasoning_from(DeltaMessage(reasoning="first"))
        self.assertEqual(_pending_reasoning.get(), "first")
        stash_reasoning_from(DeltaMessage(reasoning="second"))
        self.assertEqual(_pending_reasoning.get(), "second")


class TestReattachReasoningTo(_RelayTestBase):
    """Tool-parser side: ``reattach_reasoning_to(delta)``."""

    def test_no_pending_returns_delta_unchanged(self) -> None:
        delta = DeltaMessage(content="hello")
        out = reattach_reasoning_to(delta)
        self.assertIs(out, delta)
        self.assertEqual(out.content, "hello")
        self.assertIsNone(getattr(out, "reasoning", None))

    def test_no_pending_none_delta_returns_none(self) -> None:
        self.assertIsNone(reattach_reasoning_to(None))

    def test_pending_attaches_to_existing_content_delta(self) -> None:
        # Tool parser produced a content-only delta — without the fix,
        # this would be the chunk where reasoning is lost forever.
        _pending_reasoning.set("the last reasoning bits")
        out = reattach_reasoning_to(DeltaMessage(content="hello"))
        self.assertEqual(out.reasoning, "the last reasoning bits")
        # Mirror to the deprecated alias must be populated too — the
        # @model_validator only runs at construction, so we set both
        # fields explicitly in the assignment path.
        self.assertEqual(out.reasoning_content, "the last reasoning bits")
        self.assertEqual(out.content, "hello")
        self.assertIsNone(_pending_reasoning.get())

    def test_pending_attaches_to_none_delta_creates_new(self) -> None:
        _pending_reasoning.set("the last reasoning bits")
        out = reattach_reasoning_to(None)
        self.assertIsNotNone(out)
        self.assertEqual(out.reasoning, "the last reasoning bits")
        # Construction path: model_validator mirrors automatically.
        self.assertEqual(out.reasoning_content, "the last reasoning bits")
        self.assertIsNone(out.content)
        self.assertIsNone(_pending_reasoning.get())

    def test_pending_attaches_to_tool_calls_delta(self) -> None:
        _pending_reasoning.set("reasoning before tools")
        tool_delta = DeltaMessage()
        tool_delta.tool_calls = [MagicMock(index=0)]
        out = reattach_reasoning_to(tool_delta)
        self.assertEqual(out.reasoning, "reasoning before tools")
        self.assertEqual(out.reasoning_content, "reasoning before tools")
        self.assertEqual(len(out.tool_calls), 1)
        self.assertIsNone(_pending_reasoning.get())

    def test_pending_does_not_overwrite_existing_reasoning(self) -> None:
        _pending_reasoning.set("stashed")
        out = reattach_reasoning_to(DeltaMessage(reasoning="from-tool"))
        self.assertEqual(out.reasoning, "from-tool")
        # Existing reasoning was set via construction, so its
        # reasoning_content mirror is already populated.
        self.assertEqual(out.reasoning_content, "from-tool")
        self.assertIsNone(_pending_reasoning.get())

    def test_pending_attaches_when_existing_reasoning_is_empty_string(
        self,
    ) -> None:
        # Falsy-but-not-None: existing reasoning is "" — treat as missing
        # and let the stash through (consistent with the truthiness gate
        # in ``stash_reasoning_from``).
        _pending_reasoning.set("stashed")
        out = reattach_reasoning_to(DeltaMessage(reasoning=""))
        self.assertEqual(out.reasoning, "stashed")
        self.assertEqual(out.reasoning_content, "stashed")
        self.assertIsNone(_pending_reasoning.get())

    def test_consumed_after_read(self) -> None:
        _pending_reasoning.set("once")
        reattach_reasoning_to(DeltaMessage(content="a"))
        out = reattach_reasoning_to(DeltaMessage(content="b"))
        self.assertIsNone(getattr(out, "reasoning", None))


class TestEndToEndRelay(_RelayTestBase):
    """Full reasoning → tool relay pattern as ``serving_chat`` would drive it."""

    def test_pre_boundary_iteration_then_boundary(self) -> None:
        # Iter 1: pre-boundary, reasoning-only delta. Tool parser NOT
        # called by serving_chat. Stash sits.
        stash_reasoning_from(DeltaMessage(reasoning="building up"))
        self.assertEqual(_pending_reasoning.get(), "building up")

        # Iter 2: boundary delta carrying both. Stash overwritten.
        stash_reasoning_from(
            DeltaMessage(reasoning="final thought", content="content!")
        )
        self.assertEqual(_pending_reasoning.get(), "final thought")

        # Tool parser runs (reasoning_end_arr just flipped), produces
        # content-only delta. Stash gets attached.
        out = reattach_reasoning_to(DeltaMessage(content="content!"))
        self.assertEqual(out.reasoning, "final thought")
        self.assertEqual(out.reasoning_content, "final thought")
        self.assertEqual(out.content, "content!")

        # Iter 3 (post-boundary): reasoning parser NOT called by
        # serving_chat. Stash empty.
        out_3 = reattach_reasoning_to(DeltaMessage(content="more content"))
        self.assertIsNone(getattr(out_3, "reasoning", None))

    def test_pre_boundary_only_no_leak(self) -> None:
        stash_reasoning_from(DeltaMessage(reasoning="old"))
        self.assertEqual(_pending_reasoning.get(), "old")
        stash_reasoning_from(DeltaMessage(content="post"))
        self.assertIsNone(_pending_reasoning.get())

    def test_disabled_thinking_path_no_op(self) -> None:
        # When PanguReasoningParser short-circuits to content-only
        # (thinking_enabled=False), the stash should clear.
        stash_reasoning_from(DeltaMessage(content="all content"))
        self.assertIsNone(_pending_reasoning.get())
        out = reattach_reasoning_to(DeltaMessage(content="all content"))
        self.assertIsNone(getattr(out, "reasoning", None))


class TestPanguToolParserWrapperDispatch(_RelayTestBase):
    """Smoke: the public ``extract_tool_calls_streaming`` dispatches through
    ``reattach_reasoning_to``. Uses a thin subclass to bypass tokenizer setup."""

    def setUp(self) -> None:
        super().setUp()
        # Build a stub that mimics PanguToolParser's public dispatch
        # without invoking __init__ (which needs a real tokenizer/vocab).
        from omni_npu.v1.parsers.pangu_tool_parser import PanguToolParser

        class _Stub(PanguToolParser):
            def __init__(self) -> None:  # type: ignore[override]
                self._inner_called_with: tuple = ()
                self._inner_return: DeltaMessage | None = None

            def _extract_tool_calls_streaming(self, *args, **kwargs):  # type: ignore[override]
                self._inner_called_with = (args, kwargs)
                return self._inner_return

        self.parser = _Stub()

    def _call(self) -> DeltaMessage | None:
        return self.parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=MagicMock(),
        )

    def test_no_pending_passes_inner_through(self) -> None:
        self.parser._inner_return = DeltaMessage(content="hello")
        out = self._call()
        self.assertEqual(out.content, "hello")
        self.assertIsNone(getattr(out, "reasoning", None))

    def test_pending_reattaches_after_dispatch(self) -> None:
        _pending_reasoning.set("stashed")
        self.parser._inner_return = DeltaMessage(content="hello")
        out = self._call()
        self.assertEqual(out.content, "hello")
        self.assertEqual(out.reasoning, "stashed")
        self.assertEqual(out.reasoning_content, "stashed")
        self.assertIsNone(_pending_reasoning.get())

    def test_pending_reattaches_when_inner_returns_none(self) -> None:
        _pending_reasoning.set("orphan")
        self.parser._inner_return = None
        out = self._call()
        self.assertIsNotNone(out)
        self.assertEqual(out.reasoning, "orphan")
        self.assertEqual(out.reasoning_content, "orphan")
        self.assertIsNone(_pending_reasoning.get())


class TestSerialDriveSimulation(_RelayTestBase):
    """Drive a 3-iteration ``serving_chat``-like sequence through both parsers.

    Crucially, this test exercises the **real** production wrapper methods
    on :class:`PanguReasoningParser` and :class:`PanguToolParser` — only
    the super-call (``DeepSeekR1ReasoningParser.extract_reasoning_streaming``)
    and the private inner of the tool parser are stubbed. That means a
    future commit that deletes the ``stash_reasoning_from(ret)`` line in
    ``pangu_reasoning_parser.py`` or removes the wrapper dispatch in
    ``pangu_tool_parser.py`` will break this test.
    """

    def setUp(self) -> None:
        super().setUp()
        from unittest.mock import patch
        from vllm.reasoning.deepseek_r1_reasoning_parser import (
            DeepSeekR1ReasoningParser,
        )
        from omni_npu.v1.parsers.pangu_reasoning_parser import (
            PanguReasoningParser,
        )
        from omni_npu.v1.parsers.pangu_tool_parser import PanguToolParser

        # Build parser instances WITHOUT invoking __init__ (avoids needing
        # a real tokenizer/vocab). The real wrapper bodies still run, so
        # we have to seed any instance attribute the body reads — when
        # PanguReasoningParser.__init__ grows a new attr (e.g. !1267
        # added tool_call_start_token_id), add it here too.
        self.r = PanguReasoningParser.__new__(PanguReasoningParser)
        self.r.thinking_enabled = True
        self.r.delta_token_ids = []
        # token-id attrs aren't reached on the boundary path we drive
        # (which has end_token_id ∈ previous_token_ids → super handles
        # the split), but seed them defensively for the multi-token-
        # case branch in case a future test extends the matrix.
        self.r.start_token_id = 100
        self.r.end_token_id = 200
        # Disable the implicit-end-via-tool-call-start path (matches the
        # default when PANGU_TOOL_CALL_ENDS_THINKING is unset, which is
        # what __init__ would have left it at anyway).
        self.r.tool_call_start_token_id = None

        self._r_returns: list[DeltaMessage | None] = []
        # Patch the super-call so the REAL PanguReasoningParser body
        # runs (including the production ``stash_reasoning_from(ret)``
        # at the single exit point). The patcher is class-level so
        # ``super().extract_reasoning_streaming(...)`` dispatches to it.
        super_patcher = patch.object(
            DeepSeekR1ReasoningParser,
            "extract_reasoning_streaming",
            side_effect=lambda *a, **kw: (
                self._r_returns.pop(0) if self._r_returns else None
            ),
        )
        super_patcher.start()
        self.addCleanup(super_patcher.stop)

        # Tool parser: build instance, replace only the private inner.
        # The public ``extract_tool_calls_streaming`` is the production
        # dispatcher we want to exercise.
        self.t = PanguToolParser.__new__(PanguToolParser)
        self._t_returns: list[DeltaMessage | None] = []
        self.t._extract_tool_calls_streaming = (  # type: ignore[method-assign]
            lambda *a, **kw: self._t_returns.pop(0) if self._t_returns else None
        )

    def _reason(self, ret: DeltaMessage | None) -> DeltaMessage | None:
        self._r_returns.append(ret)
        return self.r.extract_reasoning_streaming(
            "", "", "", [], [], []
        )

    def _tool(self, ret: DeltaMessage | None) -> DeltaMessage | None:
        self._t_returns.append(ret)
        return self.t.extract_tool_calls_streaming(
            previous_text="",
            current_text="",
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=MagicMock(),
        )

    def test_three_iteration_boundary_sequence(self) -> None:
        # Iter 1 (in-reasoning): reasoning parser emits a reasoning-only
        # delta. Tool parser is NOT called by serving_chat in this branch.
        out_1 = self._reason(DeltaMessage(reasoning="warming up…"))
        self.assertEqual(out_1.reasoning, "warming up…")

        # Iter 2 (boundary): reasoning parser emits both fields. Then
        # tool parser runs (reasoning_end_arr just flipped) and the
        # original bug overwrites the reasoning-bearing delta.
        # The relay should rescue the reasoning text.
        self._reason(DeltaMessage(reasoning="final thought", content="ans"))
        # Simulate serving_chat stripping .content (relay already stashed
        # by this point, so the strip doesn't affect the rescue).
        out_2 = self._tool(DeltaMessage(content="ans"))
        self.assertEqual(out_2.reasoning, "final thought")
        self.assertEqual(out_2.reasoning_content, "final thought")
        self.assertEqual(out_2.content, "ans")

        # Iter 3 (post-boundary): reasoning parser is NOT called (Block A
        # is gated on ``not reasoning_end_arr[i]``). Tool parser alone.
        out_3 = self._tool(DeltaMessage(content="more"))
        self.assertEqual(out_3.content, "more")
        self.assertIsNone(getattr(out_3, "reasoning", None))


if __name__ == "__main__":
    unittest.main()
