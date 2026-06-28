# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

"""Per-task relay carrying the streaming reasoning text across the boundary
between :class:`PanguReasoningParser` and :class:`PanguToolParser`.

Why this exists
---------------

Workaround for an upstream bug in vLLM v0.14.0's
``OpenAIServingChat.chat_completion_stream_generator``
(``vllm/entrypoints/openai/serving_chat.py:1006-1077``). In the streaming
``elif tool_choice_auto and self.reasoning_parser:`` branch, when
``reasoning_end_arr[i]`` flips ``True`` (the iteration where ``</think>``
appears in the delta), the reasoning parser is called first and may return
``DeltaMessage(reasoning="…", content="…")`` for a multi-token chunk that
straddles the boundary. The serving layer strips ``.content`` to ``None``
and stashes it for the next iteration, then in the SAME iteration (line
1067, plain assignment) calls::

    delta_message = tool_parser.extract_tool_calls_streaming(...)

which unconditionally overwrites the reasoning-bearing ``DeltaMessage``,
discarding its ``.reasoning`` text. The boundary chunk's reasoning is
never yielded to the client.

Under MTP-K (K>=2) speculative decoding multi-token boundary chunks are
the common case; under K=1, ``</think>`` typically arrives standalone and
the base parser returns ``None`` (``basic_parsers.py:107-110``), so the
overwrite hits a no-op and the bug stays invisible.

Upstream fix: vLLM PR #42691 (on ``main``, post-v0.21.0) patches
``DelegatingParser.parse_delta`` with snapshot+restore. This module
mirrors that pattern, but localized to the omni-npu Pangu parsers since
:class:`PanguReasoningParser` doesn't inherit from ``DelegatingParser``
and we cannot wait for the upstream tag.

TODO: Delete this module (and the two anchor blocks + the wrapper in
:class:`PanguToolParser`) once vLLM is upgraded to a release containing
PR #42691 AND :class:`PanguReasoningParser`'s parent chain goes through
``DelegatingParser.parse_delta``. Cleanup is ~570 LoC across 4 files.

How it works
------------

:func:`stash_reasoning_from` is called at the end of every Pangu reasoning
streaming call, eagerly clearing the relay then setting it to the
delta's ``.reasoning`` text iff non-empty. :func:`reattach_reasoning_to`
is called at the end of every Pangu tool-parser streaming call: if a
non-empty reasoning is pending, it is attached to the tool parser's
return ``DeltaMessage`` (one is created if the tool parser returned
``None``) and the stash is consumed.

Concurrency note
----------------

The relay is a :class:`contextvars.ContextVar`, scoped per-asyncio-task.
vLLM's streaming generator runs one task per request, and the reasoning
parser → tool parser calls inside one iteration are synchronous (no
``await`` between them). Concurrent requests have isolated stashes.
"""

from __future__ import annotations

import contextvars

from vllm.entrypoints.openai.protocol import DeltaMessage


_pending_reasoning: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "omni_npu_pending_streaming_reasoning", default=None
)


def reset_for_tests() -> None:
    """Reset the relay state. **Test-only.**

    Production code does not need to call this — the relay is a
    :class:`ContextVar`, and each request runs in its own asyncio task,
    so concurrent requests are isolated automatically. ``unittest``
    ``TestCase`` instances, however, share one process-wide Context: a
    test that drives :meth:`PanguReasoningParser.extract_reasoning_streaming`
    leaves a stashed value that a subsequent test invoking
    :meth:`PanguToolParser.extract_tool_calls_streaming` (which expects
    ``None`` for some inputs) will silently inherit. Wire this into an
    autouse fixture (see ``tests/unit/parsers/conftest.py``) or a
    ``setUp``/``tearDown`` hook to enforce per-test isolation.
    """
    _pending_reasoning.set(None)


def stash_reasoning_from(delta: DeltaMessage | None) -> None:
    """Update the relay from a reasoning-parser streaming return value.

    Always clears the relay first so a non-reasoning iteration cannot
    leave a stale stash for the next tool-parser call. Sets the relay to
    ``delta.reasoning`` iff present and non-empty.
    """
    _pending_reasoning.set(None)
    if delta is None:
        return
    reasoning = getattr(delta, "reasoning", None)
    if reasoning:
        _pending_reasoning.set(reasoning)


def reattach_reasoning_to(
    delta: DeltaMessage | None,
) -> DeltaMessage | None:
    """Consume the relay and attach the stashed reasoning to ``delta``.

    If a non-empty reasoning is pending and ``delta`` does not already
    carry one, it is attached (creating a fresh :class:`DeltaMessage` if
    the tool parser returned ``None``). The relay is cleared regardless,
    so the same reasoning never reattaches twice.

    .. note::

        ``DeltaMessage`` has a ``@model_validator(mode="after")`` that
        mirrors ``reasoning`` into the deprecated ``reasoning_content``
        field at construction time (vLLM ``protocol.py:1566-1570``).
        Pydantic v2 validators do **not** re-fire on plain attribute
        assignment, so when we attach to an existing ``delta`` we have
        to set both fields explicitly — otherwise the boundary chunk
        ships with ``reasoning="…", reasoning_content=null`` on the
        wire, and clients still reading the deprecated field see
        ``null`` precisely on the chunk this fix exists to rescue.

        When the tool parser returned ``None`` we build a fresh
        ``DeltaMessage(reasoning=…)`` so the validator handles the
        mirror for us.
    """
    pending = _pending_reasoning.get()
    if not pending:
        return delta
    _pending_reasoning.set(None)
    if delta is None:
        # Construction path: model_validator mirrors reasoning_content.
        return DeltaMessage(reasoning=pending)
    if getattr(delta, "reasoning", None):
        return delta
    # Assignment path: set both fields explicitly because the validator
    # does not re-run for attribute writes.
    delta.reasoning = pending
    delta.reasoning_content = pending
    return delta
