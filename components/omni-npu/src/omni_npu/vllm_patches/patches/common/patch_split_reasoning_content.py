# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Split combined reasoning+content DeltaMessage chunks into separate SSE events.

vLLM's streaming format permits a single ``DeltaMessage`` to carry both
``.reasoning`` and ``.content`` (or ``.tool_calls``) — and the omni-npu
reasoning relay at ``omni_npu/v1/parsers/_streaming_relay.py`` produces
exactly that shape on the ``</think>`` boundary chunk under MTP-K
speculative decoding. Some OpenAI-style clients render only the first
non-null delta field per chunk and silently drop the other; this patch
rewrites every boundary chunk as TWO consecutive SSE events on the
wire:

  data: {"choices":[{"delta":{"reasoning":"…"}}]}\\n\\n
  data: {"choices":[{"delta":{"content":"…"}}]}\\n\\n

No parser logic is touched — the split happens at the SSE-string layer
inside the chat-completion streaming generator, after the upstream
chain has already produced its final ``DeltaMessage``.

Chain position
--------------

This patch is the tail of the ``chat_completion_stream_generator``
chain. Filename sort order places it after ``patch_serving_apc.py``
(which is the current tail), so the import of
``OpenAIServingChatStreamAPCPatch`` resolves, and re-registering the
same ``ExpertIdServingChatStream`` name overrides the registry entry
serving_apc inserted, while still chaining via the captured
``_orig_chat_stream``.
"""

from __future__ import annotations

import json
import re
from itertools import count

from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.logger import logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.patches.common.patch_serving_apc import (
    OpenAIServingChatStreamAPCPatch,
)


# Per-process counter for "split #N" log lines. Atomicity isn't critical
# (asyncio is single-threaded; if multiple workers each run their own
# event loop, each gets its own counter), it's just for human counting.
_split_counter = count(1)


# Capture the current tail of the chain so our wrapper can delegate to it.
_orig_chat_stream = OpenAIServingChatStreamAPCPatch.__dict__[
    "chat_completion_stream_generator"
]


# Match a single SSE data line. The streaming generator yields one event
# per ``yield``, formatted as ``data: <json>\\n\\n`` (or ``data: [DONE]
# \\n\\n`` at the very end). Lines that don't match this exact shape —
# heartbeats, comments, multi-event yields — are passed through.
_SSE_DATA_LINE_RE = re.compile(r"^data: (.+)\n\n$", re.DOTALL)


def _maybe_split_sse_line(line: str) -> list[str] | None:
    """Return ``[reasoning_event, content_event]`` if ``line`` is a single
    SSE data event whose first choice's delta carries BOTH reasoning
    (``reasoning`` or ``reasoning_content``) AND content (``content`` or
    ``tool_calls``); otherwise return ``None`` so the caller passes the
    original line through unchanged.

    Splitting rules:

    * Reasoning chunk gets the ``role`` field if present (preserves the
      "first chunk carries role" invariant some clients rely on).
    * Content chunk keeps every trailing choice-level field that
      semantically describes the chunk's emitted tokens —
      ``finish_reason``, ``logprobs``, ``stop_reason``, ``token_ids``
      (the last two are vLLM extensions; ``token_ids`` ships on every
      chunk when ``request.return_token_ids=True``). The reasoning
      half clears all four so clients that concatenate per-chunk
      ``token_ids`` don't double-count, and so a stream-terminating
      ``finish_reason``/``stop_reason`` only fires on the content half.
    * Only ``n=1`` events are split. Multi-choice (``n>1``) events pass
      through unchanged — naïvely rebuilding the event with only
      ``choices[0]``'s halves would silently drop ``choices[1:]``, which
      is data loss for parallel-sampling clients. Real ``n>1`` support
      would require splitting each choice's delta independently and
      reconciling the two halves of every choice into the rebuilt
      events, which is out of scope.
    """
    m = _SSE_DATA_LINE_RE.match(line)
    if not m:
        return None
    body = m.group(1)
    if body == "[DONE]":
        return None
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None

    choices = obj.get("choices")
    if not choices:
        return None
    # Refuse to split multi-choice (n>1) events. The rebuilt events at
    # the bottom of this function carry ONLY choices[0]'s halves; for
    # n>1 that would silently drop choices[1:] from both events, which
    # is data loss, not a documented limitation. Pass the original
    # combined chunk through unchanged — for n>1 clients the legacy
    # combined-fields wire format remains, which is at worst a missed
    # split, not a corrupted stream.
    if len(choices) != 1:
        return None
    choice0 = choices[0]
    delta = choice0.get("delta") or {}
    
    if delta.get("content") == "" and delta.get("reasoning_content") is None:
        if delta.get("role") == "assistant":
            del delta["content"]
        else:
            del delta["reasoning_content"]
        choice = {**choice0, "delta": delta}
        event = {**obj, "choices": [choice]}
        return [f"data: {json.dumps(event)}\n\n"]
    
    has_reasoning = bool(delta.get("reasoning")) or bool(delta.get("reasoning_content"))
    has_content_or_tools = bool(delta.get("content")) or bool(delta.get("tool_calls"))
    
    if has_reasoning and delta.get("content") == "":
        del delta["content"]
        choice = {**choice0, "delta": delta}
        event = {**obj, "choices": [choice]}
        return [f"data: {json.dumps(event)}\n\n"]
    
    if not has_content_or_tools:
        return None

    # Log each actual split so the patch's effect is verifiable in
    # server logs. One INFO line per boundary chunk, which on a typical
    # Pangu workload is at most one per request that mixes thinking +
    # a tool call.
    n = next(_split_counter)
    reasoning_str = (
        delta.get("reasoning") or delta.get("reasoning_content") or ""
    )
    content_str = delta.get("content") or ""
    tool_calls_count = len(delta.get("tool_calls") or [])
    if has_reasoning:
        logger.info(
            "patch_split_reasoning_content: split #%d id=%s "
            "reasoning=%d chars %r content=%d chars %r tool_calls=%d",
            n,
            obj.get("id", "?"),
            len(reasoning_str),
            reasoning_str[:32],
            len(content_str),
            content_str[:32],
            tool_calls_count,
        )

    reasoning_delta = {
        k: v for k, v in delta.items() if k not in ("content", "tool_calls")
    }
    content_delta = {
        k: v
        for k, v in delta.items()
        if k not in ("reasoning", "reasoning_content", "role")
    }

    reasoning_choice = {**choice0, "delta": reasoning_delta}
    content_choice = {**choice0, "delta": content_delta}
    # Trailing choice-level fields belong to the final chunk of the pair
    # (i.e. the content half) — clear them on the reasoning half.
    #
    # - ``finish_reason`` / ``stop_reason``: the stream's terminating
    #   reason; can only fire when the actual final tokens are emitted,
    #   which is the content side.
    # - ``logprobs`` / ``token_ids``: per-chunk token traces (``token_ids``
    #   is a vLLM extension populated on every chunk when
    #   ``request.return_token_ids=True``, see
    #   ``vllm/entrypoints/openai/serving_chat.py:1255-1259``). The
    #   reasoning half is a synthetic re-presentation that emits no
    #   additional tokens, so duplicating either would cause clients
    #   that concatenate per-chunk traces to double-count.
    for _trailing in ("finish_reason", "logprobs", "stop_reason", "token_ids"):
        if _trailing in choice0:
            reasoning_choice[_trailing] = None

    reasoning_event = {**obj, "choices": [reasoning_choice]}
    content_event = {**obj, "choices": [content_choice]}
    lines = []
    if has_reasoning:
        lines.append(f"data: {json.dumps(reasoning_event)}\n\n")
    lines.append(f"data: {json.dumps(content_event)}\n\n")
    return lines


@register_patch("ExpertIdServingChatStream", OpenAIServingChat)
class OpenAIServingChatStreamSplitPatch(VLLMPatch):
    """Tail patch on ``chat_completion_stream_generator`` — split combined
    reasoning+content SSE events into two separate events on the wire.
    """

    _attr_names_to_apply = ["chat_completion_stream_generator"]

    async def chat_completion_stream_generator(self, *args, **kwargs):
        async for raw in _orig_chat_stream(self, *args, **kwargs):
            split = _maybe_split_sse_line(raw)
            if split is None:
                yield raw
            else:
                for event in split:
                    yield event


# One-shot info log at module import so the server log confirms the patch
# is wired into the chain. If you don't see this line at worker startup,
# the file wasn't imported (auto-discovery skipped it or filename order
# didn't reach it).
logger.info(
    "patch_split_reasoning_content: loaded — combined reasoning+content "
    "SSE events will be split into two events on the wire"
)
