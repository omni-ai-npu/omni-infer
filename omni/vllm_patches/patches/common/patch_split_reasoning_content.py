# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Split combined reasoning+content DeltaMessage chunks into separate SSE events.

vLLM's streaming format permits a single ``DeltaMessage`` to carry both
``.reasoning`` and ``.content`` (or ``.tool_calls``). Legacy and third-party
parser combinations may still produce that shape on the ``</think>``
boundary chunk under speculative decoding. Some OpenAI-style clients render only
the first
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

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
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


def _token_count(tokenizer, *texts: str) -> int:
    """Token count of ``texts``; 0 when there is nothing to encode."""
    text = "".join(texts)
    encode = getattr(tokenizer, "encode", None)
    if not text or encode is None:
        return 0
    try:
        return len(encode(text, add_special_tokens=False))
    except TypeError:
        logger.debug(
            "tokenizer.encode does not accept add_special_tokens; "
            "retrying without that argument"
        )
        return len(encode(text))


def _content_token_count(choice0: dict, tokenizer) -> int:
    """How many tokens of this mixed chunk sit after ``</think>``.

    Prefer ``token_ids`` (the engine step's own ids, present when
    ``return_token_ids=True``); otherwise encode the text that moves to
    the content SSE.
    """
    vocab = getattr(tokenizer, "get_vocab", dict)() or {}
    end_id = vocab.get("</think>", vocab.get("[unused17]"))
    ids = choice0.get("token_ids")
    if isinstance(ids, list) and end_id in ids:
        return len(ids) - ids.index(end_id) - 1

    delta = choice0.get("delta") or {}
    n = _token_count(tokenizer, delta.get("content") or "")
    for tool_call in delta.get("tool_calls") or []:
        fn = tool_call.get("function") or {}
        n += _token_count(tokenizer, fn.get("name") or "", fn.get("arguments") or "")
    return n


def _reasoning_event_with_usage(
    obj: dict,
    reasoning_choice: dict,
    choice0: dict,
    tokenizer,
) -> dict:
    """Keep engine ``usage`` on the content SSE; shrink it on the reasoning SSE.

    ``completion_tokens`` on the mixed chunk already counts the content
    tokens after ``</think>``, so subtract that real count -- at least
    one, so the two halves never report the same total.
    """
    event = {**obj, "choices": [reasoning_choice]}
    usage = obj.get("usage")
    ct = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if not isinstance(ct, int) or ct <= 0:
        return event

    reasoning_ct = ct - min(max(_content_token_count(choice0, tokenizer), 1), ct - 1)
    event["usage"] = {**usage, "completion_tokens": reasoning_ct}
    prompt_tokens = usage.get("prompt_tokens")
    if isinstance(prompt_tokens, int):
        event["usage"]["total_tokens"] = prompt_tokens + reasoning_ct
    return event


def _rewrite_named_finish_reason_sse(
    event: str,
    *,
    saw_tool_calls: bool,
) -> tuple[str, bool]:
    """For named tool_choice, map terminal finish_reason stop→tool_calls.

    Only rewrites after a tool_calls delta has already been seen in this
    stream (same gate as the non-streaming full-generator override).
    """
    match = _SSE_DATA_LINE_RE.match(event)
    if match is None:
        return event, saw_tool_calls
    payload_text = match.group(1).strip()
    if not payload_text or payload_text == "[DONE]":
        return event, saw_tool_calls
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        logger.debug(
            "skipping non-JSON SSE payload in named finish_reason rewrite"
        )
        return event, saw_tool_calls

    changed = False
    for choice in payload.get("choices") or []:
        if (choice.get("delta") or {}).get("tool_calls"):
            saw_tool_calls = True
        if saw_tool_calls and choice.get("finish_reason") == "stop":
            choice["finish_reason"] = "tool_calls"
            changed = True
    if changed:
        event = f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    return event, saw_tool_calls


def _maybe_split_sse_line(line: str, tokenizer=None) -> list[str] | None:
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
    match = _SSE_DATA_LINE_RE.match(line)
    if match is None:
        return None
    body = match.group(1)
    if body == "[DONE]":
        return None
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        logger.debug("skipping non-JSON SSE payload in mixed-delta split")
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
            delta.pop("content", None)
        else:
            delta.pop("reasoning_content", None)
        choice = {**choice0, "delta": delta}
        event = {**obj, "choices": [choice]}
        return [f"data: {json.dumps(event, separators=(',', ':'))}\n\n"]

    has_reasoning = bool(delta.get("reasoning")) or bool(
        delta.get("reasoning_content")
    )
    has_content_or_tools = bool(delta.get("content")) or bool(
        delta.get("tool_calls")
    )

    if has_reasoning and delta.get("content") == "":
        delta.pop("content", None)
        choice = {**choice0, "delta": delta}
        event = {**obj, "choices": [choice]}
        return [f"data: {json.dumps(event, separators=(',', ':'))}\n\n"]

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
    #   ``vllm/entrypoints/openai/chat_completion/serving.py``). The
    #   reasoning half is a synthetic re-presentation that emits no
    #   additional tokens, so duplicating either would cause clients
    #   that concatenate per-chunk traces to double-count.
    for _trailing in ("finish_reason", "logprobs", "stop_reason", "token_ids"):
        if _trailing in choice0:
            reasoning_choice[_trailing] = None

    reasoning_event = _reasoning_event_with_usage(
        obj, reasoning_choice, choice0, tokenizer
    )
    content_event = {**obj, "choices": [content_choice]}
    lines = []
    if has_reasoning:
        lines.append(f"data: {json.dumps(reasoning_event, separators=(',', ':'))}\n\n")
    lines.append(f"data: {json.dumps(content_event, separators=(',', ':'))}\n\n")
    return lines


@register_patch("ExpertIdServingChatStream", OpenAIServingChat)
class OpenAIServingChatStreamSplitPatch(VLLMPatch):
    """Tail patch on ``chat_completion_stream_generator`` — split combined
    reasoning+content SSE events into two separate events on the wire.
    """

    _attr_names_to_apply = ["chat_completion_stream_generator"]

    async def chat_completion_stream_generator(self, *args, **kwargs):
        """Yield SSE events, splitting mixed reasoning+content chunks."""
        request = args[0] if args else kwargs.get("request")
        is_named = isinstance(
            getattr(request, "tool_choice", None),
            ChatCompletionNamedToolChoiceParam,
        )
        saw_tool_calls = False
        tokenizer = args[5] if len(args) > 5 else kwargs.get("tokenizer")
        async for raw in _orig_chat_stream(self, *args, **kwargs):
            events = _maybe_split_sse_line(raw, tokenizer) or [raw]
            for event in events:
                if is_named:
                    event, saw_tool_calls = _rewrite_named_finish_reason_sse(
                        event, saw_tool_calls=saw_tool_calls
                    )
                yield event


# One-shot info log at module import so the server log confirms the patch
# is wired into the chain. If you don't see this line at worker startup,
# the file wasn't imported (auto-discovery skipped it or filename order
# didn't reach it).
logger.info(
    "patch_split_reasoning_content: loaded — combined reasoning+content "
    "SSE events will be split into two events on the wire"
)
