# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for the chat-stream splitter patch.

Covers the pure SSE-line splitter helper ``_maybe_split_sse_line``.
The patch class itself (``OpenAIServingChatStreamSplitPatch``) is a
thin async-generator wrapper around the helper — separately exercising
the wrapper would require driving the full ``chat_completion_stream_
generator`` chain, which is heavier than the regression value justifies.
"""

from __future__ import annotations

import json
import unittest

from omni_npu.vllm_patches.patches.common.patch_split_reasoning_content import (
    _maybe_split_sse_line,
)


def _sse(payload: dict) -> str:
    """Build an SSE data line that mirrors what vLLM's stream generator emits."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _parse_sse(line: str) -> dict:
    """Inverse of :func:`_sse` — pull the JSON payload out of an SSE line."""
    assert line.startswith("data: ") and line.endswith("\n\n"), line
    return json.loads(line[len("data: ") : -2])


class TestPassThrough(unittest.TestCase):
    """Cases where the splitter should return None (caller passes through)."""

    def test_done_marker(self) -> None:
        self.assertIsNone(_maybe_split_sse_line("data: [DONE]\n\n"))

    def test_non_data_line(self) -> None:
        self.assertIsNone(_maybe_split_sse_line(": keepalive\n\n"))
        self.assertIsNone(_maybe_split_sse_line(""))
        self.assertIsNone(_maybe_split_sse_line("\n"))

    def test_malformed_json(self) -> None:
        self.assertIsNone(_maybe_split_sse_line("data: {not json}\n\n"))

    def test_no_choices(self) -> None:
        self.assertIsNone(_maybe_split_sse_line(_sse({"id": "x"})))

    def test_reasoning_only_chunk(self) -> None:
        line = _sse({"choices": [{"delta": {"reasoning": "thinking…"}}]})
        self.assertIsNone(_maybe_split_sse_line(line))

    def test_content_only_chunk(self) -> None:
        line = _sse({"choices": [{"delta": {"content": "hello"}}]})
        self.assertIsNone(_maybe_split_sse_line(line))

    def test_tool_calls_only_chunk(self) -> None:
        line = _sse(
            {"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]}
        )
        self.assertIsNone(_maybe_split_sse_line(line))

    def test_empty_reasoning_string(self) -> None:
        # Falsy reasoning + content shouldn't trigger a split.
        line = _sse({"choices": [{"delta": {"reasoning": "", "content": "x"}}]})
        self.assertIsNone(_maybe_split_sse_line(line))


class TestSplit(unittest.TestCase):
    """Cases where the splitter should produce two SSE events."""

    def test_reasoning_and_content_split(self) -> None:
        line = _sse(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning": "final thought",
                            "reasoning_content": "final thought",
                            "content": "answer!",
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )
        out = _maybe_split_sse_line(line)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 2)

        first = _parse_sse(out[0])
        second = _parse_sse(out[1])

        # Reasoning event carries reasoning fields only.
        self.assertEqual(
            first["choices"][0]["delta"],
            {"reasoning": "final thought", "reasoning_content": "final thought"},
        )
        # Content event carries content only.
        self.assertEqual(second["choices"][0]["delta"], {"content": "answer!"})

        # Top-level metadata (id, object) is preserved on both events.
        self.assertEqual(first["id"], "chatcmpl-1")
        self.assertEqual(second["id"], "chatcmpl-1")

    def test_reasoning_and_tool_calls_split(self) -> None:
        # Tool envelope opening alongside the final reasoning text.
        line = _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning": "decided to call get_weather",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "get_weather"},
                                }
                            ],
                        },
                    }
                ],
            }
        )
        out = _maybe_split_sse_line(line)
        self.assertIsNotNone(out)

        first = _parse_sse(out[0])
        second = _parse_sse(out[1])

        self.assertEqual(first["choices"][0]["delta"].get("reasoning"),
                         "decided to call get_weather")
        self.assertNotIn("tool_calls", first["choices"][0]["delta"])
        self.assertIn("tool_calls", second["choices"][0]["delta"])
        self.assertNotIn("reasoning", second["choices"][0]["delta"])

    def test_role_stays_on_reasoning_event(self) -> None:
        # role belongs on the first chunk for OpenAI compatibility —
        # if the boundary chunk happens to be first-with-role, the
        # reasoning event must carry it.
        line = _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning": "first thought",
                            "content": "first content",
                        },
                    }
                ],
            }
        )
        out = _maybe_split_sse_line(line)
        first = _parse_sse(out[0])
        second = _parse_sse(out[1])
        self.assertEqual(first["choices"][0]["delta"].get("role"), "assistant")
        self.assertNotIn("role", second["choices"][0]["delta"])

    def test_finish_reason_lands_on_content_event(self) -> None:
        # finish_reason ends the stream — must appear only on the LAST
        # of the split pair (the content event).
        line = _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning": "wrapping up",
                            "content": "done.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        out = _maybe_split_sse_line(line)
        first = _parse_sse(out[0])
        second = _parse_sse(out[1])
        self.assertIsNone(first["choices"][0]["finish_reason"])
        self.assertEqual(second["choices"][0]["finish_reason"], "stop")

    def test_logprobs_lands_on_content_event(self) -> None:
        line = _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning": "r",
                            "content": "c",
                        },
                        "logprobs": {"content": [{"token": "c"}]},
                    }
                ],
            }
        )
        out = _maybe_split_sse_line(line)
        first = _parse_sse(out[0])
        second = _parse_sse(out[1])
        self.assertIsNone(first["choices"][0]["logprobs"])
        self.assertEqual(second["choices"][0]["logprobs"],
                         {"content": [{"token": "c"}]})

    def test_stop_reason_lands_on_content_event(self) -> None:
        # stop_reason is a vLLM extension on ChatCompletionResponseStreamChoice
        # (protocol.py:1578). Same placement rule as finish_reason: only
        # the content half can carry a stream-terminating reason.
        line = _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning": "r", "content": "c"},
                        "stop_reason": "max_tokens",
                    }
                ],
            }
        )
        out = _maybe_split_sse_line(line)
        first = _parse_sse(out[0])
        second = _parse_sse(out[1])
        self.assertIsNone(first["choices"][0]["stop_reason"])
        self.assertEqual(second["choices"][0]["stop_reason"], "max_tokens")

    def test_token_ids_lands_on_content_event(self) -> None:
        # token_ids ships on every streaming chunk when
        # request.return_token_ids=True (serving_chat.py:1255-1259).
        # The reasoning half emits no additional tokens — duplicating
        # token_ids would make clients that concatenate per-chunk
        # token_ids double-count.
        line = _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning": "r", "content": "c"},
                        "token_ids": [101, 102, 103],
                    }
                ],
            }
        )
        out = _maybe_split_sse_line(line)
        first = _parse_sse(out[0])
        second = _parse_sse(out[1])
        self.assertIsNone(first["choices"][0]["token_ids"])
        self.assertEqual(second["choices"][0]["token_ids"], [101, 102, 103])

    def test_all_four_trailing_fields_land_on_content_event(self) -> None:
        # Lock the full contract: all of finish_reason, logprobs,
        # stop_reason, token_ids belong only to the content half.
        # A regression that drops any of the four from the cleared set
        # would fail this test.
        line = _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning": "r", "content": "c"},
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"token": "c"}]},
                        "stop_reason": 49152,
                        "token_ids": [101, 102, 103],
                    }
                ],
            }
        )
        out = _maybe_split_sse_line(line)
        first = _parse_sse(out[0])
        second = _parse_sse(out[1])
        for field in ("finish_reason", "logprobs", "stop_reason", "token_ids"):
            self.assertIsNone(
                first["choices"][0][field],
                f"{field} should be None on the reasoning half",
            )
        self.assertEqual(second["choices"][0]["finish_reason"], "stop")
        self.assertEqual(
            second["choices"][0]["logprobs"],
            {"content": [{"token": "c"}]},
        )
        self.assertEqual(second["choices"][0]["stop_reason"], 49152)
        self.assertEqual(second["choices"][0]["token_ids"], [101, 102, 103])

    def test_n_gt_1_passes_through_unchanged(self) -> None:
        # Splitting a multi-choice event by rebuilding with only
        # ``choices[0]``'s halves would silently drop ``choices[1:]``
        # from both wire events — data loss for parallel-sampling
        # clients. The patch refuses to split when ``len(choices) != 1``
        # and passes the original event through unchanged instead.
        line = _sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning": "r0", "content": "c0"},
                    },
                    {
                        "index": 1,
                        "delta": {"content": "c1"},
                    },
                ],
            }
        )
        self.assertIsNone(_maybe_split_sse_line(line))

    def test_split_lines_are_well_formed_sse(self) -> None:
        line = _sse(
            {"choices": [{"delta": {"reasoning": "r", "content": "c"}}]}
        )
        out = _maybe_split_sse_line(line)
        for event in out:
            self.assertTrue(event.startswith("data: "))
            self.assertTrue(event.endswith("\n\n"))
            # The interior must be valid JSON.
            json.loads(event[len("data: ") : -2])


if __name__ == "__main__":
    unittest.main()
