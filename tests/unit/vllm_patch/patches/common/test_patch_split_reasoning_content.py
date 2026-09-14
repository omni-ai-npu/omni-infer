# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Offline tests for patch_split_reasoning_content.

Stubs every vLLM / omni_npu import so this file runs without torch/NPU::

    pytest tests/unit/vllm_patch/patches/common/test_patch_split_reasoning_content.py -v
    python tests/unit/vllm_patch/patches/common/test_patch_split_reasoning_content.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

PATCH_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / "omni" / "vllm_patches").is_dir())
    / "omni/vllm_patches/patches/common/patch_split_reasoning_content.py"
)

_TESTS_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "unit" / "vllm_patch").is_dir()
)
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from unit.vllm_patch.patches.patch_test_utils import run_standalone_tests  # noqa: E402

REGISTERED: list[tuple[str, object]] = []
UPSTREAM_CHUNKS: list[str] = []


class _VLLMPatch:
    _attr_names_to_apply: list[str] = []


class ChatCompletionNamedToolChoiceParam:
    pass


def _register_patch(name, target):

    def decorator(cls):
        REGISTERED.append((name, target))
        return cls

    return decorator


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class _FakeAPC:
    async def chat_completion_stream_generator(self, *args, **kwargs):
        for chunk in UPSTREAM_CHUNKS:
            yield chunk


class _FakeTokenizer:
    def get_vocab(self):
        return {"</think>": 17, "[unused17]": 99}

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class _EncodeKwonlyTokenizer:
    """encode() rejects unexpected kwargs so the TypeError fallback is hit."""

    def get_vocab(self):
        return {}

    def encode(self, text):
        return list(range(len(text)))


def _install_stubs():
    for pkg in (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
        "omni_npu",
        "omni_npu.vllm_patches",
        "omni_npu.vllm_patches.patches",
        "omni_npu.vllm_patches.patches.common",
    ):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))

    class OpenAIServingChat:
        async def chat_completion_stream_generator(self, *a, **kw):
            raise AssertionError("vLLM fallback should not run")

    _module(
        "vllm.entrypoints.openai.chat_completion.protocol",
        ChatCompletionNamedToolChoiceParam=ChatCompletionNamedToolChoiceParam,
    )
    _module(
        "vllm.entrypoints.openai.chat_completion.serving",
        OpenAIServingChat=OpenAIServingChat,
    )
    _module("vllm.logger", logger=logging.getLogger("test_split_reasoning"))
    _module(
        "omni_npu.vllm_patches.core",
        VLLMPatch=_VLLMPatch,
        register_patch=_register_patch,
    )
    _module(
        "omni_npu.vllm_patches.patches.common.patch_serving_apc",
        OpenAIServingChatStreamAPCPatch=_FakeAPC,
    )


def _load_patch_module():
    saved = dict(sys.modules)
    try:
        _install_stubs()
        spec = importlib.util.spec_from_file_location(
            "_patch_split_reasoning_content", PATCH_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, prev in saved.items():
            if sys.modules.get(name) is not prev:
                sys.modules[name] = prev
        for name in list(sys.modules):
            if name not in saved:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    split = _load_patch_module()
else:
    split = None

    @pytest.fixture(scope="module", autouse=True)
    def _isolated_split_module():
        global split
        with patch.dict(sys.modules):
            split = _load_patch_module()
            yield
        split = None


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


def _payload(line: str) -> dict:
    assert line.startswith("data: ")
    return json.loads(line[len("data: "):].strip())


async def _collect(agen):
    return [chunk async for chunk in agen]


def _mixed(**extra):
    choice = {
        "index": 0,
        "delta": {"role": "assistant", "reasoning": "think-text", "content": "hi"},
        "finish_reason": "stop",
        "stop_reason": 17,
        "token_ids": [1, 17, 8, 9],
        **extra,
    }
    return {
        "id": "cmpl-1",
        "choices": [choice],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 988,
            "total_tokens": 998,
        },
    }


def test_registers_as_chat_stream_tail():
    names = [name for name, _ in REGISTERED]
    assert names[-1] == "ExpertIdServingChatStream"
    assert split._orig_chat_stream is _FakeAPC.__dict__[
        "chat_completion_stream_generator"
    ]


def test_passthrough_done_and_non_sse():
    assert split._maybe_split_sse_line("data: [DONE]\n\n") is None
    assert split._maybe_split_sse_line(": heartbeat\n\n") is None
    assert split._maybe_split_sse_line(_sse({"choices": []})) is None


def test_passthrough_reasoning_only():
    line = _sse({"choices": [{"delta": {"reasoning": "only"}}]})
    assert split._maybe_split_sse_line(line) is None


def test_refuses_to_split_n_greater_than_one():
    line = _sse({
        "choices": [
            {"delta": {"reasoning": "a", "content": "b"}},
            {"delta": {"reasoning": "c", "content": "d"}},
        ]
    })
    assert split._maybe_split_sse_line(line) is None


def test_strips_empty_content_on_role_chunk():
    line = _sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
    out = split._maybe_split_sse_line(line)
    assert len(out) == 1
    assert "content" not in _payload(out[0])["choices"][0]["delta"]


def test_empty_content_without_reasoning_key_does_not_raise():
    """content=='' and missing reasoning_content must not KeyError."""
    line = _sse({"choices": [{"delta": {"content": ""}}]})
    out = split._maybe_split_sse_line(line)
    assert len(out) == 1
    assert _payload(out[0])["choices"][0]["delta"].get("content") == ""


def test_strips_empty_content_when_reasoning_present():
    """``reasoning_content`` present (even empty) skips the role-chunk branch."""
    line = _sse({
        "choices": [{
            "delta": {"reasoning": "abc", "reasoning_content": "", "content": ""}
        }]
    })
    out = split._maybe_split_sse_line(line)
    assert len(out) == 1
    delta = _payload(out[0])["choices"][0]["delta"]
    assert delta["reasoning"] == "abc"
    assert "content" not in delta


def test_splits_reasoning_and_content_and_clears_trailing_fields():
    tok = _FakeTokenizer()
    out = split._maybe_split_sse_line(_sse(_mixed()), tok)
    assert len(out) == 2
    reasoning, content = map(_payload, out)

    assert reasoning["choices"][0]["delta"] == {
        "role": "assistant",
        "reasoning": "think-text",
    }
    assert reasoning["choices"][0]["finish_reason"] is None
    assert reasoning["choices"][0]["token_ids"] is None
    assert reasoning["choices"][0]["stop_reason"] is None

    assert content["choices"][0]["delta"] == {"content": "hi"}
    assert content["choices"][0]["finish_reason"] == "stop"
    assert content["choices"][0]["token_ids"] == [1, 17, 8, 9]
    assert content["usage"]["completion_tokens"] == 988


def test_reasoning_usage_uses_token_ids_after_think_end():
    """token_ids [1, 17, 8, 9] → two tokens after </think>=17."""
    tok = _FakeTokenizer()
    reasoning = _payload(split._maybe_split_sse_line(_sse(_mixed()), tok)[0])
    assert reasoning["usage"]["completion_tokens"] == 986
    assert reasoning["usage"]["total_tokens"] == 996


def test_reasoning_usage_encodes_content_when_token_ids_missing():
    tok = _FakeTokenizer()
    obj = _mixed()
    obj["choices"][0].pop("token_ids")
    obj["choices"][0]["delta"]["content"] = "xyz"  # 3 tokens via encode
    reasoning = _payload(split._maybe_split_sse_line(_sse(obj), tok)[0])
    assert reasoning["usage"]["completion_tokens"] == 985


def test_reasoning_usage_encodes_tool_call_text():
    tok = _FakeTokenizer()
    obj = {
        "choices": [{
            "delta": {
                "reasoning": "r",
                "tool_calls": [{"function": {"name": "ab", "arguments": "cd"}}],
            }
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 20, "total_tokens": 24},
    }
    reasoning = _payload(split._maybe_split_sse_line(_sse(obj), tok)[0])
    # name "ab" + args "cd" → 4 tokens
    assert reasoning["usage"]["completion_tokens"] == 16


def test_reasoning_usage_falls_back_to_one_without_tokenizer():
    reasoning = _payload(split._maybe_split_sse_line(_sse(_mixed()), None)[0])
    assert reasoning["usage"]["completion_tokens"] == 987


def test_reasoning_usage_encode_without_special_tokens_kwarg():
    tok = _EncodeKwonlyTokenizer()
    obj = _mixed()
    obj["choices"][0].pop("token_ids")
    obj["choices"][0]["delta"]["content"] = "xy"
    reasoning = _payload(split._maybe_split_sse_line(_sse(obj), tok)[0])
    assert reasoning["usage"]["completion_tokens"] == 986


def test_no_usage_still_splits():
    obj = _mixed()
    del obj["usage"]
    out = split._maybe_split_sse_line(_sse(obj), _FakeTokenizer())
    assert len(out) == 2
    assert "usage" not in _payload(out[0])


def test_legacy_reasoning_content_field():
    obj = {
        "choices": [{"delta": {"reasoning_content": "old", "content": "x"}}],
        "usage": {"completion_tokens": 5, "prompt_tokens": 1},
    }
    out = split._maybe_split_sse_line(_sse(obj), _FakeTokenizer())
    assert _payload(out[0])["choices"][0]["delta"]["reasoning_content"] == "old"
    assert _payload(out[1])["choices"][0]["delta"]["content"] == "x"


def test_named_finish_reason_rewritten_after_tool_calls():
    event = _sse({
        "choices": [{"delta": {"tool_calls": [{"index": 0}]}, "finish_reason": "stop"}]
    })
    rewritten, saw = split._rewrite_named_finish_reason_sse(
        event, saw_tool_calls=False
    )
    assert saw
    assert _payload(rewritten)["choices"][0]["finish_reason"] == "tool_calls"


def test_named_finish_reason_unchanged_before_tool_calls():
    event = _sse({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]})
    rewritten, saw = split._rewrite_named_finish_reason_sse(
        event, saw_tool_calls=False
    )
    assert not saw
    assert rewritten == event


def test_stream_generator_emits_two_sse_then_done():
    UPSTREAM_CHUNKS[:] = [_sse(_mixed()), "data: [DONE]\n\n"]
    request = types.SimpleNamespace(tool_choice=None)
    gen = split.OpenAIServingChatStreamSplitPatch.chat_completion_stream_generator(
        object(),
        request,
        None,
        "req",
        "model",
        [],
        _FakeTokenizer(),
    )
    chunks = asyncio.run(_collect(gen))
    assert len(chunks) == 3
    assert _payload(chunks[0])["choices"][0]["delta"]["reasoning"] == "think-text"
    assert _payload(chunks[1])["choices"][0]["delta"]["content"] == "hi"
    assert chunks[2] == "data: [DONE]\n\n"


def test_stream_generator_rewrites_named_tool_choice_stop():
    mixed = {
        "choices": [{
            "delta": {
                "reasoning": "r",
                "tool_calls": [{"index": 0, "function": {"name": "f"}}],
            },
            "finish_reason": "stop",
        }]
    }
    UPSTREAM_CHUNKS[:] = [_sse(mixed)]
    request = types.SimpleNamespace(tool_choice=ChatCompletionNamedToolChoiceParam())
    gen = split.OpenAIServingChatStreamSplitPatch.chat_completion_stream_generator(
        object(), request, None, "req", "model", [], None,
    )
    chunks = asyncio.run(_collect(gen))
    assert _payload(chunks[0])["choices"][0].get("finish_reason") is None
    assert _payload(chunks[1])["choices"][0]["finish_reason"] == "tool_calls"


if __name__ == "__main__":
    run_standalone_tests(globals())
