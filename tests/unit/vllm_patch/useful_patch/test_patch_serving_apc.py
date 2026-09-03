# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Offline tests for patch_serving_apc (vLLM 0.25.1 adaptation).

Every vLLM and omni_npu dependency is stubbed, so this runs with the standard
library alone -- no vllm, no torch, no pydantic, no NPU.  Deliberately placed
under tests/vllm_patches/ rather than tests/unit/ for the same reason as
tests/config/: tests/unit/conftest.py imports torch at module level, which would
defeat the point.  Run either way::

    pytest tests/unit/vllm_patch/useful_patch/test_patch_serving_apc.py -v
    python tests/unit/vllm_patch/useful_patch/test_patch_serving_apc.py

The fake "upstream" generators below reproduce 0.25.1's observable behaviour:
  * the final SSE usage chunk carries prompt_tokens_details only when
    enable_prompt_tokens_details is on (exclude_none=True drops it otherwise);
  * the non-streaming response already has usage.prompt_tokens_details filled in
    from the local engine count;
  * the new chat_template_kwargs / mm_token_counts / parser keywords are passed
    by keyword.
"""

import asyncio
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# <repo>/tests/unit/vllm_patch/useful_patch/ -> parents[4] is <repo>
PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "omni/vllm_patches/usefull_patch/common/patch_serving_apc.py"
)

REGISTERED: list[tuple[str, object]] = []


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


class _PromptTokenUsageInfo:
    def __init__(self, cached_tokens=None, cached_rate=None, multimodal_tokens=None):
        self.cached_tokens = cached_tokens
        self.cached_rate = cached_rate
        self.multimodal_tokens = multimodal_tokens


class _ErrorResponse:
    pass


class _VLLMPatch:
    _attr_names_to_apply: list[str] = []


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


def _install_stubs():
    """Fake out every import patch_serving_apc performs at module level."""
    for pkg in (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.entrypoints.openai.completion",
        "vllm.entrypoints.openai.engine",
        "omni_npu",
        "omni_npu.vllm_patches",
        "omni_npu.vllm_patches.patches",
        "omni_npu.vllm_patches.patches.common",
    ):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))

    # These stand in for the real vLLM classes, so they must expose every method
    # patch_serving_apc captures at import time as a chain fallback.
    class OpenAIServingChat:
        async def chat_completion_stream_generator(self, *a, **kw):
            raise AssertionError("vLLM fallback should not run in these tests")

        async def chat_completion_full_generator(self, *a, **kw):
            raise AssertionError("vLLM fallback should not run in these tests")

    class OpenAIServingCompletion:
        async def completion_stream_generator(self, *a, **kw):
            raise AssertionError("vLLM fallback should not run in these tests")

        # captured as _orig_compl_create at import time
        async def create_completion(self, request, raw_request=None):
            raise AssertionError("stub original should be monkeypatched per test")

    _module(
        "vllm.entrypoints.openai.chat_completion.serving",
        OpenAIServingChat=OpenAIServingChat,
    )
    _module(
        "vllm.entrypoints.openai.completion.serving",
        OpenAIServingCompletion=OpenAIServingCompletion,
    )
    _module(
        "vllm.entrypoints.openai.engine.protocol",
        ErrorResponse=_ErrorResponse,
        PromptTokenUsageInfo=_PromptTokenUsageInfo,
    )
    _module("vllm.logger", init_logger=lambda name: logging.getLogger(name))
    _module(
        "omni_npu.vllm_patches.core",
        VLLMPatch=_VLLMPatch,
        register_patch=_register_patch,
    )

    # The two sibling patches this module chains onto.  Their bodies stand in for
    # "everything below us in the chain, ending at real vLLM 0.25.1".
    class ExpertIdServingChatStream:
        async def chat_completion_stream_generator(
            self, request, result_generator, *args, **kwargs
        ):
            CALLS["chat_stream"] = {"args": args, "kwargs": kwargs}
            num_cached = None
            async for res in result_generator:
                num_cached = res.num_cached_tokens
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            usage = {"prompt_tokens": 600, "completion_tokens": 2, "total_tokens": 602}
            # 0.25.1: key present only when details are on (exclude_none=True).
            if self.enable_prompt_tokens_details and num_cached is not None:
                usage["prompt_tokens_details"] = {"cached_tokens": num_cached}
            yield f"data: {json.dumps({'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

    class ExpertIdServingCompletionStream:
        async def completion_stream_generator(
            self, request, engine_inputs, result_generator, *args, **kwargs
        ):
            CALLS["compl_stream"] = {"args": args, "kwargs": kwargs}
            num_cached = None
            async for _idx, res in result_generator:
                num_cached = res.num_cached_tokens
            usage = {"prompt_tokens": 600, "completion_tokens": 2, "total_tokens": 602}
            if self.enable_prompt_tokens_details and num_cached is not None:
                usage["prompt_tokens_details"] = {"cached_tokens": num_cached}
            yield f"data: {json.dumps({'usage': usage})}\n\n"

    class OpenAIServingChatPatch:
        async def chat_completion_full_generator(
            self, request, result_generator, *args, **kwargs
        ):
            CALLS["chat_full"] = {"args": args, "kwargs": kwargs}
            num_cached = None
            async for res in result_generator:
                num_cached = res.num_cached_tokens
            details = None
            if self.enable_prompt_tokens_details and num_cached is not None:
                details = _PromptTokenUsageInfo(cached_tokens=num_cached)
            return FakeResponse(
                usage=FakeUsage(prompt_tokens=600, prompt_tokens_details=details),
                kv_transfer_params=self._kv_out,
            )

    _module(
        "omni_npu.vllm_patches.patches.common.patch_routed_experts",
        ExpertIdServingChatStream=ExpertIdServingChatStream,
        ExpertIdServingCompletionStream=ExpertIdServingCompletionStream,
    )
    _module(
        "omni_npu.vllm_patches.patches.common.patch_prefilled_token_skip_tokenize",
        OpenAIServingChatPatch=OpenAIServingChatPatch,
    )


CALLS: dict = {}


class FakeUsage:
    def __init__(self, prompt_tokens=0, prompt_tokens_details=None):
        self.prompt_tokens = prompt_tokens
        self.prompt_tokens_details = prompt_tokens_details


class FakeResponse:
    def __init__(self, usage=None, kv_transfer_params=None):
        self.usage = usage
        self.kv_transfer_params = kv_transfer_params


class FakeRequest:
    def __init__(self, kv_transfer_params=None):
        self.kv_transfer_params = kv_transfer_params


class FakeOutput:
    def __init__(self, num_cached_tokens):
        self.num_cached_tokens = num_cached_tokens


class FakeServing:
    """Stands in for OpenAIServingChat / OpenAIServingCompletion instances."""

    def __init__(self, enable_details=True, kv_out=None):
        self.enable_prompt_tokens_details = enable_details
        self._kv_out = kv_out


def _load_patch_module():
    # This file runs at collection time (module-level ``apc = _load_patch_module()``),
    # so without a restore the stubs would clobber the real vllm/omni_npu modules for
    # every later test file in the session. Snapshot sys.modules before installing the
    # fakes and restore it in a finally. The loaded module keeps direct references to
    # the fake modules/classes, so removing them from sys.modules is safe.
    saved = dict(sys.modules)
    try:
        _install_stubs()
        spec = importlib.util.spec_from_file_location("_patch_serving_apc", PATCH_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, prev in saved.items():
            if sys.modules.get(name) is not prev:
                if prev is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev
        for name in list(sys.modules):
            if name not in saved:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    apc = _load_patch_module()
else:
    apc = None

    @pytest.fixture(scope="module", autouse=True)
    def _isolated_apc_module():
        global apc

        with patch.dict(sys.modules):
            apc = _load_patch_module()
            yield

        apc = None


async def _agen(items):
    for item in items:
        yield item


async def _collect(agen):
    return [chunk async for chunk in agen]


def _usage_of(chunks):
    """Pull the usage dict out of the SSE chunk that carries one."""
    for chunk in chunks:
        if chunk.startswith("data: {") and '"usage"' in chunk:
            return json.loads(chunk[len("data: "):])["usage"]
    raise AssertionError(f"no usage chunk in {chunks}")


# --------------------------------------------------------------------------
# helper-level tests
# --------------------------------------------------------------------------


def test_hit_rate_rounds_and_guards_zero_denominator():
    assert apc._prompt_cache_hit_rate(800, 1000) == 0.8
    assert apc._prompt_cache_hit_rate(1, 3) == 0.3333
    assert apc._prompt_cache_hit_rate(5, 0) == 0.0
    assert apc._prompt_cache_hit_rate(5, -1) == 0.0


def test_resolve_prefers_forwarded_value_including_zero():
    # D side: P forwarded a genuine 0 -- must not fall back to the local count.
    req = FakeRequest({"prefill_cached_tokens": 0})
    assert apc._resolve_num_cached_tokens_for_usage(req, 512) == 0

    req = FakeRequest({"prefill_cached_tokens": 800})
    assert apc._resolve_num_cached_tokens_for_usage(req, 0) == 800

    # Mixed deployment: no kv_transfer_params at all -> local engine count.
    assert apc._resolve_num_cached_tokens_for_usage(FakeRequest(None), 123) == 123
    assert apc._resolve_num_cached_tokens_for_usage(FakeRequest(None), None) == 0


def test_denominator_prefers_prefill_prompt_tokens():
    req = FakeRequest({"prefill_prompt_tokens": 1000})
    assert apc._prompt_tokens_denominator_pd(req, 600) == 1000
    # A non-positive forwarded value is ignored rather than trusted.
    assert apc._prompt_tokens_denominator_pd(FakeRequest({"prefill_prompt_tokens": 0}), 600) == 600
    assert apc._prompt_tokens_denominator_pd(FakeRequest(None), 600) == 600


def test_kv_transfer_params_written_only_when_channel_exists():
    # PD: the dict exists -> fill it in.
    res = FakeResponse(kv_transfer_params={"remote_engine_id": "x"})
    apc._merge_apc_into_kv_transfer_params_if_present(res, 800, 1000, 1000)
    assert res.kv_transfer_params["prefill_cached_tokens"] == 800
    assert res.kv_transfer_params["prefill_prompt_tokens"] == 1000
    assert res.kv_transfer_params["remote_engine_id"] == "x"  # untouched

    # Mixed: None must stay None, not become a dict.
    res = FakeResponse(kv_transfer_params=None)
    apc._merge_apc_into_kv_transfer_params_if_present(res, 800, 1000, 1000)
    assert res.kv_transfer_params is None


# --------------------------------------------------------------------------
# SSE rewrite tests
# --------------------------------------------------------------------------


def test_sse_keeps_details_key_as_null_when_disabled():
    """Upstream dumps with exclude_none=True, so the key would vanish entirely."""
    chunk = 'data: {"usage":{"prompt_tokens":1000}}\n\n'
    out = apc._normalize_usage_chunk(chunk, 800, enable_details=False)
    usage = json.loads(out[len("data: "):])["usage"]
    assert "prompt_tokens_details" in usage
    assert usage["prompt_tokens_details"] is None


def test_sse_adds_cached_rate_to_existing_details():
    chunk = 'data: {"usage":{"prompt_tokens":1000,"prompt_tokens_details":{"cached_tokens":0}}}\n\n'
    out = apc._normalize_usage_chunk(chunk, 800, enable_details=True, request=FakeRequest(None))
    details = json.loads(out[len("data: "):])["usage"]["prompt_tokens_details"]
    assert details["cached_tokens"] == 800
    assert details["cached_rate"] == 0.8


def test_sse_rate_never_exceeds_one_when_d_sees_a_shorter_prompt():
    """P counted 800/1000; D only sees 600 prompt tokens (prefilled-token skip)."""
    chunk = 'data: {"usage":{"prompt_tokens":600,"prompt_tokens_details":{"cached_tokens":0}}}\n\n'
    req = FakeRequest({"prefill_cached_tokens": 800, "prefill_prompt_tokens": 1000})
    out = apc._normalize_usage_chunk(chunk, 800, enable_details=True, request=req)
    details = json.loads(out[len("data: "):])["usage"]["prompt_tokens_details"]
    assert details["cached_rate"] == 0.8  # 800/1000, not 800/600 = 1.33

    # And without the forwarded denominator the min() cap still holds the line.
    out = apc._normalize_usage_chunk(chunk, 800, enable_details=True, request=FakeRequest(None))
    details = json.loads(out[len("data: "):])["usage"]["prompt_tokens_details"]
    assert details["cached_rate"] <= 1.0


def test_sse_leaves_non_usage_chunks_untouched():
    for chunk in ('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', "data: [DONE]\n\n", ": ping\n\n"):
        assert apc._normalize_usage_chunk(chunk, 800, True) == chunk


# --------------------------------------------------------------------------
# end-to-end wrapper tests (the parts that actually break on 0.25.1)
# --------------------------------------------------------------------------


def test_chat_stream_d_side_reports_p_side_hit():
    """The whole point: D's own engine count is 0, the client must still see 800."""
    serving = FakeServing(enable_details=True)
    request = FakeRequest({"prefill_cached_tokens": 800, "prefill_prompt_tokens": 1000})
    gen = apc.OpenAIServingChatStreamAPCPatch.chat_completion_stream_generator(
        serving, request, _agen([FakeOutput(0)]), "req-1", "model",
        chat_template_kwargs={"enable_thinking": False},
        mm_token_counts={"image": 32},
    )
    usage = _usage_of(asyncio.run(_collect(gen)))
    assert usage["prompt_tokens_details"]["cached_tokens"] == 800
    assert usage["prompt_tokens_details"]["cached_rate"] == 0.8


def test_chat_stream_forwards_new_0251_keywords():
    """chat_template_kwargs / mm_token_counts are passed by keyword at the call site.

    Self-contained: it populates CALLS itself (the sibling stub records args/kwargs
    during the generator run), so it no longer depends on
    test_chat_stream_d_side_reports_p_side_hit having executed first in the same
    process -- CI may distribute the two tests across separate workers.
    """
    serving = FakeServing(enable_details=True)
    request = FakeRequest({"prefill_cached_tokens": 800, "prefill_prompt_tokens": 1000})
    gen = apc.OpenAIServingChatStreamAPCPatch.chat_completion_stream_generator(
        serving, request, _agen([FakeOutput(0)]), "req-1", "model",
        chat_template_kwargs={"enable_thinking": False},
        mm_token_counts={"image": 32},
    )
    asyncio.run(_collect(gen))

    assert CALLS["chat_stream"]["kwargs"]["mm_token_counts"] == {"image": 32}
    assert CALLS["chat_stream"]["kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert CALLS["chat_stream"]["args"] == ("req-1", "model")


def test_chat_stream_mixed_deployment_uses_local_count():
    serving = FakeServing(enable_details=True)
    gen = apc.OpenAIServingChatStreamAPCPatch.chat_completion_stream_generator(
        serving, FakeRequest(None), _agen([FakeOutput(300)]), "req-2", "model",
    )
    usage = _usage_of(asyncio.run(_collect(gen)))
    assert usage["prompt_tokens_details"]["cached_tokens"] == 300
    assert usage["prompt_tokens_details"]["cached_rate"] == 0.5  # 300/600


def test_chat_full_p_side_writes_back_into_kv_transfer_params():
    serving = FakeServing(enable_details=True, kv_out={"remote_engine_id": "p0"})
    result = asyncio.run(
        apc.OpenAIServingChatFullAPCPatch.chat_completion_full_generator(
            serving, FakeRequest(None), _agen([FakeOutput(480)]), "req-3", "model",
            parser=None, mm_token_counts=None,
        )
    )
    assert result.kv_transfer_params["prefill_cached_tokens"] == 480
    assert result.kv_transfer_params["prefill_prompt_tokens"] == 600
    assert result.usage.prompt_tokens_details.cached_rate == 0.8
    assert CALLS["chat_full"]["kwargs"]["parser"] is None


def test_chat_full_relays_even_when_details_disabled():
    """With details off upstream writes nothing into usage, so the generator capture
    is the only source -- P must still be able to tell D what it hit."""
    serving = FakeServing(enable_details=False, kv_out={"remote_engine_id": "p0"})
    result = asyncio.run(
        apc.OpenAIServingChatFullAPCPatch.chat_completion_full_generator(
            serving, FakeRequest(None), _agen([FakeOutput(480)]), "req-4", "model",
        )
    )
    assert result.usage.prompt_tokens_details is None  # not fabricated
    assert result.kv_transfer_params["prefill_cached_tokens"] == 480


def test_completion_stream_reads_result_generator_by_name_not_by_index():
    serving = FakeServing(enable_details=True)
    request = FakeRequest({"prefill_cached_tokens": 800, "prefill_prompt_tokens": 1000})
    gen = apc.OpenAIServingCompletionStreamAPCPatch.completion_stream_generator(
        serving, request, ["engine-input"], _agen([(0, FakeOutput(0))]),
        "req-5", 12345, "model", num_prompts=1, tokenizer=None, request_metadata=None,
    )
    usage = _usage_of(asyncio.run(_collect(gen)))
    assert usage["prompt_tokens_details"]["cached_tokens"] == 800
    assert CALLS["compl_stream"]["kwargs"]["num_prompts"] == 1


def test_completion_create_preserves_original_and_skips_streams():
    async def fake_create(self, request, raw_request=None):
        FakeServing.original_called = True
        return FakeResponse(
            usage=FakeUsage(600, _PromptTokenUsageInfo(cached_tokens=480)),
            kv_transfer_params={"remote_engine_id": "p0"},
        )

    apc._orig_compl_create = fake_create
    result = asyncio.run(
        apc.OpenAIServingCompletionAPCPatch.create_completion(
            FakeServing(enable_details=True), FakeRequest(None)
        )
    )
    assert FakeServing.original_called is True
    assert result.usage.prompt_tokens_details.cached_rate == 0.8
    assert result.kv_transfer_params["prefill_cached_tokens"] == 480

    # An async generator (streaming) must pass straight through untouched.
    async def fake_stream_create(self, request, raw_request=None):
        async def _g():
            yield "data: x\n\n"

        return _g()

    apc._orig_compl_create = fake_stream_create
    out = asyncio.run(
        apc.OpenAIServingCompletionAPCPatch.create_completion(
            FakeServing(enable_details=True), FakeRequest(None)
        )
    )
    assert hasattr(out, "__anext__")


def test_chain_falls_back_to_vllm_when_sibling_patch_is_unavailable(monkeypatch):
    """OMNI_NPU_SKIP_PATCH_FILES / an unported sibling must not break this patch.

    _chain_to should hand back the vLLM implementation instead of raising, so the
    APC line still works when the relay chain is one link short.
    """
    sentinel = object()

    missing_module = apc._chain_to(
        "omni_npu.vllm_patches.patches.common.definitely_not_a_module",
        "Whatever", "some_method", sentinel,
    )
    assert missing_module is sentinel

    missing_class = apc._chain_to(
        "omni_npu.vllm_patches.patches.common.patch_routed_experts",
        "NoSuchClass", "chat_completion_stream_generator", sentinel,
    )
    assert missing_class is sentinel

    missing_method = apc._chain_to(
        "omni_npu.vllm_patches.patches.common.patch_routed_experts",
        "ExpertIdServingChatStream", "no_such_method", sentinel,
    )
    assert missing_method is sentinel

    # And the happy path still resolves to the sibling's own method.  The real
    # patch_routed_experts module is still being ported to vLLM 0.25.1 (it imports
    # v0.14-only symbols such as _BUFFER_PREFIX), so install a self-contained fake
    # sibling instead of importing it -- this test only exercises _chain_to's
    # resolution, not the sibling patch's body.  monkeypatch restores sys.modules at
    # teardown, so nothing leaks into the rest of the session.
    sibling = types.ModuleType(
        "omni_npu.vllm_patches.patches.common.patch_routed_experts")

    class ExpertIdServingChatStream:
        async def chat_completion_stream_generator(self, *a, **kw):
            pass

    sibling.ExpertIdServingChatStream = ExpertIdServingChatStream
    monkeypatch.setitem(sys.modules, sibling.__name__, sibling)

    resolved = apc._chain_to(
        "omni_npu.vllm_patches.patches.common.patch_routed_experts",
        "ExpertIdServingChatStream", "chat_completion_stream_generator", sentinel,
    )
    assert resolved is ExpertIdServingChatStream.__dict__[
        "chat_completion_stream_generator"
    ]


def test_patch_names_still_shadow_the_upstream_registrations():
    """The relay chain depends on re-registering the *same* PatchManager names."""
    names = [name for name, _ in REGISTERED]
    for expected in (
        "ExpertIdServingChatStream",
        "PrefilledTokenSkipOpenAIServingChat",
        "ExpertIdServingCompletionStream",
        "OpenAIServingCompletionAPCPatch",
    ):
        assert expected in names, f"{expected} no longer registered"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("=" * 60)
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    if failed:
        raise RuntimeError(f"{failed} tests failed")
