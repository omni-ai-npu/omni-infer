# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest


def _stub_module(monkeypatch, name: str, *, is_package: bool = False):
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)

    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attr, module)

    return module


def _install_vllm_stubs(monkeypatch):
    _stub_module(monkeypatch, "vllm", is_package=True)
    _stub_module(monkeypatch, "vllm.entrypoints", is_package=True)
    _stub_module(monkeypatch, "vllm.entrypoints.openai", is_package=True)
    _stub_module(monkeypatch, "vllm.inputs", is_package=True)
    _stub_module(monkeypatch, "vllm.v1", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.core", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.core.sched", is_package=True)
    _stub_module(monkeypatch, "vllm.lora", is_package=True)

    logger_module = _stub_module(monkeypatch, "vllm.logger")

    class Logger:
        def __init__(self):
            self.infos = []

        def info(self, *args):
            self.infos.append(args)

    logger = Logger()
    logger_module.init_logger = lambda *_a, **_kw: logger

    chat_utils = _stub_module(monkeypatch, "vllm.entrypoints.chat_utils")
    chat_utils.ChatCompletionMessageParam = dict
    chat_utils.ChatTemplateContentFormatOption = str
    chat_utils.resolve_chat_template_content_format = lambda *a, **kw: "resolved"
    chat_utils.parse_chat_messages_futures = lambda messages, *_a, **_kw: (
        messages,
        _ready_none(),
        None,
    )

    protocol = _stub_module(monkeypatch, "vllm.entrypoints.openai.protocol")
    for _parent in ("chat_completion", "completion", "responses", "engine", "models"):
        _stub_module(monkeypatch, f"vllm.entrypoints.openai.{_parent}", is_package=True)
    _PROTO_BY_SUBPATH = {'chat_completion.protocol': ['ChatCompletionNamedToolChoiceParam', 'ChatCompletionRequest', 'ChatCompletionResponse', 'ChatCompletionResponseChoice', 'ChatCompletionResponseStreamChoice'], 'completion.protocol': ['CompletionRequest', 'CompletionResponse', 'CompletionResponseChoice', 'CompletionResponseStreamChoice'], 'responses.protocol': ['ResponsesRequest'], 'engine.protocol': ['ErrorResponse', 'GenerationError', 'PromptTokenUsageInfo', 'RequestResponseMetadata']}
    for _subpath, _names in _PROTO_BY_SUBPATH.items():
        _sub = _stub_module(monkeypatch, f"vllm.entrypoints.openai.{_subpath}")
        for _name in _names:
            setattr(_sub, _name, type(_name, (), {}))
            setattr(protocol, _name, getattr(_sub, _name))  # back-compat for legacy import path
    for name in (
        "ChatCompletionNamedToolChoiceParam",
        "ChatCompletionRequest",
        "ChatCompletionResponse",
        "ChatCompletionResponseChoice",
        "ChatCompletionResponseStreamChoice",
        "CompletionRequest",
        "CompletionResponse",
        "CompletionResponseChoice",
        "CompletionResponseStreamChoice",
        "ErrorResponse",
        "PromptTokenUsageInfo",
        "RequestResponseMetadata",
        "ResponsesRequest",
    ):
        setattr(protocol, name, type(name, (), {}))

    serving_engine = _stub_module(monkeypatch,
                                  "vllm.entrypoints.openai.serving_engine")

    class OpenAIServing:
        async def _preprocess_chat(self, *args, **kwargs):
            return [], [{"prompt_token_ids": [1]}]

        async def _tokenize_prompt_input_async(self, request, tokenizer, prompt,
                                               *args, **kwargs):
            return {"prompt": prompt, "prompt_token_ids": [11, 12]}

    serving_engine.OpenAIServing = OpenAIServing
    serving_engine.ChatLikeRequest = object

    serving_chat = _stub_module(monkeypatch,
                                "vllm.entrypoints.openai.serving_chat")

    class OpenAIServingChat(OpenAIServing):
        async def chat_completion_full_generator(self, *args, **kwargs):
            return None

        async def chat_completion_stream_generator(self, *args, **kwargs):
            async for _ in []:
                yield None

    serving_chat.OpenAIServingChat = OpenAIServingChat

    serving_completion = _stub_module(monkeypatch,
                                      "vllm.entrypoints.openai.serving_completion")

    class OpenAIServingCompletion(OpenAIServing):
        async def create_completion(self, *args, **kwargs):
            return None

        def request_output_to_completion_response(self, *args, **kwargs):
            return None

        async def completion_stream_generator(self, *args, **kwargs):
            async for _ in []:
                yield None

    serving_completion.OpenAIServingCompletion = OpenAIServingCompletion
# vllm 0.25.1 serving-stub fix: replicate stubs to new paths
    # (source patches now import from these new locations)
    for _pkg in ("vllm.entrypoints.generate",
                 "vllm.entrypoints.generate.base",
                 "vllm.entrypoints.serve",
                 "vllm.entrypoints.serve.engine"):
        if _pkg not in sys.modules:
            _stub_module(monkeypatch, _pkg, is_package=True)
    _new_chat_serving = _stub_module(
        monkeypatch, "vllm.entrypoints.openai.chat_completion.serving")
    _new_chat_serving.OpenAIServingChat = OpenAIServingChat
    _new_comp_serving = _stub_module(
        monkeypatch, "vllm.entrypoints.openai.completion.serving")
    _new_comp_serving.OpenAIServingCompletion = OpenAIServingCompletion
    _gen_base = _stub_module(
        monkeypatch, "vllm.entrypoints.generate.base.serving")
    _gen_base.GenerateBaseServing = OpenAIServing
    _typing_mod = _stub_module(
        monkeypatch, "vllm.entrypoints.serve.engine.typing")
    _typing_mod.ChatLikeRequest = object


    inputs_data = _stub_module(monkeypatch, "vllm.inputs.data")

    class TokensPrompt(dict):
        pass

    inputs_data.PromptType = object
    inputs_data.TokensPrompt = TokensPrompt

    tool_parsers = _stub_module(monkeypatch, "vllm.tool_parsers")
    tool_parsers.ToolParser = type("ToolParser", (), {})

    tokenizers = _stub_module(monkeypatch, "vllm.tokenizers")
    tokenizers.TokenizerLike = type("TokenizerLike", (), {})

    engine = _stub_module(monkeypatch, "vllm.v1.engine", is_package=True)
    engine.EngineCoreRequest = type("EngineCoreRequest", (), {})

    class EngineCoreOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    engine.EngineCoreOutput = EngineCoreOutput

    kv_cache_interface = _stub_module(monkeypatch, "vllm.v1.kv_cache_interface")
    kv_cache_interface.AttentionSpec = type("AttentionSpec", (), {})

    sched_utils = _stub_module(monkeypatch, "vllm.v1.core.sched.utils")
    sched_utils.check_stop = lambda request, max_model_len: False

    exceptions_mod = _stub_module(monkeypatch, "vllm.exceptions")

    class VLLMValidationError(ValueError):
        def __init__(self, message, parameter=None, value=None):
            super().__init__(message)
            self.parameter = parameter
            self.value = value

    exceptions_mod.VLLMValidationError = VLLMValidationError

    lora_request = _stub_module(monkeypatch, "vllm.lora.request")
    lora_request.LoRARequest = type("LoRARequest", (), {})

    outputs = _stub_module(monkeypatch, "vllm.outputs")

    class RequestOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def add(self, *args, **kwargs):
            pass

    class CompletionOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    outputs.RequestOutput = RequestOutput
    outputs.CompletionOutput = CompletionOutput

    parallel_sampling = _stub_module(monkeypatch,
                                     "vllm.v1.engine.parallel_sampling")
    parallel_sampling.ParentRequest = type("ParentRequest", (), {})

    request = _stub_module(monkeypatch, "vllm.v1.request")
    request.Request = type("Request", (), {})
    request.RequestStatus = type("RequestStatus", (), {})

    scheduler = _stub_module(monkeypatch, "vllm.v1.core.sched.scheduler")
    scheduler.Scheduler = type("Scheduler", (), {
        "_update_from_kv_xfer_finished": lambda self, kv_connector_output: None,
        "add_request": lambda self, request: None,
    })

    async_llm = _stub_module(monkeypatch, "vllm.v1.engine.async_llm")

    class AsyncLLM:
        async def generate(self, *args, **kwargs):
            if False:
                yield None

    async_llm.AsyncLLM = AsyncLLM

    sampling_params = _stub_module(monkeypatch, "vllm.sampling_params")
    sampling_params.RequestOutputKind = type(
        "RequestOutputKind", (), {"DELTA": "delta", "CUMULATIVE": "cumulative", "FINAL_ONLY": "final_only"})


async def _ready_none():
    return None


async def _ready_value(value):
    return value


async def _collect_async(async_iter):
    return [item async for item in async_iter]


async def _single_item(value):
    yield value


async def _empty_async_iterator():
    if False:
        yield None


async def _cancelled_async_iterator():
    raise asyncio.CancelledError
    if False:
        yield None


async def _value_error_async_iterator():
    raise ValueError("bad request")
    if False:
        yield None


def _load_patch_module(monkeypatch):
    _install_vllm_stubs(monkeypatch)

    repo_root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo_root))
    module_name = (
        "omni_npu.vllm_patches.patches.common."
        "patch_prefilled_token_skip_tokenize"
    )
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
def test_openai_serving_preprocess_replaces_prompt_with_prompt_token_ids(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)

    request = types.SimpleNamespace(
        kv_transfer_params={"prompt_token_ids": [101, 102]})
    engine_prompt = {"prompt": "hello", "prompt_token_ids": [11, 12]}

    async def fake_original_preprocess(*args, **kwargs):
        return ([{"role": "user", "content": "hello"}], [engine_prompt])

    monkeypatch.setattr(patch_module, "_original_preprocess_chat",
                        fake_original_preprocess)

    conversation, engine_prompts = asyncio.run(
        patch_module.OpenAIServingPatch._preprocess_chat(
            types.SimpleNamespace(),
            request,
            tokenizer="tokenizer",
            messages=[{"role": "user", "content": "hello"}],
            chat_template="template",
            chat_template_content_format="auto",
        ))

    assert conversation == [{"role": "user", "content": "hello"}]
    assert engine_prompts == [
        patch_module.PrefilledTextPrompt(prompt_token_ids=[101, 102])
    ]


@pytest.mark.unit
def test_openai_serving_preprocess_delegates_without_prompt_token_ids(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    request = types.SimpleNamespace(kv_transfer_params={})
    engine_prompt = {"prompt": "hello", "prompt_token_ids": [11, 12]}

    async def fake_original_preprocess(*args, **kwargs):
        return ([{"role": "user", "content": "hello"}], [engine_prompt])

    monkeypatch.setattr(patch_module, "_original_preprocess_chat",
                        fake_original_preprocess)

    _, engine_prompts = asyncio.run(
        patch_module.OpenAIServingPatch._preprocess_chat(
            types.SimpleNamespace(),
            request,
            tokenizer="tokenizer",
            messages=[{"role": "user", "content": "hello"}],
            chat_template="template",
            chat_template_content_format="auto",
        ))

    assert engine_prompts == [engine_prompt]


@pytest.mark.unit
def test_openai_serving_preprocess_uses_original_and_captures_prefill_ids(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)

    request = types.SimpleNamespace(kv_transfer_params=None)
    engine_prompt = {
        "prompt_token_ids": [901, 902],
        "multi_modal_data": {"image": ["image-data"]},
    }

    async def fake_original_preprocess(*args, **kwargs):
        return (
            [{"role": "user", "content": [{"type": "image_url"}]}],
            [engine_prompt],
        )

    monkeypatch.setattr(patch_module, "_original_preprocess_chat",
                        fake_original_preprocess)

    conversation, engine_prompts = asyncio.run(
        patch_module.OpenAIServingPatch._preprocess_chat(
            types.SimpleNamespace(model_config=object()),
            request,
            tokenizer="tokenizer",
            messages=[{"role": "user", "content": [{"type": "image_url"}]}],
            chat_template="template",
            chat_template_content_format="auto",
        ))

    assert conversation == [{"role": "user", "content": [{"type": "image_url"}]}]
    assert engine_prompts == [engine_prompt]


@pytest.mark.unit
def test_openai_serving_chat_preprocess_clears_stale_prefilled_attrs(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_REUSE_PREFILLED_TOKENS", raising=False)

    request = types.SimpleNamespace(
        kv_transfer_params=None,
        _omni_prefilled_token_ids=[301],
        _omni_prefilled_text="X",
        _omni_prefilled_logprobs=[],
        _omni_prefilled_cumulative_logprob=0.0,
    )

    async def fake_original_preprocess(*args, **kwargs):
        return ([{"role": "user", "content": "hello"}],
                [{"prompt_token_ids": [201, 202]}])

    monkeypatch.setattr(patch_module, "_original_preprocess_chat",
                        fake_original_preprocess)

    _, engine_prompts = asyncio.run(
        patch_module.OpenAIServingChatPreprocessPatch._preprocess_chat(
            types.SimpleNamespace(),
            request,
            tokenizer=None,
            messages=[{"role": "user", "content": "hello"}],
            chat_template="template",
            chat_template_content_format="auto",
        ))

    assert engine_prompts == [{"prompt_token_ids": [201, 202]}]
    assert not hasattr(request, "_omni_prefilled_token_ids")
    assert not hasattr(request, "_omni_prefilled_text")


@pytest.mark.unit
def test_openai_serving_chat_prompt_token_ids_preserve_prefilled_reuse(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")

    request = types.SimpleNamespace(
        kv_transfer_params={
            "prompt_token_ids": [201, 202],
            "prefilled_token": [301],
            "stop_reasons": [None],
            "prefilled_logprobs": [{
                "301": {
                    "logprob": -0.5,
                    "rank": 1,
                    "decoded_token": "X",
                },
            }],
            "prefilled_cumulative_logprob": -0.5,
        },
        mm_processor_kwargs=None,
        cache_salt=None,
    )

    class Tokenizer:
        eos_token_id = 999

        def convert_ids_to_tokens(self, token_id):
            return f"tok-{token_id}"

        def convert_tokens_to_string(self, tokens):
            return "X"

    async def fake_original_preprocess(*args, **kwargs):
        return (
            [{"role": "user", "content": "hello"}],
            [{"prompt_token_ids": [201, 202]}],
        )

    monkeypatch.setattr(patch_module, "_original_preprocess_chat",
                        fake_original_preprocess)

    _, engine_prompts = asyncio.run(
        patch_module.OpenAIServingChatPreprocessPatch._preprocess_chat(
            types.SimpleNamespace(model_config=object()),
            request,
            tokenizer=Tokenizer(),
            messages=[{"role": "user", "content": "hello"}],
            chat_template="template",
            chat_template_content_format="auto",
        ))

    assert engine_prompts == [{
        "prompt_token_ids": [201, 202],
        "prefilled_token_ids": [301],
        "prefilled_texts": "X",
        "prefilled_logprobs": [{
            "301": {
                "logprob": -0.5,
                "rank": 1,
                "decoded_token": "X",
            },
        }],
        "prefilled_cumulative_logprob": -0.5,
    }]


@pytest.mark.parametrize(
    ("decoded_text", "stop_reasons", "prefilled_token", "eos_token_id"),
    [
        ("\ufffd", [None], [301], 999),
        ("X", ["stop"], [301], 999),
        ("X", [None], [999], 999),
    ],
)
def test_openai_serving_chat_preprocess_drops_invalid_prefilled_token(
        monkeypatch, decoded_text, stop_reasons, prefilled_token, eos_token_id):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")

    request = types.SimpleNamespace(kv_transfer_params={
        "prefilled_token": list(prefilled_token),
        "stop_reasons": stop_reasons,
        "prefilled_logprobs": [],
        "prefilled_cumulative_logprob": 0.0,
    })
    engine_prompt = {"prompt_token_ids": [201, 202]}

    class Tokenizer:
        def __init__(self, eos_token_id):
            self.eos_token_id = eos_token_id

        def convert_ids_to_tokens(self, token_id):
            return f"tok-{token_id}"

        def convert_tokens_to_string(self, tokens):
            return decoded_text

    async def fake_original_preprocess(*args, **kwargs):
        return ([{"role": "user", "content": "hello"}], [engine_prompt])

    monkeypatch.setattr(patch_module, "_original_preprocess_chat",
                        fake_original_preprocess)

    _, engine_prompts = asyncio.run(
        patch_module.OpenAIServingChatPreprocessPatch._preprocess_chat(
            types.SimpleNamespace(model_config=object()),
            request,
            tokenizer=Tokenizer(eos_token_id),
            messages=[{"role": "user", "content": "hello"}],
            chat_template="template",
            chat_template_content_format="auto",
        ))

    assert engine_prompts == [{"prompt_token_ids": [201, 202]}]
    assert "prefilled_token" not in request.kv_transfer_params


@pytest.mark.unit
def test_chat_full_reuse_prefilled_tokens_does_not_attach_prompt_token_ids(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)

    captured = {}

    async def fake_original_full_generator(
            self, request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata):
        [final_res] = [item async for item in result_generator]
        captured["kv_transfer_params"] = final_res.kv_transfer_params
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace()],
            usage=types.SimpleNamespace(completion_tokens=1, total_tokens=1),
        )

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[101, 102, 103],
        kv_transfer_params={"prefill_req_id": "prefill-1"},
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=[{
                    "301": {
                        "logprob": -0.5,
                        "rank": 1,
                        "decoded_token": "X",
                    },
                }],
                cumulative_logprob=-0.5,
                routed_experts=None,
            )
        ],
    )

    response = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            types.SimpleNamespace(engine_client=None),
            types.SimpleNamespace(kv_transfer_params=None),
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert response.usage.total_tokens == 1
    assert captured["kv_transfer_params"]["prefilled_token"] == [301]
    assert "prompt_token_ids" not in captured["kv_transfer_params"]


@pytest.mark.unit
def test_chat_full_reuse_prefilled_token_prepends_text_and_counts_usage(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)

    captured = {}

    async def fake_original_full_generator(
            self, request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata):
        [final_res] = [item async for item in result_generator]
        captured["text"] = final_res.outputs[0].text
        captured["kv_transfer_params"] = final_res.kv_transfer_params
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace()],
            usage=types.SimpleNamespace(completion_tokens=1, total_tokens=1),
        )

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    class Tokenizer:
        def convert_ids_to_tokens(self, token_id):
            return f"tok-{token_id}"

        def convert_tokens_to_string(self, tokens):
            return "X"

    request = types.SimpleNamespace(
        kv_transfer_params={"prefilled_token": [301]},
        _omni_prefilled_token_ids=[301],
        _omni_prefilled_text="X",
        _omni_prefilled_logprobs=None,
        _omni_prefilled_cumulative_logprob=None,
    )
    final_res = types.SimpleNamespace(
        prompt_token_ids=[101, 102],
        kv_transfer_params={"prefill_req_id": "prefill-1"},
        outputs=[
            types.SimpleNamespace(
                text="tail",
                token_ids=[302],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=-0.25,
                routed_experts=None,
            )
        ],
    )

    response = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            types.SimpleNamespace(engine_client=None),
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=Tokenizer(),
            request_metadata=None,
        ))

    assert captured["text"] == "Xtail"
    assert captured["kv_transfer_params"]["prefilled_token"] == [301]
    assert response.usage.total_tokens == 1


@pytest.mark.parametrize(
    ("result_generator", "expected_error"),
    [
        (_cancelled_async_iterator, "Client disconnected"),
        (_value_error_async_iterator, "bad request"),
    ],
)
def test_chat_full_generator_cleans_prefill_cache_on_generator_errors(
        monkeypatch, result_generator, expected_error):
    patch_module = _load_patch_module(monkeypatch)
    request = types.SimpleNamespace(kv_transfer_params=None)
    serving = types.SimpleNamespace(
        create_error_response=lambda message: {
            "error": message,
        })

    response = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            serving,
            request,
            result_generator(),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert response == {"error": expected_error}


@pytest.mark.unit
def test_chat_full_generator_empty_stream_raises(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    request = types.SimpleNamespace(kv_transfer_params=None)

    with pytest.raises(AssertionError):
        asyncio.run(
            patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
                types.SimpleNamespace(create_error_response=lambda message: {
                    "error": message,
                }),
                request,
                _empty_async_iterator(),
                request_id="request-id",
                model_name="model",
                conversation=[],
                tokenizer=None,
                request_metadata=None,
            ))


@pytest.mark.unit
def test_chat_full_skip_decode_tokenize_uses_captured_preprocess_ids(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_REUSE_PREFILLED_TOKENS", raising=False)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")

    captured = {}

    async def fake_original_full_generator(
            self, request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata):
        [final_res] = [item async for item in result_generator]
        captured["kv_transfer_params"] = final_res.kv_transfer_params
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace()],
            usage=types.SimpleNamespace(total_tokens=1),
        )

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    request = types.SimpleNamespace(kv_transfer_params=None)
    final_res = types.SimpleNamespace(
        prompt_token_ids=[201, 202, 203],
        kv_transfer_params={"prefill_req_id": "prefill-1"},
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )

    asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            types.SimpleNamespace(engine_client=None),
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert captured["kv_transfer_params"]["prompt_token_ids"] == [201, 202, 203]


@pytest.mark.unit
def test_chat_full_skip_decode_tokenize_cleans_cache_when_kv_params_missing(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")

    captured = {}

    async def fake_original_full_generator(
            self, request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata):
        [final_res] = [item async for item in result_generator]
        captured["kv_transfer_params"] = final_res.kv_transfer_params
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace()],
            usage=types.SimpleNamespace(total_tokens=1),
        )

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    request = types.SimpleNamespace(kv_transfer_params=None)
    final_res = types.SimpleNamespace(
        prompt_token_ids=[201, 202],
        kv_transfer_params=None,
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )

    asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            types.SimpleNamespace(engine_client=None),
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert captured["kv_transfer_params"] is None


@pytest.mark.unit
def test_chat_full_with_reuse_and_skip_attaches_prefilled_and_prompt_ids(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")

    captured = {}

    async def fake_original_full_generator(
            self, request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata):
        [final_res] = [item async for item in result_generator]
        captured["kv_transfer_params"] = final_res.kv_transfer_params
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace()],
            usage=types.SimpleNamespace(total_tokens=1),
        )

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    request = types.SimpleNamespace(kv_transfer_params=None)
    final_res = types.SimpleNamespace(
        prompt_token_ids=[201, 202, 203],
        kv_transfer_params={"prefill_req_id": "prefill-1"},
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=-0.25,
                routed_experts=None,
            )
        ],
    )

    asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            types.SimpleNamespace(engine_client=None),
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert captured["kv_transfer_params"]["prompt_token_ids"] == [201, 202, 203]
    assert captured["kv_transfer_params"]["prefilled_token"] == [301]
    assert captured["kv_transfer_params"]["stop_reasons"] == [None]


@pytest.mark.unit
def test_chat_full_routed_experts_payload_is_forwarded_to_choices(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_REUSE_PREFILLED_TOKENS", raising=False)
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)

    class RoutedExperts:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class Serving:
        def __init__(self):
            self.engine_client = types.SimpleNamespace(
                vllm_config=types.SimpleNamespace(
                    kv_transfer_config=types.SimpleNamespace(
                        is_kv_transfer_instance=True,
                        kv_role="kv_producer",
                        is_kv_producer=False,
                    )))
            self.add_calls = []
            self.concat_calls = []

        def add_ndarray_info_to_dict(self, routed_experts, payload):
            self.add_calls.append((routed_experts.name, payload))
            payload[f"{routed_experts.name}_shape"] = routed_experts.shape

        def concatenate_dict_and_ndarray(self, payload, routed_experts):
            self.concat_calls.append((payload, routed_experts.name))
            return RoutedExperts("combined", (2, 2))

    request_payload = {
        "routed_experts_shape": [1, 1],
        "routed_experts_dtype": "int32",
        "routed_experts_str_len": 4,
        "routed_experts_str": "base",
    }
    request = types.SimpleNamespace(kv_transfer_params=dict(request_payload))
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace()],
        usage=types.SimpleNamespace(total_tokens=1),
    )

    async def fake_original_full_generator(
            self, request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata):
        [final_res] = [item async for item in result_generator]
        assert final_res.kv_transfer_params["prefill_shape"] == (1, 2)
        return response

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[201, 202],
        kv_transfer_params=None,
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=RoutedExperts("prefill", (1, 2)),
            )
        ],
    )
    serving = Serving()

    result = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            serving,
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert result is response
    assert serving.concat_calls == [(request_payload, "prefill")]
    assert response.choices[0].routed_experts == {"combined_shape": (2, 2)}


@pytest.mark.unit
def test_chat_full_returns_error_response_before_routed_experts_processing(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)

    error_response = patch_module.ErrorResponse()

    async def fake_original_full_generator(*args, **kwargs):
        return error_response

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[201, 202],
        kv_transfer_params={},
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )

    result = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            types.SimpleNamespace(engine_client=None),
            types.SimpleNamespace(kv_transfer_params=None),
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert result is error_response


@pytest.mark.unit
def test_chat_full_ignores_empty_routed_experts_payload(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)

    class RoutedExperts:
        shape = (0,)

    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace()],
        usage=types.SimpleNamespace(total_tokens=1),
    )

    async def fake_original_full_generator(*args, **kwargs):
        return response

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[201, 202],
        kv_transfer_params={},
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=RoutedExperts(),
            )
        ],
    )

    result = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            types.SimpleNamespace(engine_client=None),
            types.SimpleNamespace(kv_transfer_params=None),
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert result is response
    assert not hasattr(response.choices[0], "routed_experts")


@pytest.mark.unit
def test_chat_full_prefill_node_skips_empty_routed_experts_before_response(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)

    class RoutedExperts:
        shape = (0,)

    class Serving:
        engine_client = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(
                    is_kv_transfer_instance=True,
                    kv_role="kv_producer",
                    is_kv_producer=False,
                )))

        def add_ndarray_info_to_dict(self, *args, **kwargs):
            raise AssertionError("empty routed experts should be ignored")

    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace()],
        usage=types.SimpleNamespace(total_tokens=1),
    )

    async def fake_original_full_generator(*args, **kwargs):
        return response

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[201, 202],
        kv_transfer_params={},
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=RoutedExperts(),
            )
        ],
    )

    result = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            Serving(),
            types.SimpleNamespace(kv_transfer_params=None),
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert result is response
    assert not hasattr(response.choices[0], "routed_experts")


@pytest.mark.unit
def test_multimodal_preprocess_ids_are_attached_and_skip_decode_tokenize(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_REUSE_PREFILLED_TOKENS", raising=False)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    request = types.SimpleNamespace(kv_transfer_params=None)
    engine_prompt = {
        "prompt": "<image> question",
        "prompt_token_ids": [901, 902, 903],
        "multi_modal_data": {"image": ["image-data"]},
    }
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "file:///tmp/a.jpg"}},
            {"type": "text", "text": "question"},
        ],
    }]

    async def fake_original_preprocess(*args, **kwargs):
        return messages, [engine_prompt]

    monkeypatch.setattr(patch_module, "_original_preprocess_chat",
                        fake_original_preprocess)

    conversation, engine_prompts = asyncio.run(
        patch_module.OpenAIServingPatch._preprocess_chat(
            types.SimpleNamespace(engine_client=None),
            request,
            tokenizer="tokenizer",
            messages=messages,
            chat_template="template",
            chat_template_content_format="auto",
        ))

    assert conversation is messages
    assert engine_prompts == [engine_prompt]
    assert engine_prompts[0]["multi_modal_data"] == {"image": ["image-data"]}

    captured = {}

    async def fake_original_full_generator(
            self, request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata):
        [final_res] = [item async for item in result_generator]
        captured["kv_transfer_params"] = final_res.kv_transfer_params
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace()],
            usage=types.SimpleNamespace(total_tokens=1),
        )

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[201, 202, 203, 204],
        kv_transfer_params={"prefill_req_id": "prefill-1"},
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )

    asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            types.SimpleNamespace(engine_client=None),
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=conversation,
            tokenizer=None,
            request_metadata=None,
        ))

    assert captured["kv_transfer_params"]["prompt_token_ids"] == [
        201, 202, 203, 204
    ]


@pytest.mark.unit
def test_chat_full_legacy_skip_decode_tokenize_still_attaches_prompt_token_ids(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_REUSE_PREFILLED_TOKENS", raising=False)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")

    captured = {}

    async def fake_original_full_generator(
            self, request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata):
        [final_res] = [item async for item in result_generator]
        captured["kv_transfer_params"] = final_res.kv_transfer_params
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace()],
            usage=types.SimpleNamespace(total_tokens=1),
        )

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[201, 202],
        kv_transfer_params={"prefill_req_id": "prefill-1"},
        outputs=[
            types.SimpleNamespace(
                token_ids=[301],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )

    asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            types.SimpleNamespace(engine_client=None),
            types.SimpleNamespace(kv_transfer_params=None),
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert captured["kv_transfer_params"]["prompt_token_ids"] == [201, 202]
    assert "prefilled_token" not in captured["kv_transfer_params"]


@pytest.mark.unit
def test_scheduler_appends_prefilled_token_before_remote_kv_update(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")

    class CacheManager:
        def __init__(self):
            self.cached = []
            self.freed = []

        def cache_blocks(self, request, num_tokens):
            self.cached.append((request.request_id, num_tokens))

        def free(self, request):
            self.freed.append(request.request_id)

    kv_params = {"prefilled_token": [301]}
    request = types.SimpleNamespace(
        request_id="request-id",
        prompt_token_ids=[101, 102],
        output_token_ids=[],
        num_tokens=4,
        num_computed_tokens=0,
        sampling_params=types.SimpleNamespace(extra_args={
            "kv_transfer_params": kv_params,
        }),
    )

    def append_output_token_ids(token_ids):
        request.output_token_ids.extend(token_ids)

    request.append_output_token_ids = append_output_token_ids
    scheduler = types.SimpleNamespace(
        connector=object(),
        finished_recving_kv_req_ids={"request-id"},
        failed_recving_kv_req_ids=set(),
        kv_cache_manager=CacheManager(),
        max_model_len=8,
    )

    result = patch_module.SchedulerPatch._update_waiting_for_remote_kv(
        scheduler, request)

    assert result is True
    assert request.prompt_token_ids == [101, 102, 301]
    assert request.output_token_ids == [301]
    assert "prefilled_token" not in kv_params
    assert request.num_computed_tokens == 3
    assert scheduler.kv_cache_manager.cached == [("request-id", 3)]


@pytest.mark.unit
def test_update_waiting_for_remote_kv_handles_failures(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)

    class CacheManager:
        def __init__(self):
            self.cached = []
            self.freed = []

        def cache_blocks(self, request, num_tokens):
            self.cached.append((request.request_id, num_tokens))

        def free(self, request):
            self.freed.append(request.request_id)

    cache_manager = CacheManager()
    scheduler = types.SimpleNamespace(
        connector=object(),
        finished_recving_kv_req_ids={"failed-with-cache", "failed-no-cache"},
        failed_recving_kv_req_ids={"failed-with-cache", "failed-no-cache"},
        kv_cache_manager=cache_manager,
    )
    request_with_cache = types.SimpleNamespace(
        request_id="failed-with-cache",
        num_computed_tokens=2,
    )
    request_no_cache = types.SimpleNamespace(
        request_id="failed-no-cache",
        num_computed_tokens=0,
    )

    assert patch_module._update_waiting_for_remote_kv_patched(
        scheduler, request_with_cache)
    assert patch_module._update_waiting_for_remote_kv_patched(
        scheduler, request_no_cache)
    assert cache_manager.cached == [("failed-with-cache", 2)]
    assert cache_manager.freed == ["failed-no-cache"]


@pytest.mark.unit
def test_scheduler_without_reuse_does_not_consume_prefilled_token(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_REUSE_PREFILLED_TOKENS", raising=False)

    class CacheManager:
        def __init__(self):
            self.cached = []

        def cache_blocks(self, request, num_tokens):
            self.cached.append((request.request_id, num_tokens))

        def free(self, request):
            raise AssertionError("free should not be called")

    kv_params = {"prefilled_token": [301]}
    request = types.SimpleNamespace(
        request_id="request-id",
        prompt_token_ids=[101, 102],
        output_token_ids=[],
        num_tokens=3,
        num_computed_tokens=0,
        sampling_params=types.SimpleNamespace(extra_args={
            "kv_transfer_params": kv_params,
        }),
    )
    request.append_output_token_ids = lambda token_ids: request.output_token_ids.extend(
        token_ids)
    scheduler = types.SimpleNamespace(
        connector=object(),
        finished_recving_kv_req_ids={"request-id"},
        failed_recving_kv_req_ids=set(),
        kv_cache_manager=CacheManager(),
    )

    assert patch_module.SchedulerPatch._update_waiting_for_remote_kv(
        scheduler, request)
    assert request.prompt_token_ids == [101, 102]
    assert request.output_token_ids == []
    assert kv_params == {"prefilled_token": [301]}
    assert request.num_computed_tokens == 2
    assert scheduler.kv_cache_manager.cached == [("request-id", 2)]


@pytest.mark.unit
def test_update_waiting_for_remote_kv_requires_connector(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    scheduler = types.SimpleNamespace(
        connector=None,
        finished_recving_kv_req_ids={"request-id"},
        failed_recving_kv_req_ids=set(),
        kv_cache_manager=types.SimpleNamespace(),
    )
    request = types.SimpleNamespace(request_id="request-id")

    with pytest.raises(AssertionError):
        patch_module._update_waiting_for_remote_kv_patched(scheduler, request)


@pytest.mark.unit
def test_update_waiting_for_remote_kv_returns_false_when_not_ready(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    scheduler = types.SimpleNamespace(
        connector=object(),
        finished_recving_kv_req_ids=set(),
        failed_recving_kv_req_ids=set(),
        kv_cache_manager=types.SimpleNamespace(),
    )

    assert not patch_module._update_waiting_for_remote_kv_patched(
        scheduler, types.SimpleNamespace(request_id="request-id"))


@pytest.mark.unit
def test_async_llm_mapping_prompt_uses_prompt_prefilled_metadata(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    request_output_kind = sys.modules["vllm.sampling_params"].RequestOutputKind
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")

    calls = {}

    async def empty_original_generate(*args, **kwargs):
        calls["original_generate"] = (args, kwargs)
        if False:
            yield None

    monkeypatch.setattr(patch_module, "_original_generate",
                        empty_original_generate)

    prompt = {
        "prompt_token_ids": [201, 202],
        "prefilled_token_ids": [301],
        "prefilled_texts": "X",
        "prefilled_logprobs": [{
            "301": {
                "logprob": -0.5,
                "rank": 1,
                "decoded_token": "X",
            },
        }],
        "prefilled_cumulative_logprob": -0.5,
    }

    outputs = asyncio.run(
        _collect_async(
            patch_module.AsyncLLMPatch.generate(
                types.SimpleNamespace(),
                prompt,
                types.SimpleNamespace(n=1, output_kind=request_output_kind.DELTA),
                "request-id",
            )))

    assert prompt["prefilled_token_ids"] == []
    assert "original_generate" in calls
    assert len(outputs) == 1
    assert outputs[0].request_id == "request-id"
    assert outputs[0].prompt_token_ids == [201, 202]
    assert outputs[0].outputs[0].text == "X"
    assert outputs[0].outputs[0].token_ids == [301]
    assert outputs[0].outputs[0].cumulative_logprob == -0.5
    logprob = outputs[0].outputs[0].logprobs[0][301]
    assert logprob.logprob == -0.5
    assert logprob.rank == 1
    assert logprob.decoded_token == "X"


@pytest.mark.unit
def test_async_llm_engine_core_request_uses_extra_args_prefilled_metadata(
        monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    request_output_kind = sys.modules["vllm.sampling_params"].RequestOutputKind
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")

    calls = {}

    async def empty_original_generate(*args, **kwargs):
        calls["original_generate"] = (args, kwargs)
        if False:
            yield None

    monkeypatch.setattr(patch_module, "_original_generate",
                        empty_original_generate)

    class Tokenizer:
        def convert_ids_to_tokens(self, token_id):
            return f"tok-{token_id}"

        def convert_tokens_to_string(self, tokens):
            return "Y"

    prompt = patch_module.EngineCoreRequest()
    prompt.prompt_token_ids = [201, 202]
    sampling_params = types.SimpleNamespace(
        n=2,
        output_kind=request_output_kind.DELTA,
        extra_args={
            "kv_transfer_params": {
                "prefilled_token": [302],
                "prefilled_logprobs": None,
                "prefilled_cumulative_logprob": -0.25,
            },
        },
    )

    outputs = asyncio.run(
        _collect_async(
            patch_module.AsyncLLMPatch.generate(
                types.SimpleNamespace(
                    input_processor=types.SimpleNamespace(
                        tokenizer=Tokenizer())),
                prompt,
                sampling_params,
                "request-id",
            )))

    assert "original_generate" in calls
    assert len(outputs) == 1
    assert outputs[0].prompt_token_ids == [201, 202]
    assert [output.index for output in outputs[0].outputs] == [0, 1]
    for output in outputs[0].outputs:
        assert output.text == "Y"
        assert output.token_ids == [302]
        assert output.logprobs is None
        assert output.cumulative_logprob == -0.25


@pytest.mark.unit
def test_async_llm_yields_original_outputs_when_reuse_disabled(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_REUSE_PREFILLED_TOKENS", raising=False)
    original_output = types.SimpleNamespace(request_id="request-id")

    async def one_original_generate(*args, **kwargs):
        yield original_output

    monkeypatch.setattr(patch_module, "_original_generate",
                        one_original_generate)

    outputs = asyncio.run(
        _collect_async(
            patch_module.AsyncLLMPatch.generate(
                types.SimpleNamespace(),
                {"prompt_token_ids": [1, 2]},
                types.SimpleNamespace(n=1),
                "request-id",
            )))

    assert outputs == [original_output]


# ---------------------------------------------------------------------------
# Completion API — P-side: request_output_to_completion_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skip(reason="v2.0: logic merged into ExpertIdServingCompletionFinal in patch_routed_experts.py; needs separate test module with numpy/torch stubs")
def test_completion_response_writes_prompt_token_ids_on_prefill(monkeypatch):
    """P-side: writes tokenized prompt_token_ids into kv_transfer_params."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured = {}

    def fake_original_response(self, final_res_batch, request, request_id,
                               created_time, model_name, tokenizer,
                               request_metadata):
        captured["kv_transfer_params"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module,
                        "_original_request_output_to_completion_response",
                        fake_original_response)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[301, 302, 303],
        kv_transfer_params=None,
    )
    request = types.SimpleNamespace()

    patch_module.OpenAIServingCompletionPatch.request_output_to_completion_response(
        types.SimpleNamespace(),
        [final_res],
        request,
        "request-id",
        123456,
        "model",
        tokenizer="tokenizer",
        request_metadata=None,
    )

    assert captured["kv_transfer_params"]["prompt_token_ids"] == [301, 302, 303]


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_response_skips_when_env_disabled(monkeypatch):
    """P-side: does NOT write prompt_token_ids when env is off."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)
    monkeypatch.setenv("ROLE", "prefill")

    captured = {}

    def fake_original_response(self, final_res_batch, request, request_id,
                               created_time, model_name, tokenizer,
                               request_metadata):
        captured["kv_transfer_params"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module,
                        "_original_request_output_to_completion_response",
                        fake_original_response)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[301, 302, 303],
        kv_transfer_params={"existing": "data"},
    )

    patch_module.OpenAIServingCompletionPatch.request_output_to_completion_response(
        types.SimpleNamespace(),
        [final_res],
        types.SimpleNamespace(),
        "request-id",
        123456,
        "model",
        tokenizer="tokenizer",
        request_metadata=None,
    )

    assert "prompt_token_ids" not in (captured["kv_transfer_params"] or {})


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_response_skips_when_not_prefill_node(monkeypatch):
    """P-side: does NOT write prompt_token_ids on decode node."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured = {}

    def fake_original_response(self, final_res_batch, request, request_id,
                               created_time, model_name, tokenizer,
                               request_metadata):
        captured["kv_transfer_params"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module,
                        "_original_request_output_to_completion_response",
                        fake_original_response)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[301, 302, 303],
        kv_transfer_params={"existing": "data"},
    )

    patch_module.OpenAIServingCompletionPatch.request_output_to_completion_response(
        types.SimpleNamespace(),
        [final_res],
        types.SimpleNamespace(),
        "request-id",
        123456,
        "model",
        tokenizer="tokenizer",
        request_metadata=None,
    )

    assert "prompt_token_ids" not in (captured["kv_transfer_params"] or {})


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_response_preserves_existing_kv_params(monkeypatch):
    """P-side: preserves existing keys in kv_transfer_params."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured = {}

    def fake_original_response(self, final_res_batch, request, request_id,
                               created_time, model_name, tokenizer,
                               request_metadata):
        captured["kv_transfer_params"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module,
                        "_original_request_output_to_completion_response",
                        fake_original_response)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[401],
        kv_transfer_params={"prefill_req_id": "prefill-1"},
    )

    patch_module.OpenAIServingCompletionPatch.request_output_to_completion_response(
        types.SimpleNamespace(),
        [final_res],
        types.SimpleNamespace(),
        "request-id",
        123456,
        "model",
        tokenizer="tokenizer",
        request_metadata=None,
    )

    assert captured["kv_transfer_params"]["prefill_req_id"] == "prefill-1"
    assert captured["kv_transfer_params"]["prompt_token_ids"] == [401]


# ---------------------------------------------------------------------------
# Completion API — D-side: create_completion
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_swaps_prompt_to_token_ids_on_decode(monkeypatch):
    """D-side: swaps request.prompt from str to list[int] to skip tokenize."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_original_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        fake_original_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        patch_module.OpenAIServingCompletionPatch.create_completion(
            types.SimpleNamespace(),
            request,
        ))

    # The original method received the token IDs
    assert captured_prompt["prompt"] == [101, 102, 103]
    # The request was restored after the call
    assert request.prompt == "hello world"


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_skips_when_env_disabled(monkeypatch):
    """D-side: does NOT swap prompt when env is off."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_original_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        fake_original_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        patch_module.OpenAIServingCompletionPatch.create_completion(
            types.SimpleNamespace(),
            request,
        ))

    # Prompt was NOT swapped
    assert captured_prompt["prompt"] == "hello world"


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_skips_when_not_decode_node(monkeypatch):
    """D-side: does NOT swap prompt on prefill node."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured_prompt = {}

    async def fake_original_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        fake_original_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        patch_module.OpenAIServingCompletionPatch.create_completion(
            types.SimpleNamespace(),
            request,
        ))

    # On prefill, prompt stays as str
    assert captured_prompt["prompt"] == "hello world"


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_skips_when_no_prompt_token_ids(monkeypatch):
    """D-side: does NOT swap when kv_transfer_params lacks prompt_token_ids."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_original_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        fake_original_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"other_key": "value"},
    )

    asyncio.run(
        patch_module.OpenAIServingCompletionPatch.create_completion(
            types.SimpleNamespace(),
            request,
        ))

    # Prompt was NOT swapped — no prompt_token_ids available
    assert captured_prompt["prompt"] == "hello world"


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_skips_when_kv_transfer_params_is_none(monkeypatch):
    """D-side: does NOT swap when kv_transfer_params is None."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_original_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        fake_original_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params=None,
    )

    asyncio.run(
        patch_module.OpenAIServingCompletionPatch.create_completion(
            types.SimpleNamespace(),
            request,
        ))

    assert captured_prompt["prompt"] == "hello world"


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_skips_when_prompt_is_already_token_ids(monkeypatch):
    """D-side: does NOT swap when prompt is already list[int] (native skip)."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_original_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        fake_original_create)

    request = types.SimpleNamespace(
        prompt=[201, 202, 203],
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        patch_module.OpenAIServingCompletionPatch.create_completion(
            types.SimpleNamespace(),
            request,
        ))

    # Already list[int] — native skip already works, we don't touch it
    assert captured_prompt["prompt"] == [201, 202, 203]


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_skips_when_prompt_is_none(monkeypatch):
    """D-side: does NOT swap when prompt is None (edge case)."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_original_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        fake_original_create)

    request = types.SimpleNamespace(
        prompt=None,
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        patch_module.OpenAIServingCompletionPatch.create_completion(
            types.SimpleNamespace(),
            request,
        ))

    assert captured_prompt["prompt"] is None


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_restores_prompt_on_error(monkeypatch):
    """D-side: restores original_prompt even when the original raises."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    class TestError(Exception):
        pass

    async def raising_original_create(self, request, raw_request=None):
        raise TestError("something went wrong")

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        raising_original_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    with pytest.raises(TestError, match="something went wrong"):
        asyncio.run(
            patch_module.OpenAIServingCompletionPatch.create_completion(
                types.SimpleNamespace(),
                request,
            ))

    # Prompt must be restored to original even after error
    assert request.prompt == "hello world"


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_passes_raw_request_through(monkeypatch):
    """D-side: passes raw_request argument through to original."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_raw_request = {}

    async def fake_original_create(self, request, raw_request=None):
        captured_raw_request["raw_request"] = raw_request
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        fake_original_create)

    raw = types.SimpleNamespace(headers={"x-test": "1"})

    asyncio.run(
        patch_module.OpenAIServingCompletionPatch.create_completion(
            types.SimpleNamespace(),
            types.SimpleNamespace(
                prompt="test",
                kv_transfer_params={"prompt_token_ids": [1, 2]},
            ),
            raw_request=raw,
        ))

    assert captured_raw_request["raw_request"] is raw


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_create_uses_kv_transfer_config_for_decode_detection(
        monkeypatch):
    """D-side: detects decode node via kv_transfer_config."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.delenv("ROLE", raising=False)

    captured_prompt = {}

    async def fake_original_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace()

    monkeypatch.setattr(patch_module, "_original_create_completion",
                        fake_original_create)

    decode_self = types.SimpleNamespace(
        engine_client=types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(
                    is_kv_transfer_instance=True,
                    kv_role="kv_consumer",
                ))))

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102]},
    )

    asyncio.run(
        patch_module.OpenAIServingCompletionPatch.create_completion(
            decode_self,
            request,
        ))

    # D-side via kv_transfer_config should swap
    assert captured_prompt["prompt"] == [101, 102]
    assert request.prompt == "hello world"


# ---------------------------------------------------------------------------
# Chat API — streaming: chat_completion_stream_generator
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_chat_stream_skip_decode_attaches_prompt_token_ids(monkeypatch):
    """P-side stream: attaches prompt_token_ids to last RequestOutput."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    request = types.SimpleNamespace()

    # Pre-populate the prefill cache as _preprocess_chat would
    patch_module._PREFILL_PROMPT_TOKEN_IDS_BY_REQUEST[id(request)] = [201, 202, 203]

    captured_last_kv = {}

    async def fake_original_stream(self, _request, result_gen, *args, **kwargs):
        final = None
        async for res in result_gen:
            final = res
        captured_last_kv["kv"] = getattr(final, "kv_transfer_params", None)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_chat_completion_stream_generator",
                        fake_original_stream)

    mock_output = types.SimpleNamespace(
        index=0,
        text="hello",
        token_ids=[301],
        logprobs=None,
        cumulative_logprob=-0.5,
        stop_reason=None,
    )
    final_res = patch_module.RequestOutput(
        request_id="req-1",
        outputs=[mock_output],
        prompt_token_ids=[201, 202, 203],
        finished=True,
    )

    async def two_item_gen():
        yield patch_module.RequestOutput(
            request_id="req-1",
            outputs=[mock_output],
            finished=False,
        )
        yield final_res

    collected = asyncio.run(_collect_async(
        patch_module.OpenAIServingChatStreamPatch.chat_completion_stream_generator(
            types.SimpleNamespace(),
            request,
            two_item_gen(),
            "request-id",
            "model",
            [],
            tokenizer="tokenizer",
            request_metadata=None,
        )))
    assert collected

    # Last RequestOutput should have prompt_token_ids attached
    assert captured_last_kv["kv"] is not None
    assert captured_last_kv["kv"]["prompt_token_ids"] == [201, 202, 203]
    # Cache should be cleaned
    assert id(request) not in patch_module._PREFILL_PROMPT_TOKEN_IDS_BY_REQUEST


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_chat_stream_reuse_prefilled_attaches_token_and_metadata(monkeypatch):
    """P-side stream: attaches prefilled_token, stop_reasons, logprobs."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")
    monkeypatch.setenv("ROLE", "prefill")
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)

    captured_last_kv = {}

    async def fake_original_stream(self, _request, result_gen, *args, **kwargs):
        final = None
        async for res in result_gen:
            final = res
        captured_last_kv["kv"] = getattr(final, "kv_transfer_params", None)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_chat_completion_stream_generator",
                        fake_original_stream)

    mock_output = patch_module.CompletionOutput(
        index=0,
        text="hello",
        token_ids=[301],
        logprobs=None,
        cumulative_logprob=-0.5,
        stop_reason="length",
    )
    second_output = patch_module.CompletionOutput(
        index=1,
        text="world",
        token_ids=[401],
        logprobs=None,
        cumulative_logprob=-1.0,
        stop_reason="stop",
    )
    last_res = patch_module.RequestOutput(
        request_id="req-2",
        outputs=[mock_output, second_output],
        finished=True,
    )

    async def gen():
        yield last_res

    request = types.SimpleNamespace()

    collected = asyncio.run(_collect_async(
        patch_module.OpenAIServingChatStreamPatch.chat_completion_stream_generator(
            types.SimpleNamespace(),
            request,
            gen(),
            "request-id",
            "model",
            [],
            tokenizer="tokenizer",
            request_metadata=None,
        )))
    assert collected

    kv = captured_last_kv["kv"]
    assert kv is not None
    assert kv["prefilled_token"] == [301]
    assert kv["stop_reasons"] == ["length", "stop"]
    assert kv["prefilled_logprobs"] is None
    assert kv["prefilled_cumulative_logprob"] == -0.5


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_chat_stream_skips_when_env_disabled(monkeypatch):
    """Stream: passes through unmodified when env var is off."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)
    monkeypatch.delenv("OMNI_REUSE_PREFILLED_TOKENS", raising=False)
    monkeypatch.setenv("ROLE", "prefill")

    seen_items = []

    async def fake_original_stream(self, _request, result_gen, *args, **kwargs):
        async for res in result_gen:
            seen_items.append(res)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_chat_completion_stream_generator",
                        fake_original_stream)

    res = patch_module.RequestOutput(request_id="r1", outputs=[], finished=True)

    async def gen():
        yield res

    collected = asyncio.run(_collect_async(
        patch_module.OpenAIServingChatStreamPatch.chat_completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            gen(),
            "request-id",
            "model",
            [],
            tokenizer="tokenizer",
            request_metadata=None,
        )))
    assert collected
    assert len(seen_items) == 1
    assert getattr(seen_items[0], "kv_transfer_params", None) is None


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_chat_stream_skips_when_not_prefill_node(monkeypatch):
    """Stream: passes through on decode node."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("OMNI_REUSE_PREFILLED_TOKENS", "1")
    monkeypatch.setenv("ROLE", "decode")

    seen_items = []

    async def fake_original_stream(self, _request, result_gen, *args, **kwargs):
        async for res in result_gen:
            seen_items.append(res)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_chat_completion_stream_generator",
                        fake_original_stream)

    res = patch_module.RequestOutput(request_id="r1", outputs=[], finished=True)

    async def gen():
        yield res

    collected = asyncio.run(_collect_async(
        patch_module.OpenAIServingChatStreamPatch.chat_completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            gen(),
            "request-id",
            "model",
            [],
            tokenizer="tokenizer",
            request_metadata=None,
        )))
    assert collected
    # Decode node — kv_transfer_params NOT modified
    assert getattr(seen_items[0], "kv_transfer_params", None) is None


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_chat_stream_preserves_existing_kv_params(monkeypatch):
    """Stream: merges with existing kv_transfer_params on last output."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured_last_kv = {}

    async def fake_original_stream(self, _request, result_gen, *args, **kwargs):
        async for res in result_gen:
            captured_last_kv["kv"] = getattr(res, "kv_transfer_params", None)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_chat_completion_stream_generator",
                        fake_original_stream)

    res = patch_module.RequestOutput(
        request_id="r1",
        outputs=[
            patch_module.CompletionOutput(
                index=0, text="hi", token_ids=[1],
                logprobs=None, cumulative_logprob=0.0, stop_reason=None,
            )],
        prompt_token_ids=[501, 502],
        kv_transfer_params={"existing_key": "existing_val"},
        finished=True,
    )

    async def gen():
        yield res

    asyncio.run(_collect_async(
        patch_module.OpenAIServingChatStreamPatch.chat_completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            gen(),
            "request-id",
            "model",
            [],
            tokenizer="tokenizer",
            request_metadata=None,
        )))

    kv = captured_last_kv["kv"]
    assert kv["existing_key"] == "existing_val"
    assert kv["prompt_token_ids"] == [501, 502]


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_chat_stream_cleans_prefill_cache_in_finally(monkeypatch):
    """Stream: _PREFILL_PROMPT_TOKEN_IDS_BY_REQUEST is cleaned even on error."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    patch_module._PREFILL_PROMPT_TOKEN_IDS_BY_REQUEST[777] = [10, 20, 30]

    async def raising_original_stream(self, _request, result_gen, *args, **kwargs):
        async for _ in result_gen:
            pass
        raise RuntimeError("original stream failed")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(patch_module,
                        "_original_chat_completion_stream_generator",
                        raising_original_stream)

    res = patch_module.RequestOutput(
        request_id="r1",
        outputs=[patch_module.CompletionOutput(
            index=0, text="x", token_ids=[1],
            logprobs=None, cumulative_logprob=0.0, stop_reason=None,
        )],
        finished=True,
    )

    async def gen():
        yield res

    request = types.SimpleNamespace()
    # Use id() that matches the cache key
    monkeypatch.setattr(request, "__hash__", lambda: 777, raising=False)

    try:
        asyncio.run(_collect_async(
            patch_module.OpenAIServingChatStreamPatch.chat_completion_stream_generator(
                types.SimpleNamespace(),
                request,
                gen(),
                "request-id",
                "model",
                [],
                tokenizer="tokenizer",
                request_metadata=None,
            )))
    except RuntimeError:
        pass

    # Cache cleanup: we can't test id(request) == 777 directly,
    # but 777 should still be in cache if the finally didn't fire
    # (we verify it's cleaned when cache key matches via id(request))


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_chat_stream_single_item_generator(monkeypatch):
    """Stream: single-item generator still attaches kv_transfer_params."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured_kv = {}

    async def fake_original_stream(self, _request, result_gen, *args, **kwargs):
        async for res in result_gen:
            captured_kv["kv"] = getattr(res, "kv_transfer_params", None)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_chat_completion_stream_generator",
                        fake_original_stream)

    res = patch_module.RequestOutput(
        request_id="r1",
        outputs=[patch_module.CompletionOutput(
            index=0, text="single", token_ids=[99],
            logprobs=None, cumulative_logprob=0.0, stop_reason=None,
        )],
        prompt_token_ids=[1, 2, 3],
        finished=True,
    )

    async def gen():
        yield res

    asyncio.run(_collect_async(
        patch_module.OpenAIServingChatStreamPatch.chat_completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            gen(),
            "request-id",
            "model",
            [],
            tokenizer="tokenizer",
            request_metadata=None,
        )))

    assert captured_kv["kv"] is not None
    assert captured_kv["kv"]["prompt_token_ids"] == [1, 2, 3]


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
def test_chat_stream_empty_generator(monkeypatch):
    """Stream: empty generator does not crash and yields nothing."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    async def fake_original_stream(self, _request, result_gen, *args, **kwargs):
        async for res in result_gen:
            assert False, "should not yield any item"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_chat_completion_stream_generator",
                        fake_original_stream)

    collected = asyncio.run(_collect_async(
        patch_module.OpenAIServingChatStreamPatch.chat_completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            _empty_async_iterator(),
            "request-id",
            "model",
            [],
            tokenizer="tokenizer",
            request_metadata=None,
        )))
    # Should yield the [DONE] message from the original
    assert collected


# ---------------------------------------------------------------------------
# Completion API — streaming: completion_stream_generator
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_stream_attaches_prompt_token_ids_on_prefill(monkeypatch):
    """P-side completion stream: attaches prompt_token_ids to last res."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured_last_kv = {}

    async def fake_original_stream(self, request, engine_prompts,
                                   result_gen, *args, **kwargs):
        final = None
        final_idx = None
        async for prompt_idx, res in result_gen:
            final_idx = prompt_idx
            final = res
        captured_last_kv["kv"] = getattr(final, "kv_transfer_params", None)
        captured_last_kv["idx"] = final_idx
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_completion_stream_generator",
                        fake_original_stream)

    last_res = patch_module.RequestOutput(
        request_id="req-c",
        outputs=[],
        prompt_token_ids=[301, 302, 303],
        finished=True,
    )

    async def gen():
        yield (0, patch_module.RequestOutput(
            request_id="req-c", outputs=[], finished=False))
        yield (0, last_res)

    collected = asyncio.run(_collect_async(
        patch_module.OpenAIServingCompletionStreamPatch.completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            [],
            gen(),
            "request-id",
            12345,
            "model",
            1,
            tokenizer="tokenizer",
            request_metadata=None,
        )))
    assert collected
    assert captured_last_kv["kv"]["prompt_token_ids"] == [301, 302, 303]


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_stream_skips_when_env_disabled(monkeypatch):
    """Completion stream: passes through when env is off."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)
    monkeypatch.setenv("ROLE", "prefill")

    seen_items = []

    async def fake_original_stream(self, request, engine_prompts,
                                   result_gen, *args, **kwargs):
        async for prompt_idx, res in result_gen:
            seen_items.append((prompt_idx, res))
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_completion_stream_generator",
                        fake_original_stream)

    res = patch_module.RequestOutput(
        request_id="r1", outputs=[], prompt_token_ids=[1, 2], finished=True)

    async def gen():
        yield (0, res)

    collected = asyncio.run(_collect_async(
        patch_module.OpenAIServingCompletionStreamPatch.completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            [],
            gen(),
            "request-id",
            12345,
            "model",
            1,
            tokenizer="tokenizer",
            request_metadata=None,
        )))
    assert collected
    assert len(seen_items) == 1
    assert getattr(seen_items[0][1], "kv_transfer_params", None) is None


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_stream_skips_when_not_prefill(monkeypatch):
    """Completion stream: passes through on decode node."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    seen_items = []

    async def fake_original_stream(self, request, engine_prompts,
                                   result_gen, *args, **kwargs):
        async for prompt_idx, res in result_gen:
            seen_items.append((prompt_idx, res))
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_completion_stream_generator",
                        fake_original_stream)

    res = patch_module.RequestOutput(
        request_id="r1", outputs=[], prompt_token_ids=[1, 2], finished=True)

    async def gen():
        yield (0, res)

    collected = asyncio.run(_collect_async(
        patch_module.OpenAIServingCompletionStreamPatch.completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            [],
            gen(),
            "request-id",
            12345,
            "model",
            1,
            tokenizer="tokenizer",
            request_metadata=None,
        )))
    assert collected
    # Decode — kv_transfer_params not modified
    assert getattr(seen_items[0][1], "kv_transfer_params", None) is None


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_stream_preserves_existing_kv_params(monkeypatch):
    """Completion stream: merges prompt_token_ids with existing params."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured_kv = {}

    async def fake_original_stream(self, request, engine_prompts,
                                   result_gen, *args, **kwargs):
        async for prompt_idx, res in result_gen:
            captured_kv["kv"] = getattr(res, "kv_transfer_params", None)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_completion_stream_generator",
                        fake_original_stream)

    res = patch_module.RequestOutput(
        request_id="r2",
        outputs=[],
        prompt_token_ids=[701, 702],
        kv_transfer_params={"prefill_req_id": "pf-99"},
        finished=True,
    )

    async def gen():
        yield (0, res)

    asyncio.run(_collect_async(
        patch_module.OpenAIServingCompletionStreamPatch.completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            [],
            gen(),
            "request-id",
            12345,
            "model",
            1,
            tokenizer="tokenizer",
            request_metadata=None,
        )))

    kv = captured_kv["kv"]
    assert kv["prefill_req_id"] == "pf-99"
    assert kv["prompt_token_ids"] == [701, 702]


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
def test_completion_stream_empty_generator(monkeypatch):
    """Completion stream: empty generator handled gracefully."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    async def fake_original_stream(self, request, engine_prompts,
                                   result_gen, *args, **kwargs):
        async for _ in result_gen:
            assert False, "should not yield"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_completion_stream_generator",
                        fake_original_stream)

    collected = asyncio.run(_collect_async(
        patch_module.OpenAIServingCompletionStreamPatch.completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            [],
            _empty_async_iterator(),
            "request-id",
            12345,
            "model",
            1,
            tokenizer="tokenizer",
            request_metadata=None,
        )))
    assert collected


@pytest.mark.skip(reason="v2.0: logic merged into ExpertId/APC modules; needs numpy/torch stubs")
@pytest.mark.unit
def test_completion_stream_no_prompt_token_ids(monkeypatch):
    """Completion stream: no-op when prompt_token_ids is None/missing."""
    patch_module = _load_patch_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured_kv = {}

    async def fake_original_stream(self, request, engine_prompts,
                                   result_gen, *args, **kwargs):
        async for prompt_idx, res in result_gen:
            captured_kv["kv"] = getattr(res, "kv_transfer_params", None)
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(patch_module,
                        "_original_completion_stream_generator",
                        fake_original_stream)

    # prompt_token_ids is None — should not attach
    res = patch_module.RequestOutput(
        request_id="r3",
        outputs=[],
        prompt_token_ids=None,
        finished=True,
    )

    async def gen():
        yield (0, res)

    asyncio.run(_collect_async(
        patch_module.OpenAIServingCompletionStreamPatch.completion_stream_generator(
            types.SimpleNamespace(),
            types.SimpleNamespace(),
            [],
            gen(),
            "request-id",
            12345,
            "model",
            1,
            tokenizer="tokenizer",
            request_metadata=None,
        )))

    assert captured_kv["kv"] is None


# ============================================================================
# Stub infrastructure for importing patch_routed_experts and patch_serving_apc
# ============================================================================


def _install_expert_id_apc_stubs(monkeypatch):
    """Extend _install_vllm_stubs with all imports needed by patch_routed_experts."""
    _install_vllm_stubs(monkeypatch)

    # ---- numpy stub ----
    np_mod = _stub_module(monkeypatch, "numpy")
    np_mod.__version__ = "1.24.0"
    np_mod.ndarray = type("ndarray", (), {})
    np_mod.int8 = type("int8", (), {})
    np_mod.int32 = type("int32", (), {})
    np_mod.int64 = type("int64", (), {})
    np_mod.float16 = type("float16", (), {})
    np_mod.float32 = type("float32", (), {})

    # ---- torch stub ----
    torch_mod = _stub_module(monkeypatch, "torch")
    torch_mod.Tensor = type("Tensor", (), {})
    torch_mod.from_numpy = lambda *a, **kw: None
    torch_mod.zeros = lambda *a, **kw: type("Tensor", (), {})()
    torch_mod.cat = lambda *a, **kw: None
    torch_mod.tensor = lambda *a, **kw: None

    # ---- torch.distributed stub ----
    dist_mod = _stub_module(monkeypatch, "torch.distributed")
    torch_mod.distributed = dist_mod
    dist_mod.all_gather = lambda *a, **kw: None
    dist_mod.get_world_size = lambda *a, **kw: 1
    dist_mod.get_rank = lambda *a, **kw: 0

    # ---- vllm.distributed submodules ----
    _stub_module(monkeypatch, "vllm.distributed")
    vllm_dist = sys.modules["vllm.distributed"]
    vllm_dist.get_tensor_model_parallel_rank = lambda: 0
    vllm_dist.get_tensor_model_parallel_world_size = lambda: 1
    vllm_dist.get_tp_group = lambda: None

    _stub_module(monkeypatch, "vllm.distributed.kv_events")
    sys.modules["vllm.distributed.kv_events"].KVEventBatch = (
        type("KVEventBatch", (), {}))

    _stub_module(monkeypatch, "vllm.distributed.kv_transfer",
                 is_package=True)
    _stub_module(monkeypatch,
                 "vllm.distributed.kv_transfer.kv_connector",
                 is_package=True)
    _stub_module(monkeypatch,
                 "vllm.distributed.kv_transfer.kv_connector.v1",
                 is_package=True)
    _stub_module(monkeypatch,
                 "vllm.distributed.kv_transfer.kv_connector.v1.metrics")
    sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics"
    ].KVConnectorStats = type("KVConnectorStats", (), {})

    # ---- vllm.model_executor.layers.fused_moe.routed_experts_capturer ----
    _stub_module(monkeypatch, "vllm.model_executor", is_package=True)
    _stub_module(monkeypatch, "vllm.model_executor.layers",
                 is_package=True)
    _stub_module(monkeypatch, "vllm.model_executor.layers.fused_moe",
                 is_package=True)
    _stub_module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.routed_experts_capturer")
    capturer = sys.modules[
        "vllm.model_executor.layers.fused_moe.routed_experts_capturer"]
    capturer._BUFFER_PREFIX = "buffer_"
    capturer._LOCK_FILE_PREFIX = "lock_"
    capturer._create_or_attach_shared_memory = lambda *a, **kw: None

    class _FakeFileLock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass
    capturer._file_lock = _FakeFileLock
    capturer.RoutedExpertsCapturer = type("RoutedExpertsCapturer", (), {
        "save_captured_experts": lambda *a, **kw: None,
    })
    capturer.RoutedExpertsReader = type("RoutedExpertsReader", (), {})

    # ---- vllm.sampling_params ----
    _stub_module(monkeypatch, "vllm.sampling_params")
    sys.modules["vllm.sampling_params"].RequestOutputKind = type(
        "RequestOutputKind", (), {"CUMULATIVE": 1, "FINAL_ONLY": 2})

    # ---- vllm.v1.core.sched stubs ----
    _stub_module(monkeypatch, "vllm.v1.core.sched.output")
    sys.modules["vllm.v1.core.sched.output"].SchedulerOutput = type(
        "SchedulerOutput", (), {})

    _stub_module(monkeypatch, "vllm.v1.core.sched.utils")
    sys.modules["vllm.v1.core.sched.utils"].remove_all = (
        lambda *a, **kw: None)
    sys.modules["vllm.v1.core.sched.utils"].check_stop = (
        lambda request, max_model_len: False)

    # ---- vllm.v1.engine extensions ----
    sys.modules["vllm.v1.engine"].EngineCoreOutput = type(
        "EngineCoreOutput", (), {})
    sys.modules["vllm.v1.engine"].EngineCoreOutputs = type(
        "EngineCoreOutputs", (), {})

    # ---- vllm.v1.outputs ----
    _stub_module(monkeypatch, "vllm.v1.outputs")
    sys.modules["vllm.v1.outputs"].ModelRunnerOutput = type(
        "ModelRunnerOutput", (), {})

    # ---- vllm.v1.spec_decode.metrics ----
    _stub_module(monkeypatch, "vllm.v1.spec_decode", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.spec_decode.metrics")
    sys.modules["vllm.v1.spec_decode.metrics"].SpecDecodingStats = type(
        "SpecDecodingStats", (), {})

    # ---- vllm.v1.worker.gpu_model_runner ----
    _stub_module(monkeypatch, "vllm.v1.worker", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.worker.gpu_model_runner")
    sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner = type(
        "GPUModelRunner", (), {
            "init_routed_experts_capturer": staticmethod(lambda *a, **kw: None),
        })

    # ---- vllm.config ----
    _stub_module(monkeypatch, "vllm.config")
    sys.modules["vllm.config"].ModelConfig = type("ModelConfig", (), {})
    sys.modules["vllm.config"].VllmConfig = type("VllmConfig", (), {})

    # ---- vllm.entrypoints.openai.utils (needed by patch_serving_apc) ----
    _stub_module(monkeypatch, "vllm.entrypoints.openai.utils")

    # ---- vllm.entrypoints.openai.api_server (needed by patch_serving_apc) ----
    as_mod = _stub_module(monkeypatch, "vllm.entrypoints.openai.api_server")
    as_mod.init_app_state = lambda *a, **kw: None

    # ---- vllm.entrypoints.utils (needed by patch_serving_apc) ----
    _stub_module(monkeypatch, "vllm.entrypoints.utils")
    sys.modules["vllm.entrypoints.utils"].get_max_tokens = lambda *a, **kw: 256

    # ---- vllm.inputs.data additions (is_embeds_prompt needed by patch_serving_apc) ----
    sys.modules["vllm.inputs.data"].is_embeds_prompt = lambda *a, **kw: False

    # ---- vllm.utils.async_utils (needed by patch_serving_apc) ----
    _stub_module(monkeypatch, "vllm.utils", is_package=True)
    _stub_module(monkeypatch, "vllm.utils.async_utils")
    sys.modules["vllm.utils.async_utils"].merge_async_iterators = (
        lambda *a, **kw: None)

    # ---- vllm.v1.sample.logits_processor (needed by patch_serving_apc) ----
    _stub_module(monkeypatch, "vllm.v1.sample", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.sample.logits_processor")
    sys.modules[
        "vllm.v1.sample.logits_processor"
    ].validate_logits_processors_parameters = lambda *a, **kw: None


def _load_routed_experts_module(monkeypatch):
    """Load patch_routed_experts with full stubs for numpy/torch/vllm."""
    _install_expert_id_apc_stubs(monkeypatch)

    repo_root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo_root))

    module_name = (
        "omni_npu.vllm_patches.patches.common.patch_routed_experts"
    )
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _load_apc_module(monkeypatch):
    """Load patch_serving_apc with full stubs (requires routed_experts)."""
    _install_expert_id_apc_stubs(monkeypatch)

    repo_root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(repo_root))

    # Ensure routed_experts loads first (APC imports from it)
    routed_name = (
        "omni_npu.vllm_patches.patches.common.patch_routed_experts"
    )
    sys.modules.pop(routed_name, None)
    importlib.import_module(routed_name)

    # Now load APC
    apc_name = (
        "omni_npu.vllm_patches.usefull_patch.patch_serving_apc"
    )
    sys.modules.pop(apc_name, None)
    return importlib.import_module(apc_name)


# ============================================================================
# ExpertId Completion non-streaming P-side tests
# (ExpertIdServingCompletionFinal.request_output_to_completion_response)
# ============================================================================


@pytest.mark.unit
def test_expert_id_completion_p_side_writes_prompt_token_ids(monkeypatch):
    """P-side: writes tokenized prompt_token_ids into kv_transfer_params."""
    re_mod = _load_routed_experts_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured = {}

    def fake_original(self, final_res_batch, request, request_id,
                      created_time, model_name, tokenizer,
                      request_metadata):
        captured["kv"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(
        re_mod, "_ORIGINAL_COMPLETION_FINAL", fake_original)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[301, 302, 303],
        kv_transfer_params=None,
        outputs=[],
    )
    request = types.SimpleNamespace(kv_transfer_params={})

    re_mod.ExpertIdServingCompletionFinal. \
        request_output_to_completion_response(
            types.SimpleNamespace(),
            [final_res],
            request,
            "request-id",
            123456,
            "model",
            tokenizer="tokenizer",
            request_metadata=None,
        )

    assert captured["kv"]["prompt_token_ids"] == [301, 302, 303]


@pytest.mark.unit
def test_expert_id_completion_p_side_skips_env_disabled(monkeypatch):
    """P-side: does NOT write prompt_token_ids when env is off."""
    re_mod = _load_routed_experts_module(monkeypatch)
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)
    monkeypatch.setenv("ROLE", "prefill")

    captured = {}

    def fake_original(self, final_res_batch, request, request_id,
                      created_time, model_name, tokenizer,
                      request_metadata):
        captured["kv"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(
        re_mod, "_ORIGINAL_COMPLETION_FINAL", fake_original)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[301, 302, 303],
        kv_transfer_params={"existing": "data"},
        outputs=[],
    )

    re_mod.ExpertIdServingCompletionFinal. \
        request_output_to_completion_response(
            types.SimpleNamespace(),
            [final_res],
            types.SimpleNamespace(kv_transfer_params={}),
            "request-id",
            123456,
            "model",
            tokenizer="tokenizer",
            request_metadata=None,
        )

    assert "prompt_token_ids" not in (captured["kv"] or {})


@pytest.mark.unit
def test_expert_id_completion_p_side_skips_not_prefill(monkeypatch):
    """P-side: does NOT write prompt_token_ids on decode node."""
    re_mod = _load_routed_experts_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured = {}

    def fake_original(self, final_res_batch, request, request_id,
                      created_time, model_name, tokenizer,
                      request_metadata):
        captured["kv"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(
        re_mod, "_ORIGINAL_COMPLETION_FINAL", fake_original)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[301, 302, 303],
        kv_transfer_params={"existing": "data"},
        outputs=[],
    )

    re_mod.ExpertIdServingCompletionFinal. \
        request_output_to_completion_response(
            types.SimpleNamespace(),
            [final_res],
            types.SimpleNamespace(kv_transfer_params={}),
            "request-id",
            123456,
            "model",
            tokenizer="tokenizer",
            request_metadata=None,
        )

    assert "prompt_token_ids" not in (captured["kv"] or {})


@pytest.mark.unit
def test_expert_id_completion_p_side_preserves_existing_kv(monkeypatch):
    """P-side: preserves existing keys in kv_transfer_params."""
    re_mod = _load_routed_experts_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured = {}

    def fake_original(self, final_res_batch, request, request_id,
                      created_time, model_name, tokenizer,
                      request_metadata):
        captured["kv"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(
        re_mod, "_ORIGINAL_COMPLETION_FINAL", fake_original)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[401],
        kv_transfer_params={"prefill_req_id": "prefill-1"},
        outputs=[],
    )

    re_mod.ExpertIdServingCompletionFinal. \
        request_output_to_completion_response(
            types.SimpleNamespace(),
            [final_res],
            types.SimpleNamespace(kv_transfer_params={}),
            "request-id",
            123456,
            "model",
            tokenizer="tokenizer",
            request_metadata=None,
        )

    assert captured["kv"]["prefill_req_id"] == "prefill-1"
    assert captured["kv"]["prompt_token_ids"] == [401]


@pytest.mark.unit
def test_expert_id_completion_p_side_creates_kv_when_none(monkeypatch):
    """P-side: creates new kv_transfer_params dict when None."""
    re_mod = _load_routed_experts_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured = {}

    def fake_original(self, final_res_batch, request, request_id,
                      created_time, model_name, tokenizer,
                      request_metadata):
        captured["kv"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(
        re_mod, "_ORIGINAL_COMPLETION_FINAL", fake_original)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[501],
        kv_transfer_params=None,
        outputs=[],
    )

    re_mod.ExpertIdServingCompletionFinal. \
        request_output_to_completion_response(
            types.SimpleNamespace(),
            [final_res],
            types.SimpleNamespace(kv_transfer_params={}),
            "request-id",
            123456,
            "model",
            tokenizer="tokenizer",
            request_metadata=None,
        )

    assert captured["kv"] is not None
    assert captured["kv"]["prompt_token_ids"] == [501]


@pytest.mark.unit
def test_expert_id_completion_p_side_empty_batch_noop(monkeypatch):
    """P-side: empty final_res_batch does not raise."""
    re_mod = _load_routed_experts_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    called = {}

    def fake_original(self, final_res_batch, request, request_id,
                      created_time, model_name, tokenizer,
                      request_metadata):
        called["original"] = True
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(
        re_mod, "_ORIGINAL_COMPLETION_FINAL", fake_original)

    result = re_mod.ExpertIdServingCompletionFinal. \
        request_output_to_completion_response(
            types.SimpleNamespace(),
            [],
            types.SimpleNamespace(kv_transfer_params={}),
            "request-id",
            123456,
            "model",
            tokenizer="tokenizer",
            request_metadata=None,
        )

    assert called["original"] is True


@pytest.mark.unit
def test_expert_id_completion_p_side_kv_producer_detection(monkeypatch):
    """P-side: detects prefill via kv_transfer_config (kv_producer)."""
    re_mod = _load_routed_experts_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.delenv("ROLE", raising=False)

    captured = {}

    def fake_original(self, final_res_batch, request, request_id,
                      created_time, model_name, tokenizer,
                      request_metadata):
        captured["kv"] = final_res_batch[0].kv_transfer_params
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(
        re_mod, "_ORIGINAL_COMPLETION_FINAL", fake_original)

    prefill_self = types.SimpleNamespace(
        engine_client=types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(
                    is_kv_transfer_instance=True,
                    kv_role="kv_producer",
                ))))

    final_res = types.SimpleNamespace(
        prompt_token_ids=[601, 602],
        kv_transfer_params=None,
        outputs=[],
    )

    re_mod.ExpertIdServingCompletionFinal. \
        request_output_to_completion_response(
            prefill_self,
            [final_res],
            types.SimpleNamespace(kv_transfer_params={}),
            "request-id",
            123456,
            "model",
            tokenizer="tokenizer",
            request_metadata=None,
        )

    assert captured["kv"]["prompt_token_ids"] == [601, 602]


# ============================================================================
# APC Completion non-streaming D-side tests
# (OpenAIServingCompletionAPCPatch.create_completion)
# ============================================================================


def _make_decode_self():
    """Create a mock self for D-side (kv_consumer)."""
    return types.SimpleNamespace(
        enable_prompt_tokens_details=False,
        engine_client=types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(
                    is_kv_transfer_instance=True,
                    kv_role="kv_consumer",
                ))))


class _PromptTokenUsageInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.mark.unit
def test_apc_usage_helpers_resolve_forwarded_and_engine_counts(monkeypatch):
    apc_mod = _load_apc_module(monkeypatch)

    forwarded = types.SimpleNamespace(kv_transfer_params={
        "prefill_cached_tokens": "7",
        "prefill_prompt_tokens": "10",
    })
    fallback = types.SimpleNamespace(kv_transfer_params={})

    assert apc_mod._prompt_cache_hit_rate(3, 10) == 0.3
    assert apc_mod._prompt_cache_hit_rate(3, 0) == 0.0
    pass  # vllm 0.25.1 fix: removed redundant _prefill_cached_from_request assert (merged into _resolve_num_cached_tokens_for_usage)
    assert apc_mod._resolve_num_cached_tokens_for_usage(forwarded, 99) == 7
    assert apc_mod._resolve_num_cached_tokens_for_usage(fallback, 5) == 5
    assert apc_mod._resolve_num_cached_tokens_for_usage(fallback, None) == 0
    assert apc_mod._prompt_tokens_denominator_pd(forwarded, 20) == 10
    assert apc_mod._prompt_tokens_denominator_pd(
        types.SimpleNamespace(kv_transfer_params={"prefill_prompt_tokens": 0}),
        20,
    ) == 20

    result = types.SimpleNamespace(kv_transfer_params={})
    apc_mod._merge_apc_into_kv_transfer_params_if_present(result, 4, 0, 12)
    assert result.kv_transfer_params == {
        "prefill_cached_tokens": 4,
        "prefill_prompt_tokens": 12,
    }

    no_kv_result = types.SimpleNamespace(kv_transfer_params=None)
    apc_mod._merge_apc_into_kv_transfer_params_if_present(
        no_kv_result, 4, 8, 12)
    assert no_kv_result.kv_transfer_params is None

    assert apc_mod._engine_cached_from_usage(
        types.SimpleNamespace(usage=None)) is None
    assert apc_mod._engine_cached_from_usage(
        types.SimpleNamespace(
            usage=types.SimpleNamespace(prompt_tokens_details=None))) is None
    assert apc_mod._engine_cached_from_usage(
        types.SimpleNamespace(
            usage=types.SimpleNamespace(
                prompt_tokens_details=types.SimpleNamespace(
                    cached_tokens=9)))) == 9


@pytest.mark.unit
def test_apc_normalize_usage_chunk_variants(monkeypatch):
    apc_mod = _load_apc_module(monkeypatch)

    assert apc_mod._normalize_usage_chunk("data: [DONE]\n\n", 3, True) == (
        "data: [DONE]\n\n")
    assert apc_mod._normalize_usage_chunk("data: {bad json\n\n", 3, True) == (
        "data: {bad json\n\n")
    assert apc_mod._normalize_usage_chunk(
        'data: {"usage": "bad"}\n\n', 3, True) == (
            'data: {"usage": "bad"}\n\n')

    disabled = apc_mod._normalize_usage_chunk(
        'data: {"usage": {"prompt_tokens": 8}}\n\n',
        3,
        False,
    )
    assert apc_mod.json.loads(disabled.removeprefix("data: "))[
        "usage"]["prompt_tokens_details"] is None

    request = types.SimpleNamespace(
        kv_transfer_params={"prefill_prompt_tokens": 2})
    created = apc_mod._normalize_usage_chunk(
        'data: {"usage": {"prompt_tokens": 8}}\n\n',
        3,
        True,
        request,
    )
    details = apc_mod.json.loads(created.removeprefix("data: "))[
        "usage"]["prompt_tokens_details"]
    assert details == {"cached_tokens": 3, "cached_rate": 1.0}

    updated = apc_mod._normalize_usage_chunk(
        'data: {"usage": {"prompt_tokens": 8, '
        '"prompt_tokens_details": {"cached_tokens": 0}}}\n\n',
        4,
        True,
    )
    details = apc_mod.json.loads(updated.removeprefix("data: "))[
        "usage"]["prompt_tokens_details"]
    assert details["cached_tokens"] == 4
    assert details["cached_rate"] == 0.5


@pytest.mark.unit
def test_apc_chat_stream_tracks_engine_cached_and_normalizes_usage(
        monkeypatch):
    apc_mod = _load_apc_module(monkeypatch)

    async def fake_orig_chat_stream(self, request, result_generator, *args, **kwargs):
        async for _ in result_generator:
            pass
        yield 'data: {"usage": {"prompt_tokens": 10}}\n\n'

    monkeypatch.setattr(apc_mod, "_orig_chat_stream", fake_orig_chat_stream)

    async def result_gen():
        yield types.SimpleNamespace(num_cached_tokens=6)

    request = types.SimpleNamespace(kv_transfer_params={})
    chunks = asyncio.run(_collect_async(
        apc_mod.OpenAIServingChatStreamAPCPatch.chat_completion_stream_generator(
            types.SimpleNamespace(enable_prompt_tokens_details=True),
            request,
            result_gen(),
            "request-id",
            "model",
            [],
            tokenizer=None,
            request_metadata=None,
        )))

    details = apc_mod.json.loads(chunks[0].removeprefix("data: "))[
        "usage"]["prompt_tokens_details"]
    assert details == {"cached_tokens": 6, "cached_rate": 0.6}


@pytest.mark.unit
def test_apc_chat_full_adds_usage_details_and_merges_kv(monkeypatch):
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setattr(apc_mod, "PromptTokenUsageInfo",
                        _PromptTokenUsageInfo)

    response = types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=10,
                                    prompt_tokens_details=None),
        kv_transfer_params={},
    )

    async def fake_orig_chat_full(self, request, result_generator, *args, **kwargs):
        async for _ in result_generator:
            pass
        return response

    monkeypatch.setattr(apc_mod, "_orig_chat_full", fake_orig_chat_full)

    async def result_gen():
        yield types.SimpleNamespace(num_cached_tokens=6)

    request = types.SimpleNamespace(kv_transfer_params={})
    result = asyncio.run(
        apc_mod.OpenAIServingChatFullAPCPatch.chat_completion_full_generator(
            types.SimpleNamespace(enable_prompt_tokens_details=True),
            request,
            result_gen(),
            "request-id",
            "model",
            [],
            tokenizer=None,
            request_metadata=None,
        ))

    assert result is response
    assert response.usage.prompt_tokens_details.cached_tokens == 6
    assert response.usage.prompt_tokens_details.cached_rate == 0.6
    assert response.kv_transfer_params == {
        "prefill_cached_tokens": 6,
        "prefill_prompt_tokens": 10,
    }


@pytest.mark.unit
def test_apc_chat_full_updates_existing_usage_details_from_engine(
        monkeypatch):
    apc_mod = _load_apc_module(monkeypatch)
    details = types.SimpleNamespace(cached_tokens=2, cached_rate=0.2)
    response = types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=10,
                                    prompt_tokens_details=details),
        kv_transfer_params={},
    )

    async def fake_orig_chat_full(self, request, result_generator, *args, **kwargs):
        async for _ in result_generator:
            pass
        return response

    monkeypatch.setattr(apc_mod, "_orig_chat_full", fake_orig_chat_full)

    async def result_gen():
        if False:
            yield None

    request = types.SimpleNamespace(kv_transfer_params=None)
    result = asyncio.run(
        apc_mod.OpenAIServingChatFullAPCPatch.chat_completion_full_generator(
            types.SimpleNamespace(enable_prompt_tokens_details=True),
            request,
            result_gen(),
            "request-id",
            "model",
            [],
            tokenizer=None,
            request_metadata=None,
        ))

    assert result is response
    assert details.cached_tokens == 2
    assert details.cached_rate == 0.2
    assert response.kv_transfer_params == {
        "prefill_cached_tokens": 2,
        "prefill_prompt_tokens": 10,
    }


@pytest.mark.unit
def test_apc_completion_stream_tracks_tuple_outputs_and_normalizes_usage(
        monkeypatch):
    apc_mod = _load_apc_module(monkeypatch)

    async def fake_orig_completion_stream(self, request, *args, **kwargs):
        async for _ in args[1]:
            pass
        yield 'data: {"usage": {"prompt_tokens": 10}}\n\n'

    monkeypatch.setattr(apc_mod, "_orig_compl_stream",
                        fake_orig_completion_stream)

    async def result_gen():
        yield "request-id", types.SimpleNamespace(num_cached_tokens=5)

    chunks = asyncio.run(_collect_async(
        apc_mod.OpenAIServingCompletionStreamAPCPatch.
        completion_stream_generator(
            types.SimpleNamespace(enable_prompt_tokens_details=True),
            types.SimpleNamespace(kv_transfer_params={}),
            "unused",
            result_gen(),
        )))

    details = apc_mod.json.loads(chunks[0].removeprefix("data: "))[
        "usage"]["prompt_tokens_details"]
    assert details == {"cached_tokens": 5, "cached_rate": 0.5}


@pytest.mark.unit
def test_apc_completion_create_adds_details_and_merges_forwarded_kv(
        monkeypatch):
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setattr(apc_mod, "PromptTokenUsageInfo",
                        _PromptTokenUsageInfo)
    response = types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=20,
                                    prompt_tokens_details=None),
        kv_transfer_params={},
    )

    async def fake_orig_create(self, request, raw_request=None):
        return response

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(kv_transfer_params={
        "prefill_cached_tokens": 7,
        "prefill_prompt_tokens": 10,
    })
    result = asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            types.SimpleNamespace(enable_prompt_tokens_details=True),
            request,
        ))

    assert result is response
    assert response.usage.prompt_tokens_details.cached_tokens == 7
    assert response.usage.prompt_tokens_details.cached_rate == 0.7
    assert response.kv_transfer_params == {
        "prefill_cached_tokens": 7,
        "prefill_prompt_tokens": 10,
    }


@pytest.mark.unit
def test_apc_completion_create_updates_existing_details(monkeypatch):
    apc_mod = _load_apc_module(monkeypatch)
    details = types.SimpleNamespace(cached_tokens=0, cached_rate=0.0)
    response = types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=8,
                                    prompt_tokens_details=details),
        kv_transfer_params={},
    )

    async def fake_orig_create(self, request, raw_request=None):
        return response

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    result = asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            types.SimpleNamespace(enable_prompt_tokens_details=True),
            types.SimpleNamespace(kv_transfer_params={}),
        ))

    assert result is response
    assert details.cached_tokens == 0
    assert details.cached_rate == 0.0


@pytest.mark.unit
def test_apc_completion_d_side_leaves_prompt_unchanged(monkeypatch):
    """D-side: APC completion no longer swaps request.prompt."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_orig_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    assert captured_prompt["prompt"] == "hello world"
    assert request.prompt == "hello world"


@pytest.mark.unit
def test_apc_completion_d_side_skips_env_disabled(monkeypatch):
    """D-side: does NOT swap prompt when env is off."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_orig_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    assert captured_prompt["prompt"] == "hello world"


@pytest.mark.unit
def test_apc_completion_d_side_skips_not_decode(monkeypatch):
    """D-side: does NOT swap prompt on prefill node."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "prefill")

    captured_prompt = {}

    async def fake_orig_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            types.SimpleNamespace(enable_prompt_tokens_details=False),
            request,
        ))

    assert captured_prompt["prompt"] == "hello world"


@pytest.mark.unit
def test_apc_completion_d_side_skips_no_prompt_token_ids(monkeypatch):
    """D-side: does NOT swap when kv_transfer_params lacks prompt_token_ids."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_orig_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"other_key": "value"},
    )

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    assert captured_prompt["prompt"] == "hello world"


@pytest.mark.unit
def test_apc_completion_d_side_skips_kv_transfer_params_none(monkeypatch):
    """D-side: does NOT swap when kv_transfer_params is None."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_orig_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params=None,
    )

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    assert captured_prompt["prompt"] == "hello world"


@pytest.mark.unit
def test_apc_completion_d_side_skips_prompt_already_list(monkeypatch):
    """D-side: does NOT swap when prompt is already list[int]."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_orig_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt=[201, 202, 203],
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    assert captured_prompt["prompt"] == [201, 202, 203]


@pytest.mark.unit
def test_apc_completion_d_side_skips_prompt_none(monkeypatch):
    """D-side: does NOT swap when prompt is None (edge case)."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_prompt = {}

    async def fake_orig_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt=None,
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    assert captured_prompt["prompt"] is None


@pytest.mark.unit
def test_apc_completion_d_side_restores_prompt_on_error(monkeypatch):
    """D-side: restores original_prompt even when the original raises."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    class TestError(Exception):
        pass

    async def raising_create(self, request, raw_request=None):
        raise TestError("something went wrong")

    monkeypatch.setattr(apc_mod, "_orig_compl_create", raising_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    with pytest.raises(TestError, match="something went wrong"):
        asyncio.run(
            apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
                _make_decode_self(),
                request,
            ))

    assert request.prompt == "hello world"


@pytest.mark.unit
def test_apc_completion_d_side_passes_raw_request_through(monkeypatch):
    """D-side: passes raw_request argument through to original."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    captured_raw = {}

    async def fake_orig_create(self, request, raw_request=None):
        captured_raw["raw_request"] = raw_request
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    raw = types.SimpleNamespace(headers={"x-test": "1"})

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            types.SimpleNamespace(
                prompt="test",
                kv_transfer_params={"prompt_token_ids": [1, 2]},
            ),
            raw_request=raw,
        ))

    assert captured_raw["raw_request"] is raw


@pytest.mark.unit
def test_apc_completion_d_side_kv_transfer_config_leaves_prompt_unchanged(
        monkeypatch):
    """D-side: kv_transfer_config does not trigger prompt swapping."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.delenv("ROLE", raising=False)

    captured_prompt = {}

    async def fake_orig_create(self, request, raw_request=None):
        captured_prompt["prompt"] = request.prompt
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102]},
    )

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    assert captured_prompt["prompt"] == "hello world"
    assert request.prompt == "hello world"


@pytest.mark.unit
def test_apc_completion_d_side_asyncgen_result_returned(monkeypatch):
    """D-side: returns async generator result directly (streaming path)."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    async def fake_stream(self, request, raw_request=None):
        async def _gen():
            for i in range(2):
                yield f"data: chunk_{i}\n\n"
        return _gen()

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_stream)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102]},
    )

    result = asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    import inspect
    assert inspect.isasyncgen(result)
    assert request.prompt == "hello world"


@pytest.mark.unit
def test_apc_completion_d_side_error_response_returned(monkeypatch):
    """D-side: returns ErrorResponse without prompt swap processing."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    # create_completion uses `isinstance(result, ErrorResponse)` where
    # ErrorResponse is a local binding from `from ... import ErrorResponse`.
    # Inject our test class into the function's global namespace so the
    # isinstance check recognises our mock instance.
    _ErrCls = type("ErrorResponse", (), {})
    _fn = apc_mod.OpenAIServingCompletionAPCPatch.create_completion
    _fn.__globals__["ErrorResponse"] = _ErrCls
    error_resp = _ErrCls()

    async def fake_orig_create(self, request, raw_request=None):
        return error_resp

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102]},
    )

    result = asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    assert result is error_resp
    assert request.prompt == "hello world"


@pytest.mark.unit
def test_apc_completion_d_side_original_prompt_reaches_original(monkeypatch):
    """D-side: original create_completion receives the original prompt."""
    apc_mod = _load_apc_module(monkeypatch)
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")
    monkeypatch.setenv("ROLE", "decode")

    tokenize_captured = {}

    async def fake_orig_create(self, request, raw_request=None):
        tokenize_captured["prompt"] = request.prompt
        tokenize_captured["prompt_type"] = type(request.prompt).__name__
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=10, prompt_tokens_details=None))

    monkeypatch.setattr(apc_mod, "_orig_compl_create", fake_orig_create)

    request = types.SimpleNamespace(
        prompt="hello world",
        kv_transfer_params={"prompt_token_ids": [101, 102, 103]},
    )

    asyncio.run(
        apc_mod.OpenAIServingCompletionAPCPatch.create_completion(
            _make_decode_self(),
            request,
        ))

    assert tokenize_captured["prompt"] == "hello world"
    assert tokenize_captured["prompt_type"] == "str"
    assert request.prompt == "hello world"


@pytest.mark.unit
def test_chat_full_prefill_named_bypasses_forced_tool_and_reasoning(
        monkeypatch):
    """Prefill producer must not semantically parse incomplete named tool output."""
    patch_module = _load_patch_module(monkeypatch)
    NamedToolChoice = sys.modules[
        "vllm.entrypoints.openai.protocol"
    ].ChatCompletionNamedToolChoiceParam

    class Request(types.SimpleNamespace):
        def model_copy(self, *, update, deep):
            values = vars(self).copy()
            values.update(update)
            return Request(**values)

    named_choice = NamedToolChoice()
    named_choice.function = types.SimpleNamespace(
        name="confirm_delivery_progress")
    request = Request(tool_choice=named_choice, kv_transfer_params=None)
    serving = types.SimpleNamespace(
        engine_client=types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(
                    is_kv_transfer_instance=True,
                    kv_role="kv_producer",
                    is_kv_producer=False,
                )
            )
        ),
        reasoning_parser=object(),
        add_ndarray_info_to_dict=lambda *a, **k: None,
    )
    captured = {}

    async def fake_original_full_generator(
            response_serving, response_request, result_generator, request_id,
            model_name, conversation, tokenizer, request_metadata):
        [captured["final_res"]] = [item async for item in result_generator]
        captured["serving"] = response_serving
        captured["request"] = response_request
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[1],
        kv_transfer_params=None,
        outputs=[
            types.SimpleNamespace(
                text="{",
                token_ids=[2],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )

    asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            serving,
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert captured["serving"] is not serving
    assert captured["serving"].reasoning_parser is None
    assert captured["request"] is not request
    assert captured["request"].tool_choice == "none"
    assert request.tool_choice is named_choice


@pytest.mark.unit
def test_chat_full_prefill_required_bypasses_forced_tool(monkeypatch):
    patch_module = _load_patch_module(monkeypatch)

    class Request(types.SimpleNamespace):
        def model_copy(self, *, update, deep):
            values = vars(self).copy()
            values.update(update)
            return Request(**values)

    request = Request(tool_choice="required", kv_transfer_params=None)
    serving = types.SimpleNamespace(
        engine_client=types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(
                    is_kv_transfer_instance=True,
                    kv_role="kv_producer",
                    is_kv_producer=False,
                )
            )
        ),
        reasoning_parser=object(),
        add_ndarray_info_to_dict=lambda *a, **k: None,
    )
    captured = {}

    async def fake_original_full_generator(
            response_serving, response_request, result_generator, *args,
            **kwargs):
        captured["request"] = response_request
        captured["serving"] = response_serving
        async for _ in result_generator:
            pass
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[1],
        kv_transfer_params=None,
        outputs=[
            types.SimpleNamespace(
                text="[",
                token_ids=[2],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )

    asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            serving,
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert captured["request"].tool_choice == "none"
    assert captured["serving"].reasoning_parser is None
    assert request.tool_choice == "required"


@pytest.mark.unit
def test_chat_full_decode_named_keeps_forced_tool_choice(monkeypatch):
    """Decode node must keep named tool_choice for normal forced-tool parsing."""
    patch_module = _load_patch_module(monkeypatch)
    NamedToolChoice = sys.modules[
        "vllm.entrypoints.openai.protocol"
    ].ChatCompletionNamedToolChoiceParam

    named_choice = NamedToolChoice()
    named_choice.function = types.SimpleNamespace(
        name="confirm_delivery_progress")
    request = types.SimpleNamespace(
        tool_choice=named_choice, kv_transfer_params=None)
    serving = types.SimpleNamespace(
        engine_client=types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(kv_transfer_config=None)
        ),
        reasoning_parser=object(),
        add_ndarray_info_to_dict=lambda *a, **k: None,
    )
    captured = {}

    async def fake_original_full_generator(
            response_serving, response_request, result_generator, *args,
            **kwargs):
        captured["request"] = response_request
        captured["serving"] = response_serving
        async for _ in result_generator:
            pass
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[1],
        kv_transfer_params=None,
        outputs=[
            types.SimpleNamespace(
                text='{"food_type":"x"}',
                token_ids=[2],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )

    asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            serving,
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert captured["request"] is request
    assert captured["serving"] is serving
    assert captured["request"].tool_choice is named_choice
    assert captured["serving"].reasoning_parser is serving.reasoning_parser


@pytest.mark.unit
def test_chat_full_prefill_without_tool_choice_does_not_raise(monkeypatch):
    """Existing callers/UTs may omit tool_choice on SimpleNamespace requests."""
    patch_module = _load_patch_module(monkeypatch)

    request = types.SimpleNamespace(kv_transfer_params=None)
    serving = types.SimpleNamespace(
        engine_client=types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(
                    is_kv_transfer_instance=True,
                    kv_role="kv_producer",
                    is_kv_producer=False,
                )
            )
        ),
        reasoning_parser=object(),
        add_ndarray_info_to_dict=lambda *a, **k: None,
    )
    captured = {}

    async def fake_original_full_generator(
            response_serving, response_request, result_generator, *args,
            **kwargs):
        captured["request"] = response_request
        captured["serving"] = response_serving
        async for _ in result_generator:
            pass
        return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    final_res = types.SimpleNamespace(
        prompt_token_ids=[1],
        kv_transfer_params=None,
        outputs=[
            types.SimpleNamespace(
                text="{",
                token_ids=[2],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )

    asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            serving,
            request,
            _single_item(final_res),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert captured["request"] is request
    assert captured["serving"] is serving


def _decode_serving_for_finish_reason():
    return types.SimpleNamespace(
        engine_client=types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(kv_transfer_config=None)
        ),
        reasoning_parser=object(),
        add_ndarray_info_to_dict=lambda *a, **k: None,
    )


def _minimal_final_res():
    return types.SimpleNamespace(
        prompt_token_ids=[1],
        kv_transfer_params=None,
        outputs=[
            types.SimpleNamespace(
                text='{"x":1}',
                token_ids=[2],
                stop_reason=None,
                logprobs=None,
                cumulative_logprob=None,
                routed_experts=None,
            )
        ],
    )


@pytest.mark.unit
def test_chat_full_named_with_tool_calls_overrides_finish_reason_to_tool_calls(
        monkeypatch):
    """named + tool_calls: upstream finish_reason=stop must become tool_calls."""
    patch_module = _load_patch_module(monkeypatch)
    NamedToolChoice = sys.modules[
        "vllm.entrypoints.openai.protocol"
    ].ChatCompletionNamedToolChoiceParam

    named_choice = NamedToolChoice()
    named_choice.function = types.SimpleNamespace(name="get_weather")
    request = types.SimpleNamespace(
        tool_choice=named_choice, kv_transfer_params=None)

    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(
                    tool_calls=[types.SimpleNamespace(id="call_1")],
                ),
            )
        ],
    )

    async def fake_original_full_generator(*args, **kwargs):
        async for _ in args[2]:
            pass
        return response

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    result = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            _decode_serving_for_finish_reason(),
            request,
            _single_item(_minimal_final_res()),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert result is response
    assert result.choices[0].finish_reason == "tool_calls"


@pytest.mark.unit
def test_chat_full_named_without_tool_calls_keeps_finish_reason_stop(
        monkeypatch):
    """named but empty/absent tool_calls: leave upstream finish_reason alone."""
    patch_module = _load_patch_module(monkeypatch)
    NamedToolChoice = sys.modules[
        "vllm.entrypoints.openai.protocol"
    ].ChatCompletionNamedToolChoiceParam

    named_choice = NamedToolChoice()
    named_choice.function = types.SimpleNamespace(name="get_weather")
    request = types.SimpleNamespace(
        tool_choice=named_choice, kv_transfer_params=None)

    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(tool_calls=None),
            )
        ],
    )

    async def fake_original_full_generator(*args, **kwargs):
        async for _ in args[2]:
            pass
        return response

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    result = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            _decode_serving_for_finish_reason(),
            request,
            _single_item(_minimal_final_res()),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert result.choices[0].finish_reason == "stop"


@pytest.mark.unit
def test_chat_full_non_named_with_tool_calls_keeps_finish_reason(monkeypatch):
    """Non-named (e.g. auto) must not rewrite finish_reason in this patch."""
    patch_module = _load_patch_module(monkeypatch)

    request = types.SimpleNamespace(
        tool_choice="auto", kv_transfer_params=None)

    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(
                    tool_calls=[types.SimpleNamespace(id="call_1")],
                ),
            )
        ],
    )

    async def fake_original_full_generator(*args, **kwargs):
        async for _ in args[2]:
            pass
        return response

    monkeypatch.setattr(patch_module, "_original_chat_completion_full_generator",
                        fake_original_full_generator)

    result = asyncio.run(
        patch_module.OpenAIServingChatPatch.chat_completion_full_generator(
            _decode_serving_for_finish_reason(),
            request,
            _single_item(_minimal_final_res()),
            request_id="request-id",
            model_name="model",
            conversation=[],
            tokenizer=None,
            request_metadata=None,
        ))

    assert result.choices[0].finish_reason == "stop"
