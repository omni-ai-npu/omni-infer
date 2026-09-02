# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import asyncio
import importlib
import sys
import types

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


async def _ready_value(value):
    return value


async def _collect_async(async_iter):
    return [item async for item in async_iter]


def _install_chat_stubs(monkeypatch):
    _stub_module(monkeypatch, "vllm", is_package=True)
    _stub_module(monkeypatch, "vllm.entrypoints", is_package=True)
    _stub_module(monkeypatch, "vllm.entrypoints.openai", is_package=True)
    inputs_mod = _stub_module(monkeypatch, "vllm.inputs", is_package=True)
    _stub_module(monkeypatch, "vllm.v1", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.engine", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.core", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.core.sched", is_package=True)
    _stub_module(monkeypatch, "vllm.lora", is_package=True)

    # Modules imported by patch_input_ids_piggyback (v0.25.1). Without these,
    # the stub `vllm` package (empty __path__) raises ModuleNotFoundError.
    parser_mod = _stub_module(monkeypatch, "vllm.parser")
    parser_mod.Parser = type("Parser", (), {})
    _stub_module(monkeypatch, "vllm.utils", is_package=True)
    utils_mistral = _stub_module(monkeypatch, "vllm.utils.mistral")

    def _mistral_feature_false(*a, **kw):
        return False

    utils_mistral.is_mistral_tokenizer = _mistral_feature_false
    utils_mistral.is_mistral_tool_parser = _mistral_feature_false
    _stub_module(monkeypatch, "vllm.renderers", is_package=True)
    hf_renderers = _stub_module(monkeypatch, "vllm.renderers.hf")

    def _resolve_content_format(*args, **kwargs):
        return "resolved"

    hf_renderers.resolve_chat_template_content_format = _resolve_content_format

    online_renderer = _stub_module(monkeypatch,
                                   "vllm.renderers.online_renderer")

    class OnlineRenderer:
        async def preprocess_chat(self, *args, **kwargs):
            return ["orig"], [{"prompt_token_ids": [9, 9]}]

    online_renderer.OnlineRenderer = OnlineRenderer

    logger_mod = _stub_module(monkeypatch, "vllm.logger")

    class Logger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    def _init_logger(*_a, **_kw):
        return Logger()

    logger_mod.init_logger = _init_logger
    logger_mod.logger = Logger()

    chat_utils = _stub_module(monkeypatch, "vllm.entrypoints.chat_utils")
    chat_utils.ChatCompletionMessageParam = dict
    chat_utils.ChatTemplateContentFormatOption = str
    chat_utils.ConversationMessage = dict
    chat_utils.resolve_chat_template_content_format = _resolve_content_format

    def _parse_chat_messages_futures(messages, *_a, **_kw):
        return (messages, _ready_value(None), None)

    chat_utils.parse_chat_messages_futures = _parse_chat_messages_futures

    # v0.25.1: parse_chat_messages is sync and returns (conv, mm_data, mm_uuids).
    def _parse_chat_messages(messages, *_a, **_kw):
        return (messages, None, None)

    chat_utils.parse_chat_messages = _parse_chat_messages

    protocol = _stub_module(monkeypatch, "vllm.entrypoints.openai.protocol")
    for _parent in ("chat_completion", "completion", "responses", "engine", "models"):
        _stub_module(monkeypatch, f"vllm.entrypoints.openai.{_parent}", is_package=True)
    _PROTO_BY_SUBPATH = {'chat_completion.protocol': ['ChatCompletionNamedToolChoiceParam', 'ChatCompletionRequest', 'ChatCompletionResponse', 'ChatCompletionResponseChoice', 'ChatCompletionResponseStreamChoice'], 'completion.protocol': ['CompletionRequest', 'CompletionResponse', 'CompletionResponseChoice', 'CompletionResponseStreamChoice'], 'responses.protocol': ['ResponsesRequest'], 'engine.protocol': ['ErrorResponse', 'GenerationError', 'PromptTokenUsageInfo', 'RequestResponseMetadata']}
    for _subpath, _names in _PROTO_BY_SUBPATH.items():
        _sub = _stub_module(monkeypatch, f"vllm.entrypoints.openai.{_subpath}")
        for _name in _names:
            setattr(_sub, _name, type(_name, (), {}))
            setattr(protocol, _name, getattr(_sub, _name))  # back-compat for legacy import path

    class ChatCompletionRequest:
        model_fields = {}
        field_names = None

        @classmethod
        def model_rebuild(cls, force=False):
            cls.rebuild_force = force

    protocol.ChatCompletionRequest = ChatCompletionRequest
    # The 0.25.1 patch imports from the chat_completion.protocol SUBMODULE (not
    # the legacy parent path), so the rich class (model_fields / model_rebuild
    # needed by patch_input_ids_piggyback._register_input_ids_field) must land
    # there too, not just on `protocol`.
    chat_comp_proto = sys.modules[
        "vllm.entrypoints.openai.chat_completion.protocol"]
    chat_comp_proto.ChatCompletionRequest = ChatCompletionRequest
    protocol.ChatCompletionNamedToolChoiceParam = type(
        "ChatCompletionNamedToolChoiceParam", (), {}
    )
    protocol.ResponsesRequest = type("ResponsesRequest", (), {})
    protocol.ChatCompletionResponse = type("ChatCompletionResponse", (), {})
    protocol.ErrorResponse = type("ErrorResponse", (), {})

    serving_chat = _stub_module(monkeypatch,
                                "vllm.entrypoints.openai.serving_chat")

    class OpenAIServingChat:
        async def _preprocess_chat(self, *args, **kwargs):
            return ["orig"], [{"prompt_token_ids": [9, 9]}]
        
        async def chat_completion_full_generator(self, *args, **kwargs):
            pass

    serving_chat.OpenAIServingChat = OpenAIServingChat

    serving_engine = _stub_module(monkeypatch,
                                  "vllm.entrypoints.openai.serving_engine")
    serving_engine.ChatLikeRequest = object

    class OpenAIServing:
        async def _preprocess_chat(self, *args, **kwargs):
            return ["orig"], [{"prompt_token_ids": [9, 9]}]
        
    serving_engine.OpenAIServing = OpenAIServing

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

    class _DummyCompletion(OpenAIServing):
        pass
    _new_comp_serving.OpenAIServingCompletion = _DummyCompletion
    _gen_base = _stub_module(
        monkeypatch, "vllm.entrypoints.generate.base.serving")
    _gen_base.GenerateBaseServing = OpenAIServing
    _typing_mod = _stub_module(
        monkeypatch, "vllm.entrypoints.serve.engine.typing")
    _typing_mod.ChatLikeRequest = object

    inputs_data = _stub_module(monkeypatch, "vllm.inputs.data")

    class TokensPrompt(dict):
        pass

    inputs_data.TokensPrompt = TokensPrompt
    inputs_data.PromptType = object

    # v0.25.1: patch_input_ids_piggyback does `from vllm.inputs import
    # EngineInput, tokens_input`. Match the real tokens_input() shape
    # (vllm/inputs/engine.py: TokensInput(type="token", prompt_token_ids,
    # cache_salt)).
    inputs_mod.EngineInput = dict

    def _tokens_input(prompt_token_ids, *, prompt=None, cache_salt=None):
        return {
            "type": "token",
            "prompt_token_ids": list(prompt_token_ids),
            **({"cache_salt": cache_salt} if cache_salt is not None else {}),
        }

    inputs_mod.tokens_input = _tokens_input

    tokenizers = _stub_module(monkeypatch, "vllm.tokenizers")
    tokenizers.TokenizerLike = type("TokenizerLike", (), {})

    tool_parsers = _stub_module(monkeypatch, "vllm.tool_parsers")
    tool_parsers.ToolParser = type("ToolParser", (), {})

    v1_engine = _stub_module(monkeypatch, "vllm.v1.engine")
    v1_engine.EngineCoreRequest = type("EngineCoreRequest", (), {})

    class EngineCoreOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    v1_engine.EngineCoreOutput = EngineCoreOutput

    kv_cache_interface = _stub_module(monkeypatch, "vllm.v1.kv_cache_interface")
    kv_cache_interface.AttentionSpec = type("AttentionSpec", (), {})

    sched_utils = _stub_module(monkeypatch, "vllm.v1.core.sched.utils")

    def _check_stop_stub(request, max_model_len):
        return False

    sched_utils.check_stop = _check_stop_stub

    exceptions_mod = _stub_module(monkeypatch, "vllm.exceptions")

    class VLLMValidationError(ValueError):
        def __init__(self, message, parameter=None, value=None):
            super().__init__(message)
            self.parameter = parameter
            self.value = value

    exceptions_mod.VLLMValidationError = VLLMValidationError

    async_llm = _stub_module(monkeypatch, "vllm.v1.engine.async_llm")

    class AsyncLLM:
        async def generate(self, *args, **kwargs):
            yield None
    
    async_llm.AsyncLLM = AsyncLLM

    parallel_sampling = _stub_module(monkeypatch, "vllm.v1.engine.parallel_sampling")
    parallel_sampling.ParentRequest = type("ParentRequest", (), {})

    v1_request = _stub_module(monkeypatch, "vllm.v1.request")
    v1_request.Request = type("Request", (), {})
    v1_request.RequestStatus = type("RequestStatus", (), {})

    scheduler_mod = _stub_module(monkeypatch, "vllm.v1.core.sched.scheduler")
    scheduler_mod.Scheduler = type("Scheduler", (), {
        "_update_from_kv_xfer_finished": lambda self, kv_connector_output: None,
        "add_request": lambda self, request: None,
    })

    lora_request = _stub_module(monkeypatch, "vllm.lora.request")
    lora_request.LoRARequest = type("LoRARequest", (), {})

    outputs = _stub_module(monkeypatch, "vllm.outputs")
    outputs.RequestOutput = type("RequestOutput", (), {})
    outputs.CompletionOutput = type("CompletionOutput", (), {})

    return ChatCompletionRequest


def _load_input_ids_module(monkeypatch):
    _install_chat_stubs(monkeypatch)
    module_name = (
        "omni_npu.vllm_patches.usefull_patch.common.patch_input_ids_piggyback")
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
def test_input_ids_helpers_read_declared_and_extra_ids(monkeypatch):
    mod = _load_input_ids_module(monkeypatch)

    assert mod._caller_input_ids(types.SimpleNamespace(input_ids=[1, 2])) == [1, 2]
    assert mod._caller_input_ids(
        types.SimpleNamespace(input_ids=None,
                              model_extra={"input_ids": [3]})) == [3]
    assert mod._caller_input_ids(types.SimpleNamespace(model_extra={})) is None

    assert mod._has_multimodal([
        {"content": [{"type": "text"}, {"type": "image_url"}]},
    ])
    assert not mod._has_multimodal([{"content": [{"type": "text"}]}])


@pytest.mark.unit
def test_input_ids_fast_path_reuses_piggyback_ids(monkeypatch):
    mod = _load_input_ids_module(monkeypatch)
    monkeypatch.setenv("OMNI_PIGGYBACK_INPUT_IDS", "1")
    monkeypatch.delenv("OMNI_SKIP_DECODE_TOKENIZE", raising=False)

    request = mod.ChatCompletionRequest()
    request.input_ids = [10, 11]
    request.truncate_prompt_tokens = None
    request.cache_salt = "salt"

    serving = types.SimpleNamespace(
        renderer=types.SimpleNamespace(tokenizer=object()),
        model_config=object(),
    )

    conversation, engine_inputs = asyncio.run(
        mod.InputIdsPiggybackPatch.preprocess_chat(
            serving,
            request,
            messages=[{"role": "user", "content": "hello"}],
            default_template="template",
            default_template_content_format="auto",
            default_template_kwargs=None,
        ))

    assert conversation == [{"role": "user", "content": "hello"}]
    # tokens_input() drops mm_processor_kwargs (text-only fast path); the
    # piggybacked ids are preserved.
    assert engine_inputs == [{
        "type": "token",
        "prompt_token_ids": [10, 11],
        "cache_salt": "salt",
    }]


@pytest.mark.unit
def test_input_ids_validation_mismatch_raises(monkeypatch):
    mod = _load_input_ids_module(monkeypatch)
    monkeypatch.setenv("OMNI_PIGGYBACK_INPUT_IDS", "1")
    monkeypatch.setenv("OMNI_VALIDATE_PIGGYBACK_INPUT_IDS", "1")

    async def fake_original(*args, **kwargs):
        return ["orig"], [{"prompt_token_ids": [99]}]

    monkeypatch.setattr(mod, "_original_preprocess_chat", fake_original)

    request = mod.ChatCompletionRequest()
    request.input_ids = [10]
    request.truncate_prompt_tokens = None

    tokenizer = types.SimpleNamespace(decode=lambda ids: f"tok-{ids[0]}")

    with pytest.raises(ValueError, match="Input IDs verification failed"):
        asyncio.run(
            mod.InputIdsPiggybackPatch.preprocess_chat(
                types.SimpleNamespace(
                    renderer=types.SimpleNamespace(tokenizer=tokenizer),
                    model_config=object(),
                ),
                request,
                messages=[{"role": "user", "content": "hello"}],
                default_template="template",
                default_template_content_format="auto",
                default_template_kwargs=None,
            ))


@pytest.mark.unit
def test_input_ids_fallback_calls_original_for_multimodal(monkeypatch):
    mod = _load_input_ids_module(monkeypatch)
    monkeypatch.setenv("OMNI_PIGGYBACK_INPUT_IDS", "1")

    async def fake_original(*args, **kwargs):
        return ["fallback"], [{"prompt_token_ids": [1]}]

    monkeypatch.setattr(mod, "_original_preprocess_chat", fake_original)

    request = mod.ChatCompletionRequest()
    request.input_ids = [10]
    request.truncate_prompt_tokens = None

    result = asyncio.run(
        mod.InputIdsPiggybackPatch.preprocess_chat(
            types.SimpleNamespace(
                renderer=types.SimpleNamespace(tokenizer=object()),
                model_config=object(),
            ),
            request,
            messages=[{"content": [{"type": "image_url"}]}],
            default_template="template",
            default_template_content_format="auto",
            default_template_kwargs=None,
        ))

    assert result == (["fallback"], [{"prompt_token_ids": [1]}])


@pytest.mark.unit
def test_input_ids_conflicting_skip_decode_setting_asserts(monkeypatch):
    mod = _load_input_ids_module(monkeypatch)
    monkeypatch.setenv("OMNI_PIGGYBACK_INPUT_IDS", "1")
    monkeypatch.setenv("OMNI_SKIP_DECODE_TOKENIZE", "1")

    with pytest.raises(AssertionError, match="requires OMNI_SKIP_DECODE_TOKENIZE=0"):
        asyncio.run(
            mod.InputIdsPiggybackPatch.preprocess_chat(
                types.SimpleNamespace(),
                types.SimpleNamespace(input_ids=[1]),
                messages=[],
                default_template=None,
                default_template_content_format="auto",
                default_template_kwargs=None,
            ))


@pytest.mark.unit
def test_grammar_bitmask_backend_forces_native_on_npu(monkeypatch):
    # The bitmask patch lives in ``patch_vllm_structured_output``, whose import
    # chain pulls in vllm's structured-output backends. Those reference xgrammar
    # symbols (e.g. ``GrammarMatcher``) at class-definition time, so use the real
    # xgrammar module here and only spy on the function under test rather than
    # stubbing the whole module out.
    import xgrammar

    calls = []
    monkeypatch.setattr(
        xgrammar, "apply_token_bitmask_inplace",
        lambda logits, bitmask, *args, **kwargs: calls.append(kwargs))

    module_name = (
        "omni_npu.vllm_patches.usefull_patch.common.patch_vllm_structured_output")
    sys.modules.pop(module_name, None)
    mod = importlib.import_module(module_name)

    mod.NPUGrammarBitmaskBackendPatch.apply_token_bitmask_inplace(
        types.SimpleNamespace(device=types.SimpleNamespace(type="npu")),
        "mask",
    )
    mod.NPUGrammarBitmaskBackendPatch.apply_token_bitmask_inplace(
        types.SimpleNamespace(device=types.SimpleNamespace(type="cpu")),
        "mask",
    )

    assert calls == [{"backend": "torch_native"}, {"backend": "auto"}]


@pytest.mark.unit
def test_split_stream_patch_splits_and_passthrough(monkeypatch):
    _install_chat_stubs(monkeypatch)
    apc_name = "omni_npu.vllm_patches.usefull_patch.common.patch_serving_apc"
    apc = types.ModuleType(apc_name)

    async def raw_stream(self):
        yield 'data: {"choices": [{"delta": {"reasoning": "r", "content": "c"}}]}\n\n'
        yield "data: [DONE]\n\n"

    apc.OpenAIServingChatStreamAPCPatch = type(
        "OpenAIServingChatStreamAPCPatch", (),
        {"chat_completion_stream_generator": raw_stream})
    monkeypatch.setitem(sys.modules, apc_name, apc)

    module_name = (
        "omni_npu.vllm_patches.patches.common.patch_split_reasoning_content")
    sys.modules.pop(module_name, None)
    mod = importlib.import_module(module_name)

    chunks = asyncio.run(_collect_async(
        mod.OpenAIServingChatStreamSplitPatch.chat_completion_stream_generator(
            types.SimpleNamespace())))

    assert len(chunks) == 3
    assert '"reasoning":"r"' in chunks[0]
    assert '"content":"c"' in chunks[1]
    assert chunks[2] == "data: [DONE]\n\n"
