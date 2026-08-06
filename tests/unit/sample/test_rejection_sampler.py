# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import importlib
import importlib.machinery
import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch


class _DummyStreamCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyDefaultStream:
    def __init__(self):
        self.waited_streams = []

    def wait_stream(self, stream):
        self.waited_streams.append(stream)


@pytest.fixture
def rejection_mod(monkeypatch):
    fake_default_stream = _DummyDefaultStream()
    # Load real torch_npu first so package submodules (e.g. _inductor) stay
    # importable after we overlay a lightweight stub for unit tests.
    import torch_npu as real_torch_npu  # noqa: F401
    try:
        importlib.import_module("torch_npu._inductor")
    except Exception:
        monkeypatch.setitem(
            sys.modules,
            "torch_npu._inductor",
            types.ModuleType("torch_npu._inductor"),
        )

    fake_torch_npu = types.ModuleType("torch_npu")
    fake_aiter_ops = types.ModuleType("vllm._aiter_ops")
    fake_torch_npu.__spec__ = importlib.machinery.ModuleSpec(
        "torch_npu", loader=None, is_package=True
    )
    fake_torch_npu.__path__ = []
    fake_aiter_ops.__spec__ = importlib.machinery.ModuleSpec("vllm._aiter_ops", loader=None)
    def _fallback_attr(name):
        if name.startswith("npu_"):
            return lambda *args, **kwargs: None
        raise AttributeError(f"module 'torch_npu' has no attribute {name}")
    setattr(
        fake_torch_npu,
        "npu",
        types.SimpleNamespace(
            stream=lambda _stream: _DummyStreamCtx(),
            Stream=lambda: "fake_npu_stream",
            get_device_name=lambda index=0: "Ascend910B2",
        ),
    )
    setattr(fake_torch_npu, "__getattr__", _fallback_attr)
    setattr(fake_torch_npu, "npu_top_k_top_p_sample", lambda *args, **kwargs: None)
    setattr(fake_torch_npu, "npu_fusion_attention", lambda *args, **kwargs: None)
    setattr(
        fake_torch_npu,
        "_C",
        types.SimpleNamespace(_NPUTaskGroupHandle=object),
    )
    setattr(
        fake_aiter_ops,
        "rocm_aiter_ops",
        types.SimpleNamespace(is_enabled=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", fake_aiter_ops)
    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(
            current_stream=lambda: fake_default_stream,
            Stream=object,
            Event=object,
            NPUGraph=object,
            ExternalEvent=object,
            is_available=lambda: False,
            config=types.SimpleNamespace(),
        ),
        raising=False,
    )

    import omni.sample.rejection_sampler as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(
        mod.torch,
        "npu",
        types.SimpleNamespace(
            current_stream=lambda: fake_default_stream,
            stream=lambda s=None: _DummyStreamCtx() if s is not None else "stream_from_npu",
        ),
        raising=False,
    )
    return mod, fake_torch_npu, fake_default_stream


def test_expand_batch_to_tokens_with_replacement(rejection_mod):
    mod, _, _ = rejection_mod
    x = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
    cu = torch.tensor([2, 5, 6], dtype=torch.int64)

    out = mod.expand_batch_to_tokens(x, cu, num_tokens=6, replace_from=0, replace_to=9)

    expected = torch.tensor([9.0, 9.0, 2.0, 2.0, 2.0, 9.0], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_compute_probs_returns_logits_for_all_greedy(rejection_mod):
    mod, _, _ = rejection_mod
    logits = torch.randn(3, 4)
    metadata = SimpleNamespace(all_greedy=True, temperature=None, top_k=None, top_p=None)

    out = mod.compute_probs(logits, cu_num_draft_tokens=torch.tensor([1, 3]), sampling_metadata=metadata)

    assert out is logits


def test_compute_probs_applies_temperature_and_topk_topp(rejection_mod, monkeypatch):
    mod, _, _ = rejection_mod
    captured = {}

    def fake_apply(logits, k, p):
        captured["logits"] = logits.clone()
        captured["k"] = k.clone()
        captured["p"] = p.clone()
        return logits + 3

    monkeypatch.setattr(mod, "apply_top_k_top_p_npu", fake_apply)

    logits = torch.tensor([[4.0, 2.0], [10.0, 6.0], [8.0, 4.0]], dtype=torch.float32)
    metadata = SimpleNamespace(
        all_greedy=False,
        temperature=torch.tensor([mod.GREEDY_TEMPERATURE, 2.0], dtype=torch.float32),
        top_k=torch.tensor([1, 2], dtype=torch.int64),
        top_p=torch.tensor([0.7, 0.9], dtype=torch.float32),
    )
    cu = torch.tensor([1, 3], dtype=torch.int64)

    out = mod.compute_probs(logits, cu_num_draft_tokens=cu, sampling_metadata=metadata)

    expected_after_div = torch.tensor([[4.0, 2.0], [5.0, 3.0], [4.0, 2.0]], dtype=torch.float32)
    assert torch.allclose(captured["logits"], expected_after_div)
    assert torch.equal(captured["k"], torch.tensor([1, 2, 2], dtype=torch.int64))
    assert torch.allclose(captured["p"], torch.tensor([0.7, 0.9, 0.9], dtype=torch.float32))
    assert torch.allclose(out, expected_after_div + 3)


def test_sample_recovered_tokens_native_without_draft_probs(rejection_mod):
    mod, _, _ = rejection_mod
    recovered = torch.empty((2,), dtype=torch.int32)
    cu = torch.tensor([2], dtype=torch.int64)
    draft_ids = torch.tensor([0, 1], dtype=torch.int64)
    target_probs = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.6, 0.0]], dtype=torch.float32)
    q = torch.ones((1, 3), dtype=torch.float32)

    mod.sample_recovered_tokens_native(
        recovered_token_ids=recovered,
        cu_num_draft_tokens=cu,
        draft_token_ids=draft_ids,
        draft_probs=None,
        target_probs=target_probs,
        q=q,
        vocab_size=3,
        PADDED_VOCAB_SIZE=4,
        NO_DRAFT_PROBS=True,
    )

    assert torch.equal(recovered, torch.tensor([2, 0], dtype=torch.int32))


def test_select_tokens_by_accepted_with_fill_mask(rejection_mod):
    mod, _, _ = rejection_mod
    output = torch.full((2, 3), mod.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    accepted = torch.tensor([True, False, True], dtype=torch.bool)
    cu = torch.tensor([2, 3], dtype=torch.int64)
    draft = torch.tensor([10, 11, 12], dtype=torch.int32)
    recovered = torch.tensor([20, 21, 22], dtype=torch.int32)
    bonus = torch.tensor([[30], [31]], dtype=torch.int32)
    fill_this_time = torch.tensor([True, False], dtype=torch.bool)

    mod.select_tokens_by_accepted(
        output_token_ids=output,
        accepted=accepted,
        cu_num_draft_tokens=cu,
        draft_token_ids=draft,
        recovered_token_ids=recovered,
        bonus_token_ids=bonus,
        fill_this_time=fill_this_time,
        max_spec_len=2,
    )

    expected = torch.tensor([[10, 21, mod.PLACEHOLDER_TOKEN_ID], [mod.PLACEHOLDER_TOKEN_ID] * 3], dtype=torch.int32)
    assert torch.equal(output, expected)


def test_rejection_greedy_sample_native_basic(rejection_mod):
    mod, _, _ = rejection_mod
    output = torch.full((1, 3), mod.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)

    mod.rejection_greedy_sample_native(
        output_token_ids=output,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        target_argmax=torch.tensor([1, 5], dtype=torch.int32),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        is_greedy=None,
        max_spec_len=2,
    )

    assert torch.equal(output, torch.tensor([[1, 5, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32))


def test_rejection_random_sample_native_with_draft_probs(rejection_mod):
    mod, _, _ = rejection_mod
    output = torch.full((1, 3), mod.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    target_probs = torch.tensor([[0.9, 0.1], [0.9, 0.1]], dtype=torch.float32)
    draft_probs = torch.tensor([[0.95, 0.05], [0.05, 0.95]], dtype=torch.float32)

    mod.rejection_random_sample_native(
        output_token_ids=output,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_token_ids=torch.tensor([0, 1], dtype=torch.int32),
        draft_probs=draft_probs,
        target_probs=target_probs,
        bonus_token_ids=torch.tensor([[7]], dtype=torch.int32),
        recovered_token_ids=torch.tensor([3, 4], dtype=torch.int32),
        uniform_probs=torch.tensor([0.5, 0.95], dtype=torch.float32),
        is_greedy=torch.tensor([False]),
        max_spec_len=2,
        vocab_size=2,
        NO_DRAFT_PROBS=False,
    )

    assert torch.equal(output, torch.tensor([[0, 4, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32))


def test_simple_verify_returns_expected_shape(rejection_mod):
    mod, _, _ = rejection_mod
    out, accepted_num = mod.simple_verify(
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        target_token_ids=torch.tensor([1, 3], dtype=torch.int32),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(),
    )

    expected = torch.tensor([[1, 3, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    assert torch.equal(out, expected)
    assert torch.equal(accepted_num, torch.tensor([1], dtype=torch.int32))


def test_generate_random_sequence_respects_seeded_generators(rejection_mod):
    mod, _, _ = rejection_mod
    probs = torch.zeros((1, 4), dtype=torch.float32)
    spec_meta = SimpleNamespace(num_draft_tokens=[1, 0])

    g0 = torch.Generator(device=probs.device.type).manual_seed(42)
    out1 = mod.generate_random_sequence(
        probs=probs,
        sampling_metadata=SimpleNamespace(generators={0: g0}),
        spec_metadata=spec_meta,
    )
    g0 = torch.Generator(device=probs.device.type).manual_seed(42)
    out2 = mod.generate_random_sequence(
        probs=probs,
        sampling_metadata=SimpleNamespace(generators={0: g0}),
        spec_metadata=spec_meta,
    )

    assert out1.shape == probs.shape
    assert torch.allclose(out1, out2)


def test_compute_probs_and_sample_all_greedy_path(rejection_mod):
    mod, _, _ = rejection_mod
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)
    token_ids, out_logits = mod.compute_probs_and_sample(
        logits=logits.clone(),
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        sampling_metadata=SimpleNamespace(all_greedy=True),
        q=None,
        use_npu_sample=True,
        not_support_float=False,
    )

    assert torch.equal(token_ids, torch.tensor([1, 0], dtype=torch.int32))
    assert torch.allclose(out_logits, logits)


def test_compute_probs_and_sample_non_greedy_calls_npu(rejection_mod, monkeypatch):
    mod, fake_torch_npu, fake_default_stream = rejection_mod
    called = {}

    def fake_sample(logits, top_k, top_p, q, is_need_logits):
        called["logits_dtype"] = logits.dtype
        called["top_k"] = top_k.clone()
        called["top_p"] = top_p.clone()
        called["q"] = q.clone()
        called["is_need_logits"] = is_need_logits
        return torch.tensor([1, 0], dtype=torch.int32), logits + 2

    fake_torch_npu.npu_top_k_top_p_sample = fake_sample
    stream = object()
    logits = torch.tensor([[4.0, 2.0], [6.0, 3.0]], dtype=torch.float32)
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        temperature=torch.tensor([1.0], dtype=torch.float32),
        top_k=None,
        top_p=None,
        generators={},
    )
    metadata = SimpleNamespace(num_draft_tokens=[1])

    token_ids, out_logits = mod.compute_probs_and_sample(
        logits=logits,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        sampling_metadata=sampling_metadata,
        q=torch.ones_like(logits, dtype=torch.float32),
        use_npu_sample=True,
        not_support_float=False,
    )

    assert called["logits_dtype"] == torch.float32
    assert torch.equal(called["top_k"], torch.tensor([2, 2], dtype=torch.int32))
    assert torch.all(called["top_p"] == 1)
    assert called["q"].dtype == torch.float32
    assert called["is_need_logits"] is True
    assert torch.equal(token_ids, torch.tensor([1, 0], dtype=torch.int32))
    assert out_logits.dtype == torch.float32


def test_compute_probs_and_sample_not_support_float_casts_logits(rejection_mod, monkeypatch):
    mod, fake_torch_npu, _ = rejection_mod
    called = {}

    def fake_sample(logits, top_k, top_p, q, is_need_logits):
        called["logits_dtype"] = logits.dtype
        called["top_p_dtype"] = top_p.dtype
        called["logits_max"] = logits.max().item()
        return torch.tensor([1, 0], dtype=torch.int32), logits

    fake_torch_npu.npu_top_k_top_p_sample = fake_sample
    monkeypatch.setattr(
        mod,
        "model_extra_config",
        SimpleNamespace(dtype=torch.bfloat16),
    )

    logits = torch.tensor([[4.0, 2.0], [6.0, 3.0]], dtype=torch.float32)
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        temperature=torch.tensor([1.0], dtype=torch.float32),
        top_k=None,
        top_p=None,
        generators={},
    )

    token_ids, _ = mod.compute_probs_and_sample(
        logits=logits,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        sampling_metadata=sampling_metadata,
        q=torch.ones_like(logits, dtype=torch.float32),
        use_npu_sample=True,
        not_support_float=True,
    )

    assert called["logits_dtype"] == torch.bfloat16
    assert called["top_p_dtype"] == torch.bfloat16
    assert called["logits_max"] <= 0.0
    assert torch.equal(token_ids, torch.tensor([1, 0], dtype=torch.int32))


def test_rejection_sample_all_greedy_short_circuit(rejection_mod):
    mod, _, _ = rejection_mod
    output = mod.rejection_sample(
        draft_token_ids=torch.tensor([1, 1], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_probs=None,
        target_probs=torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32),
        bonus_token_ids=torch.tensor([[7]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(
            all_greedy=True,
            all_random=False,
            temperature=torch.tensor([mod.GREEDY_TEMPERATURE], dtype=torch.float32),
            generators={},
        ),
        stream=_DummyDefaultStream(),
    )

    assert torch.equal(output, torch.tensor([[0, mod.PLACEHOLDER_TOKEN_ID, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32))


def test_rejection_sample_random_path_invokes_helpers(rejection_mod, monkeypatch):
    mod, _, _ = rejection_mod
    called = {}

    monkeypatch.setattr(
        mod,
        "generate_uniform_probs",
        lambda *args, **kwargs: torch.tensor([0.2, 0.9], dtype=torch.float32),
    )

    def fake_sample_recovered_tokens(*args, **kwargs):
        called["sample_recovered_tokens_called"] = True
        return torch.tensor([4, 5], dtype=torch.int32)

    monkeypatch.setattr(mod, "sample_recovered_tokens", fake_sample_recovered_tokens)

    def fake_rejection_random_sample_native(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS,
    ):
        called["random_called"] = True
        called["no_draft_probs"] = NO_DRAFT_PROBS
        output_token_ids[:] = torch.tensor([[4, 5, 6]], dtype=torch.int32)

    monkeypatch.setattr(mod, "rejection_random_sample_native", fake_rejection_random_sample_native)

    out = mod.rejection_sample(
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_probs=None,
        target_probs=torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(
            all_greedy=False,
            all_random=True,
            temperature=torch.tensor([0.7], dtype=torch.float32),
            generators={},
        ),
        stream=_DummyDefaultStream(),
    )

    assert called["sample_recovered_tokens_called"] is True
    assert called["random_called"] is True
    assert called["no_draft_probs"] is True
    assert torch.equal(out, torch.tensor([[4, 5, 6]], dtype=torch.int32))


@dataclass
class _ForwardSamplingMetadata:
    max_num_logprobs: int | None
    all_greedy: bool = False
    generators: dict = field(default_factory=dict)


def _build_npu_rejection_sampler(mod, monkeypatch, enable_mtp_invariant=False):
    monkeypatch.setattr(mod, "on_ascend950", lambda: False)
    monkeypatch.setattr(
        mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                disable_npu_top_k_top_p_sample=False,
                enable_mtp_invariant=enable_mtp_invariant,
            )
        ),
    )
    base_sampler = SimpleNamespace(
        logprobs_mode="raw_logits",
        dsa_stream=_DummyDefaultStream(),
    )
    return mod.NPURejectionSampler(base_sampler)


def _build_spec_decode_metadata():
    return SimpleNamespace(
        max_spec_len=1,
        bonus_logits_indices=torch.tensor([0], dtype=torch.int64),
        target_logits_indices=torch.tensor([1], dtype=torch.int64),
        draft_token_ids=torch.tensor([1], dtype=torch.int32),
        num_draft_tokens=[1],
        cu_num_draft_tokens=torch.tensor([1], dtype=torch.int64),
    )


def _patch_npu_rejection_forward_deps(mod, rejection, monkeypatch, output_ids):
    bonus_output = SimpleNamespace(
        sampled_token_ids=torch.tensor([[9]], dtype=torch.int32),
        logprobs_tensors=SimpleNamespace(logprobs=torch.zeros(1, 4)),
    )
    rejection.sampler = lambda *args, **kwargs: bonus_output
    monkeypatch.setattr(
        rejection, "apply_logits_processors", lambda target_logits, sm, md: target_logits
    )
    monkeypatch.setattr(
        mod,
        "compute_probs_and_sample",
        lambda target_logits, cu, sm, q, use_npu_sample, not_support_float=False: (
            torch.tensor([1], dtype=torch.int32),
            target_logits,
        ),
    )
    monkeypatch.setattr(
        mod, "simple_verify", lambda *args, **kwargs: (output_ids, torch.tensor([0], dtype=torch.int32))
    )


def _patch_npu_rejection_forward_deps_mtp(mod, rejection, monkeypatch, output_ids, accepted_num):
    bonus_output = SimpleNamespace(
        sampled_token_ids=torch.tensor([[9]], dtype=torch.int32),
        logprobs_tensors=SimpleNamespace(logprobs=torch.zeros(1, 4)),
    )
    rejection.sampler = lambda *args, **kwargs: bonus_output
    rejection.is_processed_logprobs_mode = False
    rejection.apply_logits_processors = lambda target_logits, sm, md: target_logits
    monkeypatch.setattr(
        mod,
        "compute_probs_and_sample",
        lambda target_logits, cu, sm, q, use_npu_sample, not_support_float=False: (
            torch.tensor([0], dtype=torch.int32),
            target_logits,
        ),
    )
    monkeypatch.setattr(
        mod, "simple_verify", lambda *args, **kwargs: (output_ids, accepted_num)
    )


def test_npu_rejection_sampler_forward_collects_logprobs_when_max_num_logprobs_zero(
    rejection_mod, monkeypatch
):
    """top_logprobs=0 still requires sampled-token logprobs."""
    mod, _, _ = rejection_mod
    rejection = _build_npu_rejection_sampler(mod, monkeypatch)
    metadata = _build_spec_decode_metadata()
    output_ids = torch.tensor([[1, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    _patch_npu_rejection_forward_deps(mod, rejection, monkeypatch, output_ids)

    fake_logprobs = SimpleNamespace(dummy=True)
    get_logprobs_calls = []
    monkeypatch.setattr(
        rejection,
        "_get_logprobs_tensors",
        lambda max_num_logprobs, *args, **kwargs: (
            get_logprobs_calls.append(max_num_logprobs) or fake_logprobs
        ),
    )

    out = rejection.forward(
        metadata=metadata,
        draft_probs=None,
        logits=torch.randn(2, 4),
        sampling_metadata=_ForwardSamplingMetadata(max_num_logprobs=0),
    )

    assert get_logprobs_calls == [0]
    assert out.logprobs_tensors is fake_logprobs
    assert torch.equal(out.sampled_token_ids, output_ids)


def test_npu_rejection_sampler_forward_skips_logprobs_when_max_num_logprobs_none(
    rejection_mod, monkeypatch
):
    mod, _, _ = rejection_mod
    rejection = _build_npu_rejection_sampler(mod, monkeypatch)
    metadata = _build_spec_decode_metadata()
    output_ids = torch.tensor([[1, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    _patch_npu_rejection_forward_deps(mod, rejection, monkeypatch, output_ids)

    get_logprobs_called = []
    monkeypatch.setattr(
        rejection,
        "_get_logprobs_tensors",
        lambda *args, **kwargs: get_logprobs_called.append(True),
    )

    out = rejection.forward(
        metadata=metadata,
        draft_probs=None,
        logits=torch.randn(2, 4),
        sampling_metadata=_ForwardSamplingMetadata(max_num_logprobs=None),
    )

    assert get_logprobs_called == []
    assert out.logprobs_tensors is None
    assert torch.equal(out.sampled_token_ids, output_ids)


def test_select_tokens_by_accepted_returns_accepted_num(rejection_mod):
    mod, _, _ = rejection_mod
    output = torch.full((2, 3), mod.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    accepted = torch.tensor([True, False, True], dtype=torch.bool)
    cu = torch.tensor([2, 3], dtype=torch.int64)
    draft = torch.tensor([10, 11, 12], dtype=torch.int32)
    recovered = torch.tensor([20, 21, 22], dtype=torch.int32)
    bonus = torch.tensor([[30], [31]], dtype=torch.int32)

    accepted_num = mod.select_tokens_by_accepted(
        output_token_ids=output,
        accepted=accepted,
        cu_num_draft_tokens=cu,
        draft_token_ids=draft,
        recovered_token_ids=recovered,
        bonus_token_ids=bonus,
        fill_this_time=None,
        max_spec_len=2,
    )

    assert torch.equal(accepted_num, torch.tensor([1, 1], dtype=torch.int32))


def test_rejection_greedy_sample_native_returns_accepted_num(rejection_mod):
    mod, _, _ = rejection_mod
    output = torch.full((1, 3), mod.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)

    accepted_num = mod.rejection_greedy_sample_native(
        output_token_ids=output,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        target_argmax=torch.tensor([1, 5], dtype=torch.int32),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        is_greedy=None,
        max_spec_len=2,
    )

    assert torch.equal(output, torch.tensor([[1, 5, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32))
    assert torch.equal(accepted_num, torch.tensor([1], dtype=torch.int32))


def test_rejection_random_sample_native_returns_accepted_num(rejection_mod):
    mod, _, _ = rejection_mod
    output = torch.full((1, 3), mod.PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    target_probs = torch.tensor([[0.9, 0.1], [0.9, 0.1]], dtype=torch.float32)
    draft_probs = torch.tensor([[0.95, 0.05], [0.05, 0.95]], dtype=torch.float32)

    accepted_num = mod.rejection_random_sample_native(
        output_token_ids=output,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        draft_token_ids=torch.tensor([0, 1], dtype=torch.int32),
        draft_probs=draft_probs,
        target_probs=target_probs,
        bonus_token_ids=torch.tensor([[7]], dtype=torch.int32),
        recovered_token_ids=torch.tensor([3, 4], dtype=torch.int32),
        uniform_probs=torch.tensor([0.5, 0.95], dtype=torch.float32),
        is_greedy=torch.tensor([False]),
        max_spec_len=2,
        vocab_size=2,
        NO_DRAFT_PROBS=False,
    )

    assert torch.equal(output, torch.tensor([[0, 4, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32))
    assert torch.equal(accepted_num, torch.tensor([1], dtype=torch.int32))


def test_simple_verify_returns_accepted_num(rejection_mod):
    mod, _, _ = rejection_mod
    output, accepted_num = mod.simple_verify(
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int64),
        target_token_ids=torch.tensor([1, 3], dtype=torch.int32),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(),
    )

    expected = torch.tensor([[1, 3, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    assert torch.equal(output, expected)
    assert torch.equal(accepted_num, torch.tensor([1], dtype=torch.int32))


def test_generate_random_sequence_mtp_invariant_row_wise(rejection_mod, monkeypatch):
    mod, _, _ = rejection_mod
    monkeypatch.setattr(
        mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(enable_mtp_invariant=True)
        ),
    )
    probs = torch.zeros((3, 5), dtype=torch.float32)
    spec_meta = SimpleNamespace(num_draft_tokens=[2])

    g = torch.Generator(device=probs.device.type).manual_seed(42)
    out = mod.generate_random_sequence(
        probs=probs,
        sampling_metadata=SimpleNamespace(generators={0: g}),
        spec_metadata=spec_meta,
    )

    g_ref = torch.Generator(device=probs.device.type).manual_seed(42)
    expected = torch.empty_like(probs)
    for row in range(probs.shape[0]):
        expected[row : row + 1].exponential_(generator=g_ref)

    assert out.shape == probs.shape
    assert torch.equal(out, expected)


def test_npu_rejection_sampler_forward_rolls_back_rng_on_rejection(
    rejection_mod, monkeypatch
):
    mod, _, _ = rejection_mod
    rejection = _build_npu_rejection_sampler(mod, monkeypatch, enable_mtp_invariant=True)
    metadata = _build_spec_decode_metadata()
    output_ids = torch.tensor([[0, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    accepted_num = torch.tensor([0], dtype=torch.int32)
    _patch_npu_rejection_forward_deps_mtp(
        mod, rejection, monkeypatch, output_ids, accepted_num
    )

    device = torch.device("npu")
    gen = torch.Generator(device=device).manual_seed(42)
    ref_gen = torch.Generator(device=device).manual_seed(42)
    ref_state = ref_gen.get_state()
    # consumed = accepted_num + 1 = 1 draw from the saved state
    dummy = torch.empty((1, 4), dtype=torch.float32, device=device)
    dummy.exponential_(generator=ref_gen)
    expected_state = ref_gen.get_state()

    out = rejection.forward(
        metadata=metadata,
        draft_probs=None,
        logits=torch.randn(2, 4, dtype=torch.float32, device=device),
        sampling_metadata=_ForwardSamplingMetadata(
            max_num_logprobs=None,
            all_greedy=False,
            generators={0: gen},
        ),
    )

    assert torch.equal(out.sampled_token_ids, output_ids)
    assert torch.equal(gen.get_state(), expected_state)


def test_npu_rejection_sampler_forward_skips_rollback_when_all_accepted(
    rejection_mod, monkeypatch
):
    mod, _, _ = rejection_mod
    rejection = _build_npu_rejection_sampler(mod, monkeypatch, enable_mtp_invariant=True)
    metadata = _build_spec_decode_metadata()
    output_ids = torch.tensor([[0, mod.PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    accepted_num = torch.tensor([1], dtype=torch.int32)
    _patch_npu_rejection_forward_deps_mtp(
        mod, rejection, monkeypatch, output_ids, accepted_num
    )

    device = torch.device("npu")
    gen = torch.Generator(device=device).manual_seed(42)
    ref_gen = torch.Generator(device=device).manual_seed(42)
    # num_draft_tokens=[1], all accepted: generator advanced n + 1 = 2 draws
    dummy = torch.empty((1, 4), dtype=torch.float32, device=device)
    dummy.exponential_(generator=ref_gen)
    dummy.exponential_(generator=ref_gen)
    expected_state = ref_gen.get_state()

    out = rejection.forward(
        metadata=metadata,
        draft_probs=None,
        logits=torch.randn(2, 4, dtype=torch.float32, device=device),
        sampling_metadata=_ForwardSamplingMetadata(
            max_num_logprobs=None,
            all_greedy=False,
            generators={0: gen},
        ),
    )

    assert torch.equal(out.sampled_token_ids, output_ids)
    assert torch.equal(gen.get_state(), expected_state)
