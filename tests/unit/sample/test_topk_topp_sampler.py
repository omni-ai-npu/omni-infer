# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import importlib
import importlib.machinery
import sys
import types

import pytest
import torch
import torch_npu

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
def topk_mod(monkeypatch):
    fake_default_stream = _DummyDefaultStream()
    # Keep real torch_npu package for vllm/torch.compile imports; only stub
    # the sampling-related entry points used by this module.
    fake_aiter_ops = types.ModuleType("vllm._aiter_ops")
    fake_aiter_ops.__spec__ = importlib.machinery.ModuleSpec("vllm._aiter_ops", loader=None)
    setattr(fake_aiter_ops, "rocm_aiter_ops", types.SimpleNamespace(is_enabled=lambda: False))
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", fake_aiter_ops)

    import omni_npu.sample.ops.topk_topp_sampler as mod

    mod = importlib.reload(mod)
    monkeypatch.setattr(
        mod.torch_npu,
        "npu_top_k_top_p_sample",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        mod.torch_npu,
        "npu_fusion_attention",
        lambda *args, **kwargs: None,
        raising=False,
    )
    if hasattr(mod.torch_npu, "npu"):
        monkeypatch.setattr(
            mod.torch_npu.npu,
            "stream",
            lambda _stream: _DummyStreamCtx(),
            raising=False,
        )
        monkeypatch.setattr(
            mod.torch_npu.npu,
            "Stream",
            lambda: "fake_npu_stream",
            raising=False,
        )
        monkeypatch.setattr(
            mod.torch_npu.npu,
            "get_device_name",
            lambda index=0: "Ascend910B2",
            raising=False,
        )
    monkeypatch.setattr(
        mod.torch,
        "npu",
        types.SimpleNamespace(
            current_stream=lambda: fake_default_stream,
            stream=lambda s=None: _DummyStreamCtx() if s is not None else "stream_from_npu",
        ),
        raising=False,
    )
    return mod, mod.torch_npu, fake_default_stream


def test_apply_top_k_top_p_npu_passthrough_when_k_p_none(topk_mod):
    mod, _, _ = topk_mod
    logits = torch.randn(2, 5, dtype=torch.float32)

    out = mod.apply_top_k_top_p_npu(logits, None, None)

    assert out is logits


def test_apply_top_k_top_p_npu_default_inputs_and_dtype_conversion(topk_mod):
    mod, fake_torch_npu, _ = topk_mod
    called = {}
    expected_dtype = mod.model_extra_config.dtype

    def fake_sample(logits, k, p, q, is_need_logits):
        called["logits_dtype"] = logits.dtype
        called["k"] = k
        called["p"] = p
        called["q"] = q
        called["is_need_logits"] = is_need_logits
        return torch.zeros((logits.shape[0],), dtype=torch.int32), logits + 1

    fake_torch_npu.npu_top_k_top_p_sample = fake_sample
    logits = torch.randn(3, 7, dtype=torch.float32)
    top_p = torch.tensor([0.8, 0.9, 1.0], dtype=torch.float32)

    out = mod.apply_top_k_top_p_npu(logits, k=None, p=top_p)

    assert out.dtype == expected_dtype
    assert called["logits_dtype"] == expected_dtype
    assert called["k"].dtype == torch.int32
    assert torch.equal(called["k"], torch.tensor([7, 7, 7], dtype=torch.int32))
    assert called["p"].dtype == expected_dtype
    assert torch.allclose(called["p"].float(), top_p, atol=2e-3, rtol=0)
    assert called["q"] is None
    assert called["is_need_logits"] is True


def test_apply_top_k_top_p_npu_respects_fp16_dtype(topk_mod, monkeypatch):
    mod, fake_torch_npu, _ = topk_mod
    called = {}

    fake_config = types.SimpleNamespace(
        dtype=torch.float16,
        operator_opt_config=types.SimpleNamespace(disable_npu_top_k_top_p_sample=False),
    )
    monkeypatch.setattr(mod, "model_extra_config", fake_config)

    def fake_sample(logits, k, p, q, is_need_logits):
        called["logits_dtype"] = logits.dtype
        called["p"] = p
        return torch.zeros((logits.shape[0],), dtype=torch.int32), logits

    fake_torch_npu.npu_top_k_top_p_sample = fake_sample
    logits = torch.randn(2, 5, dtype=torch.float32)
    top_p = torch.tensor([0.9, 1.0], dtype=torch.float32)

    out = mod.apply_top_k_top_p_npu(logits, k=None, p=top_p)

    assert out.dtype == torch.float16
    assert called["logits_dtype"] == torch.float16
    assert called["p"].dtype == torch.float16


def test_generate_coins_waits_stream_and_supports_seeded_generators(topk_mod):
    mod, _, fake_default_stream = topk_mod
    probs = torch.zeros((2, 4), dtype=torch.bfloat16)
    stream = _DummyDefaultStream()

    g0 = torch.Generator(device=probs.device.type).manual_seed(11)
    g1 = torch.Generator(device=probs.device.type).manual_seed(22)
    samp = types.SimpleNamespace(bonus_q=None, dsa_stream=stream)
    q1 = mod.generate_coins(probs, {0: g0, 1: g1}, samp)

    g0 = torch.Generator(device=probs.device.type).manual_seed(11)
    g1 = torch.Generator(device=probs.device.type).manual_seed(22)
    q2 = mod.generate_coins(probs, {0: g0, 1: g1}, samp)

    assert q1.dtype == torch.float32
    assert q1.shape == probs.shape
    assert torch.allclose(q1, q2)
    assert fake_default_stream.waited_streams[-1] is stream


@pytest.mark.parametrize(
    "logprobs_mode,expect_none",
    [
        ("raw_logprobs", True),
        ("processed_logits", False),
        ("processed_logprobs", False),
    ],
)
def test_forward_npu_handles_logprobs_modes(topk_mod, monkeypatch, logprobs_mode, expect_none):
    mod, fake_torch_npu, _ = topk_mod

    fake_logits_out = torch.tensor([[2.0, 1.0], [0.5, 0.5]], dtype=torch.bfloat16)
    fake_token_ids = torch.tensor([0, 1], dtype=torch.int32)
    fake_torch_npu.npu_top_k_top_p_sample = (
        lambda logits, k, p, q, is_need_logits: (fake_token_ids, fake_logits_out)
    )
    monkeypatch.setattr(
        mod,
        "generate_coins",
        lambda probs, generators, stream: torch.ones_like(probs, dtype=torch.float32),
    )

    sampler = mod.NPUTopKTopPSampler.__new__(mod.NPUTopKTopPSampler)
    sampler.logprobs_mode = logprobs_mode
    sampler.dsa_stream = object()
    sampler.sampler = None

    logits = torch.randn(2, 2, dtype=torch.float32)
    token_ids, logits_to_return = sampler.forward_npu(logits, generators={}, k=None, p=None)

    assert torch.equal(token_ids, fake_token_ids)
    if expect_none:
        assert logits_to_return is None
    elif logprobs_mode == "processed_logits":
        assert torch.equal(logits_to_return, fake_logits_out)
    else:
        expected = fake_logits_out.log_softmax(dim=-1, dtype=torch.float32)
        assert torch.allclose(logits_to_return, expected)


@pytest.mark.parametrize(
    "logprobs_mode,expect_none",
    [
        ("raw_logprobs", True),
        ("processed_logits", False),
        ("processed_logprobs", False),
    ],
)
def test_forward_npu_disable_bypass_path(topk_mod, monkeypatch, logprobs_mode, expect_none):
    """Test the disable_npu_top_k_top_p_sample bypass path."""
    mod, _, _ = topk_mod

    # Mock model_extra_config to enable the bypass
    fake_config = types.SimpleNamespace(
        operator_opt_config=types.SimpleNamespace(disable_npu_top_k_top_p_sample=True)
    )
    monkeypatch.setattr(mod, "model_extra_config", fake_config)

    # Mock apply_top_k_top_p and random_sample (now in-module, no longer imported)
    fake_logits = torch.tensor([[2.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
    fake_idx = torch.tensor([0, 1], dtype=torch.int64)
    fake_token_ids = torch.tensor([0, 1], dtype=torch.int32)

    monkeypatch.setattr(mod, "apply_top_k_top_p", lambda logits, k, p: (fake_logits, fake_idx))
    monkeypatch.setattr(mod, "random_sample", lambda probs, idx, gens, stream: fake_token_ids)

    sampler = mod.NPUTopKTopPSampler.__new__(mod.NPUTopKTopPSampler)
    sampler.logprobs_mode = logprobs_mode
    sampler.dsa_stream = object()
    sampler.sampler = None

    logits = torch.randn(2, 2, dtype=torch.float32)
    token_ids, logits_to_return = sampler.forward_npu(logits, generators={}, k=None, p=None)

    assert torch.equal(token_ids, fake_token_ids)
    if expect_none:
        assert logits_to_return is None
    elif logprobs_mode == "processed_logits":
        assert logits_to_return is not None
    else:
        assert logits_to_return is not None
        assert logits_to_return.dtype == torch.float32


def test_sampler_init_sets_npu_specific_attributes(topk_mod, monkeypatch):
    mod, fake_torch_npu, _ = topk_mod
    parent_called = {}

    def fake_parent_init(self, logprobs_mode):
        parent_called["logprobs_mode"] = logprobs_mode

    monkeypatch.setattr(mod.V1TopKTopPSampler, "__init__", fake_parent_init)
    monkeypatch.setattr(mod, "on_ascend950", lambda: False)

    mock_sampler = types.SimpleNamespace(dsa_stream="stream_from_npu")
    sampler = mod.NPUTopKTopPSampler(logprobs_mode="processed_logits", sampler=mock_sampler)

    assert parent_called["logprobs_mode"] == "processed_logits"
    assert sampler.apply_top_k_top_p is mod.apply_top_k_top_p_npu
    assert sampler.forward == sampler.forward_npu
    assert sampler.dsa_stream == "stream_from_npu"


# ── apply_top_k_top_p ──────────────────────────────────────────────


def test_apply_top_k_top_p_both_none(topk_mod):
    """Both k=None and p=None → passthrough: (logits, None)."""
    mod, _, _ = topk_mod
    logits = torch.randn(2, 5)
    out, idx = mod.apply_top_k_top_p(logits, None, None)
    assert out is logits
    assert idx is None


def test_apply_top_k_top_p_k_only(topk_mod):
    """p=None, k provided → delegates to apply_top_k_only, returns (result, None)."""
    mod, _, _ = topk_mod
    logits = torch.randn(2, 10, dtype=torch.float32)
    k = torch.tensor([3, 7])
    out, idx = mod.apply_top_k_top_p(logits, k, None)
    assert idx is None
    # Top-k: exactly k values per row should survive
    for row in range(2):
        n_valid = torch.sum(out[row] != -float("inf")).item()
        assert n_valid == k[row].item(), f"row {row}: expected {k[row]} valid, got {n_valid}"


def test_apply_top_k_top_p_p_only(topk_mod):
    """k=None, p provided → sorts and applies top-p mask, returns (sorted_logits, idx)."""
    mod, _, _ = topk_mod
    logits = torch.tensor([[1.0, 10.0, 100.0]], dtype=torch.float32)
    p = torch.tensor([0.0])  # keep top 0% mass → only the highest element survives
    out, idx = mod.apply_top_k_top_p(logits, k=None, p=p)
    assert idx is not None
    assert out.shape == logits.shape
    # Only the last element (highest in ascending sort) should survive
    assert out[0, -1] != -float("inf"), "highest element must survive"
    assert torch.sum(out != -float("inf")) == 1, "only one element should survive with p=0"


def test_apply_top_k_top_p_both(topk_mod):
    """Both k and p provided → applies top-k then top-p mask."""
    mod, _, _ = topk_mod
    logits = torch.randn(2, 20, dtype=torch.float32)
    k = torch.tensor([10, 10])
    p = torch.tensor([0.5, 0.5])
    out, idx = mod.apply_top_k_top_p(logits, k, p)
    assert idx is not None
    assert out.shape == logits.shape
    # At most k elements survive per row, at least 1
    for row in range(2):
        n_valid = torch.sum(out[row] != -float("inf")).item()
        assert 1 <= n_valid <= 10, f"row {row}: expected 1–10 valid, got {n_valid}"
    # Last element (highest in ascending sort) must always survive
    assert out[0, -1] != -float("inf")
    assert out[1, -1] != -float("inf")


def test_apply_top_k_top_p_p_one(topk_mod):
    """p=1.0 keeps all elements (1-p=0, no cumsum ≤ 0)."""
    mod, _, _ = topk_mod
    logits = torch.randn(2, 6, dtype=torch.float32)
    p = torch.tensor([1.0, 1.0])
    out, idx = mod.apply_top_k_top_p(logits, k=None, p=p)
    assert idx is not None
    assert out.shape == logits.shape
    n_valid_per_row = torch.sum(out != -float("inf"), dim=-1)
    assert torch.all(n_valid_per_row == 6)


# ── apply_top_k_only ───────────────────────────────────────────────


def test_apply_top_k_only_basic(topk_mod):
    """Top-k filtering with k < vocab_size."""
    mod, _, _ = topk_mod
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]], dtype=torch.float32)
    k = torch.tensor([3])
    result = mod.apply_top_k_only(logits, k)
    assert result is logits  # in-place
    # Top-3 values: 5.0, 4.0, 3.0 → everything else -inf
    expected = torch.tensor([[-float("inf"), 5.0, -float("inf"), 4.0, 3.0]], dtype=torch.float32, device=result.device)
    torch.testing.assert_close(result, expected)


def test_apply_top_k_only_full_vocab(topk_mod):
    """When k == vocab_size, no values are masked."""
    mod, _, _ = topk_mod
    logits = torch.randn(2, 5)
    original = logits.clone()
    k = torch.tensor([5, 5])
    result = mod.apply_top_k_only(logits, k)
    assert result is logits
    assert torch.equal(result, original)


def test_apply_top_k_only_k1(topk_mod):
    """k=1: only the maximum value per row survives."""
    mod, _, _ = topk_mod
    logits = torch.tensor([
        [1.0, 5.0, 2.0, 4.0, 3.0],
        [9.0, 1.0, 3.0, 2.0, 8.0],
    ], dtype=torch.float32)
    k = torch.tensor([1, 1])
    result = mod.apply_top_k_only(logits, k)
    # Exactly 1 valid value per row
    n_valid = torch.sum(result != -float("inf"), dim=-1)
    assert torch.equal(n_valid, torch.tensor([1, 1]))
    # Those values should be the row-wise maxima
    assert result[0, 1] == 5.0
    assert result[1, 0] == 9.0


# ── random_sample ──────────────────────────────────────────────────


def test_random_sample_basic(topk_mod):
    """Basic call with empty generators produces valid output shape/dtype."""
    mod, _, fake_default_stream = topk_mod
    probs = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.1, 0.6]], dtype=torch.float32)
    stream = _DummyDefaultStream()
    result = mod.random_sample(probs.clone(), idx=None, generators={}, sampler=types.SimpleNamespace(bonus_q=None, dsa_stream=stream))
    assert result.shape == (2,)
    assert result.dtype == torch.long
    assert torch.all(result >= 0) and torch.all(result < 3)
    assert fake_default_stream.waited_streams[-1] is stream


def test_random_sample_with_generators(topk_mod):
    """Seeded per-row generators produce deterministic results."""
    mod, _, _ = topk_mod
    probs = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.1, 0.6]], dtype=torch.float32)
    stream = _DummyDefaultStream()
    g0 = torch.Generator(device=probs.device.type).manual_seed(42)
    g1 = torch.Generator(device=probs.device.type).manual_seed(99)
    g0b = torch.Generator(device=probs.device.type).manual_seed(42)
    g1b = torch.Generator(device=probs.device.type).manual_seed(99)

    result1 = mod.random_sample(probs.clone(), idx=None, generators={0: g0, 1: g1}, sampler=types.SimpleNamespace(bonus_q=None, dsa_stream=stream))
    result2 = mod.random_sample(probs.clone(), idx=None, generators={0: g0b, 1: g1b}, sampler=types.SimpleNamespace(bonus_q=None, dsa_stream=_DummyDefaultStream()))

    assert result1.shape == (2,)
    assert torch.equal(result1, result2)


def test_random_sample_with_idx(topk_mod):
    """When idx is provided, output is gathered from idx using sampled indices."""
    mod, _, _ = topk_mod
    probs = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.1, 0.6]], dtype=torch.float32)
    # Identity mapping → same result as without idx
    idx = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    torch.manual_seed(42)
    torch_npu.npu.manual_seed(42)
    r1 = mod.random_sample(probs.clone(), idx=None, generators={}, sampler=types.SimpleNamespace(bonus_q=None, dsa_stream=_DummyDefaultStream()))
    torch.manual_seed(42)
    torch_npu.npu.manual_seed(42)
    r2 = mod.random_sample(probs.clone(), idx=idx, generators={}, sampler=types.SimpleNamespace(bonus_q=None, dsa_stream=_DummyDefaultStream()))
    # With identity idx, results should match
    assert torch.equal(r1, r2)

    # Non-identity idx: output values are gathered from idx
    probs2 = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float32)
    idx2 = torch.tensor([[10, 20, 30]], dtype=torch.long)
    # Deterministic via seeding
    torch.manual_seed(123)
    result = mod.random_sample(probs2.clone(), idx=idx2, generators={}, sampler=types.SimpleNamespace(bonus_q=None, dsa_stream=_DummyDefaultStream()))
    assert result.shape == (1,)
    assert result.dtype == torch.long
    # The output should be one of the values from idx2
    assert result[0].item() in (10, 20, 30)

