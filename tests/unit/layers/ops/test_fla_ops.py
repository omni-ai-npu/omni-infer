# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import torch

from omni.layers.ops import fla_ops


def test_fused_recurrent_gated_delta_rule_fwd_updates_indexed_states(monkeypatch):
    def fake_scatter(target, indices, updates):
        target[indices.squeeze(1)] = updates

    monkeypatch.setattr(fla_ops.torch_npu, "npu_scatter_nd_update_", fake_scatter)
    q = torch.ones(1, 2, 1, 2)
    k = torch.ones(1, 2, 1, 2)
    v = torch.ones(1, 2, 1, 2)
    g = torch.zeros(1, 2, 1)
    beta = torch.ones(1, 2, 1)
    initial_state = torch.zeros(2, 1, 2, 2, dtype=torch.bfloat16)

    out, state = fla_ops.fused_recurrent_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=None,
        initial_state=initial_state,
        cu_seqlens=torch.tensor([0, 1, 2]),
        ssm_state_indices=torch.tensor([1, 0]),
        num_accepted_tokens=torch.tensor([1, 1]),
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == (1, 2, 1, 2)
    assert state.shape == initial_state.shape


def test_fused_recurrent_gated_delta_rule_npu_passes_contiguous_inputs(monkeypatch):
    calls = {}

    def fake_rule(**kwargs):
        calls.update(kwargs)
        return torch.ones(2, 1, 2, dtype=torch.bfloat16)

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(npu_ai_infra_recurrent_gated_delta_rule=fake_rule),
        raising=False,
    )
    q = torch.randn(1, 2, 1, 2)
    k = torch.randn(1, 2, 1, 2)
    v = torch.randn(1, 2, 1, 2)
    g = torch.randn(1, 2, 1)
    beta = torch.randn(1, 2, 1)
    state = torch.randn(2, 1, 2, 2)

    out, final_state = fla_ops._fused_recurrent_gated_delta_rule_npu(
        q,
        k,
        v,
        g,
        beta,
        scale=0.5,
        initial_state=state,
        actual_seqlens=torch.tensor([1, 2]),
        ssm_state_indices=torch.tensor([[0], [1]]),
        num_accepted_tokens=torch.tensor([1, 2]),
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == (2, 1, 2)
    assert final_state is state
    assert calls["query"].shape == (2, 1, 2)
    assert calls["actual_seq_lengths"].dtype == torch.int32
    assert calls["ssm_state_indices"].dtype == torch.int32


def test_fused_recurrent_gated_delta_rule_defaults_beta_and_scale(monkeypatch):
    calls = {}

    def fake_fused(**kwargs):
        calls.update(kwargs)
        return "out", "state"

    monkeypatch.setattr(fla_ops, "_fused_recurrent_gated_delta_rule_npu", fake_fused)
    q = torch.randn(1, 2, 1, 2)
    k = torch.randn(1, 2, 1, 2)
    v = torch.randn(1, 2, 1, 2)
    g = torch.randn(1, 2, 1)

    out, state = fla_ops.fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta=None,
        initial_state=torch.randn(2, 1, 2, 2),
        actual_seqlens=torch.tensor([1, 2]),
        ssm_state_indices=torch.tensor([[0], [1]]),
    )

    assert (out, state) == ("out", "state")
    assert calls["scale"] == k.shape[-1] ** -0.5
    assert torch.equal(calls["beta"], torch.ones_like(q[..., 0]))


def test_chunk_gated_delta_rule_npu_uses_custom_recurrence(monkeypatch):
    def fake_inverse(attn):
        return torch.linalg.inv(attn)

    def fake_recurrence(state, kgexp, value, k_cumdecay, qgexp, gexp, seqlen):
        return torch.zeros_like(value), torch.zeros_like(value)

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(
            npu_lower_triangular_inverse=fake_inverse,
            npu_chunk_gated_delta_rule_recurrence=fake_recurrence,
        ),
        raising=False,
    )
    query = torch.randn(1, 2, 1, 2)
    key = torch.randn(1, 2, 1, 2)
    value = torch.randn(1, 2, 1, 2)
    g = torch.zeros(1, 2, 1)
    beta = torch.ones(1, 2, 1)

    out, state = fla_ops.chunk_gated_delta_rule_npu(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=2,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        initial_dtype=torch.float32,
    )

    assert out.shape == value.shape
    assert state is None


def test_chunk_gated_delta_rule_calls_npu_per_batch(monkeypatch):
    calls = []

    def fake_chunk_gdn(**kwargs):
        calls.append(kwargs)
        query = kwargs["query"]
        state = kwargs["initial_state"]
        return torch.ones_like(query), torch.ones_like(state)

    monkeypatch.setattr(fla_ops, "chunk_gated_delta_rule_npu", fake_chunk_gdn)
    q = torch.randn(1, 3, 1, 2)
    k = torch.randn(1, 3, 1, 2)
    v = torch.randn(1, 3, 1, 2)
    g = torch.zeros(1, 3, 1)
    beta = torch.ones(1, 3, 1)
    initial_state = torch.zeros(2, 1, 2, 2)

    out, states = fla_ops.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 1, 3]),
        cu_seqlens_cpu=[0, 1, 3],
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == q.shape
    assert states.shape == initial_state.shape
    assert len(calls) == 2


def test_chunk_gated_delta_rule_ref_runs_cpu_reference_path():
    q = torch.randn(1, 1, 1, 2)
    k = torch.randn(1, 1, 1, 2)
    v = torch.randn(1, 1, 1, 2)
    g = torch.zeros(1, 1, 1)
    beta = torch.ones(1, 1, 1)
    initial_state = torch.zeros(1, 1, 2, 2)

    out, states = fla_ops.chunk_gated_delta_rule_ref(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 1]),
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == q.shape
    assert states.shape == initial_state.shape
