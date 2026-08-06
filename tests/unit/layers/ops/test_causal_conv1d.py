# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch

from omni.layers.ops import causal_conv1d as conv_mod


def test_npu_fused_causal_conv1d_squeezes_and_transposes_weight(monkeypatch):
    calls = {}

    def fake_op(**kwargs):
        calls.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(npu_ai_infra_fused_causal_conv1d=fake_op),
        raising=False,
    )
    x = torch.randn(2, 3)
    weight = torch.randn(4, 1, 3)
    conv_states = torch.randn(1, 3, 4)

    out = conv_mod.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=torch.tensor([0, 2]),
        cache_indices=torch.tensor([0]),
        num_accepted_tokens=torch.tensor([1]),
        activation="swish",
        max_query_len=2,
        inplace=True,
    )

    assert out is x
    assert calls["weight"].shape == (3, 4)
    assert calls["activation"] == "swish"
    assert calls["max_query_len"] == 2
    assert calls["conv_mode"] == 0
    assert calls["inplace"] is True


def test_causal_conv1d_ref_handles_legacy_layout_and_final_states():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    weight = torch.ones(2, 2, dtype=torch.float32)
    final_states = torch.empty(1, 2, 1)

    out, state = conv_mod.causal_conv1d_ref(
        x,
        weight,
        bias=None,
        initial_states=None,
        return_final_states=True,
        final_states_out=final_states,
        activation=None,
    )

    assert out.shape == (3, 2)
    assert state is final_states
    assert torch.equal(final_states, x.transpose(0, 1)[-1:].transpose(0, 1).unsqueeze(0))


def test_causal_conv1d_ref_uses_initial_states_and_rejects_activation():
    x = torch.randn(3, 2)
    weight = torch.ones(2, 2)
    initial_states = torch.randn(1, 1, 2)
    final_states = torch.empty(1, 1, 2)

    out, state = conv_mod.causal_conv1d_ref(
        x,
        weight,
        bias=torch.ones(2),
        initial_states=initial_states,
        return_final_states=True,
        final_states_out=final_states,
        activation="silu",
    )

    assert out.shape == (3, 2)
    assert state is final_states
    with pytest.raises(NotImplementedError):
        conv_mod.causal_conv1d_ref(x, weight, None, None, activation="relu")


def test_causal_conv1d_update_updates_state_and_uses_num_accepted(monkeypatch):
    def fake_scatter(target, indices, updates):
        target[indices.squeeze(1)] = updates

    monkeypatch.setattr(conv_mod.torch_npu, "npu_scatter_nd_update_", fake_scatter)
    x = torch.randn(2, 2)
    conv_state = torch.randn(2, 3, 2)
    weight = torch.ones(2, 3)

    out = conv_mod.causal_conv1d_update(
        x,
        conv_state,
        weight,
        activation=True,
        conv_state_indices=torch.tensor([0, 1]),
        num_accepted_tokens=torch.tensor([1, 2]),
    )

    assert out.shape == (2, 2)
    with pytest.raises(AssertionError):
        conv_mod.causal_conv1d_update(
            x,
            conv_state,
            weight,
            activation="relu",
            conv_state_indices=torch.tensor([0, 1]),
        )


def test_causal_conv1d_batch_helpers_skip_padding_slots():
    x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    weight = torch.ones(2, 2)
    conv_states = torch.randn(3, 2, 1)

    out_bubble = conv_mod.causal_conv1d_fn_bubble(
        x,
        weight,
        bias=None,
        conv_states=conv_states,
        query_start_loc=torch.tensor([0, 2, 5]),
        cache_indices=torch.tensor([0, conv_mod.PAD_SLOT_ID]),
        has_initial_state=torch.tensor([True, False]),
        activation=None,
    )
    out_fn = conv_mod.causal_conv1d_fn(
        x,
        weight,
        bias=None,
        conv_states=conv_states,
        seqlens_list=[2, 0, 3],
        cache_indices_list=[0, 1, conv_mod.PAD_SLOT_ID],
        has_initial_state_list=[True, False, False],
        activation=None,
    )

    assert out_bubble.shape == (2, 2)
    assert out_fn.shape == (2, 2)
