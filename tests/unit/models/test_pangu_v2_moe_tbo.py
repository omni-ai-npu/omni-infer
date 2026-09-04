# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections import namedtuple
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit

_GatherRouting = namedtuple(
    "_GatherRouting",
    ("index", "order_sp", "data", "parse", "done", "scale"),
)


class _FakeEvent:
    def record(self, *_args, **_kwargs):
        return None

    def wait(self, *_args, **_kwargs):
        return None


def _npu_runtime():
    """CPU-safe torch.npu stream/event namespace."""
    return SimpleNamespace(
        current_stream=lambda: object(),
        Event=_FakeEvent,
        stream=lambda _stream: nullcontext(),
    )


def _stat(total, avg, ep=2):
    """Build a RoutingStat-like namespace for the TBO all_to_all phase."""
    avg_n = min(total, avg)
    rest_n = max(0, total - avg)
    avg_recvs = torch.zeros(ep, dtype=torch.int32)
    rest_recvs = torch.zeros(ep, dtype=torch.int32)
    avg_recvs[0] = avg_n
    rest_recvs[0] = rest_n
    token_map = torch.zeros(ep, ep, dtype=torch.int32)
    token_map[0, 0] = total
    zeros = torch.zeros(ep, dtype=torch.int32)
    return SimpleNamespace(
        hist1=torch.ones(2, dtype=torch.int32),
        hist2=torch.ones(2, dtype=torch.int32),
        avg_map=torch.ones(ep, ep, dtype=torch.int32),
        rest_map=torch.ones(ep, ep, dtype=torch.int32),
        map=token_map,
        avg_sends=zeros.clone(),
        avg_recvs=avg_recvs,
        rest_sends=zeros.clone(),
        rest_recvs=rest_recvs,
        full_sends=avg_recvs + rest_recvs,
        full_recvs=avg_recvs + rest_recvs,
    )


def _make_moe(tokens=4, top_k=2, hidden_size=4):
    """Bare OpenPanguV2MOE with TBO knobs and a stubbed shared expert."""
    moe = model_mod.OpenPanguV2MOE.__new__(model_mod.OpenPanguV2MOE)
    moe.n_routed_experts = 4
    moe.routed_scaling_factor = 1.0
    moe.e_score_correction_bias = None
    moe.moe_tbo_threshold = 4
    moe.moe_seq_split_length = 64
    moe._is_quant = True
    moe.gate = MagicMock(return_value=(torch.zeros(tokens, 4), None))
    moe.experts = SimpleNamespace(top_k=top_k, topk_group=1, num_expert_group=1)
    moe.shared_calls = []

    def shared_experts(hidden_states):
        moe.shared_calls.append(hidden_states)
        rows = hidden_states.size(0) if isinstance(hidden_states, torch.Tensor) else tokens
        return torch.zeros(rows, hidden_size, dtype=torch.bfloat16)

    moe.shared_experts = shared_experts
    return moe


def _stub_parse_input(moe, tokens, top_k):
    """Bypass gating so TBO tests only exercise the overlap pipeline."""

    def parse_input(hidden_states):
        if isinstance(hidden_states, dict):
            hidden_states = hidden_states["hidden_states_bf16"]
        weights = torch.ones(tokens, top_k, dtype=torch.float32)
        ids = torch.zeros(tokens, top_k, dtype=torch.int32)
        return hidden_states, weights, ids

    moe._parse_input = parse_input


def _enter_tbo_patches(stack, stat, index, order_sp, x_sc, tokens, hidden_size):
    """Patch TBO helpers so _forward_tbo can run on CPU."""
    sp = SimpleNamespace(
        world_size=2,
        all_gather=lambda tensor, dim=0: (
            tensor.repeat(2, 1) if tensor.dim() == 2 else tensor.repeat(2)
        ),
    )
    ep = SimpleNamespace(device_group="ep", world_size=2, rank_in_group=0)
    gather_out = _GatherRouting(
        index=index,
        order_sp=order_sp,
        data=torch.arange(8),
        parse=lambda _data: stat,
        done=_FakeEvent(),
        scale=x_sc,
    )

    def fake_gather(*_args, **_kwargs):
        return gather_out

    def fake_quant(_experts, x_i8, _sc, _hist):
        return torch.zeros(x_i8.size(0), hidden_size, dtype=torch.bfloat16)

    def fake_all_to_all(_group, tensor, _sends, _recvs, out=None):
        if out is None:
            return tensor
        return out

    def fake_reroute(value, _mat=None):
        if type(value) is int:
            return torch.arange(value)
        return value

    def fake_quant_op(tensor):
        scales = torch.ones(tensor.size(0), dtype=torch.float32)
        i8 = torch.zeros(tensor.size(0), tensor.size(1), dtype=torch.int8)
        return i8, scales

    patches = [
        patch.object(model_mod.torch, "npu", _npu_runtime()),
        patch.object(model_mod, "named_stream", return_value=object()),
        patch.object(model_mod, "record_event", side_effect=lambda: _FakeEvent()),
        patch.object(model_mod, "get_tp_group", return_value=sp),
        patch.object(model_mod, "get_ep_group", return_value=ep),
        patch.object(model_mod, "no_aiv", side_effect=lambda group: group),
        patch.object(model_mod, "gather_routing", side_effect=fake_gather),
        patch.object(model_mod, "quant_ffn", side_effect=fake_quant),
        patch.object(model_mod, "all_to_all", side_effect=fake_all_to_all),
        patch.object(model_mod, "rerouting", side_effect=fake_reroute),
        patch.object(
            model_mod, "finalize_routing",
            return_value=torch.zeros(tokens, hidden_size, dtype=torch.bfloat16),
        ),
        patch.object(
            model_mod.torch_npu, "npu_dynamic_quant", side_effect=fake_quant_op
        ),
    ]
    for item in patches:
        stack.enter_context(item)


def test_forward_routes_to_tbo_when_threshold_exceeded():
    """forward() uses TBO when token*TP is above moe_tbo_threshold."""
    moe = _make_moe()
    moe._forward_tbo = MagicMock(return_value=torch.ones(4, 4))
    hidden_states = torch.zeros(4, 4)
    with patch.object(
        model_mod, "get_tp_group", return_value=SimpleNamespace(world_size=2)
    ):
        out = moe.forward(hidden_states)
    moe._forward_tbo.assert_called_once_with(hidden_states)
    torch.testing.assert_close(out, torch.ones(4, 4))


def test_forward_does_not_use_tbo_when_threshold_disabled():
    """moe_tbo_threshold <= 0 keeps the single-batch path."""
    moe = _make_moe()
    moe.moe_tbo_threshold = -1
    moe._forward_single = MagicMock(return_value=torch.zeros(4, 4))
    hidden_states = torch.zeros(4, 4)
    with patch.object(
        model_mod, "get_tp_group", return_value=SimpleNamespace(world_size=2)
    ):
        out = moe.forward(hidden_states)
    moe._forward_single.assert_called_once_with(hidden_states)
    assert tuple(out.shape) == (4, 4)


def test_parse_input_accepts_tensor_and_dict():
    """_parse_input runs gating on fp32 activations from a tensor or a dict."""
    moe = _make_moe()
    tensor = torch.ones(4, 4, dtype=torch.bfloat16)
    with patch.object(
        model_mod.torch_npu, "npu_moe_gating_top_k",
        return_value=(
            torch.ones(4, 2), torch.zeros(4, 2, dtype=torch.int32), None,
        ),
    ):
        x, weights, ids = moe._parse_input(tensor)
        packed = moe._parse_input(
            {
                "hidden_states_fp32": tensor.float(),
                "hidden_states_bf16": tensor,
            }
        )
    assert x is tensor
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert packed[0] is tensor


def test_forward_tbo_avg_only_path():
    """When total tokens <= avg, TBO skips the rest FFN wave."""
    tokens, top_k, hidden_size = 4, 2, 4
    moe = _make_moe(tokens=tokens, top_k=top_k, hidden_size=hidden_size)
    _stub_parse_input(moe, tokens, top_k)
    avg = int(tokens * top_k * 1.1)
    total = avg
    stat = _stat(total, avg)
    index = torch.zeros(avg + 4, dtype=torch.int64)
    order_sp = torch.arange(tokens * top_k, dtype=torch.int32)
    x_sc = torch.ones(8, dtype=torch.float32)
    hidden_states = torch.ones(tokens, hidden_size, dtype=torch.bfloat16)
    with ExitStack() as stack:
        _enter_tbo_patches(stack, stat, index, order_sp, x_sc, tokens, hidden_size)
        out = moe._forward_tbo(hidden_states, quant_combine=True)
    assert tuple(out.shape) == (tokens, hidden_size)
    assert len(moe.shared_calls) == 1


def test_forward_tbo_rest_wave_and_dict_input():
    """When total > avg, TBO runs the rest FFN and accepts dict hidden states."""
    tokens, top_k, hidden_size = 4, 2, 4
    moe = _make_moe(tokens=tokens, top_k=top_k, hidden_size=hidden_size)
    _stub_parse_input(moe, tokens, top_k)
    avg = int(tokens * top_k * 1.1)
    total = avg + 3
    stat = _stat(total, avg)
    index = torch.zeros(total + 4, dtype=torch.int64)
    order_sp = torch.arange(tokens * top_k, dtype=torch.int32)
    x_sc = torch.ones(8, dtype=torch.float32)
    hidden_states = {
        "hidden_states_bf16": torch.ones(tokens, hidden_size, dtype=torch.bfloat16),
        "hidden_states_fp32": torch.ones(tokens, hidden_size, dtype=torch.float32),
    }
    with ExitStack() as stack:
        _enter_tbo_patches(stack, stat, index, order_sp, x_sc, tokens, hidden_size)
        out = moe._forward_tbo(hidden_states, quant_combine=False)
    assert tuple(out.shape) == (tokens, hidden_size)
    assert len(moe.shared_calls) == 1
