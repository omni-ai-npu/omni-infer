# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``OpenPanguV2MOE._forward_allgather`` on the int8 path.

Exercises gating -> dynamic quant -> init_routing_v2 -> GMM -> fused
GMM+FinalizeRouting, plus the all-reduce / shared-expert tail.

Every ``torch_npu`` operator is mocked. Following the project convention for
NPU kernels, the assertions pin the *call contract* (which branch ran, which
tensors and kwargs were handed to each operator, output shape/dtype) and never
operator numerics -- the operator package is replaceable.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit

TOKENS = 4
TOP_K = 2
HIDDEN = 4
EXPERTS = 4
INTERMEDIATE = 8
ROWS = TOKENS * TOP_K


def _forward_allgather(moe, *args, **kwargs):
    """Call the module-private forward under test."""
    return getattr(moe, "_forward_allgather")(*args, **kwargs)


def _make_quant_moe(gmm_fr_token_threshold):
    """Bare int8 OpenPanguV2MOE; `threshold` decides the GMM-FR branch."""
    moe = model_mod.OpenPanguV2MOE.__new__(model_mod.OpenPanguV2MOE)
    setattr(moe, "_is_quant", True)
    setattr(moe, "_is_w4a8", False)
    moe.side_stream = None
    moe.use_moe_force_load_balance = False
    moe.routed_scaling_factor = 1.0
    moe.e_score_correction_bias = None
    moe.n_routed_experts = EXPERTS
    moe.n_physical_experts = EXPERTS
    moe.physical_expert_start = 0
    moe.physical_expert_end = EXPERTS
    moe.gmm_fr_token_threshold = gmm_fr_token_threshold
    moe.gate = MagicMock(return_value=(torch.zeros(TOKENS, EXPERTS), None))
    moe.experts = SimpleNamespace(
        top_k=TOP_K,
        topk_group=1,
        num_expert_group=1,
        w13_weight=torch.zeros(EXPERTS, HIDDEN, 2 * INTERMEDIATE, dtype=torch.int8),
        w13_weight_scale=torch.ones(EXPERTS, 2 * INTERMEDIATE, dtype=torch.float32),
        w2_weight=torch.zeros(EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.int8),
        w2_weight_scale=torch.ones(EXPERTS, HIDDEN, dtype=torch.float32),
    )
    moe.shared_calls = []

    def shared_experts(hidden_states):
        moe.shared_calls.append(hidden_states)
        return torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16)

    moe.shared_experts = shared_experts
    return moe


def _patch_quant_ops(monkeypatch, calls, *, expert_tokens=None):
    """Mock the int8 operator chain with shape-correct outputs."""
    # `_is_graph_mode` is a read-only property driven by the forward context;
    # eager mode keeps the npugraph scopes out of the way.
    monkeypatch.setattr(
        model_mod.OpenPanguV2MOE, "_is_graph_mode", property(lambda self: False)
    )
    expert_tokens = (
        expert_tokens
        if expert_tokens is not None
        else torch.full((EXPERTS,), ROWS // EXPERTS, dtype=torch.int64)
    )

    def gating_top_k(router_logits, **kwargs):
        calls["gating"] = (router_logits, kwargs)
        return (
            torch.full((TOKENS, TOP_K), 0.5, dtype=torch.float32),
            torch.zeros(TOKENS, TOP_K, dtype=torch.int32),
            None,
        )

    def dynamic_quant(hidden_states):
        calls["dynamic_quant"] = hidden_states
        return (
            torch.zeros(TOKENS, HIDDEN, dtype=torch.int8),
            torch.ones(TOKENS, dtype=torch.float32),
        )

    def init_routing_v2(hidden_states_int8, topk_ids, **kwargs):
        calls["init_routing"] = (hidden_states_int8, topk_ids, kwargs)
        return (
            torch.zeros(ROWS, HIDDEN, dtype=torch.int8),
            torch.arange(ROWS, dtype=torch.int32),
            expert_tokens,
            torch.ones(ROWS, dtype=torch.float32),
        )

    def grouped_matmul(x, weight, **kwargs):
        calls.setdefault("grouped_matmul", []).append((x, weight, kwargs))
        return [torch.zeros(ROWS, 2 * INTERMEDIATE, dtype=torch.int32)]

    def dequant_swiglu_quant(**kwargs):
        calls["swiglu"] = kwargs
        return (
            torch.zeros(ROWS, INTERMEDIATE, dtype=torch.int8),
            torch.ones(ROWS, dtype=torch.float32),
        )

    def gmm_finalize_routing(intermediate_h, w2_weight, **kwargs):
        calls["gmm_fr"] = (intermediate_h, w2_weight, kwargs)
        return torch.zeros(TOKENS, HIDDEN, dtype=torch.float32)

    def moe_finalize_routing(down_proj_output, *args, **kwargs):
        calls["finalize_routing"] = (down_proj_output, args, kwargs)
        return torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16)

    for name, fn in (
        ("npu_moe_gating_top_k", gating_top_k),
        ("npu_dynamic_quant", dynamic_quant),
        ("npu_moe_init_routing_v2", init_routing_v2),
        ("npu_grouped_matmul", grouped_matmul),
        ("npu_dequant_swiglu_quant", dequant_swiglu_quant),
        ("npu_grouped_matmul_finalize_routing", gmm_finalize_routing),
        ("npu_moe_finalize_routing", moe_finalize_routing),
    ):
        monkeypatch.setattr(model_mod.torch_npu, name, fn, raising=False)

    monkeypatch.setattr(
        model_mod,
        "get_ep_group",
        lambda: SimpleNamespace(
            all_reduce=lambda t: t,
            reduce_scatter=lambda t, dim=0: t,
            all_gather=lambda t, dim=0: t,
        ),
        raising=False,
    )
    monkeypatch.setattr(model_mod, "nullcontext", nullcontext, raising=False)


def test_forward_allgather_quant_uses_fused_gmm_finalize_routing(monkeypatch):
    """Small batches take the fused GMM2+FinalizeRouting branch."""
    calls = {}
    # A threshold at least as large as the batch enables the fused branch.
    moe = _make_quant_moe(gmm_fr_token_threshold=TOKENS)
    _patch_quant_ops(monkeypatch, calls)

    hidden_states = torch.ones(TOKENS, HIDDEN, dtype=torch.bfloat16)
    out = _forward_allgather(moe, hidden_states, use_allreduce=True)

    # Gating ran on float32 logits derived from the bf16 hidden states.
    gating_logits, gating_kwargs = calls.get("gating")
    assert gating_logits.dtype == torch.float32
    assert gating_kwargs.get("k") == TOP_K
    # init_routing gets the quantized activations and row_idx_type=1 for GMM-FR.
    routing_x, routing_ids, routing_kwargs = calls.get("init_routing")
    assert routing_x.dtype == torch.int8
    assert routing_kwargs.get("row_idx_type") == 1
    assert routing_kwargs.get("expert_num") == EXPERTS
    assert routing_kwargs.get("active_expert_range") == [0, EXPERTS]
    # Only the gate_up GMM runs; the down-proj GMM is folded into the fused op.
    assert len(calls.get("grouped_matmul")) == 1
    assert "finalize_routing" not in calls
    # The fused op receives a zero shared_input placeholder shaped like the input.
    fr_kwargs = calls.get("gmm_fr")[2]
    shared_input_fake = fr_kwargs.get("shared_input")
    assert shared_input_fake.shape == (TOKENS, HIDDEN)
    assert shared_input_fake.dtype == torch.bfloat16
    assert torch.count_nonzero(shared_input_fake) == 0
    assert fr_kwargs.get("output_bs") == TOKENS
    assert fr_kwargs.get("group_list_type") == 1
    # row_index maps each expanded row back to its source token.
    assert fr_kwargs.get("row_index").dtype == torch.int64
    assert fr_kwargs.get("logit").dtype == torch.float32
    # Shared experts consume the original (pre-quant) hidden states.
    assert moe.shared_calls == [hidden_states]
    assert out.shape == (TOKENS, HIDDEN)
    assert out.dtype == torch.bfloat16


def test_forward_allgather_quant_uses_separate_down_proj_when_batch_large(monkeypatch):
    """Above the threshold the down-proj GMM and finalize_routing stay separate."""
    calls = {}
    # A threshold below the batch size disables the fused branch.
    moe = _make_quant_moe(gmm_fr_token_threshold=TOKENS - 1)
    _patch_quant_ops(monkeypatch, calls)

    out = _forward_allgather(
        moe, torch.ones(TOKENS, HIDDEN, dtype=torch.bfloat16), use_allreduce=True
    )

    assert calls.get("init_routing")[2].get("row_idx_type") == 0
    # Both the gate_up and the down-proj matmuls run as separate calls.
    assert len(calls.get("grouped_matmul")) == 2
    assert "gmm_fr" not in calls
    assert calls.get("finalize_routing")[2].get("drop_pad_mode") == 3
    assert out.shape == (TOKENS, HIDDEN)


def test_forward_allgather_accepts_dict_input(monkeypatch):
    """A {"hidden_states_bf16", "hidden_states_fp32"} dict feeds the fp32 gate."""
    calls = {}
    moe = _make_quant_moe(gmm_fr_token_threshold=TOKENS)
    _patch_quant_ops(monkeypatch, calls)

    hidden_bf16 = torch.ones(TOKENS, HIDDEN, dtype=torch.bfloat16)
    hidden_fp32 = torch.full((TOKENS, HIDDEN), 2.0, dtype=torch.float32)

    _forward_allgather(
        moe,
        {"hidden_states_bf16": hidden_bf16, "hidden_states_fp32": hidden_fp32},
        use_allreduce=True,
    )

    # The gate is called with the provided fp32 tensor, not a cast of the bf16 one.
    assert moe.gate.call_args[0][0] is hidden_fp32
    # Downstream still uses the bf16 tensor.
    assert calls.get("dynamic_quant") is hidden_bf16
    assert moe.shared_calls == [hidden_bf16]
