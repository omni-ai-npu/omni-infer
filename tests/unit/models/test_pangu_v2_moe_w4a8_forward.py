# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Execute OpenPanguV2MOE W4A8 compute branches with mocked NPU ops.

The flag-only tests in ``test_pangu_v2_moe_w4a8_flags.py`` do not run the
production GMM / swiglu / finalize lines. These tests drive the three
communication paths that contain the incremental W4A8 code:

* ``_forward_allgather`` (W4A8 GMM2 / GMM-FR, and the W8A8 else branch)
* ``_dispatch_combine_single_batch``
* ``_forward_all2allv``
"""
from contextlib import ExitStack
from types import SimpleNamespace
from typing import NamedTuple, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit

TOKENS = 2
HIDDEN = 4
INTERMEDIATE = 4
TOP_K = 2
NUM_EXPERTS = 2


class GatingResult(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    extra: Optional[object]


class InitRoutingResult(NamedTuple):
    sorted_tokens: torch.Tensor
    expanded_idx: torch.Tensor
    expert_tokens: torch.Tensor
    scale: torch.Tensor


class DispatchResult(NamedTuple):
    expand_x: torch.Tensor
    dynamic_scale: torch.Tensor
    expand_idx: torch.Tensor
    expert_token_nums: torch.Tensor
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor


class ReRoutingResult(NamedTuple):
    hidden_states_sorted: torch.Tensor
    pertoken_scale: torch.Tensor
    idxs_unsort: torch.Tensor
    tokens_per_expert: torch.Tensor


def _w8a8_experts():
    """W8A8 experts: int8 weights + channel scales, no int4_scale."""
    return SimpleNamespace(
        top_k=TOP_K,
        topk_group=1,
        num_expert_group=1,
        w13_weight=torch.zeros(NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE, dtype=torch.int8),
        w13_weight_scale=torch.ones(NUM_EXPERTS, 2 * INTERMEDIATE),
        w2_weight=torch.zeros(NUM_EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.int8),
        w2_weight_scale=torch.ones(NUM_EXPERTS, HIDDEN),
    )


def _w4a8_experts(*, asymmetric=False):
    experts = SimpleNamespace(
        top_k=TOP_K,
        topk_group=1,
        num_expert_group=1,
        w13_weight=torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.int8),
        w13_weight_bias=torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE),
        w13_weight_int4_scale=torch.ones(NUM_EXPERTS, 1, 2 * INTERMEDIATE, dtype=torch.int64),
        w2_weight=torch.zeros(NUM_EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.int8),
        w2_weight_bias=torch.zeros(NUM_EXPERTS, HIDDEN),
        w2_weight_int4_scale=torch.ones(NUM_EXPERTS, 1, HIDDEN, dtype=torch.int64),
    )
    if asymmetric:
        experts.w13_weight_offset = torch.ones(NUM_EXPERTS, 1, 2 * INTERMEDIATE)
        experts.w2_weight_offset = torch.ones(NUM_EXPERTS, 1, HIDDEN)
    else:
        experts.w13_weight_offset = None
        experts.w2_weight_offset = None
    return experts


def _make_moe(*, asymmetric=False, gmm_fr_threshold=0, w4a8=True):
    moe = model_mod.OpenPanguV2MOE.__new__(model_mod.OpenPanguV2MOE)
    # _is_graph_mode is a read-only property over get_forward_context().
    moe._is_quant = True
    moe._is_w4a8 = w4a8
    moe._is_w4a8_weight_asymmetric = asymmetric if w4a8 else False
    moe.experts = (
        _w4a8_experts(asymmetric=asymmetric) if w4a8 else _w8a8_experts()
    )
    moe.n_routed_experts = NUM_EXPERTS
    moe.n_physical_experts = NUM_EXPERTS
    moe.n_local_physical_experts = NUM_EXPERTS
    moe.physical_expert_start = 0
    moe.physical_expert_end = NUM_EXPERTS
    moe.ep_size = 1
    moe.ep_rank = 0
    moe.ep_group = "ep"
    moe.routed_scaling_factor = 1.0
    moe.e_score_correction_bias = None
    moe.use_moe_force_load_balance = False
    moe.enable_eplb = False
    moe.side_stream = None
    moe.fetch_stream = None
    moe.gmm_fr_token_threshold = gmm_fr_threshold
    moe.layer_idx = 0
    hidden = torch.ones(TOKENS, HIDDEN, dtype=torch.bfloat16)
    moe.gate = MagicMock(
        return_value=(torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.float32), None)
    )
    moe.shared_experts = MagicMock(
        return_value=torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16)
    )
    moe._hidden = hidden
    return moe


def _identity_ep():
    return SimpleNamespace(
        rank=0,
        rank_in_group=0,
        world_size=1,
        all_gather=lambda tensor, dim=0: tensor,
        all_reduce=lambda tensor: tensor,
        reduce_scatter=lambda tensor, dim=0: tensor,
    )


def _gating_result():
    return GatingResult(
        topk_weights=torch.ones(TOKENS, TOP_K, dtype=torch.float32),
        topk_ids=torch.zeros(TOKENS, TOP_K, dtype=torch.int32),
        extra=None,
    )


def _init_routing_result(*, tokens=TOKENS, hidden=HIDDEN):
    return InitRoutingResult(
        sorted_tokens=torch.ones(tokens * TOP_K, hidden, dtype=torch.int8),
        expanded_idx=torch.arange(tokens * TOP_K, dtype=torch.int32),
        expert_tokens=torch.tensor([tokens, tokens], dtype=torch.int64),
        scale=torch.ones(tokens * TOP_K, dtype=torch.float32),
    )


def _patch_common_npu(stack, *, gmm_out_hidden=HIDDEN, gmm_out_inter=2 * INTERMEDIATE):
    def fake_gmm(inputs, weights, **kwargs):
        tokens = inputs[0]
        out_dim = gmm_out_inter if tokens.shape[-1] == HIDDEN else gmm_out_hidden
        return [torch.ones(tokens.shape[0], out_dim, dtype=kwargs.get("output_dtype", torch.bfloat16))]

    def fake_swiglu(*args, **kwargs):
        x = args[0] if args else kwargs["x"]
        return (
            torch.ones(x.shape[0], INTERMEDIATE, dtype=torch.int8),
            torch.ones(x.shape[0], dtype=torch.float32),
        )

    stack.enter_context(
        patch.object(
            model_mod.torch_npu,
            "npu_moe_gating_top_k",
            return_value=_gating_result(),
        )
    )
    stack.enter_context(
        patch.object(
            model_mod.torch_npu,
            "npu_dynamic_quant",
            side_effect=lambda x: (
                torch.zeros(x.shape[0], x.shape[-1], dtype=torch.int8),
                torch.ones(x.shape[0], dtype=torch.float32),
            ),
        )
    )
    stack.enter_context(
        patch.object(model_mod.torch_npu, "npu_grouped_matmul", side_effect=fake_gmm)
    )
    stack.enter_context(
        patch.object(
            model_mod.torch_npu, "npu_dequant_swiglu_quant", side_effect=fake_swiglu
        )
    )
    stack.enter_context(patch.object(model_mod, "get_ep_group", side_effect=_identity_ep))
    stack.enter_context(
        patch.object(
            model_mod,
            "get_forward_context",
            return_value=SimpleNamespace(
                cudagraph_runtime_mode=model_mod.CUDAGraphMode.NONE,
            ),
        )
    )
    stack.enter_context(
        patch.object(
            model_mod,
            "current_platform",
            SimpleNamespace(device_type="cpu"),
        )
    )
    return fake_gmm


def _capturing_gmm(captured):
    def _gmm(inputs, weights, **kwargs):
        captured.append(kwargs)
        tokens = inputs[0]
        out_dim = 2 * INTERMEDIATE if tokens.shape[-1] == HIDDEN else HIDDEN
        dtype = kwargs.get("output_dtype", torch.bfloat16)
        return [torch.ones(tokens.shape[0], out_dim, dtype=dtype)]

    return _gmm


def _fake_all_to_all_single(output, inp, *args, **kwargs):
    if output.shape == inp.shape:
        output.copy_(inp)
    elif output.numel() == inp.numel():
        output.copy_(inp.reshape(output.shape))
    else:
        output.zero_()


def _dispatch_result():
    """Build the 6-tuple returned by ``npu_moe_distribute_dispatch_v2``."""
    return DispatchResult(
        expand_x=torch.ones(4, HIDDEN, dtype=torch.int8),
        dynamic_scale=torch.ones(4, dtype=torch.float32),
        expand_idx=torch.arange(4, dtype=torch.int32),
        expert_token_nums=torch.tensor([2, 2], dtype=torch.int32),
        ep_recv_counts=torch.ones(1, dtype=torch.int32),
        tp_recv_counts=torch.ones(1, dtype=torch.int32),
    )


def _resorted_result(routing):
    """Build the re-routing tuple from an init-routing result."""
    sorted_tokens = routing[0]
    return ReRoutingResult(
        hidden_states_sorted=sorted_tokens.to(torch.int8),
        pertoken_scale=torch.ones(sorted_tokens.shape[0], dtype=torch.float32),
        idxs_unsort=torch.arange(sorted_tokens.shape[0], dtype=torch.int32),
        tokens_per_expert=torch.tensor([2, 2], dtype=torch.int64),
    )


def _patch_init_routing(stack, routing=None):
    stack.enter_context(
        patch.object(
            model_mod.torch_npu,
            "npu_moe_init_routing_v2",
            return_value=routing if routing is not None else _init_routing_result(),
        )
    )


def _patch_finalize_routing(stack):
    stack.enter_context(
        patch.object(
            model_mod.torch_npu,
            "npu_moe_finalize_routing",
            return_value=torch.ones(TOKENS, HIDDEN, dtype=torch.bfloat16),
        )
    )


def _patch_grouped_matmul(stack, captured):
    stack.enter_context(
        patch.object(
            model_mod.torch_npu, "npu_grouped_matmul", side_effect=_capturing_gmm(captured)
        )
    )


def _patch_grouped_matmul_finalize_routing(stack, fr_mock):
    stack.enter_context(
        patch.object(
            model_mod.torch_npu,
            "npu_grouped_matmul_finalize_routing",
            fr_mock,
        )
    )


def _patch_re_routing(stack, resorted):
    stack.enter_context(
        patch.object(
            model_mod.torch_npu, "npu_moe_re_routing", return_value=resorted
        )
    )


def _patch_all_to_all_single(stack):
    stack.enter_context(
        patch.object(model_mod.dist, "all_to_all_single", side_effect=_fake_all_to_all_single)
    )


def _patch_distribute_dispatch(stack, dispatch=None):
    stack.enter_context(
        patch.object(
            model_mod.torch_npu,
            "npu_moe_distribute_dispatch_v2",
            return_value=dispatch if dispatch is not None else _dispatch_result(),
        )
    )


def _patch_distribute_combine(stack, combine_out=None):
    stack.enter_context(
        patch.object(
            model_mod.torch_npu,
            "npu_moe_distribute_combine_v2",
            return_value=combine_out if combine_out is not None
            else torch.ones(TOKENS, HIDDEN, dtype=torch.bfloat16),
        )
    )


def _fr_mock():
    """Build the grouped_matmul_finalize_routing return-value mock."""
    return MagicMock(return_value=torch.ones(TOKENS, HIDDEN, dtype=torch.float32))


def _run_forward_allgather(moe, captured):
    """Patch common NPU + routing and run ``_forward_allgather``."""
    with ExitStack() as stack:
        _patch_common_npu(stack)
        _patch_grouped_matmul(stack, captured)
        _patch_init_routing(stack)
        _patch_finalize_routing(stack)
        return moe._forward_allgather(moe._hidden, use_allreduce=True)


def _run_forward_all2allv(moe, captured, routing, resorted):
    """Patch common NPU + routing/re-routing and run ``_forward_all2allv``."""
    with ExitStack() as stack:
        _patch_common_npu(stack)
        _patch_grouped_matmul(stack, captured)
        _patch_init_routing(stack, routing)
        _patch_re_routing(stack, resorted)
        _patch_all_to_all_single(stack)
        _patch_finalize_routing(stack)
        return moe._forward_all2allv(moe._hidden)


def _dispatch_combine_call(moe):
    """Call ``_dispatch_combine_single_batch`` with the standard test args."""
    return moe._dispatch_combine_single_batch(
        hidden_states=moe._hidden,
        topk_weights=torch.ones(TOKENS, TOP_K),
        topk_ids=torch.zeros(TOKENS, TOP_K, dtype=torch.int32),
        quant_mode=-1,
        ep_comm_name="ep",
    )


def _run_forward_allgather_fr(moe, gmm_calls, fr_mock):
    """Patch common NPU + routing + GMM-FR and run ``_forward_allgather``."""
    with ExitStack() as stack:
        _patch_common_npu(stack)
        _patch_grouped_matmul(stack, gmm_calls)
        _patch_init_routing(stack)
        _patch_grouped_matmul_finalize_routing(stack, fr_mock)
        return moe._forward_allgather(moe._hidden, use_allreduce=True)


def _run_dispatch_combine(moe, captured, dispatch, combine_out=None):
    """Patch common NPU + dispatch/combine and run ``_dispatch_combine_single_batch``."""
    with ExitStack() as stack:
        _patch_common_npu(stack)
        _patch_grouped_matmul(stack, captured)
        _patch_distribute_dispatch(stack, dispatch)
        _patch_distribute_combine(stack, combine_out)
        return _dispatch_combine_call(moe)


def test_forward_allgather_w4a8_symmetric_uses_no_offset():
    """Symmetric W4A8 allgather path calls GMM with offset=None."""
    moe = _make_moe(asymmetric=False, gmm_fr_threshold=0)
    captured = []
    out = _run_forward_allgather(moe, captured)

    assert out.shape == (TOKENS, HIDDEN)
    assert len(captured) == 2
    assert captured[0]["offset"] is None
    assert captured[1]["offset"] is None
    assert captured[0]["scale"][0] is moe.experts.w13_weight_int4_scale
    assert captured[1]["scale"][0] is moe.experts.w2_weight_int4_scale
    moe.shared_experts.assert_called_once()


def test_forward_allgather_w4a8_asymmetric_passes_offset_and_token_scale():
    """Asymmetric W4A8 allgather path wires offset tensors and unsqueezed scales."""
    moe = _make_moe(asymmetric=True, gmm_fr_threshold=0)
    captured = []
    out = _run_forward_allgather(moe, captured)

    assert out.shape == (TOKENS, HIDDEN)
    assert captured[0]["offset"][0] is moe.experts.w13_weight_offset
    assert captured[1]["offset"][0] is moe.experts.w2_weight_offset
    assert captured[0]["per_token_scale"][0].ndim == 2
    assert captured[1]["per_token_scale"][0].ndim == 2


def test_forward_allgather_w4a8_gmm_fr_path():
    """Small-batch W4A8 allgather uses grouped_matmul_finalize_routing for GMM2."""
    moe = _make_moe(asymmetric=False, gmm_fr_threshold=1024)
    fr_mock = _fr_mock()
    gmm_calls = []
    out = _run_forward_allgather_fr(moe, gmm_calls, fr_mock)

    assert out.shape == (TOKENS, HIDDEN)
    assert len(gmm_calls) == 1
    fr_mock.assert_called_once()
    assert fr_mock.call_args.kwargs["offset"] is None
    assert fr_mock.call_args.kwargs["scale"] is moe.experts.w2_weight_int4_scale


def test_forward_allgather_w4a8_gmm_fr_asymmetric_offset():
    """GMM-FR W4A8 path forwards the w2 offset when weights are asymmetric."""
    moe = _make_moe(asymmetric=True, gmm_fr_threshold=1024)
    fr_mock = _fr_mock()
    with ExitStack() as stack:
        _patch_common_npu(stack)
        _patch_init_routing(stack)
        _patch_grouped_matmul_finalize_routing(stack, fr_mock)
        moe._forward_allgather(moe._hidden, use_allreduce=True)

    assert fr_mock.call_args.kwargs["offset"] is moe.experts.w2_weight_offset


def test_dispatch_combine_w4a8_symmetric_and_asymmetric():
    """Dispatch/combine W4A8 path covers both offset=None and offset tensors."""
    dispatch = _dispatch_result()
    combine_out = torch.ones(TOKENS, HIDDEN, dtype=torch.bfloat16)

    for asymmetric in (False, True):
        moe = _make_moe(asymmetric=asymmetric)
        captured = []
        out = _run_dispatch_combine(moe, captured, dispatch, combine_out)

        assert out.shape == (TOKENS, HIDDEN)
        assert len(captured) == 2
        if asymmetric:
            assert captured[0]["offset"][0] is moe.experts.w13_weight_offset
            assert captured[1]["offset"][0] is moe.experts.w2_weight_offset
        else:
            assert captured[0]["offset"] is None
            assert captured[1]["offset"] is None


def test_forward_all2allv_w4a8_symmetric_path():
    """All-to-all W4A8 path runs GMM1/swiglu/GMM2 and finalize_routing."""
    moe = _make_moe(asymmetric=False)
    captured = []

    routing = _init_routing_result()
    resorted = _resorted_result(routing)
    out = _run_forward_all2allv(moe, captured, routing, resorted)

    assert out.shape == (TOKENS, HIDDEN)
    assert len(captured) == 2
    assert captured[0]["offset"] is None
    assert captured[1]["offset"] is None
    moe.shared_experts.assert_called_once()


def test_forward_all2allv_w4a8_asymmetric_path():
    """All-to-all asymmetric W4A8 path forwards both expert offsets."""
    moe = _make_moe(asymmetric=True)
    captured = []

    routing = _init_routing_result()
    resorted = _resorted_result(routing)
    out = _run_forward_all2allv(moe, captured, routing, resorted)

    assert out.shape == (TOKENS, HIDDEN)
    assert captured[0]["offset"][0] is moe.experts.w13_weight_offset
    assert captured[1]["offset"][0] is moe.experts.w2_weight_offset


def test_forward_allgather_accepts_dict_hidden_states():
    """W4A8 allgather reads bf16/fp32 activations from a dict input."""
    moe = _make_moe(asymmetric=False, gmm_fr_threshold=0)
    packed = {
        "hidden_states_bf16": moe._hidden,
        "hidden_states_fp32": moe._hidden.float(),
    }
    with ExitStack() as stack:
        _patch_common_npu(stack)
        _patch_init_routing(stack)
        _patch_finalize_routing(stack)
        out = moe._forward_allgather(packed, use_allreduce=True)

    assert out.shape == (TOKENS, HIDDEN)
    moe.gate.assert_called_once()
    assert moe.gate.call_args.args[0].dtype == torch.float32


def test_forward_allgather_w8a8_gmm_and_finalize():
    """Quantized non-W4A8 allgather hits the W8A8 GMM1 / swiglu / GMM2 / finalize else."""
    moe = _make_moe(w4a8=False, gmm_fr_threshold=0)
    captured = []
    out = _run_forward_allgather(moe, captured)

    assert out.shape == (TOKENS, HIDDEN)
    assert len(captured) == 2
    assert captured[0]["output_dtype"] is torch.int32
    assert captured[1]["output_dtype"] is torch.bfloat16
    assert torch.equal(captured[1]["scale"][0], moe.experts.w2_weight_scale.to(torch.bfloat16))
    moe.shared_experts.assert_called_once()


def test_forward_allgather_w8a8_gmm_fr_path():
    """Small-batch W8A8 allgather uses grouped_matmul_finalize_routing in the else branch."""
    moe = _make_moe(w4a8=False, gmm_fr_threshold=1024)
    fr_mock = _fr_mock()
    gmm_calls = []
    out = _run_forward_allgather_fr(moe, gmm_calls, fr_mock)

    assert out.shape == (TOKENS, HIDDEN)
    assert len(gmm_calls) == 1
    fr_mock.assert_called_once()
    assert fr_mock.call_args.kwargs["scale"] is moe.experts.w2_weight_scale


def test_dispatch_combine_w8a8_path():
    """Dispatch/combine W8A8 else branch uses weight_scale and int32 GMM1."""
    moe = _make_moe(w4a8=False)
    captured = []
    dispatch = _dispatch_result()
    out = _run_dispatch_combine(moe, captured, dispatch)

    assert out.shape == (TOKENS, HIDDEN)
    assert len(captured) == 2
    assert captured[0]["output_dtype"] is torch.int32
    assert captured[0]["scale"] is None
    assert torch.equal(captured[1]["scale"][0], moe.experts.w2_weight_scale.to(torch.bfloat16))


def test_forward_all2allv_w8a8_path():
    """All-to-all W8A8 else branch runs int32 GMM1, dequant-swiglu and scaled GMM2."""
    moe = _make_moe(w4a8=False)
    captured = []
    routing = _init_routing_result()
    resorted = _resorted_result(routing)
    out = _run_forward_all2allv(moe, captured, routing, resorted)

    assert out.shape == (TOKENS, HIDDEN)
    assert len(captured) == 2
    assert captured[0]["output_dtype"] is torch.int32
    assert captured[0]["scale"] is None
    assert torch.equal(captured[1]["scale"][0], moe.experts.w2_weight_scale.to(torch.bfloat16))
    moe.shared_experts.assert_called_once()
