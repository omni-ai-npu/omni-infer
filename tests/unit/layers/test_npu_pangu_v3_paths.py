# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from omni_npu.v1.layers.attention import npu_pangu as mod


@pytest.fixture
def case(monkeypatch):
    attention = mod.NPUPanguSparseAttention.__new__(mod.NPUPanguSparseAttention)
    torch.nn.Module.__init__(attention)
    for name, value in dict(
        num_heads=2, num_local_heads=2, kv_lora_rank=3, q_lora_rank=3,
        qk_nope_head_dim=3, qk_rope_head_dim=1, qk_head_dim=4, v_head_dim=2,
        block_size=16, sliding_window=32, scaling=0.5, param_sink_number=0,
        use_gpt_oss_sink=False, use_gpt_oss_sink_rescale=False, use_aicpu_fa_tiling=False,
        on_ascend950=False, is_fa_metadata_producer=True, _fa_meta_suffix="_mla",
        is_cla_fa_metadata_producer=True, cla_swa_gate_window=8, use_mome=False,
        is_cla_reuse_layer=True, is_dsa_layer=False, first_chunk_pa=True,
        sharded_o_proj=False, _fia_aic_core_num=24, _fia_aiv_core_num=48,
        layer_name="model.layers.7.self_attn",
    ).items():
        setattr(attention, name, value)
    attention.W_UK_T = torch.eye(3).repeat(2, 1, 1)
    attention.W_UK_T_cla_swa = attention.W_UK_T * 2
    attention.W_UV = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    attention.gpt_oss_sink = torch.zeros(2)
    attention.gpt_oss_sink_swa = torch.ones(2)
    attention.sink_k_nope = torch.ones(1, 1, 3)
    attention.sink_k_pe = torch.ones(1, 1, 1)
    cache = (torch.randn(1, 16, 3), torch.randn(1, 16, 1))
    attention.attn = SimpleNamespace(
        layer_name="global", kv_cache=cache, impl=SimpleNamespace(SHARE_MASK_TRIL_SPARSE=torch.empty(0)),
    )
    attention.attn_cla_swa = attention.attn
    phase = SimpleNamespace(
        query_cumlens=torch.tensor([3], dtype=torch.int32), seq_lens=torch.tensor([3], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32), chunked_context=None, num_tokens=3,
    )
    metadata = SimpleNamespace(prefill=phase, decode=None, num_actual_tokens=3, num_decode_tokens=0)
    context = SimpleNamespace(capturing=False)
    monkeypatch.setattr(mod, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        mod.torch_npu, "npu_transpose_batchmatmul",
        lambda value, weight, **kwargs: (value.transpose(0, 1) @ weight).transpose(0, 1),
    )
    return SimpleNamespace(
        attention=attention, metadata=metadata, context=context, cache=cache,
        q=torch.arange(18, dtype=torch.float32).reshape(3, 2, 3), pe=torch.ones(3, 2, 1),
    )


def test_cla_query_and_cache_preparation(case, monkeypatch):
    a = case.attention
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    a.q_a_proj = Mock(side_effect=lambda value: value + 1)
    a.q_a_layernorm = Mock(side_effect=lambda value: value * 2)
    a.q_b_proj = Mock(side_effect=lambda value: value)
    a._q_rope = Mock(side_effect=lambda value, cos, sin: value + cos.unsqueeze(1))
    cos = torch.ones(3, 1)
    q, absorbed, pe = a._prepare_cla_queries(hidden, cos, cos)
    expected = ((hidden + 1) * 2).reshape(3, 2, 4)
    torch.testing.assert_close(q, expected[..., :3])
    torch.testing.assert_close(absorbed, q)
    torch.testing.assert_close(a._w_uk_t_absorb(q, a.W_UK_T_cla_swa), q * 2)
    torch.testing.assert_close(pe, expected[..., 3:] + 1)
    a.kv_a_proj_with_mqa_swa = Mock(side_effect=lambda value: value[:, :4])
    updated = (torch.ones(3, 3), cos, torch.zeros_like(case.cache[0]), torch.zeros_like(case.cache[1]))
    update = Mock(return_value=updated)
    monkeypatch.setattr(torch.ops.vllm, "npu_pangu_cla_swa_kv_cache_update", update)
    global_cache, local_cache, current = a._prepare_cla_kv(hidden, cos, cos)
    assert global_cache is case.cache
    assert local_cache[0] is updated[2] and local_cache[1] is updated[3]
    assert current[0] is updated[0] and current[1] is updated[1]
    torch.testing.assert_close(update.call_args.args[0], hidden[:, :4])
    assert update.call_args.args[-1] == a.layer_name


@pytest.mark.parametrize("sp_tokens", [None, 2, 0])
def test_v3_absorb_uses_phase_or_sp_metadata(case, monkeypatch, sp_tokens):
    a, metadata = case.attention, case.metadata
    a.use_gpt_oss_sink = True
    manager = None
    if sp_tokens is not None:
        manager = SimpleNamespace(valid_token_count=sp_tokens, sp_attn_meta=lambda: ("q", "kv", "table"))
    count = 3 if sp_tokens is None else sp_tokens
    latent = case.q[:count]
    pa = Mock(return_value=latent)
    monkeypatch.setattr(a, "_apply_swa_pa_attention", pa)
    output = a._apply_SWA_attention_prefill_absorb(
        case.q, case.pe, case.cache, attn_metadata=metadata, sp_manager=manager,
        recompute_metadata=False if sp_tokens is not None else None,
    )
    expected = torch.einsum("tnl,nlv->tnv", latent, a.W_UV).reshape(count, 4)
    torch.testing.assert_close(output, expected)
    if count == 0:
        pa.assert_not_called()
    else:
        kwargs = pa.call_args.kwargs
        assert kwargs["num_actual_tokens"] == count
        assert kwargs["recompute_metadata"] is (sp_tokens is None)
        assert kwargs["metadata_caller"] == "prefill_absorb_mla"
        if manager is not None:
            assert (kwargs["query_cumlens"], kwargs["seq_lens"], kwargs["block_table"]) == ("q", "kv", "table")
        else:
            assert kwargs["query_cumlens"] is metadata.prefill.query_cumlens


@pytest.mark.parametrize("backend,capturing,aicpu,sinks,padded", [
    ("sink", False, False, 1, True), ("sink", True, False, 1, False),
    ("sink", False, False, 0, False), ("sink", False, True, 0, False),
    ("pioneer", False, False, 1, False), ("pioneer", True, False, 1, False),
    ("pioneer", False, True, 0, False),
    ("gpt", False, False, 0, False), ("gpt", True, False, 0, False),
])
def test_swa_pa_backends_capture_and_padding(case, monkeypatch, backend, capturing, aicpu, sinks, padded):
    a = case.attention
    a.on_ascend950, a.use_gpt_oss_sink = backend == "pioneer", backend == "gpt"
    a.use_aicpu_fa_tiling, a.param_sink_number = aicpu, sinks
    case.context.capturing = capturing
    count = 2 if padded else 3
    op = Mock(return_value=(case.q[:count].transpose(0, 1).contiguous(), torch.empty(0)))
    monkeypatch.setattr(torch.ops.custom, "npu_fused_infer_attention_sink", op)
    monkeypatch.setattr(torch.ops.custom, "npu_ai_infra_attention_pioneer", op)
    monkeypatch.setattr(mod.torch_npu, "_npu_attention_pioneer", op, raising=False)
    monkeypatch.setattr(mod.torch_npu, "npu_fused_infer_attention_score_v2", op)
    tiling = Mock(return_value=torch.empty(1, dtype=torch.int32))
    monkeypatch.setattr(mod, "npu_fused_infer_attention_sink_metadata", tiling)
    monkeypatch.setattr(mod, "npu_ai_infra_attention_pioneer_metadata", tiling)
    capture = Mock()
    monkeypatch.setattr(mod, "capture_graph_task", capture)
    caller = "decode_mla" if a.on_ascend950 and aicpu else "cla_prefill_absorb"
    if backend == "gpt":
        case.metadata.decode, case.metadata.prefill = case.metadata.prefill, None
        output = a._apply_SWA_attention_decode(case.q, case.pe, case.cache, attn_metadata=case.metadata)
    else:
        output = a._apply_swa_pa_attention(
            case.q, case.pe, case.cache, case.metadata.prefill, num_tokens=3, num_actual_tokens=count,
            attention_layer=a.attn, sliding_window=32, metadata_caller=caller, recompute_metadata=True,
        )
    if capturing:
        torch.testing.assert_close(output, torch.zeros_like(case.q))
        op.assert_not_called()
        kwargs = capture.call_args.kwargs["op_kwargs"]
        expected_op = {"sink": mod.OP_FIA_SINK, "pioneer": mod.OP_FIA_PIONEER, "gpt": mod.OP_FIA_V2}[backend]
        assert capture.call_args.kwargs["op_desc"] == expected_op
    else:
        torch.testing.assert_close(output[:count], case.q[:count])
        torch.testing.assert_close(output[count:], torch.zeros_like(output[count:]))
        capture.assert_not_called()
        kwargs = op.call_args.kwargs
    for key, value in (("key_sink", a.sink_k_nope), ("value_sink", a.sink_k_nope), ("key_rope_sink", a.sink_k_pe)):
        if sinks:
            assert kwargs[key] is value
        else:
            assert key not in kwargs
    assert kwargs["pre_tokens"] == 31
    assert kwargs["query"].shape[0] == count
    if aicpu:
        assert kwargs["metaData" if a.on_ascend950 else "meta_data"] is tiling.return_value
        args, recompute, actual_caller = tiling.call_args.args
        assert args["block_size"] == a.block_size and recompute is True
        assert actual_caller == caller
    else:
        tiling.assert_not_called()


@pytest.mark.parametrize("ascend950,aicpu,sinks", [
    (False, False, 0), (False, True, 1), (True, False, 1), (True, True, 1),
])
def test_direct_prefill_preserves_token_sinks(case, monkeypatch, ascend950, aicpu, sinks):
    a = case.attention
    a.on_ascend950, a.use_aicpu_fa_tiling, a.param_sink_number = ascend950, aicpu, sinks
    a.kv_a_layernorm = Mock(side_effect=lambda value: value)
    a.param_sink_compressed_kv = torch.ones(1, 3)
    a.param_sink_k_pe = torch.ones(1, 1)
    a.kv_b_proj = Mock(side_effect=lambda value: value[:, :1].expand(-1, 10))
    native = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    op = Mock(return_value=(native, torch.empty(0)))
    monkeypatch.setattr(torch.ops.custom, "npu_fused_infer_attention_sink", op)
    monkeypatch.setattr(torch.ops.custom, "npu_ai_infra_attention_pioneer", op)
    monkeypatch.setattr(mod.torch_npu, "_npu_attention_pioneer", op, raising=False)
    tiling = Mock(return_value=torch.empty(1))
    monkeypatch.setattr(mod, "npu_ai_infra_attention_pioneer_metadata", tiling)
    monkeypatch.setattr(mod, "npu_fused_infer_attention_sink_metadata", tiling)
    output = a._apply_SWA_attention_prefill(case.q, case.pe, case.q, case.pe, native, case.metadata)
    torch.testing.assert_close(output, native.flatten(1))
    assert ("key_sink" in op.call_args.kwargs) == bool(sinks)
    assert op.call_args.kwargs["pre_tokens"] == 31
    if aicpu:
        assert tiling.call_args.args[-1] == "prefill_mla"


@pytest.mark.parametrize("error,message", [
    ("layout", "Unsupported GPT-OSS sink output layout"), ("sink", "was not initialized"),
    ("capture", "num_tokens is required"), ("rescale_capture", "num_tokens is required"),
    ("local", "Missing CLA local SWA attention metadata"), ("cp", "Missing CLA local SWA prefill metadata"),
    ("cp_metadata", "CLA CP requires prefill SWA metadata"),
    ("fc2", "Missing CLA local SWA prefill metadata"), ("fc2_mome", "does not support MoME"),
])
def test_v3_rejects_invalid_attention_inputs(case, error, message):
    a, metadata = case.attention, case.metadata
    case.context.capturing = True
    calls = {
        "layout": lambda: a._rescale_gpt_oss_sink_output(case.q, torch.ones(3, 2), a.gpt_oss_sink, "invalid"),
        "sink": lambda: a._apply_gpt_oss_fia({"query": case.q}),
        "capture": lambda: a._apply_gpt_oss_fia({"query": case.q}, output_shape=(2, 3, 3)),
        "rescale_capture": lambda: a._apply_gpt_oss_fia_rescale(
            {"query": case.q}, a.gpt_oss_sink, (2, 3, 3), None, None, "decode",
        ),
        "local": lambda: a._forward_cla(case.q, case.pe, case.pe, metadata, None, None),
        "cp": lambda: a._forward_cla_prefill_cp(case.q, case.pe, case.pe, metadata, None),
        "cp_metadata": lambda: a._apply_cla_swa_attention_cp(
            case.q, case.pe, case.cache, SimpleNamespace(prefill=None), None,
        ),
        "fc2": lambda: a._forward_cla_prefill_FC2(case.q, case.pe, case.pe, metadata, None),
        "fc2_mome": lambda: a._forward_cla_prefill_FC2(case.q, case.pe, case.pe, metadata, metadata),
    }
    if error == "sink":
        a.gpt_oss_sink = None
    if error == "fc2_mome":
        a.use_mome = True
    call = calls.get(error)
    assert call is not None
    with pytest.raises((ValueError, RuntimeError), match=message):
        call()


def test_no_token_sink_skips_parameter_allocation(case, monkeypatch):
    empty = Mock(side_effect=AssertionError("zero sinks must not allocate parameters"))
    monkeypatch.setattr(torch, "empty", empty)
    case.attention._init_param_sinks()
    empty.assert_not_called()
    assert not hasattr(case.attention, "param_sink_compressed_kv")


@pytest.mark.parametrize("mode", ["cla", "dsa", "pa", "direct"])
def test_prefill_dispatch_and_cla_decode(case, monkeypatch, mode):
    a, metadata = case.attention, case.metadata
    a.is_cla_reuse_layer = mode == "cla"
    a.is_dsa_layer = mode == "dsa"
    a.first_chunk_pa = mode != "direct"
    output = torch.ones(3, 4)
    callees = ["_forward_cla", "_apply_DSA_attention", "_apply_SWA_attention_prefill_absorb",
               "_apply_SWA_attention_prefill"]
    mocks = {name: Mock(return_value=output) for name in callees}
    for name, mock in mocks.items():
        monkeypatch.setattr(a, name, mock)
    prolog = (case.q, case.pe, case.cache, None) if mode != "direct" else (case.q,) * 5
    monkeypatch.setattr(a, "_mla_prolog", Mock(return_value=prolog))
    monkeypatch.setattr(a, "_mla_epilog", lambda value, *_args: value)
    assert a._forward_prefill(case.q, case.pe, case.pe, metadata, metadata) is output
    selected = callees[["cla", "dsa", "pa", "direct"].index(mode)]
    mocks[selected].assert_called_once()
    for name in set(callees) - {selected}:
        mocks[name].assert_not_called()
    if mode == "cla":
        assert a._forward_decode(case.q, case.pe, case.pe, metadata, metadata) is output
        assert mocks[selected].call_count == 2


def test_cla_dsa_combines_global_and_local_outputs(case, monkeypatch):
    a = case.attention
    a.is_dsa_layer = True
    a.W_UV_cla_swa = a.W_UV * 2
    monkeypatch.setattr(a, "_prepare_cla_queries", lambda *args: (case.q, case.q, case.pe))
    monkeypatch.setattr(a, "_prepare_cla_kv", lambda *args: (case.cache, case.cache, (case.q, case.pe)))
    monkeypatch.setattr(a, "_prepare_cla_swa_inputs", lambda *args: (case.q, case.cache))
    indices = torch.zeros(3, 1, 2, dtype=torch.int32)
    monkeypatch.setattr(a, "_get_topk_indices", lambda metadata: indices)
    dsa = Mock(return_value=torch.ones(3, 4))
    monkeypatch.setattr(a, "_apply_DSA_attention", dsa)
    monkeypatch.setattr(a, "_apply_swa_pa_attention", lambda *args, **kwargs: case.q)
    monkeypatch.setattr(a, "_mla_epilog", lambda value, *args: value)
    a.swa_gate_global = Mock(side_effect=lambda hidden, output: output * 2)
    a.swa_gate_local = Mock(side_effect=lambda hidden, output: output * 3)
    output = a._forward_cla(case.q, case.pe, case.pe, case.metadata, case.metadata, None)
    expected = 2 + torch.einsum("tnl,nlv->tnv", case.q, a.W_UV_cla_swa).reshape(3, 4) * 3
    torch.testing.assert_close(output, expected)
    assert dsa.call_args.args[3] is indices


@pytest.mark.parametrize("mome,sp,gather", [(False, False, False), (True, False, True), (True, True, True)])
def test_fc2_epilog_preserves_mome_and_token_padding(case, monkeypatch, mome, sp, gather):
    a = case.attention
    a.use_mome, a.enable_mome_sp = mome, sp
    a.num_local_heads, a.num_heads = 1, 2
    a.o_conv = SimpleNamespace(input_size_per_partition=4)
    monkeypatch.setattr(a, "_apply_o_proj", lambda value: value)
    group = SimpleNamespace(rank_in_group=1, all_gather=Mock(side_effect=lambda value, dim: value.repeat(1, 2)))
    monkeypatch.setattr(mod, "get_tp_group", lambda: group)
    all_to_all = Mock(return_value=torch.full((2, 4), 4.0))
    monkeypatch.setattr(a, "_flashcomm2_all_to_all", all_to_all)

    def apply_mome(value, *_args, inplace=False, **kwargs):
        return value.add_(1) if inplace else value + 1

    conv = Mock(side_effect=apply_mome)
    monkeypatch.setattr(a, "_apply_MOME", conv)
    output = a._flashcomm2_epilog(torch.ones(3, 2), case.metadata, None, 2, 4, 0, 3, gather)
    if not mome:
        torch.testing.assert_close(output, torch.ones(3, 4))
        conv.assert_not_called()
    elif sp:
        torch.testing.assert_close(output, torch.full((2, 4), 5.0))
        assert conv.call_args.kwargs["inplace"] and conv.call_args.kwargs["ena_sp"]
        all_to_all.assert_called_once()
    else:
        torch.testing.assert_close(output, torch.tensor([[2.0] * 4, [0.0] * 4]))
        all_to_all.assert_not_called()


def test_unquant_kv_uses_pa_cache_without_expanding_values(case, monkeypatch):
    a, metadata = case.attention, case.metadata
    metadata.slot_mapping = torch.arange(3)
    monkeypatch.setattr(a, "_kv_rmsnorm_rope_cache_v2_kwargs", lambda *args: {})
    op = Mock(return_value=(case.pe, case.q))
    monkeypatch.setattr(torch.ops.custom, "npu_ai_infra_kv_rmsnorm_rope_cache_v2", op)
    result, indices = a._npu_kvrmsnorm_rope_cache_unquant(case.q, case.cache, case.pe, case.pe, metadata)
    assert result is case.cache and indices is None
    torch.testing.assert_close(op.call_args.kwargs["k_cache"], case.cache[1].unsqueeze(2))


def test_mixed_forward_restores_shared_metadata(case, monkeypatch):
    a = case.attention
    a.prefix, a.cla_swa_attn_name = "global", "local"
    a.is_cp_layer = a.is_attn_sp_layer = a.enable_flashcomm2 = False
    a.tp_size = 1
    a.moe_comm_strategy = "allreduce"

    def metadata(offset):
        return SimpleNamespace(
            prefill=SimpleNamespace(), decode=SimpleNamespace(), num_actual_tokens=3, num_decode_tokens=1,
            num_decodes=1, num_prefills=1, slot_mapping=torch.arange(3) + offset,
            slot_mapping_2d=torch.arange(6).reshape(3, 2) + offset,
        )

    global_meta, local_meta = metadata(0), metadata(10)
    initial = [(m.prefill, m.slot_mapping, m.slot_mapping_2d) for m in (global_meta, local_meta)]
    case.context.no_compile_layers = {"layer": a}
    case.context.attn_metadata = {"global.attn": global_meta, "local": local_meta}

    def forward(hidden, cos, sin, meta, local, mome):
        assert meta.num_actual_tokens == local.num_actual_tokens == hidden.shape[0]
        return hidden + (10 if meta.prefill is None else 20)

    monkeypatch.setattr(a, "_forward_prefill", forward)
    monkeypatch.setattr(a, "_forward_decode", forward)
    hidden = torch.zeros(3, 4)
    output = mod.npu_pangu_forward(hidden, case.pe, case.pe, "layer")
    torch.testing.assert_close(output, torch.tensor([[10.0] * 4, [20.0] * 4, [20.0] * 4]))
    for meta, (prefill, slots, slots_2d) in zip((global_meta, local_meta), initial):
        assert meta.prefill is prefill and meta.num_actual_tokens == 3
        torch.testing.assert_close(meta.slot_mapping, slots)
        torch.testing.assert_close(meta.slot_mapping_2d, slots_2d)


@pytest.mark.parametrize("mome,inplace", [(False, False), (True, False), (True, True)])
def test_cp_kv_projection_preserves_optional_mome(case, monkeypatch, mome, inplace):
    a, metadata = case.attention, case.metadata
    a.use_mome, a.use_mome_inplace_update, a.enable_mome_sp = mome, inplace, False
    a.skip_topk, a.cache_config = True, SimpleNamespace(cache_dtype="auto")
    a.rope_interleave = False
    a.o_proj = SimpleNamespace(tp_size=1)
    a.qa_conv = a.compresskv_conv = a.o_conv = object()
    a.q_a_proj = a.q_a_layernorm = a.q_b_proj = Mock(side_effect=lambda value: value)
    a.kv_a_proj_with_mqa = Mock(side_effect=lambda value: value[:, :4].clone())
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    methods = ("slice_tokens", "cp_slice", "sp_to_cp", "ag_tokens", "cp_to_sp")
    manager = SimpleNamespace(**{name: (lambda value, **kwargs: value) for name in methods})
    metadata.prefill.sp_manager = manager

    def mome_op(value, *_args, inplace=False, **kwargs):
        return value.add_(1) if inplace else value + 1

    monkeypatch.setattr(a, "_apply_MOME", mome_op)
    monkeypatch.setattr(a, "_get_topk_indices", lambda meta: None)
    monkeypatch.setattr(mod.torch_npu, "npu_rotary_mul", lambda value, *args, **kwargs: value)
    kv_args = Mock(side_effect=lambda kv, *args, **kwargs: {"kv": kv})
    monkeypatch.setattr(a, "_kv_rmsnorm_rope_cache_v2_kwargs", kv_args)
    cache_update = Mock(return_value=(case.pe, case.q))
    monkeypatch.setattr(torch.ops.custom, "npu_ai_infra_kv_rmsnorm_rope_cache_v2", cache_update)
    monkeypatch.setattr(a, "_apply_DSA_attention_cp", lambda *args, **kwargs: torch.ones(3, 4))
    monkeypatch.setattr(a, "_apply_o_proj", lambda value: value)
    output = a._forward_prefill_cp(hidden, torch.ones(3, 1), torch.ones(3, 1), metadata)
    expected_kv = hidden[:, :4].clone()
    expected_kv[:, :3] += int(mome)
    torch.testing.assert_close(cache_update.call_args.kwargs["kv"], expected_kv)
    torch.testing.assert_close(output, torch.full((3, 4), 1.0 + int(mome)))
