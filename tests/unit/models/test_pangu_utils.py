# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from omni_npu.v1.models.pangu import utils as pangu_utils


pytestmark = pytest.mark.unit


class _FakeEvent:
    def record(self, *_args, **_kwargs):
        return None

    def wait(self, *_args, **_kwargs):
        return None


class _FakeStream:
    def wait_stream(self, _other):
        return None

    def wait(self, *_args, **_kwargs):
        return None


def _patch_npu_runtime(monkeypatch):
    """Replace torch.npu stream/event APIs with CPU-safe fakes."""
    npu = SimpleNamespace(
        Stream=_FakeStream,
        Event=_FakeEvent,
        current_stream=lambda: _FakeStream(),
        stream=lambda _stream: nullcontext(),
    )
    monkeypatch.setattr(pangu_utils.torch, "npu", npu, raising=False)
    return npu


def _uncompiled_make_stat():
    """Load make_stat without the torchair compile wrapper."""
    tree = ast.parse(Path(pangu_utils.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "make_stat":
            node.decorator_list = []
            code = compile(
                ast.Module(body=[node], type_ignores=[]),
                pangu_utils.__file__,
                "exec",
            )
            namespace = dict(vars(pangu_utils))
            exec(code, namespace)
            return namespace["make_stat"]
    raise RuntimeError("make_stat not found")


def _clear_fn_cache(fn, *names):
    """Drop lazy attributes cached on a helper function."""
    for name in names:
        if hasattr(fn, name):
            delattr(fn, name)


def _stub_gather_routing(monkeypatch, pack_cols):
    """Stub sort/stat/reroute helpers used by gather_routing tests."""
    _patch_npu_runtime(monkeypatch)
    _clear_fn_cache(pangu_utils.gather_routing, "_buf_size")
    ids = torch.zeros(2, 2, dtype=torch.int32)
    hist = torch.ones(4, dtype=torch.int32)
    order = torch.arange(4, dtype=torch.int32)
    monkeypatch.setattr(
        pangu_utils, "sort_hist_fused", lambda *_a, **_k: (hist, order)
    )
    data = torch.arange(30, dtype=torch.int32)
    reorg = torch.zeros(3, 3, dtype=torch.int32)
    pack = torch.zeros(2, pack_cols, dtype=torch.int32)
    monkeypatch.setattr(
        pangu_utils, "make_stat", lambda *_a, **_k: (data, reorg, pack)
    )
    monkeypatch.setattr(pangu_utils, "record_event", lambda: _FakeEvent())
    monkeypatch.setattr(
        pangu_utils, "rerouting", lambda *_a, **_k: torch.arange(4)
    )
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    return ids, data, pack, group


def test_npu_compile_uses_torchair_backend(monkeypatch):
    """npu_compile should compile with the torchair NPU backend."""
    captured = {}
    monkeypatch.setattr(pangu_utils.torchair, "CompilerConfig", lambda: "cfg")
    monkeypatch.setattr(
        pangu_utils.torchair,
        "get_npu_backend",
        lambda compiler_config: "backend",
    )

    def fake_compile(fn, backend=None):
        captured["backend"] = backend
        return fn

    monkeypatch.setattr(pangu_utils.torch, "compile", fake_compile)

    def identity(value):
        return value

    assert pangu_utils.npu_compile(identity) is identity
    assert captured["backend"] == "backend"


def test_no_aiv_returns_original_group_outside_aiv(monkeypatch):
    """Without AIV expansion, no_aiv is a passthrough."""
    monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    group = SimpleNamespace(
        ranks=[0], world_size=1, rank_in_group=0, device_group="g"
    )
    assert pangu_utils.no_aiv(group) is group


def test_no_aiv_builds_and_caches_cpu_group_in_aiv_mode(monkeypatch):
    """AIV mode builds an AI_CPU process group and caches it per group id."""
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    group0 = SimpleNamespace(
        ranks=[0, 1], world_size=2, rank_in_group=0, device_group="g0"
    )
    options = SimpleNamespace(hccl_config={})
    hccl = SimpleNamespace(Options=lambda: options)
    c10d = SimpleNamespace(ProcessGroupHCCL=hccl)
    monkeypatch.setattr(
        pangu_utils.torch_npu,
        "_C",
        SimpleNamespace(_distributed_c10d=c10d),
        raising=False,
    )
    monkeypatch.setattr(pangu_utils.dist, "get_backend", lambda _g: "hccl")
    monkeypatch.setattr(
        pangu_utils.dist, "new_group", lambda *args, **kwargs: "pg_cpu"
    )
    _clear_fn_cache(pangu_utils.no_aiv, str(id(group0)))
    out = pangu_utils.no_aiv(group0)
    assert out.world_size == 2
    assert out.rank_in_group == 0
    assert out.device_group == "pg_cpu"
    assert options.hccl_config["hccl_op_expansion_mode"] == 2
    assert pangu_utils.no_aiv(group0) is out


def test_named_stream_current_and_named_cache(monkeypatch):
    """named_stream('current') is the default stream; other names are cached."""
    _patch_npu_runtime(monkeypatch)
    current = pangu_utils.named_stream("current")
    named = pangu_utils.named_stream("com_stream")
    assert isinstance(current, _FakeStream)
    assert pangu_utils.named_stream("com_stream") is named
    assert named is not current


def test_record_event_creates_and_records(monkeypatch):
    """record_event constructs an NPU event and records it."""
    recorded = []

    class TrackingEvent(_FakeEvent):
        def record(self, *_args, **_kwargs):
            recorded.append(True)

    monkeypatch.setattr(
        pangu_utils.torch,
        "npu",
        SimpleNamespace(Event=TrackingEvent),
        raising=False,
    )
    event = pangu_utils.record_event()
    assert isinstance(event, TrackingEvent)
    assert recorded == [True]


def test_reinterpret_round_trips_float_bits():
    """reinterpret is a zero-copy dtype view that preserves bits."""
    source = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    as_int = pangu_utils.reinterpret(source, torch.int32)
    assert as_int.dtype == torch.int32
    assert tuple(as_int.shape) == tuple(source.shape)
    restored = pangu_utils.reinterpret(as_int, torch.float32)
    torch.testing.assert_close(restored, source)


def test_all_to_all_allocates_out_and_forwards_splits(monkeypatch):
    """all_to_all converts tensor splits and writes into a new output."""
    captured = {}

    def fake_a2a(out, inp, recv, send, group=None):
        captured["recv"] = recv
        captured["send"] = send
        captured["group"] = group
        out.copy_(inp)

    monkeypatch.setattr(
        pangu_utils.torch.distributed, "all_to_all_single", fake_a2a
    )
    group = SimpleNamespace(device_group="ep")
    inp = torch.arange(4, dtype=torch.float32).view(2, 2)
    out = pangu_utils.all_to_all(
        group, inp, torch.tensor([1, 1]), torch.tensor([1, 1])
    )
    assert tuple(out.shape) == (2, 2)
    assert captured["recv"] == [1, 1]
    assert captured["send"] == [1, 1]
    assert captured["group"] == "ep"
    torch.testing.assert_close(out, inp)


def test_all_to_all_reuses_provided_out(monkeypatch):
    """When out is given, all_to_all writes into that buffer."""
    monkeypatch.setattr(
        pangu_utils.torch.distributed,
        "all_to_all_single",
        lambda out, inp, recv, send, group=None: out.copy_(inp),
    )
    group = SimpleNamespace(device_group="ep")
    inp = torch.ones(2, 3)
    dest = torch.zeros(2, 3)
    result = pangu_utils.all_to_all(group, inp, [2], [2], out=dest)
    assert result is dest
    torch.testing.assert_close(dest, inp)


def test_sort_hist_fused_forwards_routing_flags(monkeypatch):
    """sort_hist_fused maps cumsum/inv_order flags into npu_moe_init_routing_v2."""
    captured = {}

    def fake_init(_dummy, expert_idx=None, **kwargs):
        captured.update(kwargs)
        hist = torch.ones(kwargs["expert_num"], dtype=torch.int32)
        idx = torch.arange(expert_idx.numel(), dtype=torch.int32)
        return None, idx, hist, None

    monkeypatch.setattr(
        pangu_utils.torch_npu, "npu_moe_init_routing_v2", fake_init
    )
    ids = torch.zeros(4, dtype=torch.int32)
    hist, idx = pangu_utils.sort_hist_fused(
        ids, 8, inv_order=True, cumsum=True
    )
    assert captured["expert_tokens_num_type"] == 0
    assert captured["row_idx_type"] == 0
    assert tuple(hist.shape) == (8,)
    assert tuple(idx.shape) == (4,)


def test_make_stat_builds_routing_pack(monkeypatch):
    """Uncompiled make_stat all-gathers hist and returns packed routing stats."""
    make_stat = _uncompiled_make_stat()

    def fake_ag(dst, src, group=None):
        copies = dst.numel() // src.numel()
        dst.copy_(src.repeat(copies))

    monkeypatch.setattr(
        pangu_utils.torch.distributed, "all_gather_into_tensor", fake_ag
    )
    ep, experts = 2, 4
    pack_sp = torch.arange(experts + 2, dtype=torch.int32)
    avg = torch.tensor([1], dtype=torch.int32)
    group = SimpleNamespace(rank_in_group=0, world_size=ep, device_group="g")
    data, reorg, pack = make_stat(pack_sp, avg, experts, group)
    assert data.dtype == torch.int32
    assert reorg.dtype == torch.int32
    assert tuple(pack.shape) == (ep, pack_sp.numel())
    assert reorg.dim() == 2


def test_routing_stat_splits_packed_data():
    """RoutingStat unpacks the concatenated gather_routing payload."""
    ep, experts = 2, 4
    splits = [ep] * 6 + [experts // ep] * 3 + [experts] * 3
    data = torch.arange(sum(splits), dtype=torch.int32)
    stat = pangu_utils.RoutingStat(data, ep, experts)
    assert tuple(stat.avg_sends.shape) == (ep,)
    assert tuple(stat.hist0.shape) == (experts // ep,)
    assert tuple(stat.map.shape) == (experts // ep, ep)
    assert tuple(stat.rest_map.shape) == (experts // ep, ep)
    assert tuple(stat.avg_map.shape) == (experts // ep, ep)


def test_gather_routing_without_scale(monkeypatch):
    """gather_routing packs hist+index when no gating scale is provided."""
    ids, data, _pack, group = _stub_gather_routing(monkeypatch, pack_cols=8)
    index, order_sp, out_data, parse, done, scale = pangu_utils.gather_routing(
        group, ids, 4, 2
    )
    assert scale is None
    assert order_sp.dtype == torch.int32
    assert out_data is data
    stat = parse(data)
    assert isinstance(stat, pangu_utils.RoutingStat)
    assert isinstance(done, _FakeEvent)
    assert tuple(index.shape) == (4,)


def test_gather_routing_with_scale_reinterprets_float(monkeypatch):
    """gather_routing packs the int-cast scale and restores it after gather."""
    ids, _data, pack, group = _stub_gather_routing(monkeypatch, pack_cols=10)
    scale = torch.tensor([1.5, 2.5], dtype=torch.float32)
    *_rest, out_scale = pangu_utils.gather_routing(group, ids, 4, 2, scale)
    assert out_scale.dtype == torch.float32
    assert out_scale.numel() == pack[:, 8:10].numel()


def test_rerouting_int_preallocates_workspace(monkeypatch):
    """Integer input means allocate a workspace and return the routed index."""
    routed = torch.arange(3, dtype=torch.int32)

    def fake_reroute(workspace, mat):
        assert workspace.dtype == torch.int8
        assert workspace.size(0) == 3
        return None, None, routed, None

    monkeypatch.setattr(
        pangu_utils.torch_npu, "npu_moe_re_routing", fake_reroute
    )
    mat = torch.ones(2, 2, dtype=torch.int32)
    out = pangu_utils.rerouting(3, mat)
    assert torch.equal(out, routed)


def test_rerouting_2d_index_selects_rows(monkeypatch):
    """2D input is reordered by the indices returned from npu_moe_re_routing."""
    idx = torch.tensor([1, 0], dtype=torch.int64)
    monkeypatch.setattr(
        pangu_utils.torch_npu,
        "npu_moe_re_routing",
        lambda *_a, **_k: (None, None, idx, None),
    )
    mat = torch.ones(2, 2, dtype=torch.int32)
    inp = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = pangu_utils.rerouting(inp, mat)
    torch.testing.assert_close(out, inp[[1, 0]])


def test_rerouting_1d_reinterprets_storage(monkeypatch):
    """1D int/float tensors are viewed as bf16 pairs for the reroute kernel."""
    monkeypatch.setattr(
        pangu_utils.torch_npu,
        "npu_moe_re_routing",
        lambda buf, mat: (buf, None, None, None),
    )
    mat = torch.ones(2, 2, dtype=torch.int32)
    inp = torch.arange(4, dtype=torch.int32)
    out = pangu_utils.rerouting(inp, mat)
    assert out.untyped_storage().data_ptr() != 0


def test_quant_ffn_chains_grouped_matmul_and_swiglu(monkeypatch):
    """quant_ffn runs grouped matmul, dequant-swiglu, then the down projection."""
    calls = []

    def fake_gmm(*args, **kwargs):
        calls.append(kwargs.get("output_dtype"))
        tokens = args[0][0]
        if kwargs.get("output_dtype") == torch.int32:
            return [torch.zeros(tokens.size(0), 4, dtype=torch.int32)]
        return [torch.ones(tokens.size(0), 3, dtype=torch.bfloat16)]

    def fake_swiglu(**kwargs):
        tokens = kwargs["x"]
        return (
            torch.zeros(tokens.size(0), 2, dtype=torch.int8),
            torch.ones(tokens.size(0), dtype=torch.float32),
        )

    monkeypatch.setattr(pangu_utils.torch_npu, "npu_grouped_matmul", fake_gmm)
    monkeypatch.setattr(
        pangu_utils.torch_npu, "npu_dequant_swiglu_quant", fake_swiglu
    )
    experts = SimpleNamespace(
        w13_weight=torch.zeros(2, 2),
        w13_weight_scale=torch.ones(2),
        w2_weight=torch.zeros(2, 2),
        w2_weight_scale=torch.ones(1, dtype=torch.bfloat16),
    )
    x_i8 = torch.zeros(3, 2, dtype=torch.int8)
    x_sc = torch.ones(3, dtype=torch.float32)
    hist = torch.tensor([1, 2], dtype=torch.int32)
    out = pangu_utils.quant_ffn(experts, x_i8, x_sc, hist)
    assert out.dtype == torch.bfloat16
    assert tuple(out.shape) == (3, 3)
    assert calls == [torch.int32, torch.bfloat16]


def test_finalize_routing_forwards_drop_pad_mode(monkeypatch):
    """finalize_routing wraps npu_moe_finalize_routing with drop_pad_mode=2."""
    captured = {}
    expected = torch.ones(2, 3)

    def fake_finalize(*args, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        pangu_utils.torch_npu, "npu_moe_finalize_routing", fake_finalize
    )
    x = torch.zeros(4, 3)
    scales = torch.ones(2, 2)
    reorg = torch.arange(4)
    out = pangu_utils.finalize_routing(x, scales, reorg)
    assert out is expected
    assert captured["drop_pad_mode"] == 2
    assert captured["scales"] is scales
    assert captured["expanded_src_to_dst_row"] is reorg
