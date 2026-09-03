# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import torch
import torchair
import torch_npu
import torch.distributed as dist
from types import SimpleNamespace
from collections import defaultdict

from vllm.distributed import GroupCoordinator
from omni_npu.layers.fused_moe.layer import NPUSharedFusedMoE


def npu_compile(fn):
    # torchair compile helper: compile fn into a static NPU graph
    config = torchair.CompilerConfig()
    backend = torchair.get_npu_backend(compiler_config=config)
    return torch.compile(fn, backend=backend)


def no_aiv(group0: GroupCoordinator) -> GroupCoordinator | SimpleNamespace:
    # when AIV mode, create AI_CPU group as replacement for all_to_all compatibility
    if os.environ.get("HCCL_OP_EXPANSION_MODE", "") != "AIV":
        return group0
    sig = str(id(group0))
    if not hasattr(no_aiv, sig):
        cfg = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        cfg.hccl_config = {
            "hccl_op_expansion_mode": 2, # AI_CPU
            "hccl_buffer_size": 300,     # 300M
        }
        device_group = dist.new_group(
            group0.ranks,
            backend=dist.get_backend(group0.device_group),
            pg_options=cfg,
        )
        group = SimpleNamespace(
            world_size=group0.world_size,
            rank_in_group=group0.rank_in_group,
            device_group=device_group,
        )
        setattr(no_aiv, sig, group)
    return getattr(no_aiv, sig)


def named_stream(name: str) -> torch.npu.Stream:
    # lazy named stream factory; "current" returns default, others cached by name
    if name == "current":
        return torch.npu.current_stream()
    if not hasattr(named_stream, name):
        setattr(named_stream, name, torch.npu.Stream())
    return getattr(named_stream, name)


def record_event() -> torch.npu.Event:
    # create and record an NPU event for cross-stream sync points
    e = torch.npu.Event()
    e.record()
    return e


def reinterpret(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # zero-copy bit-level dtype reinterpret via untyped_storage
    y = x.new_empty(0, dtype=dtype)
    unit = x.element_size()
    base = x.storage_offset() * unit
    size = x.numel() * unit
    buf = x.untyped_storage()[base: base + size]
    y.set_(buf)
    return y.view(*x.shape)


def all_to_all(
    group: GroupCoordinator,
    x: torch.Tensor,
    s: torch.Tensor | list[int], # send_splits
    r: torch.Tensor | list[int], # recv_splits
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    # asymmetric all_to_all with unequal send/recv splits, used for MoE token exchange
    r = r if isinstance(r, list) else r.tolist()
    s = s if isinstance(s, list) else s.tolist()
    assert sum(s) == x.size(0)
    if out is not None:
        assert out.size(0) == sum(r)
        assert out.size(1) == x.size(1)
    else:
        out = x.new_empty(sum(r), *x.shape[1:])
    torch.distributed.all_to_all_single(
        out, x, r, s, group=group.device_group)
    return out


def sort_hist_fused(
    ids: torch.Tensor,
    exps: int,
    inv_order: bool = False,
    cumsum: bool = False,
) -> tuple[torch.Tensor, ...]:
    # fused sort + histogram via npu_moe_init_routing_v2 (local, no comm)
    _, idx, hist, _ = torch_npu.npu_moe_init_routing_v2(
        ids.new_empty(ids.size(0), 1, dtype=torch.bfloat16),
        expert_idx=ids,
        active_num=ids.numel(),
        expert_num=exps,
        active_expert_range=[0, exps],
        expert_tokens_num_type=0 if cumsum else 1,
        expert_tokens_num_flag=True,
        row_idx_type=0 if inv_order else 1,
        quant_mode=-1)
    return hist, idx


@npu_compile
def make_stat(
    pack_sp: torch.Tensor, # [N+x] packed hist + index (+ scale)
    avg: torch.Tensor,     # [1] fixed lower bound
    N: int,
    group: GroupCoordinator,
):
    # compiled NPU graph: all_gather hist → avg/rest split → reorg → pack+all_gather
    # avg is a fixed lower bound (independent of D2H), allowing early FFN dispatch.
    # rest = the dynamic remainder beyond avg; its exact count requires D2H sync (stat_cpu).
    assert pack_sp.dim() == 1
    rank = group.rank_in_group
    ep = group.world_size
    n = N // ep                                 # experts per rank

    pack = pack_sp.new_empty(ep, pack_sp.size(0))
    torch.distributed.all_gather_into_tensor(
        pack.flatten(), pack_sp, group=group.device_group)

    hist_rank = pack[:, :N].float().contiguous()  # [ep, N]
    hist_exp = hist_rank.transpose(0, 1).contiguous() # [N, ep] global histogram

    z1 = hist_rank.new_zeros(1)
    z2 = hist_rank.new_zeros(N - n * rank)
    z3 = hist_rank.new_zeros(n * rank)

    cum_exp = hist_exp.view(ep, -1).cumsum(1)   # cumulative tokens per expert across ranks
    rest_exp = (cum_exp - avg[0]).clamp(min=0).flatten()
    rest_exp = torch.cat([z1, rest_exp.diff().clamp(min=0)]).view(-1, ep)
    avg_exp = hist_exp - rest_exp               # [N, ep] fixed (avg) vs dynamic remainder (rest)

    full_hist = hist_exp.sum(1)                 # [N] total tokens per expert
    rest_hist = rest_exp.sum(1)                 # [N] dynamic remainder tokens per expert
    avg_hist = full_hist - rest_hist            # [N] fixed tokens per expert

    rest_map = rest_exp.view(-1, ep).transpose(0, 1)
    rest_map = rest_map.view(ep, ep, -1).sum(-1)  # [ep, ep] rest token exchange matrix
    full_map = hist_rank.view(ep, ep, -1).sum(-1) # [ep, ep] full token exchange matrix
    avg_map = full_map - rest_map                 # [ep, ep] avg token exchange matrix

    hist = torch.cat([z2, hist_rank.flatten(), z3])
    hist = hist.view(-1, N)                     # [ep+1, N] shifted histogram
    rest = hist[:, n:].sum(1).view(-1, 1)       # [ep+1, 1] dynamic remainder beyond n experts/rank
    reorg = torch.cat([hist[:, :n], rest], 1)   # [ep+1, n+1] reorg routing matrix

    data = torch.cat([
        avg_map[:, rank],             # [ep] avg_sends
        avg_map[rank, :],             # [ep] avg_recvs
        rest_map[:, rank],            # [ep] rest_sends
        rest_map[rank, :],            # [ep] rest_recvs
        full_map[:, rank],            # [ep] full_sends
        full_map[rank, :],            # [ep] full_recvs
        full_hist.view(ep, -1)[rank], # [n] hist0
        avg_hist.view(ep, -1)[rank],  # [n] hist1
        rest_hist.view(ep, -1)[rank], # [n] hist2
        hist_exp.view(ep, -1)[rank],  # [N] full_detail
        rest_exp.view(ep, -1)[rank],  # [N] rest_detail
        avg_exp.view(ep, -1)[rank],   # [N] avg_detail
    ])
    return data.int(), reorg.int(), pack


class RoutingStat:
    # per-rank statistics container returned by gather_routing/OpGatherRouting, constructed via RoutingStat(data, ep, N)
    def __init__(self, data: torch.Tensor, ep: int, N: int):
        splits = [ep] * 6 + [N // ep] * 3 + [N] * 3
        self.data = data
        (
            self.avg_sends,
            self.avg_recvs,
            self.rest_sends,
            self.rest_recvs,
            self.full_sends,
            self.full_recvs,
            self.hist0,
            self.hist1,
            self.hist2,
            self.map,
            self.rest_map,
            self.avg_map,
        ) = torch.split(data, splits)
        self.map = self.map.view(-1, ep)
        self.rest_map = self.rest_map.view(-1, ep)
        self.avg_map = self.avg_map.view(-1, ep)


def gather_routing(
    ep_group: GroupCoordinator,  # EP communication group
    ids: torch.Tensor,           # [T, K] expert indices from gating
    exp: int,                    # total logical experts N
    avg: int,                    # fixed lower bound (no D2H dependency)
    scale: torch.Tensor | None = None, # [T] gating scales
):
    # gather-routing entry (non-compiled path): sort+hist → make_stat → RoutingStat → rerouting
    # avg is a fixed lower bound independent of D2H; rest is the dynamic remainder requiring D2H.
    assert ids.dtype == torch.int
    assert ids.dim() == 2
    ep = ep_group.world_size
    rank = ep_group.rank_in_group
    T, K = ids.shape

    hist_sp, order_sp = sort_hist_fused(ids, exp, cumsum=False)
    hist_sp, order_sp = hist_sp.int(), order_sp.int()

    def get_buffer(size):
        if not hasattr(gather_routing, "_buf_size"):
            setattr(gather_routing, "_buf_size", [0])
        buf_size = getattr(gather_routing, "_buf_size")
        if buf_size[0] < size:
            buf_size[0] = size
        return ids.new_empty(buf_size[0])

    idx_sp = (order_sp / K).int() + (rank * T)
    if scale is None:
        pack_sp = get_buffer(exp + T * K)
        torch.cat([hist_sp, idx_sp],
            dim=0, out=pack_sp[:exp + T * K])
    else:
        assert scale.dim() == 1
        assert scale.size(0) == T
        assert scale.dtype == torch.float
        scale = reinterpret(scale, torch.int)
        pack_sp = get_buffer(exp + T * K + T)
        torch.cat([hist_sp, idx_sp, scale],
            dim=0, out=pack_sp[:exp + T * K + T])

    avg = ids.new_full([1], avg)
    data, reorg, pack = make_stat(pack_sp, avg, exp, ep_group)
    done = record_event()

    if scale is None:
        orders = pack[:, exp: exp + T * K].contiguous()
    else:
        orders: torch.Tensor = pack[:, exp: exp + T * K].contiguous()
        scales: torch.Tensor = pack[:, exp + T * K: exp + T * K + T].contiguous()
        scale = reinterpret(scales, torch.float).flatten()

    def parse(data: torch.Tensor) -> RoutingStat:
        return RoutingStat(data, ep, exp)

    index = rerouting(orders.flatten(), reorg)
    results = index, order_sp, data, parse, done, scale
    return results


def rerouting(
    x: torch.Tensor | int,
    mat: torch.Tensor,
) -> torch.Tensor:
    # MoE rerouting via routing matrix; int=pre-allocate, 1D=zero-copy bf16 cast
    assert mat.dim() == 2
    if type(x) is int:
        x = mat.new_empty(x, 1, dtype=torch.int8)
        _, _, y, _ = torch_npu.npu_moe_re_routing(x, mat)
        return y

    if x.dim() == 2:
        tmp = x.new_empty(x.size(0), 1, dtype=torch.int8)
        _, _, idx, _ = torch_npu.npu_moe_re_routing(tmp, mat)
        return torch.index_select(x, 0, idx)

    assert x.dim() == 1
    assert x.dtype in [torch.int, torch.float]
    buf = x.new_empty(0, dtype=torch.bfloat16)
    buf.set_(x.untyped_storage())
    offset = x.storage_offset()
    buf = buf.view(-1, 2)[offset: offset + x.size(0)]
    buf, _, _, _ = torch_npu.npu_moe_re_routing(buf, mat)
    y = x.new_empty(0)
    y.set_(buf.untyped_storage())
    return y


def quant_ffn(
    experts: NPUSharedFusedMoE,
    x_i8: torch.Tensor,  # [T, D] int8
    x_sc: torch.Tensor,  # [T] fp32
    hist: torch.Tensor,  # [N]
) -> torch.Tensor:
    assert x_i8.dtype == torch.int8
    assert x_sc.dtype == torch.float
    hist = hist.to(torch.int64)
    if hasattr(experts, "w13_weight_int4_scale"):
        return _quant_ffn_w4a8(experts, x_i8, x_sc, hist)
    return _quant_ffn_w8a8(experts, x_i8, x_sc, hist)


def _quant_ffn_w4a8(
    experts: NPUSharedFusedMoE,
    x_i8: torch.Tensor,  # [T, D] int8
    x_sc: torch.Tensor,  # [T] fp32
    hist: torch.Tensor,  # [N] int64
) -> torch.Tensor:
    asym = experts.w13_weight_offset is not None
    tuning_config = [0, 1, -1] if experts.quant_method.gmm_autotiling else None

    h = _w4a8_grouped_matmul(
        x_i8,
        experts.w13_weight,
        experts.w13_weight_bias,
        experts.w13_weight_int4_scale,
        experts.w13_weight_offset,
        x_sc,
        hist,
        tuning_config,
        asym,
    )
    h, h_sc = torch_npu.npu_dequant_swiglu_quant(
        h,
        weight_scale=None,
        activation_scale=None,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=hist,
        activate_left=True,
        quant_mode=1,
    )
    return _w4a8_grouped_matmul(
        h,
        experts.w2_weight,
        experts.w2_weight_bias,
        experts.w2_weight_int4_scale,
        experts.w2_weight_offset,
        h_sc,
        hist,
        tuning_config,
        asym,
    )


def _w4a8_grouped_matmul(
    x,
    weight,
    bias,
    scale,
    offset,
    token_scale,
    hist,
    tuning_config,
    asym,
):
    return torch.ops.custom.npu_ai_infra_grouped_matmul(
        [x],
        [weight],
        bias=[bias],
        scale=[scale],
        offset=[offset] if asym else None,
        antiquant_scale=None,
        antiquant_offset=None,
        per_token_scale=[token_scale.unsqueeze(-1) if asym else token_scale],
        group_list=hist,
        activation_input=None,
        activation_quant_scale=None,
        activation_quant_offset=None,
        split_item=3,
        group_type=0,
        group_list_type=1,
        act_type=0,
        tuning_config=tuning_config,
        output_dtype=torch.bfloat16,
    )[0]


def _quant_ffn_w8a8(
    experts: NPUSharedFusedMoE,
    x_i8: torch.Tensor,  # [T, D] int8
    x_sc: torch.Tensor,  # [T] fp32
    hist: torch.Tensor,  # [N]
) -> torch.Tensor:
    # int8 quantized MoE FFN: grouped_matmul → dequant_swiglu_quant → grouped_matmul
    w13 = experts.w13_weight
    s13 = experts.w13_weight_scale
    w2 = experts.w2_weight
    s2 = experts.w2_weight_scale
    assert s2.dtype == torch.bfloat16

    x_i32 = torch_npu.npu_grouped_matmul(
        [x_i8],
        [w13],
        bias=None,
        group_list=hist,
        split_item=3,
        output_dtype=torch.int32,
        group_type=0,
        group_list_type=1,
        act_type=0,
    )[0]
    x_i8, x_sc = torch_npu.npu_dequant_swiglu_quant(
        x=x_i32,
        weight_scale=s13,
        activation_scale=x_sc,
        bias=None,
        quant_offset=None,
        quant_scale=None,
        group_index=hist,
        activate_left=True,
        quant_mode=1,
    )
    y = torch_npu.npu_grouped_matmul(
        [x_i8],
        [w2],
        bias=None,
        scale=[s2],
        per_token_scale=[x_sc],
        group_list=hist,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]
    return y


def finalize_routing(
    x: torch.Tensor,      # [T*K]
    scales: torch.Tensor, # [T, K]
    reorg: torch.Tensor,  # [T*K]
) -> torch.Tensor:        # [T]
    return torch_npu.npu_moe_finalize_routing(
        x,
        skip1=None,
        skip2=None,
        bias=None,
        scales=scales,
        expanded_src_to_dst_row=reorg,
        export_for_source_row=None,
        drop_pad_mode=2,
    )
