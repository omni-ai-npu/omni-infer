# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Pangu-coupled smooth + INT8 quantize for one layer.

Single-device implementation of joint_smooth_and_quantize_layer, with the
distributed / ThreadPool branches dropped (single-path).

This file IS the Pangu-specific absorption topology — q_a_layernorm,
kv_b_proj head-interleaved V-rows, the un-quantized indexer.wq_b, pre_mlp_layernorm,
the router (mlp.gate), and per-expert down_proj. When a second model needs
jointfix, THIS is the file that moves into a backend.

Single-device routed-expert behaviour: each expert is searched individually,
then a MEDIAN (a,b) is applied to all of them. The per-expert-(a,b) apply only
exists on the distributed path.

Reads collectors by PanguBackend.install_stat_hooks's key convention:
    {pfx}.q_b_in  {pfx}.o_in  {pfx}.mlp_in
    {pfx}.exp{eid}_down_in  {pfx}.shared_down_in  {pfx}.dense_down_in
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from jointfix.core import profiling as _prof
from jointfix.core.primitives import select_write_quantize, should_quantize
from jointfix.core.stats import refresh_stats_after_smooth
from jointfix.methods.jointfix import (
    JointSearchConfig, joint_grid_search_ab, joint_grid_search_ab_distributed,
)
from jointfix.methods.smooth_search import make_smooth_scale


def _precompute_Y_ref(W_bf16: torch.Tensor, X_sample: torch.Tensor) -> torch.Tensor:
    """Y_ref = X @ W^T — BF16 ground truth for the output-recon objective."""
    return X_sample.float() @ W_bf16.float().T


def _refine_row_scales_for_weight_mse(
    int8_w: torch.Tensor, scale: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Least-squares row-scale refinement with a strict no-regression gate.

    INT8 codes remain unchanged, so the checkpoint/runtime format is identical.
    Each accepted row scale minimizes ||q*s-W||² for its fixed integer codes.
    """
    q = int8_w.float()
    w = target.float().to(q.device)
    old = scale.float().to(q.device)
    old_shape = old.shape
    old_row = old.reshape(old.shape[0], -1)[:, 0]
    denom = q.square().sum(dim=1).clamp_min(1e-20)
    candidate = (q * w).sum(dim=1) / denom
    candidate = candidate.clamp_min(1e-20)
    old_err_row = (q * old_row.unsqueeze(1) - w).square().mean(dim=1)
    new_err_row = (q * candidate.unsqueeze(1) - w).square().mean(dim=1)
    accept = torch.isfinite(candidate) & (new_err_row <= old_err_row)
    # Deployment stores BF16 scales. Re-evaluate after that exact rounding so
    # the strict gate describes the actual checkpoint, not an FP32 proxy.
    candidate_stored = candidate.to(scale.dtype).float()
    stored_err_row = (q * candidate_stored.unsqueeze(1) - w).square().mean(dim=1)
    accept = accept & (stored_err_row <= old_err_row)
    refined = torch.where(accept, candidate_stored, old_row)
    final_err_row = torch.where(accept, stored_err_row, old_err_row)
    norm = w.square().mean().clamp_min(1e-20)
    metrics = {
        "weight_nmse_before_scale_refine": float(old_err_row.mean().div(norm).item()),
        "weight_nmse_after_scale_refine": float(final_err_row.mean().div(norm).item()),
        "scale_refined_rows": int(accept.sum().item()),
        "scale_total_rows": int(accept.numel()),
    }
    return refined.reshape(old_shape).to(dtype=scale.dtype, device=scale.device), metrics


def _ab_converged(prev_ab: dict, cur_ab: dict, tol: float) -> bool:
    """True if every (a,b) shared by prev/cur moved by <= tol (K-iter early-exit)."""
    for k, (a, b) in cur_ab.items():
        if k in prev_ab:
            pa, pb = prev_ab[k]
            if abs(a - pa) > tol or abs(b - pb) > tol:
                return False
    return True


def _smooth_disabled_for_layer(config: JointSearchConfig, layer_idx: int) -> bool:
    return (config.skip_smooth_from_layer is not None
            and layer_idx >= config.skip_smooth_from_layer)


def smooth_quantize_pangu_layer(
    layer_tensors: Dict[str, torch.Tensor],
    collectors: dict,
    model_config: dict,
    layer_idx: int,
    config: JointSearchConfig,
    skip_patterns: List[str],
    device,
    devices=None,
) -> Tuple[Dict[str, torch.Tensor], int, int, dict]:
    """
    Apply joint (a,b) smooth search + INT8 quantize to one layer.

    Returns (out_tensors, n_smooth, n_quantized, traces). out_tensors holds int8
    weights + bf16 scales for quantized linears, and bf16 passthrough for the rest
    (incl. the smooth-modified norms / un-quantized consumers).

    `devices` (default [device]): with >1, the gate+up search distributes weights
    across devices and routed experts get PER-EXPERT (a,b) (the production
    multi-device path); with 1, gate+up is single-device and routed experts
    collapse to a MEDIAN (a,b).
    """
    pfx = f"model.layers.{layer_idx}"
    t = {k: v.clone().float() for k, v in layer_tensors.items()}
    n_smooth = 0
    traces: Dict[str, dict] = {}
    sd = device
    devices = devices or [device]
    n_dev = len(devices)

    def _sq(name: str) -> bool:
        return should_quantize(name, layer_tensors[name], skip_patterns)

    _layer_skip = _smooth_disabled_for_layer(config, layer_idx)
    _attn_skip = config.skip_smooth_attn or _layer_skip
    route_summary = None
    protected_experts = None
    route_key = f"{pfx}.moe_routes"
    if config.mdmixq_enabled and route_key in collectors:
        route_summary = collectors[route_key].finalize()
        n_experts = int(model_config.get("n_routed_experts", 256))
        n_text = max(1, min(
            n_experts, int(round(n_experts * config.mdmixq_text_expert_fraction))))
        text_rank = torch.argsort(
            route_summary["text_salience"], descending=True, stable=True)
        text_top = [
            int(eid) for eid in text_rank[:n_text]
            if route_summary["text_salience"][eid] > 0
        ]
        n_nontext = max(0, min(n_experts, int(config.mdmixq_nontext_topk)))
        nontext_rank = torch.argsort(
            route_summary["nontext_salience"], descending=True, stable=True)
        nontext_top = ([
            int(eid) for eid in nontext_rank[:n_nontext]
            if route_summary["nontext_salience"][eid] > 0
        ] if n_nontext else [])
        protected_experts = set(text_top) | set(nontext_top)
        traces[f"{pfx}.mdmixq"] = {
            "text_tokens": route_summary["text_tokens"],
            "nontext_tokens": route_summary["nontext_tokens"],
            "tau_text": route_summary["tau_text"],
            "tau_nontext": route_summary["tau_nontext"],
            "text_experts": text_top,
            "nontext_experts": nontext_top,
            "protected_expert_count": len(protected_experts),
        }
    elif config.mdmixq_enabled and model_config.get("n_routed_experts") and layer_idx >= int(
            model_config.get("first_k_dense_replace", 0)):
        raise RuntimeError(
            f"{pfx}: jointfix-mdmixq requires modality-aligned Omni calibration; "
            "no routed text/non-text statistics were collected"
        )
    if route_summary is not None and (
            route_summary["text_tokens"] + route_summary["nontext_tokens"] == 0):
        raise RuntimeError(
            f"{pfx}: jointfix-mdmixq route collector is empty; use --calib-format omni"
        )

    # ── 1. q_b_proj → absorb into q_a_layernorm (+ compensate indexer.wq_b) ──
    q_b_k = f"{pfx}.self_attn.q_b_proj.weight"
    qa_nk = f"{pfx}.self_attn.q_a_layernorm.weight"
    q_b_ask = f"{pfx}.q_b_in"
    can_smooth_q_b = (
        not _attn_skip
        and q_b_k in t
        and qa_nk in t
        and q_b_ask in collectors
        and _sq(q_b_k)
    )
    if can_smooth_q_b:
        stats = collectors[q_b_ask].finalize(sd)
        W = t[q_b_k].to(sd)
        w_stat = W.abs().amax(dim=0)
        if config.objective == "output-recon":
            stats["Y_refs"] = [_precompute_Y_ref(W, stats["X_sample"])]
        s, trace = joint_grid_search_ab([W], stats, w_stat, config)
        traces[q_b_k] = trace
        s = s.cpu()
        t[qa_nk] /= s
        t[q_b_k] *= s.unsqueeze(0)
        n_smooth += 1
        wq_b_k = f"{pfx}.self_attn.indexer.wq_b.weight"
        if wq_b_k in t:                       # un-quantized DSA consumer, still needs ×s
            t[wq_b_k] *= s.unsqueeze(0)

    # ── 1b. o_proj → absorb into kv_b_proj V-rows (head-interleaved) ──
    o_proj_k = f"{pfx}.self_attn.o_proj.weight"
    kv_b_k = f"{pfx}.self_attn.kv_b_proj.weight"
    o_in_ask = f"{pfx}.o_in"
    NH = model_config["num_attention_heads"]
    NOPE = model_config["qk_nope_head_dim"]
    VHD = model_config["v_head_dim"]
    can_smooth_o_proj = (
        not _attn_skip
        and o_proj_k in t
        and kv_b_k in t
        and o_in_ask in collectors
        and _sq(o_proj_k)
    )
    if can_smooth_o_proj:
        stats = collectors[o_in_ask].finalize(sd)
        W = t[o_proj_k].to(sd)
        w_stat = W.abs().amax(dim=0)
        if config.objective == "output-recon":
            stats["Y_refs"] = [_precompute_Y_ref(W, stats["X_sample"])]
        s, trace = joint_grid_search_ab([W], stats, w_stat, config)
        traces[o_proj_k] = trace
        s = s.cpu()
        traces[o_proj_k]["s"] = s
        traces[o_proj_k]["ask"] = o_in_ask
        t[o_proj_k] *= s.unsqueeze(0)
        for h in range(NH):                   # V-rows only, per head
            v_start = h * (NOPE + VHD) + NOPE
            v_end = v_start + VHD
            s_slice = s[h * VHD:(h + 1) * VHD]
            t[kv_b_k][v_start:v_end, :] /= s_slice.unsqueeze(1)
        n_smooth += 1

    # ── K-iter coordinate descent over (gate+up ↔ down) ──
    K = max(1, config.num_iterations)
    _stats_cache: Dict[str, dict] = {}

    def _get_stats(ask: str) -> dict:
        if ask not in _stats_cache:
            _stats_cache[ask] = collectors[ask].finalize(sd)
        return _stats_cache[ask]

    prev_ab: Dict[str, Tuple[float, float]] = {}
    for _it in range(K):
        n_smooth = 0   # diagnostic count only (reset per iteration)

        # ── 2. gate+up → absorb into pre_mlp_layernorm ──
        pre_mlp_k = f"{pfx}.pre_mlp_layernorm.weight"
        if pre_mlp_k not in t:
            pre_mlp_k = None
        mlp_in_ask = f"{pfx}.mlp_in"
        if (pre_mlp_k is not None and mlp_in_ask in collectors
                and not (config.skip_smooth_gate_up or _layer_skip)):
            stats = _get_stats(mlp_in_ask)
            all_gate_up_keys: List[str] = []
            gate_up_keys: List[str] = []
            for proj in ("gate_proj", "up_proj"):
                for owner in (f"{pfx}.mlp", f"{pfx}.mlp.shared_experts"):
                    k = f"{owner}.{proj}.weight"
                    if k in t and _sq(k):
                        all_gate_up_keys.append(k)
                        gate_up_keys.append(k)
                for eid in range(model_config.get("n_routed_experts", 256)):
                    k = f"{pfx}.mlp.experts.{eid}.{proj}.weight"
                    if k in t and _sq(k):
                        all_gate_up_keys.append(k)
                        if protected_experts is None or eid in protected_experts:
                            gate_up_keys.append(k)

            if gate_up_keys:
                if n_dev > 1:
                    # distribute gate+up weights round-robin across devices
                    w_stat = torch.stack(
                        [t[k].abs().amax(dim=0) for k in gate_up_keys]).amax(dim=0).to(sd)
                    weight_groups = [[] for _ in range(n_dev)]
                    for i, k in enumerate(gate_up_keys):
                        weight_groups[i % n_dev].append(t[k].to(devices[i % n_dev]))
                    s, trace = joint_grid_search_ab_distributed(
                        weight_groups, devices, stats, w_stat, config)
                else:
                    search_weights = [t[k].to(sd) for k in gate_up_keys]
                    w_stat = torch.stack([W.abs().amax(dim=0) for W in search_weights]).amax(dim=0)
                    if config.objective == "output-recon":
                        stats["Y_refs"] = [_precompute_Y_ref(W, stats["X_sample"]) for W in search_weights]
                    s, trace = joint_grid_search_ab(search_weights, stats, w_stat, config)
                traces[pre_mlp_k] = trace
                s = s.cpu()
                t[pre_mlp_k] /= s
                for k in all_gate_up_keys:
                    t[k] *= s.unsqueeze(0)
                n_smooth += len(all_gate_up_keys)
                # shared experts not searched but share the norm input
                for proj in ("gate_proj", "up_proj"):
                    sk = f"{pfx}.mlp.shared_experts.{proj}.weight"
                    if sk in t and sk not in gate_up_keys:
                        t[sk] *= s.unsqueeze(0)
                # router reads pre_mlp_layernorm too — un-quantized, still ×s
                # (missing this caused ~40% routing flip)
                gate_router_k = f"{pfx}.mlp.gate.weight"
                if gate_router_k in t:
                    t[gate_router_k] *= s.unsqueeze(0)

        # ── 3. down_proj → absorb s^{-1} into up_proj rows ──
        _down_skip = config.skip_smooth_down or _layer_skip

        def _smooth_down(
            dk: str,
            uk: str,
            ask: str,
            down_skip: bool = _down_skip,
            iteration: int = _it,
        ) -> None:
            nonlocal n_smooth
            if down_skip:
                return
            required_data_available = dk in t and uk in t and ask in collectors
            if not required_data_available:
                return
            if not _sq(dk):
                return
            st = _get_stats(ask)
            W = t[dk].to(sd)
            w_stat_i = W.abs().amax(dim=0)
            if config.objective == "output-recon":
                st = dict(st)
                st["Y_refs"] = [_precompute_Y_ref(W, st["X_sample"])]
            s, trace = joint_grid_search_ab([W], st, w_stat_i, config)
            traces[dk] = trace
            s = s.cpu()
            traces[dk]["s"] = s
            traces[dk]["ask"] = ask
            t[uk] /= s.unsqueeze(1)
            t[dk] *= s.unsqueeze(0)
            if iteration < K - 1:
                _stats_cache[ask] = refresh_stats_after_smooth(_stats_cache[ask], s.to(sd))
            n_smooth += 1

        _smooth_down(f"{pfx}.mlp.down_proj.weight",
                     f"{pfx}.mlp.up_proj.weight", f"{pfx}.dense_down_in")
        _smooth_down(f"{pfx}.mlp.shared_experts.down_proj.weight",
                     f"{pfx}.mlp.shared_experts.up_proj.weight", f"{pfx}.shared_down_in")

        # Routed experts: search each individually, then apply MEDIAN (a,b) to all
        # (single-device behaviour — see module docstring).
        n_routed = model_config.get("n_routed_experts", 256)
        routed = []
        for eid in range(n_routed):
            ep = f"{pfx}.mlp.experts.{eid}"
            dk, uk, ask = f"{ep}.down_proj.weight", f"{ep}.up_proj.weight", f"{pfx}.exp{eid}_down_in"
            can_smooth_expert = dk in t and uk in t and ask in collectors and _sq(dk)
            if can_smooth_expert:
                routed.append((eid, dk, uk, ask))

        if routed and not _down_skip:
            # ── search each expert's (a,b) ──
            _prof.reset_peak(devices)        # profile (opt-in): time + peak HBM of
            _t_routed = _prof.clock()        # the per-expert search — sizes batching
            expert_traces: Dict[str, dict] = {}
            if n_dev > 1:
                # distribute experts round-robin across devices (parallel search)
                from concurrent.futures import ThreadPoolExecutor
                for _, _, _, ask in routed:
                    _get_stats(ask)               # pre-warm cache (avoid ThreadPool race)

                def _search_expert(arg):
                    i, (eid, dk, uk, ask) = arg
                    dev = devices[i % n_dev]
                    st = {k: (v.to(dev) if torch.is_tensor(v) else v)
                          for k, v in _get_stats(ask).items()}
                    W = t[dk].to(dev)
                    w_stat_i = W.abs().amax(dim=0)
                    if config.objective == "output-recon":
                        st["Y_refs"] = [_precompute_Y_ref(W, st["X_sample"])]
                    _, tr = joint_grid_search_ab([W], st, w_stat_i, config)
                    return dk, tr

                with ThreadPoolExecutor(max_workers=n_dev) as ex:
                    for dk_r, tr_r in ex.map(_search_expert, enumerate(routed)):
                        expert_traces[dk_r] = tr_r
            else:
                for _, dk, _, ask in routed:
                    st = dict(_get_stats(ask))
                    W = t[dk].to(sd)
                    w_stat_i = W.abs().amax(dim=0)
                    if config.objective == "output-recon":
                        st["Y_refs"] = [_precompute_Y_ref(W, st["X_sample"])]
                    _, tr = joint_grid_search_ab([W], st, w_stat_i, config)
                    expert_traces[dk] = tr
            traces.update(expert_traces)
            _prof.emit_routed_down(layer_idx, routed, t, _get_stats, devices,
                                   _t_routed, config)

            # ── apply: per-expert (a,b) on multi-device; MEDIAN on single-device ──
            if n_dev == 1:
                a_list = [tr["a"] for tr in expert_traces.values()]
                b_list = [tr["b"] for tr in expert_traces.values()]
                median_a = sorted(a_list)[len(a_list) // 2]
                median_b = sorted(b_list)[len(b_list) // 2]

            for _, dk, uk, ask in routed:
                if n_dev > 1:
                    expert_trace = expert_traces[dk]
                    a_opt, b_opt = expert_trace["a"], expert_trace["b"]
                else:
                    a_opt, b_opt = median_a, median_b
                x_stat = _get_stats(ask)["amax"]
                W_dev = t[dk].to(sd)
                w_stat_i = W_dev.abs().amax(dim=0)
                s = make_smooth_scale(x_stat, w_stat_i, a_opt, b_opt).cpu()
                traces.setdefault(dk, {})["s"] = s
                traces[dk]["ask"] = ask
                t[uk] /= s.unsqueeze(1)
                t[dk] *= s.unsqueeze(0)
                if _it < K - 1 and ask in _stats_cache:
                    _stats_cache[ask] = refresh_stats_after_smooth(_stats_cache[ask], s.to(sd))
                n_smooth += 1

        # K-iter convergence early-exit
        curr_ab = {k: (v["a"], v["b"]) for k, v in traces.items()
                   if isinstance(v, dict) and "a" in v and "b" in v}
        if _it > 0 and _ab_converged(prev_ab, curr_ab, config.iter_ab_tol):
            break
        prev_ab = curr_ab

    # ── 4. INT8 quantize (output-side → GPTQ/RTN; input-side → RTN) ──
    out: Dict[str, torch.Tensor] = {}
    n_quantized = 0

    def _is_write_end(name: str) -> bool:
        if "self_attn.o_proj.weight" in name:
            return True
        if "mlp.down_proj.weight" in name and ".experts." not in name:
            return True
        if "mlp.shared_experts.down_proj.weight" in name:
            return True
        if ".experts." in name and ".down_proj.weight" in name:
            return True
        return False

    for name, orig in sorted(layer_tensors.items()):
        if should_quantize(name, orig, skip_patterns):
            X_smooth_for_gptq = None
            if _is_write_end(name) and name in traces:
                tr = traces[name]
                s_stored, ask_stored = tr.get("s"), tr.get("ask")
                if s_stored is not None and ask_stored is not None and ask_stored in collectors:
                    X_sample = collectors[ask_stored].finalize(sd)["X_sample"]
                    X_smooth_for_gptq = X_sample.float() / s_stored.to(X_sample.device).clamp(min=1e-12)
            # GPTQ with no X_smooth (trace/collector missing) -> silent RTN inside
            # select_write_quantize; its absence from [GPTQ-RUN] is the signal.
            int8_w, scale = select_write_quantize(
                t[name], X_smooth_for_gptq,
                write_quant=config.write_quant, gptq_damp=config.gptq_damp,
                gptq_block_size=config.gptq_block_size, tag=name)
            if config.mdmixq_enabled:
                scale, weight_metrics = _refine_row_scales_for_weight_mse(
                    int8_w, scale, t[name])
                traces.setdefault(name, {}).update(weight_metrics)
            out[name] = int8_w
            out[name.replace(".weight", ".weight_scale")] = scale
            n_quantized += 1
        else:
            out[name] = t[name].to(orig.dtype)

    return out, n_smooth, n_quantized, traces
