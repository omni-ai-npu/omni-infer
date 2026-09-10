# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
JointFix — the W8A8 quantization method.

joint (a,b) smooth-scale search + output-reconstruction objective + K-iter
coordinate descent + output-side GPTQ. The first concrete QuantMethod.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from jointfix.backends.base import LayerSpec, ModelBackend
from jointfix.core.stats import StatsConfig
from jointfix.methods.base import QuantMethod
from jointfix.methods.smooth_search import (
    compute_output_recon_objective,
    compute_output_recon_objective_batched_ab,
    compute_output_recon_objective_batched_ab_distributed,
    make_smooth_scale,
)
from jointfix.registry import register_method


@dataclass
class JointSearchConfig:
    """
    Hyperparameters for joint (a,b) SmoothQuant search.

    Lives with the method (not in core) — core/ never imports this.
    Stat-collection knobs are projected out via `stats_config()` so AccumActStats
    stays method-agnostic.
    """

    lam: float = 1.0
    gamma: float = 1.0
    n_hist_bins: int = 256
    hist_min: float = 1e-8
    hist_max: float = 1e4
    sink_exclude: int = 1
    sample_limit: int = 128
    tokens_per_sample: int = 8
    stage1_grid: list = field(default_factory=lambda: [
        (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
        (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1),
    ])
    stage2_radius: float = 0.2
    stage2_step: float = 0.1
    eps: float = 1e-12
    # channel weighting + output-side quantizer
    channel_weight: str = "hessian"
    hessian_alpha: float = 1.0
    write_quant: str = "gptq"
    gptq_damp: float = 0.01
    gptq_block_size: int = 128
    objective: str = "output-recon"
    # early-skip heuristics (both disabled by default)
    early_skip_enabled: bool = True
    early_skip_l1_enabled: bool = False
    early_skip_l2_enabled: bool = False
    early_skip_l1_factor: float = 1.0
    early_skip_l2_factor: float = 1.0
    ab_chunk: int = 9
    batched_ab_enabled: bool = True
    batched_ab_distributed_enabled: bool = True
    ab_chunk_distributed: int = 25
    # per-NPU batched expert search
    expert_batch_enabled: bool = True
    # K-iter coordinate descent across s_gate_up <-> s_down
    num_iterations: int = 1
    iter_ab_tol: float = 0.05
    # ablation switches (default False -> no behaviour change)
    skip_smooth_down: bool = False
    skip_smooth_gate_up: bool = False
    skip_smooth_attn: bool = False
    skip_smooth_from_layer: Optional[int] = None
    # leave the shared experts un-quantized (adds "mlp.shared_experts" to the
    # skip patterns when --skip-shared-experts is set)
    skip_shared_experts: bool = False
    # MDMixQ-guided uniform W8A8. These knobs affect calibration/search only;
    # checkpoint precision and runtime kernels remain identical to JointFix.
    mdmixq_enabled: bool = False
    mdmixq_text_sample_ratio: float = 0.8
    mdmixq_text_expert_fraction: float = 0.25
    mdmixq_nontext_topk: int = 8
    mdmixq_candidate_multiplier: int = 4

    def stats_config(self) -> StatsConfig:
        """Project the stat-collection knobs into a method-agnostic StatsConfig."""
        _needs_channel_weight = (self.objective == "weight-error")
        return StatsConfig(
            n_hist_bins=self.n_hist_bins,
            hist_min=self.hist_min,
            hist_max=self.hist_max,
            sample_limit=self.sample_limit,
            tokens_per_sample=self.tokens_per_sample,
            # Channel weights (outlier->p99/median histogram, hessian->E[x²] moments) are
            # only consumed by the weight-error objective. output-recon is a pure output
            # MSE and reads neither, so collect only the stat the active config can use —
            # for the default (output-recon) that means amax + sample cache only.
            collect_histogram=_needs_channel_weight and self.channel_weight == "outlier",
            collect_moments=_needs_channel_weight and self.channel_weight == "hessian",
            modality_aware=self.mdmixq_enabled,
            text_sample_ratio=self.mdmixq_text_sample_ratio,
            modality_candidate_multiplier=self.mdmixq_candidate_multiplier,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage (a,b) search driver
# ─────────────────────────────────────────────────────────────────────────────
def joint_grid_search_ab(
    weights: list,
    stats: dict,
    w_stat: torch.Tensor,
    config: JointSearchConfig,
    warm_start: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Two-stage (a,b) search via the output-reconstruction objective.

    Returns (smooth_scale [h_in], trace_dict). Output-recon path only (the
    default objective); the weight-error path is not yet implemented.

    `warm_start` (a0,b0) seeds K-iter coordinate descent: Stage 1 is skipped and
    Stage 2 refines around (a0,b0). warm_start=None does a full two-stage search.

    Invariant (the search never regresses): a safety net forces s=1 (a=b=0)
    whenever J(s=1) beats every searched candidate.
    """
    x_stat = stats["amax"]
    Y_refs = stats.get("Y_refs") if config.objective == "output-recon" else None
    if config.objective != "output-recon" or not Y_refs or len(Y_refs) != len(weights):
        raise NotImplementedError(
            "joint_grid_search_ab: only the output-recon objective is "
            "implemented; the weight-error objective path is not yet available.")

    def J(a: float, b: float) -> float:
        return compute_output_recon_objective(
            weights, Y_refs, stats["X_sample"], x_stat, w_stat, a, b, config.eps)

    def J_batch(a_list, b_list):
        if not a_list:
            return []
        if not config.batched_ab_enabled:
            return [J(a, b) for a, b in zip(a_list, b_list)]
        a_t = torch.tensor(a_list, dtype=torch.float32)
        b_t = torch.tensor(b_list, dtype=torch.float32)
        J_t = compute_output_recon_objective_batched_ab(
            weights, Y_refs, stats["X_sample"], x_stat, w_stat,
            a_t, b_t, eps=config.eps, ab_chunk=config.ab_chunk)
        return J_t.detach().cpu().tolist()

    return _two_stage_ab_search(J, J_batch, x_stat, w_stat, config, warm_start)


def _two_stage_ab_search(J, J_batch, x_stat, w_stat, config, warm_start=None):
    """
    Shared two-stage (a,b) search: coarse anti-diagonal grid -> fine 2D box, with
    L1/L2 early-skip (default off) and the safety net (never worse than s=1). `J` /
    `J_batch` are the objective evaluators — single-device or distributed — so this
    logic is identical for both paths. Returns (smooth_scale, trace).
    """
    J_at_zero = J(0.0, 0.0)
    search_skipped = "none"

    # L1 skip: probe (0.5,0.5) — DEFAULT DISABLED (single-point probe misfires).
    if config.early_skip_enabled and config.early_skip_l1_enabled:
        J_probe = J(0.5, 0.5)
        if J_probe >= J_at_zero * config.early_skip_l1_factor:
            s = make_smooth_scale(x_stat, w_stat, 0.0, 0.0, config.eps)
            return s, {"a": 0.0, "b": 0.0, "J_final": J_at_zero,
                       "a_stage1": 0.0, "b_stage1": 0.0, "J_stage1": J_at_zero,
                       "J_probe_55": J_probe, "J_at_zero": J_at_zero,
                       "search_skipped": "stage1+2"}

    # Stage 1: anti-diagonal grid (or warm-start seed).
    if warm_start is not None:
        a1, b1 = warm_start
        best_s1, best_Js1, _t_stage1 = (a1, b1), J(a1, b1), 0.0
    else:
        a_list = [ab[0] for ab in config.stage1_grid]
        b_list = [ab[1] for ab in config.stage1_grid]
        _t_s1 = time.time()
        J_list = J_batch(a_list, b_list)
        _t_stage1 = time.time() - _t_s1
        best_idx = int(min(range(len(J_list)), key=lambda i: J_list[i]))
        best_Js1 = J_list[best_idx]
        best_s1 = (a_list[best_idx], b_list[best_idx])
        a1, b1 = best_s1

    # L2 skip: DEFAULT DISABLED (stage1 anti-diagonal ≠ predictor of stage2 box).
    if (config.early_skip_enabled and config.early_skip_l2_enabled
            and best_Js1 >= J_at_zero * config.early_skip_l2_factor):
        if J_at_zero < best_Js1:
            best_final, best_Jfinal = (0.0, 0.0), J_at_zero
        else:
            best_final, best_Jfinal = best_s1, best_Js1
        s = make_smooth_scale(x_stat, w_stat, best_final[0], best_final[1], config.eps)
        return s, {"a": best_final[0], "b": best_final[1], "J_final": best_Jfinal,
                   "a_stage1": a1, "b_stage1": b1, "J_stage1": best_Js1,
                   "J_at_zero": J_at_zero, "search_skipped": "stage2"}

    # Stage 2: local 2D box refinement around (a1,b1).
    radius, step = config.stage2_radius, config.stage2_step
    n = int(2 * radius / step) + 1 if step > 0 else 1
    a_grid = [max(0.0, min(1.0, a1 + (i - n // 2) * step)) for i in range(n)]
    b_grid = [max(0.0, min(1.0, b1 + (i - n // 2) * step)) for i in range(n)]
    stage2_a_list, stage2_b_list = [], []
    for a in a_grid:
        for b in b_grid:
            stage2_a_list.append(a)
            stage2_b_list.append(b)
    _t_s2 = time.time()
    J_list2 = J_batch(stage2_a_list, stage2_b_list)
    _t_stage2 = time.time() - _t_s2

    best_final, best_Jfinal = best_s1, best_Js1
    for idx, J_val in enumerate(J_list2):
        if J_val < best_Jfinal:
            best_Jfinal, best_final = J_val, (stage2_a_list[idx], stage2_b_list[idx])

    # Safety net: the grids never include (0,0); if s=1 beats the searched best, take it.
    if J_at_zero < best_Jfinal:
        best_final, best_Jfinal = (0.0, 0.0), J_at_zero

    s = make_smooth_scale(x_stat, w_stat, best_final[0], best_final[1], config.eps)
    return s, {"a": best_final[0], "b": best_final[1], "J_final": best_Jfinal,
               "a_stage1": a1, "b_stage1": b1, "J_stage1": best_Js1,
               "t_stage1_s": _t_stage1, "t_stage2_s": _t_stage2,
               "n_stage2_candidates": len(stage2_a_list),
               "J_at_zero": J_at_zero, "search_skipped": search_skipped}


def joint_grid_search_ab_distributed(weight_groups, devices, stats, w_stat,
                                     config, warm_start=None):
    """
    Distributed two-stage (a,b) search (output-recon only).

    weight_groups[i] live on devices[i]; stats / w_stat on devices[0]. Builds
    Y_refs per device (resident there) and reduces the objective across devices via
    compute_output_recon_objective_batched_ab_distributed. Same two-stage logic as
    the single-device driver — only the objective evaluator differs. Returns
    (smooth_scale [h_in] on devices[0], trace); the caller .cpu()s it.
    """
    from concurrent.futures import ThreadPoolExecutor

    x_stat = stats["amax"]
    if config.objective != "output-recon":
        raise NotImplementedError("distributed search: output-recon objective only")

    Y_refs_per_dev = [None] * len(weight_groups)

    def _build_yref(idx):
        dev = devices[idx]
        X_dev = stats["X_sample"].to(dev).float()
        Y_refs_per_dev[idx] = [X_dev @ W.to(dev).float().T for W in weight_groups[idx]]

    if len(devices) == 1:
        _build_yref(0)
    else:
        with ThreadPoolExecutor(max_workers=len(devices)) as ex:
            list(ex.map(_build_yref, range(len(devices))))

    ab_chunk = (config.ab_chunk_distributed if config.ab_chunk_distributed > 0
                else config.ab_chunk)

    def J_batch(a_list, b_list):
        if not a_list:
            return []
        a_t = torch.tensor(a_list, dtype=torch.float32)
        b_t = torch.tensor(b_list, dtype=torch.float32)
        J_t = compute_output_recon_objective_batched_ab_distributed(
            weight_groups, Y_refs_per_dev, devices, stats["X_sample"], x_stat, w_stat,
            a_t, b_t, eps=config.eps, ab_chunk=ab_chunk)
        return J_t.detach().cpu().tolist()

    def J(a, b):
        return J_batch([a], [b])[0]

    return _two_stage_ab_search(J, J_batch, x_stat, w_stat, config, warm_start)


@register_method("jointfix")
class JointFixMethod(QuantMethod):
    name = "jointfix"
    needs_activations = True

    def __init__(self, config: Optional[JointSearchConfig] = None):
        self.config = config or JointSearchConfig()
        self._traces: Dict[str, dict] = {}     # weight_name -> search trace (a,b,J,...)

    def stats_config(self):
        return self.config.stats_config()

    def configure(self, args) -> None:
        """
        Apply parsed CLI args onto JointSearchConfig (argparse dest names match
        the config field names).
        """
        for field_name in ("objective", "write_quant", "channel_weight", "num_iterations",
                           "iter_ab_tol", "lam", "gamma", "gptq_damp", "gptq_block_size",
                           "skip_shared_experts"):
            if hasattr(args, field_name):
                setattr(self.config, field_name, getattr(args, field_name))

    def add_cli_args(self, parser) -> None:
        g = parser.add_argument_group("jointfix: joint (a,b) search")
        g.add_argument("--objective", choices=["weight-error", "output-recon"],
                       default="output-recon")
        g.add_argument("--write-quant", choices=["rtn", "gptq"], default="gptq")
        g.add_argument("--channel-weight", choices=["outlier", "hessian"],
                       default="hessian")
        g.add_argument("--num-iterations", type=int, default=1)
        g.add_argument("--iter-ab-tol", type=float, default=0.05)
        g.add_argument("--skip-shared-experts", action="store_true", default=False,
                       help="leave the shared experts un-quantized (BF16)")
        g.add_argument("--lam", type=float, default=1.0)
        g.add_argument("--gamma", type=float, default=1.0)
        g.add_argument("--gptq-damp", type=float, default=0.01)
        g.add_argument("--gptq-block-size", type=int, default=128)
        # (remaining v4/v6 knobs added during the search-math migration)

    def process_layer(self, layer_tensors, collectors, spec: LayerSpec,
                      backend: ModelBackend, device, devices=None) -> Dict[str, object]:
        # Lazy import breaks the cycle (jointfix -> _smooth_pangu -> jointfix).
        # jointfix absorption is Pangu-coupled in v1: backend is
        # expected to be a PanguBackend (provides config()/skip_patterns()).
        from jointfix.methods._smooth_pangu import smooth_quantize_pangu_layer

        skip = backend.skip_patterns()
        if self.config.skip_shared_experts:
            skip = skip + ["mlp.shared_experts"]   # -> un-quantized + not smoothed
        out, _n_smooth, _n_quant, _traces = smooth_quantize_pangu_layer(
            layer_tensors, collectors, backend.config(), spec.layer_idx,
            self.config, skip, device, devices=devices,
        )
        self._traces.update(_traces)           # keep (a,b)/J per weight for the trace dump
        return out

    def dump_traces(self, out_dir) -> None:
        """
        Write joint_search_traces.json (weight_name -> (a,b,J,...)), the same artifact
        the monolith writes — so tools/compare_joint_traces.py can diff the searched (a,b)
        between a jointfix run and a monolith run. Tensor fields (e.g. the 's' scale) are
        dropped to keep the file small; the (a,b)/J scalars are what the diff compares.
        """
        if not self._traces:
            return
        import json
        from pathlib import Path
        scalar = {name: {k: v for k, v in tr.items() if not torch.is_tensor(v)}
                  for name, tr in self._traces.items()}
        path = Path(out_dir) / "joint_search_traces.json"
        with open(path, "w") as f:
            json.dump(scalar, f, indent=2)
        print(f"  search traces       : {path}  ({len(scalar)} groups)")
