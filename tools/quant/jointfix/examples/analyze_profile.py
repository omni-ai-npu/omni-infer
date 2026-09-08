#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Turn a JOINTFIX_PROFILE run log into the table that sizes the batched expert search.

Usage:
    JOINTFIX_PROFILE=1 jointfix quantize ... 2>&1 | tee run.log
    python examples/analyze_profile.py run.log [--peak-tflops 100] [--hbm-frac 0.5]

It reads the ``[PROF] {json}`` lines emitted by ``jointfix.core.profiling`` and reports:

  Q1  launch-bound vs compute-bound — achieved TFLOP/s of the per-expert search vs the
      device peak. Low utilisation => the device idles between many small launches =>
      fusing experts wins big. High utilisation => GEMMs already saturate it => small win.
  Q2  token raggedness across experts — decides ragged-token handling: near-uniform means
      a plain stack works; a wide spread means bucket equal-length experts or zero-pad+mask.
  Q3  a suggested initial (expert_chunk x ab_chunk) tile that fits the measured free HBM.

The arithmetic lives in pure functions (`analyze`, `suggest_tile`) so it is unit-tested on
CPU without an NPU — see tests/test_profiling.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

_LOGGER = logging.getLogger(__name__)


def parse_prof_lines(text: str) -> List[dict]:
    """Extract the JSON payload of every '[PROF] {...}' line in a run log."""
    records = []
    for line in text.splitlines():
        i = line.find("[PROF] ")
        if i == -1:
            continue
        try:
            records.append(json.loads(line[i + len("[PROF] "):]))
        except json.JSONDecodeError as error:
            _LOGGER.debug("ignoring malformed profiling record: %s", error)
            continue
    return records


def _quantile(sorted_vals: List[float], q: float) -> float:
    """Nearest-rank quantile of an already-sorted, non-empty list."""
    n = len(sorted_vals)
    idx = min(n - 1, max(0, int(q * (n - 1) + 0.5)))
    return sorted_vals[idx]


def _min_free_bytes(free_gb: List[Optional[float]]) -> Optional[float]:
    vals = [f for f in (free_gb or []) if f is not None]
    return min(vals) * 1e9 if vals else None


def layer_flop(rec: dict) -> float:
    """
    Matmul FLOPs of one layer's per-expert search: every expert evaluates
    n_candidates (a,b) points (plus one Y_ref build) via a [tok,h_in]x[h_in,h_out] GEMM.
    """
    h_in, h_out = rec.get("h_in") or 0, rec.get("h_out") or 0
    n_cand = rec.get("n_candidates") or 0
    return sum(2 * tok * h_in * h_out * (n_cand + 1) for tok in rec.get("toks", []))


def suggest_tile(h_in: int, h_out: int, tok_max: int, free_bytes: float,
                 budget_frac: float, n_experts: int, n_candidates: int) -> dict:
    """
    Largest (expert_chunk, ab_chunk) whose fp32 intermediates fit budget_frac*free.

    Per (expert, candidate) the search materialises the smoothed+quantised weight copy
    [h_out,h_in], the smoothed activation [tok,h_in] and the GEMM output [tok,h_out];
    the weight copy usually dominates, which is why ab_chunk must be bounded too.
    """
    pair_bytes = 4 * (h_out * h_in + tok_max * h_in + tok_max * h_out)
    budget = budget_frac * free_bytes
    max_pairs = int(budget // pair_bytes)
    if max_pairs < 1:
        return {"expert_chunk": 1, "ab_chunk": 1, "pair_bytes": pair_bytes,
                "tile_bytes": pair_bytes, "note": "one (expert,candidate) pair already "
                "exceeds the budget — drop to fp16 intermediates or tile tok"}
    ab_chunk = min(25, n_candidates, max_pairs)            # stage-2 box size, the larger stage
    expert_chunk = max(1, min(n_experts, max_pairs // ab_chunk))
    return {"expert_chunk": expert_chunk, "ab_chunk": ab_chunk, "pair_bytes": pair_bytes,
            "tile_bytes": expert_chunk * ab_chunk * pair_bytes, "note": ""}


def analyze(records: List[dict], peak_tflops: float = 100.0,
            hbm_budget_frac: float = 0.5) -> dict:
    """Pure summary of parsed [PROF] records — the unit-tested core of the report."""
    routed = [r for r in records if r.get("kind") == "routed_down"]
    layers = [r for r in records if r.get("kind") == "layer"]

    per_layer = []
    for r in routed:
        toks = sorted(r.get("toks", []))
        wall = r.get("wall_s") or 0.0
        flop = layer_flop(r)
        tflops = (flop / wall / 1e12) if wall > 0 else 0.0
        free_b = _min_free_bytes(r.get("free_gb"))
        tile = (suggest_tile(r["h_in"], r["h_out"], toks[-1] if toks else 0, free_b,
                             hbm_budget_frac, r.get("n_experts", len(toks)),
                             r.get("n_candidates", 34))
                if free_b and toks else None)
        per_layer.append({
            "layer": r.get("layer"),
            "n_experts": r.get("n_experts"),
            "h_in": r.get("h_in"), "h_out": r.get("h_out"),
            "tok_min": toks[0] if toks else 0,
            "tok_p50": _quantile(toks, 0.5) if toks else 0,
            "tok_max": toks[-1] if toks else 0,
            "wall_s": wall, "tflops": tflops, "util": tflops / peak_tflops if peak_tflops else 0.0,
            "peak_gb": max([p for p in (r.get("peak_gb") or []) if p is not None], default=None),
            "free_gb": (free_b / 1e9) if free_b else None,
            "tile": tile,
        })

    util_vals = [pl["util"] for pl in per_layer if pl["wall_s"] > 0]
    mean_util = sum(util_vals) / len(util_vals) if util_vals else 0.0
    if mean_util < 0.25:
        q1 = f"LAUNCH-BOUND (mean util {mean_util:.1%}) — fusing experts likely 2-4x on search"
    elif mean_util > 0.5:
        q1 = f"COMPUTE-BOUND (mean util {mean_util:.1%}) — GEMMs already saturate; batching ~1.2-1.5x"
    else:
        q1 = f"MIXED (mean util {mean_util:.1%}) — expect ~1.5-2x on search"

    ragged = max((pl["tok_max"] / pl["tok_p50"] for pl in per_layer if pl["tok_p50"]),
                 default=1.0)
    q2 = (f"raggedness max(tok_max/tok_p50) = {ragged:.1f}x — "
          + ("near-uniform; plain stack OK" if ragged < 1.5
             else "wide spread; bucket equal-length experts or zero-pad+mask"))

    constraining = min((pl for pl in per_layer if pl["tile"]),
                       key=lambda pl: pl["tile"]["expert_chunk"], default=None)
    q3 = (f"suggested initial tile: expert_chunk={constraining['tile']['expert_chunk']} x "
          f"ab_chunk={constraining['tile']['ab_chunk']} "
          f"(layer {constraining['layer']}, {hbm_budget_frac:.0%} of "
          f"{constraining['free_gb']:.1f}GB free)" if constraining
          else "no HBM info in log — run with JOINTFIX_PROFILE on an NPU/CUDA device")

    search_frac = None
    if layers:
        tot = sum((layer.get("t_forward_s", 0) + layer.get("t_search_quant_s", 0)
                   + layer.get("t_reforward_s", 0)) for layer in layers)
        sq = sum(layer.get("t_search_quant_s", 0) for layer in layers)
        search_frac = sq / tot if tot > 0 else None

    return {"per_layer": per_layer, "mean_util": mean_util, "q1": q1, "q2": q2, "q3": q3,
            "search_fraction": search_frac, "n_layers_profiled": len(routed),
            "phases": phase_breakdown(layers)}


_PHASES = [("load", "t_load_s"), ("forward", "t_forward_s"),
           ("search+q", "t_search_quant_s"), ("reforward", "t_reforward_s")]

_PHASE_HINT = {
    "forward": "BF16 stat-collection forward. stat_oh = forward - reforward isolates the hook "
               "cost; if stat_oh is still large after the histogram skip, the remaining "
               "sum_x2/amax/sample or per-hook overhead is the next target.",
    "load": "weight load from disk (SFS/network IO) — the 16 cards sit idle here. Fix is IO: "
            "prefetch / overlap load(layer N+1) with compute(layer N) / faster storage.",
    "reforward": "the 2nd, hook-free forward for error propagation. --skip-reforward removes it "
                 "(trades a per-layer accuracy check for roughly this much time).",
    "search+q": "(a,b) search + INT8 quantize. expert-search batching was rejected (~2%); the "
                "GPTQ quantize is the bulk here.",
}


def phase_breakdown(layer_records: List[dict]) -> dict:
    """
    Per-layer load/forward/search+quant/reforward split — surfaces the dominant phase
    (the next optimization target). `stat_oh = forward - reforward` uses the hook-free
    re-forward as the floor, so it isolates whatever stat-collection overhead remains.
    """
    rows, totals = [], {name: 0.0 for name, _ in _PHASES}
    for r in layer_records:
        vals = {name: float(r.get(key, 0.0) or 0.0) for name, key in _PHASES}
        rows.append({"layer": r.get("layer"), **vals,
                     "stat_oh": vals["forward"] - vals["reforward"],
                     "total": sum(vals.values())})
        for name in totals:
            totals[name] += vals[name]
    fwds = sorted(row["forward"] for row in rows)
    med_fwd = fwds[len(fwds) // 2] if fwds else 0.0
    for row in rows:
        row["warmup"] = med_fwd > 0 and row["forward"] > 3 * med_fwd
    grand = sum(totals.values())
    return {"rows": rows, "totals": totals, "grand": grand, "median_forward": med_fwd,
            "dominant": max(totals, key=totals.get) if grand > 0 else None}


def format_report(result: dict) -> str:
    lines = []
    ph = result.get("phases")
    if ph and ph["rows"]:
        lines += ["", "=== per-layer phase time (s) — where the layer wall actually goes ===",
                  f"{'layer':>5} {'load':>9} {'forward':>9} {'search+q':>9} "
                  f"{'reforward':>9} {'stat_oh':>9} {'total':>9}"]
        for row in ph["rows"]:
            flag = "  *warmup" if row["warmup"] else ""
            lines.append(f"{row['layer']:>5} {row['load']:>9.1f} {row['forward']:>9.1f} "
                         f"{row['search+q']:>9.1f} {row['reforward']:>9.1f} "
                         f"{row['stat_oh']:>9.1f} {row['total']:>9.1f}{flag}")
        grand = ph["grand"] or 1.0
        order = sorted(ph["totals"], key=lambda k: ph["totals"][k], reverse=True)
        lines.append("")
        lines.append("where time goes: "
                     + " | ".join(f"{k} {100 * ph['totals'][k] / grand:.0f}%" for k in order))
        if ph["dominant"]:
            lines.append(f"NEXT TARGET — {ph['dominant']}: {_PHASE_HINT.get(ph['dominant'], '')}")
        if any(r["warmup"] for r in ph["rows"]):
            wl = [r["layer"] for r in ph["rows"] if r["warmup"]]
            lines.append(f"note: layers {wl} flagged *warmup (forward >> 3x median) — one-time "
                         "aclnn/GE compile, not steady-state; judge the fix on non-warmup layers.")
        lines.append("")

    lines += ["=== per-layer routed-expert search ===",
             f"{'layer':>5} {'experts':>7} {'h_in':>6} {'h_out':>6} "
             f"{'tok(min/p50/max)':>18} {'wall_s':>8} {'TFLOP/s':>8} {'util':>6} "
             f"{'peakGB':>7} {'freeGB':>7}"]
    for pl in result["per_layer"]:
        tok_str = f"{pl['tok_min']}/{pl['tok_p50']}/{pl['tok_max']}"
        peak = pl['peak_gb'] if pl['peak_gb'] is not None else '-'
        free = round(pl['free_gb'], 1) if pl['free_gb'] is not None else '-'
        lines.append(
            f"{pl['layer']:>5} {pl['n_experts']:>7} {pl['h_in']:>6} {pl['h_out']:>6} "
            f"{tok_str:>18} {pl['wall_s']:>8.2f} {pl['tflops']:>8.1f} {pl['util']:>5.0%} "
            f"{peak:>7} {free:>7}")
    sf = result["search_fraction"]
    lines += ["",
              f"Q1 launch-vs-compute : {result['q1']}",
              f"Q2 token raggedness  : {result['q2']}",
              f"Q3 tile budget       : {result['q3']}",
              (f"search as fraction of layer wall: {sf:.0%}" if sf is not None
               else "search fraction: (no per-layer timing in log)"),
              ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logfile", nargs="?", help="run log (default: stdin)")
    ap.add_argument("--peak-tflops", type=float, default=100.0,
                    help="device peak for the util estimate (set to your NPU's fp32/fp16 peak)")
    ap.add_argument("--hbm-frac", type=float, default=0.5,
                    help="fraction of free HBM the tile may use")
    args = ap.parse_args()

    text = open(args.logfile).read() if args.logfile else sys.stdin.read()
    records = parse_prof_lines(text)
    if not records:
        print("no [PROF] lines found — did you run with JOINTFIX_PROFILE=1 ?", file=sys.stderr)
        sys.exit(1)
    result = analyze(records, peak_tflops=args.peak_tflops, hbm_budget_frac=args.hbm_frac)
    print(format_report(result))


if __name__ == "__main__":
    main()
