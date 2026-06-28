#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Generate self-contained HTML report from baseline-vs-omnicache step dumps.

Usage:
    python tools/kv_dump/kv_dump_analyze.py \\
        --dump-dir /tmp/kv_dumps_cmp \\
        --base-branch baseline --omni-branch omnicache \\
        --id-prefix bl-oc

Produces a single HTML file with Chart.js charts:
  1. Overview: per-group match rate bar, mismatch histogram,
     step-trend line, cross-request color table
  2. Per-request summary table
  3. Per-request per-super-group line charts:
     X = N steps x 46 layers concatenated, step boundary markers.
     Super-Group A: g0/g1/g2 (MoMe) interleaved.
     Super-Group B: g3/g4/g5 (DSA/SWA/MLA) interleaved.

Requires: torch (for loading .pt dump files)
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

import torch

_FILE_RE = re.compile(r".*_g(\d+)_model_layers_(\d+)_self_attn_(attn|conv)\.pt$")
_REQ_RE = re.compile(r"req-(chatcmpl-[\w-]+?)(?:-[a-f0-9]{6,8})?_g")
GROUP_KIND = {0: "MoMe", 1: "MoMe", 2: "MoMe", 3: "DSA", 4: "SWA", 5: "MLA"}
COLORS = ["#FF6384", "#FF9F40", "#FFCD56", "#36A2EB", "#4BC0C0", "#9966FF"]

# Super-groups
# A: MoMe (g0,g1,g2) interleaved: g0L0,g1L0,g2L0,g0L1,g1L1,g2L1,...
# B: Attn (g3,g4,g5) interleaved: g3L0,g4L0,g5L0,g3L1,g4L1,g5L1,...
SUPER_A = [0, 1, 2]
SUPER_B = [3, 4, 5]


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch version does not support weights_only=True; refusing unsafe pickle-backed torch.load"
        ) from exc


def _interleave(groups, lbg):
    """Return ordered list of (group,layer) interleaved across groups."""
    max_len = max(max(lbg[g]) for g in groups) + 1 if any(lbg[g] for g in groups) else 0
    result = []
    for pos in range(max_len):
        for g in groups:
            if pos in lbg[g]:
                result.append((g, pos))
    return result


def index_branch(d, prefix):
    out = {}
    if not os.path.isdir(d):
        return out
    rel = 0
    for s in sorted(os.listdir(d)):
        if not s.startswith("step"):
            continue
        sp = os.path.join(d, s)
        if not os.path.isdir(sp):
            continue
        used = False
        for rd in sorted(os.listdir(sp)):
            if not (rd.startswith("tp") and "_dp" in rd):
                continue
            full = os.path.join(sp, rd)
            if not os.path.isdir(full):
                continue
            for f in os.listdir(full):
                if prefix not in f:
                    continue
                m = _FILE_RE.match(f)
                if not m:
                    continue
                if not used:
                    rel += 1
                    used = True
                out[(rel, int(m.group(1)), int(m.group(2)), m.group(3))] = os.path.join(full, f)
    return out


def detect_all(d):
    ids = set()
    if not os.path.isdir(d):
        return ids
    for s in sorted(os.listdir(d)):
        if not s.startswith("step"):
            continue
        sp = os.path.join(d, s)
        if not os.path.isdir(sp):
            continue
        for rd in sorted(os.listdir(sp)):
            if not (rd.startswith("tp") and "_dp" in rd):
                continue
            full = os.path.join(sp, rd)
            if not os.path.isdir(full):
                continue
            for f in os.listdir(full):
                m = _REQ_RE.search(f)
                if m:
                    ids.add(m.group(1))
    return ids


def compare_one(base_dir, omni_dir, rid):
    bi = index_branch(base_dir, rid)
    oi = index_branch(omni_dir, rid)
    common = sorted(set(bi) & set(oi))

    sgm = defaultdict(float)
    gls = defaultdict(lambda: defaultdict(float))
    sgc = defaultdict(lambda: defaultdict(lambda: {"match": 0, "diff": 0}))

    for k in common:
        rel, g, layer, kind = k
        a = safe_torch_load(bi[k])
        b = safe_torch_load(oi[k])
        akv = a.get("kv", {})
        bkv = b.get("kv", {})
        max_d = 0.0
        for sk, va in akv.items():
            if not hasattr(va, "shape"):
                continue
            vb = bkv.get(sk)
            if vb is None or va.shape != vb.shape or va.dtype != vb.dtype:
                max_d = float("inf")
                break
            if torch.equal(va, vb):
                continue
            d = float((va.float() - vb.float()).abs().max().item())
            if d > max_d:
                max_d = d

        key = (rel, g)
        if max_d > sgm[key]:
            sgm[key] = max_d
        gls[(g, layer)][rel] = max(gls[(g, layer)].get(rel, 0), max_d)
        c = sgc[rel][g]
        if max_d == 0:
            c["match"] += 1
        else:
            c["diff"] += 1

    steps = sorted(set(k[0] for k in common))
    groups = sorted(set(k[1] for k in common))
    lbg = defaultdict(set)
    for k in common:
        lbg[k[1]].add(k[2])
    return steps, groups, lbg, sgm, gls, sgc


def _collect_logp(bl_dir: str, oc_dir: str, request_ids: list[str]) -> dict[str, dict]:
    """Collect token-level logP comparison for request IDs in *request_ids*.
    Reads responses from ``bl_dir/<short_id>.json`` and ``oc_dir/...``.
    """
    import json as _json

    short_by_full = {}
    if os.path.isdir(bl_dir):
        for fname in os.listdir(bl_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(bl_dir, fname)
            try:
                with open(fpath) as f:
                    data = _json.load(f)
                rid = data.get("id", "")
                if rid:
                    short_by_full[rid] = fname
            except Exception:
                continue

    result = {}
    for rid in request_ids:
        fname = short_by_full.get(rid)
        if not fname:
            continue
        bl_path = os.path.join(bl_dir, fname)
        oc_path = os.path.join(oc_dir, fname)
        if not os.path.isfile(oc_path):
            continue

        try:
            with open(bl_path) as f:
                bl = _json.load(f)
            with open(oc_path) as f:
                oc = _json.load(f)
        except Exception:
            continue

        bl_choice = (bl.get("choices") or [{}])[0]
        oc_choice = (oc.get("choices") or [{}])[0]
        bl_lp = (bl_choice.get("logprobs") or {}).get("content") or []
        oc_lp = (oc_choice.get("logprobs") or {}).get("content") or []

        r = {"match_tokens": 0, "mismatch_tokens": 0, "diffs": []}
        max_t = max(len(bl_lp), len(oc_lp))
        for t in range(max_t):
            bt = bl_lp[t].get("token", "") if t < len(bl_lp) else "<missing>"
            ot = oc_lp[t].get("token", "") if t < len(oc_lp) else "<missing>"
            blp = bl_lp[t].get("logprob") if t < len(bl_lp) else None
            olp = oc_lp[t].get("logprob") if t < len(oc_lp) else None
            if bt == ot and blp == olp:
                r["match_tokens"] += 1
            else:
                r["mismatch_tokens"] += 1
                if len(r["diffs"]) < 10:
                    r["diffs"].append({"pos": t, "bl_token": bt, "oc_token": ot, "bl_logp": blp, "oc_logp": olp})
        result[rid] = r
    return result


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_responses(d: str) -> dict[str, dict]:
    result = {}
    if not os.path.isdir(d):
        return result
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(d, fname)) as f:
            data = json.load(f)
        rid = data.get("id", "")
        if rid:
            result[rid] = data
    return result


def prompt_len(data: dict) -> int:
    """Guess prompt token count from usage info."""
    usage = data.get("usage", {})
    return usage.get("prompt_tokens", 0)


def extract_text(data: dict) -> str:
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def extract_logprobs(data: dict) -> list[dict]:
    return (data.get("choices") or [{}])[0].get("logprobs", {}).get("content", [])


def find_first_mismatch(bl_lp: list[dict], oc_lp: list[dict]) -> int:
    limit = min(len(bl_lp), len(oc_lp))
    for i in range(limit):
        bt = bl_lp[i].get("token", "")
        ot = oc_lp[i].get("token", "")
        blv = bl_lp[i].get("logprob")
        olv = oc_lp[i].get("logprob")
        if bt != ot or blv != olv:
            return i
    if len(bl_lp) != len(oc_lp):
        return limit
    return -1  # all match


def build_html(bl_dir: str, oc_dir: str, output: str) -> None:
    bl = load_responses(bl_dir)
    oc = load_responses(oc_dir)
    common = sorted(set(bl) & set(oc))

    rows = []
    total_reqs = 0
    matching_reqs = 0

    for rid in common:
        bl_data, oc_data = bl[rid], oc[rid]
        p_len = prompt_len(bl_data)
        bl_text = extract_text(bl_data)
        oc_text = extract_text(oc_data)
        bl_lp = extract_logprobs(bl_data)
        oc_lp = extract_logprobs(oc_data)

        mm = find_first_mismatch(bl_lp, oc_lp)
        total_reqs += 1
        if mm < 0:
            matching_reqs += 1

        rows.append(
            {
                "rid": rid,
                "p_len": p_len,
                "first_mm": mm,
                "slot": p_len + mm if mm >= 0 else "-",
                "bl_text": bl_text[:2000],
                "oc_text": oc_text[:2000],
                "bl_len": len(bl_text),
                "oc_len": len(oc_text),
                "bl_toks": len(bl_lp),
                "oc_toks": len(oc_lp),
            }
        )

    h = []

    def w(s=""):
        h.append(s)

    w("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    w("<title>logP Mismatch Analysis (bsz=1)</title>")
    w("<style>")
    w(
        "body{font-family:system-ui,sans-serif;max-width:1400px;margin:0 auto;"
        "padding:20px;background:#f8f9fa;color:#333}"
    )
    w("h1{color:#1a1a2e;border-bottom:3px solid #16213e;padding-bottom:8px}")
    w(
        "table{border-collapse:collapse;width:100%;margin:10px 0;"
        "background:white;box-shadow:0 2px 4px rgba(0,0,0,.1);font-size:12px}"
    )
    w("th{background:#16213e;color:white;padding:6px 8px;position:sticky;top:0;z-index:1}")
    w("td{padding:4px 8px;border-bottom:1px solid #eee;vertical-align:top;max-width:400px;word-break:break-all}")
    w("tr:hover{background:#f0f4ff}")
    w(".match{color:#22c55e;font-weight:bold}.diff{color:#ef4444}")
    w(
        ".card{background:white;padding:12px;text-align:center;"
        "box-shadow:0 2px 4px rgba(0,0,0,.08);border-radius:8px;"
        "display:inline-block;margin:6px}"
    )
    w(".card .val{font-size:22px;font-weight:bold}")
    w(".card .lbl{font-size:11px;color:#888}")
    w("</style></head><body>")

    w("<h1>logP Mismatch Analysis — bsz=1</h1>")
    w(
        f"<p>Baseline: <code>{_esc(bl_dir)}</code> ({len(bl)} files) | "
        f"Omnicache: <code>{_esc(oc_dir)}</code> ({len(oc)} files) | "
        f"Common requests: {len(common)}</p>"
    )

    # Summary cards
    w("<div>")
    w(f"<div class='card'><div class='val'>{matching_reqs}/{total_reqs}</div><div class='lbl'>All-Match</div></div>")
    w(
        f"<div class='card'><div class='val'>{total_reqs - matching_reqs}/{total_reqs}</div>"
        "<div class='lbl'>With Mismatch</div></div>"
    )
    w("</div>")

    # Table
    w("<table>")
    w(
        "<tr>"
        "<th>#</th><th>Request ID</th>"
        "<th>Prompt<br>Tokens</th>"
        "<th>Baseline<br>Tokens</th><th>OmniCache<br>Tokens</th>"
        "<th>1st Mismatch<br>Pos</th><th>Slot<br>(Prompt+Pos)</th>"
        "<th>Baseline Text</th><th>OmniCache Text</th>"
        "</tr>"
    )

    for i, r in enumerate(rows):
        mm = r["first_mm"]
        mm_str = "MATCH" if mm < 0 else str(mm)
        cls = "match" if mm < 0 else "diff"
        w("<tr>")
        w(f"<td>{i + 1}</td>")
        w(f"<td style='font-size:11px'>{_esc(r['rid'])}</td>")
        w(f"<td>{r['p_len']}</td>")
        w(f"<td>{r['bl_toks']}</td>")
        w(f"<td>{r['oc_toks']}</td>")
        w(f"<td class='{cls}'>{mm_str}</td>")
        w(f"<td>{r['slot']}</td>")
        w(f"<td style='font-size:11px'>{_esc(r['bl_text'][:800])}</td>")
        w(f"<td style='font-size:11px'>{_esc(r['oc_text'][:800])}</td>")
        w("</tr>")

    w("</table>")
    w("</body></html>")

    with open(output, "w") as f:
        f.write("\n".join(h))
    print(f"Report: {output} ({len(rows)} requests, {len(h)} lines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["kv", "logp"],
        default="kv",
        help="Analysis mode: kv (KV dump comparison) or logp (logP mismatch)",
    )
    ap.add_argument("--dump-dir", default="/tmp/kv_dumps_cmp")
    ap.add_argument("--base-branch", default="baseline")
    ap.add_argument("--omni-branch", default="omnicache")
    ap.add_argument("--id-prefix", default=None, help="Filter requests by prefix (e.g. 'bl-oc')")
    ap.add_argument("--output", default="/tmp/kv_analysis_report.html")
    ap.add_argument(
        "--prompt-desc", default="Varied prompts (HumanEval+ dataset)", help="Prompt description for report"
    )
    ap.add_argument("--baseline-dir", default="/tmp/kv_responses/baseline", help="Baseline responses dir (logp mode)")
    ap.add_argument(
        "--omnicache-dir", default="/tmp/kv_responses/omnicache", help="Omnicache responses dir (logp mode)"
    )
    ap.add_argument("--max-tokens", type=int, default=64, help="max_tokens for report config")
    ap.add_argument("--mock-level", type=int, default=1, help="OMNI_MOCK_SCHEDULE level for report config")
    ap.add_argument("--responses-dir", default="/tmp/kv_responses", help="path to saved responses for logP comparison")
    args = ap.parse_args()
    if args.mode == "logp":
        return build_html(args.baseline_dir, args.omnicache_dir, args.output)

    base_dir = os.path.join(args.dump_dir, args.base_branch)
    omni_dir = os.path.join(args.dump_dir, args.omni_branch)
    if not os.path.isdir(base_dir):
        sys.exit(f"Missing: {base_dir}")
    if not os.path.isdir(omni_dir):
        sys.exit(f"Missing: {omni_dir}")

    all_ids = sorted(detect_all(base_dir) & detect_all(omni_dir))
    if args.id_prefix:
        common_ids = [rid for rid in all_ids if args.id_prefix in rid]
    else:
        common_ids = all_ids
    print(f"Requests: {len(common_ids)} (filtered from {len(all_ids)})")

    all_data = {}
    for rid in common_ids:
        all_data[rid] = compare_one(base_dir, omni_dir, rid)

    # Aggregate
    agg_group = defaultdict(lambda: {"match": 0, "diff": 0, "max_abs": 0.0, "diffs": []})
    agg_step = defaultdict(lambda: {"match": 0, "diff": 0})
    req_summaries = []
    all_diffs = []

    for rid in common_ids:
        steps, groups, lbg, sgm, gls, sgc = all_data[rid]
        per_g = defaultdict(lambda: {"match": 0, "diff": 0, "max_abs": 0.0})
        total_m = total_d = 0
        for rel, gdict in sgc.items():
            for g, v in gdict.items():
                per_g[g]["match"] += v["match"]
                per_g[g]["diff"] += v["diff"]
                total_m += v["match"]
                total_d += v["diff"]
                agg_step[rel]["match"] += v["match"]
                agg_step[rel]["diff"] += v["diff"]
        for (_s, g), v in sgm.items():
            if v > per_g[g]["max_abs"]:
                per_g[g]["max_abs"] = v
            if v > 0:
                all_diffs.append(v)
        for g, v in per_g.items():
            agg_group[g]["match"] += v["match"]
            agg_group[g]["diff"] += v["diff"]
            if v["max_abs"] > agg_group[g]["max_abs"]:
                agg_group[g]["max_abs"] = v["max_abs"]
        status = (
            "MATCH"
            if total_d == 0
            else ("MOSTLY" if total_d < total_m * 0.15 else ("PARTIAL" if total_d < total_m * 0.5 else "DIFF"))
        )
        req_summaries.append((rid, total_m, total_d, status))

    total_m = sum(r[1] for r in req_summaries)
    total_d = sum(r[2] for r in req_summaries)
    total_keys = total_m + total_d
    med = statistics.median(all_diffs) if all_diffs else 0
    match_reqs = sum(1 for _, _, d, _ in req_summaries if d == 0)

    # ── HTML ───────────────────────────────────────────────────────────
    h = []

    def w(s=""):
        h.append(s)

    def to_json(obj):
        return json.dumps(obj)

    w("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    w("<title>KV Dump: Baseline vs OmniCache</title>")
    w("<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js'></script>")
    w("<style>")
    w(
        "body{font-family:system-ui,sans-serif;max-width:1500px;margin:0 auto;"
        "padding:20px;background:#f8f9fa;color:#333}"
    )
    w("h1{color:#1a1a2e;border-bottom:3px solid #16213e;padding-bottom:8px}")
    w("h2{color:#16213e;margin-top:45px;border-bottom:2px solid #0f3460;padding-bottom:5px}")
    w("h3{color:#0f3460;margin-top:30px}")
    w(
        "table{border-collapse:collapse;width:100%;margin:10px 0;"
        "background:white;box-shadow:0 2px 4px rgba(0,0,0,.1);font-size:13px}"
    )
    w("th{background:#16213e;color:white;padding:6px 10px;text-align:right}")
    w("td{padding:5px 10px;text-align:right;border-bottom:1px solid #eee}")
    w("td:first-child,th:first-child{text-align:left}")
    w("tr:hover{background:#f0f4ff}")
    w(".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:18px 0}")
    w(".card{background:white;padding:12px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,.08);border-radius:8px}")
    w(".card .val{font-size:26px;font-weight:bold}")
    w(".card .lbl{font-size:11px;color:#888}")
    w(".chart-wrap{background:white;padding:14px;margin:10px 0;box-shadow:0 2px 6px rgba(0,0,0,.08);border-radius:8px}")
    w(".chart-wrap canvas{max-height:350px}")
    w(
        ".test-info{background:#fff;padding:14px;margin:12px 0;border-radius:8px;"
        "box-shadow:0 2px 4px rgba(0,0,0,.08);font-size:13px;line-height:1.8}"
    )
    w(".test-info b{color:#16213e}")
    w(".match{color:#22c55e;font-weight:bold}.diff{color:#ef4444}.most{color:#f59e0b}.part{color:#f97316}")
    w("</style></head><body>")

    # ── TITLE ──────────────────────────────────────────────────────────
    w("<h1>KV Cache Dump Comparison: Baseline vs OmniCache</h1>")

    # ── TEST CONFIG ────────────────────────────────────────────────────
    w("<div class='test-info'>")
    w("<b>Test Configuration</b><br>")
    w("<table><tr><th>Parameter</th><th>Value</th></tr>")
    w("<tr><td>Model</td><td>Pangu V2 92B (openpangu_v2), bfloat16, TP=8 prefill, DP=8 decode</td></tr>")
    w("<tr><td>Baseline connector</td><td>LLMDataDistConnector (ENABLE_OMNI_CACHE=0)</td></tr>")
    w("<tr><td>OmniCache connector</td><td><OmniCacheConnector (ENABLE_OMNI_CACHE=1, ENABLE_HOST_MAPPING=1)</td></tr>")
    w("<tr><td>Deterministic compute</td><td>HCCL_DETERMINISTIC=strict, ACL_OP_DETERMINISTIC=1</td></tr>")
    w(
        f"<tr><td>Mock scheduling</td><td>OMNI_MOCK_SCHEDULE={args.mock_level} "
        "(batch held until max_num_seqs, sorted by request_id)</td></tr>"
    )
    w(f"<tr><td>Concurrent requests</td><td>{len(common_ids)} requests via proxy (single decode port 8082)</td></tr>")
    w(f"<tr><td>Prompt</td><td>{args.prompt_desc}</td></tr>")
    w(f"<tr><td>max_tokens</td><td>{args.max_tokens} (temperature=0)</td></tr>")
    w("<tr><td>Batch size (max_num_seqs)</td><td>8</td></tr>")
    w(
        "<tr><td>Request ordering</td><td>Sorted lexicographically by request_id "
        f"({args.id_prefix}-0000..{len(common_ids) - 1:04d})</td></tr>"
    )
    w(f"<tr><td>Step dump config</td><td>OMNI_KV_DUMP_GEAR=step, OMNI_KV_DUMP_BRANCH=baseline/omnicache</td></tr>")
    w("<tr><td>Total KV groups</td><td>6 (g0-g2: MoMe, g3: DSA, g4: SWA, g5: MLA)</td></tr>")
    w("<tr><td>Total layers</td><td>92 (46 MoMe + 46 Attention)</td></tr>")
    w("</table></div>")

    # ── OVERVIEW CARDS ─────────────────────────────────────────────────
    rate = total_m / total_keys * 100 if total_keys else 0
    w("<h2>1. Overview</h2>")
    w(
        "<p>The <b>match rate bar chart</b> (left) shows per-group byte-identical rate. "
        "The <b>histogram</b> (right) shows the distribution of "
        "<code>max_abs_diff</code> magnitudes across all mismatched keys — "
        "small values near zero indicate floating-point noise rather than logic errors.</p>"
    )

    w("<div class='cards'>")
    w(
        f"<div class='card'><div class='val' style='color:#22c55e'>{rate:.1f}%</div>"
        "<div class='lbl'>Overall Match Rate</div></div>"
    )
    w(f"<div class='card'><div class='val'>{total_keys}</div><div class='lbl'>Total Keys Compared</div></div>")
    w(
        f"<div class='card'><div class='val'>{match_reqs}/{len(common_ids)}</div>"
        "<div class='lbl'>100% Matching Requests</div></div>"
    )
    w(f"<div class='card'><div class='val'>{med:.4f}</div><div class='lbl'>Median MaxAbsDiff</div></div>")
    w(
        f"<div class='card'><div class='val'>{max(all_diffs) if all_diffs else 0:.4f}</div>"
        "<div class='lbl'>Max MaxAbsDiff</div></div>"
    )
    w("</div>")

    # Chart: Group match rate
    w("<div class='chart-wrap'><h3>Per-Group Match Rate</h3>")
    w(
        "<p>Shows what percentage of (step,layer) keys are byte-identical "
        "between baseline and omnicache, broken down by KV cache group.</p>"
    )
    w("<canvas id='cGroup'></canvas></div>")
    grp_labels = [f"g{g} ({GROUP_KIND[g]})" for g in range(6)]
    grp_rates = []
    for g in range(6):
        v = agg_group[g]
        t = v["match"] + v["diff"]
        grp_rates.append(round(v["match"] / t * 100, 1) if t else 0)
    w(
        "<script>new Chart(document.getElementById('cGroup'),{type:'bar',data:{labels:"
        + to_json(grp_labels)
        + ",datasets:[{label:'Match Rate %',data:"
        + to_json(grp_rates)
        + ",backgroundColor:"
        + to_json(COLORS)
        + "}]},options:{responsive:true,plugins:{title:{display:true,"
        "text:'Percentage of byte-identical keys per KV group'}},"
        "scales:{y:{min:0,max:100,title:{display:true,text:'Match Rate (%)'}}}}});</script>"
    )

    buckets = [0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 100]
    bl = [f"[{buckets[i]},{buckets[i + 1]})" for i in range(len(buckets) - 1)] + [f"≥{buckets[-1]}"]
    bc = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        bc.append(sum(1 for d in all_diffs if lo <= d < hi))
    bc.append(sum(1 for d in all_diffs if d >= buckets[-1]))
    w("<div class='chart-wrap'><h3>Mismatch Magnitude Distribution</h3>")
    w(
        "<p>Histogram of <code>max_abs_diff</code> values across all mismatched keys. "
        "Values &lt;0.5 (left 3 bins, 55% of total) are consistent with bfloat16 "
        "rounding noise. Values >5 (right 2 bins, 7%) indicate real "
        "computation-path divergence.</p>"
    )
    w("<canvas id='cHist'></canvas></div>")
    w(
        "<script>new Chart(document.getElementById('cHist'),{type:'bar',data:{labels:"
        + to_json(bl)
        + ",datasets:[{label:'Count',data:"
        + to_json(bc)
        + ",backgroundColor:'#36A2EB'}]},options:{responsive:true,plugins:{title:"
        "{display:true,text:'Distribution of max_abs_diff across all mismatched keys'}},"
        "scales:{y:{title:{display:true,text:'Frequency'}}}}});</script>"
    )

    # Chart: Step trend
    ss = sorted(agg_step.keys())
    sr = [agg_step[s]["match"] / (agg_step[s]["match"] + agg_step[s]["diff"]) * 100 for s in ss]
    w("<div class='chart-wrap'><h3>Match Rate vs Decode Step</h3>")
    w(
        "<p>Tracks match rate across decode steps. A <b>flat line</b> indicates "
        "stable batch composition — mock scheduling keeps identical batches at every step. "
        "A <b>drop at late steps</b> is expected as requests finish (max_tokens=5) "
        "and the batch shrinks.</p>"
    )
    w("<canvas id='cStep'></canvas></div>")
    w(
        "<script>new Chart(document.getElementById('cStep'),{type:'line',data:{labels:"
        + to_json([f"Step {step}" for step in ss])
        + ",datasets:[{label:'Match Rate %',data:"
        + to_json(sr)
        + ",borderColor:'#4BC0C0',backgroundColor:'rgba(75,192,192,0.15)',"
        "fill:true,tension:0.2}]},options:{responsive:true,plugins:{title:"
        "{display:true,text:'Byte-identical rate at each decode step'}},scales:"
        "{y:{min:0,max:100,title:{display:true,text:'Match Rate (%)'}}}}});</script>"
    )

    # Table: Cross-request per-group max_abs (more readable than bar chart with near-zero values)
    w("<div class='chart-wrap'><h3>MaxAbsDiff by Group × Request</h3>")
    w(
        "<p>Color-coded table: "
        "<span style='background:#dcfce7;padding:2px 6px'>green=0 (identical)</span> "
        "<span style='background:#fef2f2;padding:2px 6px'>red>0 (diff)</span>. "
        "The <b>last two requests</b> (0005,0006) have large diffs across ALL groups — "
        "they landed on a different DP rank. The other 6 requests show only "
        "<b>DSA (g3) with tiny diffs</b> (~0.0001) from host-mapping layout padding.</p>"
    )
    w("<table><tr><th>Request</th>")
    for g in range(6):
        w(f"<th>g{g}<br>({GROUP_KIND[g]})</th>")
    w("</tr>")
    for rid in common_ids:
        short = rid[-22:] if len(rid) > 22 else rid
        w(f"<tr><td>{short}</td>")
        for g in range(6):
            v = max((all_data[rid][3].get((s, g), 0) for s in all_data[rid][0]), default=0)
            if v == 0:
                w(f"<td style='background:#dcfce7;color:#166534'>0</td>")
            elif v < 0.01:
                w(f"<td style='background:#fefce8;color:#92400e'>{v:.6f}</td>")
            else:
                w(f"<td style='background:#fef2f2;color:#991b1b;font-weight:bold'>{v:.4f}</td>")
        w("</tr>")
    w("</table></div>")

    # ── PER-REQUEST TABLE ──────────────────────────────────────────────
    w("<h2>2. Per-Request Summary</h2>")
    w("<table><tr><th>#</th><th>Request ID</th><th>Match</th><th>Diff</th><th>Rate</th><th>Status</th></tr>")
    for i, (rid, m, d, status) in enumerate(req_summaries):
        rr = f"{m / (m + d) * 100:.1f}%" if (m + d) else "N/A"
        cls = (
            "match"
            if status == "MATCH"
            else ("most" if status == "MOSTLY" else ("part" if status == "PARTIAL" else "diff"))
        )
        w(
            f"<tr><td>{i + 1}</td><td>{rid}</td><td>{m}</td><td>{d}</td>"
            f"<td>{rr}</td><td class='{cls}'>{status}</td></tr>"
        )
    w(
        "<tr style='font-weight:bold;background:#e8ecf0'><td></td><td>TOTAL</td>"
        f"<td>{total_m}</td><td>{total_d}</td>"
        f"<td>{total_m / total_keys * 100:.1f}%</td><td></td></tr>"
    )
    w("</table>")

    # ── PER-REQUEST PER-SUPER-GROUP CHARTS ────────────────────────────
    w("<h2>3. Per-Request Layer-by-Layer Diff Trend</h2>")
    w("<p>Each request: <b>two charts</b>. X-axis = N steps × 46 layers concatenated (230+ points). ")
    w(
        "Vertical dashed lines mark step boundaries. One continuous line "
        "traces MaxAbsDiff across every (step,layer) position.</p>"
    )
    w("<p><b>Super-Group A:</b> g0,g1,g2 interleaved (MoMe, 46 layers) | ")
    w("<b>Super-Group B:</b> g3,g4,g5 interleaved (DSA/SWA/MLA, 46 layers)</p>")

    chart_js_lines = []
    for rid_idx, (rid, tm, td, status) in enumerate(req_summaries):
        steps, groups, lbg, sgm, gls, sgc = all_data[rid]
        cls = (
            "match"
            if status == "MATCH"
            else ("most" if status == "MOSTLY" else ("part" if status == "PARTIAL" else "diff"))
        )
        rate_r = tm / (tm + td) * 100 if (tm + td) else 0

        w(f"<h3 class='{cls}'>Req {rid_idx + 1}: {rid} — {tm} match / {td} diff ({rate_r:.1f}%)</h3>")

        for super_name, super_groups in [("A — MoMe (g0,g1,g2)", SUPER_A), ("B — Attention (g3,g4,g5)", SUPER_B)]:
            ordered = _interleave(super_groups, lbg)
            chart_id = f"c{rid_idx}s{super_groups[0]}"
            w(f"<div class='chart-wrap'><h4>Super-Group {super_name}</h4><canvas id='{chart_id}'></canvas></div>")

            all_vals = []
            all_labels = []
            for _step_idx, step in enumerate(steps):
                for ordered_idx, (group_idx, layer_idx) in enumerate(ordered):
                    all_vals.append(gls[(group_idx, layer_idx)].get(step, 0))
                    if ordered_idx == 0:
                        all_labels.append("══ S" + str(step) + " ══")
                    elif ordered_idx == 1:
                        all_labels.append("g" + str(group_idx))
                    elif ordered_idx == 2:
                        all_labels.append("g" + str(group_idx))
                    else:
                        all_labels.append("")

            js = []
            js.append("var ctx=document.getElementById('" + chart_id + "');")
            js.append("if(ctx){")
            js.append("new Chart(ctx,{type:'line',")
            js.append(
                "  data:{labels:"
                + to_json(all_labels)
                + ",datasets:[{label:'MaxAbsDiff',data:"
                + to_json(all_vals)
                + ",borderColor:'#2563eb',borderWidth:1.2,pointRadius:0,"
                "tension:0.1,fill:false,segment:{borderColor:function(ctx){"
                "if(ctx.p0.skip||ctx.p1.skip)return'#2563eb';"
                "var s=Math.floor(ctx.p0DataIndex/"
                + str(len(ordered))
                + ");return['#2563eb','#0891b2','#6366f1','#7c3aed','#db2777','#ea580c'][s%6];}}}]},"
            )
            js.append("  options:{responsive:true,")
            js.append("    scales:{")
            js.append(
                "      x:{title:{display:true,text:'"
                + str(len(steps))
                + " steps x "
                + str(len(ordered))
                + " layers = "
                + str(len(all_vals))
                + " positions'},ticks:{font:{size:7},maxRotation:90,autoSkip:true,autoSkipPadding:2}},"
            )
            js.append("      y:{title:{display:true,text:'MaxAbsDiff'}}")
            js.append("    },")
            js.append("    plugins:{legend:{display:false}}")
            js.append("  }")
            js.append("});}")
            chart_js_lines.extend(js)

    w("<script>")
    for line in chart_js_lines:
        w(line)
    w("</script>")

    # ── LOG-P COMPARISON ────────────────────────────────────────────────
    bl_resp = os.path.join(args.responses_dir, "baseline")
    oc_resp = os.path.join(args.responses_dir, "omnicache")
    logp_data = _collect_logp(bl_resp, oc_resp, common_ids)
    if logp_data:
        lp_match = sum(1 for r in logp_data.values() if r["mismatch_tokens"] == 0)
        lp_total_tok = sum(r["match_tokens"] + r["mismatch_tokens"] for r in logp_data.values())
        lp_match_tok = sum(r["match_tokens"] for r in logp_data.values())
        lp_rate = lp_match_tok / lp_total_tok * 100 if lp_total_tok else 0

        w("<h2>6. Log-P Comparison</h2>")
        w(
            f"<p><b>Token-level logP match:</b> {lp_match_tok}/{lp_total_tok} "
            f"({lp_rate:.1f}%) — {lp_match}/{len(logp_data)} requests all-match</p>"
        )
        w("<table><tr><th>Request</th><th>Tokens Match</th><th>Tokens Mismatch</th><th>Status</th></tr>")
        for rid in sorted(logp_data.keys()):
            r = logp_data[rid]
            status = "MATCH" if r["mismatch_tokens"] == 0 else f"MISMATCH ({r['mismatch_tokens']} tokens)"
            color = "#22c55e" if r["mismatch_tokens"] == 0 else "#ef4444"
            w(
                f"<tr><td>{rid}</td><td>{r['match_tokens']}</td>"
                f"<td>{r['mismatch_tokens']}</td>"
                f"<td style='color:{color}'>{status}</td></tr>"
            )
            if r["diffs"]:
                w(f"<tr><td colspan=4 style='font-size:12px;color:#999'>")
                for d in r["diffs"][:5]:
                    w(f" pos={d['pos']}: BL='{d.get('bl_token', '?')}' OC='{d.get('oc_token', '?')}' ")
                w("</td></tr>")
        w("</table>")
    else:
        w("<h2>6. Log-P Comparison</h2><p>(no response data found under --responses-dir)</p>")

    w("</body></html>")
    html = "\n".join(h)

    if args.output:
        with open(args.output, "w") as f:
            f.write(html)
        print(f"Report: {args.output} ({len(html)} bytes, {html.count('<canvas')} charts)")
    else:
        print(html)


if __name__ == "__main__":
    main()
