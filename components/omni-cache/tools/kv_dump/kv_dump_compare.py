#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unified KV dump comparator for omni-cache PD diagnostics.

Modes
-----
* ``transfer``: 4-stage probe data (.npz under ``<dir>/{prefill,decode}/req_<id>/``).
  Wraps :mod:`kv_consistency_check.py` to compare per-stage transitions.
  Use to confirm KV transmission is byte-faithful at each hop.

* ``step``: per-decode-step .pt dumps under
  ``<dir>/_step/<branch>/step{NNNN}/tp{R}_dp{R}/req-<id>_g*_*.pt``.
  Cross-correlates baseline vs omnicache layer-by-layer.

Usage
-----
::

    # auto-detect mode from layout (writes mismatches.tsv by default)
    python tools/kv_dump/kv_dump_compare.py --dump-dir /tmp/kv_dumps

    # force step mode with explicit branches/req-ids
    python tools/kv_dump/kv_dump_compare.py --mode step \\
        --baseline-branch baseline --omni-branch omnicache

    # compare ALL requests with matching deterministic IDs across branches
    python tools/kv_dump/kv_dump_compare.py --mode step \\
        --dump-dir /path/to/dumps --all-requests

    # disable mismatch file output
    python tools/kv_dump/kv_dump_compare.py --mode step --all-requests \\
        --no-mismatch-file

    # quick peek mode: only export mismatches to TSV (fast, no summary)
    python tools/kv_dump/kv_dump_compare.py --mode step --all-requests \\
        --quick-peek

    # 4-stage probe consistency for one request id
    python tools/kv_dump/kv_dump_compare.py --mode transfer \\
        --request-id chatcmpl-XXXX

Output Files
------------
By default, mismatches are written to ``mismatches.tsv`` with columns:

    request_id, step, group, layer, kind, subkey, shape, file, type

Where ``type`` is either ``value_mismatch`` or ``shape_mismatch``.
Use ``--no-mismatch-file`` to disable, or ``--mismatch-file <path>`` to customize.

Quick Peek Mode
---------------
When ``--quick-peek`` is specified, the script skips full aggregation and only
writes mismatches to the TSV file. This mode is faster for large-scale scans
where you only need to identify which keys differ.

Deterministic Request IDs
-------------------------
To compare the same request across branches under concurrent load, use
the ``X-Request-Id`` HTTP header when sending requests through the nginx
router.  The header value propagates through the PD pipeline and appears
in ``input_batch.req_ids`` (prefixed with ``chatcmpl-``).  Example::

    curl -H "X-Request-Id: test-req-0001" http://localhost:7077/v1/chat/completions ...

The ``--all-requests`` flag discovers all request IDs present in both
branches and compares them pairwise.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
import torch
import os
import re
import subprocess
import sys
from collections import defaultdict

# Progress bar support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

_FILE_RE = re.compile(
    r".*_g(\d+)_model_layers_(\d+)_self_attn_(attn|conv)\.pt$"
)
# Match both old format (with hash suffix) and new format (no hash).
# Old: req-chatcmpl-<uuid>-<hash>_g*_*.pt
# New: req-chatcmpl-<id>_g*_*.pt
# Match request ID with optional vLLM-generated hash suffix.
# Old format: req-chatcmpl-<uuid>-<hash>_g... (hash stripped)
# New format: req-chatcmpl-<x-request-id>_g... or req-chatcmpl-<x-request-id>-<hash>_g...
# The non-greedy +? lets the optional hash group consume the suffix when present.
_REQ_RE = re.compile(r"req-(chatcmpl-[\w-]+?)(?:-[a-f0-9]{6,8})?_g")


def step_root(dump_dir: str) -> str:
    """Return the root directory holding `<branch>/step{NNNN}/` dumps.

    Tries `<dump_dir>/_step` first (unified layout); then checks if
    dump_dir itself contains branch subdirectories; falls back to
    `/tmp/kv_dumps_step` (legacy `OMNI_KV_STEP_DUMP_DIR` default).
    """
    cand = os.path.join(dump_dir, "_step")
    if os.path.isdir(cand):
        return cand
    # Check if dump_dir itself has branch dirs (baseline/omnicache)
    if os.path.isdir(dump_dir):
        for name in ("baseline", "omnicache"):
            if os.path.isdir(os.path.join(dump_dir, name)):
                return dump_dir
    legacy = "/tmp/kv_dumps_step"
    if os.path.isdir(legacy):
        return legacy
    return cand


def detect_mode(dump_dir: str) -> str:
    root = step_root(dump_dir)
    has_step = os.path.isdir(root) and any(
        os.path.isdir(os.path.join(root, b))
        for b in os.listdir(root)
        if not b.startswith("_")
    )
    has_transfer = os.path.isdir(os.path.join(dump_dir, "prefill")) or \
                   os.path.isdir(os.path.join(dump_dir, "decode"))
    if has_step:
        return "step"
    if has_transfer:
        return "transfer"
    return "step"


def _find_rank_dir(step_dir: str):
    """Find the first tp*_dp* subdir that exists under a step directory."""
    if not os.path.isdir(step_dir):
        return None
    for entry in sorted(os.listdir(step_dir)):
        if entry.startswith("tp") and "_dp" in entry:
            full = os.path.join(step_dir, entry)
            if os.path.isdir(full):
                return full
    return None


def detect_request_in_branch(branch_dir: str):
    """Return the last request ID found across all rank dirs."""
    if not os.path.isdir(branch_dir):
        return None
    last = None
    for s in sorted(os.listdir(branch_dir)):
        if not s.startswith("step"):
            continue
        step_path = os.path.join(branch_dir, s)
        if not os.path.isdir(step_path):
            continue
        for rank_dir in sorted(os.listdir(step_path)):
            if not (rank_dir.startswith("tp") and "_dp" in rank_dir):
                continue
            full = os.path.join(step_path, rank_dir)
            if not os.path.isdir(full):
                continue
            for f in os.listdir(full):
                m = _REQ_RE.search(f)
                if m:
                    last = m.group(1)
    return last


def detect_all_requests_in_branch(branch_dir: str):
    """Return a set of all request IDs found in a branch.

    Scans ALL tp*_dp* rank directories within each step, not just
    the first one — different DP ranks may hold different requests.
    """
    ids = set()
    if not os.path.isdir(branch_dir):
        return ids
    for s in sorted(os.listdir(branch_dir)):
        if not s.startswith("step"):
            continue
        step_path = os.path.join(branch_dir, s)
        if not os.path.isdir(step_path):
            continue
        for rank_dir in sorted(os.listdir(step_path)):
            if not (rank_dir.startswith("tp") and "_dp" in rank_dir):
                continue
            full = os.path.join(step_path, rank_dir)
            if not os.path.isdir(full):
                continue
            for f in os.listdir(full):
                m = _REQ_RE.search(f)
                if m:
                    ids.add(m.group(1))
    return ids


def index_branch(branch_dir: str, request_prefix: str):
    """Index files by (relative_step, group, layer, kind).

    Uses a relative step counter — the first step directory that contains
    dump files for `request_prefix` gets rel=1, the next gets rel=2, etc.
    This aligns steps across branches even when warmup step counts differ.

    Scans ALL tp*_dp* rank directories within each step.
    """
    out = {}
    if not os.path.isdir(branch_dir):
        return out
    rel = 0
    for s in sorted(os.listdir(branch_dir)):
        if not s.startswith("step"):
            continue
        step_path = os.path.join(branch_dir, s)
        if not os.path.isdir(step_path):
            continue
        used_this_step = False
        for rank_dir in sorted(os.listdir(step_path)):
            if not (rank_dir.startswith("tp") and "_dp" in rank_dir):
                continue
            full = os.path.join(step_path, rank_dir)
            if not os.path.isdir(full):
                continue
            for f in os.listdir(full):
                if request_prefix not in f:
                    continue
                m = _FILE_RE.match(f)
                if not m:
                    continue
                if not used_this_step:
                    rel += 1
                    used_this_step = True
                out[(rel, int(m.group(1)), int(m.group(2)), m.group(3))] = os.path.join(full, f)
    return out


def _compare_one_request(base_dir, omni_dir, base_id, omni_id, atol, mismatch_file=None, quick_peek=False):
    """Compare one request across baseline and omni branches.

    Returns (per_g dict, sample list, mismatch_count) where per_g[g] = {match, diff, max}.
    If mismatch_file is provided, writes mismatch info to file.
    If quick_peek=True, skips aggregation and only writes mismatches.
    """
    import torch  # type: ignore
    base_idx = index_branch(base_dir, base_id)
    omni_idx = index_branch(omni_dir, omni_id)

    common = sorted(set(base_idx) & set(omni_idx))

    if not common:
        return None, None, len(base_idx), len(omni_idx), 0, 0

    # Quick peek mode: just log mismatches to file, no aggregation
    if quick_peek and mismatch_file:
        mismatch_count = 0
        key_iter = tqdm(common, desc="  Quick peek", unit="key", leave=False) if HAS_TQDM else common
        for k in key_iter:
            rel, g, layer, kind = k
            a = torch.load(base_idx[k], map_location="cpu", weights_only=False)
            b = torch.load(omni_idx[k], map_location="cpu", weights_only=False)
            akv = a.get("kv", {})
            bkv = b.get("kv", {})
            for sk, va in akv.items():
                if not hasattr(va, "shape"):
                    continue
                vb = bkv.get(sk)
                if vb is None or va.shape != vb.shape or va.dtype != vb.dtype:
                    fname = os.path.basename(base_idx[k])
                    mismatch_file.write(f"{base_id}\t{rel}\t{g}\t{layer}\t{kind}\t{sk}\t{va.shape}\t{fname}\tshape_mismatch\n")
                    mismatch_count += 1
                    continue
                if torch.equal(va, vb):
                    continue
                fname = os.path.basename(base_idx[k])
                mismatch_file.write(f"{base_id}\t{rel}\t{g}\t{layer}\t{kind}\t{sk}\t{va.shape}\t{fname}\tvalue_mismatch\n")
                mismatch_count += 1
        return None, None, len(base_idx), len(omni_idx), len(common), mismatch_count

    # Normal mode: full aggregation (with optional mismatch file)
    per_g = defaultdict(lambda: {"match": 0, "diff": 0, "max": 0.0})
    sample = []
    mismatch_count = 0
    key_iter = tqdm(common, desc="  Comparing keys", unit="key", leave=False) if HAS_TQDM else common
    for k in key_iter:
        rel, g, layer, kind = k
        a = torch.load(base_idx[k], map_location="cpu", weights_only=False)
        b = torch.load(omni_idx[k], map_location="cpu", weights_only=False)
        akv = a.get("kv", {})
        bkv = b.get("kv", {})
        ok = True
        max_d = 0.0
        for sk, va in akv.items():
            if not hasattr(va, "shape"):
                continue
            vb = bkv.get(sk)
            if vb is None or va.shape != vb.shape or va.dtype != vb.dtype:
                ok = False
                break
            if torch.equal(va, vb):
                continue
            # Log mismatch before computing diff
            fname = os.path.basename(base_idx[k])
            if mismatch_file:
                mismatch_file.write(f"{base_id}\t{rel}\t{g}\t{layer}\t{kind}\t{sk}\t{va.shape}\t{fname}\tvalue_mismatch\n")
                mismatch_count += 1
            d = (va.float() - vb.float()).abs().max().item()
            if d > max_d:
                max_d = d
            if d > atol:
                ok = False
        bucket = per_g[g]
        if ok:
            bucket["match"] += 1
        else:
            bucket["diff"] += 1
            if max_d > bucket["max"]:
                bucket["max"] = max_d
            if len(sample) < 4:
                sample.append({"step": rel, "g": g, "layer": layer,
                               "kind": kind, "max_abs": round(max_d, 5)})
    return per_g, sample, len(base_idx), len(omni_idx), len(common), mismatch_count


def step_compare(args):
    import torch  # type: ignore
    root = step_root(args.dump_dir)
    base_dir = os.path.join(root, args.baseline_branch)
    omni_dir = os.path.join(root, args.omni_branch)
    if not os.path.isdir(base_dir):
        sys.exit(f"baseline dir missing: {base_dir}")
    if not os.path.isdir(omni_dir):
        sys.exit(f"omni dir missing: {omni_dir}")

    if args.all_requests:
        base_ids = detect_all_requests_in_branch(base_dir)
        omni_ids = detect_all_requests_in_branch(omni_dir)
        common_ids = sorted(base_ids & omni_ids)
        print(f"# step compare — all matching requests")
        print(f"#   baseline: branch={args.baseline_branch} requests={len(base_ids)}")
        print(f"#   omni:     branch={args.omni_branch} requests={len(omni_ids)}")
        print(f"#   common:   {len(common_ids)} requests")
        if not common_ids:
            print(f"#   baseline IDs sample: {sorted(base_ids)[:5]}")
            print(f"#   omni IDs sample: {sorted(omni_ids)[:5]}")
            sys.exit("No matching request IDs between branches.")

        # Limit to first N requests for quick testing
        compare_ids = common_ids[:args.max_requests] if args.max_requests else common_ids

        # Determine output file for mismatches
        mismatch_file_path = None if args.no_mismatch_file else args.mismatch_file
        is_quick_peek = args.quick_peek

        # Quick peek mode: only write mismatches, no summary
        if is_quick_peek and mismatch_file_path:
            print(f"# Quick peek mode, writing to: {mismatch_file_path}")
            with open(mismatch_file_path, 'w') as f:
                f.write("request_id\tstep\tgroup\tlayer\tkind\tsubkey\tshape\tfile\ttype\n")
                total_mismatches = 0
                req_iter = tqdm(compare_ids, desc="Quick peek", unit="req") if HAS_TQDM else compare_ids
                for rid in req_iter:
                    _, _, _, _, _, mismatch_count = _compare_one_request(
                        base_dir, omni_dir, rid, rid, args.atol, mismatch_file=f, quick_peek=True)
                    total_mismatches += mismatch_count
            print(f"# Done. Total mismatches: {total_mismatches}")
            print(f"# Output: {mismatch_file_path}")
            return

        # Normal mode: full aggregation (with optional mismatch file)
        agg_g = defaultdict(lambda: {"match": 0, "diff": 0, "max": 0.0})
        req_summary = []
        total_mismatches = 0

        # Open mismatch file if provided
        mismatch_file_ctx = open(mismatch_file_path, 'w') if mismatch_file_path else None
        if mismatch_file_ctx:
            mismatch_file_ctx.write("request_id\tstep\tgroup\tlayer\tkind\tsubkey\tshape\tfile\ttype\n")

        req_iter = tqdm(compare_ids, desc="Comparing requests", unit="req") if HAS_TQDM else compare_ids
        try:
            for rid in req_iter:
                per_g, sample, n_base, n_omni, n_common, mismatch_count = _compare_one_request(
                    base_dir, omni_dir, rid, rid, args.atol,
                    mismatch_file=mismatch_file_ctx, quick_peek=False)
                total_mismatches += mismatch_count
                total_match = sum(v["match"] for v in per_g.values())
                total_diff = sum(v["diff"] for v in per_g.values())
                status = "MATCH" if total_diff == 0 else "DIFF"
                req_summary.append((rid, total_match, total_diff, n_common, status))
                for g, v in per_g.items():
                    agg_g[g]["match"] += v["match"]
                    agg_g[g]["diff"] += v["diff"]
                    if v["max"] > agg_g[g]["max"]:
                        agg_g[g]["max"] = v["max"]
        finally:
            if mismatch_file_ctx:
                mismatch_file_ctx.close()

        print()
        print("=== Per-request summary ===")
        for rid, m, d, c, st in req_summary:
            print(f"  {rid}: {c} keys, {m} match, {d} diff [{st}]")

        print()
        print("=== Aggregate by group ===")
        print(f"{'g':>2}  {'match':>6} {'diff':>5}  {'max_abs':>10}")
        for g in sorted(agg_g):
            v = agg_g[g]
            print(f"{g:>2} {v['match']:>7} {v['diff']:>5}  {v['max']:>10.4f}")

        if mismatch_file_path:
            print(f"\n# Mismatches written to: {mismatch_file_path} ({total_mismatches} entries)")
        return

    base_id = args.baseline_id or detect_request_in_branch(base_dir)
    omni_id = args.omni_id or detect_request_in_branch(omni_dir)
    if not base_id or not omni_id:
        sys.exit(f"need --baseline-id/--omni-id; got {base_id=}, {omni_id=}")

    # Determine output file for mismatches
    mismatch_file_path = None if args.no_mismatch_file else args.mismatch_file
    is_quick_peek = args.quick_peek

    # Quick peek mode: only write mismatches, no summary
    if is_quick_peek and mismatch_file_path:
        print(f"# Quick peek mode, writing to: {mismatch_file_path}")
        with open(mismatch_file_path, 'w') as f:
            f.write("request_id\tstep\tgroup\tlayer\tkind\tsubkey\tshape\tfile\ttype\n")
            _, _, n_base, n_omni, n_common, mismatch_count = _compare_one_request(
                base_dir, omni_dir, base_id, omni_id, args.atol,
                mismatch_file=f, quick_peek=True)
        print(f"# Done. Mismatches: {mismatch_count}")
        print(f"# Output: {mismatch_file_path}")
        return

    # Normal mode with optional mismatch file
    mismatch_file_ctx = open(mismatch_file_path, 'w') if mismatch_file_path else None
    if mismatch_file_ctx:
        mismatch_file_ctx.write("request_id\tstep\tgroup\tlayer\tkind\tsubkey\tshape\tfile\ttype\n")

    try:
        per_g, sample, n_base, n_omni, n_common, mismatch_count = _compare_one_request(
            base_dir, omni_dir, base_id, omni_id, args.atol,
            mismatch_file=mismatch_file_ctx, quick_peek=False)
    finally:
        if mismatch_file_ctx:
            mismatch_file_ctx.close()

    print(f"# step compare")
    print(f"#   baseline: branch={args.baseline_branch} req={base_id} files={n_base}")
    print(f"#   omni:     branch={args.omni_branch} req={omni_id} files={n_omni}")

    if not n_common:
        sys.exit("No matching keys between baseline and omni.")

    print()
    print(f"{'g':>2}  {'match':>6} {'diff':>5}  {'max_abs':>10}")
    for g in sorted(per_g):
        v = per_g[g]
        print(f"{g:>2} {v['match']:>7} {v['diff']:>5}  {v['max']:>10.4f}")

    if mismatch_file_path:
        print(f"\n# Mismatches written to: {mismatch_file_path} ({mismatch_count} entries)")


_FILE_RE_SELF = re.compile(
    r"req-(chatcmpl-[\w-]+?)(?:-[a-f0-9]{6,8})?_g(\d+)_model_layers_(\d+)_self_attn_(attn|conv)\.pt$"
)

def self_consistency_compare(args):
    """Single-branch KV self-consistency check."""
    branch_dir = args.dump_dir
    exclude_last = not getattr(args, 'include_last_block', False)
    
    if not os.path.isdir(branch_dir):
        sys.exit(f"Not a directory: {branch_dir}")

    idx = defaultdict(list)
    for step_name in sorted(os.listdir(branch_dir)):
        step_path = os.path.join(branch_dir, step_name)
        if not os.path.isdir(step_path):
            continue
        for rank_name in sorted(os.listdir(step_path)):
            rank_path = os.path.join(step_path, rank_name)
            if not os.path.isdir(rank_path):
                continue
            for fname in os.listdir(rank_path):
                m = _FILE_RE_SELF.search(fname)
                if not m:
                    continue
                req_id = m.group(1)
                g = int(m.group(2))
                layer = int(m.group(3))
                kind = m.group(4)
                fpath = os.path.join(rank_path, fname)
                step_num = int(step_name.replace("step", ""))
                idx[(req_id, g, layer, kind)].append((step_num, fpath))

    for k in idx:
        idx[k].sort()

    total_checks = total_corrupted = 0
    corruption_samples = []
    req_stats = defaultdict(lambda: {"checks": 0, "corrupted": 0})

    for k in sorted(idx):
        entries = idx[k]
        if len(entries) < 2:
            continue
        rid, g, layer, kind = k
        prev_step, prev_path = entries[0]
        prev_data = torch.load(prev_path, map_location="cpu", weights_only=False)
        prev_kv = prev_data.get("kv", {})

        for step, path in entries[1:]:
            cur_data = torch.load(path, map_location="cpu", weights_only=False)
            cur_kv = cur_data.get("kv", {})
            for comp_name, prev_t in prev_kv.items():
                if not hasattr(prev_t, "shape"):
                    continue
                cur_t = cur_kv.get(comp_name)
                if cur_t is None or not hasattr(cur_t, "shape"):
                    continue
                n_common = min(prev_t.shape[0], cur_t.shape[0])
                if exclude_last:
                    n_common = max(0, n_common - 1)
                if n_common == 0:
                    continue
                prev_slice = prev_t[:n_common]
                cur_slice = cur_t[:n_common]
                if prev_slice.shape != cur_slice.shape:
                    continue
                total_checks += 1
                req_stats[rid]["checks"] += 1
                if not torch.equal(prev_slice, cur_slice):
                    diff = (prev_slice.float() - cur_slice.float()).abs().max().item()
                    total_corrupted += 1
                    req_stats[rid]["corrupted"] += 1
                    if len(corruption_samples) < 10:
                        corruption_samples.append({
                            "req": rid[:20], "g": g, "layer": layer, "kind": kind,
                            "comp": comp_name, "prev_step": prev_step, "cur_step": step,
                            "n_common_blocks": n_common, "max_abs_diff": round(diff, 4),
                            "prev_blocks": prev_data.get("hbm_block_ids"),
                            "cur_blocks": cur_data.get("hbm_block_ids"),
                        })
            prev_step = step
            prev_data = cur_data
            prev_kv = cur_kv

    print(f"=== KV Self-Consistency Report ===")
    print(f"Branch: {branch_dir}")
    print(f"Total checks: {total_checks}, Corrupted: {total_corrupted}, Clean: {total_checks - total_corrupted}")
    for rid in sorted(req_stats):
        s = req_stats[rid]
        status = "CLEAN" if s["corrupted"] == 0 else "CORRUPTED"
        print(f"  {rid[:40]}: {s['checks']} checks, {s['corrupted']} corrupted [{status}]")
    if corruption_samples:
        print(f"\nSample corruptions:")
        for c in corruption_samples:
            print(f"  req={c['req']} g={c['g']} layer={c['layer']} {c['kind']}.{c['comp']} steps={c['prev_step']}->{c['cur_step']}")
    else:
        print("No corruption detected.")


def transfer_compare(args):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "kv_consistency_check.py"),
        "--dump-dir", args.dump_dir,
        "--request-id", args.request_id,
        "--compare-pairs",
        "prefill_hbm:prefill_host",
        "prefill_host:decode_host",
        "decode_host:decode_hbm",
        "prefill_hbm:decode_hbm",
    ]
    print(f"# Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default="/tmp/kv_dumps")
    ap.add_argument("--mode", choices=["auto", "transfer", "step", "self-consistency"], default="auto")
    ap.add_argument("--request-id", default=None)
    ap.add_argument("--baseline-branch", default="baseline")
    ap.add_argument("--omni-branch", default="omnicache")
    ap.add_argument("--baseline-id", default=None)
    ap.add_argument("--omni-id", default=None)
    ap.add_argument("--all-requests", action="store_true",
                    help="Compare ALL requests with matching IDs across branches")
    ap.add_argument("--max-requests", type=int, default=None,
                    help="Limit to first N requests (use with --all-requests for quick testing)")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--include-last-block", action="store_true",
                     help="Include last block in self-consistency check")
    ap.add_argument("--quick-peek", action="store_true",
                    help="Quick peek mode: only export mismatches to TSV, skip summary")
    ap.add_argument("--mismatch-file", default="mismatches.tsv",
                    help="Export mismatches to TSV file (default: mismatches.tsv)")
    ap.add_argument("--no-mismatch-file", action="store_true",
                    help="Disable mismatch file output")
    args = ap.parse_args()

    mode = args.mode if args.mode != "auto" else detect_mode(args.dump_dir)
    print(f"[kv_dump_compare] mode={mode} dump-dir={args.dump_dir}")
    if mode == "transfer":
        if not args.request_id:
            sys.exit("--request-id required for transfer mode")
        return transfer_compare(args)
    return step_compare(args)


if __name__ == "__main__":
    main()
