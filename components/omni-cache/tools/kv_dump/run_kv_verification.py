#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""One-shot KV-cache verification: baseline vs omnicache under mock schedule.

Sends concurrent requests to two runs (baseline + omnicache), collects KV
dumps and full responses, then compares layer-by-layer KV byte-equivalence
and token-level logP alignment.

Usage::

    python tools/kv_dump/run_kv_verification.py -n 8 --max-tokens 5 --ignore-eos

The caller is responsible for starting the PD servers with the correct env
vars BEFORE running this script:

  # Run 1: baseline
  ENABLE_OMNI_CACHE=0 OMNI_MOCK_SCHEDULE=1 ...
  python tools/kv_dump/run_kv_verification.py -n 8 --max-tokens 5 --ignore-eos --mode baseline

  # Run 2: omnicache (after restarting servers with ENABLE_OMNI_CACHE=1)
  ENABLE_OMNI_CACHE=1 OMNI_MOCK_SCHEDULE=1 ...
  python tools/kv_dump/run_kv_verification.py -n 8 --max-tokens 5 --ignore-eos --mode omnicache

  # Compare (after both runs complete):
  python tools/kv_dump/run_kv_verification.py --compare
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent  # tools/
REPO_DIR = TOOLS_DIR.parent  # repo root

DEFAULT_DUMP_STEP = "/tmp/kv_dumps_step"
DEFAULT_RESPONSES = "/tmp/kv_responses"
_SAFE_BRANCH_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _run(cmd: list[str], **kwargs):
    print(f"  RUN: {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, cwd=str(REPO_DIR), **kwargs)


def _validate_branch_name(branch: str) -> str:
    if not _SAFE_BRANCH_RE.fullmatch(branch):
        raise ValueError(
            f"unsafe dump branch name: {branch!r}; expected {_SAFE_BRANCH_RE.pattern}"
        )
    return branch


def _resolve_under_root(root: str, child: str) -> Path:
    root_path = Path(root).resolve()
    target = (root_path / child).resolve()
    if root_path != target and root_path not in target.parents:
        raise ValueError(f"resolved path escapes dump root: {target}")
    return target


def _send_requests(n: int, max_tokens: int, id_prefix: str,
                   ignore_eos: bool, vary: bool, output_dir: str,
                   url: str, model: str,
                   prompts_file: str | None = None,
                   prompts_key: str = "prompt",
                   seed: int = 42):
    cmd = [
        sys.executable, str(TOOLS_DIR / "scripts" / "send_concurrent.py"),
        "-n", str(n),
        "--max-tokens", str(max_tokens),
        "--id-prefix", id_prefix,
        "--url", url,
        "--model", model,
        "--save-dir", output_dir,
        "--logprobs",
    ]
    if ignore_eos:
        cmd.append("--ignore-eos")
    if vary:
        cmd.append("--vary")
    if prompts_file:
        cmd.extend(["--prompts-file", prompts_file])
        cmd.extend(["--prompts-key", prompts_key])
        cmd.extend(["--seed", str(seed)])
    _run(cmd)




def do_send(args):
    """Send requests for one run (baseline or omnicache)."""
    branch = _validate_branch_name(
        args.mode if args.mode else os.environ.get("OMNI_KV_DUMP_BRANCH", "run")
    )
    resp_dir = os.path.join(args.responses_dir, branch)

    # Clean up previous step dumps
    target = _resolve_under_root(DEFAULT_DUMP_STEP, branch)
    if target.exists():
        _run(["rm", "-rf", str(target)])

    print(f"\n=== Sending {args.n} requests ({branch}) ===")
    _send_requests(
        n=args.n,
        max_tokens=args.max_tokens,
        id_prefix=args.id_prefix,
        ignore_eos=args.ignore_eos,
        vary=args.vary,
        output_dir=resp_dir,
        url=args.url,
        model=args.model,
        prompts_file=args.prompts_file,
        prompts_key=args.prompts_key,
        seed=args.seed,
    )
    print(f"  Responses saved to {resp_dir}")


def do_compare(args):
    """Compare KV dumps and logP between baseline and omnicache."""
    dump_dir = args.dump_dir or DEFAULT_DUMP_STEP
    bl_ok = Path(dump_dir, "baseline").is_dir()
    oc_ok = Path(dump_dir, "omnicache").is_dir()

    if bl_ok and oc_ok:
        print(f"\n=== KV dump comparison (dir={dump_dir}) ===")
        try:
            _run([
                sys.executable, str(TOOLS_DIR / "kv_dump" / "kv_dump_compare.py"),
                "--dump-dir", dump_dir,
                "--mode", "step",
                "--all-requests",
                "--baseline-branch", "baseline",
                "--omni-branch", "omnicache",
            ])
        except subprocess.CalledProcessError:
            print("  WARNING: KV comparison failed (missing torch dependency?)")

        if args.analyze:
            try:
                _run([
                    sys.executable, str(TOOLS_DIR / "kv_dump" / "kv_dump_analyze.py"),
                    "--dump-dir", dump_dir,
                    "--id-prefix", args.id_prefix,
                ])
            except subprocess.CalledProcessError:
                print("  WARNING: KV analysis failed")
    else:
        print(f"\n=== Skipping KV dump comparison (missing {dump_dir}/baseline or /omnicache) ===")

    # --- logP comparison ---
    bl_resp = Path(args.responses_dir) / "baseline"
    oc_resp = Path(args.responses_dir) / "omnicache"
    if not bl_resp.is_dir() or not oc_resp.is_dir():
        print("\n=== Skipping logP comparison (no response dirs) ===")
        return

    print("\n=== logP comparison ===")
    _compare_logp(bl_resp, oc_resp, args.id_prefix)


def _compare_logp(bl_dir: Path, oc_dir: Path, id_prefix: str):
    """Compare token-level log probabilities between baseline and omnicache."""
    match = mismatch = missing = 0
    per_request: dict[str, dict] = {}

    for fname in sorted(os.listdir(bl_dir)):
        if not fname.startswith(id_prefix) or not fname.endswith(".json"):
            continue
        bl_path = bl_dir / fname
        oc_path = oc_dir / fname
        if not oc_path.is_file():
            missing += 1
            continue

        with open(bl_path) as f:
            bl = json.load(f)
        with open(oc_path) as f:
            oc = json.load(f)

        bl_choices = bl.get("choices", [])
        oc_choices = oc.get("choices", [])
        if not bl_choices or not oc_choices:
            continue

        bl_logprobs = bl_choices[0].get("logprobs")
        oc_logprobs = oc_choices[0].get("logprobs")
        bl_content = bl_choices[0].get("message", {}).get("content", "")
        oc_content = oc_choices[0].get("message", {}).get("content", "")

        bl_tokens = _extract_tokens(bl_choices[0])
        oc_tokens = _extract_tokens(oc_choices[0])

        r = {"match_tokens": 0, "mismatch_tokens": 0, "diffs": []}
        if bl_logprobs and oc_logprobs:
            bl_lp = bl_logprobs.get("content", []) or []
            oc_lp = oc_logprobs.get("content", []) or []
            max_t = max(len(bl_tokens), len(oc_tokens))
            for t in range(max_t):
                bl_tok = bl_tokens[t] if t < len(bl_tokens) else "<missing>"
                oc_tok = oc_tokens[t] if t < len(oc_tokens) else "<missing>"
                bl_lp_val = bl_lp[t].get("logprob") if t < len(bl_lp) else None
                oc_lp_val = oc_lp[t].get("logprob") if t < len(oc_lp) else None
                if bl_tok == oc_tok and bl_lp_val == oc_lp_val:
                    r["match_tokens"] += 1
                    match += 1
                else:
                    r["mismatch_tokens"] += 1
                    mismatch += 1
                    if len(r["diffs"]) < 10:
                        r["diffs"].append({
                            "pos": t,
                            "bl_token": bl_tok, "oc_token": oc_tok,
                            "bl_logp": bl_lp_val, "oc_logp": oc_lp_val,
                        })
        else:
            # No logprobs — compare content token-by-token
            for t, (bt, ot) in enumerate(zip(bl_tokens, oc_tokens)):
                if bt == ot:
                    r["match_tokens"] += 1
                    match += 1
                else:
                    r["mismatch_tokens"] += 1
                    mismatch += 1
                    if len(r["diffs"]) < 10:
                        r["diffs"].append({
                            "pos": t, "bl_token": bt, "oc_token": ot,
                        })

        if bl_content != oc_content:
            r["content_mismatch"] = True

        per_request[fname] = r
        status = "MATCH" if r["mismatch_tokens"] == 0 else f"MISMATCH ({r['mismatch_tokens']} tokens)"
        print(f"  {fname}: {status}")

    print(f"\n  Summary: {match}/{match+mismatch} tokens match across {len(per_request)} requests")
    if missing:
        print(f"  ({missing} requests missing from omnicache)")
    if mismatch == 0:
        print("  ALL MATCH")
    else:
        for req, r in per_request.items():
            if r["mismatch_tokens"] > 0:
                print(f"\n  First diffs for {req}:")
                for d in r["diffs"]:
                    bl_lp_str = f" lp={d.get('bl_logp', '?')}" if "bl_logp" in d else ""
                    oc_lp_str = f" lp={d.get('oc_logp', '?')}" if "oc_logp" in d else ""
                    print(f"    pos={d['pos']}: BL='{d['bl_token']}'{bl_lp_str} OC='{d['oc_token']}'{oc_lp_str}")


def _extract_tokens(choice: dict) -> list[str]:
    """Extract token strings from a choice, preferring logprobs content."""
    logprobs = choice.get("logprobs", {})
    content = logprobs.get("content")
    if content:
        return [t.get("token", "") for t in content]
    text = choice.get("text", "") or choice.get("message", {}).get("content", "")
    return [text] if text else []


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="action")

    # --- send ---
    p_send = sp.add_parser("send", help="Send requests for one run")
    p_send.add_argument("-n", type=int, required=True,
                        help="Concurrent requests (must match server --max-num-seqs)")
    p_send.add_argument("--max-tokens", type=int, default=20)
    p_send.add_argument("--ignore-eos", action="store_true")
    p_send.add_argument("--vary", action="store_true",
                        help="Use varied prompts per request")
    p_send.add_argument("--prompts-file", default=None,
                        help="JSONL file to load prompts from")
    p_send.add_argument("--prompts-key", default="prompt",
                        help="Key in JSONL objects")
    p_send.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt selection")
    p_send.add_argument("--id-prefix", default="mytest")
    p_send.add_argument("--url", default="http://localhost:7077/v1/chat/completions")
    p_send.add_argument("--model", default="deepseek")
    p_send.add_argument("--mode", default=None,
                        help="baseline | omnicache | custom-branch-name")
    p_send.add_argument("--responses-dir", default=DEFAULT_RESPONSES)

    # --- compare ---
    p_cmp = sp.add_parser("compare", help="Compare KV dumps + logP")
    p_cmp.add_argument("--dump-dir", default=DEFAULT_DUMP_STEP)
    p_cmp.add_argument("--responses-dir", default=DEFAULT_RESPONSES)
    p_cmp.add_argument("--id-prefix", default="mytest")
    p_cmp.add_argument("--analyze", action="store_true",
                       help="Also generate HTML analysis report")

    args = ap.parse_args()

    if args.action == "send":
        do_send(args)
    elif args.action == "compare":
        do_compare(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
