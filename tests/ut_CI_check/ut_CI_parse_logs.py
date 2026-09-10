#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Parse multiple pytest logs, aggregate summary counts and failed/error nodeids,
and write a merged log for local retention.

usage
  python3 ut_CI_parse_logs.py --log a.log --log b.log --merged-log merged.log
  python3 ut_CI_parse_logs.py --log a.log --known known_fails.txt
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
HTML_RE = re.compile(r"<[^>]*>")
NODEID_RE = re.compile(r"(?P<nodeid>\S+::\S+)")
SUMMARY_LINE_RE = re.compile(r"^=+.*?\bin\b.*?=+$")

COUNT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("failed", re.compile(r"(\d+)\s+failed\b")),
    ("passed", re.compile(r"(\d+)\s+passed\b")),
    ("errors", re.compile(r"(\d+)\s+errors?\b")),
    ("skipped", re.compile(r"(\d+)\s+skipped\b")),
    ("warnings", re.compile(r"(\d+)\s+warnings?\b")),
    ("xfailed", re.compile(r"(\d+)\s+xfailed\b")),
    ("xpassed", re.compile(r"(\d+)\s+xpassed\b")),
    ("deselected", re.compile(r"(\d+)\s+deselected\b")),
    ("rerun", re.compile(r"(\d+)\s+rerun\b")),
]


def _clean_line(s: str) -> str:
    s = ANSI_RE.sub("", s)
    s = HTML_RE.sub("", s)
    return s.rstrip("\n")


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_counts_from_summary_line(line: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for k, pat in COUNT_PATTERNS:
        m = pat.search(line)
        if m:
            counts[k] = counts.get(k, 0) + int(m.group(1))
    return counts


def _extract_nodeid(line: str) -> str | None:
    line = line.strip()
    if line.startswith("FAILED "):
        rest = line[len("FAILED "):].strip()
    elif line.startswith("ERROR "):
        rest = line[len("ERROR "):].strip()
    else:
        return None

    rest = rest.split(" - ", 1)[0].strip()
    m = NODEID_RE.search(rest)
    if m:
        return m.group("nodeid")
    if "::" in rest:
        return rest.split()[0].strip()
    return None


@dataclass
class ParseResult:
    sessions: int
    counts: Dict[str, int]
    failed_tests: List[str]
    error_tests: List[str]
    summary_lines: List[str]


@dataclass
class KnownFailureCheck:
    ok: bool
    actual_total: int
    known_total: int
    new_failures: List[str]
    fixed_failures: List[str]
    remaining_known: List[str]


def parse_log_text(text: str) -> ParseResult:
    raw_lines = text.splitlines()
    lines = [_clean_line(x) for x in raw_lines]

    sessions = 0
    counts_total: Dict[str, int] = {k: 0 for k, _ in COUNT_PATTERNS}
    summary_lines: List[str] = []

    for ln in lines:
        if SUMMARY_LINE_RE.match(ln) and any(p.search(ln) for _, p in COUNT_PATTERNS):
            sessions += 1
            summary_lines.append(ln)
            c = _parse_counts_from_summary_line(ln)
            for k, v in c.items():
                counts_total[k] = counts_total.get(k, 0) + v

    failed: List[str] = []
    errors: List[str] = []
    for ln in lines:
        if ln.startswith("FAILED "):
            nid = _extract_nodeid(ln)
            if nid:
                failed.append(nid)
        elif ln.startswith("ERROR "):
            nid = _extract_nodeid(ln)
            if nid:
                errors.append(nid)

    if not failed and not errors:
        in_summary = False
        for ln in lines:
            if "short test summary info" in ln:
                in_summary = True
                continue
            if in_summary and SUMMARY_LINE_RE.match(ln):
                in_summary = False
                continue
            if not in_summary:
                continue
            if ln.startswith("FAILED "):
                nid = _extract_nodeid(ln)
                if nid:
                    failed.append(nid)
            elif ln.startswith("ERROR "):
                nid = _extract_nodeid(ln)
                if nid:
                    errors.append(nid)

    return ParseResult(
        sessions=sessions,
        counts=counts_total,
        failed_tests=_uniq_keep_order(failed),
        error_tests=_uniq_keep_order(errors),
        summary_lines=summary_lines,
    )


def parse_log_file(path: str) -> ParseResult:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_log_text(f.read())


def aggregate(results: List[ParseResult]) -> ParseResult:
    total_sessions = sum(r.sessions for r in results)
    all_keys = set()
    for r in results:
        all_keys.update(r.counts.keys())
    total_counts = {k: 0 for k in all_keys}
    for r in results:
        for k, v in r.counts.items():
            total_counts[k] = total_counts.get(k, 0) + int(v)

    all_failed = _uniq_keep_order([t for r in results for t in r.failed_tests])
    all_error = _uniq_keep_order([t for r in results for t in r.error_tests])
    all_summary = [ln for r in results for ln in r.summary_lines]

    return ParseResult(
        sessions=total_sessions,
        counts=total_counts,
        failed_tests=all_failed,
        error_tests=all_error,
        summary_lines=all_summary,
    )


def load_known_failures_txt(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "::" not in line:
                continue
            out.append(line)
    return _uniq_keep_order(out)


def check_known_failures(result: ParseResult, known: List[str]) -> KnownFailureCheck:
    actual_list = _uniq_keep_order(result.failed_tests + result.error_tests)
    actual_set = set(actual_list)
    known_set = set(known)

    new_failures = [x for x in actual_list if x not in known_set]
    remaining_known = [x for x in actual_list if x in known_set]
    fixed_failures = sorted([x for x in known_set if x not in actual_set])

    return KnownFailureCheck(
        ok=(len(new_failures) == 0),
        actual_total=len(actual_list),
        known_total=len(known_set),
        new_failures=new_failures,
        fixed_failures=fixed_failures,
        remaining_known=remaining_known,
    )


def write_merged_log(logs: List[str], merged_path: str) -> None:
    os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as out:
        for p in logs:
            out.write(f"===== LOG BEGIN: {p} =====\n")
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    out.write(f.read())
            except FileNotFoundError:
                out.write(f"[WARN] log file not found: {p}\n")
            out.write(f"\n===== LOG END: {p} =====\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True, help="pytest log file path (repeatable)")
    ap.add_argument("--merged-log", default="", help="optional: write merged log to this path")
    ap.add_argument("--known", default="", help="known failures txt path (optional)")
    args = ap.parse_args()

    logs = args.log
    results = [parse_log_file(p) for p in logs]
    merged = aggregate(results)

    if args.merged_log:
        write_merged_log(logs, args.merged_log)
        print(f"[INFO] merged log written: {args.merged_log}")

    print(f"sessions: {merged.sessions}")
    for k in ["passed", "failed", "errors", "skipped", "warnings", "xfailed", "xpassed", "rerun"]:
        if k in merged.counts:
            print(f"{k}: {merged.counts.get(k, 0)}")
    print(f"failed_tests: {len(merged.failed_tests)}")
    print(f"error_tests : {len(merged.error_tests)}")
    if merged.failed_tests:
        print("failed_tests:")
        for t in merged.failed_tests:
            print(f"  {t}")
    if merged.error_tests:
        print("error_tests:")
        for t in merged.error_tests:
            print(f"  {t}")

    if args.known:
        known = load_known_failures_txt(args.known)
        check = check_known_failures(merged, known)
        print(f"known_total: {check.known_total}")
        print(f"actual_total: {check.actual_total}")
        print(f"new_failures: {len(check.new_failures)}")
        print(f"fixed_failures: {len(check.fixed_failures)}")
        print(f"remaining_known: {len(check.remaining_known)}")
        if check.new_failures:
            print("new_failures:")
            for t in check.new_failures:
                print(f"  {t}")
        if check.fixed_failures:
            print("fixed_failures:")
            for t in check.fixed_failures:
                print(f"  {t}")
        if check.remaining_known:
            print("remaining_known:")
            for t in check.remaining_known:
                print(f"  {t}")


if __name__ == "__main__":
    main()
