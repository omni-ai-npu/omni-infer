#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Parse coverage rate from a text report (coverage report output).

usage
  python3 ut_CI_cover_rate_check.py --report /path/to/coverage_report.txt
  python3 ut_CI_cover_rate_check.py --report /path/to/coverage_report.txt --min 60
"""

from __future__ import annotations

import argparse
import re
import sys

TOTAL_LINE_RE = re.compile(r"^TOTAL\s+.*?(\d+(?:\.\d+)?)%\s*$")


def parse_total_percent(text: str) -> float:
    for raw in text.splitlines():
        line = raw.strip()
        m = TOTAL_LINE_RE.match(line)
        if m:
            return float(m.group(1))
    raise ValueError("TOTAL line with percent not found in report")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="coverage report txt file")
    ap.add_argument("--min", type=float, default=None, help="minimum coverage percent")
    args = ap.parse_args()

    try:
        with open(args.report, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"[ERROR] report not found: {args.report}", file=sys.stderr)
        raise SystemExit(2)

    try:
        percent = parse_total_percent(text)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(2)

    print(f"[INFO] coverage total: {percent:.0f}%")
    if args.min is not None:
        if percent < args.min:
            print(f"[ERROR] coverage below threshold: {percent:.0f}% < {args.min:.0f}%")
            raise SystemExit(1)
        print(f"[INFO] coverage threshold OK: {percent:.0f}% >= {args.min:.0f}%")


if __name__ == "__main__":
    main()
