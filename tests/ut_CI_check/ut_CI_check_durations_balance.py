#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

# usage
# python3 ut_CI_check_durations_balance.py --dir /path/to/dir --threshold 1.5
# python3 ut_CI_check_durations_balance.py --container-json a.json --container-json b.json


def load_json(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a dict")
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except Exception as exc:
            raise ValueError(f"{path} has non-numeric duration for {k}: {v}") from exc
    return out


def infer_name(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"test_durations_(.+)\.json$", base)
    if m:
        return m.group(1)
    return base


def collect_totals(paths: List[str]) -> List[Tuple[str, float]]:
    totals: List[Tuple[str, float]] = []
    for path in paths:
        if not os.path.isfile(path):
            print(f"[WARN] missing file: {path}")
            continue
        name = infer_name(path)
        data = load_json(path)
        totals.append((name, sum(data.values())))
    return totals


def balance_ratio(totals: List[Tuple[str, float]]) -> float:
    if not totals:
        return 1.0
    min_t = min(t for _, t in totals)
    max_t = max(t for _, t in totals)
    if min_t <= 0.0:
        return float("inf") if max_t > 0 else 1.0
    return max_t / min_t


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "usage\n"
            "  python3 ut_CI_check_durations_balance.py --dir /path/to/dir\n"
            "  python3 ut_CI_check_durations_balance.py --container-json a.json --container-json b.json"
        )
    )
    ap.add_argument("--container-json", action="append", default=[], help="path to a container json (repeatable)")
    ap.add_argument("--dir", default="", help="directory containing test_durations_*.json")
    ap.add_argument("--pattern", default="test_durations_*.json", help="glob pattern when using --dir")
    args = ap.parse_args()

    paths = list(args.container_json)
    if args.dir:
        import glob
        paths.extend(sorted(glob.glob(os.path.join(args.dir, args.pattern))))

    if not paths:
        print("[ERROR] no container json files provided", file=sys.stderr)
        return 2

    totals = collect_totals(paths)
    if not totals:
        print("[ERROR] no data loaded from container json files", file=sys.stderr)
        return 3

    print("[INFO] per-container total durations (seconds):")
    for name, total in sorted(totals, key=lambda x: x[0]):
        print(f"  {name}: {total:.2f}")

    threshold = 2
    ratio = balance_ratio(totals)
    ok = ratio <= threshold
    print(f"[INFO] balance check: max/min = {ratio:.2f} (threshold {threshold:.2f})")
    if not ok:
        print("[WARN] containers are imbalanced. Please update durations.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
