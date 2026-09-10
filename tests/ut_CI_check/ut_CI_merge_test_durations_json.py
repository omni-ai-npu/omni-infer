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


def collect_durations(
    container_json_paths: List[str],
) -> Tuple[Dict[str, float], List[Tuple[str, str, float, float]]]:
    merged: Dict[str, float] = {}
    duplicates: List[Tuple[str, str, float, float]] = []

    for path in container_json_paths:
        if not os.path.isfile(path):
            print(f"[WARN] missing file: {path}")
            continue
        name = infer_name(path)
        data = load_json(path)

        for k, v in data.items():
            if k in merged and abs(merged[k] - v) > 1e-9:
                duplicates.append((name, k, merged[k], v))
                merged[k] = max(merged[k], v)
            else:
                merged[k] = v

    return merged, duplicates


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "usage\n"
            "  python3 ut_CI_merge_test_durations_json.py --out /path/merged.json "
            "--container-json a.json --container-json b.json"
        )
    )
    ap.add_argument("--out", required=True, help="output merged json path")
    ap.add_argument("--container-json", action="append", default=[], help="path to a container json (repeatable)")
    args = ap.parse_args()

    if not args.container_json:
        print("[ERROR] no --container-json provided", file=sys.stderr)
        return 2

    merged, duplicates = collect_durations(args.container_json)

    if not merged:
        print("[ERROR] no data loaded from container json files", file=sys.stderr)
        return 3

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True, ensure_ascii=False)

    print(f"[INFO] merged durations json written: {args.out}")
    print(f"[INFO] total tests: {len(merged)}")
    if duplicates:
        print(f"[WARN] duplicate nodeids detected: {len(duplicates)} (kept max duration)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
