#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Compute incremental (diff) coverage from a full coverage.xml using diff-cover.

Workflow (simple):
  1) git diff --cached > demo_diff.txt
  2) rewrite coverage.xml paths by stripping an absolute prefix
  3) run diff-cover with the diff file, export html/txt

Usage example:
  python3 ut_diff_cov.py \
    --repo-root /path/to/repo \
    --coverage-xml /path/to/coverage.xml \
    --old-prefix /workspace/omni \
    --out-html /path/to/diffcov.html \
    --out-txt /path/to/diffcov.txt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def run_cmd(args: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=True)


def ensure_diff_cover() -> str:
    cmd = shutil.which("diff-cover")
    if cmd:
        return cmd
    print("[INFO] diff-cover not found, installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "diff-cover"], check=True)
    cmd = shutil.which("diff-cover")
    if not cmd:
        raise RuntimeError("diff-cover still not found after installation")
    return cmd


def normalize_prefix(p: str) -> str:
    if p != "/" and p.endswith("/"):
        return p[:-1]
    return p


def rewrite_coverage_paths(
    xml_path: Path,
    old_prefix: str,
    new_prefix: str,
    set_source: str | None,
    strip_leading_slash: bool = True,
) -> int:
    old_prefix = normalize_prefix(old_prefix)
    new_prefix = normalize_prefix(new_prefix)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    changed = 0

    if set_source is not None:
        for sources in root.iter("sources"):
            for src in list(sources):
                if src.tag != "source":
                    continue
                if (src.text or "") != set_source:
                    src.text = set_source

    candidates = {old_prefix, "/" + old_prefix}

    for class_elem in root.iter("class"):
        filename = class_elem.get("filename")
        if not filename:
            continue

        fn = filename
        if strip_leading_slash and fn.startswith("/"):
            fn_cmp = fn[1:]
        else:
            fn_cmp = fn

        for pref in candidates:
            pref_cmp = pref[1:] if (strip_leading_slash and pref.startswith("/")) else pref
            if fn_cmp.startswith(pref_cmp):
                tail = fn_cmp[len(pref_cmp):]
                if new_prefix == "":
                    new_fn = tail.lstrip("/")
                else:
                    new_fn = new_prefix.rstrip("/") + "/" + tail.lstrip("/")
                class_elem.set("filename", new_fn)
                changed += 1
                break

    if changed == 0 and set_source is None:
        return 0

    backup_path = xml_path.with_suffix(xml_path.suffix + ".bak")
    shutil.copy(xml_path, backup_path)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"[INFO] backup saved to: {backup_path}")
    print("[INFO] coverage.xml updated.")
    return changed


SUMMARY_TOTAL_RE = re.compile(r"^Total:\s+(\d+)\s+lines", re.IGNORECASE)
SUMMARY_MISSING_RE = re.compile(r"^Missing:\s+(\d+)\s+lines", re.IGNORECASE)
SUMMARY_COVER_RE = re.compile(r"^Coverage:\s+(\d+)%", re.IGNORECASE)
NO_LINES_RE = re.compile(r"^No lines with coverage information in this diff\.", re.IGNORECASE)


def parse_diff_cover_summary(text: str) -> tuple[int | None, int | None, int | None, bool]:
    total = missing = cover = None
    no_lines = False
    for raw in text.splitlines():
        line = raw.strip()
        if NO_LINES_RE.match(line):
            no_lines = True
            return 0, 0, 100, no_lines
        m = SUMMARY_TOTAL_RE.match(line)
        if m:
            total = int(m.group(1))
            continue
        m = SUMMARY_MISSING_RE.match(line)
        if m:
            missing = int(m.group(1))
            continue
        m = SUMMARY_COVER_RE.match(line)
        if m:
            cover = int(m.group(1))
            continue
    return total, missing, cover, no_lines


def print_summary(total: int, missing: int) -> None:
    covered = max(total - missing, 0)
    cover_pct = int(round((covered / total) * 100)) if total > 0 else 100
    print(f"[INFO] diff covered lines : {covered}")
    print(f"[INFO] diff missing lines : {missing}")
    print(f"[INFO] diff coverage       : {cover_pct}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute diff coverage using diff-cover.")
    ap.add_argument("--repo-root", required=True, help="git repo root (required)")
    ap.add_argument("--coverage-xml", required=True, help="coverage xml file")
    ap.add_argument(
        "--old-prefix",
        default="",
        help="absolute path prefix to strip/replace (leave empty to skip rewrite)",
    )
    ap.add_argument("--new-prefix", default="", help="replace prefix with this (default: empty)")
    ap.add_argument("--out-html", required=True, help="html report output path")
    ap.add_argument("--out-txt", required=True, help="text report output path")
    ap.add_argument("--diff-file", default="", help="use an existing diff file (optional)")
    ap.add_argument("--set-source", default=".", help="rewrite <sources><source>...</source> to this value")
    ap.add_argument("--no-strip-leading-slash", action="store_true", help="do not strip leading '/' before match")
    ap.add_argument("--min", type=float, default=None, help="minimum diff coverage percent")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    cov_xml = os.path.abspath(args.coverage_xml)

    if not os.path.isdir(repo_root):
        print(f"[ERROR] repo root not found: {repo_root}", file=sys.stderr)
        return 2
    if not os.path.isfile(cov_xml):
        print(f"[ERROR] coverage xml not found: {cov_xml}", file=sys.stderr)
        return 2

    diff_cover_cmd = ensure_diff_cover()

    diff_file = args.diff_file
    if not diff_file:
        diff_file = os.path.join(repo_root, "demo_diff.txt")
        try:
            cp = run_cmd(["git", "diff", "--cached"], cwd=repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr.strip() if exc.stderr else str(exc)
            print(f"[ERROR] git diff --cached failed: {msg}", file=sys.stderr)
            return 2
        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(cp.stdout)
        print(f"[INFO] diff written: {diff_file}")

    if os.path.isfile(diff_file) and os.path.getsize(diff_file) == 0:
        print("[INFO] diff is empty; no changed lines to measure.")
        print_summary(total=0, missing=0)
        return 0

    set_source = None if args.set_source == "NONE" else args.set_source
    if args.old_prefix:
        rewrite_coverage_paths(
            xml_path=Path(cov_xml),
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
            set_source=set_source,
            strip_leading_slash=not args.no_strip_leading_slash,
        )

    cmd = [
        diff_cover_cmd,
        cov_xml,
        "--diff-file",
        diff_file,
        "--format",
        f"html:{args.out_html}",
    ]
    print(f"[INFO] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    try:
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write(proc.stdout or "")
        print(f"[INFO] wrote text report: {args.out_txt}")
    except Exception as exc:
        print(f"[WARN] failed to write text report: {exc}", file=sys.stderr)

    combined_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    total, missing, cover, no_lines = parse_diff_cover_summary(combined_text)
    if total is None or missing is None:
        return proc.returncode

    if no_lines:
        print("[INFO] no lines with coverage information in diff; treating coverage as 100%.")

    print_summary(total=total, missing=missing)

    if total == 0:
        print("[INFO] no lines in diff; treating coverage as 100%.")
        return 0

    if args.min is not None:
        if cover is None:
            cover = int(round(((total - missing) / total) * 100))
        if cover < args.min:
            print(f"[ERROR] diff coverage below threshold: {cover}% < {args.min:.0f}%", file=sys.stderr)
            return 1
        print(f"[INFO] diff coverage threshold OK: {cover}% >= {args.min:.0f}%")

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
