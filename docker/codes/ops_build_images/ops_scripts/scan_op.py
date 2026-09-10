#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path


# ==================== 算子根定位（复用 V4 体系）====================
LEAF_DIRS = {
    "op_host",
    "op_kernel",
    "op_kernel_aicpu",
    "op_api",
    "op_tiling",
    "op_graph",
    "tests",
    "docs",
    "csrc",
    "converter",
}
BUCKET_PREFIXES = ("ascendc/src",)


def find_op_root_from_disk(fpath, base):
    """从文件路径反向定位算子根目录的 relative path（相对于 base）。

    规则：文件必须在某个 op_host 目录下，op_host 的父目录即算子根。
    返回 None 表示不在有效算子结构内。
    """
    try:
        rel = fpath.relative_to(base)
    except ValueError:
        return None

    parts = rel.parts
    if "op_host" not in parts:
        return None

    op_host_idx = parts.index("op_host")
    if op_host_idx == 0:
        return None

    op_root_parts = parts[:op_host_idx]
    return "/".join(op_root_parts)


def get_scan_bases(dir_path, op):
    """根据 BUCKET_PREFIXES 生成所有扫描起点"""
    op_path = Path(dir_path) / op
    bases = []
    for prefix in BUCKET_PREFIXES:
        base = op_path / prefix
        if base.is_dir():
            bases.append(base)
    return bases


def scan_configs(dir_path: str, op: str, devices_str: str) -> str:
    """
    扫描目录: ${DIR_PATH}/${op}/{BUCKET_PREFIXES}/**/op_host/*_def.cpp

    规则:
    1. device 支持逗号分隔多值，如 "A,B,C"
    2. 包含任一 .AddConfig("X") 即算匹配（OR 关系）
    3. 若匹配，提取 _def.cpp 前缀加入 set
    4. 检查 {prefix}_metadata/op_host/ 存在且无 *_def.cpp，则额外加入 {prefix}_metadata
    """
    devices = [d.strip() for d in devices_str.split(",") if d.strip()]
    if not devices:
        return ""

    target_strs = [f'.AddConfig("{d}"' for d in devices]

    result_set = set()

    bases = get_scan_bases(dir_path, op)
    if not bases:
        return ""

    for base in bases:
        for fpath in base.rglob("*_def.cpp"):
            op_root_rel = find_op_root_from_disk(fpath, base)
            if op_root_rel is None:
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except (IOError, OSError):
                continue

            # ===== OR 关系：包含任一 target_str 即匹配 =====
            if not any(t in content for t in target_strs):
                continue

            prefix = fpath.name[:-8]
            result_set.add(prefix)

            # metadata 规则
            op_root_abs = base / op_root_rel
            op_root_parent = op_root_abs.parent
            metadata_dir = op_root_parent / f"{prefix}_metadata"

            if metadata_dir.is_dir():
                metadata_op_host = metadata_dir / "op_host"
                if metadata_op_host.is_dir() and not any(
                    metadata_op_host.glob("*_def.cpp")
                ):
                    result_set.add(f"{prefix}_metadata")

    return ";".join(sorted(result_set)) if result_set else ""


def main():
    parser = argparse.ArgumentParser(
        description="Scan ascendc src op_host with multi-device OR filter."
    )
    parser.add_argument("dir_path", help="根目录路径")
    parser.add_argument("op", help="op 子目录名")
    parser.add_argument("devices", help="设备名，支持逗号分隔多值，如 A,B,C")
    args = parser.parse_args()

    print(scan_configs(args.dir_path, args.op, args.devices))


if __name__ == "__main__":
    main()
