# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Atomic shard write + integrity check + resume — model/method-agnostic.

Pure I/O. The print() resume messages are kept for parity.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Set, Tuple

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

_LOGGER = logging.getLogger(__name__)


def atomic_save(tensors: Dict[str, torch.Tensor], path: Path) -> None:
    """
    Write safetensors via tmp file + atomic rename to prevent corruption.

    Falls back to copy+unlink when os.replace() is blocked (e.g. SFS Turbo /
    Lustre network filesystems raise EPERM on cross-entry rename).
    """
    tmp = path.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp))
    try:
        os.replace(str(tmp), str(path))
    except OSError:
        # Network filesystems (SFS Turbo, NFS, Lustre) may not support atomic
        # rename. Copy then delete is not crash-atomic but is functionally safe
        # because verify_shard() detects a truncated file on resume.
        import shutil
        shutil.copy2(str(tmp), str(path))
        try:
            os.unlink(str(tmp))
        except OSError as error:
            _LOGGER.warning("failed to remove temporary checkpoint %s: %s", tmp, error)


def verify_shard(path: Path) -> bool:
    """Check if a safetensors shard is readable and not truncated."""
    try:
        t = load_file(str(path))
        if len(t) == 0:
            return False
        # Spot check: touch first tensor's data to verify it's not corrupted.
        first_key = next(iter(t))
        _ = t[first_key].sum()
        del t
        return True
    except (OSError, RuntimeError, SafetensorError, StopIteration, ValueError):
        return False


def save_checkpoint(output_dir: Path, last_layer: int, written_shards: Set[str]) -> None:
    """Save resume checkpoint to output dir."""
    ckpt = {
        "last_completed_layer": last_layer,
        "written_shards": sorted(written_shards),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ckpt_path = output_dir / ".quantize_checkpoint.json"
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f, indent=2)


def load_checkpoint(output_dir: Path, weight_map: Dict[str, str]) -> Tuple[int, Set[str]]:
    """
    Auto-detect resume point from checkpoint + shard integrity.

    `weight_map` maps tensor-name -> shard-filename (the model's safetensors
    index), used to find which layers a corrupted shard touches.

    Returns (start_layer, verified_shards).
    """
    ckpt_path = output_dir / ".quantize_checkpoint.json"
    if not ckpt_path.exists():
        return 0, set()

    with open(ckpt_path) as f:
        ckpt = json.load(f)

    recorded_shards = set(ckpt.get("written_shards", []))
    last_layer = ckpt.get("last_completed_layer", -1)

    # Verify each recorded shard
    verified: Set[str] = set()
    corrupted = []
    for sfn in recorded_shards:
        path = output_dir / sfn
        if not path.exists():
            corrupted.append(sfn)
            continue
        if verify_shard(path):
            verified.add(sfn)
        else:
            corrupted.append(sfn)

    if corrupted:
        print(f"  [resume] Corrupted/missing shards detected: {corrupted}")
        for sfn in corrupted:
            p = output_dir / sfn
            if p.exists():
                p.unlink()
                print(f"    Deleted corrupted: {sfn}")

        # Find the earliest layer that touches a corrupted shard — re-process from there.
        earliest_bad_layer = last_layer + 1
        for sfn in corrupted:
            for name, s in weight_map.items():
                if s == sfn and "layers." in name:
                    parts = name.split(".")
                    li = int(parts[parts.index("layers") + 1])
                    earliest_bad_layer = min(earliest_bad_layer, li)

        # Remove shards for layers >= earliest_bad_layer from the verified set.
        bad_shards = set()
        for sfn in verified:
            for name, s in weight_map.items():
                if s == sfn and "layers." in name:
                    parts = name.split(".")
                    li = int(parts[parts.index("layers") + 1])
                    if li >= earliest_bad_layer:
                        bad_shards.add(sfn)
                        break
        verified -= bad_shards
        resume_layer = earliest_bad_layer
    else:
        resume_layer = last_layer + 1

    print(f"  [resume] Checkpoint: last_layer={last_layer}, "
          f"verified={len(verified)} shards, resume from layer {resume_layer}")
    return resume_layer, verified
