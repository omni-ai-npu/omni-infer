# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Opt-in debug helpers for OmniCache diagnostics."""

from __future__ import annotations

import os
from typing import Any, Mapping


def apc_debug_enabled() -> bool:
    return os.getenv("OMNI_CACHE_APC_DEBUG", "0") == "1"


def _rank_from_env() -> int | None:
    for name in ("TP_RANK", "RANK", "LOCAL_RANK", "VLLM_RANK"):
        value = os.getenv(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def should_log_rank(cache: Any = None) -> bool:
    if not apc_debug_enabled():
        return False
    if os.getenv("OMNI_CACHE_APC_DEBUG_ALL_RANKS", "0") == "1":
        return True
    from vllm.distributed import get_tp_group

    return int(get_tp_group().rank) == 0


def _shape_dtype(value: Any) -> tuple[Any, Any]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None and isinstance(value, (list, tuple)):
        shape = (len(value),)
    return shape, dtype


def _preview(value: Any, rows: int, cols: int) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dim() == 0:
            return tensor.cpu().item()
        if tensor.dim() == 1:
            return tensor[:cols].cpu().tolist()
        return tensor[:rows, :cols].cpu().tolist()

    import numpy as np

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.ndim == 1:
            return value[:cols].tolist()
        return value[:rows, :cols].tolist()

    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return [list(row[:cols]) for row in value[:rows]]
        return list(value[:cols])
    return repr(value)


def summarize_array(name: str, value: Any, rows: int = 2, cols: int = 36) -> str:
    shape, dtype = _shape_dtype(value)
    return (
        f"{name}.shape={shape} {name}.dtype={dtype} "
        f"{name}.preview={_preview(value, rows, cols)}"
    )


def summarize_map(name: str, mapping: Mapping[Any, Any] | None, limit: int = 36) -> str:
    if not mapping:
        return f"{name}.len=0 {name}.items=[]"
    items = list(mapping.items())[:limit]
    return f"{name}.len={len(mapping)} {name}.items={items}"
