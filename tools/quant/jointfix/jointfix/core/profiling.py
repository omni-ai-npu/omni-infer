# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Opt-in profiling for the smooth-search hot path — sizes the batched expert search.

Default OFF (zero overhead). Enable with the env var ``JOINTFIX_PROFILE=1``. When on,
each profiled block prints one ``[PROF] {json}`` line to stdout; feed the run log to
``examples/analyze_profile.py`` to get the table that decides whether — and how — to
batch the per-expert output-recon search across experts.

Three questions that batching must answer first, and the signal each line carries:

  Q1  launch-bound or compute-bound?  -> search wall + per-expert GEMM dims, so the
      analyzer can back out achieved TFLOP/s. Low utilisation means the device sits
      idle between many small launches, which is exactly where fusing experts wins;
      high utilisation means the GEMMs already saturate it and batching buys little.
  Q2  per-expert token spread?        -> the token count of every routed expert, which
      drives the ragged-token strategy (bucket equal-length experts vs zero-pad+mask)
      and the ``tok`` dimension of the tile.
  Q3  search-phase peak HBM + free?   -> the budget the ``expert_chunk x ab_chunk`` tile
      has to fit inside.

This module only emits raw numbers; all arithmetic (FLOPs, utilisation, the suggested
tile) lives in the analyzer so it can be unit-tested on CPU without an NPU.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Optional

_LOGGER = logging.getLogger(__name__)

import torch


def enabled() -> bool:
    """True when JOINTFIX_PROFILE is set to a truthy value."""
    return os.environ.get("JOINTFIX_PROFILE", "") not in ("", "0", "false", "False", "no")


def clock() -> float:
    """time.time() when profiling is on, else 0.0 (callers subtract two clocks)."""
    return time.time() if enabled() else 0.0


def fmt(seconds: float) -> str:
    """Human duration: 3725 -> '1h02m', 309 -> '5m09s', 42 -> '42s'."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def heartbeat(msg: str) -> None:
    """
    Live progress line for a human watching a long run: ``[PROF HH:MM:SS] msg``.

    Distinct from the ``[PROF] {json}`` data lines (analyzer fodder): heartbeats carry
    no machine-read payload — they just show the run is alive, which stage it is in, and
    how long each stage took. Crucial for a 505B run whose first useful output otherwise
    appears only after layer 0 fully completes. No-op when profiling is off (zero overhead).
    """
    if not enabled():
        return
    print(f"[PROF {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def eta(done: int, total: int, t_run_start: float) -> None:
    """
    After ``done``/``total`` layers, print percent-done, elapsed, avg/layer and ETA —
    the text-mode equivalent of a progress bar for the layer loop. No-op when off.
    """
    if not enabled() or done <= 0 or total <= 0:
        return
    elapsed = clock() - t_run_start
    avg = elapsed / done
    print(f"[PROF {time.strftime('%H:%M:%S')}] progress {done}/{total} layers "
          f"({100 * done / total:.0f}%) | elapsed {fmt(elapsed)} | "
          f"avg {fmt(avg)}/layer | ETA ~{fmt(avg * max(0, total - done))}", flush=True)


def _mem_backend(dev: torch.device):
    """The torch memory module for a device (torch.npu / torch.cuda), or None for cpu."""
    if dev.type == "npu" and hasattr(torch, "npu"):
        return torch.npu
    if dev.type == "cuda":
        return torch.cuda
    return None


def reset_peak(devices) -> None:
    """
    Zero the peak-allocated counter on every device so the next emit measures the
    peak of just the block that follows. No-op when disabled or on CPU.
    """
    if not enabled():
        return
    for dev in devices:
        m = _mem_backend(dev)
        if m is not None:
            try:
                m.reset_peak_memory_stats(dev)
            except (AttributeError, RuntimeError, TypeError) as error:
                _LOGGER.debug("failed to reset peak memory stats on %s: %s", dev, error)


def _peak_and_free_gb(devices):
    """Return (peak_gb, free_gb) lists, one entry per device (None where unavailable)."""
    peak, free = [], []
    for dev in devices:
        m = _mem_backend(dev)
        if m is None:
            peak.append(None)
            free.append(None)
            continue
        try:
            peak.append(round(m.max_memory_allocated(dev) / 1e9, 3))
        except (AttributeError, RuntimeError, TypeError) as error:
            _LOGGER.debug("failed to read peak memory on %s: %s", dev, error)
            peak.append(None)
        try:
            free_bytes, _total = m.mem_get_info(dev)
            free.append(round(free_bytes / 1e9, 3))
        except (AttributeError, RuntimeError, TypeError) as error:
            _LOGGER.debug("failed to read free memory on %s: %s", dev, error)
            free.append(None)
    return peak, free


def _emit(record: dict) -> None:
    if not enabled():
        return
    # prepend an absolute wall-clock stamp so the log shows *when* each line landed
    # (durations alone can't reveal a stall); analyzer reads by key, extra field is safe.
    print("[PROF] " + json.dumps({"ts": time.strftime("%H:%M:%S"), **record},
                                 ensure_ascii=False), flush=True)


def _n_candidates(config) -> int:
    """Stage-1 grid size + Stage-2 box size, i.e. how many (a,b) each expert evaluates."""
    n_stage1 = len(getattr(config, "stage1_grid", []) or []) or 9
    radius = getattr(config, "stage2_radius", 0.2)
    step = getattr(config, "stage2_step", 0.1)
    side = int(2 * radius / step) + 1 if step > 0 else 1
    return n_stage1 + side * side


def emit_routed_down(layer_idx: int, routed, tensors, get_stats, devices,
                     t_start: float, config) -> None:
    """
    Emit the per-layer routed-expert search profile (Q1/Q2/Q3).

    `routed` is the list of (eid, down_key, up_key, ask) the driver just searched;
    `get_stats(ask)` returns the (cached) finalized stats for an expert; `tensors`
    holds the layer weights. Call this right after the search loop, before the apply
    step, with `t_start` from a `clock()` taken at `reset_peak()` time.
    """
    if not enabled() or not routed:
        return
    toks: List[int] = []
    h_in: Optional[int] = None
    h_out: Optional[int] = None
    for _eid, dk, _uk, ask in routed:
        try:
            xs = get_stats(ask).get("X_sample")
            if xs is not None:
                toks.append(int(xs.shape[0]))
        except (AttributeError, KeyError, RuntimeError, TypeError) as error:
            _LOGGER.debug("failed to read profiling stats for %s: %s", ask, error)
        if h_out is None and dk in tensors:
            h_out, h_in = int(tensors[dk].shape[0]), int(tensors[dk].shape[1])
    peak, free = _peak_and_free_gb(devices)
    _emit({
        "kind": "routed_down",
        "layer": layer_idx,
        "n_dev": len(devices),
        "n_experts": len(routed),
        "h_in": h_in,
        "h_out": h_out,
        "n_candidates": _n_candidates(config),
        "toks": toks,
        "wall_s": round(clock() - t_start, 4),
        "peak_gb": peak,
        "free_gb": free,
    })


def emit_layer(layer_idx: int, n_samples: int, t_forward: float, t_search_quant: float,
               t_reforward: float, t_load: float = 0.0) -> None:
    """
    Emit the per-layer wall-time split, so search-as-a-fraction-of-layer is visible
    (confirms how much the expert search actually costs end-to-end). ``t_load`` is the
    weight-load (often SFS/network IO) time — the suspected bottleneck on big models, and
    the one phase the old timing missed because the load ran before the first clock().
    """
    if not enabled():
        return
    _emit({
        "kind": "layer",
        "layer": layer_idx,
        "n_samples": n_samples,
        "t_load_s": round(t_load, 4),
        "t_forward_s": round(t_forward, 4),
        "t_search_quant_s": round(t_search_quant, 4),
        "t_reforward_s": round(t_reforward, 4),
    })
