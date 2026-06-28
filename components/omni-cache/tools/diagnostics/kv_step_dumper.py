# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Per-step decode KV dumper, packaged as an omni-cache plugin.

After every decode step on this worker, save one `.pt` per
(step, tp_rank, dp_rank, request_id, kv_cache_group, layer_name)
containing the HBM-resident KV slice the request occupies this step.

The HBM slot the dumper reads from is chosen to match exactly what
`omni_cache.cache.transfer_engine.decode.prepare_h2d_copy_args_hbm_buffer`
writes to:

  - mamba / MoMe groups : slot = idx_req + 1
  - sliding-window      : slots = req_offset * idx_req + 1 + (0..N-1)
  - DSA / regular attn  : slot = host pool block_id (1:1)

`idx_req` is the omni-cache batch lane index assigned to the request,
read from `runner.omni_cache.req_id_to_idx`. On baseline (no omni-cache
present) every group falls through to the block_id branch, so the
scheduler's block_table directly indexes the HBM slab.

Single env var to enable:

    OMNI_KV_STEP_DUMP=1
    OMNI_KV_STEP_DUMP_DIR     output root (default /tmp/kv_dumps_step)
    OMNI_KV_STEP_DUMP_BRANCH  subdir label; auto-picks
                              "omnicache" / "baseline" from
                              ENABLE_OMNI_CACHE
    OMNI_KV_STEP_DUMP_MAX     stop after N decode steps (0 = ∞)

`install()` is wired via the `vllm.general_plugins` entry point
`omni_cache_kv_step_dumper`.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV_DIR = "OMNI_KV_STEP_DUMP_DIR"
_ENV_BRANCH = "OMNI_KV_STEP_DUMP_BRANCH"
_ENV_MAX = "OMNI_KV_STEP_DUMP_MAX"
_ENV_TARGET_REQ = "OMNI_KV_DUMP_TARGET_REQ"
_ENV_PRE_STEP_MODE = "OMNI_KV_DUMP_PRE_STEP_MODE"  # "full" (default) or "sliced"

_install_lock = threading.Lock()
_installed = False


def _enabled() -> bool:
    from tools.diagnostics.dump_controller import gear_is_step
    return gear_is_step()


def _target_request_id() -> Optional[str]:
    v = (os.environ.get(_ENV_TARGET_REQ) or "").strip()
    return v if v else None


def _root_dir() -> str:
    return os.environ.get(_ENV_DIR) or "/tmp/kv_dumps_step"


def _branch_name() -> str:
    v = (os.environ.get(_ENV_BRANCH) or "").strip()
    if v:
        return v
    return ("omnicache"
            if os.environ.get("ENABLE_OMNI_CACHE", "0") == "1"
            else "baseline")


def _max_steps() -> int:
    from tools.diagnostics.dump_controller import max_steps
    return max_steps()


def install() -> None:
    """Install the NPUModelRunner.execute_model monkey patch.

    This operation is idempotent and is a no-op when OMNI_KV_STEP_DUMP
    is not set.
    """
    global _installed
    if not _enabled():
        return
    with _install_lock:
        if _installed:
            return
        try:
            from omni_npu.worker.npu_model_runner import NPUModelRunner
        except Exception as exc:
            logger.info(
                "[kv_step_dumper] NPUModelRunner not importable: %s", exc)
            return
        original = NPUModelRunner.execute_model

        def wrapped(self, scheduler_output, intermediate_tensors=None):
            # Pre-step dump: capture KV *before* the first decode forward
            # for a target request id.  Controlled by OMNI_KV_DUMP_TARGET_REQ.
            if getattr(self, "_kv_step_counter", 0) == 0:
                try:
                    _dump_pre_step(self, scheduler_output)
                except Exception as exc:
                    logger.exception(
                        "[kv_step_dumper] pre-step dump failed: %s", exc,
                    )
            result = original(self, scheduler_output, intermediate_tensors)
            try:
                _dump(self, scheduler_output)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[kv_step_dumper] dump failed on step=%s: %s",
                    getattr(self, "_kv_step_counter", "?"), exc,
                )
            return result

        wrapped.__wrapped__ = original
        NPUModelRunner.execute_model = wrapped
        _installed = True
        logger.warning(
            "[kv_step_dumper] installed: branch=%s dir=%s max_steps=%s",
            _branch_name(), _root_dir(), _max_steps() or "inf",
        )


# ---- runtime detection ---------------------------------------------------


def _is_consumer(runner) -> bool:
    ktc = getattr(runner.vllm_config, "kv_transfer_config", None)
    if ktc is None:
        return True
    return getattr(ktc, "kv_role", "") == "kv_consumer"


def _omnicache_active(runner) -> bool:
    return (
        getattr(runner, "omni_cache", None) is not None
        and getattr(runner.omni_cache, "hbm_buffer_pool", None) is not None
    )


def _hbm_req_offset(runner, gidx: int) -> Optional[int]:
    oc = getattr(runner, "omni_cache", None)
    if oc is None:
        return None
    pool = getattr(oc, "hbm_buffer_block_table_pool", None)
    if not pool or gidx >= len(pool):
        return None
    entry = pool[gidx]
    if isinstance(entry, dict):
        v = entry.get("req_offset")
        if isinstance(v, int) and v > 0:
            return v
    return None


def _omni_idx_req(runner, request_id: str) -> Optional[int]:
    """Resolve omni-cache's HBM batch-lane index for `request_id`.

    `prepare_h2d_copy_args_hbm_buffer` writes mamba / SWA blocks at slots
    derived from this index (and `record_batch_idx_to_req` is what assigns
    it). We must read from the same slot or we'll see stale / wrong data.
    """
    oc = getattr(runner, "omni_cache", None)
    if oc is None:
        return None
    table = getattr(oc, "req_id_to_idx", None)
    if isinstance(table, dict):
        v = table.get(request_id)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    rmap = getattr(oc, "record_batch_idx_to_req", None)
    if isinstance(rmap, dict):
        for idx, rid in rmap.items():
            if rid == request_id:
                try:
                    return int(idx)
                except (TypeError, ValueError):
                    pass
    return None


def _spec_path(spec) -> str:
    """Same dispatch logic `prepare_h2d_copy_args_hbm_buffer` uses to pick
    the HBM destination slot. Returns "mamba", "swa", or "block_id"."""
    try:
        from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec
    except Exception:
        MambaSpec = None
        SlidingWindowSpec = None
    try:
        from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid.patch_kv_cache_interface import (  # noqa: E501
            MomeSpec, ShareKVSlidingWindowSpec,
        )
    except Exception:
        MomeSpec = None
        ShareKVSlidingWindowSpec = None

    if MambaSpec is not None and isinstance(spec, MambaSpec):
        return "mamba"
    if MomeSpec is not None and isinstance(spec, MomeSpec):
        return "mamba"
    if (ShareKVSlidingWindowSpec is not None
            and isinstance(spec, ShareKVSlidingWindowSpec)):
        return "swa"
    if SlidingWindowSpec is not None and isinstance(spec, SlidingWindowSpec):
        return "swa"
    return "block_id"


def _hbm_slots(spec_path_value: str,
               host_blocks: List[int],
               idx_req: Optional[int],
               req_offset: Optional[int]) -> List[int]:
    """Translate logical block ids → HBM-buffer slots."""
    if spec_path_value == "mamba":
        if idx_req is None:
            return []
        return [idx_req + 1]
    if spec_path_value == "swa":
        if idx_req is None or not req_offset:
            return list(host_blocks)
        n = min(len(host_blocks), req_offset)
        base = req_offset * idx_req
        return [base + i + 1 for i in range(n)]
    return list(host_blocks)


def _group_block_table_np(input_batch, gidx: int) -> Optional[np.ndarray]:
    mgt = getattr(input_batch, "block_table", None)
    if mgt is None:
        return None
    tables = getattr(mgt, "block_tables", None)
    if not isinstance(tables, (list, tuple)) or gidx >= len(tables):
        return None
    entry = tables[gidx]
    inner = getattr(entry, "block_table", None)
    arr = getattr(inner, "np", None) if inner is not None else None
    if isinstance(arr, np.ndarray):
        return arr
    gpu = getattr(entry, "gpu", None)
    if isinstance(gpu, torch.Tensor):
        return gpu.detach().cpu().numpy()
    return None


def _row_blocks(
    per_group_np: np.ndarray,
    row: int,
    max_blocks: Optional[int] = None,
) -> List[int]:
    """Deduped non-zero block ids occupied by `row`.

    `max_blocks` caps the number of leading row entries inspected.
    The scheduler keeps a fixed-width block-table per row; when a
    row is reused for a shorter request, the trailing slots may
    still carry stale block-ids from the previous occupant. Pass
    ``ceil(num_tokens_no_spec[row] / block_size)`` so those stale
    ids don't widen the dump's first dim past the real block count.
    """
    if row >= per_group_np.shape[0]:
        return []
    row_arr = per_group_np[row].tolist()
    if max_blocks is not None:
        row_arr = row_arr[: max(0, int(max_blocks))]
    seen: Dict[int, None] = {}
    for v in row_arr:
        iv = int(v)
        if iv > 0 and iv not in seen:
            seen[iv] = None
    return list(seen.keys())


def _rebuild_dense_if_strided(t: torch.Tensor) -> torch.Tensor:
    """For DSA layers under ENABLE_HOST_MAPPING=1 the host pool stores kv
    densely (block_size * head_dim contiguous per block, with a per-block
    tail of (actual_head_dim - head_dim) * block_size padding bytes the
    decode kernel never reads). The vLLM-side view set up by
    ``_host_cache_view`` reads at stride=actual_head_dim per token, which
    is wrong for this dense layout: it treats the per-block pad as if it
    were per-token gap and shifts every token after the first by
    (actual_head_dim - head_dim). Detect that (per-token stride larger
    than feature dim) and rebuild a dense view from underlying storage.
    Tensors that are already dense pass through unchanged."""
    if t.ndim != 3:
        return t
    num_blocks, block_size, head_dim = t.shape
    actual_head_dim = t.stride(1)
    if actual_head_dim == head_dim:
        return t
    flat = t.as_strided(
        size=(num_blocks, block_size * actual_head_dim),
        stride=(block_size * actual_head_dim, 1),
        storage_offset=t.storage_offset(),
    )
    return flat[:, : block_size * head_dim].reshape(
        num_blocks, block_size, head_dim,
    )


def _layer_components(attn_mod) -> Dict[str, torch.Tensor]:
    raw = getattr(attn_mod, "kv_cache", None)
    if raw is None:
        return {}
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    if isinstance(raw, torch.Tensor):
        return {"c0": _rebuild_dense_if_strided(raw)}
    if isinstance(raw, (list, tuple)):
        return {
            f"c{i}": _rebuild_dense_if_strided(t)
            for i, t in enumerate(raw)
            if isinstance(t, torch.Tensor) and t.ndim > 0
        }
    return {}


def _slice_by_blocks(
    comps: Dict[str, torch.Tensor], block_ids: List[int],
) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    for name, t in comps.items():
        if t.shape[0] == 0:
            continue
        valid = [b for b in block_ids if 0 <= b < t.shape[0]]
        if not valid:
            continue
        idx = torch.as_tensor(valid, dtype=torch.long, device=t.device)
        sliced = t.index_select(0, idx).detach().to("cpu").contiguous()
        out[name] = sliced
        out[f"{name}_blocks"] = list(map(int, valid))
    return out or None


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)


def _dump_pre_step(runner, scheduler_output) -> None:
    """Dump KV for a target request BEFORE the first decode step.

    Controlled by:
      OMNI_KV_DUMP_TARGET_REQ  — request-id substring to match (e.g. "humaneval-0010")
      OMNI_KV_STEP_DUMP_DIR    — output root (default /tmp/kv_dumps_step)
      OMNI_KV_STEP_DUMP_BRANCH — branch label

    Dumps all KV groups for the matched request: MoME / SWA / Indexer (HBM)
    and DSA (host).  Includes per-group block IDs so the dump can serve as
    a reference for attention-operator unit tests.
    """
    target_req = _target_request_id()
    if not target_req:
        return

    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return
    num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
    if num_reqs == 0:
        return

    kv_cfg = getattr(runner, "kv_cache_config", None)
    if kv_cfg is None or not kv_cfg.kv_cache_groups:
        return
    fwd_ctx = getattr(
        runner.vllm_config.compilation_config, "static_forward_context", None,
    )
    if not isinstance(fwd_ctx, dict) or not fwd_ctx:
        return

    req_ids = list(getattr(input_batch, "req_ids", []))[:num_reqs]
    target_row = None
    target_rid = None
    for row, rid in enumerate(req_ids):
        if target_req in str(rid):
            target_row = row
            target_rid = str(rid)
            break
    if target_row is None:
        return

    tp_rank = int(getattr(runner, "rank", 0) or 0)
    pconf = getattr(runner, "parallel_config", None)
    dp_rank = int(getattr(pconf, "data_parallel_rank", 0) or 0) if pconf else 0
    branch = _branch_name()
    out_dir = os.path.join(_root_dir(), branch, "pre_step", f"tp{tp_rank}_dp{dp_rank}")
    os.makedirs(out_dir, exist_ok=True)

    omnicache = _omnicache_active(runner)

    for gidx, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        per_group_np = _group_block_table_np(input_batch, gidx)
        if per_group_np is None:
            continue
        spec_kind = _spec_path(group.kv_cache_spec) if omnicache else "block_id"
        req_offset = _hbm_req_offset(runner, gidx) if omnicache else None
        block_size = getattr(group.kv_cache_spec, "block_size", None)
        num_tokens_arr = getattr(input_batch, "num_tokens_no_spec", None)

        max_blocks_for_row: Optional[int] = None
        if num_tokens_arr is not None and block_size:
            n_tok = int(num_tokens_arr[target_row])
            if n_tok > 0:
                max_blocks_for_row = (n_tok + int(block_size) - 1) // int(block_size)
        host_blocks = _row_blocks(per_group_np, target_row,
                                  max_blocks=max_blocks_for_row)
        if not host_blocks:
            continue

        idx_req = _omni_idx_req(runner, target_rid) if omnicache else None
        hbm_blocks = _hbm_slots(spec_kind, host_blocks, idx_req, req_offset)

        for layer_name in group.layer_names:
            attn_mod = fwd_ctx.get(layer_name)
            if attn_mod is None:
                continue
            comps = _layer_components(attn_mod)
            if not comps:
                continue

            pre_step_mode = (os.environ.get(_ENV_PRE_STEP_MODE) or "full").strip()

            if pre_step_mode == "sliced":
                # Only the blocks belonging to this request.
                kv_spec = _slice_by_blocks(comps, hbm_blocks) if hbm_blocks else None
                kv_block = _slice_by_blocks(comps, host_blocks)
                kv_payload = kv_spec or kv_block or {}
            else:
                # Full PA-format KV tensor (all blocks) — for operator-level
                # verification where the caller needs the complete cache.
                kv_payload = {}
                for cname, ct in comps.items():
                    kv_payload[cname] = ct.detach().to("cpu").contiguous()

            fname = (
                f"req-{_safe(target_rid)}"
                f"_g{gidx}_{_safe(layer_name)}.pt"
            )
            out_path = os.path.join(out_dir, fname)
            torch.save(
                {
                    "step": 0,
                    "tp_rank": tp_rank,
                    "dp_rank": dp_rank,
                    "branch": branch,
                    "request_id": target_rid,
                    "request_row": target_row,
                    "kv_cache_group": gidx,
                    "layer_name": layer_name,
                    "spec_path": spec_kind,
                    "idx_req": idx_req,
                    "host_block_ids": list(map(int, host_blocks)),
                    "hbm_block_ids": list(map(int, hbm_blocks)) if hbm_blocks else [],
                    "omni_cache": omnicache,
                    "kv": kv_payload,
                },
                out_path + ".tmp",
            )
            os.replace(out_path + ".tmp", out_path)

    logger.info(
        "[kv_step_dumper] pre-step dump: req=%s row=%d branch=%s dir=%s",
        target_rid, target_row, branch, out_dir,
    )


def _dump(runner, scheduler_output) -> None:
    if not _is_consumer(runner):
        return
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return
    num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
    if num_reqs == 0:
        return

    step = int(getattr(runner, "_kv_step_counter", 0)) + 1
    runner._kv_step_counter = step
    limit = _max_steps()
    if limit and step > limit:
        return

    kv_cfg = getattr(runner, "kv_cache_config", None)
    if kv_cfg is None or not kv_cfg.kv_cache_groups:
        return
    fwd_ctx = getattr(
        runner.vllm_config.compilation_config, "static_forward_context", None,
    )
    if not isinstance(fwd_ctx, dict) or not fwd_ctx:
        if num_reqs > 0:
            logger.warning(
                "[kv_step_dumper] static_forward_context is empty; "
                "cannot resolve layer KV tensors. step=%d reqs=%d",
                step, num_reqs,
            )
        return

    req_ids = list(getattr(input_batch, "req_ids", []))[:num_reqs]
    while len(req_ids) < num_reqs:
        req_ids.append(f"row{len(req_ids)}")

    # Filter by target request id prefix (OMNI_KV_DUMP_TARGET_REQ)
    target_prefix = _target_request_id()
    target_rows = None
    if target_prefix:
        target_rows = {
            row for row, rid in enumerate(req_ids)
            if target_prefix in str(rid)
        }

    tp_rank = int(getattr(runner, "rank", 0) or 0)
    pconf = getattr(runner, "parallel_config", None)
    dp_rank = int(getattr(pconf, "data_parallel_rank", 0) or 0) if pconf else 0
    branch = _branch_name()
    step_dir = os.path.join(
        _root_dir(), branch, f"step{step:04d}", f"tp{tp_rank}_dp{dp_rank}",
    )
    os.makedirs(step_dir, exist_ok=True)

    omnicache = _omnicache_active(runner)
    written = 0

    for gidx, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        per_group_np = _group_block_table_np(input_batch, gidx)
        if per_group_np is None:
            continue
        spec_kind = _spec_path(group.kv_cache_spec) if omnicache else "block_id"
        req_offset = _hbm_req_offset(runner, gidx) if omnicache else None
        block_size = getattr(group.kv_cache_spec, "block_size", None)
        num_tokens_arr = getattr(input_batch, "num_tokens_no_spec", None)

        for layer_name in group.layer_names:
            attn_mod = fwd_ctx.get(layer_name)
            if attn_mod is None:
                continue
            comps = _layer_components(attn_mod)
            if not comps:
                continue

            for row in range(num_reqs):
                if target_rows is not None and row not in target_rows:
                    continue
                # Cap by the actual number of blocks this request
                # occupies (ceil(num_tokens / block_size)) so stale
                # block-table entries left by a previous, longer
                # request on the same row are ignored.
                max_blocks_for_row: Optional[int] = None
                if num_tokens_arr is not None and block_size:
                    n_tok = int(num_tokens_arr[row])
                    if n_tok > 0:
                        bs = int(block_size)
                        max_blocks_for_row = (n_tok + bs - 1) // bs
                host_blocks = _row_blocks(
                    per_group_np, row, max_blocks=max_blocks_for_row,
                )
                if not host_blocks:
                    continue
                rid = req_ids[row]
                idx_req = _omni_idx_req(runner, str(rid)) if omnicache else None
                hbm_blocks = _hbm_slots(
                    spec_kind, host_blocks, idx_req, req_offset,
                )
                if not hbm_blocks:
                    continue
                payload = _slice_by_blocks(comps, hbm_blocks)
                if not payload:
                    continue
                # Also dump using host block_ids directly (mode-agnostic).
                # For baseline / DSA this equals payload; for MoMe / SWA in
                # omnicache it reads the block_id slot of the underlying
                # tensor instead of the spec-aware slot — useful as a
                # control signal when comparing to baseline byte-for-byte.
                payload_block = _slice_by_blocks(comps, host_blocks)

                fname = (
                    f"req-{_safe(str(rid))}"
                    f"_g{gidx}_{_safe(layer_name)}.pt"
                )
                out_path = os.path.join(step_dir, fname)
                torch.save(
                    {
                        "step": step,
                        "tp_rank": tp_rank,
                        "dp_rank": dp_rank,
                        "branch": branch,
                        "request_id": str(rid),
                        "request_row": row,
                        "kv_cache_group": gidx,
                        "layer_name": layer_name,
                        "spec_path": spec_kind,
                        "idx_req": idx_req,
                        "host_block_ids": list(map(int, host_blocks)),
                        "hbm_block_ids": list(map(int, hbm_blocks)),
                        "omni_cache": omnicache,
                        "kv": payload,
                        "kv_by_block_id": payload_block,
                    },
                    out_path + ".tmp",
                )
                os.replace(out_path + ".tmp", out_path)
                written += 1

    logger.info(
        "[kv_step_dumper] step=%d branch=%s running=%d wrote=%d dir=%s",
        step, branch, num_reqs, written, step_dir,
    )
