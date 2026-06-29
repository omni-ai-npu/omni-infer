# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Deterministic input-batch request-order swap for KV-cache testing.

Activated by OMNI_MOCK_SCHEDULE=2.  At a fixed point in the decode
sequence, swaps the positions of two requests in ``InputBatch`` before
attention metadata is built.  The swap is deterministic (seeded by the
sorted request IDs), so baseline and omnicache runs produce identical
post-swap batch layouts.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

ENV_VAR = "OMNI_MOCK_SCHEDULE"

_orig_prepare_inputs = None
_swap_done: dict[int, bool] = {}  # pid -> already swapped
_step_counter: dict[int, int] = {}  # pid -> decode step count

_SWAP_AT_STEP = 5  # do the swap on this decode step (1-indexed)


def _is_active() -> bool:
    return os.environ.get(ENV_VAR, "0") == "2"


def _swap_input_batch_rows(input_batch, i: int, j: int) -> None:
    """Swap per-request data for rows *i* and *j*, mirroring condense() pattern."""
    if i == j:
        return

    # ---- request-id bookkeeping -------------------------------------------
    req_ids = input_batch._req_ids
    ri, rj = req_ids[i], req_ids[j]
    req_ids[i], req_ids[j] = rj, ri
    input_batch.req_id_to_index[ri] = j
    input_batch.req_id_to_index[rj] = i

    # ---- per-row scalar arrays (mirrors condense lines 698-729) ------------
    def _swap1(arr, a, b):
        if arr is not None and hasattr(arr, "__getitem__"):
            copy_a = arr[a].copy() if hasattr(arr[a], "copy") else arr[a]
            arr[a], arr[b] = arr[b], copy_a

    _swap1(input_batch.num_tokens_no_spec, i, j)
    _swap1(input_batch.num_prompt_tokens, i, j)
    _swap1(input_batch.num_computed_tokens_cpu, i, j)
    _swap1(input_batch.temperature_cpu, i, j)
    _swap1(input_batch.top_p_cpu, i, j)
    _swap1(input_batch.top_k_cpu, i, j)
    _swap1(input_batch.frequency_penalties_cpu, i, j)
    _swap1(input_batch.presence_penalties_cpu, i, j)
    _swap1(input_batch.repetition_penalties_cpu, i, j)
    _swap1(input_batch.request_lora_mapping, i, j)

    # ---- token_ids: copy rows (mirrors condense lines 685-689) -----------
    bt = input_batch.block_table
    if hasattr(input_batch, "token_ids_cpu"):
        row_i = input_batch.token_ids_cpu[i].copy()
        input_batch.token_ids_cpu[i] = input_batch.token_ids_cpu[j].copy()
        input_batch.token_ids_cpu[j] = row_i

    if hasattr(input_batch, "is_token_ids"):
        row_i = input_batch.is_token_ids[i].copy()
        input_batch.is_token_ids[i] = input_batch.is_token_ids[j].copy()
        input_batch.is_token_ids[j] = row_i

    # output / spec token tracking
    if hasattr(input_batch, "req_output_token_ids"):
        input_batch.req_output_token_ids[i], input_batch.req_output_token_ids[j] = (
            input_batch.req_output_token_ids[j],
            input_batch.req_output_token_ids[i],
        )

    if hasattr(input_batch, "spec_token_ids"):
        input_batch.spec_token_ids[i], input_batch.spec_token_ids[j] = (
            input_batch.spec_token_ids[j],
            input_batch.spec_token_ids[i],
        )

    # req_prompt_embeds — like condense line 691-693
    pe = getattr(input_batch, "req_prompt_embeds", None)
    if pe is not None:
        has_i, has_j = i in pe, j in pe
        vi = pe.pop(i, None) if has_i else None
        vj = pe.pop(j, None) if has_j else None
        if has_j:
            pe[i] = vj
        if has_i:
            pe[j] = vi

    # block_table: use proper swap_row() which handles num_blocks_per_row
    if hasattr(bt, "swap_row"):
        bt.swap_row(i, j)
    elif hasattr(bt, "block_tables"):
        for tbl in bt.block_tables:
            if hasattr(tbl, "swap_row"):
                tbl.swap_row(i, j)

    _swap1(getattr(input_batch, "num_spec_tokens", None), i, j)


def _patched_prepare_inputs(self, scheduler_output, num_tokens_after_padding):
    """Wrap ``_prepare_inputs`` to inject a deterministic row swap."""
    global _swap_done

    if _is_active():
        pid = os.getpid()
        if not _swap_done.get(pid):
            # Increment decode step counter (1-indexed)
            step = _step_counter.get(pid, 0) + 1
            _step_counter[pid] = step

            if step == _SWAP_AT_STEP:
                num_reqs = getattr(self.input_batch, "num_reqs", 0)
                if num_reqs >= 2:
                    from vllm.logger import init_logger

                    _log = init_logger("vllm.v1.omni")
                    _swap_done[pid] = True

                    req_ids = getattr(self.input_batch, "_req_ids", [])[:num_reqs]

                    def _short_id(rid: str) -> str:
                        parts = rid.split("-")
                        if rid.startswith("chatcmpl-"):
                            return "-".join(parts[1:3])
                        return rid

                    short_ids = tuple(sorted(_short_id(r) for r in req_ids))
                    import hashlib

                    seed = int(hashlib.md5(str(short_ids).encode()).hexdigest()[:8], 16)
                    rng = __import__("random").Random(seed)
                    i = rng.randint(0, num_reqs - 1)
                    j = rng.randint(0, num_reqs - 1)
                    while j == i:
                        j = rng.randint(0, num_reqs - 1)

                    _log.warning(
                        "[MOCK-SWAP] step=%d: swapping req[%d]=%s <-> req[%d]=%s seed=%d num_reqs=%d",
                        step,
                        i,
                        req_ids[i],
                        j,
                        req_ids[j],
                        seed,
                        num_reqs,
                    )

                    _swap_input_batch_rows(self.input_batch, i, j)

    return _orig_prepare_inputs(self, scheduler_output, num_tokens_after_padding)


def install() -> None:
    """Monkey-patch NPUModelRunner._prepare_inputs."""
    global _orig_prepare_inputs
    if not _is_active():
        return
    if _orig_prepare_inputs is not None:
        return

    if not _try_patch():
        import threading as _th
        import time as _ti

        def _poll():
            while _orig_prepare_inputs is None:
                _ti.sleep(0.1)
                if _try_patch():
                    return

        _th.Thread(target=_poll, daemon=True).start()


def _try_patch() -> bool:
    global _orig_prepare_inputs
    import sys as _sys

    mod = _sys.modules.get("omni_npu.worker.npu_model_runner")
    if mod is None:
        try:
            import importlib

            mod = importlib.import_module("omni_npu.worker.npu_model_runner")
        except Exception:
            return False
    _orig_prepare_inputs = mod.NPUModelRunner._prepare_inputs
    mod.NPUModelRunner._prepare_inputs = _patched_prepare_inputs
    return True
