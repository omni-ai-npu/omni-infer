# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

"""Per-batch thinking-ban state holder.

Suppresses Pangu's tool-call boundary failure mode under spec-decode:

* While a request is **inside** ``<think>``, bans ``<|tool_call_start|>``
  (forces the model to close ``</think>`` before opening a tool call).
* Once ``</think>`` has been emitted, bans the further re-emission of
  ``</think>`` (prevents fragmenting the output).

Designed as a sibling of :class:`ThinkingBudgetStateHolder` so it can ride
the same ``InputBatch`` lifecycle hooks (``sync_batch``,
``_make_sampling_metadata``) and the same ``Sampler`` /
``RejectionSampler`` apply paths in ``patch_thinking_limit.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from omni_npu.v1.config import ReasoningConfig

logger = init_logger(__name__)

# FSM constants.
PRE_THINK = 0
IN_THINK = 1
POST_THINK = 2


def maybe_create_thinking_ban_state_holder(
    reasoning_config: "ReasoningConfig | None",
    max_num_seqs: int,
    num_spec_tokens: int,
    device: torch.device,
    is_pin_memory: bool,
) -> "ThinkingBanStateHolder | None":
    """Factory: returns a holder iff the ban is opted-in and the token id resolved.

    Returns ``None`` when ``reasoning_config`` is missing, when
    ``ban_tool_call_in_thinking`` is off, or when
    ``reasoning_config.tool_call_start_token_id`` is ``None`` (the
    tokenizer had neither ``<|tool_call_start|>`` nor the ``[unused11]``
    fallback when ``initialize_token_ids`` ran). A ``None`` holder makes
    the ban a silent no-op end-to-end.
    """
    if reasoning_config is None:
        return None
    if not getattr(reasoning_config, "ban_tool_call_in_thinking", False):
        return None
    tid = getattr(reasoning_config, "tool_call_start_token_id", None)
    if tid is None:
        logger.warning(
            "ThinkingBanStateHolder: ban_tool_call_in_thinking=True but the "
            "tokenizer had no <|tool_call_start|>/[unused11] when "
            "initialize_token_ids ran; ban disabled"
        )
        return None
    return ThinkingBanStateHolder(
        reasoning_config=reasoning_config,
        tool_call_start_tid=tid,
        max_num_seqs=max_num_seqs,
        num_spec_tokens=num_spec_tokens,
        device=device,
        is_pin_memory=is_pin_memory,
    )


class ThinkingBanStateHolder:
    """Three-state FSM per request, derived from the last positions of
    ``<think>`` and ``</think>`` in the running history.

    State transitions are last-occurrence based (mirrors the backwards-scan
    semantics in ``qwen3_reasoning_parser.is_reasoning_end``):

    ============================================  ===========
    last_<think>_pos vs last_</think>_pos          state
    ============================================  ===========
    both ``-1``                                    PRE_THINK
    ``last_start > last_end``                      IN_THINK
    ``last_end > last_start``                      POST_THINK
    ============================================  ===========

    Apply rules:

    * ``IN_THINK``   → ban ``<|tool_call_start|>`` (one column)
    * ``POST_THINK`` → ban the last token of ``</think>`` (one column)
    * ``PRE_THINK``  → no-op
    """

    def __init__(
        self,
        *,
        reasoning_config: "ReasoningConfig",
        tool_call_start_tid: int,
        max_num_seqs: int,
        num_spec_tokens: int,
        device: torch.device,
        is_pin_memory: bool,
    ):
        _ = is_pin_memory  # API parity with ThinkingBudgetStateHolder
        _ = max_num_seqs   # not needed for dict-based state but kept in signature
        self.device = device
        self.num_spec_tokens = num_spec_tokens
        self.in_spec_mode = num_spec_tokens > 0
        self.tool_call_start_tid = int(tool_call_start_tid)

        rs = getattr(reasoning_config, "reasoning_start_token_ids", None)
        re_ = getattr(reasoning_config, "reasoning_end_token_ids", None)
        self.think_start_token_ids: list[int] = list(rs) if rs else []
        self.think_end_token_ids: list[int] = list(re_) if re_ else []
        # The fast single-token path (Pangu's case) keys off the last id; the
        # general path uses ``_find_last_sequence_index`` which handles
        # multi-token sentinels for ``update_state``. Per-row classification
        # under MTP-K only honors single-token sentinels — a multi-token
        # ``<think>`` would need suffix matching across draft boundaries,
        # which is out of scope.
        self.is_enabled = bool(
            self.think_start_token_ids and self.think_end_token_ids
        )
        if not self.is_enabled:
            logger.warning(
                "ThinkingBanStateHolder: reasoning_start/end token ids are "
                "empty (reasoning_config.initialize_token_ids was probably "
                "not called); ban disabled"
            )

        # Per-slot state: {req_index -> dict}. Mirrors the keying used by
        # ThinkingBudgetStateHolder so condense/move from BatchUpdate works
        # the same way.
        self._state: dict[int, dict[str, Any]] = {}
        # cu_num_tokens[slot] = starting row index for this slot in a flat
        # logits tensor; reused per ``apply_to_logits`` call to keep the row
        # layout consistent with the budget holder.
        self.cu_num_tokens: dict[int, int] = {}

        logger.info(
            "ThinkingBanStateHolder enabled=%s tool_call_start_tid=%d "
            "think_start=%s think_end=%s num_spec_tokens=%d",
            self.is_enabled,
            self.tool_call_start_tid,
            self.think_start_token_ids,
            self.think_end_token_ids,
            self.num_spec_tokens,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _find_last_sequence_index(
        target_list: list[int], token_ids: list[int]
    ) -> int:
        """Last starting index where ``token_ids`` occurs in ``target_list``.

        Identical to ``ThinkingBudgetStateHolder._find_last_sequence_index``.
        """
        if not token_ids:
            return -1
        target = target_list if isinstance(target_list, list) else list(target_list)
        n = len(token_ids)
        for i in range(len(target) - n, -1, -1):
            if target[i : i + n] == token_ids:
                return i
        return -1

    @staticmethod
    def _derive_state(last_start_pos: int, last_end_pos: int) -> int:
        if last_start_pos < 0 and last_end_pos < 0:
            return PRE_THINK
        if last_start_pos > last_end_pos:
            return IN_THINK
        return POST_THINK

    def _init_state_entry(
        self, prompt_tok_ids: list[int] | None
    ) -> dict[str, Any]:
        prompt = list(prompt_tok_ids) if prompt_tok_ids else []
        last_start = self._find_last_sequence_index(prompt, self.think_start_token_ids)
        last_end = self._find_last_sequence_index(prompt, self.think_end_token_ids)
        return {
            "last_start_pos": last_start,
            "last_end_pos": last_end,
            "prompt_len": len(prompt),
            "prev_output_length": 0,
            "output_tok_ids": [],
            "spec_token_ids": [],
        }

    # ------------------------------------------------------------------ public

    def has_tracked_requests(self) -> bool:
        """Mirrors ``ThinkingBudgetStateHolder.has_tracked_requests``."""
        return bool(self._state)

    def sync_batch(self, batch_update: BatchUpdate | None) -> None:
        """Add/remove/move per-request state. Mirrors the budget holder.

        Unlike the budget holder (which gates per request on
        ``extra_args.thinking_token_budget``), the ban is a *global*
        feature: every new request gets tracked while the holder is
        enabled. The factory is the single switch.
        """
        if not self.is_enabled or not batch_update:
            return

        # removed BEFORE added (matches BatchUpdate's documented order)
        for index in batch_update.removed:
            self._state.pop(index, None)

        for index, _params, prompt_tok_ids, output_tok_ids in batch_update.added:
            entry = self._init_state_entry(prompt_tok_ids)
            # Hold the live reference (BatchUpdate's `output_tok_ids` is the
            # request's running list — same contract as ``BatchUpdate.added``
            # at vllm/v1/sample/logits_processor/interface.py:46-48).
            entry["output_tok_ids"] = output_tok_ids
            self._state[index] = entry

        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                s1 = self._state.get(i1)
                s2 = self._state.get(i2)
                if s1 is not None:
                    self._state[i2] = s1
                else:
                    self._state.pop(i2, None)
                if s2 is not None:
                    self._state[i1] = s2
                else:
                    self._state.pop(i1, None)
            else:
                state = self._state.pop(i1, None)
                if state is not None:
                    self._state[i2] = state

    def update_state(
        self,
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None,
        repeat_indices: torch.Tensor | None = None,
    ) -> None:
        """Re-derive ``last_start_pos`` / ``last_end_pos`` from the running
        committed history (output minus the trailing spec suffix).

        Mirrors ``ThinkingBudgetStateHolder.update_state`` signature exactly,
        including the spec-suffix-stripping rule and the ``repeat_indices``
        layout used by the RejectionSampler path.
        """
        if not self.is_enabled or not self._state:
            return

        spec_lists = spec_token_ids or []
        last_row_for_req: dict[int, int] | None = None
        if repeat_indices is not None:
            last_row_for_req = {}
            rpt = repeat_indices.cpu().tolist()
            for batch_row, req_i in enumerate(rpt):
                last_row_for_req[req_i] = batch_row

        for seq_idx, state in list(self._state.items()):
            if last_row_for_req is not None:
                output_row = last_row_for_req.get(seq_idx)
                if output_row is None or output_row >= len(output_token_ids):
                    continue
                state["output_tok_ids"] = output_token_ids[output_row]
            elif seq_idx >= len(output_token_ids):
                continue
            else:
                state["output_tok_ids"] = output_token_ids[seq_idx]
            if seq_idx < len(spec_lists):
                state["spec_token_ids"] = list(spec_lists[seq_idx])
            else:
                state["spec_token_ids"] = []

            # ``output_tok_ids`` may already include the spec suffix from
            # ``_combine_outputs_with_spec_tokens``; strip it to recover the
            # truly-committed history (mirrors budget holder lines 181-186).
            output = list(state["output_tok_ids"]) if state["output_tok_ids"] else []
            spec_len = len(state["spec_token_ids"])
            if spec_len > 0 and len(output) >= spec_len:
                output = output[: -spec_len]

            # Incrementally extend last_*_pos using ONLY the newly committed
            # tokens. For single-token sentinels (Pangu) this is a simple
            # equality test. For multi-token sentinels we fall back to a
            # bounded suffix rescan against the prompt+output tail (rare).
            prev_len = state["prev_output_length"]
            new_tokens = output[prev_len:]
            for i, tid in enumerate(new_tokens):
                abs_pos = state["prompt_len"] + prev_len + i
                if (
                    len(self.think_start_token_ids) == 1
                    and tid == self.think_start_token_ids[0]
                ):
                    state["last_start_pos"] = abs_pos
                if (
                    len(self.think_end_token_ids) == 1
                    and tid == self.think_end_token_ids[0]
                ):
                    state["last_end_pos"] = abs_pos
            if len(self.think_start_token_ids) > 1 or len(self.think_end_token_ids) > 1:
                # Multi-token: rescan the suffix that could span the boundary
                # between previously-known last_*_pos and the new tokens.
                full = output  # output only; prompt scan already done at init
                tail_start = max(
                    0,
                    prev_len
                    - max(
                        len(self.think_start_token_ids),
                        len(self.think_end_token_ids),
                    )
                    + 1,
                )
                tail = full[tail_start:]
                rel_s = self._find_last_sequence_index(tail, self.think_start_token_ids)
                rel_e = self._find_last_sequence_index(tail, self.think_end_token_ids)
                if rel_s >= 0:
                    state["last_start_pos"] = max(
                        state["last_start_pos"], state["prompt_len"] + tail_start + rel_s
                    )
                if rel_e >= 0:
                    state["last_end_pos"] = max(
                        state["last_end_pos"], state["prompt_len"] + tail_start + rel_e
                    )

            state["prev_output_length"] = len(output)

    def apply_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids: list[list[int]] | None,
    ) -> torch.Tensor:
        """Mask banned token columns at ``-inf`` per row.

        Row layout mirrors ``ThinkingBudgetStateHolder._apply_forcing_to_logits``
        (lines 481-498) so the cursor arithmetic stays identical:

        * non-spec mode → 1 row per request
        * spec mode, ``predict_bonus_token=False`` → ``num_draft_tokens`` rows
          per request (the K target rows, ordered position 0..K-1)
        * spec mode, ``predict_bonus_token=True`` → 1 row per request (the
          bonus row, fed the state after consuming all K drafts)
        """
        if not self.is_enabled or not self._state:
            return logits

        spec_lists = spec_token_ids or []
        self.cu_num_tokens.clear()
        cumulative = 0
        n_layout = len(spec_lists)
        if self._state:
            n_layout = max(n_layout, max(self._state.keys()) + 1)

        for index in range(n_layout):
            self.cu_num_tokens[index] = cumulative
            spec_tokens = (
                spec_lists[index] if index < len(spec_lists) else []
            )
            if self.in_spec_mode:
                cumulative += len(spec_tokens) if not predict_bonus_token else 1
            else:
                cumulative += 1

        # Single-token shortcuts (Pangu's case)
        think_start_single = (
            self.think_start_token_ids[0]
            if len(self.think_start_token_ids) == 1
            else None
        )
        think_end_single = (
            self.think_end_token_ids[0]
            if len(self.think_end_token_ids) == 1
            else None
        )
        think_end_last = (
            self.think_end_token_ids[-1] if self.think_end_token_ids else None
        )

        def _advance(state: int, tid: int) -> int:
            # Per-row classification helper. Only honors single-token sentinels;
            # multi-token sentinels would need suffix matching across drafts
            # (out of scope — Pangu uses single tokens).
            if think_start_single is not None and tid == think_start_single:
                return IN_THINK
            if think_end_single is not None and tid == think_end_single:
                return POST_THINK
            return state

        for seq_idx in sorted(self._state.keys()):
            if seq_idx not in self.cu_num_tokens:
                continue
            entry = self._state[seq_idx]
            base_state = self._derive_state(
                entry["last_start_pos"], entry["last_end_pos"]
            )
            spec_tokens = spec_lists[seq_idx] if seq_idx < len(spec_lists) else []
            start_row = self.cu_num_tokens[seq_idx]

            if predict_bonus_token:
                # Bonus row sees ALL K drafts speculatively.
                final = base_state
                for d in spec_tokens:
                    final = _advance(final, d)
                rows: list[tuple[int, int]] = [(start_row, final)]
            elif self.in_spec_mode:
                # K target rows: row r samples after history + drafts[0..r-1].
                cur = base_state
                rows = []
                for r in range(len(spec_tokens)):
                    rows.append((start_row + r, cur))
                    cur = _advance(cur, spec_tokens[r])
            else:
                rows = [(start_row, base_state)]

            for row, st in rows:
                if row >= logits.shape[0]:
                    break
                if st == IN_THINK:
                    logits[row, self.tool_call_start_tid] = float("-inf")
                elif st == POST_THINK and think_end_last is not None:
                    logits[row, think_end_last] = float("-inf")

        return logits
