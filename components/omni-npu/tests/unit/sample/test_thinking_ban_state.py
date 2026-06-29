# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for ``thinking_ban_state`` (ThinkingBanStateHolder)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import torch

from omni_npu.v1.config import ReasoningConfig
from omni_npu.v1.sample.thinking_ban_state import (
    IN_THINK,
    POST_THINK,
    PRE_THINK,
    ThinkingBanStateHolder,
    maybe_create_thinking_ban_state_holder,
)
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality


# Sentinel token ids used throughout the tests. Pangu's real ids are
# <think>=148905, </think>=148906, <|tool_call_start|>=148903 but we keep
# the test set small for legibility.
THINK_START = 100
THINK_END = 200
TOOL_CALL_START = 300
VOCAB_SIZE = 512


@dataclass
class ParamsWithArgs:
    extra_args: dict[str, Any] | None = None


def _make_reasoning_cfg(
    think_start: list[int] | None = None,
    think_end: list[int] | None = None,
    ban: bool = True,
    tool_call_start_tid: int | None = TOOL_CALL_START,
) -> ReasoningConfig:
    rc = ReasoningConfig(ban_tool_call_in_thinking=ban)
    rc._reasoning_start_token_ids = [THINK_START] if think_start is None else think_start
    rc._reasoning_end_token_ids = [THINK_END] if think_end is None else think_end
    rc._tool_call_start_token_id = tool_call_start_tid
    rc._enabled = True
    return rc


def _make_holder(
    *,
    think_start: list[int] | None = None,
    think_end: list[int] | None = None,
    tool_call_start_tid: int = TOOL_CALL_START,
    num_spec_tokens: int = 0,
    device: torch.device | None = None,
) -> ThinkingBanStateHolder:
    return ThinkingBanStateHolder(
        reasoning_config=_make_reasoning_cfg(think_start, think_end),
        tool_call_start_tid=tool_call_start_tid,
        max_num_seqs=4,
        num_spec_tokens=num_spec_tokens,
        device=device or torch.device("cpu"),
        is_pin_memory=False,
    )


def _batch_update(
    *,
    added: list[tuple[int, Any, list[int] | None, list[int]]] | None = None,
    removed: list[int] | None = None,
    moved: list[tuple[int, int, MoveDirectionality]] | None = None,
    batch_size: int | None = None,
) -> BatchUpdate:
    added = added or []
    removed = removed or []
    moved = moved or []
    if batch_size is None:
        batch_size = max(
            (a[0] for a in added),
            default=0,
        ) + 1
    return BatchUpdate(
        batch_size=batch_size, added=added, removed=removed, moved=moved
    )


# ---------------------------------------------------------------- factory tests


class TestMaybeCreateHolder(unittest.TestCase):
    """Factory gating. Tokenizer resolution happens earlier, in
    ``ReasoningConfig.initialize_token_ids`` — covered separately in
    ``tests/unit/config/test_reasoning.py``."""

    def test_none_when_config_is_none(self) -> None:
        self.assertIsNone(
            maybe_create_thinking_ban_state_holder(
                None, 2, 0, torch.device("cpu"), False
            )
        )

    def test_none_when_flag_is_off(self) -> None:
        rc = _make_reasoning_cfg(ban=False)
        self.assertIsNone(
            maybe_create_thinking_ban_state_holder(
                rc, 2, 0, torch.device("cpu"), False
            )
        )

    def test_none_when_tool_call_start_tid_unresolved(self) -> None:
        # Flag on but tokenizer couldn't find <|tool_call_start|> /
        # [unused11] (so initialize_token_ids left the field at None).
        rc = _make_reasoning_cfg(ban=True, tool_call_start_tid=None)
        self.assertIsNone(
            maybe_create_thinking_ban_state_holder(
                rc, 2, 0, torch.device("cpu"), False
            )
        )

    def test_returns_holder_when_resolved(self) -> None:
        rc = _make_reasoning_cfg(ban=True, tool_call_start_tid=42)
        h = maybe_create_thinking_ban_state_holder(
            rc, 2, 1, torch.device("cpu"), False
        )
        self.assertIsInstance(h, ThinkingBanStateHolder)
        self.assertEqual(h.tool_call_start_tid, 42)
        self.assertTrue(h.in_spec_mode)


# -------------------------------------------------------- pure helpers / FSM


class TestFindLastSequenceIndex(unittest.TestCase):
    def test_single_token(self) -> None:
        self.assertEqual(
            ThinkingBanStateHolder._find_last_sequence_index(
                [1, 2, 100, 3, 100, 4], [100]
            ),
            4,
        )

    def test_multi_token(self) -> None:
        self.assertEqual(
            ThinkingBanStateHolder._find_last_sequence_index(
                [1, 2, 3, 1, 2, 3, 4], [1, 2]
            ),
            3,
        )

    def test_not_found(self) -> None:
        self.assertEqual(
            ThinkingBanStateHolder._find_last_sequence_index([1, 2, 3], [9]),
            -1,
        )

    def test_empty_pattern(self) -> None:
        self.assertEqual(
            ThinkingBanStateHolder._find_last_sequence_index([1, 2], []),
            -1,
        )


class TestDeriveState(unittest.TestCase):
    def test_pre_when_both_neg(self) -> None:
        self.assertEqual(ThinkingBanStateHolder._derive_state(-1, -1), PRE_THINK)

    def test_in_when_start_after_end(self) -> None:
        self.assertEqual(ThinkingBanStateHolder._derive_state(5, 2), IN_THINK)
        self.assertEqual(ThinkingBanStateHolder._derive_state(0, -1), IN_THINK)

    def test_post_when_end_after_start(self) -> None:
        self.assertEqual(ThinkingBanStateHolder._derive_state(2, 5), POST_THINK)
        self.assertEqual(ThinkingBanStateHolder._derive_state(-1, 0), POST_THINK)
        # Equal positions (the same start==end shouldn't happen in practice
        # since sentinels are distinct, but the predicate degrades to POST).
        self.assertEqual(ThinkingBanStateHolder._derive_state(3, 3), POST_THINK)


class TestInitStateEntry(unittest.TestCase):
    def setUp(self) -> None:
        self.h = _make_holder()

    def test_empty_prompt_is_pre_think(self) -> None:
        entry = self.h._init_state_entry(None)
        self.assertEqual(entry["last_start_pos"], -1)
        self.assertEqual(entry["last_end_pos"], -1)
        self.assertEqual(entry["prompt_len"], 0)

    def test_prompt_with_open_think_is_in(self) -> None:
        prompt = [1, 2, THINK_START, 3, 4]
        entry = self.h._init_state_entry(prompt)
        self.assertEqual(entry["last_start_pos"], 2)
        self.assertEqual(entry["last_end_pos"], -1)
        self.assertEqual(entry["prompt_len"], 5)
        self.assertEqual(
            ThinkingBanStateHolder._derive_state(
                entry["last_start_pos"], entry["last_end_pos"]
            ),
            IN_THINK,
        )

    def test_prompt_with_closed_think_is_post(self) -> None:
        prompt = [THINK_START, 1, 2, THINK_END, 3]
        entry = self.h._init_state_entry(prompt)
        self.assertEqual(entry["last_start_pos"], 0)
        self.assertEqual(entry["last_end_pos"], 3)
        self.assertEqual(
            ThinkingBanStateHolder._derive_state(
                entry["last_start_pos"], entry["last_end_pos"]
            ),
            POST_THINK,
        )

    def test_reopen_after_close_is_in(self) -> None:
        prompt = [THINK_START, 1, THINK_END, 2, THINK_START, 3]
        entry = self.h._init_state_entry(prompt)
        # last <think> at index 4, last </think> at index 2 → IN
        self.assertEqual(entry["last_start_pos"], 4)
        self.assertEqual(entry["last_end_pos"], 2)
        self.assertEqual(
            ThinkingBanStateHolder._derive_state(
                entry["last_start_pos"], entry["last_end_pos"]
            ),
            IN_THINK,
        )

    def test_lone_end_in_prompt_is_post(self) -> None:
        # Multi-turn leak: an earlier turn's </think> survives in the prompt
        # without a matching <think>. The state seeds to POST so we ban the
        # model from re-emitting </think>.
        prompt = [1, THINK_END, 2]
        entry = self.h._init_state_entry(prompt)
        self.assertEqual(entry["last_start_pos"], -1)
        self.assertEqual(entry["last_end_pos"], 1)
        self.assertEqual(
            ThinkingBanStateHolder._derive_state(
                entry["last_start_pos"], entry["last_end_pos"]
            ),
            POST_THINK,
        )


# ----------------------------------------------------------------- lifecycle


class TestSyncBatch(unittest.TestCase):
    def setUp(self) -> None:
        self.h = _make_holder()

    def test_disabled_holder_no_op(self) -> None:
        # An empty-sentinel holder is is_enabled=False and refuses all updates.
        h = ThinkingBanStateHolder(
            reasoning_config=_make_reasoning_cfg(think_start=[], think_end=[]),
            tool_call_start_tid=TOOL_CALL_START,
            max_num_seqs=4,
            num_spec_tokens=0,
            device=torch.device("cpu"),
            is_pin_memory=False,
        )
        bu = _batch_update(
            added=[(0, ParamsWithArgs(), [THINK_START, 1], [])],
        )
        h.sync_batch(bu)
        self.assertFalse(h.has_tracked_requests())

    def test_none_batch_update_no_op(self) -> None:
        self.h.sync_batch(None)
        self.assertFalse(self.h.has_tracked_requests())

    def test_add_seeds_initial_state(self) -> None:
        bu = _batch_update(
            added=[(0, ParamsWithArgs(), [THINK_START, 1], [])],
        )
        self.h.sync_batch(bu)
        self.assertTrue(self.h.has_tracked_requests())
        s = self.h._state[0]
        self.assertEqual(s["last_start_pos"], 0)
        self.assertEqual(s["last_end_pos"], -1)
        self.assertEqual(s["prompt_len"], 2)

    def test_remove_pops_state(self) -> None:
        self.h.sync_batch(_batch_update(added=[(0, ParamsWithArgs(), [1], [])]))
        self.h.sync_batch(_batch_update(removed=[0]))
        self.assertFalse(self.h.has_tracked_requests())

    def test_swap_swaps_two_states(self) -> None:
        self.h.sync_batch(
            _batch_update(
                added=[
                    (0, ParamsWithArgs(), [THINK_START], []),
                    (1, ParamsWithArgs(), [THINK_END], []),
                ]
            )
        )
        self.assertEqual(self.h._state[0]["last_start_pos"], 0)
        self.assertEqual(self.h._state[1]["last_end_pos"], 0)
        self.h.sync_batch(
            _batch_update(moved=[(0, 1, MoveDirectionality.SWAP)])
        )
        # Slot 1 now holds the formerly-at-0 state (last_start_pos=0)
        self.assertEqual(self.h._state[1]["last_start_pos"], 0)
        self.assertEqual(self.h._state[0]["last_end_pos"], 0)

    def test_swap_with_one_side_missing(self) -> None:
        self.h.sync_batch(
            _batch_update(added=[(0, ParamsWithArgs(), [THINK_START], [])])
        )
        # Swap slot 0 (present) with slot 5 (missing).
        self.h.sync_batch(
            _batch_update(
                moved=[(0, 5, MoveDirectionality.SWAP)], batch_size=6
            )
        )
        self.assertNotIn(0, self.h._state)
        self.assertEqual(self.h._state[5]["last_start_pos"], 0)

    def test_unidirectional_moves_state(self) -> None:
        self.h.sync_batch(
            _batch_update(added=[(2, ParamsWithArgs(), [THINK_START], [])])
        )
        self.h.sync_batch(
            _batch_update(
                moved=[(2, 0, MoveDirectionality.UNIDIRECTIONAL)],
                batch_size=3,
            )
        )
        self.assertNotIn(2, self.h._state)
        self.assertEqual(self.h._state[0]["last_start_pos"], 0)


# ------------------------------------------------------------- update_state


class TestUpdateState(unittest.TestCase):
    def setUp(self) -> None:
        self.h = _make_holder()

    def _add(self, idx: int, prompt: list[int]) -> None:
        self.h.sync_batch(
            _batch_update(added=[(idx, ParamsWithArgs(), prompt, [])])
        )

    def test_no_op_when_disabled(self) -> None:
        h = ThinkingBanStateHolder(
            reasoning_config=_make_reasoning_cfg(think_start=[], think_end=[]),
            tool_call_start_tid=TOOL_CALL_START,
            max_num_seqs=4,
            num_spec_tokens=0,
            device=torch.device("cpu"),
            is_pin_memory=False,
        )
        h.update_state([[1, 2]], None)
        self.assertFalse(h.has_tracked_requests())

    def test_pre_to_in_transition(self) -> None:
        self._add(0, [1, 2])  # PRE_THINK
        self.h.update_state(output_token_ids=[[5, THINK_START]], spec_token_ids=None)
        s = self.h._state[0]
        # last_start_pos = prompt_len (2) + 0 (prev_output_length=0) + 1 (index in new tokens)
        self.assertEqual(s["last_start_pos"], 2 + 1)
        self.assertEqual(s["last_end_pos"], -1)
        self.assertEqual(s["prev_output_length"], 2)

    def test_in_to_post_transition(self) -> None:
        self._add(0, [THINK_START, 1])  # IN_THINK
        self.h.update_state(output_token_ids=[[5, THINK_END]], spec_token_ids=None)
        s = self.h._state[0]
        self.assertEqual(s["last_start_pos"], 0)  # from prompt
        # last_end at prompt_len(2) + 1 = 3
        self.assertEqual(s["last_end_pos"], 3)

    def test_post_to_in_reentry(self) -> None:
        # Closed first block, then re-opened thinking inside generation.
        self._add(0, [THINK_START, 1, THINK_END])  # POST
        self.h.update_state(
            output_token_ids=[[2, THINK_START, 3]], spec_token_ids=None
        )
        s = self.h._state[0]
        # last_start at prompt_len(3) + 1 = 4 — beats last_end_pos (2 in prompt)
        self.assertEqual(s["last_start_pos"], 4)
        self.assertEqual(s["last_end_pos"], 2)

    def test_strips_spec_suffix(self) -> None:
        self._add(0, [1])
        # output ends with 3 spec tokens that should NOT advance committed state.
        self.h.update_state(
            output_token_ids=[[5, 6, THINK_START, 7, 8]],
            spec_token_ids=[[THINK_START, 7, 8]],
        )
        s = self.h._state[0]
        # After stripping spec suffix, committed = [5, 6]. No sentinel hit.
        self.assertEqual(s["last_start_pos"], -1)
        self.assertEqual(s["last_end_pos"], -1)
        self.assertEqual(s["prev_output_length"], 2)

    def test_no_grow_skip(self) -> None:
        self._add(0, [1])
        # First step: output [5]
        self.h.update_state(output_token_ids=[[5]], spec_token_ids=None)
        self.assertEqual(self.h._state[0]["prev_output_length"], 1)
        # Second step: output unchanged (e.g. recovery)
        self.h.update_state(output_token_ids=[[5]], spec_token_ids=None)
        self.assertEqual(self.h._state[0]["prev_output_length"], 1)
        self.assertEqual(self.h._state[0]["last_start_pos"], -1)

    def test_with_repeat_indices(self) -> None:
        # Mimics the RejectionSampler row layout: each request contributes
        # `num_draft_tokens[i]` rows. repeat_indices says which request each
        # row belongs to; the holder reads only the last row per request.
        self._add(0, [1])
        self._add(1, [THINK_START])
        # 4 rows: req 0 has 2 rows (rows 0,1), req 1 has 2 rows (rows 2,3).
        # We pretend last row 1 has [5, THINK_END] and last row 3 has [5, 6].
        rpt = torch.tensor([0, 0, 1, 1])
        output = [
            [4, 5],           # row 0 (req 0)
            [5, THINK_END],   # row 1 (req 0) — last for req 0
            [9, 8],           # row 2 (req 1)
            [5, 6],           # row 3 (req 1) — last for req 1
        ]
        self.h.update_state(output, None, repeat_indices=rpt)
        self.assertGreaterEqual(self.h._state[0]["last_end_pos"], 0)
        # Req 1 had prompt [THINK_START] then committed [5, 6] — no end seen.
        self.assertEqual(self.h._state[1]["last_end_pos"], -1)


# ------------------------------------------------------------ apply_to_logits


class TestApplyToLogitsNonSpec(unittest.TestCase):
    def setUp(self) -> None:
        self.h = _make_holder(num_spec_tokens=0)

    def _add(self, idx: int, prompt: list[int]) -> None:
        self.h.sync_batch(
            _batch_update(added=[(idx, ParamsWithArgs(), prompt, [])])
        )

    def test_pre_think_no_change(self) -> None:
        self._add(0, [1, 2])
        logits = torch.zeros(1, VOCAB_SIZE)
        out = self.h.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=None)
        self.assertTrue(torch.equal(out, torch.zeros(1, VOCAB_SIZE)))

    def test_in_think_bans_tool_call(self) -> None:
        self._add(0, [THINK_START, 1])
        logits = torch.zeros(1, VOCAB_SIZE)
        self.h.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=None)
        self.assertEqual(logits[0, TOOL_CALL_START].item(), float("-inf"))
        # other tokens untouched
        self.assertEqual(logits[0, 0].item(), 0.0)
        self.assertEqual(logits[0, THINK_END].item(), 0.0)

    def test_post_think_bans_think_end(self) -> None:
        self._add(0, [THINK_START, 1, THINK_END])
        logits = torch.zeros(1, VOCAB_SIZE)
        self.h.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=None)
        self.assertEqual(logits[0, THINK_END].item(), float("-inf"))
        # tool_call NOT banned in POST_THINK
        self.assertEqual(logits[0, TOOL_CALL_START].item(), 0.0)

    def test_mixed_batch_per_row(self) -> None:
        self._add(0, [THINK_START])               # IN
        self._add(1, [THINK_START, 1, THINK_END]) # POST
        self._add(2, [1, 2])                       # PRE
        logits = torch.zeros(3, VOCAB_SIZE)
        self.h.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=None)
        self.assertEqual(logits[0, TOOL_CALL_START].item(), float("-inf"))
        self.assertEqual(logits[1, THINK_END].item(), float("-inf"))
        # row 2 untouched
        self.assertEqual(logits[2, TOOL_CALL_START].item(), 0.0)
        self.assertEqual(logits[2, THINK_END].item(), 0.0)


class TestApplyToLogitsSpec(unittest.TestCase):
    """Per-row classification under MTP-K=3."""

    def setUp(self) -> None:
        self.h = _make_holder(num_spec_tokens=3)

    def _add(self, idx: int, prompt: list[int]) -> None:
        self.h.sync_batch(
            _batch_update(added=[(idx, ParamsWithArgs(), prompt, [])])
        )

    def test_target_rows_walk_drafts_in_think(self) -> None:
        # Single request, base state = IN_THINK.
        # K=3 drafts: ['.', '</think>', '<|tool_call_start|>'-equivalent token].
        # Rows: 0 (no drafts consumed) IN, 1 (after '.') IN, 2 (after '</think>')
        # POST.
        self._add(0, [THINK_START])
        spec_tokens = [[42, THINK_END, 5]]
        logits = torch.zeros(3, VOCAB_SIZE)
        self.h.apply_to_logits(
            logits, predict_bonus_token=False, spec_token_ids=spec_tokens
        )
        # Row 0: IN → tool_call banned
        self.assertEqual(logits[0, TOOL_CALL_START].item(), float("-inf"))
        # Row 1: still IN (after '.') → tool_call banned
        self.assertEqual(logits[1, TOOL_CALL_START].item(), float("-inf"))
        # Row 2: POST (after </think>) → think_end banned, tool_call NOT banned
        self.assertEqual(logits[2, THINK_END].item(), float("-inf"))
        self.assertEqual(logits[2, TOOL_CALL_START].item(), 0.0)

    def test_bonus_row_walks_all_drafts(self) -> None:
        # Same setup, but predict_bonus_token=True → 1 row per request, state
        # after consuming all 3 drafts (final = POST).
        self._add(0, [THINK_START])
        spec_tokens = [[42, THINK_END, 5]]
        logits = torch.zeros(1, VOCAB_SIZE)
        self.h.apply_to_logits(
            logits, predict_bonus_token=True, spec_token_ids=spec_tokens
        )
        self.assertEqual(logits[0, THINK_END].item(), float("-inf"))
        self.assertEqual(logits[0, TOOL_CALL_START].item(), 0.0)

    def test_two_reqs_distinct_states(self) -> None:
        # Req 0 IN, Req 1 PRE. K=2 drafts each (we only fill 2 below).
        self._add(0, [THINK_START])
        self._add(1, [1, 2])
        spec_tokens = [[42, 43], [44, THINK_START]]
        # 4 target rows total (2 per req).
        logits = torch.zeros(4, VOCAB_SIZE)
        self.h.apply_to_logits(
            logits, predict_bonus_token=False, spec_token_ids=spec_tokens
        )
        # req 0 rows 0 and 1: both IN (drafts are arbitrary non-sentinels)
        self.assertEqual(logits[0, TOOL_CALL_START].item(), float("-inf"))
        self.assertEqual(logits[1, TOOL_CALL_START].item(), float("-inf"))
        # req 1 row 0 (start_row = 2): PRE → untouched
        self.assertEqual(logits[2, TOOL_CALL_START].item(), 0.0)
        # req 1 row 1 (start_row + 1 = 3): after THINK_START draft? No — the
        # walk consumes draft *at index 0* before row 1. So row 1 is "after
        # drafts[0..0]" = "after 44" = PRE still. Row 2 would be IN if it
        # existed, but with only 2 drafts there's no row 2 in target layout.
        self.assertEqual(logits[3, TOOL_CALL_START].item(), 0.0)

    def test_disabled_when_no_tracked(self) -> None:
        # No requests added: apply_to_logits should be a no-op.
        logits = torch.zeros(3, VOCAB_SIZE)
        before = logits.clone()
        self.h.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=[[1, 2, 3]])
        self.assertTrue(torch.equal(logits, before))


if __name__ == "__main__":
    unittest.main()
