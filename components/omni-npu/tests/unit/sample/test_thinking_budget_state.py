# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for ``thinking_budget_state`` (ThinkingBudgetStateHolder)."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import torch

from omni_npu.v1.config import ReasoningConfig
from omni_npu.v1.sample.thinking_budget_state import (
    ThinkingBudgetStateHolder,
    _get_thinking_token_budget,
    _normalize_thinking_token_budget,
    maybe_create_thinking_budget_state_holder,
)
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality


@dataclass
class ParamsWithArgs:
    extra_args: dict[str, Any] | None = None


def _make_reasoning_cfg(
    think_start: list[int] | None = None,
    think_end: list[int] | None = None,
) -> ReasoningConfig:
    rc = ReasoningConfig()
    rc._reasoning_start_token_ids = [100] if think_start is None else think_start
    rc._reasoning_end_token_ids = [200] if think_end is None else think_end
    rc._enabled = True
    return rc


def _make_holder(
    *,
    think_start: list[int] | None = None,
    think_end: list[int] | None = None,
    reasoning_config: ReasoningConfig | None = "default",
    max_num_seqs: int = 4,
    num_spec_tokens: int = 0,
    device: torch.device | None = None,
) -> ThinkingBudgetStateHolder:
    if reasoning_config == "default":
        rc = _make_reasoning_cfg(think_start, think_end)
    else:
        rc = reasoning_config  # type: ignore[assignment]
    dev = device or torch.device("cpu")
    return ThinkingBudgetStateHolder(rc, max_num_seqs, num_spec_tokens, dev, False)


class TestNormalizeThinkingTokenBudget(unittest.TestCase):
    def test_none(self) -> None:
        self.assertIsNone(_normalize_thinking_token_budget(None))

    def test_valid_non_negative_integers(self) -> None:
        self.assertEqual(_normalize_thinking_token_budget(0), 0)
        self.assertEqual(_normalize_thinking_token_budget(7), 7)

    def test_invalid_values(self) -> None:
        for value in (-4, -1, 1.5, "3", True, False):
            self.assertIsNone(_normalize_thinking_token_budget(value))


class TestGetThinkingTokenBudget(unittest.TestCase):
    def test_no_extra_args_attr(self) -> None:
        self.assertIsNone(_get_thinking_token_budget(object()))

    def test_extra_args_none(self) -> None:
        self.assertIsNone(_get_thinking_token_budget(ParamsWithArgs(None)))

    def test_missing_key(self) -> None:
        self.assertIsNone(_get_thinking_token_budget(ParamsWithArgs({})))

    def test_present(self) -> None:
        self.assertEqual(
            _get_thinking_token_budget(ParamsWithArgs({"thinking_token_budget": 7})), 7
        )

    def test_negative_in_extra_args(self) -> None:
        self.assertIsNone(
            _get_thinking_token_budget(ParamsWithArgs({"thinking_token_budget": -4}))
        )


class TestMaybeCreateHolder(unittest.TestCase):
    def test_none_config(self) -> None:
        h = maybe_create_thinking_budget_state_holder(None, 2, 0, torch.device("cpu"), False)
        self.assertIsNone(h)

    def test_with_config(self) -> None:
        rc = _make_reasoning_cfg()
        h = maybe_create_thinking_budget_state_holder(rc, 2, 1, torch.device("cpu"), False)
        self.assertIsInstance(h, ThinkingBudgetStateHolder)
        self.assertTrue(h.in_spec_mode)


class TestThinkingBudgetStateHolder(unittest.TestCase):
    def setUp(self) -> None:
        self.h = _make_holder()

    def test_holder_disabled_when_no_reasoning_config(self) -> None:
        h = ThinkingBudgetStateHolder(None, 2, 0, torch.device("cpu"), False)
        self.assertFalse(h.is_enabled)
        self.assertEqual(h.think_start_token_ids, [])
        self.assertEqual(h.think_end_token_ids, [])

    def test_has_tracked_requests_after_sync(self) -> None:
        self.assertFalse(self.h.has_tracked_requests())
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 1}), [1, 2], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu)
        self.assertTrue(self.h.has_tracked_requests())

    def test_sync_batch_ignores_invalid_budget(self) -> None:
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": -4}), [1, 2], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu)
        self.assertFalse(self.h.has_tracked_requests())

    def test_sync_batch_noop_when_disabled(self) -> None:
        h = ThinkingBudgetStateHolder(None, 2, 0, torch.device("cpu"), False)
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 1}), [1], [])],
            removed=[],
            moved=[],
        )
        h.sync_batch(bu)
        self.assertFalse(h.has_tracked_requests())

    def test_sync_batch_noop_when_no_update(self) -> None:
        self.h.sync_batch(None)
        self.assertFalse(self.h.has_tracked_requests())

    def test_find_last_sequence_index(self) -> None:
        self.assertEqual(
            ThinkingBudgetStateHolder._find_last_sequence_index([1, 2, 3, 1, 2, 3, 4], [1, 2]),
            3,
        )
        self.assertEqual(
            ThinkingBudgetStateHolder._find_last_sequence_index([1, 2, 3], [9, 9]),
            -1,
        )
        self.assertEqual(ThinkingBudgetStateHolder._find_last_sequence_index([1, 2], []), -1)

    def test_init_state_entry_from_prompt_in_think(self) -> None:
        prompt_ids = [1, 2, 100, 3, 4]
        state = self.h._init_state_entry(prompt_ids, thinking_token_budget=10)
        self.assertTrue(state["in_think"])
        self.assertEqual(state["think_count"], 2)
        self.assertFalse(state["in_end"])
        self.assertIs(state["prompt_tok_ids"], prompt_ids)

    def test_init_state_entry_think_closed_in_prompt(self) -> None:
        prompt_ids = [1, 100, 3, 200, 5]
        state = self.h._init_state_entry(prompt_ids, thinking_token_budget=10)
        self.assertFalse(state["in_think"])
        self.assertEqual(state["think_count"], 0)

    def test_init_state_entry_no_prompt(self) -> None:
        state = self.h._init_state_entry(None, thinking_token_budget=3)
        self.assertFalse(state["in_think"])
        self.assertEqual(state["check_count_down"], 3)
        self.assertIsNone(state["prompt_tok_ids"])

    def test_init_state_entry_budget_zero_while_in_think_sets_in_end(self) -> None:
        prompt_ids = [100, 1, 2]
        state = self.h._init_state_entry(prompt_ids, thinking_token_budget=0)
        self.assertTrue(state["in_think"])
        self.assertTrue(state["in_end"])

    def test_update_think_state_flat_in_end_force_index_mid_spec(self) -> None:
        """``0 < remaining_budget < spec_len`` when output length does not grow."""
        state = {
            "in_think": False,
            "in_end": True,
            "think_count": 4,
            "thinking_token_budget": 5,
            "prev_output_length": 2,
            "check_count_down": 0,
            "output_tok_ids": [1, 2],
            "spec_token_ids": [40, 41, 42],
            "start_thinking": 0,
            "continue_thinking": False,
            "end_thinking": -1,
            "end_count": 0,
        }
        self.h._update_think_state(state)
        self.assertEqual(state["force_index"], [1])

    def test_update_think_state_flat_in_end_force_zero_when_exhausted(self) -> None:
        """``remaining_budget <= 0`` branch in flat output / ``in_end`` path."""
        state = {
            "in_think": False,
            "in_end": True,
            "think_count": 6,
            "thinking_token_budget": 5,
            "prev_output_length": 1,
            "check_count_down": 0,
            "output_tok_ids": [9],
            "spec_token_ids": [40],
            "start_thinking": 0,
            "continue_thinking": False,
            "end_thinking": -1,
            "end_count": 0,
        }
        self.h._update_think_state(state)
        self.assertEqual(state["force_index"], [0])

    def test_update_think_state_continue_thinking_adjusts_absolute_end(self) -> None:
        """``continue_thinking`` plus resolved ``end_thinking`` shifts absolute end."""
        self.h.think_start_token_ids = [10]
        self.h.think_end_token_ids = [20]
        state = {
            "in_think": True,
            "in_end": False,
            "think_count": 1,
            "thinking_token_budget": 20,
            "prev_output_length": 1,
            "output_tok_ids": [10, 5, 6, 20, 7],
            "spec_token_ids": [],
            "check_count_down": 0,
            "start_thinking": 0,
            "continue_thinking": True,
            "end_thinking": 3,
            "prompt_tok_ids": [10, 1],
            "end_count": 0,
        }
        self.h._update_think_state(state)

    def test_update_think_state_start_after_end_sets_think_count(self) -> None:
        """``absolute_start_pos > absolute_end_pos`` enters inner think branch."""
        self.h.think_start_token_ids = [100]
        self.h.think_end_token_ids = [200]
        state = {
            "in_think": False,
            "in_end": False,
            "think_count": 0,
            "thinking_token_budget": 30,
            "prev_output_length": 0,
            "check_count_down": 0,
            "output_tok_ids": [200, 1, 2, 100, 9, 10],
            "spec_token_ids": [],
            "start_thinking": 3,
            "end_thinking": 0,
            "continue_thinking": False,
            "prompt_tok_ids": [],
            "end_count": 0,
        }
        self.h._update_think_state(state)
        self.assertTrue(state["in_think"])

    def test_update_think_state_in_think_increment_from_prompt(self) -> None:
        """``elif state['in_think']`` uses prompt offset for think_count."""
        self.h.think_start_token_ids = [10]
        self.h.think_end_token_ids = [20]
        state = {
            "in_think": True,
            "in_end": False,
            "think_count": 0,
            "thinking_token_budget": 40,
            "prev_output_length": 2,
            "check_count_down": 0,
            "output_tok_ids": [10, 1, 2, 3],
            "spec_token_ids": [],
            "start_thinking": 0,
            "end_thinking": -1,
            "continue_thinking": False,
            "prompt_tok_ids": [10, 5, 5, 5],
            "end_count": 0,
        }
        self.h._update_think_state(state)
        self.assertGreater(state["think_count"], 0)

    def test_update_think_state_budget_exceed_force_mid_spec(self) -> None:
        """Over budget with draft tokens: ``0 < remaining_budget < spec_len``."""
        self.h.think_start_token_ids = [10]
        self.h.think_end_token_ids = [20]
        state = {
            "in_think": True,
            "in_end": False,
            "think_count": 3,
            "thinking_token_budget": 4,
            "prev_output_length": 3,
            "check_count_down": 0,
            "output_tok_ids": [10, 1, 2, 3, 4],
            "spec_token_ids": [30, 31, 32],
            "start_thinking": 0,
            "continue_thinking": False,
            "end_thinking": -1,
            "prompt_tok_ids": [],
            "end_count": 0,
        }
        self.h._update_think_state(state)
        self.assertTrue(state["in_end"])
        self.assertGreater(len(state.get("force_index", [])), 0)

    def test_update_think_state_not_in_think_resets_check_countdown(self) -> None:
        """``else`` branch sets ``check_count_down`` back to full budget."""
        self.h.think_start_token_ids = [10]
        self.h.think_end_token_ids = [20]
        state = {
            "in_think": False,
            "in_end": False,
            "think_count": 0,
            "thinking_token_budget": 11,
            "prev_output_length": 0,
            "check_count_down": 0,
            "output_tok_ids": [10, 1, 2, 20, 3],
            "spec_token_ids": [],
            "start_thinking": 0,
            "end_thinking": 3,
            "continue_thinking": False,
            "prompt_tok_ids": [],
            "end_count": 0,
        }
        self.h._update_think_state(state)
        self.assertEqual(state["check_count_down"], 11)

    def test_update_think_state_short_circuit_no_budget(self) -> None:
        state = {"thinking_token_budget": -1, "output_tok_ids": []}
        self.h._update_think_state(state)
        self.assertEqual(state["thinking_token_budget"], -1)

    def test_update_think_state_no_end_token_ids(self) -> None:
        h = _make_holder(think_start=[1], think_end=[])
        st = {
            "thinking_token_budget": 5,
            "in_end": True,
            "output_tok_ids": [],
            "force_index": [0],
            "start_thinking": 0,
            "end_thinking": -1,
        }
        h._update_think_state(st)
        self.assertEqual(st["thinking_token_budget"], -1)
        self.assertFalse(st["in_end"])
        self.assertEqual(st["force_index"], [])

    def test_update_think_state_countdown_early_return(self) -> None:
        state = {
            "in_think": True,
            "in_end": False,
            "check_count_down": 3,
            "output_tok_ids": [1, 2, 3],
            "prev_output_length": 0,
            "thinking_token_budget": 10,
            "spec_token_ids": [],
            "start_thinking": 0,
            "continue_thinking": False,
            "end_thinking": -1,
        }
        self.h._update_think_state(state)
        self.assertEqual(state["check_count_down"], 1)

    def test_update_think_state_empty_output_in_end_sets_force_zero(self) -> None:
        state = {
            "in_end": True,
            "in_think": False,
            "check_count_down": 0,
            "output_tok_ids": [],
            "prev_output_length": 0,
            "end_count": 0,
            "thinking_token_budget": 5,
            "spec_token_ids": [],
            "start_thinking": 0,
            "think_count": 0,
            "continue_thinking": False,
            "end_thinking": -1,
        }
        self.h._update_think_state(state)
        self.assertEqual(state.get("force_index"), [0])

    def test_update_think_state_enters_in_end_when_budgetExceeded(self) -> None:
        state = {
            "in_think": True,
            "in_end": False,
            "think_count": 5,
            "thinking_token_budget": 5,
            "output_tok_ids": [100, 11, 12, 13, 14, 15],
            "prev_output_length": 5,
            "check_count_down": 0,
            "spec_token_ids": [],
            "start_thinking": 0,
            "end_thinking": -1,
            "continue_thinking": False,
            "prompt_tok_ids": None,
            "end_count": 0,
        }
        self.h._update_think_state(state)
        self.assertTrue(state["in_end"])
        self.assertFalse(state["in_think"])

    def test_update_think_state_in_end_abort_resets_think(self) -> None:
        self.h.think_end_token_ids = [999]
        state = {
            "in_think": False,
            "in_end": True,
            "end_count": 0,
            "prev_output_length": 0,
            "output_tok_ids": [1, 2, 3],
            "thinking_token_budget": 10,
            "check_count_down": 0,
            "spec_token_ids": [],
            "start_thinking": 0,
            "think_count": 1,
            "continue_thinking": False,
            "end_thinking": -1,
            "bonus_token_forced": False,
        }
        self.h._update_think_state(state)
        self.assertTrue(state["in_think"])
        self.assertFalse(state["in_end"])

    def test_update_think_state_continue_thinking_branch(self) -> None:
        self.h.think_start_token_ids = [10]
        self.h.think_end_token_ids = [20]
        state = {
            "in_think": True,
            "in_end": False,
            "think_count": 1,
            "prev_output_length": 1,
            "output_tok_ids": [10, 1, 2, 3],
            "thinking_token_budget": 10,
            "check_count_down": 0,
            "spec_token_ids": [],
            "start_thinking": 0,
            "continue_thinking": True,
            "end_thinking": -1,
        }
        self.h._update_think_state(state)
        self.assertTrue(state["in_think"])

    def test_update_think_state_in_end_spec_branches(self) -> None:
        """Cover ``current_length <= prev_length`` spec_len branches in ``in_end``."""
        state = {
            "in_think": False,
            "in_end": True,
            "think_count": 0,
            "thinking_token_budget": 2,
            "prev_output_length": 2,
            "check_count_down": 0,
            "output_tok_ids": [1, 2],
            "spec_token_ids": [40, 41],
            "start_thinking": 0,
            "continue_thinking": False,
            "end_thinking": -1,
            "end_count": 0,
        }
        self.h._update_think_state(state)
        self.assertIn(state["force_index"][0], (0, 1, 2))

    def test_update_think_state_multi_end_tokens_in_end_block(self) -> None:
        h = _make_holder(think_start=[1], think_end=[300, 301])
        # Flat output step: stay in ``in_end`` without ``end_count==0`` abort
        # (new token would need to match ``think_end[0]`` when length grows).
        state = {
            "in_think": False,
            "in_end": True,
            "end_count": 0,
            "prev_output_length": 6,
            "output_tok_ids": [1, 2, 3, 4, 5, 6],
            "thinking_token_budget": 7,
            "check_count_down": 0,
            "spec_token_ids": [300],
            "start_thinking": 0,
            "think_count": 0,
            "continue_thinking": False,
            "end_thinking": -1,
        }
        h._update_think_state(state)
        self.assertTrue(state["in_end"])
        self.assertGreater(len(state.get("force_index", [])), 0)

    def test_sync_batch_add_remove_move(self) -> None:
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 10}), [1, 2, 100], [5, 5])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu)
        self.assertIn(0, self.h._state)

        bu2 = BatchUpdate(batch_size=0, added=[], removed=[0], moved=[])
        self.h.sync_batch(bu2)
        self.assertNotIn(0, self.h._state)

        self.h._state[5] = {"data": "a"}
        bu3 = BatchUpdate(
            batch_size=2, added=[], removed=[], moved=[(5, 10, MoveDirectionality.UNIDIRECTIONAL)]
        )
        self.h.sync_batch(bu3)
        self.assertNotIn(5, self.h._state)
        self.assertEqual(self.h._state[10]["data"], "a")

    def test_sync_batch_swap(self) -> None:
        a = self.h._init_state_entry([], 1)
        b = self.h._init_state_entry([], 2)
        self.h._state[0] = a
        self.h._state[1] = b
        bu = BatchUpdate(batch_size=2, added=[], removed=[], moved=[(0, 1, MoveDirectionality.SWAP)])
        self.h.sync_batch(bu)
        self.assertIs(self.h._state[0], b)
        self.assertIs(self.h._state[1], a)

    def test_update_state_repeat_indices_row(self) -> None:
        bu = BatchUpdate(
            batch_size=2,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 5}), [100, 1], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu)
        repeat = torch.tensor([1, 0], dtype=torch.long)
        self.h.update_state([[9], [10, 11]], None, repeat_indices=repeat)
        self.assertEqual(self.h._state[0]["output_tok_ids"], [10, 11])

    def test_update_state_repeat_indices_missing_seq_skips(self) -> None:
        bu = BatchUpdate(
            batch_size=2,
            added=[(2, ParamsWithArgs({"thinking_token_budget": 1}), [], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu)
        before = list(self.h._state[2]["output_tok_ids"])
        repeat = torch.tensor([0, 0], dtype=torch.long)
        self.h.update_state([[5], [6]], None, repeat_indices=repeat)
        self.assertEqual(self.h._state[2]["output_tok_ids"], before)

    def test_update_state_strips_spec_suffix(self) -> None:
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 5}), [], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu)
        self.h.update_state([[1, 2, 40, 41]], [[40, 41]])
        self.assertEqual(self.h._state[0]["output_tok_ids"], [1, 2])

    def test_update_state_returns_when_empty_tracking(self) -> None:
        self.h.update_state([[1, 2, 3]], None)
        self.assertFalse(self.h.has_tracked_requests())

    def test_update_state_skips_row_when_output_shorter_than_index(self) -> None:
        bu = BatchUpdate(
            batch_size=1,
            added=[(2, ParamsWithArgs({"thinking_token_budget": 3}), [], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu)
        self.h.update_state([[1], [2]], None)
        self.assertEqual(self.h._state[2]["output_tok_ids"], [])

    def test_sync_batch_replace_row_clears_budget(self) -> None:
        bu1 = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 1}), [], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu1)
        bu2 = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({}), [], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu2)
        self.assertFalse(self.h.has_tracked_requests())

    def test_apply_mask_skip_when_force_index_empty(self) -> None:
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 1}), [100], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu)
        st = self.h._state[0]
        st["in_end"] = True
        st["end_count"] = 0
        st["force_index"] = []
        logits = torch.zeros((1, 256), dtype=torch.float32)
        out = self.h.apply_to_logits(logits, False, None)
        self.assertTrue(torch.all(out[0] == 0))

    def test_apply_predict_bonus_resets_force_index_to_zero(self) -> None:
        h = _make_holder(num_spec_tokens=1)
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 1}), [100], [])],
            removed=[],
            moved=[],
        )
        h.sync_batch(bu)
        st = h._state[0]
        st["in_end"] = True
        st["end_count"] = 0
        st["force_index"] = [1]
        st["spec_token_ids"] = []
        logits = torch.zeros((2, 256), dtype=torch.float32)
        h.apply_to_logits(logits, predict_bonus_token=True, spec_token_ids=[[]])
        self.assertEqual(st["force_index"], [0])

    def test_apply_to_logits_noop(self) -> None:
        h = ThinkingBudgetStateHolder(None, 2, 0, torch.device("cpu"), False)
        logits = torch.zeros((1, 10))
        out = h.apply_to_logits(logits, False, None)
        self.assertIs(out, logits)

    def test_apply_to_logits_forces_active_end(self) -> None:
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 3}), [100, 1, 2], [])],
            removed=[],
            moved=[],
        )
        self.h.sync_batch(bu)
        self.h._state[0]["in_end"] = True
        self.h._state[0]["end_count"] = 0
        self.h._state[0]["force_index"] = [0]
        logits = torch.zeros((1, 300), dtype=torch.float32)
        out = self.h.apply_to_logits(logits, False, None)
        self.assertEqual(out[0, 200].item(), 1e9)

    def test_apply_to_logits_predict_bonus_skips_when_force_in_spec_range(self) -> None:
        """Bonus pass: force targets draft row — first branch ``continue`` in forcing."""
        h = _make_holder(num_spec_tokens=1)
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 2}), [100], [])],
            removed=[],
            moved=[],
        )
        h.sync_batch(bu)
        h._state[0].update(
            {
                "in_end": True,
                "end_count": 0,
                "force_index": [0],
                "spec_token_ids": [55],
                "output_tok_ids": [1],
            }
        )
        logits = torch.zeros((3, 50), dtype=torch.float32)
        _ = h.apply_to_logits(logits, predict_bonus_token=True, spec_token_ids=[[55]])
        self.assertEqual(h._state[0]["force_index"], [0])

    def test_apply_forcing_non_bonus_row_from_spec_layout(self) -> None:
        """Target logits row aligns with draft layout when ``predict_bonus_token`` is False."""
        h = _make_holder(num_spec_tokens=1)
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 1}), [100], [])],
            removed=[],
            moved=[],
        )
        h.sync_batch(bu)
        st = h._state[0]
        st["in_end"] = True
        st["end_count"] = 0
        st["force_index"] = [1]
        st["spec_token_ids"] = [77]
        st["output_tok_ids"] = [1]
        st["bonus_token_forced"] = False
        logits = torch.zeros((3, 512), dtype=torch.float32)
        h.apply_to_logits(logits, predict_bonus_token=False, spec_token_ids=[[77]])
        self.assertGreater(logits[1, 200].item(), 1e8)

    def test_apply_with_end_count_gt_zero_clears_bonus_flag(self) -> None:
        h = _make_holder(num_spec_tokens=0)
        bu = BatchUpdate(
            batch_size=1,
            added=[(0, ParamsWithArgs({"thinking_token_budget": 1}), [100], [])],
            removed=[],
            moved=[],
        )
        h.sync_batch(bu)
        st = h._state[0]
        st.update(
            {
                "in_end": True,
                "end_count": 1,
                "force_index": [0],
                "spec_token_ids": [],
                "output_tok_ids": [1, 2],
                "bonus_token_forced": True,
            }
        )
        logits = torch.ones((1, 512), dtype=torch.float32)
        h.apply_to_logits(logits, predict_bonus_token=True, spec_token_ids=[[]])
        self.assertFalse(st["bonus_token_forced"])

    def test_spec_mode_cu_num_tokens_layout(self) -> None:
        h = _make_holder(max_num_seqs=2, num_spec_tokens=2)
        self.assertEqual(h.mask.numel(), 2 * (2 + 1))


if __name__ == "__main__":
    unittest.main()
