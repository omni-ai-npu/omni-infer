# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import torch

from omni.v1.sample.logits_processor.mineru_logits_processor import (
    MinerULogitsProcessor,
    _get_int_value,
)
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor.interface import MoveDirectionality


@dataclass
class ParamsWithArgs:
    extra_args: dict[str, Any] | None = None


def _make_processor() -> MinerULogitsProcessor:
    vllm_config = MagicMock(spec=VllmConfig)
    return MinerULogitsProcessor(vllm_config, torch.device("cpu"), False)


class TestGetIntValue(unittest.TestCase):
    def test_returns_int(self):
        self.assertEqual(_get_int_value({"key": 5}, "key"), 5)

    def test_converts_str_to_int(self):
        self.assertEqual(_get_int_value({"key": "10"}, "key"), 10)

    def test_returns_none_for_missing_key(self):
        self.assertIsNone(_get_int_value({"key": 5}, "other"))

    def test_returns_none_for_none_args(self):
        self.assertIsNone(_get_int_value(None, "key"))

    def test_returns_none_for_non_dict(self):
        self.assertIsNone(_get_int_value("not_a_dict", "key"))

    def test_returns_none_for_invalid_value(self):
        self.assertIsNone(_get_int_value({"key": "invalid"}, "key"))

    def test_returns_none_for_none_value(self):
        self.assertIsNone(_get_int_value({"key": None}, "key"))


class TestMinerULogitsProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = _make_processor()

    def test_is_argmax_invariant(self):
        self.assertFalse(self.processor.is_argmax_invariant())

    def test_update_state_none_does_nothing(self):
        self.processor.req_info[0] = (2, [1, 2, 3], {})
        self.processor.update_state(None)
        self.assertIn(0, self.processor.req_info)

    def test_update_state_added_with_ngram_size(self):
        bu = MagicMock()
        bu.added = [(0, ParamsWithArgs({"no_repeat_ngram_size": 3}), [1, 2], [5, 6])]
        bu.removed = []
        bu.moved = []
        self.processor.update_state(bu)
        ngram_size, output_tok_ids, cached_ngrams = self.processor.req_info[0]
        self.assertEqual(ngram_size, 3)
        self.assertEqual(output_tok_ids, [5, 6])
        self.assertEqual(cached_ngrams, {})

    def test_update_state_added_without_extra_args(self):
        bu = MagicMock()
        bu.added = [(0, ParamsWithArgs(None), [1, 2], [5, 6])]
        bu.removed = []
        bu.moved = []
        self.processor.update_state(bu)
        self.assertEqual(self.processor.req_info[0][0], 0)

    def test_update_state_added_with_negative_ngram_size(self):
        bu = MagicMock()
        bu.added = [(0, ParamsWithArgs({"no_repeat_ngram_size": -5}), [1, 2], [])]
        bu.removed = []
        bu.moved = []
        self.processor.update_state(bu)
        self.assertEqual(self.processor.req_info[0][0], 0)

    def test_update_state_removed(self):
        self.processor.req_info[0] = (2, [1, 2], {})
        bu = MagicMock()
        bu.added = []
        bu.removed = [0]
        bu.moved = []
        self.processor.update_state(bu)
        self.assertNotIn(0, self.processor.req_info)

    def test_update_state_moved_unidirectional(self):
        self.processor.req_info[5] = (3, [10, 20, 30], {(10, 20): [30]})
        bu = MagicMock()
        bu.added = []
        bu.removed = []
        bu.moved = [(5, 10, MoveDirectionality.UNIDIRECTIONAL)]
        self.processor.update_state(bu)
        self.assertNotIn(5, self.processor.req_info)
        ngram_size, output_tok_ids, cached_ngrams = self.processor.req_info[10]
        self.assertEqual(ngram_size, 3)
        self.assertEqual(output_tok_ids, [10, 20, 30])
        self.assertEqual(cached_ngrams, {(10, 20): [30]})

    def test_update_state_moved_swap(self):
        info_a = (2, [1, 2], {})
        info_b = (3, [3, 4, 5], {(3, 4): [5]})
        self.processor.req_info[0] = info_a
        self.processor.req_info[1] = info_b
        bu = MagicMock()
        bu.added = []
        bu.removed = []
        bu.moved = [(0, 1, MoveDirectionality.SWAP)]
        self.processor.update_state(bu)
        self.assertEqual(self.processor.req_info[0], info_b)
        self.assertEqual(self.processor.req_info[1], info_a)

    def test_apply_no_req_info(self):
        logits = torch.zeros((2, 100))
        result = self.processor.apply(logits)
        self.assertIs(result, logits)

    def test_apply_ngram_size_zero_does_nothing(self):
        self.processor.req_info[0] = (0, [1, 2, 3], {})
        logits = torch.zeros((1, 100))
        logits[0, 5] = 1.0
        result = self.processor.apply(logits)
        self.assertEqual(result[0, 5].item(), 1.0)

    def test_apply_insufficient_tokens_does_nothing(self):
        self.processor.req_info[0] = (3, [1, 2], {})
        logits = torch.zeros((1, 100))
        logits[0, 5] = 1.0
        result = self.processor.apply(logits)
        self.assertEqual(result[0, 5].item(), 1.0)

    def test_apply_bans_repeated_ngram(self):
        # output: [1, 2, 3, 1, 2], ngram_size=3
        # prev_ngram = (3, 1), last_token = 2 -> cache (3, 1): [2]
        # current_prefix = (1, 2) -> banned from cache: [] (no match)
        # After apply, cache will have (3, 1): [2]
        # To ban token 3, we need (1, 2): [3] in cache
        cached_ngrams = {(1, 2): [3]}
        self.processor.req_info[0] = (3, [1, 2, 3, 1, 2], cached_ngrams)

        logits = torch.zeros((1, 100))
        logits[0, 3] = 1.0
        logits[0, 2] = 2.0
        logits[0, 5] = 3.0

        result = self.processor.apply(logits)

        # current_prefix = (1, 2), banned = cached_ngrams[(1,2)] = [3]
        self.assertEqual(result[0, 3].item(), float("-inf"))
        self.assertEqual(result[0, 2].item(), 2.0)  # not banned
        self.assertEqual(result[0, 5].item(), 3.0)

    def test_apply_bans_multiple_tokens(self):
        # output: [1, 5], ngram_size=2
        # prev_ngram = (1,), last_token = 5 -> cache (1,): [5]
        # current_prefix = (5,) -> banned from cache: [10, 20]
        cached_ngrams = {(5,): [10, 20]}
        self.processor.req_info[0] = (2, [1, 5], cached_ngrams)

        logits = torch.zeros((1, 100))
        logits[0, 10] = 1.0
        logits[0, 20] = 2.0
        logits[0, 30] = 3.0

        result = self.processor.apply(logits)

        self.assertEqual(result[0, 10].item(), float("-inf"))
        self.assertEqual(result[0, 20].item(), float("-inf"))
        self.assertEqual(result[0, 30].item(), 3.0)

    def test_apply_multiple_requests(self):
        self.processor.req_info[0] = (2, [1, 2], {(2,): [5]})
        self.processor.req_info[1] = (2, [3, 4], {(4,): [10]})

        logits = torch.zeros((2, 100))
        logits[0, 5] = 1.0
        logits[1, 10] = 2.0

        result = self.processor.apply(logits)

        self.assertEqual(result[0, 5].item(), float("-inf"))
        self.assertEqual(result[1, 10].item(), float("-inf"))

    def test_apply_missing_req_info_for_index(self):
        self.processor.req_info[1] = (2, [1, 2], {(2,): [5]})

        logits = torch.zeros((2, 100))
        logits[0, 5] = 1.0
        logits[1, 5] = 2.0

        result = self.processor.apply(logits)

        self.assertEqual(result[0, 5].item(), 1.0)
        self.assertEqual(result[1, 5].item(), float("-inf"))

    def test_apply_updates_cached_ngrams(self):
        cached_ngrams = {}
        self.processor.req_info[0] = (3, [1, 2, 3], cached_ngrams)

        logits = torch.zeros((1, 100))
        self.processor.apply(logits)
        self.assertEqual(cached_ngrams, {(1, 2): [3]})

    def test_apply_accumulates_banned_tokens(self):
        cached_ngrams = {}
        output_tok_ids = [1, 2, 3]
        self.processor.req_info[0] = (3, output_tok_ids, cached_ngrams)

        logits = torch.zeros((1, 100))
        self.processor.apply(logits)
        self.assertEqual(cached_ngrams, {(1, 2): [3]})

        output_tok_ids.append(4)
        self.processor.apply(logits)
        self.assertEqual(cached_ngrams[(2, 3)], [4])

        output_tok_ids.extend([1, 2])
        logits[0, 3] = 1.0
        result = self.processor.apply(logits)
        self.assertEqual(result[0, 3].item(), float("-inf"))


if __name__ == "__main__":
    unittest.main()