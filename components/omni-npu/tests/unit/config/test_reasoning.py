# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for ``omni_npu.v1.config.reasoning.ReasoningConfig``."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from omni_npu.v1.config import ReasoningConfig
from vllm.config.model import ModelConfig


class TestReasoningConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = MagicMock(spec=ModelConfig)

    def test_defaults_not_enabled(self) -> None:
        cfg = ReasoningConfig()
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.reasoning_start_token_ids)
        self.assertIsNone(cfg.reasoning_end_token_ids)

    def test_ban_tool_call_in_thinking_default_false(self) -> None:
        """The opt-in switch for the worker-side tool-call ban defaults to off."""
        cfg = ReasoningConfig()
        self.assertFalse(cfg.ban_tool_call_in_thinking)

    def test_as_argparse_dict(self) -> None:
        arg_dict = ReasoningConfig.as_argparse_dict()
        self.assertEqual(arg_dict["type"], str)
        self.assertIsNone(arg_dict["default"])
        self.assertIn("JSON string", arg_dict["help"])
        self.assertIn("reasoning", arg_dict["help"].lower())

    def test_as_argparse_dict_fallback_doc(self) -> None:
        with patch.object(ReasoningConfig, "__doc__", None):
            d = ReasoningConfig.as_argparse_dict()
        self.assertIn("Reasoning configuration.", d["help"])

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_short_circuit_already_set(self, mock_tok: MagicMock) -> None:
        cfg = ReasoningConfig()
        cfg._reasoning_start_token_ids = [1]
        cfg._reasoning_end_token_ids = [2]
        cfg.initialize_token_ids(self.model_config)
        mock_tok.assert_not_called()
        self.assertTrue(cfg.enabled)

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_missing_start_no_tokenizer(self, mock_tok: MagicMock) -> None:
        cfg = ReasoningConfig(reasoning_end_str="</t>")
        cfg.initialize_token_ids(self.model_config)
        mock_tok.assert_not_called()
        self.assertFalse(cfg.enabled)

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_missing_end_no_tokenizer(self, mock_tok: MagicMock) -> None:
        cfg = ReasoningConfig(reasoning_start_str="<t>")
        cfg.initialize_token_ids(self.model_config)
        mock_tok.assert_not_called()
        self.assertFalse(cfg.enabled)

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_uses_think_alias_strings(self, mock_tok: MagicMock) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [10] if s == "<a>" else [20]
        mock_tok.return_value = mock_enc

        cfg = ReasoningConfig(think_start_str="<a>", think_end_str="<b>")
        cfg.initialize_token_ids(self.model_config)

        self.assertEqual(cfg.reasoning_start_token_ids, [10])
        self.assertEqual(cfg.reasoning_end_token_ids, [20])
        self.assertTrue(cfg.enabled)
        mock_tok.assert_called_once_with(model_config=self.model_config)

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_reasoning_str_over_alias(self, mock_tok: MagicMock) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = (
            lambda s, add_special_tokens=False: [1] if s == "RS" else [2] if s == "RE" else [99]
        )
        mock_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="RS",
            reasoning_end_str="RE",
            think_start_str="XX",
            think_end_str="YY",
        )
        cfg.initialize_token_ids(self.model_config)
        mock_enc.encode.assert_any_call("RS", add_special_tokens=False)
        mock_enc.encode.assert_any_call("RE", add_special_tokens=False)
        self.assertEqual(cfg.reasoning_start_token_ids, [1])
        self.assertEqual(cfg.reasoning_end_token_ids, [2])

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_raises_when_encode_empty(self, mock_tok: MagicMock) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.return_value = []
        mock_tok.return_value = mock_enc

        cfg = ReasoningConfig(reasoning_start_str="<x>", reasoning_end_str="<y>")
        with self.assertRaises(ValueError) as ctx:
            cfg.initialize_token_ids(self.model_config)
        self.assertIn("failed to tokenize", str(ctx.exception))

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_start_token_id_not_resolved_when_ban_off(
        self, mock_tok: MagicMock
    ) -> None:
        """Ban off → resolution skipped, field stays None even if vocab has the key."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"<|tool_call_start|>": 999}
        mock_tok.return_value = mock_enc

        cfg = ReasoningConfig(reasoning_start_str="<s>", reasoning_end_str="<e>")
        cfg.initialize_token_ids(self.model_config)
        self.assertIsNone(cfg.tool_call_start_token_id)

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_start_token_id_resolved_when_ban_on(
        self, mock_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"<|tool_call_start|>": 999}
        mock_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_call_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.tool_call_start_token_id, 999)

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_start_token_id_falls_back_to_unused11(
        self, mock_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        # No <|tool_call_start|>, but [unused11] is present.
        mock_enc.get_vocab.return_value = {"[unused11]": 11}
        mock_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_call_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.tool_call_start_token_id, 11)

    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_start_token_id_none_when_no_key(
        self, mock_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"foo": 1, "bar": 2}
        mock_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_call_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertIsNone(cfg.tool_call_start_token_id)


if __name__ == "__main__":
    unittest.main()
