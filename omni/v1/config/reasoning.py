# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.model import ModelConfig
from vllm.config.reasoning import ReasoningConfig as VllmReasoningConfig
from vllm.config.utils import config
from vllm.tokenizers import cached_tokenizer_from_config


@config
class ReasoningConfig(VllmReasoningConfig):
    """vLLM reasoning configuration extended with Omni thinking bans.

    The upstream fields and token initialisation remain authoritative. Omni
    adds only compatibility aliases and the Pangu tool-marker ban switches.
    `thinking_token_budget` is retained solely so existing JSON configs can
    still be parsed; request budgets are handled by vLLM's native field and
    native `ThinkingBudgetStateHolder`.
    """

    think_start_str: str = ""
    """Deprecated alias for `reasoning_start_str`."""
    think_end_str: str = ""
    """Deprecated alias for `reasoning_end_str`."""

    thinking_token_budget: int | None = None
    """Deprecated server-config compatibility field; not used for wiring."""

    ban_tool_start_in_thinking: bool = False
    """Ban the tool-call start marker while reasoning is still open."""
    ban_tool_end_in_thinking: bool = False
    """Ban the tool-call end marker while reasoning is still open."""

    _tool_call_start_token_id: int | None = field(
        default=None, init=False, repr=False
    )
    _tool_call_end_token_id: int | None = field(
        default=None, init=False, repr=False
    )

    @property
    def tool_call_start_token_id(self) -> int | None:
        return self._tool_call_start_token_id

    @property
    def tool_call_end_token_id(self) -> int | None:
        return self._tool_call_end_token_id

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Run native reasoning initialisation, then resolve ban marker IDs."""
        if not self.reasoning_start_str and self.think_start_str:
            self.reasoning_start_str = self.think_start_str
        if not self.reasoning_end_str and self.think_end_str:
            self.reasoning_end_str = self.think_end_str

        # This preserves vLLM 0.25.1's parser-derived marker support.
        super().initialize_token_ids(model_config)
        if not self.enabled or not (
            self.ban_tool_start_in_thinking or self.ban_tool_end_in_thinking
        ):
            return

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        vocab = tokenizer.get_vocab()

        # Pangu's legacy markers are kept as fallbacks for old checkpoints.
        if self.ban_tool_start_in_thinking:
            for marker in ("<|tool_call_start|>", "[unused11]"):
                if marker in vocab:
                    self._tool_call_start_token_id = int(vocab[marker])
                    break
        if self.ban_tool_end_in_thinking:
            for marker in ("<|tool_call_end|>", "[unused12]"):
                if marker in vocab:
                    self._tool_call_end_token_id = int(vocab[marker])
                    break
