# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

# NOTE: This parser is coupled to :class:`PanguToolParser` via the
# per-asyncio-task relay in ``_streaming_relay.py``. The streaming method
# below stashes reasoning text so that PanguToolParser can re-attach it
# after vLLM's serving_chat unconditionally overwrites the delta_message
# (workaround for the v0.14.0 streaming bug fixed upstream by PR #42691).
# See ``_streaming_relay.py`` for the full rationale.


import os
from collections.abc import Sequence
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.entrypoints.openai.protocol import DeltaMessage
from vllm.tokenizers import TokenizerLike
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.logger import logger

from omni_npu.v1.parsers._streaming_relay import stash_reasoning_from


class PanguReasoningParser(DeepSeekR1ReasoningParser):
    """
    Reasoning parser for the Pangu model.

    The Pangu model uses either [unused16]...[unused17] or <think>...</think>
    tokens to enclose reasoning text within its output. This parser dynamically
    identifies the available tokens and extracts the reasoning content.
    It also handles edge cases in streaming where the start token and initial
    reasoning text are combined in a single chunk.

    Reasoning may also end implicitly when a tool-call-start token
    (<|tool_call_start|> or [unused11]) appears without a preceding
    </think>/[unused17]. The tool-call-start marker itself is kept as the
    leading character of `content` because PanguToolParser locates the
    tool-call payload by searching for that literal token.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.delta_token_ids = []
        self.is_reasoning_end_count = 0
        # Optional implicit-end signal; None unless opted-in via env var
        # PANGU_TOOL_CALL_ENDS_THINKING=1, or when the tokenizer doesn't
        # carry the marker. Default (env unset) matches pre-9c14d17e.
        self.tool_call_start_token_id = (
            self.vocab.get(self.tool_call_start_token)
            if self.tool_call_start_token is not None
            else None
        )
        if os.environ.get("PANGU_TOOL_CALL_ENDS_THINKING", "0") == "1":
            logger.info(
                "PANGU_TOOL_CALL_ENDS_THINKING=1: tool-call-start marker %r "
                "will implicitly end reasoning", self.tool_call_start_token
            )
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Pangu defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        think = chat_kwargs.get("think", True)
        thinking = chat_kwargs.get("thinking", True)
        
        self.thinking_enabled = True
        if not think or not thinking:
            self.thinking_enabled = False
    
    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>" if self.vocab.get("<think>") else "[unused16]"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>" if self.vocab.get("</think>") else "[unused17]"

    @property
    def tool_call_start_token(self) -> str | None:
        """The active tool-call-start token (implicit reasoning-end marker), or
        None unless opted-in via env var PANGU_TOOL_CALL_ENDS_THINKING=1.
        Default (env unset) matches pre-9c14d17e behavior — marker is ignored."""
        if os.environ.get("PANGU_TOOL_CALL_ENDS_THINKING", "0") != "1":
            return None
        if self.vocab.get("<|tool_call_start|>"):
            return "<|tool_call_start|>"
        if self.vocab.get("[unused11]"):
            return "[unused11]"
        return None

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        delta_ids = self.delta_token_ids
        tool_start_id = self.tool_call_start_token_id

        end_in_delta = self.end_token_id in delta_ids and self.end_token_id in input_ids
        tool_start_in_delta = (
            tool_start_id is not None
            and tool_start_id in delta_ids
            and tool_start_id in input_ids
        )
        if end_in_delta or tool_start_in_delta:
            self.is_reasoning_end_count += 1
            return self.is_reasoning_end_count == 1

        if input_ids[-1] == self.end_token_id:
            return True
        if tool_start_id is not None and input_ids[-1] == tool_start_id:
            return True
        return False

    def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Every return path below funnels through ``stash_reasoning_from`` so
        # the relay carries any ``.reasoning`` text on to ``PanguToolParser``.
        # See ``_streaming_relay.py`` for why this matters at the boundary.
        if not self.thinking_enabled:
            ret = DeltaMessage(content=delta_text)
            stash_reasoning_from(ret)
            return ret

        self.delta_token_ids = delta_token_ids

        tool_start_id = self.tool_call_start_token_id
        tool_start_text = self.tool_call_start_token

        # Reasoning already ended implicitly in a previous step — emit content.
        if (
            tool_start_id is not None
            and tool_start_id in previous_token_ids
            and self.end_token_id not in previous_token_ids
        ):
            ret = DeltaMessage(content=delta_text)
            stash_reasoning_from(ret)
            return ret

        # Implicit end fires in this delta. Mirror Kimi's tool-section branch:
        # split text around the marker but KEEP the marker in `content` so the
        # downstream tool parser can still locate the tool-call payload.
        if (
            tool_start_id is not None
            and tool_start_id in delta_token_ids
            and self.end_token_id not in delta_token_ids
            and self.end_token_id not in previous_token_ids
        ):
            tool_idx = delta_text.find(tool_start_text) if tool_start_text else -1
            if tool_idx != -1:
                reasoning_part = delta_text[:tool_idx]
                # Strip a leading start_token if it's in this same delta.
                if (
                    self.start_token_id in delta_token_ids
                    and reasoning_part.startswith(self.start_token)
                ):
                    reasoning_part = reasoning_part[len(self.start_token):]
                content_part = delta_text[tool_idx:]
                logger.info(
                    "Pangu reasoning implicitly ended by tool-call-start "
                    "marker %r (streaming)", tool_start_text
                )
                ret = DeltaMessage(reasoning=reasoning_part, content=content_part)
                stash_reasoning_from(ret)
                return ret

        ret = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        if (
                ret is not None
                and self.start_token_id in delta_token_ids
                and self.end_token_id not in delta_token_ids
        ):
            # multi token case: start token in delta such as '<think>Hello' extract 'Hello' to reasoning
            start_index = delta_text.find(self.start_token)
            delta_text = delta_text[start_index + len(self.start_token):]
            ret = DeltaMessage(reasoning=delta_text)

        stash_reasoning_from(ret)
        return ret

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Check if the <think> is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # Check if the model output contains the </think> tokens.
        # If the end token is not found, return the model output as is.
        if not self.thinking_enabled:
            # Thinking explicitly disabled — treat everything as content.
            return None, model_output

        if self.end_token in model_output:
            # Explicit end takes priority over implicit tool-call-start end.
            reasoning, _, content = model_output.partition(self.end_token)
            return reasoning, content or None

        # Implicit reasoning end: the model emitted a tool-call-start marker
        # without an explicit </think>. Keep the marker as the start of
        # `content` so PanguToolParser can still locate the payload.
        tool_start_text = self.tool_call_start_token
        if tool_start_text and tool_start_text in model_output:
            tool_idx = model_output.find(tool_start_text)
            logger.info(
                "Pangu reasoning implicitly ended by tool-call-start "
                "marker %r (non-streaming)", tool_start_text
            )
            return model_output[:tool_idx], model_output[tool_idx:] or None

        return model_output, None