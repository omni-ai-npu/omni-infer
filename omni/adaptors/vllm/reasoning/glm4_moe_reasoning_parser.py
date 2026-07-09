# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for size in range(max_len, 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


@ReasoningParserManager.register_module("glm52")
@ReasoningParserManager.register_module("glm51")
@ReasoningParserManager.register_module("glm5")
@ReasoningParserManager.register_module("glm47")
@ReasoningParserManager.register_module("glm45")
@ReasoningParserManager.register_module("GlmMoe")
class Glm4MoeModelReasoningParser(ReasoningParser):
    """
    Reasoning parser for the Glm4MoeModel model.

    The Glm4MoeModel model uses <think>...</think> tokens to denote reasoning
    text within its output. The model provides a strict switch to disable
    reasoning output via the 'enable_thinking=False' parameter. This parser
    extracts the reasoning content enclosed by <think> and </think> tokens
    from the model's output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.start_token = "<think>"
        self.end_token = "</think>"
        self.assistant_token = "<|assistant|>"
        self.tool_call_start_token = "<tool_call>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)
        self.assistant_token_id = self.vocab.get(self.assistant_token)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self._reasoning_end_seen = False
        self._deferred_reasoning = ""
        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                "Glm4MoeModel reasoning parser could not locate "
                "think start/end tokens in the tokenizer!"
            )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        GLM's chat template has <think></think> tokens after every
        <|assistant|> token. Thus, we need to check if </think> is
        after the most recent <|assistant|> token (if present).
        """
        if self._reasoning_end_seen:
            return True

        for token_id in input_ids[::-1]:
            if token_id == self.end_token_id:
                return True
            if (
                self.tool_call_start_token_id is not None
                and token_id == self.tool_call_start_token_id
            ):
                return True
            if token_id == self.start_token_id:
                return False
            if (
                self.assistant_token_id is not None
                and token_id == self.assistant_token_id
            ):
                return False
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extract content ids after </think>, or from <tool_call> onward."""
        search_start = 0
        if self.assistant_token_id is not None and self.assistant_token_id in input_ids:
            search_start = (
                len(input_ids)
                - 1
                - input_ids[::-1].index(self.assistant_token_id)
                + 1
            )
        current_turn_ids = input_ids[search_start:]

        if self.end_token_id in current_turn_ids:
            end_index = (
                len(current_turn_ids)
                - 1
                - current_turn_ids[::-1].index(self.end_token_id)
            )
            return current_turn_ids[end_index + 1 :]

        if (
            self.tool_call_start_token_id is not None
            and self.tool_call_start_token_id in current_turn_ids
        ):
            tool_index = current_turn_ids.index(self.tool_call_start_token_id)
            return current_turn_ids[tool_index:]

        return []

    @staticmethod
    def _thinking_enabled(request: ChatCompletionRequest | None) -> bool:
        if request is None or getattr(request, "chat_template_kwargs", None) is None:
            return True

        chat_template_kwargs = request.chat_template_kwargs
        thinking = chat_template_kwargs.get("thinking", None)
        enable_thinking = chat_template_kwargs.get("enable_thinking", None)
        if thinking is None and enable_thinking is None:
            return True
        return bool(thinking) or bool(enable_thinking)

    def _split_reasoning_from_content(
        self,
        text: str,
        boundary_token: str,
        include_boundary_in_content: bool,
    ) -> tuple[str | None, str | None]:
        boundary_index = text.find(boundary_token)
        if boundary_index == -1:
            overlap = _partial_tag_overlap(text, boundary_token)
            reasoning = text[: len(text) - overlap] if overlap else text
            return reasoning or None, None

        content_start = (
            boundary_index
            if include_boundary_in_content
            else boundary_index + len(boundary_token)
        )
        reasoning_content = text[:boundary_index] or None
        content = text[content_start:] or None
        return reasoning_content, content

    def _reasoning_delta(self, reasoning_content: str | None) -> DeltaMessage | None:
        if reasoning_content is None:
            return None
        combined = self._deferred_reasoning + reasoning_content
        trimmed = combined.rstrip()
        self._deferred_reasoning = combined[len(trimmed) :]
        if not trimmed:
            return None
        return DeltaMessage(reasoning_content=trimmed)

    def _with_reasoning_delta(
        self,
        reasoning_content: str | None,
        content: str | None = None,
    ) -> DeltaMessage | None:
        reasoning_delta = self._reasoning_delta(reasoning_content)
        reasoning = reasoning_delta.reasoning_content if reasoning_delta else None
        if self._reasoning_end_seen:
            self._deferred_reasoning = ""
        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning_content=reasoning, content=content)

    def _split_delta_on_tool_call(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        tool_index = current_text.find(self.tool_call_start_token)
        if tool_index == -1:
            overlap = _partial_tag_overlap(current_text, self.tool_call_start_token)
            if not overlap:
                return None
            holdback = min(overlap, len(delta_text))
            reasoning_content = delta_text[:-holdback] if holdback else delta_text
            return self._reasoning_delta(reasoning_content or None)

        self._reasoning_end_seen = True
        reasoning_end = max(0, tool_index - len(previous_text))
        reasoning_content = delta_text[:reasoning_end] or None
        content = current_text[tool_index:] or None
        return self._with_reasoning_delta(reasoning_content, content)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest = None,
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content
        
        GLM-4.7 support: When thinking is enabled by default, the prompt 
        includes '<think>' but model output may only contain '</think>'.
        In this case, content before '</think>' is reasoning.
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.start_token_id, self.end_token_id]
        ):
            return None

        # If thinking is explicitly disabled, all output should be treated
        # as normal content (no reasoning stream).
        if (
            request is not None
            and getattr(request, "chat_template_kwargs", None) is not None
            and not self._thinking_enabled(request)
        ):
            return DeltaMessage(content=delta_text)

        if self._reasoning_end_seen or self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)

        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                self._reasoning_end_seen = True
                return self._with_reasoning_delta(
                    reasoning_content,
                    content if content else None,
                )

            tool_boundary_delta = self._split_delta_on_tool_call(
                previous_text,
                current_text,
                delta_text,
            )
            if tool_boundary_delta is not None:
                return tool_boundary_delta
            return self._reasoning_delta(delta_text)

        if self.start_token_id in delta_token_ids:
            start_index = delta_text.find(self.start_token)
            reasoning_start = start_index + len(self.start_token)
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token, reasoning_start)
                reasoning_content = delta_text[reasoning_start:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                self._reasoning_end_seen = True
                return self._with_reasoning_delta(
                    reasoning_content,
                    content if content else None,
                )

            reasoning_text = delta_text[reasoning_start:]
            tool_index = reasoning_text.find(self.tool_call_start_token)
            if tool_index != -1:
                self._reasoning_end_seen = True
                return self._with_reasoning_delta(
                    reasoning_text[:tool_index] or None,
                    reasoning_text[tool_index:] or None,
                )
            overlap = _partial_tag_overlap(reasoning_text, self.tool_call_start_token)
            if overlap:
                reasoning_text = reasoning_text[:-overlap]
            return self._reasoning_delta(reasoning_text or None)

        if self.end_token_id in delta_token_ids:
            end_index = delta_text.find(self.end_token)
            reasoning_content = delta_text[:end_index]
            content = delta_text[end_index + len(self.end_token) :]
            self._reasoning_end_seen = True
            return self._with_reasoning_delta(
                reasoning_content if reasoning_content else None,
                content if content else None,
            )

        tool_boundary_delta = self._split_delta_on_tool_call(
            previous_text,
            current_text,
            delta_text,
        )
        if tool_boundary_delta is not None:
            return tool_boundary_delta
        return self._reasoning_delta(delta_text)

    def finish_streaming(self) -> DeltaMessage | None:
        if self._deferred_reasoning and not self._reasoning_end_seen:
            reasoning = self._deferred_reasoning
            self._deferred_reasoning = ""
            return DeltaMessage(reasoning_content=reasoning)
        self._deferred_reasoning = ""
        return None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        start_id = self.start_token_id
        end_id = self.end_token_id
        if start_id is None or end_id is None:
            return 0
        count = 0
        depth = 0
        for token_id in token_ids:
            if token_id == start_id:
                depth += 1
                continue
            if token_id == end_id:
                if depth > 0:
                    depth -= 1
                continue
            if depth > 0:
                count += 1
        return count

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning
        - 'xyz' goes to content
        
        GLM-4.7 support: When thinking is enabled by default, the prompt 
        includes '<think>' but model output may only contain '</think>'.
        In this case, content before '</think>' is reasoning.

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        if not self._thinking_enabled(request):
            return None, model_output

        # When thinking is enabled by default, GLM-4.7 can emit reasoning
        # without an opening <think> token. Until </think> appears, output
        # before a tool call is reasoning and tool XML remains parseable content.
        if self.end_token not in model_output:
            reasoning_content, content = self._split_reasoning_from_content(
                model_output,
                self.tool_call_start_token,
                include_boundary_in_content=True,
            )
            if content is not None:
                if self.start_token in (reasoning_content or ""):
                    _, _, reasoning_content = reasoning_content.partition(
                        self.start_token
                    )
                return reasoning_content.rstrip() if reasoning_content else None, content
            if self.start_token in model_output:
                _, _, reasoning_content = model_output.partition(self.start_token)
                return reasoning_content.rstrip() if reasoning_content else None, None
            return model_output.rstrip(), None
        
        # Case 1: Both <think> and </think> present (GLM-4.6 or explicit thinking)
        if self.start_token in model_output:
            # Check if the <think> is present in the model output, remove it
            model_output_parts = model_output.partition(self.start_token)
            model_output = (
                model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
            )
            # Extract reasoning content from the model output
            reasoning_content, _, content = model_output.partition(self.end_token)
            final_content = content or None
            return reasoning_content.rstrip(), final_content
        
        # Case 2: Only </think> present (GLM-4.7 default thinking mode)
        # The <think> was in the prompt, model only generates content + </think>
        else:
            reasoning_content, _, content = model_output.partition(self.end_token)
            final_content = content or None
            return reasoning_content.rstrip(), final_content
